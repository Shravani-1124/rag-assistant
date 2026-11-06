"""FastAPI RAG service using LlamaIndex and a local Qdrant store."""

from __future__ import annotations

import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from pydantic import BaseModel, Field, field_validator
from qdrant_client import QdrantClient

load_dotenv()

QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "internal_docs"
SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown"}
UNKNOWN_ANSWER = "I don't know based on the available internal documents."

QA_PROMPT = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "You are an internal knowledge assistant. Answer the question using only the context.\n"
    "Each passage includes filename and chunk_id. Cite both when you use a passage.\n"
    "If the context does not contain the answer, reply exactly: "
    f"\"{UNKNOWN_ANSWER}\"\n"
    "Do not use outside knowledge or guess.\n"
    "Question: {query_str}\n"
    "Answer: "
)

_client: QdrantClient | None = None
_vector_store: QdrantVectorStore | None = None
_store_lock = asyncio.Lock()


class IngestRequest(BaseModel):
    paths: list[str] = Field(..., min_length=1)


class IngestResponse(BaseModel):
    chunks_indexed: int
    collection: str = COLLECTION_NAME


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)

    @field_validator("query")
    @classmethod
    def query_must_have_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Query cannot be empty.")
        return cleaned


class Source(BaseModel):
    filename: str
    chunk_id: str
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


def require_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key or key == "your_openai_api_key_here":
        raise RuntimeError("Set OPENAI_API_KEY in .env before running.")
    return key


def configure_models() -> None:
    api_key = require_api_key()
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=api_key,
    )


def get_vector_store() -> QdrantVectorStore:
    global _client, _vector_store
    if _vector_store is None:
        _client = QdrantClient(path=QDRANT_PATH)
        _vector_store = QdrantVectorStore(
            client=_client,
            collection_name=COLLECTION_NAME,
        )
    return _vector_store


def collection_ready(client: QdrantClient) -> bool:
    try:
        return bool(client.collection_exists(COLLECTION_NAME))
    except Exception:
        names = {item.name for item in client.get_collections().collections}
        return COLLECTION_NAME in names


def read_documents(paths: list[str]) -> list[Document]:
    documents: list[Document] = []
    missing: list[str] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if not path.is_file():
            missing.append(str(path))
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {path.name}. Use .txt or .md files.",
            )
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"{path.name} is not valid UTF-8 text.",
            ) from exc
        if text.strip():
            documents.append(Document(text=text, metadata={"filename": path.name}))
    if missing:
        raise HTTPException(
            status_code=404,
            detail="Paths not found: " + ", ".join(missing),
        )
    if not documents:
        raise HTTPException(status_code=400, detail="No document text to index.")
    return documents


def chunk_documents(documents: list[Document]):
    nodes = SentenceSplitter(chunk_size=512, chunk_overlap=50).get_nodes_from_documents(
        documents
    )
    for node in nodes:
        filename = str(node.metadata.get("filename", "unknown"))
        chunk_id = f"{filename}::{uuid.uuid4().hex[:12]}"
        node.metadata["filename"] = filename
        node.metadata["chunk_id"] = chunk_id
        node.id_ = chunk_id
    if not nodes:
        raise HTTPException(status_code=400, detail="Documents produced no chunks.")
    return nodes


def ingest_paths(paths: list[str]) -> int:
    configure_models()
    nodes = chunk_documents(read_documents(paths))
    storage_context = StorageContext.from_defaults(vector_store=get_vector_store())
    VectorStoreIndex(nodes, storage_context=storage_context, show_progress=False)
    return len(nodes)


def query_index(question: str) -> dict[str, object]:
    configure_models()
    store = get_vector_store()
    client = store.client
    if client is None or not collection_ready(client):
        return {"answer": UNKNOWN_ANSWER, "sources": []}

    engine = VectorStoreIndex.from_vector_store(store).as_query_engine(
        similarity_top_k=4,
        text_qa_template=QA_PROMPT,
    )
    response = engine.query(question)
    sources = [
        {
            "filename": str(node.metadata.get("filename", "unknown")),
            "chunk_id": str(node.metadata.get("chunk_id", node.node_id)),
            "text": node.get_content(),
        }
        for node in response.source_nodes
    ]
    answer = str(response).strip()
    if not sources or not answer:
        return {"answer": UNKNOWN_ANSWER, "sources": []}
    return {"answer": answer, "sources": sources}


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_vector_store()
    yield
    if _client is not None:
        _client.close()


app = FastAPI(title="Internal Knowledge Assistant", lifespan=lifespan)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(payload: IngestRequest) -> IngestResponse:
    async with _store_lock:
        try:
            count = await asyncio.to_thread(ingest_paths, payload.paths)
        except HTTPException:
            raise
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    return IngestResponse(chunks_indexed=count)


@app.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest) -> QueryResponse:
    async with _store_lock:
        try:
            result = await asyncio.to_thread(query_index, payload.query)
        except HTTPException:
            raise
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    return QueryResponse.model_validate(result)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
