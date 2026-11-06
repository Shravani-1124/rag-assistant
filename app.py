"""Streamlit chat UI for the internal knowledge assistant."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st

DEFAULT_API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT_SECONDS = 120


def parse_paths(raw: str) -> list[str]:
    paths: list[str] = []
    for line in raw.splitlines():
        cleaned = line.strip().strip('"').strip("'")
        if cleaned:
            paths.append(str(Path(cleaned).expanduser().resolve()))
    return paths


def error_detail(raw: str) -> str:
    text = raw.strip()
    if not text:
        return "Unknown error"
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    detail = parsed.get("detail", parsed)
    if isinstance(detail, list):
        return "; ".join(str(item.get("msg", item)) for item in detail)
    return str(detail)


def post_json(base_url: str, path: str, payload: dict) -> dict:
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = error_detail(exc.read().decode("utf-8", errors="replace"))
        raise RuntimeError(f"{path} failed ({exc.code}): {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach the API at {base_url}: {exc.reason}") from exc

    try:
        parsed = json.loads(body) if body else {}
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{path} returned a non-JSON response.") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{path} returned an unexpected response.")
    return parsed


def source_label(source: object, index: int) -> str:
    if isinstance(source, str):
        return source or f"Source {index}"
    if isinstance(source, dict):
        filename = str(source.get("filename") or source.get("source") or f"Source {index}")
        chunk_id = source.get("chunk_id")
        if chunk_id:
            return f"{filename} · {chunk_id}"
        return filename
    return f"Source {index}"


def render_sources(sources: object) -> None:
    if not isinstance(sources, list) or not sources:
        return
    for index, source in enumerate(sources, start=1):
        with st.expander(source_label(source, index)):
            if isinstance(source, dict):
                excerpt = source.get("text") or source.get("excerpt") or source.get("content")
                metadata = {
                    key: value
                    for key, value in source.items()
                    if key not in {"text", "excerpt", "content"}
                }
                if metadata:
                    st.json(metadata)
                if excerpt:
                    st.markdown(str(excerpt))
            else:
                st.write(str(source))


def render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources"))


st.set_page_config(page_title="Internal Knowledge Assistant", page_icon=":books:")
st.title("Internal Knowledge Assistant")
st.caption("Answers come from documents indexed by the API, with citations for each reply.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Ingest documents")
    api_base = st.text_input("API base URL", value=DEFAULT_API_BASE)
    path_text = st.text_area(
        "Document paths",
        height=180,
        placeholder="/path/to/handbook.md\n/path/to/policy.txt",
        help="One document path per line. These paths are sent to POST /ingest.",
    )
    if st.button("Ingest", type="primary"):
        paths = parse_paths(path_text)
        if not paths:
            st.warning("Enter at least one document path.")
        else:
            missing = [path for path in paths if not Path(path).exists()]
            if missing:
                st.error("These paths do not exist:\n" + "\n".join(missing))
            else:
                with st.spinner("Indexing documents..."):
                    try:
                        result = post_json(api_base, "/ingest", {"paths": paths})
                    except RuntimeError as exc:
                        st.error(str(exc))
                    else:
                        indexed = result.get("chunks_indexed", result.get("indexed_chunks"))
                        if indexed is None:
                            st.success(f"Sent {len(paths)} path(s) to /ingest.")
                        else:
                            st.success(f"Indexed {indexed} chunk(s) from {len(paths)} path(s).")

render_history()

question = st.chat_input("Ask a question about internal documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    try:
        with st.spinner("Searching..."):
            result = post_json(api_base, "/query", {"query": question})
    except RuntimeError as exc:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": str(exc),
                "sources": [],
            }
        )
    else:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": str(result.get("answer", "")),
                "sources": result.get("sources") or [],
            }
        )
    st.rerun()
