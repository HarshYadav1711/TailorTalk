"""Streamlit frontend for the Titanic dataset chat agent."""

from __future__ import annotations

import base64
import os
from typing import Any

import requests
import streamlit as st


def _get_backend_url() -> str:
    """Resolve backend URL from secrets/env, with local fallback."""
    secret_url = st.secrets.get("BACKEND_URL")
    env_url = os.getenv("BACKEND_URL")
    return str(secret_url or env_url or "http://localhost:8000/chat")


BACKEND_URL = _get_backend_url()

st.set_page_config(page_title="Titanic Chat Agent", page_icon=":ship:")
st.title("Titanic Dataset Chat Agent")
st.write(
    "Ask questions about the Titanic passenger dataset. "
    "The assistant uses deterministic pandas tools for every calculation."
)
st.caption(f"Backend: {BACKEND_URL}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("tool_used"):
            st.caption(f"Tool used: {message['tool_used']}")
        if message.get("visualization_base64"):
            image_bytes = base64.b64decode(message["visualization_base64"])
            st.image(image_bytes, caption="Generated visualization")

question = st.chat_input("Ask about survival, fares, ages, or passenger groups...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            response = requests.post(BACKEND_URL, json={"question": question}, timeout=90)
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            answer_text = payload.get("response", "")
            tool_used = payload.get("tool_used")
            visualization_base64 = payload.get("visualization_base64")
        except requests.RequestException as exc:
            answer_text = f"Backend request failed: {exc}"
            tool_used = None
            visualization_base64 = None

        st.write(answer_text)
        if tool_used:
            st.caption(f"Tool used: {tool_used}")
        if visualization_base64:
            image_bytes = base64.b64decode(visualization_base64)
            st.image(image_bytes, caption="Generated visualization")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer_text,
            "tool_used": tool_used,
            "visualization_base64": visualization_base64,
        }
    )

