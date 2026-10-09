"""Review the persisted Chroma knowledge base without invoking a chat model."""

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from medibank.deployment import public_mode
from medibank.vector_page import render_vector_page

if public_mode():
    st.error("Vector database administration is available in the local app only.")
    st.stop()

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
render_vector_page()
