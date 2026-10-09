"""Public Streamlit entry point: server-owned models and a chat-only interface."""

from __future__ import annotations

import hashlib
import logging
import os

import streamlit as st

from medibank.deployment import prepare_public_index, read_public_settings
from medibank.knowledge import discover_pdfs
from medibank.limits import RequestLimiter
from medibank.observability import audit, redact

# Both direct admin-page guards and this explicit page registry enforce public mode.
os.environ["APP_PUBLIC_MODE"] = "true"


@st.cache_resource(show_spinner=False)
def public_resources(settings, corpus_signature):
    return prepare_public_index(settings), RequestLimiter()


def public_chat() -> None:
    # Reading root-level Streamlit secrets also loads their environment variables.
    try:
        try:
            values = dict(st.secrets)
        except FileNotFoundError:
            values = {}
        os.environ["APP_PUBLIC_MODE"] = "true"
        settings = read_public_settings(values)
        corpus_signature = tuple(
            (pdf.name, hashlib.sha256(pdf.data).hexdigest())
            for pdf in discover_pdfs(settings.pdf_directory)
        )
        with st.spinner("Preparing Medibank information…"):
            index, limiter = public_resources(settings, corpus_signature)
    except Exception as error:
        os.environ["APP_PUBLIC_MODE"] = "true"
        audit("public_startup_failed", level="ERROR", error_type=type(error).__name__)
        logging.getLogger("medibank.deployment").error(
            "Public startup failed: %s", redact(str(error))
        )
        st.title("Medibank Assistant")
        st.error(
            "The assistant is not ready. The app owner needs to check its deployment settings."
        )
        st.link_button("Contact Medibank", "https://www.medibank.com.au/contact-us/")
        st.stop()
    from app import main

    main(public_settings=settings, public_index=index, request_limiter=limiter)


st.set_page_config(page_title="Medibank Assistant", page_icon="📚", layout="wide")
# st.navigation disables discovery of the local pages/ directory for this server.
st.navigation(
    [st.Page(public_chat, title="Medibank Assistant", default=True)], position="hidden"
).run()
