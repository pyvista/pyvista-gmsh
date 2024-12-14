"""streamlit application."""

from pathlib import Path

import streamlit as st

intro_markdown = Path("README.md").read_text()
st.markdown(intro_markdown, unsafe_allow_html=True)
