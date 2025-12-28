import pathlib
import streamlit as st

st.set_page_config(
    page_title="Documentation",
    layout="wide",
)

st.title("Documentation")

DOC_PATH = (
    pathlib.Path(__file__)
    .resolve()
    .parents[1]
    / "docs"
    / "stochastic_models.md"
)

if not DOC_PATH.exists():
    st.error(f"Documentation file not found: {DOC_PATH}")
else:
    st.markdown(DOC_PATH.read_text(encoding="utf-8"))
