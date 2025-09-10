import sys
import pathlib

import streamlit as st

from st_pages import get_nav_from_toml

# https://github.com/streamlit/streamlit/issues/10992
torch_mod = sys.modules.get("torch")
if torch_mod and hasattr(torch_mod, "classes"):
    torch_mod.classes.__path__ = []

st.set_page_config(layout="wide")
st.html(pathlib.Path(".streamlit/styles.css"))
nav = get_nav_from_toml(".streamlit/pages_sections.toml")

pg = st.navigation(pages=nav)
pg.run()
