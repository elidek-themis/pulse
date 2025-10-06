from pathlib import Path

import streamlit as st

from st_pages import get_nav_from_toml

conf_path = Path(".streamlit")

st.set_page_config(layout="wide")

st.html(conf_path / "styles.css")  # horizontal/vertical padding
st.html(conf_path / "logo.css")  # header logos

nav = get_nav_from_toml(path=str(conf_path / "pages_sections.toml"))

pg = st.navigation(pages=nav)
pg.run()
