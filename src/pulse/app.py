from pathlib import Path

import streamlit as st

from st_pages import get_nav_from_toml

from pulse.pages.state import inject_css

conf_path = Path(".streamlit")

st.set_page_config(layout="wide")

inject_css(conf_path / "styles.css")
inject_css(conf_path / "logo.css")

nav = get_nav_from_toml(path=conf_path / "pages_sections.toml")
pg = st.navigation(pages=nav)
pg.run()
