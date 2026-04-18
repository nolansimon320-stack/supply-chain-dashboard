import os
import streamlit as st


def get_api_key() -> str | None:
    try:
        key = st.secrets.get("FRED_API_KEY")
        if key:
            return key
    except Exception:
        pass
    env_key = os.getenv("FRED_API_KEY")
    if env_key:
        return env_key
    return st.session_state.get("fred_api_key")


def render_sidebar_key_input() -> str | None:
    key = get_api_key()
    if key:
        st.success("API key loaded")
        return key

    st.markdown("**FRED API Key**")
    user_input = st.text_input(
        "Enter key",
        type="password",
        key="fred_api_key",
        placeholder="abcdef1234...",
        help="Free key from fred.stlouisfed.org",
    )
    if user_input:
        st.session_state["fred_api_key"] = user_input
        return user_input
    return None
