import streamlit as st

st.set_page_config(page_title="New Streamlit App", layout="wide")

st.title("New Streamlit App")
st.write("Start building here.")

with st.sidebar:
    st.header("Controls")
    st.text_input("Example input")