import json
import streamlit as st

st.set_page_config(page_title="New Streamlit App", layout="wide")

st.title("New Streamlit App")
st.write("Start building here.")

with st.sidebar:
    st.header("Controls")
    st.text_input("Example input")
    st.header("Upload Transcript")
    uploaded_file = st.file_uploader("Upload JSON transcript", type=["json"])

if uploaded_file is not None:
    try:
        transcript = json.load(uploaded_file)
    except json.JSONDecodeError:
        st.error("Invalid JSON file. Please upload a valid JSON transcript.")
        transcript = None
    except Exception as exc:
        st.error(f"Could not read the file: {exc}")
        transcript = None
    else:
        st.success("Transcript loaded successfully.")
        st.subheader("Transcript Preview")
        st.json(transcript, expanded=False)
