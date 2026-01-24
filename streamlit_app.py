import json
import streamlit as st

st.set_page_config(page_title="New Streamlit App", layout="wide")

st.title("Chatbot Evaluator")
st.write("Start building here.")

with st.sidebar:
    st.header("Upload Transcript")
    st.caption("Upload a JSON file containing one or more chatbot conversations.")
    uploaded_file = st.file_uploader(
        "Upload JSON transcript",
        type=["json"],
        label_visibility="collapsed",
    )

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
        # Normalize to a list so the rest of the app is consistent.
        if isinstance(transcript, dict):
            transcript = [transcript]
        elif not isinstance(transcript, list):
            st.error("Transcript must be a JSON object or a list of conversation objects.")
            transcript = None

    if transcript is not None:
        valid_conversations = []
        invalid_conversations = 0

        for index, conversation in enumerate(transcript, start=1):
            if not isinstance(conversation, dict):
                invalid_conversations += 1
                continue

            # Gracefully fill missing optional fields.
            conversation_id = conversation.get("conversation_id") or f"conversation_{index}"
            llm = conversation.get("llm") or "unknown"
            messages = conversation.get("messages")

            # Messages must be a list of objects with valid role and non-empty content.
            if not isinstance(messages, list):
                invalid_conversations += 1
                continue

            is_valid = True
            for message in messages:
                if not isinstance(message, dict):
                    is_valid = False
                    break
                role = message.get("role")
                content = message.get("content")
                if role not in {"user", "assistant", "system"}:
                    is_valid = False
                    break
                if not isinstance(content, str) or not content.strip():
                    is_valid = False
                    break

            if not is_valid:
                invalid_conversations += 1
                continue

            valid_conversations.append(
                {
                    "conversation_id": conversation_id,
                    "llm": llm,
                    "messages": messages,
                }
            )

        if valid_conversations:
            st.success("Transcript loaded and validated.")
            st.subheader("Validation Summary")
            st.write(
                {
                    "total_conversations": len(transcript),
                    "valid_conversations": len(valid_conversations),
                    "invalid_conversations": invalid_conversations,
                }
            )

            if invalid_conversations:
                st.warning("Some conversations were invalid and were skipped.")

            st.subheader("Transcript Preview")
            st.json(valid_conversations[:2], expanded=False)
        else:
            st.error("No valid conversations found in the uploaded transcript.")
