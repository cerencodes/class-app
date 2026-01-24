import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Chatbot Evaluator", layout="wide")

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

            # Compute readability metrics per conversation.
            per_conversation_metrics = []

            def word_count(text: str) -> int:
                return len(text.split())

            def alnum_count(text: str) -> int:
                return sum(1 for ch in text if ch.isalnum())

            def sentence_count(text: str) -> int:
                return sum(1 for ch in text if ch in {".", "!", "?"})

            for conversation in valid_conversations:
                messages = conversation["messages"]
                total_turns = len(messages)
                user_turns = sum(1 for m in messages if m.get("role") == "user")
                assistant_turns = sum(1 for m in messages if m.get("role") == "assistant")

                assistant_messages = [m for m in messages if m.get("role") == "assistant"]
                assistant_text = " ".join(m.get("content", "") for m in assistant_messages)
                assistant_words = word_count(assistant_text)
                assistant_chars = alnum_count(assistant_text)
                assistant_sentences = sentence_count(assistant_text)
                if assistant_words and assistant_sentences == 0:
                    assistant_sentences = 1

                if assistant_words:
                    ari = 4.71 * (assistant_chars / assistant_words)
                    ari += 0.5 * (assistant_words / assistant_sentences)
                    ari -= 21.43
                else:
                    ari = 0

                avg_assistant_words = (
                    assistant_words / assistant_turns if assistant_turns else 0
                )

                per_conversation_metrics.append(
                    {
                        "conversation_id": conversation["conversation_id"],
                        "llm": conversation["llm"],
                        "total_turns": total_turns,
                        "user_turns": user_turns,
                        "assistant_turns": assistant_turns,
                        "avg_assistant_words_per_message": avg_assistant_words,
                        "ari": ari,
                    }
                )

            st.subheader("Readability Performance")
            display_rows = []
            for row in per_conversation_metrics:
                display_rows.append(
                    {
                        "conversation_id": row["conversation_id"],
                        "llm": row["llm"],
                        "total_turns": int(row["total_turns"]),
                        "user_turns": int(row["user_turns"]),
                        "assistant_turns": int(row["assistant_turns"]),
                        "avg_assistant_words_per_message": int(
                            round(row["avg_assistant_words_per_message"])
                        ),
                        "ari": round(row["ari"], 2),
                    }
                )
            display_df = pd.DataFrame(display_rows)
            styled_df = display_df.style.format(
                {
                    "avg_assistant_words_per_message": "{:.0f}",
                    "ari": "{:.2f}",
                }
            )
            st.dataframe(styled_df, use_container_width=True)

            st.subheader("Transcript Preview")
            st.json(valid_conversations[:2], expanded=False)
        else:
            st.error("No valid conversations found in the uploaded transcript.")
