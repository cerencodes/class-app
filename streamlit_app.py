import json
from urllib import error, request

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

with open("dale_chall_easy_words.txt", "r", encoding="utf-8") as file:
    DALE_CHALL_EASY_WORDS = {line.strip().lower() for line in file if line.strip()}

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
    st.subheader("Select LLM")
    openrouter_api_key = st.text_input("OpenRouter API Key", type="password")
    evaluation_model = st.selectbox(
        "Evaluation Model",
        [
            "openai/gpt-4o-mini",
            "anthropic/claude-3.5-haiku",
            "google/gemini-2.0-flash-001",
        ],
    )


def openrouter_chat_completion(api_key: str, model: str, messages: list[dict]) -> str | None:
    if not api_key:
        st.warning("OpenRouter API key is missing.")
        return None

    payload = json.dumps({"model": model, "messages": messages}).encode("utf-8")
    req = request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Chatbot Evaluator",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        st.error(f"OpenRouter request failed: {exc.code} {exc.reason}")
        return None
    except error.URLError as exc:
        st.error(f"OpenRouter request failed: {exc.reason}")
        return None
    except Exception as exc:
        st.error(f"OpenRouter request failed: {exc}")
        return None

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        st.error("OpenRouter response format was unexpected.")
        return None

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
            model = conversation.get("model") or conversation.get("llm") or "unknown"
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
                    "model": model,
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

            def syllable_count_word(word: str) -> int:
                # Simple, deterministic heuristic: count vowel groups, trim silent trailing 'e'.
                w = "".join(ch for ch in word.lower() if ch.isalpha())
                if not w:
                    return 0
                vowels = set("aeiouy")
                groups = 0
                prev_vowel = False
                for ch in w:
                    is_vowel = ch in vowels
                    if is_vowel and not prev_vowel:
                        groups += 1
                    prev_vowel = is_vowel
                if w.endswith("e") and groups > 1:
                    groups -= 1
                return max(groups, 1)

            def syllable_count_text(text: str) -> int:
                return sum(syllable_count_word(word) for word in text.split())

            def tokenize_words(text: str) -> list[str]:
                tokens = []
                for raw in text.split():
                    cleaned = "".join(ch for ch in raw.lower() if ch.isalpha())
                    if cleaned:
                        tokens.append(cleaned)
                return tokens

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
                assistant_syllables = syllable_count_text(assistant_text)
                assistant_tokens = tokenize_words(assistant_text)
                difficult_words = sum(
                    1 for word in assistant_tokens if word not in DALE_CHALL_EASY_WORDS
                )

                if assistant_words:
                    ari = 4.71 * (assistant_chars / assistant_words)
                    ari += 0.5 * (assistant_words / assistant_sentences)
                    ari -= 21.43
                    flesch_reading_ease = (
                        206.835
                        - 1.015 * (assistant_words / assistant_sentences)
                        - 84.6 * (assistant_syllables / assistant_words)
                    )
                    difficult_pct = (difficult_words / assistant_words) * 100
                    dale_chall_score = 0.1579 * difficult_pct + 0.0496 * (
                        assistant_words / assistant_sentences
                    )
                    if difficult_pct > 5:
                        dale_chall_score += 3.6365
                else:
                    ari = 0
                    flesch_reading_ease = 0
                    dale_chall_score = 0

                avg_assistant_words = (
                    assistant_words / assistant_turns if assistant_turns else 0
                )

                per_conversation_metrics.append(
                    {
                        "conversation_id": conversation["conversation_id"],
                        "model": conversation["model"],
                        "total_turns": total_turns,
                        "user_turns": user_turns,
                        "assistant_turns": assistant_turns,
                        "avg_assistant_words_per_message": avg_assistant_words,
                        "ari": ari,
                        "flesch_reading_ease": flesch_reading_ease,
                        "dale_chall_score": dale_chall_score,
                    }
                )

            st.subheader("Readability Performance")
            display_rows = []
            for row in per_conversation_metrics:
                display_rows.append(
                    {
                        "conversation_id": row["conversation_id"],
                        "model": row["model"],
                        "total_turns": int(row["total_turns"]),
                        "user_turns": int(row["user_turns"]),
                        "assistant_turns": int(row["assistant_turns"]),
                        "avg_assistant_words_per_message": int(
                            round(row["avg_assistant_words_per_message"])
                        ),
                        "ari": round(row["ari"], 2),
                        "flesch_reading_ease": round(row["flesch_reading_ease"], 2),
                        "dale_chall_score": round(row["dale_chall_score"], 2),
                        "clears_all_3_metrics": (
                            row["ari"] <= 10
                            and row["flesch_reading_ease"] >= 60
                            and row["dale_chall_score"] <= 7.0
                        ),
                    }
                )
            display_df = pd.DataFrame(display_rows)

            def color_good(value: float, threshold: float, op: str) -> str:
                if op == "le" and value <= threshold:
                    return "color: green;"
                if op == "ge" and value >= threshold:
                    return "color: green;"
                return ""

            styled_df = (
                display_df.style.format(
                    {
                        "avg_assistant_words_per_message": "{:.0f}",
                        "ari": "{:.2f}",
                        "flesch_reading_ease": "{:.2f}",
                        "dale_chall_score": "{:.2f}",
                    }
                )
                .applymap(lambda v: color_good(v, 10, "le"), subset=["ari"])
                .applymap(
                    lambda v: color_good(v, 60, "ge"),
                    subset=["flesch_reading_ease"],
                )
                .applymap(
                    lambda v: color_good(v, 7, "le"),
                    subset=["dale_chall_score"],
                )
            )
            st.dataframe(styled_df, use_container_width=True)
            csv_text = display_df.to_csv(index=False)
            components.html(
                """
                <button id="copy-readability-csv">Copy table to clipboard</button>
                <div id="readability-csv" style="display:none;"></div>
                <script>
                  const csv = %s;
                  document.getElementById("readability-csv").textContent = csv;
                  document.getElementById("copy-readability-csv").onclick = () => {
                    navigator.clipboard.writeText(
                      document.getElementById("readability-csv").textContent
                    );
                  };
                </script>
                """
                % json.dumps(csv_text),
                height=40,
            )
            clears_count = sum(display_df["clears_all_3_metrics"])
            total_count = len(display_df)
            clears_pct = (clears_count / total_count * 100) if total_count else 0
            st.write(
                f"Clears all 3 thresholds: {clears_count} of {total_count} "
                f"({clears_pct:.1f}%)"
            )
            ari_pass_count = sum(display_df["ari"] <= 10)
            ari_pass_pct = (ari_pass_count / total_count * 100) if total_count else 0
            st.write(
                f"ARI pass rate: {ari_pass_count} of {total_count} "
                f"({ari_pass_pct:.1f}%)"
            )
            flesch_pass_count = sum(display_df["flesch_reading_ease"] >= 60)
            flesch_pass_pct = (
                (flesch_pass_count / total_count * 100) if total_count else 0
            )
            st.write(
                f"Flesch Reading Ease pass rate: {flesch_pass_count} of {total_count} "
                f"({flesch_pass_pct:.1f}%)"
            )
            dale_pass_count = sum(display_df["dale_chall_score"] <= 7.0)
            dale_pass_pct = (dale_pass_count / total_count * 100) if total_count else 0
            st.write(
                f"Dale–Chall pass rate: {dale_pass_count} of {total_count} "
                f"({dale_pass_pct:.1f}%)"
            )
            with st.expander("How to interpret these readability scores"):
                st.markdown(
                    """
                    - **ARI (Automated Readability Index):**
                      Lower scores indicate easier reading. Values around 1–6 are
                      easy/elementary, 7–12 are middle to early high school, and
                      13+ indicates more complex text.
                    - **Flesch–Kincaid Reading Ease:**
                      Higher scores are easier to read. 90–100 is very easy,
                      60–70 is standard/easily understood, and below 30 is very
                      difficult.
                    - **Dale–Chall:**
                      Lower scores are easier. About 4.9 or lower is easy,
                      5.0–6.9 is average, and 7.0+ is difficult.
                    """
            )

            st.subheader("Accuracy Performance")
            default_prompt = (
                "#Task\n"
                "Summarize user problem and requested features using only user messages.\n"
                "\n"
                "#Constraints\n"
                "- Use only user messages.\n"
                "- Be concise, avoid filler words.\n"
                "- Maximum 1 short sentence, omit periods\n"
                "- No bulletpoints or other text styling\n"
                "\n"
                "#Good Example\n"
                "- 6'4\" tall, large hands, needs larger grips\n"
                "- BMX racer, wants ultra narrow grips\n"
                "\n"
                "#Bad Example\n"
                "- The user is 6'4\" and needs a larger grip suitable for their large hands.\n"
                "- The user races BMX and is specifically looking for recommendations on "
                "ultra narrow grips."
            )
            context_prompt = st.text_area(
                "Context Extraction Prompt",
                value=default_prompt,
                height=120,
            )
            default_product_prompt = (
                "#Task\n"
                "Identify all specific products recommended by the assistant in this "
                "conversation.\n"
                "\n"
                "#Constraints\n"
                "- Use only assistant messages.\n"
                "- If multiple products are recommended, separate them with a semicolon "
                "(;).\n"
                "- If no product is recommended, return an empty string.\n"
                "- Output plain text only. No explanations."
            )
            product_prompt = st.text_area(
                "Product Extraction Prompt",
                value=default_product_prompt,
                height=140,
            )
            run_accuracy = st.button("Run")
            if run_accuracy:
                if not openrouter_api_key:
                    st.warning(
                        "OpenRouter API key is missing. Context summaries were skipped."
                    )
                else:
                    context_rows = []
                    for conversation in valid_conversations:
                        user_messages = [
                            m.get("content", "")
                            for m in conversation["messages"]
                            if m.get("role") == "user"
                        ]
                        assistant_messages = [
                            m.get("content", "")
                            for m in conversation["messages"]
                            if m.get("role") == "assistant"
                        ]
                        if not user_messages:
                            context_summary = ""
                        else:
                            prompt = context_prompt.strip() or default_prompt
                            context_summary = openrouter_chat_completion(
                                openrouter_api_key,
                                evaluation_model,
                                [
                                    {"role": "system", "content": prompt},
                                    {"role": "user", "content": "\n".join(user_messages)},
                                ],
                            )
                            context_summary = context_summary or ""

                        if not assistant_messages:
                            product_summary = ""
                        else:
                            product_prompt_text = (
                                product_prompt.strip() or default_product_prompt
                            )
                            product_summary = openrouter_chat_completion(
                                openrouter_api_key,
                                evaluation_model,
                                [
                                    {"role": "system", "content": product_prompt_text},
                                    {
                                        "role": "user",
                                        "content": "\n".join(assistant_messages),
                                    },
                                ],
                            )
                            product_summary = product_summary or ""

                        context_rows.append(
                            {
                                "conversation_id": conversation["conversation_id"],
                                "context": context_summary.strip(),
                                "product_recommended": product_summary.strip(),
                            }
                        )

                    context_df = pd.DataFrame(context_rows)
                    st.table(context_df)
                    context_csv = context_df.to_csv(index=False)
                    components.html(
                        """
                        <button id="copy-context-csv">Copy to clipboard</button>
                        <div id="context-csv" style="display:none;"></div>
                        <script>
                          const contextCsv = %s;
                          document.getElementById("context-csv").textContent = contextCsv;
                          document.getElementById("copy-context-csv").onclick = () => {
                            navigator.clipboard.writeText(
                              document.getElementById("context-csv").textContent
                            );
                          };
                        </script>
                        """
                        % json.dumps(context_csv),
                        height=40,
                    )

            st.subheader("Transcript Preview")
            st.json(valid_conversations[:2], expanded=False)
        else:
            st.error("No valid conversations found in the uploaded transcript.")
