import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain.messages import AIMessage, HumanMessage

from agent_core import DISCLAIMER, build_nutrition_agent, build_user_message
from memory import SummaryMemoryManager
from tools import ensure_csv_exists


load_dotenv()
st.set_page_config(page_title="Nutrition AI Agent", page_icon="🥗", layout="wide")


def init_state() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"nutrition-{uuid.uuid4()}"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = build_nutrition_agent("gpt-4.1")
    if "memory" not in st.session_state:
        st.session_state.memory = SummaryMemoryManager(model_name="gpt-4.1-mini", max_messages=8)


def display_history() -> None:
    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(msg.content if isinstance(msg.content, str) else "User sent text + image input.")
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(msg.content)


def sectioned_output(text: str) -> None:
    headers = [
        "Meal Analysis",
        "Nutrition Summary",
        "Recommendations",
        "Search Results",
        "CSV Storage",
    ]
    for header in headers:
        marker = f"## {header}"
        if marker in text:
            start = text.index(marker)
            next_positions = [text.find(f"## {h}", start + 1) for h in headers if text.find(f"## {h}", start + 1) != -1]
            end = min(next_positions) if next_positions else len(text)
            block = text[start:end].strip()
            with st.expander(header, expanded=True):
                st.markdown(block)
    if "## " not in text:
        st.markdown(text)


init_state()
ensure_csv_exists()

st.title("Multi-Modal Nutrition AI Agent")
st.caption("Analyze meals, estimate nutrition, search healthy options, and store structured meal logs.")

with st.sidebar:
    st.subheader("Session Settings")
    model_name = st.selectbox("Model", ["gpt-4.1", "gpt-4.1-mini"], index=0)
    city = st.text_input("Default City", value="Assiut")
    goal = st.text_input("Default Goal", value="Maintain healthy balance")
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.session_state.memory = SummaryMemoryManager(model_name="gpt-4.1-mini", max_messages=8)
        st.rerun()

if model_name != "gpt-4.1":
    st.session_state.agent = build_nutrition_agent(model_name)

display_history()

with st.form("nutrition_form", clear_on_submit=True):
    user_text = st.text_area("Meal text input (required)", placeholder="Describe your meal and what you want help with...")
    uploaded = st.file_uploader("Optional meal image", type=["jpg", "jpeg", "png", "webp"])
    submit = st.form_submit_button("Analyze")

if submit:
    if not user_text.strip():
        st.error("Text input is required.")
    else:
        final_text = f"{user_text.strip()}\n\nContext:\n- City: {city}\n- Goal: {goal}"
        image_bytes = uploaded.read() if uploaded else None
        mime = uploaded.type if uploaded else "image/jpeg"

        user_message = build_user_message(final_text, image_bytes, mime)
        st.session_state.messages.append(user_message)
        effective_messages = st.session_state.memory.inject_summary(st.session_state.messages)

        with st.chat_message("user"):
            st.write(user_text)
            if uploaded:
                st.image(uploaded, caption="Uploaded meal image", use_column_width=True)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing your nutrition input..."):
                result = st.session_state.agent.invoke(
                    {"messages": effective_messages},
                    config={"configurable": {"thread_id": st.session_state.thread_id}},
                )
                ai_msg = result["messages"][-1]
                if DISCLAIMER not in ai_msg.content:
                    ai_msg.content = f"{ai_msg.content}\n\n{DISCLAIMER}"
                st.session_state.messages.append(ai_msg)
                sectioned_output(ai_msg.content)

st.info(DISCLAIMER)

