"""
app.py
------
Streamlit chat interface (FR-6).

Run with:  streamlit run app.py
"""

import streamlit as st

from chatbot import ask, build_chatbot
from schemas import ChatResponse

st.set_page_config(page_title="LangChain Router Chatbot", page_icon="🧭")


@st.cache_resource
def load_chain():
    """Build the chain once and reuse it across reruns."""
    return build_chatbot()


def render_response(response: ChatResponse) -> None:
    """Show the validated Pydantic object in the chat bubble."""
    st.markdown(f"**Route:** `{response.category.value}` — {response.routing_reason}")
    st.markdown(response.answer)

    col1, col2 = st.columns(2)
    col1.metric("Confidence", f"{response.confidence:.0%}")
    col2.metric("Difficulty", response.difficulty.value.title())

    st.markdown("**Summary**")
    st.info(response.summary)

    st.markdown("**Keywords**")
    st.write(" · ".join(f"`{k}`" for k in response.keywords))

    with st.expander("Follow-up questions"):
        for q in response.follow_up_questions:
            st.write(f"- {q}")

    with st.expander("Raw structured output (Pydantic)"):
        st.json(response.model_dump(mode="json"))


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("🧭 Router Chatbot")
    st.caption("RunnableBranch + RunnableParallel + Pydantic")
    st.divider()
    st.markdown(
        "**How it works**\n\n"
        "1. Classify the question\n"
        "2. `RunnableBranch` picks a specialist\n"
        "3. `RunnableParallel` runs answer + analysis together\n"
        "4. Result validated as a Pydantic model"
    )
    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --------------------------------------------------------------------------
# Main chat
# --------------------------------------------------------------------------
st.title("LangChain Router Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    chain = load_chain()
except ValueError as e:
    st.error(str(e))
    st.stop()

# Replay chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            render_response(message["content"])

# Handle new input
question = st.chat_input("Ask me about code, math, or anything else...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Routing and answering..."):
            try:
                response = ask(chain, question)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.stop()
        render_response(response)

    st.session_state.messages.append({"role": "assistant", "content": response})