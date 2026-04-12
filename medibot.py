import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

DB_FAISS_PATH = "vectorstore/db_faiss"

# Initialize embedding model ONCE at module level
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

@st.cache_resource
def get_vectorstore():
    print("Loading FAISS vectorstore...")
    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    print("FAISS loaded ✅")
    return db


@st.cache_resource
def load_llm():
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )
    return llm

llm = load_llm()
def set_custom_prompt():

    template = """
    You are a helpful medical AI assistant. Use the conversation history and the 
    retrieved context below to answer the user's question.

    If the user's question references something from the conversation history 
    (e.g., "it", "that", "the same thing"), resolve the reference using the history.

    If you don't know the answer, just say that you don't know.
    Do not try to make up an answer.

    Conversation History:
    {chat_history}

    Context:
    {context}

    Question:
    {question}

    Answer:
    """


    prompt = PromptTemplate(
        template=template,
        input_variables=["chat_history", "context", "question"]
    )

    return prompt

def get_chat_history(messages, max_turns=5):
    """Format recent conversation turns into a string for the prompt."""
    history_lines = []
    # Take the last `max_turns * 2` messages (each turn = user + assistant)
    recent = messages[-(max_turns * 2):]
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_lines.append(f"{role}: {msg['content']}")
    return "\n".join(history_lines)


def main():

    st.title("🩺 Medical AI Chatbot")

    # Sidebar clear chat button
    with st.sidebar:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    user_query = st.chat_input("Ask me anything about Healthcare")

    if user_query:

        chat_history = get_chat_history(st.session_state.messages)

        st.chat_message("user").markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        try:
            vectorstore = get_vectorstore()
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

            prompt = set_custom_prompt()

            

            # Retrieve docs separately to capture source metadata
            retrieved_docs = retriever.invoke(user_query)
            print(f"Retrieved docs count: {len(retrieved_docs)}")

            if not retrieved_docs:
                response = "I couldn't find relevant information in my knowledge base for your question."
                st.chat_message("assistant").markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                return

            context = format_docs(retrieved_docs)

            rag_chain = (
                {
                    "context": lambda _: context,
                    "chat_history": lambda _: chat_history,
                    "question": RunnablePassthrough()
                }
                | prompt
                | llm
                | StrOutputParser()
            )

            with st.chat_message("assistant"):
                response = rag_chain.invoke(user_query)
                st.write(response)
                # response = st.write_stream(rag_chain.stream(user_query))

                # Display sources
                sources = set()
                for doc in retrieved_docs:
                    src = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "N/A")
                    # Show just the filename, not the full path
                    filename = src.split("/")[-1].split("\\")[-1]
                    sources.add(f"📄 **{filename}** — Page {int(page) + 1}" if page != "N/A" else f"📄 **{filename}**")

                with st.expander("📚 Sources"):
                    for src in sorted(sources):
                        st.markdown(src)

        except Exception as e:
            response = f"⚠️ Error occurred: {str(e)}"
            st.chat_message("assistant").markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()