from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


import os
from dotenv import load_dotenv
load_dotenv()

def load_llm():
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        api_key=os.getenv('GOOGLE_API_KEY')
    )

    return llm


# llm = load_llm()

# response = llm.invoke("What are symptoms of diabetes?")

# print(response.content)

DB_FAISS_PATH = "vectorstore/db_faiss"
custom_prompt_template = """
Use the pieces of information provided in the context to answer the user's question.

If you don't know the answer, just say that you don't know.
Do not try to make up an answer.

Do not provide anything outside of the given context.

Context:
{context}

Question:
{question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template = custom_prompt_template, input_variables=["context", "question"])
    return prompt

prompt = set_custom_prompt(custom_prompt_template)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# qa_chain = RetrievalQA.from_chain_type(
#     llm=load_llm(),
#     chain_type="stuff",
#     retriever=db.as_retriever(search_kwargs={"k": 3}),
#     return_source_documents=True,
#     chain_type_kwargs={"prompt": set_custom_prompt(custom_prompt_template)}
# )

# user_query = input("Enter your question: ")
# response = qa_chain.invoke({"query": user_query})
# print("Answer:", response['result'])
# print("Source Documents:",response["source_documents"])

#as per the new version of langchain

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

retriever = db.as_retriever(search_kwargs={"k": 3})

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | load_llm()
    | StrOutputParser()
)

if __name__ == "__main__":
    user_query = input("Enter your question: ")

    response = rag_chain.invoke(user_query)

    print("\nAnswer:", response)