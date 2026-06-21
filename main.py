from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embedding_model = HuggingFaceEmbeddings()

vectorstore = Chroma(
    embedding_function=embedding_model,
    persist_directory="chroma_db"
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":3, 
        "fetch_k":10, 
        "lambda_mult":0.5
    }
) 

llm = ChatMistralAI(model="mistral-small-latest") 

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a helpful assistant that provides concise answers based on retrieved documents. If the retrieved documents do not contain enough information to answer the question, say "I could not find the information in the given document" """),
        ("human", "Context:{context} Question:{question}")
    ]
)

print("Rag system created ")

print("press 0 to exit ")

while True:
    query = input("You : ")
    if query == "0":
        break 
    
    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    
    final_prompt = prompt.invoke({
        "context" :context,
        "question": query
    })
    
    response = llm.invoke(final_prompt)

    print(f"\n AI: {response.content}")
    