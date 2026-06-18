from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

data = PyPDFLoader("document loaders/GRU.pdf")
docs = data.load()

template =  ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant that summarizes the following text."),
                ("human", "{text}")
            ])

model = ChatMistralAI(model = "mistral-small-2603")
prompt = template.format_messages(text = docs)
result = model.invoke(prompt)
print(result.content)