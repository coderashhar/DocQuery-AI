from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
#     separator="",
#     chunk_size=100,
#     chunk_overlap=1
# )

# text_splitter = TokenTextSplitter(
#     chunk_size=1000, 
#     chunk_overlap=10
# )

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=10
)

data = PyPDFLoader("document loaders/GRU.pdf")
docs = data.load()
chunks = text_splitter.split_documents(docs)

for chunk in chunks:
    print(chunk.page_content)