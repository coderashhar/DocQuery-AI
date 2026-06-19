from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="",
    chunk_size=100,
    chunk_overlap=1
)

data = TextLoader("document loaders/notes.txt")
docs = data.load()
chunks = text_splitter.split_documents(docs)

for chunk in chunks:
    print(chunk.page_content)