import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader

model_name = "BAAI/bge-large-en"
model_kwargs = {'device':'cpu'}
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceBgeEmbeddings(
    model_name = model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs
)

print("Embeddings Model Loaded=======================")

loader = PyPDFLoader("pet.pdf")
print(loader)
print("=============================================================")
documents = loader.load()
print(documents)
print("===============================================================")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
print(text_splitter)
print("===============================================================")
texts = text_splitter.split_documents(documents)
print(texts)
print(len(texts))
print("==================================================================")

vector_stores = Chroma.from_documents(texts,
                                      embeddings,
                                      collection_metadata = {"hnsw:space":"cosine"},
                                      persist_directory = 'stores/pet_cosine')
print(vector_stores)
