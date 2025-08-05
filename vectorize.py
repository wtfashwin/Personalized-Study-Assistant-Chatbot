from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader

loader = DirectoryLoader('./study_materials', glob='*.txt')
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="distilbert-base-uncased")

db = FAISS.from_documents(texts, embeddings)
db.save_local("faiss_index")
print("Vector store created and saved successfully.")   