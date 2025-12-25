from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

loader = PyPDFLoader("data/Swarnima_Dwivedi_Resume.pdf")
documents = loader.load()

# Split text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100
)
chunks = splitter.split_documents(documents)


embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

# Store in FAISS
vector_db = FAISS.from_documents(chunks, embeddings)
vector_db.save_local("vector_db")
print("✅ Ingestion complete")