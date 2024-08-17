# hugging face login
from huggingface_hub import login

HUGGING_FACE_API = ''

login(HUGGING_FACE_API)

from langchain_huggingface.embeddings import HuggingFaceEmbeddings

embedding_model_path = "/home/user/Desktop/semantic-search-langchain_agent/RAG/nomic"

# loading embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name = embedding_model_path,
    model_kwargs = {'trust_remote_code': True},  
)
print("embedding model loaded...:).........................")