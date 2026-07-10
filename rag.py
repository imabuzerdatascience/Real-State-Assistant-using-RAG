from uuid import uuid4
from langchain_groq import ChatGroq 
from dotenv import load_dotenv
from pathlib import Path 
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQAWithSourcesChain


print("intilized components")
load_dotenv()  
# avoid dupiction 
llm = None 
vector_store = None

def intialize_components():
  global llm , vector_store 
  if llm is None :
     llm = ChatGroq(model="llama-3.3-70b-versatile" , temperature=0.8  , max_tokens = 500) 

  ef = HuggingFaceEmbeddings(
    model_name =  "sentence-transformers/all-MiniLM-L6-v2" ,
    model_kwargs = {"trust_remote_code": True}
  )


# vector _db 
  if vector_store is None : 
    vector_store = Chroma(
      collection_name = "real_state" ,
      embedding_function= ef ,
      persist_directory= Path("./vector_store")
    )


def process_url(urls) :
  
  loader = UnstructuredURLLoader(urls=urls)
  data = loader.load() 
  

  intialize_components()
  vector_store.reset_collection()

  # chunkers 
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    separators=["\n\n", "\n", ".", ",", "?", "&"],
    chunk_overlap=50
)

  docs = text_splitter.split_documents(data)

  uuids = [str(uuid4()) for _ in range(len(docs))] 
  vector_store.add_documents(docs , ids = uuids)







if __name__ == "__main__" :
  urls = [
          "https://realty.economictimes.indiatimes.com/news/residential/former-ask-asset-directors-family-invests-179-crore-in-luxury-mumbai-apartments/132301523?utm_source=top_story&utm_medium=homepage" ,
  ]
   
  process_url(urls)

  
  results = vector_store.similarity_search(
    query="Family buys apartment",
    k=2
    
) 
print(results)
  
