from uuid import uuid4
from langchain_groq import ChatGroq 
from dotenv import load_dotenv
from pathlib import Path 
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQAWithSourcesChain 
import os 



load_dotenv()


# avoid dupiction 
llm = None 
vector_store = None

def intialize_components():
  global llm , vector_store 

  if llm is None :
     llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY") , model="llama-3.3-70b-versatile" , temperature=0.8  , max_tokens = 500) 

  if vector_store is None :
    ef = HuggingFaceEmbeddings(
      model_name =  "sentence-transformers/all-MiniLM-L6-v2" ,
      model_kwargs = {"trust_remote_code": True}
    )


# vector _db 
 
    vector_store = Chroma(
      collection_name = "real_state" ,
      embedding_function= ef ,
      persist_directory= Path("./vector_store")
    )


def process_url(urls):

  
    # yield "Initializing LLM and Vector Database..."

    intialize_components()
    vector_store.reset_collection()

    # yield "Loading URLs..."

    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    print("Documents:", len(data))

    for doc in data:
      print("=" * 50)
      print(doc.metadata)
      print(doc.page_content[:1000])

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        separators=["\n\n", "\n", ".", ",", "?", "&"]
    )

    # yield "Splitting documents into chunks..."

    docs = text_splitter.split_documents(data)

    # yield f"Creating embeddings for {len(docs)} chunks..."

    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)

    # yield "Embeddings stored in Vector Database successfully."


def genrate_answer(query):
  if not vector_store :
    raise ValueError("you database is not intialize")
  
  chain = RetrievalQAWithSourcesChain.from_llm(llm=llm ,retriever = vector_store.as_retriever())
  result = chain.invoke({"question" : query} , return_only_outputs = True)
  sources = result.get("sources" , "")
  return result["answer"] , sources 





if __name__ == "__main__" :
  urls = [
        "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html"
    ]

   
  process_url(urls)
    

  
  answer , sources = genrate_answer("Tell me what was the 30 year fixed mortagate rate along with the date?")
  print(f"answer : {answer}")
  print(f"sources : {sources}") 



