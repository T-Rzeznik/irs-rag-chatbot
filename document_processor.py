import re
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import requests

def process_pdf_documents(pdf_path: str, vector_store) -> None:

    loader = PyMuPDFLoader(
        file_path=pdf_path,
        mode="page",  # Process page by page to maintain structure
        extract_tables="markdown"  
    )
    
    docs = loader.load()
    
    cleaned_docs = []
    for doc in docs:
        
        metadata = doc.metadata
        content = doc.page_content
    
        # get title for doc
        title = ""
        if metadata.get("page") == 0:  
            lines = content.split("\n")
            if lines:
                title = lines[0].strip()
        

        content = re.sub(r'^[A-Z\s]{10,}$', '', content, flags=re.MULTILINE)
        content = re.sub(r'\s+', ' ', content).strip()
        
        cleaned_doc = Document(
            page_content=f"Title: {title}\n\nContent: {content}",
            metadata={
                "source": os.path.basename(pdf_path),  
                "page": metadata.get("page", 0),
                "title": title or os.path.basename(pdf_path),  
                "type": "pdf"
            }
        )
        cleaned_docs.append(cleaned_doc)
    
   
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    

    splits = text_splitter.split_documents(cleaned_docs)
    
 
    vector_store.add_documents(documents=splits)
    
    return len(splits)  # return number of chunks created

def process_url_documents(url: str, vector_store) -> None:
   
    loader = WebBaseLoader(
        web_paths=(url,) 
    )
    
    docs = loader.load() # list of langchain document objects
    
    cleaned_docs = []
    for doc in docs:
    
        metadata = doc.metadata
        content = doc.page_content
        
        # get title for doc
        title = metadata.get("title", "")
        if not title and content:
            lines = content.split("\n")
            if lines:
                title = lines[0].strip() # use the first line for title
         
        #clean doc
        content = re.sub(r'^[A-Z\s]{10,}$', '', content, flags=re.MULTILINE)
        content = re.sub(r'\s+', ' ', content).strip()
        
        cleaned_doc = Document(
            page_content=f"Title: {title}\n\nContent: {content}",
            metadata={
                "source": url,
                "title": title or url,
                "type": "web"
            }
        )
        cleaned_docs.append(cleaned_doc)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    splits = text_splitter.split_documents(cleaned_docs)
    vector_store.add_documents(documents=splits)
    
    return len(splits)



def get_irs_urls():
    table_of_contents_re = re.compile(r"href=\"(https://www.irs.gov/irm/part\d+)\"") 
    pages_re = re.compile(r"href=\"(https://www.irs.gov/irm/part\d+/irm_\d+-\d+-\d+)\"")
    irs_url = "https://www.irs.gov/irm"

    response = requests.get(irs_url)
    parts = table_of_contents_re.findall(response.content.decode())
    
    all_pages = []
    for part in parts:
        response = requests.get(part)
        pages = pages_re.findall(response.content.decode())
        
        all_pages.extend(pages)
    return all_pages

        




