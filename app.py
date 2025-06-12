import streamlit as st
import os
from langchain.chat_models import init_chat_model
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from document_processor import *


if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

if 'llm' not in st.session_state:
    st.session_state.llm = None

if 'show_summaries' not in st.session_state:
    st.session_state.show_summaries = True

URLS_TO_PROCESS = [
    "https://lilianweng.github.io/posts/2025-05-01-thinking/",
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-02-05-human-data-quality/",
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/",
    "https://lilianweng.github.io/posts/2023-01-10-inference-optimization/",
    "https://lilianweng.github.io/posts/2022-09-08-ntk/",
    "https://lilianweng.github.io/posts/2022-06-09-vlm/",
    "https://lilianweng.github.io/posts/2022-04-15-data-gen/",
    "https://lilianweng.github.io/posts/2021-12-05-semi-supervised/",
    "https://lilianweng.github.io/posts/2021-09-25-train-large/",
    "https://lilianweng.github.io/posts/2021-07-11-diffusion-models/",
    "https://lilianweng.github.io/posts/2021-05-31-contrastive/",
    "https://lilianweng.github.io/posts/2021-03-21-lm-toxicity/",
    "https://lilianweng.github.io/posts/2021-01-02-controllable-text-generation/",
    "https://lilianweng.github.io/posts/2020-10-29-odqa/",
    "https://lilianweng.github.io/posts/2020-08-06-nas/",
    "https://lilianweng.github.io/posts/2020-06-07-exploration-drl/",
    "https://lilianweng.github.io/posts/2020-04-07-the-transformer-family/",
    "https://lilianweng.github.io/posts/2020-01-29-curriculum-rl/"
]

def initialize_vector_store():
    """Initialize the vector store with the embedding model"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return Chroma(
        collection_name="rag_docs",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

def ask_question(query: str, vector_store):
    relevant_docs = vector_store.similarity_search(query, k=4)
    context_parts = []
    unique = []
    context_map = {}  # Store mapping of citation numbers to full context chunks

    #format contect for prompt injection
    for i, doc in enumerate(relevant_docs, 1):
        source = doc.metadata.get('source', 'Unknown source')
        title = doc.metadata.get('title', 'Untitled')
        page = doc.metadata.get('page', '')
        
        if source not in unique:
            unique.append(source)

        # Format the source information
        source_info = f"Source {unique.index(source) + 1}: {title}"
        if page != '':
            source_info += f" (Page {page})"
        if source:
            source_info += f" from {source}"
        
        citation_num = unique.index(source) + 1
        context_parts.append(f"[{citation_num}] {doc.page_content}\n{source_info}")
        # Store the full context chunk with its citation number
        context_map[citation_num] = {
            'content': doc.page_content,
            'source_info': source_info
        }
    
    context = "\n\n".join(context_parts) #build our full context to give to prompt

    # First LLM response
    prompt = (
        "You are a helpful and knowledgeable assistant. Use the context below to answer the user's question clearly, accurately, and thoroughly. "
        "Use as much relevant information from the context as needed — do not skip important details.\n\n"
        
        "Do not use the context word for word in your response, you should use the context to answer the question, but not cite it word for word\n\n"
        
        "group the facts and cite the source once at the end of the sentence or paragraph.\n\n"
        "For example, cite a fact from the first source as [1].\n\n"
        
        "After the answer, include a 'Citations:' section. Start each citation with a bullet point."
        "If a context chunk is un used, Do not include it in citations."
        "\"[n] Title – Source from URL \" and then insert a line break \n\n "
        "If only one source is used, still use [x] for the single citaion"
        
        "IMPORTANT: If there are multiple contexts eg.([1],[2][3]) all from the same url or document, Do not repeat identical citations \n\n "
        
        "IMPORTANT: If you cannot find the answer in the context, then you should output 'I don't have that information' and DO NOT CITE ANY SOURCES.\n\n "
        
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    response = st.session_state.llm.invoke(prompt)
    
   
    if st.session_state.show_summaries: # toggle
        import re
        citations = re.findall(r'\[(\d+)\]', response.content)
        citations = list(set(citations))  
        
        # Get the cited context chunks
        cited_contexts = []
        for citation in citations:
            if int(citation) in context_map:
                cited_contexts.append(f"[{citation}] {context_map[int(citation)]['content']}\n{context_map[int(citation)]['source_info']}")
        
        
        summary_prompt = (
            "You are a helpful assistant that summarizes the context chunks used in a response. "
            "Below are the raw context chunks that were cited in the response. "
            "For each cited chunk, provide a separate summary clearly labeled with its citation number.\n\n"
            "Do not use the context word for word in your response, you should use the context to answer the question, but not cite it word for word\n\n"
            
            "Cited Context Chunks:\n" + "\n\n".join(cited_contexts) + "\n\n"
            "Provide a separate summary for each cited chunk in the following format:\n"
            "Source [X] Summary: [Your summary of this specific source]\n\n"
            "Make sure to maintain the citation numbers from the original chunks and provide a clear, separate summary for each source.\n\n"
        )
        summary_response = st.session_state.llm.invoke(summary_prompt)
        return response.content, summary_response.content
    else:
        return response.content, None


#__Streamlit App__
st.title("Thomas Rzeznik")
st.subheader("Data Wrangler Chatbot")

st.header("Configuration")
with st.expander("API Keys", expanded=False):
    hf_token = st.text_input("HuggingFace Token", type="password")
    langsmith_key = st.text_input("LangSmith API Key", type="password")
    google_key = st.text_input("Google API Key", type="password")
    
    if st.button("Save API Keys"):
        os.environ["HF_TOKEN"] = hf_token
        os.environ["LANGSMITH_API_KEY"] = langsmith_key
        os.environ["GOOGLE_API_KEY"] = google_key
        os.environ["LANGSMITH_TRACING"] = "true"
    
        st.session_state.llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
        st.session_state.vector_store = initialize_vector_store()
        st.success("API keys saved and models initialized!")

with st.expander("Admin Panel", expanded=False):

    if st.button("Process All URLs"):
        if st.session_state.vector_store is None:
            st.error("Please save API keys first!")
        else:
            total_chunks = 0
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, url in enumerate(URLS_TO_PROCESS):
                status_text.text(f"Processing URL {i+1}/{len(URLS_TO_PROCESS)}: {url}")
                try:
                    chunks = process_url_documents(url, st.session_state.vector_store)
                    total_chunks += chunks
                    st.success(f"Added {chunks} chunks from {url}")
                except Exception as e:
                    st.error(f"Error processing {url}: {str(e)}")
                
                
                progress = (i + 1) / len(URLS_TO_PROCESS)
                progress_bar.progress(progress)
            
            status_text.text(f"Completed! Total chunks added: {total_chunks}")
            progress_bar.empty()
    elif st.button("Process All IRS PAGES"):

        total_chunks = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        irs_urls = get_irs_urls()
        for i, url in enumerate(irs_urls):
            status_text.text(f"Processing URL {i+1}/{len(irs_urls)}: {url}")
            try:
                chunks = process_url_documents(url, st.session_state.vector_store)
                total_chunks += chunks
                st.success(f"Added {chunks} chunks from {url}")
            except Exception as e:
                st.error(f"Error processing {url}: {str(e)}")
            
            
            progress = (i + 1) / len(irs_urls)
            progress_bar.progress(progress)
        
        status_text.text(f"Completed! Total chunks added: {total_chunks}")
        progress_bar.empty()
    elif st.button("Clear Database", type="secondary"):
        if st.session_state.vector_store is not None:
            st.session_state.vector_store.delete_collection()
            st.session_state.vector_store = initialize_vector_store()
            st.success("Database cleared successfully!")
        else:
            st.warning("No database to clear. Please initialize the system first.")

st.header("Document Upload")

if st.session_state.vector_store is not None:
    try:
        count = st.session_state.vector_store._collection.count()
        st.info(f"Current database contains {count} document chunks")
    except:
        st.info("Database is empty")

upload_tab, url_tab = st.tabs(["Upload PDF", "Add URL"])

with upload_tab:
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file and st.button("Process PDF"):
        if st.session_state.vector_store is None:
            st.error("Please save API keys first!")
        else:
            
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getvalue())
            
            
            with st.spinner("Processing PDF..."):
                chunks = process_pdf_documents("temp.pdf", st.session_state.vector_store)
                st.success(f"Added {chunks} chunks from PDF!")
            
            
            os.remove("temp.pdf")

with url_tab:
    url = st.text_input("Enter URL to process")
    if url and st.button("Process URL"):
        if st.session_state.vector_store is None:
            st.error("Please save API keys first!")
        else:
            with st.spinner("Processing URL..."):
                chunks = process_url_documents(url, st.session_state.vector_store)
                st.success(f"Added {chunks} chunks from URL!")


st.header("Chat")
if st.session_state.vector_store is None or st.session_state.llm is None:
    st.warning("Please configure API keys and initialize the system first!")
else:
    
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Reset Chat History", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()
    with col2:
        st.session_state.show_summaries = st.toggle("Show Context Summaries", value=st.session_state.show_summaries)
    
    # history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant" and "summary" in message and message["summary"] is not None:
                st.info("Context Summary: \n" + message["summary"])
    
    if prompt := st.chat_input("Ask a question"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
    
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, summary = ask_question(prompt, st.session_state.vector_store)
                st.write(response)
                if summary:
                    st.info("Context Summary: " + summary)
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": response,
                    "summary": summary
                }) 