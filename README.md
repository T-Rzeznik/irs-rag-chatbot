# DataWranglerRag1

A powerful RAG (Retrieval-Augmented Generation) application that processes and queries documents using advanced language models. This application allows users to upload PDFs, process web URLs, and interact with the content through an intuitive chat interface.

## Features

- **Document Processing**
  - PDF document upload and processing
  - Web URL processing
  - Automatic text chunking and cleaning
  - Metadata extraction and management

- **Vector Storage**
  - ChromaDB integration for efficient document storage
  - Sentence transformer embeddings for semantic search
  - Persistent storage of processed documents

- **Interactive Chat Interface**
  - Streamlit-based user interface
  - Real-time document querying
  - Source citation and attribution
  - Toggle-able source summaries

- **Admin Features**
  - Bulk URL processing
  - Database management
  - API key configuration
  - Progress tracking for batch operations

## Prerequisites

- Python 3.10+
- Required API keys:
  - HuggingFace Token
  - LangSmith API Key
  - Google API Key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DataWranglerRag1.git
cd DataWranglerRag1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables or configure API keys through the application interface.

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Configure API Keys:
   - Open the "API Keys" expander
   - Enter your HuggingFace, LangSmith, and Google API keys
   - Click "Save API Keys"

3. Process Documents:
   - Upload PDFs through the "Upload PDF" tab
   - Add web URLs through the "Add URL" tab
   - Use the Admin Panel for bulk processing

4. Query Documents:
   - Use the chat interface to ask questions about your documents
   - View source citations and summaries
   - Toggle source summaries as needed

## Project Structure

- `app.py`: Main application file containing the Streamlit interface and core functionality
- `document_processor.py`: Document processing utilities for PDFs and web URLs
- `chroma_db/`: Directory for persistent vector storage
- `requirements.txt`: Project dependencies

## Dependencies

- langchain: Core RAG functionality
- streamlit: Web interface
- PyMuPDF: PDF processing
- chromadb: Vector storage
- sentence-transformers: Text embeddings
- beautifulsoup4: Web scraping
- Additional dependencies listed in requirements.txt