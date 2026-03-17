# 🩺 Medical AI Chatbot RAG

A Retrieval-Augmented Generation (RAG) powered chatbot designed to provide accurate, context-aware answers about diabetes and related medical topics using advanced AI models.

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **Intelligent Q&A**: Answers questions about diabetes using medical literature
- **RAG Technology**: Combines retrieval from medical documents with generative AI
- **Streamlit Interface**: User-friendly web interface for easy interaction
- **Google Gemini Integration**: Powered by Google's latest Gemini 2.5-flash model
- **Vector Search**: Fast similarity search using FAISS and sentence transformers
- **Medical Focus**: Specialized for diabetes-related queries with medical accuracy

## 🏗️ Architecture

The chatbot uses a RAG (Retrieval-Augmented Generation) architecture:

1. **Document Processing**: Medical PDFs are loaded and split into manageable chunks
2. **Embedding Generation**: Text chunks are converted to vector embeddings using HuggingFace models
3. **Vector Storage**: Embeddings are stored in a FAISS vector database for fast retrieval
4. **Query Processing**: User questions are embedded and matched against stored vectors
5. **Context Retrieval**: Relevant medical information is retrieved from the vector store
6. **Answer Generation**: Google Gemini LLM generates accurate answers using retrieved context

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Google AI API key (for Gemini model)
- Virtual environment (recommended)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Medical_AI_Chatbot_RAG
```

### Step 2: Create Virtual Environment

```bash
# On Windows
python -m venv medi
medi\Scripts\activate

# On macOS/Linux
python -m venv medi
source medi/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

For development dependencies:

```bash
pip install -r requirements-dev.txt
```

## ⚙️ Setup

### 1. Prepare Medical Documents

Place your medical PDF documents in the `data/` directory. The system comes pre-configured with:
- Diabetes Care Booklet.pdf
- The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf

### 2. Create Vector Store

Run the document processing script to create the FAISS vector store:

```bash
python mem_llm.py
```

This will:
- Load all PDFs from `data/`
- Split documents into chunks (500 characters with 50 overlap)
- Generate embeddings using sentence-transformers
- Create and save FAISS index in `vectorstore/db_faiss/`

### 3. Configure API Key

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_google_ai_api_key_here
```

Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

## 💻 Usage

### Start the Chatbot

```bash
streamlit run medibot.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Interact with the Chatbot

1. Type your question about diabetes in the chat input
2. The bot will retrieve relevant information from medical documents
3. Receive accurate, context-based answers powered by AI

### Example Queries

- "What are the symptoms of diabetes?"
- "How is diabetes diagnosed?"
- "What are the treatment options for type 2 diabetes?"
- "Explain the complications of uncontrolled diabetes"

## 📁 Project Structure

```
Medical_AI_Chatbot_RAG/
├── medibot.py              # Main Streamlit application
├── conn_llm.py             # LLM connection and RAG chain setup
├── mem_llm.py              # Document processing and vector store creation
├── requirements.txt        # Production dependencies
├── requirements-dev.txt    # Development dependencies
├── .env                    # Environment variables (API keys)
├── data/                   # Medical PDF documents
│   ├── Diabetes Care Booklet.pdf
│   └── The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf
├── vectorstore/            # FAISS vector database
│   └── db_faiss/
│       └── index.faiss
├── medi/                   # Python virtual environment
└── __pycache__/            # Python cache files
```

## 📦 Dependencies

### Core Dependencies

- **Streamlit**: Web application framework
- **LangChain**: Framework for LLM applications
- **FAISS**: Vector similarity search
- **Google Generative AI**: Gemini LLM integration
- **HuggingFace Transformers**: Embedding models
- **Sentence Transformers**: Text embedding generation
- **PyPDF**: PDF document loading

### Key Libraries

- `langchain-google-genai`: Google AI integration
- `langchain-community`: Community components
- `streamlit`: Web UI
- `faiss-cpu`: Vector database
- `sentence-transformers`: Embeddings
- `python-dotenv`: Environment management

## 🔧 Configuration

### Environment Variables

- `GOOGLE_API_KEY`: Your Google AI API key

### Vector Store Settings

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Chunk Size**: 500 characters
- **Chunk Overlap**: 50 characters
- **Retriever K**: 3 (top similar documents)

### LLM Settings

- **Model**: `gemini-2.5-flash`
- **Temperature**: 0.3 (balanced creativity/accuracy)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This chatbot is for informational purposes only and should not replace professional medical advice. Always consult with qualified healthcare providers for medical concerns and treatment decisions.

---

**Built with ❤️ using Streamlit, LangChain, and Google Gemini**</content>
<parameter name="filePath">c:\Users\HP\Desktop\Dhruvil\Medical_AI_Chatbot_RAG\README.md