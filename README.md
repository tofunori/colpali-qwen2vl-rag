# ColPali + Qwen2-VL Multimodal RAG System

This repository contains a Multimodal Retrieval-Augmented Generation (RAG) system that combines:
- **ColPali**: A document retrieval model that processes both text and images in PDFs
- **Qwen2-VL**: A powerful Vision-Language Model (VLM) that understands both text and visual content

With this system, you can ask questions about PDF documents, and get answers that take into account both the textual and visual content of the documents.

## 📋 Features

- PDF document indexing that captures both text and visual information
- Semantic search across your document collection
- Question answering based on document content including images, tables, and charts
- Support for multiple PDF documents in a collection

## 🔧 Requirements

- Python 3.8+ 
- CUDA-compatible GPU with at least 12GB VRAM (24GB+ recommended for optimal performance)
- Windows/Linux/macOS

## 🚀 Installation

### Setting up a Virtual Environment in VSCode

1. Clone this repository:
   ```bash
   git clone https://github.com/tofunori/colpali-qwen2vl-rag.git
   cd colpali-qwen2vl-rag
   ```

2. Open the project in VSCode:
   ```bash
   code .
   ```

3. Create a virtual environment:
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS)
   - Type "Python: Create Environment" and select it
   - Choose "Venv"
   - Select your Python interpreter

   Or via terminal:
   ```bash
   # Windows
   python -m venv venv
   
   # Linux/macOS
   python3 -m venv venv
   ```

4. Activate the virtual environment:
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Linux/macOS
   source venv/bin/activate
   ```

5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

6. Install poppler-utils (required for PDF processing):
   - **Windows**: Download from [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases/) and add to PATH
   - **Linux**: `sudo apt-get install -y poppler-utils`
   - **macOS**: `brew install poppler`

## 💻 Usage

### 1. Index Your Documents

```python
from multimodal_rag import RAGSystem

# Initialize the system
rag_system = RAGSystem()

# Index a folder containing PDFs
rag_system.index_documents(folder_path="your_documents/")

# Or index individual PDFs
rag_system.index_documents(file_paths=["document1.pdf", "document2.pdf"])
```

### 2. Ask Questions About Your Documents

```python
# Simple question answering
answer = rag_system.answer_question("What are the key features of product X?")
print(answer)

# Advanced configuration
answer = rag_system.answer_question(
    "How do I assemble this furniture?",
    top_k=5,  # Number of relevant pages to consider
    max_tokens=500  # Maximum length of the answer
)
```

### 3. Using the CLI Tool

```bash
# Index documents
python cli.py index --folder ./my_pdfs/

# Ask a question
python cli.py ask "What does figure 3 show about sales trends?"
```

## 📊 Examples

### Example 1: Technical Documentation

```python
rag_system = RAGSystem()
rag_system.index_documents(file_paths=["technical_manual.pdf"])

# The system can interpret technical diagrams and charts
question = "Explain the signal flow in Figure 2 of the manual."
answer = rag_system.answer_question(question)
print(answer)
```

### Example 2: Research Papers

```python
rag_system = RAGSystem()
rag_system.index_documents(folder_path="research_papers/")

# The system can analyze graphs and data in research papers
question = "What were the experimental results for the control group as shown in Figure 4?"
answer = rag_system.answer_question(question)
print(answer)
```

## 🔄 How It Works

1. **Document Indexing**:
   - PDFs are converted to images page by page
   - ColPali processes each page and creates embeddings that capture both text and visual content
   - These embeddings are stored in a vector database

2. **Retrieval**:
   - When a question is asked, it's converted into the same embedding space
   - The system retrieves the most similar document pages

3. **Generation**:
   - Retrieved pages (as images) and the question are sent to Qwen2-VL
   - Qwen2-VL generates an answer based on both the visual and textual content of the pages

## 🛠️ Advanced Configuration

The system can be configured in several ways:

```python
from multimodal_rag import RAGSystem

rag_system = RAGSystem(
    colpali_model="vidore/colpali-v1.2",  # ColPali model to use
    vlm_model="Qwen/Qwen2-VL-7B-Instruct",  # VLM model to use
    index_path="./index",  # Where to store the index
    use_gpu=True,  # Whether to use GPU
    low_memory=False  # Low memory mode (slower but uses less VRAM)
)
```

## 📝 Resources

- [ColPali Paper](https://arxiv.org/abs/2403.04180) - Original paper describing the ColPali model
- [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) - Qwen2-VL model on Hugging Face
- [Byaldi](https://github.com/AnswerDotAI/byaldi) - Library used to interact with ColPali

## ⚠️ Limitations

- Requires significant GPU memory for optimal performance
- Processing time scales with the number and size of documents
- While the system performs well on many types of documents, extremely complex layouts might still present challenges
- The quality of answers depends on both the relevance of retrieved pages and the capabilities of the underlying VLM

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
