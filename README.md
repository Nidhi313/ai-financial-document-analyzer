# Financial Document Analyzer

A privacy-preserving, AI-powered Streamlit application that enables users to upload and analyze financial documents using local Large Language Models (LLMs). This tool combines retrieval-augmented generation (RAG) with dynamic data visualizations and financial metric extraction for informed decision-making.

## Key Features

- ** Private LLM Chat**: Interact with a locally hosted Mistral 7B quantized model (via `ctransformers`) for general finance queries.
- ** Document Q&A**: Upload PDFs, Excel files, or TXT documents and ask document-specific questions with AI-generated answers and source traceability.
- ** Financial Insights**: Extract and summarize key metrics such as **Revenue**, **Net Income**, and **EBITDA** from documents.
- ** Auto Visualizations**: View basic trend charts using embedded financial data.
- ** Raw Data Inspection**: Explore how your financial documents are segmented and processed internally.

## AI and NLP Techniques Used

- **LangChain RAG Pipeline** for document-based question answering.
- **Sentence Transformers** (`all-MiniLM-L6-v2`) for semantic vector embeddings.
- **ChromaDB** for fast and local vector search.
- **Local LLM via `ctransformers`** for low-latency generation with privacy.
- **Altair** for elegant data visualization.

## Getting Started

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download and Add a Local Quantized Model
Download a quantized .gguf file of Mistral-7B-Instruct from a trusted source (e.g., Hugging Face), and update the MODEL_PATH in app.py:

```bash
MODEL_PATH = r"C:\path\to\your\model\mistral-7b-instruct-v0.2.Q4_K_M.gguf"
```

3. Run the app:

```bash
streamlit run app.py
```

## File Support

This app supports:

- .pdf files (PyMuPDF)
- .txt, .text files (Plain text)
- .xls, .xlsx (Unstructured Excel)
- Other formats handled using UnstructuredFileLoader

## Example Use Cases

- Company annual report summarization
- Private investment analysis
- Automated Q&A for CFO documents
- Rapid extraction of key performance indicators

## Future Enhancements

- Auto-charting from real extracted financial data
- Named Entity Recognition (NER) for company and metric tagging
- Support for additional languages and formats
- More customizable dashboards
