# StudyMate RAG - RLHF-Enabled Question Answering System

A Retrieval-Augmented Generation (RAG) system with Reinforcement Learning from Human Feedback (RLHF) capabilities. This system allows you to ask questions about your study materials and continuously improves based on your feedback.

## Features

- 📚 **Document Ingestion**: Process PDF documents and create vector embeddings
- 🔍 **RAG System**: Retrieve relevant context and generate answers using LLM
- 🎯 **RLHF Integration**: Learn from user feedback to improve answer quality
- 💾 **Feedback Storage**: Track all user ratings and feedback for analysis
- 🔄 **Dynamic Prompt Optimization**: Automatically adjusts prompts based on feedback

## Prerequisites

Before you begin, ensure you have the following installed:

1. **Python 3.8+**
2. **Ollama** - For running local LLM models
   - Install from: https://ollama.ai
   - Pull the Mistral model: `ollama pull mistral`

## Installation

### Step 1: Clone or Navigate to the Project

```bash
cd studymate-rag
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
pip install langchain-community langchain-huggingface langchain-core sentence-transformers
```

**Note**: If you encounter issues, you may need to install additional packages:
```bash
pip install langchain langchain-community langchain-huggingface langchain-core sentence-transformers faiss-cpu pypdf
```

### Step 4: Prepare Your Documents

Place your PDF files in the `data/` directory. The default setup expects `data/nishan.pdf`, but you can modify `ingest.py` to use your own PDF file.

## Usage Guide

### Step 1: Generate Embeddings (First Time Setup)

Before you can ask questions, you need to process your documents and create vector embeddings.

```bash
python ingest.py
```

**What this does:**
- Loads PDF documents from `data/nishan.pdf`
- Splits documents into chunks (600 characters with 100 overlap)
- Generates embeddings using HuggingFace's `all-MiniLM-L6-v2` model
- Stores the vector database in `vector_db/` directory

**Expected Output:**
```
✅ Ingestion complete
```

**Note**: This step only needs to be run once per document set. Re-run if you add new documents or want to update the knowledge base.

### Step 2: Start the Q&A System

Once embeddings are generated, start the interactive Q&A system:

```bash
python qa.py
```

### Step 3: Ask Questions

The system will prompt you to ask questions. Here's how it works:

```
Ask your question (type 'exit' to quit): What is machine learning?

Answer:
[System provides answer based on your documents]

Rate this answer (1-5, or 'skip' to skip feedback): 4
```

### Step 4: Provide Feedback

After each answer, you'll be asked to rate it:

- **1-2**: Poor answer (triggers prompt optimization)
- **3**: Neutral answer
- **4-5**: Good answer
- **skip**: Skip feedback for this answer

**How RLHF Works:**
- Low ratings (1-2) → Reward = -1.0 → Prompt is optimized to request clearer answers
- Neutral rating (3) → Reward = 0.5 → No prompt change
- High ratings (4-5) → Reward = 1.0 → No prompt change (system performing well)

## Example Session

```
$ python qa.py

Ask your question (type 'exit' to quit): What is Python?

Answer:
 Python is a high-level programming language known for its simplicity...

Rate this answer (1-5, or 'skip' to skip feedback): 2
✓ Prompt optimized based on your feedback (reward: -1.0)

Ask your question (type 'exit' to quit): What is JavaScript?

Answer:
[Answer will now be clearer and more concise due to optimization]

Rate this answer (1-5, or 'skip' to skip feedback): 5

Ask your question (type 'exit' to quit): exit
👋 Bye!
```

## Project Structure

```
studymate-rag/
├── data/
│   └── nishan.pdf          # Your PDF documents (add your own)
├── vector_db/              # Generated vector database (created after ingest.py)
├── feedback_store.json     # Stores all user feedback and ratings
├── ingest.py               # Document ingestion and embedding generation
├── qa.py                   # Main Q&A system with RLHF
├── reward_model.py         # Computes rewards from ratings
├── rlhf_loop.py            # Optimizes prompts based on rewards
├── feedback.py             # Collects and stores user feedback
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## How RLHF Integration Works

1. **User asks a question** → System retrieves relevant context and generates answer
2. **User rates the answer** → Rating (1-5) is collected
3. **Feedback is stored** → Saved to `feedback_store.json`
4. **Reward is computed**:
   - Rating 4-5 → Reward = 1.0 (positive)
   - Rating 3 → Reward = 0.5 (neutral)
   - Rating 1-2 → Reward = -1.0 (negative)
5. **Prompt optimization** → If reward is negative, prompt is updated to request clearer answers
6. **RAG chain update** → Future questions use the optimized prompt

## Configuration

### Changing the LLM Model

In `qa.py`, you can change the Ollama model:

```python
llm = Ollama(model="mistral")  # Change to "llama2", "codellama", etc.
```

Make sure you've pulled the model first: `ollama pull <model-name>`

### Changing the Embedding Model

In both `ingest.py` and `qa.py`, you can change the embedding model:

```python
embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Change to other models
)
```

### Processing Different PDFs

Edit `ingest.py` to point to your PDF:

```python
loader = PyPDFLoader("data/your-document.pdf")  # Change path here
```

## Troubleshooting

### Issue: "ModuleNotFoundError"

**Solution**: Make sure all dependencies are installed:
```bash
pip install langchain langchain-community langchain-huggingface langchain-core sentence-transformers faiss-cpu pypdf
```

### Issue: "Ollama model not found"

**Solution**: Pull the model first:
```bash
ollama pull mistral
```

### Issue: "vector_db not found"

**Solution**: Run `ingest.py` first to generate the vector database.

### Issue: "PDF file not found"

**Solution**: 
1. Ensure your PDF is in the `data/` directory
2. Update the path in `ingest.py` if using a different location

## Viewing Feedback Data

All feedback is stored in `feedback_store.json`. You can view it:

```bash
cat feedback_store.json
```

Or in Python:
```python
import json
with open("feedback_store.json", "r") as f:
    for line in f:
        feedback = json.loads(line)
        print(f"Question: {feedback['question']}")
        print(f"Rating: {feedback['rating']}\n")
```

## Next Steps

- Add more documents to expand your knowledge base
- Experiment with different LLM models
- Customize the reward model and prompt optimization logic
- Analyze feedback data to understand answer quality trends

## License

This project is for educational purposes.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

