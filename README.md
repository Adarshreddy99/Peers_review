# AI-Driven Peer Review Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/Gradio-4.16.0-orange.svg)](https://gradio.app/)

An intelligent platform for academic integrity checking and automated feedback generation using AI and NLP techniques.

---

## 🌐 Live Demo

**Access the platform here:** https://huggingface.co/spaces/AdarshReddy99/ai-peer-review

---

## 📋 Table of Contents

- [Features](#features)
- [Supported File Types](#supported-file-types)
- [How to Use](#how-to-use)
- [Technical Architecture](#technical-architecture)
- [Local Installation](#local-installation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License](#license)
- [Authors](#authors)

---

## ✨ Features

### 1. **Plagiarism Detection**

#### Structural Analysis (AST-based)
- Detects code copying through Abstract Syntax Tree comparison
- Normalizes variable names and function names to catch renamed plagiarism
- Extracts structural features (loops, conditionals, nesting depth)
- Supports Python and Java files
- Requires minimum 2 compatible code files

#### Semantic Analysis (AI-powered)
- Uses Mistral-7B-Instruct LLM for deep semantic comparison
- Detects conceptual plagiarism and paraphrasing
- Identifies logic equivalence (e.g., `for` loops converted to `while`)
- Analyzes method rewriting and structural variations
- Maximum 4 files for optimal performance

#### Hybrid Scoring Algorithm
- Combines TF-IDF similarity with feature-based comparison
- Conservative weighting (70% lower score, 30% higher score)
- Reduces false positives
- Thresholds:
  - **≥70%:** High suspicion (flagged)
  - **50-70%:** Moderate similarity (reported)
  - **<50%:** Low similarity (not reported)

### 2. **AI Teaching Assistant Feedback**

- **Comprehensive Analysis:**
  - Relevance assessment
  - Pros and cons identification
  - Clarity evaluation
  - Technical accuracy verification
  - Methodology and numerical data review
  - Implementability scoring
  - Actionable improvement suggestions

- **Agentic Web Search:**
  - LLM autonomously decides when to search the web
  - Verifies factual claims in documents
  - Checks if information is up-to-date
  - Defines technical terms
  - Powered by Tavily API

- **Multi-Format Support:**
  - Code files (Python, Java)
  - Documents (PDF, TXT)
  - Combined code + documentation analysis

### 3. **Combined Comprehensive Report**

- Unified academic integrity and feedback analysis
- AI-only plagiarism check (semantic focus)
- Contextual feedback aware of integrity issues
- Professional formatting for instructors
- Single comprehensive evaluation

---

## 📁 Supported File Types

| File Type | Extensions | Used In |
|-----------|------------|---------|
| **Python Code** | `.py` | Plagiarism (AST + AI), Feedback |
| **Java Code** | `.java` | Plagiarism (AST + AI), Feedback |
| **PDF Documents** | `.pdf` | Plagiarism (AI only), Feedback |
| **Text Files** | `.txt` | Plagiarism (AI only), Feedback |

---

## 🚀 How to Use

### **Tab 1: Plagiarism Checker**

1. **Upload Files:**
   - Select 2 or more files (code, PDF, or TXT)
   - Maximum 4 files for AI check
   - Minimum 2 Python/Java files for AST check

2. **Select Analysis Types:**
   - ☑️ **Structural Check (AST)** - Code structure comparison
   - ☑️ **Semantic Check (AI)** - Deep conceptual analysis

3. **Click "Check Plagiarism"**

4. **Review Report:**
   - Structural similarity percentages
   - Semantic analysis with evidence
   - Verdict for each file pair

### **Tab 2: AI-TA Feedback**

1. **Enter Problem Statement:**
   - Describe the assignment or project goal
   - Be specific about requirements

2. **Upload Files (Optional):**
   - Code files in "Code Files" section
   - Documents in "Documents" section
   - No file limit (documents truncated to 4000 chars)

3. **Click "Generate Feedback"**

4. **Review Detailed Feedback:**
   - Relevance assessment
   - Strengths and weaknesses
   - Technical accuracy analysis
   - Improvement suggestions
   - Overall summary

### **Tab 3: Combined Report**

1. **Enter Problem Statement**

2. **Upload Files:**
   - Code files (optional)
   - Documents (optional)

3. **Click "Generate Combined Report"**

4. **Review Unified Report:**
   - Academic integrity analysis (AI-based)
   - Comprehensive feedback
   - Single professional document

---

## 🏗️ Technical Architecture

### **Core Technologies**

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Workflow Orchestration** | LangGraph | State machine for multi-step AI workflows |
| **Frontend** | Gradio 4.16.0 | Web interface with dark theme |
| **LLM** | Mistral-7B-Instruct | Natural language understanding and generation |
| **Code Analysis** | Python AST, Javalang | Parse and normalize code structure |
| **Text Similarity** | Scikit-learn TF-IDF | Document similarity calculation |
| **Web Search** | Tavily API | Fact-checking and verification |
| **PDF Processing** | PyMuPDF (fitz) | Text extraction from PDFs |

### **Key Algorithms**

#### 1. **AST Normalization**
```
Original Code → Parse to AST → Normalize Names → Extract Features → Compare
```
- Handles variable renaming
- Preserves API calls and libraries
- Tracks control flow patterns

#### 2. **TF-IDF Similarity**
```
TF-IDF(term, doc) = TF(term, doc) × IDF(term)
```
- TF: Term frequency in document
- IDF: Inverse document frequency (rarity)
- Cosine similarity for comparison

#### 3. **Hybrid Scoring**
```
combined_sim = (min_score × 0.7) + (max_score × 0.3)
```
- Conservative approach
- Reduces false positives
- Requires both structural AND textual similarity

### **LangGraph Workflow Example**
```
Plagiarism Graph:
├─ Entry & Preprocess (decode files, classify types)
├─ [Conditional: Run AST?]
│  ├─ YES → Normalize Code → Calculate Similarity
│  └─ NO → Skip
├─ [Conditional: Run AI?]
│  ├─ YES → LLM Semantic Analysis
│  └─ NO → Skip
└─ Compile Final Report → END

AI-TA Graph:
├─ Preprocess Files (extract text from PDFs)
├─ [Conditional: Has Documents?]
│  ├─ YES → Agentic Web Search (LLM decides)
│  └─ NO → Skip
└─ Generate Feedback (with web context) → END

Combined Graph:
├─ Run Plagiarism Check (AI-only)
├─ Run AI-TA Feedback (with plagiarism context)
└─ Synthesize Combined Report → END
```

---

## 💻 Local Installation

### **Prerequisites**

- Python 3.10 or higher
- pip package manager
- HuggingFace account and API token
- Tavily API key (for web search)

### **Step 1: Clone Repository**
```bash
git clone https://github.com/YOUR_USERNAME/ai-peer-review-platform.git
cd ai-peer-review-platform
```

### **Step 2: Create Virtual Environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### **Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 4: Configure Environment Variables**

Create a `.env` file in the project root:
```bash
HF_TOKEN=your_huggingface_token_here
TAVILY_API_KEY=your_tavily_api_key_here
```

**Get API Keys:**
- **HuggingFace Token:** https://huggingface.co/settings/tokens
- **Tavily API Key:** https://tavily.com (1000 free searches/month)

### **Step 5: Run Application**
```bash
python app.py
```

The app will launch at: `http://127.0.0.1:7860`

---

## 📂 Project Structure
```
ai-peer-review-platform/
│
├── src/                           # Core application modules
│   ├── __init__.py               # Package initializer
│   ├── graph_plagiarism.py       # Plagiarism detection graph
│   ├── graph_ai_ta.py            # AI feedback generation graph
│   ├── graph_combined.py         # Combined report graph
│   └── parsing_utils.py          # File parsing and AST utilities
│
├── app.py                         # Gradio web interface
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── .env.example                   # Environment variables template
└── .gitignore                     # Git ignore rules
```

### **Module Descriptions**

| File | Description |
|------|-------------|
| `graph_plagiarism.py` | LangGraph workflow for plagiarism detection (AST + AI) |
| `graph_ai_ta.py` | LangGraph workflow for feedback generation with web search |
| `graph_combined.py` | Unified report combining plagiarism and feedback |
| `parsing_utils.py` | AST normalization, feature extraction, PDF/TXT parsing |
| `app.py` | Gradio interface with three tabs and file upload handlers |

---

cators and students worldwide**
