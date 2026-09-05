# 🦷 DentGuide AI

### Evidence-based answers from trusted dental guidelines

**DentGuide AI** is a Retrieval-Augmented Generation (RAG) application that allows dental professionals, students, and researchers to ask questions about uploaded clinical guidelines and receive answers grounded in the source documents.

The application is designed specifically for large dental guideline PDFs, including documents containing scanned/image-based pages.

Users can upload documents such as:

* **Therapeutic Guidelines 2025**
* **ADA Guidelines for Infection Prevention and Control — Fifth Edition, Amended**

and ask questions such as:

> "What does the guideline recommend for acute odontogenic infections?"

> "When is antibiotic prophylaxis recommended for dental procedures?"

> "What are the standard precautions for infection control?"

> "What PPE is recommended during dental procedures?"

The system retrieves the most relevant passages from the uploaded guidelines and asks the AI model to answer **only from those retrieved sources**.

---

## ✨ Key Features

### 📚 Multi-PDF guideline library

Upload multiple dental guideline PDFs at the same time.

The current project was designed around:

* Therapeutic Guidelines 2025 — 356 pages
* ADA Infection Prevention and Control Guidelines — Fifth Edition, Amended

---

### 🔎 Semantic search

DentGuide AI does not simply search for exact keywords.

It converts guideline passages into vector embeddings and performs semantic similarity search to find passages relevant to the user's question.

For example:

> "What antibiotics are recommended for a spreading dental infection?"

can retrieve relevant sections even when the wording in the guideline differs from the question.

---

### 🤖 AI-generated answers

The retrieved passages are provided to the language model, which generates a concise answer based on the available evidence.

The system is instructed to:

* Use only retrieved guideline information
* Avoid inventing recommendations
* Preserve important qualifiers
* Preserve dosage and duration information
* Identify exceptions
* State when evidence is insufficient
* Cite the relevant document and PDF page

---

### 📄 Page-aware citations

The application maintains the original PDF page number during processing.

Answers can therefore reference sources such as:

```text
[Therapeutic Guidelines 2025 — PDF p. 132]
```

or

```text
[ADA Guidelines for Infection Prevention and Control — PDF p. 14]
```

The application also provides an expandable **Retrieved Evidence** section so users can inspect the passages used to generate the answer.

---

### 🖨️ OCR support

A major feature of this project is support for scanned/image-based PDFs.

The Therapeutic Guidelines PDF contains pages where conventional PDF text extraction provides little or no useful text.

DentGuide AI therefore uses a multi-stage extraction strategy:

```text
PDF
 │
 ├── Native text available?
 │       │
 │       └── Yes → Extract text
 │
 └── No / insufficient text
         │
         ├── Tesseract OCR
         │
         └── OpenAI vision OCR fallback
```

This makes the application suitable for older scanned guideline documents as well as digitally generated PDFs.

---

## 🧠 Architecture

DentGuide AI uses a Retrieval-Augmented Generation architecture.

```text
                  ┌─────────────────────┐
                  │     PDF Guidelines  │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ PDF Text Extraction │
                  │      + OCR           │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Page-aware Chunking │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Text Embeddings     │
                  │ text-embedding-3-   │
                  │ small               │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Semantic Retrieval  │
                  │ Top relevant chunks │
                  └──────────┬──────────┘
                             │
                     User Question
                             │
                             ▼
                  ┌─────────────────────┐
                  │    GPT-5.6 Sol      │
                  │ Grounded generation │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Answer + Citations  │
                  │ + Retrieved Evidence│
                  └─────────────────────┘
```

---

# 🛠️ Technology Stack

| Component            | Technology                         |
| -------------------- | ---------------------------------- |
| User Interface       | Streamlit                          |
| Programming Language | Python                             |
| LLM                  | GPT-5.6 Sol                        |
| Embeddings           | text-embedding-3-small             |
| PDF Processing       | PyMuPDF                            |
| OCR                  | Tesseract / OpenAI vision fallback |
| Numerical Processing | NumPy                              |
| Image Processing     | Pillow                             |
| Deployment           | Streamlit Community Cloud          |
| Version Control      | Git / GitHub                       |

---

# 📁 Project Structure

```text
DentGuide-AI/
│
├── app.py
├── requirements.txt
├── README.md
│
└── .gitignore
```

Do **not** upload your PDF guideline files or API keys to GitHub unless you have the appropriate rights and intentionally want to distribute those documents.

---

# 🚀 Getting Started

## 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/DentGuide-AI.git
```

Move into the project:

```bash
cd DentGuide-AI
```

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 Configure the OpenAI API Key

Do **not** put your API key directly inside `app.py`.

For local development, you can set an environment variable.

### Windows PowerShell

```powershell
$env:OPENAI_API_KEY="your-api-key"
```

Or use Streamlit secrets.

Create:

```text
.streamlit/secrets.toml
```

with:

```toml
OPENAI_API_KEY = "your-api-key"
```

Add `.streamlit/secrets.toml` to `.gitignore` so that your key is never committed to GitHub.

---

# ▶️ Run the Application

Start Streamlit:

```bash
streamlit run app.py
```

The application will open in your browser.

Upload your guideline PDFs through the sidebar.

After processing is complete, you can begin asking questions.

---

# 💬 Example Questions

## Therapeutic Guidelines

```text
What is the recommended management of acute odontogenic infections?
```

```text
What does the guideline recommend for spreading odontogenic infections?
```

```text
What are the indications for antibiotic prophylaxis for dental procedures?
```

```text
What antimicrobial options are recommended for patients with antimicrobial hypersensitivity?
```

---

## Infection Prevention and Control

```text
What are the standard precautions in dental practice?
```

```text
What are the recommended hand hygiene moments?
```

```text
What PPE should be used during dental procedures?
```

```text
How should reusable medical devices be processed?
```

```text
What should be done following a sharps injury?
```

---

# 🛡️ Grounding and Hallucination Protection

One of the main design goals of DentGuide AI is to prevent the AI from behaving like a general medical chatbot.

The model is explicitly instructed:

```text
Use ONLY the retrieved excerpts from the uploaded guidelines.
```

If the retrieved material does not contain sufficient information, the model is instructed to respond:

```text
I could not find enough information in the uploaded guidelines
to answer this reliably.
```

The model is also instructed not to fabricate:

* Drug doses
* Treatment durations
* Indications
* Contraindications
* Recommendations
* Guideline citations
* Page numbers

This is particularly important for clinical guideline applications.

---

# 📖 Source Prioritization

DentGuide AI recognizes the different roles of the uploaded documents.

### Therapeutic Guidelines

Used primarily for:

* Therapeutics
* Pharmacology
* Antimicrobial therapy
* Dental infections
* Antibiotic prophylaxis
* Drug-related questions
* Clinical management recommendations

### ADA Infection Prevention and Control Guidelines

Used primarily for:

* Infection prevention
* Hand hygiene
* PPE
* Sterilisation
* Disinfection
* Reprocessing
* Environmental cleaning
* Sharps
* Waste management
* Occupational exposure
* Dental practice infection control

If a question requires information from both documents, the application can retrieve evidence from both.

---

# ⚕️ Clinical Safety

DentGuide AI is an **educational and reference tool**.

It should not be considered a replacement for:

* Professional clinical judgement
* The original guideline
* Local regulations
* Local institutional policies
* Manufacturer Instructions for Use (IFUs)
* Specialist consultation when required

Users should verify important clinical decisions against the **original guideline document**.

Guidelines may also be updated over time. Always check that the version uploaded to the application is the current applicable edition.

---

# 📊 Current Retrieval Process

The application currently uses:

```text
Chunk size:       2200 characters
Chunk overlap:     350 characters
Top results:       10
Embedding model:   text-embedding-3-small
```

The chunks retain:

* Document name
* PDF page number
* Chunk number
* Extracted text

This allows the application to connect an answer back to the original source.

---

# 🖨️ OCR Pipeline

For every PDF page, the application first attempts native text extraction.

If insufficient text is found:

```text
Native PDF extraction
        │
        ▼
Enough text?
   ┌────┴────┐
  YES        NO
   │          │
   ▼          ▼
Use text   Tesseract OCR
              │
              ▼
        Enough text?
          ┌───┴───┐
         YES      NO
          │        │
          ▼        ▼
       Use OCR   Vision OCR
```

This was added because the Therapeutic Guidelines document supplied for the project contains many image/scanned pages.

---

# ☁️ Streamlit Deployment

DentGuide AI can be deployed using Streamlit Community Cloud.

Your GitHub repository should contain:

```text
app.py
requirements.txt
README.md
```

Then connect the GitHub repository to Streamlit.

Configure the API key through Streamlit's **Secrets** configuration.

Example:

```toml
OPENAI_API_KEY = "your-api-key"
```

Do not commit the API key to GitHub.

---

# 🔧 Environment Variables

The default models are defined as:

```text
OPENAI_CHAT_MODEL=gpt-5.6-sol
OPENAI_EMBED_MODEL=text-embedding-3-small
```

They can be overridden using environment variables:

```bash
OPENAI_CHAT_MODEL=your-model
```

```bash
OPENAI_EMBED_MODEL=your-embedding-model
```

---

# ⚠️ Current Limitations

This is the first production-oriented MVP.

### 1. Index persistence

The current version builds the document index during the Streamlit session.

If the application restarts, the documents may need to be uploaded and indexed again.

### 2. OCR processing time

Large scanned PDFs can require significant processing time, particularly when vision OCR is required.

### 3. In-memory vector index

The current version stores embeddings in memory rather than using a persistent vector database.

### 4. Table-heavy PDFs

Complex clinical tables may require additional parsing improvements to preserve their exact structure.

### 5. Guideline version management

The current MVP does not yet provide a full document version-management system.

---

# 🚀 Future Development

The project can be expanded into a more advanced clinical guideline platform.

Planned improvements include:

* [ ] Persistent vector database
* [ ] Persistent document library
* [ ] Automatic document version tracking
* [ ] Guideline update management
* [ ] Better table extraction
* [ ] Hybrid keyword + semantic search
* [ ] Reranking of retrieved passages
* [ ] Better citation verification
* [ ] Document/page viewer
* [ ] Highlight the exact supporting passage
* [ ] Therapeutic Guidelines / IPC category filters
* [ ] Drug-specific search
* [ ] Compare recommendations between guidelines
* [ ] Multi-turn conversational context
* [ ] User authentication
* [ ] Admin document management
* [ ] OCR quality checking
* [ ] Audit logs
* [ ] Feedback on answer quality
* [ ] Evaluation dataset for retrieval accuracy
* [ ] Automated RAG evaluation
* [ ] Support for additional dental guidelines

---

# 🧪 Recommended Evaluation

Before using DentGuide AI for serious clinical reference, create a test set of questions from the source documents.

For example:

```text
Question
Expected answer
Expected document
Expected page
Expected supporting passage
```

Then evaluate:

### Retrieval accuracy

Did the application retrieve the correct section?

### Citation accuracy

Does the cited page actually contain the supporting information?

### Answer accuracy

Does the answer correctly represent the guideline?

### Hallucination rate

Did the model introduce information not present in the guideline?

### Abstention accuracy

Does the model correctly say "not enough information" when the retrieved sources do not support an answer?

---

# 🧑‍💻 Development Philosophy

DentGuide AI follows a simple principle:

> **Retrieve first. Answer second.**

The LLM should not be treated as the knowledge base.

Instead:

```text
Guideline
    ↓
Evidence retrieval
    ↓
Relevant passages
    ↓
LLM reasoning
    ↓
Cited answer
```

This makes the application more suitable for document-grounded clinical question answering than a conventional general-purpose chatbot.

---

# 📜 Disclaimer

DentGuide AI is an experimental/educational software project for retrieving and presenting information from uploaded dental guideline documents.

It is not a medical device and should not be relied upon as the sole basis for diagnosis, treatment, prescribing, infection-control decisions, or other clinical decisions.

Always consult the original guideline and apply appropriate professional judgement and applicable local requirements.

---

# ⭐ Project Name

## DentGuide AI

**Tagline:**

> **Evidence-based answers from trusted dental guidelines.**

---

# 👨‍💻 Author

Developed as an AI-powered dental guideline retrieval and clinical knowledge project.

Built with:

**Python · Streamlit · OpenAI · PyMuPDF · OCR · RAG**

---

## ⭐ If you find this project useful

Consider starring the repository ⭐ and contributing improvements to the project.
