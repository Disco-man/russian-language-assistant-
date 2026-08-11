# Russian Language Assistant

A Streamlit app for analyzing Russian text with OCR, morphological and syntactic parsing.

## Features

- **OCR** — Extract text from images using ChatGPT Vision API
- **Morphological Analysis** — Identify parts of speech, lemmas, grammatical properties  
- **Syntactic Analysis** — Analyze word dependencies (subject, predicate, object, etc.)

## Quick Start

### Install
```bash
pip install streamlit spacy pandas pillow pytesseract python-dotenv requests
python -m spacy download ru_core_news_sm
```

### Configure
Create `сapi.env`:
```
OPENAI_API_KEY=your_api_key
```
Create `gapi.env`:
```
GEMINI_API_KEY=your_api_key
```

### Run
```bash
streamlit run app.py
```

Open browser at `http://localhost:8501`

## Usage

1. Upload image with text OR enter text manually
2. Click "📖 Распознать текст" to extract from image (optional)
3. Click "🔎 Разобрать предложение" to analyze
4. Click "🎲 Сгенерировать текст для анализа" to generate text to test yourself
5. View results in the table

