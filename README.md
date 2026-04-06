# 🇺🇸 Obama Speeches — NLP Analytics Dashboard

A full Streamlit app converting your Jupyter notebook into an interactive dashboard.

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Run the app
```bash
streamlit run app.py
```

### 3. Open in browser
Streamlit will open automatically at: **http://localhost:8501**

---

## 📁 Files
```
app.py            ← Main Streamlit app
requirements.txt  ← Python dependencies
README.md         ← This file
```

## 📊 Features / Sections

| Section | What it does |
|---|---|
| 📊 Dataset Overview | Speech counts, year distribution, document types |
| 🔤 Text Preprocessing | Tokenization, stop words, stemming vs lemmatization |
| 📦 BoW & TF-IDF | Bag of Words + TF-IDF heatmap |
| 📐 N-Grams | Unigrams, bigrams, trigrams comparison |
| 🧬 Word Embeddings | Word2Vec + t-SNE visualization |
| 🤖 Classification | Naive Bayes, SVM, Logistic Regression |
| 💬 Sentiment Analysis | VADER + Aspect-Based (per policy domain) |
| 🗺️ Topic Modelling | LDA, LSA, topic assignments |
| 🔵 Clustering | K-Means + Hierarchical dendrogram |
| 🏷️ NER | spaCy Named Entity Recognition |
| 📝 Summarization | TextRank extractive summarization |
| ☁️ Word Clouds | Overall + per-topic word clouds |

## 📂 CSV Format Expected
Your CSV should have at minimum these columns:
- `content` — speech text
- `title` — speech title
- `document_date` — date (e.g. 2012-01-24)
- `document_type_name` (optional) — type of document

## ☁️ Deploy to Streamlit Cloud (Free)

1. Push this folder to a **GitHub repo**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo → select `app.py`
4. Add this to `packages.txt` in your repo root:
   ```
   python3-dev
   ```
5. Click **Deploy** — it's live!
