import warnings, os, logging
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
for _lg in ['gensim','sklearn','matplotlib','urllib3']:
    logging.getLogger(_lg).setLevel(logging.ERROR)

import streamlit as st
import pandas as pd
import numpy as np
import re
from io import StringIO
from collections import Counter, defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from wordcloud import WordCloud
from scipy.cluster.hierarchy import dendrogram, linkage

import nltk
import spacy
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords as nltk_sw

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation as SklearnLDA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import gensim.corpora as corpora
from gensim.models import LdaModel, Word2Vec
import networkx as nx

# ── Download NLTK data ──
@st.cache_resource
def download_nltk():
    for res in ['punkt','stopwords','averaged_perceptron_tagger',
                'maxent_ne_chunker','words','wordnet','vader_lexicon',
                'punkt_tab','averaged_perceptron_tagger_eng','maxent_ne_chunker_tab']:
        nltk.download(res, quiet=True)
    return True

@st.cache_resource
def load_spacy():
    try:
        return spacy.load('en_core_web_sm')
    except:
        os.system('python -m spacy download en_core_web_sm -q')
        return spacy.load('en_core_web_sm')

# ── Plot style ──
DARK_BG  = '#0f0f1a'
MID_BG   = '#1a1a2e'
PALETTE  = ['#e040fb','#40c4ff','#69ff47','#ffd740','#ff6e40',
            '#80deea','#ff4081','#b2ff59','#ea80fc','#ffab40']

plt.rcParams.update({
    'figure.facecolor': DARK_BG, 'axes.facecolor': MID_BG,
    'axes.edgecolor':'#444466','axes.labelcolor':'#ccccdd',
    'axes.titlecolor':'#ffffff','xtick.color':'#aaaacc',
    'ytick.color':'#aaaacc','text.color':'#ccccdd',
    'grid.color':'#2a2a4a','grid.alpha':0.4
})

DEV_KW = ['economy','jobs','infrastructure','education','healthcare',
          'middle class','growth','investment','innovation','opportunity',
          'energy','manufacturing','small business','workforce','climate',
          'development','employment','trade','technology','poverty']

ENGLISH_SW = None
ALL_SW     = None
lemmatizer = WordNetLemmatizer()
stemmer    = PorterStemmer()

def init_stopwords():
    global ENGLISH_SW, ALL_SW
    ENGLISH_SW = set(nltk_sw.words('english'))
    CUSTOM_SW  = {'applause','laughter','audience','mr','ms','mrs','also',
                  'would','could','said','let','like','one','two','shall',
                  'must','tonight','today','know','get','even','back','still'}
    ALL_SW = ENGLISH_SW | CUSTOM_SW

def preprocess(text, min_len=3):
    tokens = word_tokenize(str(text).lower())
    tokens = [t for t in tokens if t.isalpha() and t not in ALL_SW and len(t) >= min_len]
    tokens = [lemmatizer.lemmatize(t, pos='v') for t in tokens]
    return tokens

@st.cache_data(show_spinner=False)
def load_and_prepare(csv_bytes):
    df_raw = pd.read_csv(StringIO(csv_bytes.decode('utf-8')))
    df_raw = df_raw.dropna(subset=['content']).reset_index(drop=True)
    df_raw['document_date'] = pd.to_datetime(df_raw['document_date'], errors='coerce')
    df_raw['year'] = df_raw['document_date'].dt.year
    df_raw['content'] = df_raw['content'].astype(str)
    df_raw['title']   = df_raw['title'].astype(str)

    mask = (
        df_raw['content'].str.lower().apply(lambda t: any(k in t for k in DEV_KW))
        | df_raw['title'].str.lower().apply(lambda t: any(k in t for k in DEV_KW))
    )
    df = df_raw[mask].reset_index(drop=True)

    init_stopwords()
    df['tokens']      = df['content'].apply(preprocess)
    df['clean_text']  = df['tokens'].apply(lambda t: ' '.join(t))
    df['token_count'] = df['tokens'].apply(len)

    def assign_era(year):
        if pd.isna(year): return 'Unknown'
        y = int(year)
        if y <= 2010: return 'Crisis Era'
        elif y <= 2013: return 'Recovery Era'
        else: return 'Legacy Era'
    df['era'] = df['year'].apply(assign_era)

    # Sentiment
    vader = SentimentIntensityAnalyzer()
    POLITICAL_LEX = {'prosperity':3.0,'innovation':2.5,'investment':1.8,
                     'revitalize':2.5,'rebuild':2.0,'strengthen':2.0,
                     'crisis':-2.5,'recession':-2.5,'unemployment':-2.2,
                     'inequality':-2.0,'stagnation':-2.0,'opportunity':2.5,
                     'promise':2.0,'progress':2.2,'future':1.5,'together':1.5}
    vader.lexicon.update(POLITICAL_LEX)

    def sentiment_score(text):
        sc  = vader.polarity_scores(str(text)[:2000])
        c   = sc['compound']
        cat = 'Optimistic' if c >= 0.15 else ('Urgent' if c <= -0.05 else 'Neutral')
        return pd.Series({'compound':round(c,4),'pos':round(sc['pos'],4),
                          'neg':round(sc['neg'],4),'neu':round(sc['neu'],4),
                          'sentiment':cat})

    sent_df = df['content'].apply(sentiment_score)
    df[['compound','pos','neg','neu','sentiment']] = sent_df

    return df_raw, df, vader

# ════════════════════════════════════════════════════════
# STREAMLIT APP
# ════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Obama Speeches — NLP Analytics",
    page_icon="🇺🇸",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] { background-color: #0f0f1a; color: #ccccdd; font-family: 'Inter', sans-serif; }

h1, h2, h3 { font-family: 'Playfair Display', serif; }

.stApp { background-color: #0f0f1a; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12121f 0%, #1a1a2e 100%);
    border-right: 1px solid #2a2a4a;
}

.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #e040fb; }
.metric-val { font-family: 'IBM Plex Mono', monospace; font-size: 2rem; font-weight: 600; color: #e040fb; }
.metric-lbl { font-size: 0.78rem; color: #7777aa; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.2rem; }

.section-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #ffffff;
    border-left: 4px solid #e040fb;
    padding-left: 1rem;
    margin: 2rem 0 1rem 0;
}

.tag {
    display: inline-block;
    background: #1e1e3a;
    border: 1px solid #3a3a5a;
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
    font-size: 0.75rem;
    color: #aaaacc;
    margin: 2px;
    font-family: 'IBM Plex Mono', monospace;
}

.stSelectbox > div > div { background: #1a1a2e; border-color: #2a2a4a; }
.stFileUploader > div { background: #1a1a2e; border: 2px dashed #3a3a5a; border-radius: 10px; }

div[data-testid="stMetric"] {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 10px;
    padding: 0.8rem 1rem;
}

.stDataFrame { background: #1a1a2e; }

.highlight-box {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    color: #aaaacc;
}

.sentiment-optimistic { color: #69ff47; font-weight: 600; }
.sentiment-urgent { color: #ff4081; font-weight: 600; }
.sentiment-neutral { color: #ffd740; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem 0;'>
        <div style='font-family: Playfair Display, serif; font-size: 1.5rem; font-weight: 900; color: #e040fb;'>🇺🇸 Obama NLP</div>
        <div style='font-size: 0.72rem; color: #555577; letter-spacing: 0.1em; text-transform: uppercase;'>Text & Web Analytics</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("<div style='font-size:0.8rem; color:#7777aa;'>📡 Dataset loads automatically from Google Drive</div>", unsafe_allow_html=True)

    st.divider()
    SECTIONS = [
        "📊 Dataset Overview",
        "🔤 Text Preprocessing",
        "📦 BoW & TF-IDF",
        "📐 N-Grams",
        "🧬 Word Embeddings",
        "🤖 Classification",
        "💬 Sentiment Analysis",
        "🗺️ Topic Modelling",
        "🔵 Clustering",
        "🏷️ NER",
        "📝 Summarization",
        "☁️ Word Clouds",
    ]
    section = st.radio("Navigate", SECTIONS, label_visibility="collapsed")

# ── Google Drive Loader ──
GDRIVE_FILE_ID = "1g-FaNVQhPARRPFZehWT02KzeYi7hUOwu"

@st.cache_data(show_spinner=False)
def load_from_gdrive(file_id):
    import gdown, os, tempfile
    url = f"https://drive.google.com/uc?id={file_id}"
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
        tmp_path = tmp.name
    gdown.download(url, tmp_path, quiet=True, fuzzy=True)
    with open(tmp_path, 'rb') as f:
        data = f.read()
    os.unlink(tmp_path)
    return data

# ── Hero ──
st.markdown("""
<div style='text-align:center; padding: 3rem 2rem 2rem 2rem;'>
    <div style='font-family: Playfair Display, serif; font-size: 3.5rem; font-weight: 900; color: #ffffff; line-height: 1.1;'>
        Obama Speeches<br><span style='color: #e040fb;'>NLP Analytics</span>
    </div>
    <div style='color: #7777aa; margin: 1.5rem auto; max-width: 560px; font-size: 1rem; line-height: 1.7;'>
        Comprehensive text mining, sentiment analysis, topic modelling,<br>
        NER, word embeddings, and more — all in one dashboard.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Load data ──
download_nltk()
nlp_model = load_spacy()
init_stopwords()

with st.spinner("🔄 Loading dataset from Google Drive..."):
    csv_bytes = load_from_gdrive(GDRIVE_FILE_ID)

with st.spinner("🔄 Preprocessing speeches (this may take a few minutes on first load)..."):
    df_raw, df, vader = load_and_prepare(csv_bytes)

corpus = df['clean_text'].tolist()
titles = df['title'].tolist()

# ── Vectorizers (cached) ──
@st.cache_resource
def build_vectorizers(_corpus):
    bow_vec = CountVectorizer(max_features=500, min_df=2)
    bow_mat = bow_vec.fit_transform(_corpus)
    tfidf_vec = TfidfVectorizer(max_features=500, min_df=2, sublinear_tf=True)
    tfidf_mat = tfidf_vec.fit_transform(_corpus)
    return bow_vec, bow_mat, tfidf_vec, tfidf_mat

bow_vec, bow_mat, tfidf_vec, tfidf_mat = build_vectorizers(tuple(corpus))
bow_feat   = bow_vec.get_feature_names_out()
tfidf_feat = tfidf_vec.get_feature_names_out()
tfidf_dense = tfidf_mat.toarray()


# ════════════════════════════════════════════════════════
# SECTION RENDERING
# ════════════════════════════════════════════════════════

# ── 1. Dataset Overview ──
if section == "📊 Dataset Overview":
    st.markdown("<div class='section-header'>📊 Dataset Overview</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><div class='metric-val'>{len(df_raw):,}</div><div class='metric-lbl'>Total Speeches</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='metric-val'>{len(df):,}</div><div class='metric-lbl'>Dev. Speeches</div></div>", unsafe_allow_html=True)
    ymin = int(df['year'].min()) if not df['year'].isna().all() else '—'
    ymax = int(df['year'].max()) if not df['year'].isna().all() else '—'
    c3.markdown(f"<div class='metric-card'><div class='metric-val'>{ymin}–{ymax}</div><div class='metric-lbl'>Year Range</div></div>", unsafe_allow_html=True)
    avg_tok = int(df['token_count'].mean())
    c4.markdown(f"<div class='metric-card'><div class='metric-val'>{avg_tok:,}</div><div class='metric-lbl'>Avg Tokens/Speech</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(DARK_BG)

    if 'document_type_name' in df_raw.columns:
        tc = df_raw['document_type_name'].value_counts().head(8)
        axes[0].barh(range(len(tc)), tc.values, color=PALETTE[:len(tc)])
        axes[0].set_yticks(range(len(tc)))
        axes[0].set_yticklabels(tc.index, fontsize=9)
        axes[0].invert_yaxis()
        axes[0].set_title('Document Types in Full Dataset', fontweight='bold')
        axes[0].set_xlabel('Count')

    yc = df['year'].value_counts().sort_index().dropna()
    axes[1].bar(yc.index.astype(int), yc.values, color=PALETTE[1], edgecolor=DARK_BG, width=0.7)
    axes[1].set_title('Development Speeches by Year', fontweight='bold')
    axes[1].set_xlabel('Year'); axes[1].set_ylabel('Count')
    for bar in axes[1].patches:
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                     int(bar.get_height()), ha='center', fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("**Sample Speeches**")
    st.dataframe(df[['title','year','era','sentiment','compound']].head(15),
                 use_container_width=True, hide_index=True)


# ── 2. Text Preprocessing ──
elif section == "🔤 Text Preprocessing":
    st.markdown("<div class='section-header'>🔤 Text Preprocessing</div>", unsafe_allow_html=True)

    speech_idx = st.selectbox("Select Speech", range(min(10, len(df))),
                               format_func=lambda i: df['title'].iloc[i][:60])

    sample_text  = str(df['content'].iloc[speech_idx])[:2000]
    word_tokens  = word_tokenize(sample_text)
    sent_tokens  = sent_tokenize(sample_text)
    tokens_lower = [t.lower() for t in word_tokens if t.isalpha()]
    tokens_clean = [t for t in tokens_lower if t not in ALL_SW]
    stems  = [stemmer.stem(t) for t in tokens_clean]
    lemmas = [lemmatizer.lemmatize(t, pos='v') for t in tokens_clean]

    tab1, tab2, tab3 = st.tabs(["Tokenization", "Stop Words", "Stem vs Lemma"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Word Tokens", f"{len(word_tokens):,}")
        c2.metric("Unique Tokens", f"{len(set(word_tokens)):,}")
        c3.metric("Sentences", f"{len(sent_tokens):,}")

        alpha_tokens = [t.lower() for t in word_tokens if t.isalpha()]
        freq = Counter(alpha_tokens)
        top30 = freq.most_common(30)
        words30, counts30 = zip(*top30)

        fig, ax = plt.subplots(figsize=(14, 4))
        fig.patch.set_facecolor(DARK_BG)
        ax.bar(range(len(words30)), counts30, color=[PALETTE[i%len(PALETTE)] for i in range(30)])
        ax.set_xticks(range(len(words30)))
        ax.set_xticklabels(words30, rotation=45, ha='right', fontsize=9)
        ax.set_title('Top 30 Word Tokens (before stop word removal)', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with tab2:
        removed_sw   = [t for t in tokens_lower if t in ALL_SW]
        content_freq = Counter(tokens_clean)

        c1, c2, c3 = st.columns(3)
        c1.metric("Before Removal", f"{len(tokens_lower):,}")
        c2.metric("After Removal", f"{len(tokens_clean):,}")
        c3.metric("Retention Rate", f"{len(tokens_clean)/max(len(tokens_lower),1)*100:.1f}%")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor(DARK_BG)
        for ax, data, title, color in [
            (axes[0], Counter(tokens_lower).most_common(15), 'Before Stop Word Removal', PALETTE[1]),
            (axes[1], content_freq.most_common(15), 'After Stop Word Removal', PALETTE[0])
        ]:
            if data:
                w, c = zip(*data)
                ax.barh(range(len(w)), c, color=color)
                ax.set_yticks(range(len(w))); ax.set_yticklabels(w, fontsize=9)
                ax.invert_yaxis(); ax.set_title(title, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with tab3:
        c1, c2, c3 = st.columns(3)
        c1.metric("Original Vocab", f"{len(set(tokens_clean)):,}")
        c2.metric("After Stemming", f"{len(set(stems)):,}")
        c3.metric("After Lemmatization", f"{len(set(lemmas)):,}")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.patch.set_facecolor(DARK_BG)
        for ax, toks, title, color in [
            (axes[0], tokens_clean, 'Original', PALETTE[2]),
            (axes[1], stems, 'After Stemming', PALETTE[3]),
            (axes[2], lemmas, 'After Lemmatization', PALETTE[0])
        ]:
            top = Counter(toks).most_common(12)
            if top:
                w, c = zip(*top)
                ax.barh(range(len(w)), c, color=color)
                ax.set_yticks(range(len(w))); ax.set_yticklabels(w, fontsize=9)
                ax.invert_yaxis(); ax.set_title(title, fontweight='bold')
                ax.text(0.98, 0.02, f'Vocab: {len(set(toks))}', transform=ax.transAxes,
                        ha='right', fontsize=9, color='#ffd740')
        plt.tight_layout()
        st.pyplot(fig); plt.close()


# ── 3. BoW & TF-IDF ──
elif section == "📦 BoW & TF-IDF":
    st.markdown("<div class='section-header'>📦 Bag of Words & TF-IDF</div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Bag of Words", "TF-IDF"])

    with tab1:
        bow_dense   = bow_mat.toarray()
        corpus_freq = bow_dense.sum(axis=0)
        top_idx     = corpus_freq.argsort()[-20:][::-1]

        c1, c2 = st.columns(2)
        c1.metric("Vocabulary Size", f"{len(bow_feat):,}")
        c2.metric("Matrix Shape", f"{bow_mat.shape[0]} × {bow_mat.shape[1]}")

        fig, ax = plt.subplots(figsize=(13, 6))
        fig.patch.set_facecolor(DARK_BG)
        top_terms  = [bow_feat[i] for i in top_idx]
        top_counts = [corpus_freq[i] for i in top_idx]
        bars = ax.barh(range(len(top_terms)), top_counts,
                       color=[PALETTE[i%len(PALETTE)] for i in range(len(top_terms))])
        ax.set_yticks(range(len(top_terms))); ax.set_yticklabels(top_terms, fontsize=10)
        ax.invert_yaxis()
        for bar in bars:
            ax.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
                    int(bar.get_width()), va='center', fontsize=8)
        ax.set_title('Top 20 Terms — Bag of Words', fontweight='bold')
        ax.set_xlabel('Total Frequency')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with tab2:
        mean_tfidf  = tfidf_dense.mean(axis=0)
        top_tfidf   = mean_tfidf.argsort()[-20:][::-1]

        c1, c2 = st.columns(2)
        c1.metric("Vocabulary Size", f"{len(tfidf_feat):,}")
        c2.metric("Matrix Shape", f"{tfidf_mat.shape[0]} × {tfidf_mat.shape[1]}")

        top20_idx = mean_tfidf.argsort()[-20:][::-1]
        subset    = tfidf_dense[:min(12, len(df)), :][:, top20_idx]
        top20_t   = tfidf_feat[top20_idx]

        fig, ax = plt.subplots(figsize=(18, max(5, min(12, len(df))*0.6)))
        fig.patch.set_facecolor(DARK_BG)
        sns.heatmap(subset, xticklabels=top20_t,
                    yticklabels=[t[:35] for t in titles[:min(12, len(df))]],
                    cmap='magma', linewidths=0.3, ax=ax, cbar_kws={'shrink':0.8})
        ax.set_title('TF-IDF Heatmap — Signature Terms per Speech', fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig); plt.close()


# ── 4. N-Grams ──
elif section == "📐 N-Grams":
    st.markdown("<div class='section-header'>📐 N-Gram Analysis</div>", unsafe_allow_html=True)

    raw_corpus = df['content'].str.lower().tolist()
    results = {}
    for label, ngram in [('Unigrams (n=1)', (1,1)), ('Bigrams (n=2)', (2,2)), ('Trigrams (n=3)', (3,3))]:
        vec = TfidfVectorizer(ngram_range=ngram, max_features=200, min_df=2, stop_words='english')
        mat = vec.fit_transform(raw_corpus)
        feat = vec.get_feature_names_out()
        mean_sc = mat.toarray().mean(axis=0)
        top_idx = mean_sc.argsort()[-15:][::-1]
        results[label] = [(feat[i], mean_sc[i]) for i in top_idx if mean_sc[i] > 0]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.patch.set_facecolor(DARK_BG)
    for ax, (label, data), color in zip(axes, results.items(), [PALETTE[1], PALETTE[0], PALETTE[2]]):
        if not data: continue
        terms, scores = zip(*data[:12])
        ax.barh(range(len(terms)), scores, color=color)
        ax.set_yticks(range(len(terms))); ax.set_yticklabels(terms, fontsize=9)
        ax.invert_yaxis(); ax.set_title(label, fontweight='bold')
        ax.set_xlabel('Mean TF-IDF')
    plt.suptitle('N-Gram Analysis — Unigrams vs Bigrams vs Trigrams',
                 fontsize=13, color='white', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("""
    <div class='highlight-box'>
    💡 <b>Key Insight:</b> Unigrams miss compound political concepts.
    Bigrams reveal: <b>middle class, clean energy, small business</b>.
    Trigrams reveal: <b>middle class families, create new jobs</b>.
    </div>""", unsafe_allow_html=True)


# ── 5. Word Embeddings ──
elif section == "🧬 Word Embeddings":
    st.markdown("<div class='section-header'>🧬 Word Embeddings — Word2Vec</div>", unsafe_allow_html=True)

    @st.cache_resource
    def train_w2v(_token_lists):
        return Word2Vec(sentences=list(_token_lists), vector_size=100, window=5,
                        min_count=2, workers=2, epochs=20, sg=0)

    token_lists = tuple(tuple(t) for t in df['tokens'].tolist())
    with st.spinner("Training Word2Vec..."):
        w2v = train_w2v(token_lists)

    c1, c2, c3 = st.columns(3)
    c1.metric("Vocabulary", f"{len(w2v.wv):,} terms")
    c2.metric("Vector Dimensions", str(w2v.vector_size))
    c3.metric("Training Speeches", str(len(token_lists)))

    query = st.selectbox("Find similar words to:", [w for w in
                          ['economy','education','jobs','energy','healthcare','infrastructure']
                          if w in w2v.wv])
    if query in w2v.wv:
        similar = w2v.wv.most_similar(query, topn=8)
        cols = st.columns(4)
        for i, (word, score) in enumerate(similar):
            cols[i%4].markdown(f"""
            <div class='metric-card' style='margin-bottom:0.5rem;'>
                <div style='font-family: IBM Plex Mono, monospace; color: #40c4ff;'>{word}</div>
                <div style='color: #e040fb; font-size:0.9rem;'>{score:.3f}</div>
            </div>""", unsafe_allow_html=True)

    # t-SNE plot
    words_to_plot = ['economy','jobs','education','energy','healthcare','invest',
                     'growth','infrastructure','innovation','america','people',
                     'work','future','tax','budget']
    words_to_plot = [w for w in words_to_plot if w in w2v.wv]
    vectors = np.array([w2v.wv[w] for w in words_to_plot])

    if len(vectors) >= 4:
        perp = min(5, len(vectors)-1)
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=1000)
        coords = tsne.fit_transform(vectors)

        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor(DARK_BG)
        for i, word in enumerate(words_to_plot):
            c = PALETTE[i % len(PALETTE)]
            ax.scatter(coords[i,0], coords[i,1], color=c, s=150, zorder=2)
            ax.annotate(word, (coords[i,0]+0.5, coords[i,1]+0.5), fontsize=11, color=c)
        ax.set_title('Word2Vec Embeddings — t-SNE Projection', fontweight='bold')
        ax.set_xlabel('t-SNE dim 1'); ax.set_ylabel('t-SNE dim 2')
        plt.tight_layout()
        st.pyplot(fig); plt.close()


# ── 6. Classification ──
elif section == "🤖 Classification":
    st.markdown("<div class='section-header'>🤖 Text Classification</div>", unsafe_allow_html=True)

    valid_eras = df['era'].value_counts()
    valid_eras = valid_eras[valid_eras >= 3].index.tolist()
    df_cls = df[df['era'].isin(valid_eras)].reset_index(drop=True)

    st.markdown(f"**Task:** Classify speeches by presidential era · **Classes:** {', '.join(valid_eras)}")

    if len(df_cls) >= 6 and len(valid_eras) >= 2:
        le = LabelEncoder()
        y  = le.fit_transform(df_cls['era'])
        X_vec = TfidfVectorizer(max_features=500, sublinear_tf=True)
        X     = X_vec.fit_transform(df_cls['clean_text'])

        MODELS = {
            'Naive Bayes': MultinomialNB(),
            'SVM (LinearSVC)': LinearSVC(max_iter=2000),
            'Logistic Regression': LogisticRegression(max_iter=2000)
        }

        with st.spinner("Running 5-fold cross validation..."):
            cv_results = {}
            for name, model in MODELS.items():
                cv = cross_val_score(model, X, y, cv=min(5, len(df_cls)), scoring='accuracy')
                cv_results[name] = (cv.mean(), cv.std())

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(DARK_BG)
        names  = list(cv_results.keys())
        means  = [v[0]*100 for v in cv_results.values()]
        stds   = [v[1]*100 for v in cv_results.values()]
        bars   = ax.bar(names, means, color=PALETTE[:3], edgecolor=DARK_BG, width=0.5)
        ax.errorbar(names, means, yerr=stds, fmt='none', color='white', capsize=6, linewidth=2)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    f'{m:.1f}%', ha='center', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 115)
        ax.set_title('Text Classification — Model Accuracy (5-fold CV)', fontweight='bold')
        ax.set_ylabel('Accuracy (%)')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        # Show results table
        res_df = pd.DataFrame({
            'Model': names,
            'CV Accuracy': [f"{v[0]*100:.1f}%" for v in cv_results.values()],
            'Std Dev': [f"±{v[1]*100:.1f}%" for v in cv_results.values()]
        })
        st.dataframe(res_df, hide_index=True, use_container_width=True)
    else:
        st.warning("Not enough samples per class for classification. Need at least 3 per era.")
        st.dataframe(df[['title','year','era']].head(20), hide_index=True)


# ── 7. Sentiment Analysis ──
elif section == "💬 Sentiment Analysis":
    st.markdown("<div class='section-header'>💬 Sentiment Analysis</div>", unsafe_allow_html=True)

    # Summary metrics
    dist = df['sentiment'].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><div class='metric-val' style='color:#69ff47'>{dist.get('Optimistic',0)}</div><div class='metric-lbl'>Optimistic</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='metric-val' style='color:#ff4081'>{dist.get('Urgent',0)}</div><div class='metric-lbl'>Urgent</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><div class='metric-val' style='color:#ffd740'>{dist.get('Neutral',0)}</div><div class='metric-lbl'>Neutral</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><div class='metric-val'>{df['compound'].mean():+.3f}</div><div class='metric-lbl'>Avg Compound</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["VADER Analysis", "Aspect-Based"])

    with tab1:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.patch.set_facecolor(DARK_BG)
        colors_pie = {'Optimistic':'#2ecc71','Urgent':'#e74c3c','Neutral':'#f39c12'}

        scounts = df['sentiment'].value_counts()
        axes[0].pie(scounts.values, labels=scounts.index,
                    colors=[colors_pie.get(c,'#888') for c in scounts.index],
                    autopct='%1.0f%%', startangle=90,
                    textprops={'color':'white','fontsize':11})
        axes[0].set_title('Sentiment Distribution', fontweight='bold')

        colors_bar = [colors_pie.get(s,'#888') for s in df['sentiment']]
        axes[1].bar(range(len(df)), df['compound'], color=colors_bar, edgecolor=DARK_BG, width=0.7)
        axes[1].axhline(0.15,  color='#2ecc71', linestyle='--', alpha=0.6, label='+0.15 Optimistic')
        axes[1].axhline(-0.05, color='#e74c3c', linestyle='--', alpha=0.6, label='-0.05 Urgent')
        axes[1].set_title('VADER Compound Score per Speech', fontweight='bold')
        axes[1].legend(fontsize=8)

        year_sent = df.groupby('year')['compound'].mean().dropna()
        if len(year_sent) > 1:
            axes[2].plot(year_sent.index.astype(int), year_sent.values,
                         color=PALETTE[0], linewidth=2.5, marker='o', markersize=8)
            axes[2].fill_between(year_sent.index.astype(int), year_sent.values,
                                 alpha=0.2, color=PALETTE[0])
            axes[2].axhline(0, color='white', alpha=0.2)
            axes[2].set_title('Avg Sentiment Arc by Year', fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig); plt.close()

        # Table
        display_df = df[['title','year','compound','sentiment']].copy()
        display_df['compound'] = display_df['compound'].apply(lambda x: f"{x:+.4f}")
        st.dataframe(display_df, hide_index=True, use_container_width=True)

    with tab2:
        ASPECTS = {
            'Economy': ['economy','economic','gdp','recession','recovery','growth','fiscal'],
            'Jobs': ['job','employ','unemployment','worker','workforce','hire','wages'],
            'Education': ['education','school','college','university','student','teacher','learn'],
            'Healthcare': ['health','healthcare','insurance','medical','hospital','patient','care'],
            'Energy': ['energy','oil','solar','wind','clean','renewable','environment','climate'],
            'Infrastructure': ['infrastructure','road','bridge','broadband','transport','build','repair']
        }

        with st.spinner("Computing aspect-based sentiment..."):
            df_asp = df.copy()
            df_asp['sentences'] = df_asp['content'].apply(lambda t: sent_tokenize(str(t)[:1500]))
            df_asp['sentence_scores'] = df_asp['sentences'].apply(
                lambda sents: [(s, vader.polarity_scores(s)['compound']) for s in sents])

            def aspect_sent(ss, kws):
                scores = [sc for s, sc in ss if any(k in s.lower() for k in kws)]
                return round(sum(scores)/len(scores), 4) if scores else 0.0

            aspect_matrix = {}
            for aspect, kws in ASPECTS.items():
                aspect_matrix[aspect] = df_asp['sentence_scores'].apply(
                    lambda s: aspect_sent(s, kws)).values

        aspect_df = pd.DataFrame(aspect_matrix, index=df['title'].str[:30]).head(25)
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor(DARK_BG)
        sns.heatmap(aspect_df, cmap='RdYlGn', center=0, linewidths=0.3, ax=ax,
                    annot=False, cbar_kws={'shrink':0.7})
        ax.set_title('Aspect-Based Sentiment — Policy Domains × Speeches', fontweight='bold')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        st.pyplot(fig); plt.close()


# ── 8. Topic Modelling ──
elif section == "🗺️ Topic Modelling":
    st.markdown("<div class='section-header'>🗺️ Topic Modelling</div>", unsafe_allow_html=True)

    NOISE = {'also','must','will','shall','every','would','could',
             'said','say','just','like','know','get','put','got',
             'one','two','new','back','now','time','year','today'}

    @st.cache_resource
    def run_lda(_token_lists):
        token_lists_clean = [
            [t for t in toks if t not in NOISE and len(t) > 3]
            for toks in _token_lists
        ]
        id2word = corpora.Dictionary(token_lists_clean)
        id2word.filter_extremes(no_below=2, no_above=0.85)
        bow_corp = [id2word.doc2bow(doc) for doc in token_lists_clean]
        lda = LdaModel(corpus=bow_corp, id2word=id2word, num_topics=5,
                       random_state=42, passes=20, alpha='auto', eta='auto')
        return lda, id2word, bow_corp

    token_lists_t = tuple(tuple(t) for t in df['tokens'].tolist())
    with st.spinner("Running LDA Topic Modelling..."):
        lda, id2word, bow_corp = run_lda(token_lists_t)

    TOPIC_LABELS = {
        0:'Economic Recovery & Jobs', 1:'Infrastructure & Clean Energy',
        2:'Education & Workforce',    3:'Healthcare & Social Safety',
        4:'Trade, Innovation & Competitiveness'
    }

    def dominant_topic(b):
        dist = lda.get_document_topics(b)
        return max(dist, key=lambda x: x[1])[0] if dist else 0

    df['lda_topic']   = [dominant_topic(b) for b in bow_corp]
    df['topic_label'] = df['lda_topic'].map(TOPIC_LABELS)

    tab1, tab2, tab3 = st.tabs(["LDA Topics", "LSA", "Topic Assignments"])

    with tab1:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor(DARK_BG)

        tc = df['topic_label'].value_counts()
        axes[0].barh(range(len(tc)), tc.values,
                     color=[PALETTE[i%len(PALETTE)] for i in range(len(tc))])
        axes[0].set_yticks(range(len(tc))); axes[0].set_yticklabels(tc.index, fontsize=9)
        axes[0].invert_yaxis(); axes[0].set_title('LDA Topic Distribution', fontweight='bold')

        top_n_words = 8
        topic_word_mat = np.zeros((5, top_n_words))
        for tid in range(5):
            ww = lda.show_topic(tid, topn=top_n_words)
            for j, (_, s) in enumerate(ww):
                topic_word_mat[tid, j] = s
        word_labels = [w for w, _ in lda.show_topic(0, topn=top_n_words)]
        topic_names = [TOPIC_LABELS[i][:20] for i in range(5)]
        sns.heatmap(topic_word_mat, xticklabels=word_labels, yticklabels=topic_names,
                    cmap='YlOrRd', ax=axes[1], linewidths=0.3, annot=True, fmt='.3f',
                    cbar_kws={'shrink':0.8})
        axes[1].set_title('Top Word Probabilities per Topic', fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig); plt.close()

        for tid in range(5):
            ww = lda.show_topic(tid, topn=10)
            words_str = ' · '.join([f'{w}' for w, _ in ww])
            st.markdown(f"<div class='highlight-box'><b style='color:{PALETTE[tid]}'>{TOPIC_LABELS[tid]}</b><br>{words_str}</div>", unsafe_allow_html=True)

    with tab2:
        n_comp  = min(5, len(df)-1)
        svd     = TruncatedSVD(n_components=n_comp, random_state=42)
        lsa_mat = svd.fit_transform(tfidf_mat)

        st.write(f"**Variance Explained:** {svd.explained_variance_ratio_.sum()*100:.1f}%")
        for i, v in enumerate(svd.explained_variance_ratio_):
            st.progress(float(v), text=f"Component {i+1}: {v*100:.1f}%")

        if n_comp >= 2:
            fig, ax = plt.subplots(figsize=(10, 7))
            fig.patch.set_facecolor(DARK_BG)
            for i, (x, y) in enumerate(zip(lsa_mat[:,0], lsa_mat[:,1])):
                c = PALETTE[df['lda_topic'].iloc[i] % len(PALETTE)]
                ax.scatter(x, y, color=c, s=120, zorder=2)
                ax.annotate(str(df['title'].iloc[i])[:20], (x+0.002, y+0.002), fontsize=8, color=c)
            ax.set_title('LSA — Document Space (Component 1 vs 2)', fontweight='bold')
            ax.set_xlabel('LSA Component 1'); ax.set_ylabel('LSA Component 2')
            plt.tight_layout()
            st.pyplot(fig); plt.close()

    with tab3:
        st.dataframe(df[['title','year','topic_label','sentiment']],
                     hide_index=True, use_container_width=True)


# ── 9. Clustering ──
elif section == "🔵 Clustering":
    st.markdown("<div class='section-header'>🔵 Document Clustering</div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["K-Means", "Hierarchical"])

    with tab1:
        K = min(4, max(2, len(df)//3))
        km = KMeans(n_clusters=K, random_state=42, n_init=10, max_iter=300)
        km_labels = km.fit_predict(tfidf_mat.toarray())
        df['kmeans_cluster'] = km_labels

        max_k = min(8, len(df)-1)
        k_range = range(2, max_k+1)
        inertias = []
        for k in k_range:
            km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
            km_tmp.fit(tfidf_mat)
            inertias.append(km_tmp.inertia_)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor(DARK_BG)
        axes[0].plot(list(k_range), inertias, color=PALETTE[0], linewidth=2.5, marker='o', markersize=8)
        axes[0].axvline(K, color=PALETTE[2], linestyle='--', alpha=0.7, label=f'Chosen K={K}')
        axes[0].set_title('Elbow Method', fontweight='bold'); axes[0].legend()
        axes[0].set_xlabel('K'); axes[0].set_ylabel('Inertia')

        cluster_counts = df['kmeans_cluster'].value_counts().sort_index()
        axes[1].bar([f'Cluster {i}' for i in cluster_counts.index], cluster_counts.values,
                    color=[PALETTE[i%len(PALETTE)] for i in range(len(cluster_counts))])
        axes[1].set_title('K-Means Cluster Sizes', fontweight='bold')
        axes[1].set_ylabel('Speeches')

        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.dataframe(df[['title','year','kmeans_cluster','topic_label']],
                     hide_index=True, use_container_width=True)

    with tab2:
        MAX_DOCS = 60
        df_hc     = df.head(MAX_DOCS)
        tfidf_hc  = tfidf_mat[:MAX_DOCS]
        n_lsa     = min(10, len(df_hc)-1)
        svd_h     = TruncatedSVD(n_components=n_lsa, random_state=42)
        X_lsa     = svd_h.fit_transform(tfidf_hc)
        Z         = linkage(X_lsa, method='ward')
        short_labels = [str(t)[:25] for t in df_hc['title'].tolist()]

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor(DARK_BG)
        dendrogram(Z, labels=short_labels, orientation='right', leaf_font_size=8,
                   color_threshold=0.7*max(Z[:,2]), ax=ax)
        ax.set_title('Hierarchical Clustering Dendrogram', fontweight='bold')
        ax.set_xlabel('Distance (Ward Linkage)')
        plt.tight_layout()
        st.pyplot(fig); plt.close()


# ── 10. NER ──
elif section == "🏷️ NER":
    st.markdown("<div class='section-header'>🏷️ Named Entity Recognition</div>", unsafe_allow_html=True)

    TARGET_ENT = {
        'GPE': 'Geopolitical Entity', 'ORG': 'Organization',
        'PERSON': 'Person', 'NORP': 'Nationality / Political Group',
        'MONEY': 'Monetary Value'
    }

    with st.spinner("Running spaCy NER (this may take a moment)..."):
        MAX_DOCS  = min(30, len(df))
        TEXT_LIMIT = 10000
        ner_freq   = defaultdict(Counter)
        docs = nlp_model.pipe(
            (str(text)[:TEXT_LIMIT] for text in df.head(MAX_DOCS)['content']),
            batch_size=8
        )
        for doc in docs:
            for ent in doc.ents:
                if ent.label_ in TARGET_ENT:
                    norm = ent.text.replace("'s","").strip().title()
                    if len(norm) > 2:
                        ner_freq[ent.label_][norm] += 1

    n_cols = 3
    labels_available = [l for l in TARGET_ENT if l in ner_freq and ner_freq[l]]
    rows = [labels_available[i:i+n_cols] for i in range(0, len(labels_available), n_cols)]
    for row in rows:
        cols = st.columns(len(row))
        for col, label in zip(cols, row):
            top = ner_freq[label].most_common(8)
            if not top: continue
            with col:
                st.markdown(f"**[{label}] {TARGET_ENT[label]}**")
                ner_df = pd.DataFrame(top, columns=['Entity','Count'])
                fig, ax = plt.subplots(figsize=(5, 3))
                fig.patch.set_facecolor(DARK_BG)
                color = PALETTE[labels_available.index(label) % len(PALETTE)]
                ax.barh(ner_df['Entity'], ner_df['Count'], color=color)
                ax.invert_yaxis(); ax.set_xlabel('Mentions')
                plt.tight_layout()
                st.pyplot(fig); plt.close()


# ── 11. Summarization ──
elif section == "📝 Summarization":
    st.markdown("<div class='section-header'>📝 Text Summarization</div>", unsafe_allow_html=True)

    def textrank_summarize(text, n_sents=5):
        sents = sent_tokenize(str(text))
        sents = [s for s in sents if len(s.split()) > 8]
        if len(sents) <= n_sents:
            return sents
        vec  = TfidfVectorizer(stop_words='english', max_features=500)
        mat  = vec.fit_transform(sents)
        sim  = cosine_similarity(mat)
        np.fill_diagonal(sim, 0)
        G    = nx.from_numpy_array(sim)
        scores = nx.pagerank(G, max_iter=200)
        ranked = sorted(scores, key=scores.get, reverse=True)[:n_sents]
        return [sents[i] for i in sorted(ranked)]

    speech_idx = st.selectbox("Select Speech to Summarize",
                               range(min(len(df), 15)),
                               format_func=lambda i: df['title'].iloc[i][:65])

    n_sents = st.slider("Summary Length (sentences)", 2, 8, 4)

    content = str(df['content'].iloc[speech_idx])
    summary = textrank_summarize(content, n_sents=n_sents)
    orig_wc = len(content.split())
    summ_wc = sum(len(s.split()) for s in summary)

    c1, c2, c3 = st.columns(3)
    c1.metric("Original Words", f"{orig_wc:,}")
    c2.metric("Summary Words", f"{summ_wc:,}")
    c3.metric("Compression", f"{(1-summ_wc/max(orig_wc,1))*100:.0f}%")

    st.markdown("**Extractive Summary (TextRank)**")
    for i, sent in enumerate(summary, 1):
        st.markdown(f"""
        <div class='highlight-box'>
            <span style='color:#e040fb; font-weight:600;'>[{i}]</span> {sent}
        </div>""", unsafe_allow_html=True)


# ── 12. Word Clouds ──
elif section == "☁️ Word Clouds":
    st.markdown("<div class='section-header'>☁️ Word Clouds</div>", unsafe_allow_html=True)

    TOPIC_LABELS = {
        0:'Economic Recovery & Jobs', 1:'Infrastructure & Clean Energy',
        2:'Education & Workforce',    3:'Healthcare & Social Safety',
        4:'Trade, Innovation & Competitiveness'
    }
    WC_COLORS = ['viridis','plasma','magma','inferno','cividis','cool']

    if 'lda_topic' not in df.columns:
        df['lda_topic'] = 0

    all_text = ' '.join(df['clean_text'].tolist())
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.patch.set_facecolor(DARK_BG)
    axes_flat = axes.flatten()

    wc = WordCloud(width=700, height=400, background_color=MID_BG,
                   colormap='plasma', max_words=80, collocations=False).generate(all_text)
    axes_flat[0].imshow(wc, interpolation='bilinear')
    axes_flat[0].axis('off')
    axes_flat[0].set_title('All Development Speeches', fontweight='bold')

    for i in range(1, 6):
        tid = i - 1
        topic_text = ' '.join(df[df['lda_topic']==tid]['clean_text'].tolist())
        if not topic_text.strip():
            axes_flat[i].axis('off'); continue
        wc2 = WordCloud(width=500, height=300, background_color=MID_BG,
                        colormap=WC_COLORS[i%len(WC_COLORS)], max_words=50,
                        collocations=False).generate(topic_text)
        axes_flat[i].imshow(wc2, interpolation='bilinear')
        axes_flat[i].axis('off')
        axes_flat[i].set_title(f'Topic: {TOPIC_LABELS.get(tid,"")[:30]}',
                                fontweight='bold', fontsize=9)

    plt.suptitle('Word Clouds — Development Speeches by Topic',
                 fontsize=14, color='white', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig); plt.close()
