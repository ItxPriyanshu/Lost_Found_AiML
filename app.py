"""
app.py
------
AI-Powered Lost & Found Matcher - Streamlit Web Application.
A complete, self-contained Machine Learning web application for college-level demonstration.

Implemented Algorithms:
1. Random Forest Classifier (Multiclass: Category prediction from description)
2. TF-IDF + Cosine Similarity (Text semantics matching)
3. Logistic Regression (Binary classification: Potential match prediction)
4. K-Means Clustering (Unsupervised learning: Topic/item grouping)
5. SQLite Database (Persistent local storage)
"""

import os
import re
import shutil
import sqlite3
import joblib
from datetime import datetime, date

# Data science & visualization libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn algorithms
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Silence loky CPU detection warning on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# =====================================================================
# PATHS & CONFIGURATION
# =====================================================================
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
DB_PATH = os.path.join(BASE_DIR, "lost_found.db")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

CATEGORY_MODEL_PATH = os.path.join(MODELS_DIR, "category_classifier.pkl")
CATEGORY_TFIDF_PATH = os.path.join(MODELS_DIR, "category_tfidf.pkl")
MATCH_MODEL_PATH = os.path.join(MODELS_DIR, "match_classifier.pkl")

CATEGORIES = [
    "Wallet",
    "Mobile Phone",
    "Bag",
    "Laptop",
    "Watch",
    "Keys",
    "Other"
]

COMMON_LOCATIONS = [
    "Central Library",
    "Cafeteria",
    "Computer Lab 3",
    "Sports Complex",
    "Lecture Hall A",
    "Admin Block",
    "Auditorium",
    "Main Gate",
    "Science Block",
    "Student Center",
    "Other Location"
]

COMMON_COLORS = [
    "Black",
    "Brown",
    "Blue",
    "Silver",
    "Grey",
    "Red",
    "White",
    "Green",
    "Golden",
    "Other"
]

# Standard self-contained English stopwords
STOPWORDS = set([
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can't", "cannot", "could", "did", "do", "does", "doing", "don't", "down", "during", "each", "few",
    "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "him", "his",
    "how", "i", "if", "in", "into", "is", "isn't", "it", "its", "me", "more", "most", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "out", "over",
    "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "with", "would", "you", "your", "yours"
])

def clean_text(text: str) -> str:
    """Preprocesses text: lowercases, strips special characters, removes stopwords."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    tokens = text.split()
    filtered = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(filtered)


# =====================================================================
# DATABASE MANAGEMENT (SQLite)
# =====================================================================
def get_db_connection():
    """Returns SQLite connection with row factory for dictionary-style access."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Creates the reports table if it doesn't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_type TEXT NOT NULL CHECK(report_type IN ('lost', 'found')),
            item_name TEXT NOT NULL,
            category TEXT NOT NULL,
            color TEXT NOT NULL,
            location TEXT NOT NULL,
            report_date TEXT NOT NULL,
            description TEXT NOT NULL,
            image_path TEXT,
            contact_info TEXT,
            status TEXT NOT NULL DEFAULT 'open',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def seed_sample_reports_if_empty():
    """Seeds realistic sample campus reports for immediate demonstration."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM reports")
    count = cursor.fetchone()[0]

    if count == 0:
        sample_reports = [
            ("lost", "Black Leather Wallet", "Wallet", "Black", "Central Library",
             "2026-11-20", "Lost my black leather bi-fold wallet on the 2nd floor study table. Has student ID and metro pass.",
             None, "student1@univ.edu", "open"),
            ("lost", "Blue iPhone 13", "Mobile Phone", "Blue", "Central Library",
             "2026-11-21", "Midnight blue iPhone 13 with cracked tempered glass protector, dropped near library entrance.",
             None, "9876543210", "open"),
            ("lost", "Dell Inspiron 15 Laptop", "Laptop", "Silver", "Computer Lab 3",
             "2026-11-22", "Silver Dell Inspiron 15.6 inch laptop with Intel Core i5 sticker. Left on desk 14.",
             None, "cs_student@univ.edu", "open"),
            ("lost", "Nike Gym Duffle Bag", "Bag", "Red", "Sports Complex",
             "2026-11-19", "Red Nike sports duffle bag with gym sneakers, towel, and water bottle.",
             None, "sports_rep@univ.edu", "open"),
            ("lost", "Casio Analog Watch", "Watch", "Silver", "Lecture Hall A",
             "2026-11-22", "Silver stainless steel Casio quartz watch lost during Physics lecture.",
             None, "physics_guy@univ.edu", "open"),
            ("lost", "Honda Bike Key", "Keys", "Silver", "Main Gate",
             "2026-11-23", "Pair of Honda Activa scooter keys on red rubber keychain lost near the parking gate.",
             None, "hostel_b@univ.edu", "open"),
            ("lost", "AirPods Pro 2", "Other", "White", "Student Center",
             "2026-11-23", "Apple AirPods Pro 2 in white charging case misplaced near the coffee shop.",
             None, "audio_fan@univ.edu", "open"),

            ("found", "Dark Leather Wallet", "Wallet", "Black", "Central Library",
             "2026-11-20", "Found a black leather bi-fold wallet on 2nd floor library reading desk with student cards.",
             None, "lib_reception@univ.edu", "open"),
            ("found", "Blue iPhone", "Mobile Phone", "Blue", "Central Library",
             "2026-11-21", "Found blue iPhone 13 with cracked screen near central library front steps.",
             None, "security_office@univ.edu", "open"),
            ("found", "Silver Dell Laptop", "Laptop", "Silver", "Computer Lab 3",
             "2026-11-22", "Found Dell Inspiron laptop in lab 3 workstation. Kept with lab assistant.",
             None, "lab3_tech@univ.edu", "open"),
            ("found", "Red Sports Duffle Bag", "Bag", "Red", "Sports Complex",
             "2026-11-20", "Red duffle bag containing sports shoes found in badminton court bench.",
             None, "coach@univ.edu", "open"),
            ("found", "Metallic Wristwatch", "Watch", "Silver", "Lecture Hall A",
             "2026-11-23", "Found silver Casio metal watch in 3rd row of Lecture Hall A.",
             None, "hall_staff@univ.edu", "open"),
            ("found", "Scooter Key with Red Ring", "Keys", "Silver", "Main Gate",
             "2026-11-23", "Two keys with red Honda keychain found near the bike parking area.",
             None, "gate_guard@univ.edu", "open"),
            ("found", "Casio Scientific Calculator", "Other", "Black", "Lecture Hall A",
             "2026-11-22", "Casio fx-991EX scientific calculator found on desk in Lecture Hall A.",
             None, "math_dept@univ.edu", "open"),
            ("found", "Green Military Rucksack", "Bag", "Green", "Main Gate",
             "2026-11-18", "Large green canvas rucksack bag found near main security post.",
             None, "gate_guard@univ.edu", "open")
        ]

        cursor.executemany("""
            INSERT INTO reports (
                report_type, item_name, category, color, location,
                report_date, description, image_path, contact_info, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, sample_reports)
        conn.commit()

    conn.close()

def add_report(report_type, item_name, category, color, location, report_date, description, image_path=None, contact_info=None):
    """Inserts a new report into the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO reports (
            report_type, item_name, category, color, location,
            report_date, description, image_path, contact_info, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')
    """, (report_type, item_name, category, color, location, str(report_date), description, image_path, contact_info))
    report_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return report_id

def get_reports(report_type=None, category=None, location=None, search_query=None):
    """Queries reports with optional filters."""
    conn = get_db_connection()
    cursor = conn.cursor()
    query = "SELECT * FROM reports WHERE 1=1"
    params = []

    if report_type and report_type != "All":
        query += " AND report_type = ?"
        params.append(report_type.lower())
    if category and category != "All":
        query += " AND category = ?"
        params.append(category)
    if location and location != "All":
        query += " AND location = ?"
        params.append(location)
    if search_query:
        query += " AND (item_name LIKE ? OR description LIKE ?)"
        term = f"%{search_query.strip()}%"
        params.extend([term, term])

    query += " ORDER BY id DESC"
    cursor.execute(query, params)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

def get_report_by_id(report_id):
    """Gets a single report by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM reports WHERE id = ?", (report_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def update_report_status(report_id, new_status):
    """Updates status ('open', 'resolved')."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE reports SET status = ? WHERE id = ?", (new_status, report_id))
    conn.commit()
    conn.close()

def get_summary_stats():
    """Computes summary metrics for cards and dashboard."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM reports WHERE report_type = 'lost'")
    total_lost = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM reports WHERE report_type = 'found'")
    total_found = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM reports WHERE status = 'resolved' OR status = 'matched'")
    total_resolved = cursor.fetchone()[0]
    conn.close()
    return {
        "total_lost": total_lost,
        "total_found": total_found,
        "total_resolved": total_resolved
    }


# =====================================================================
# MACHINE LEARNING MODULE
# =====================================================================
def train_category_model():
    """Trains Random Forest Classifier to predict category from description."""
    items_csv = os.path.join(DATA_DIR, "items_dataset.csv")
    if not os.path.exists(items_csv):
        # Fallback basic dataset if file not found
        data = [
            ("Black leather wallet", "Wallet"), ("Blue iPhone 13", "Mobile Phone"),
            ("Wildcraft backpack", "Bag"), ("Dell Inspiron laptop", "Laptop"),
            ("Casio analog watch", "Watch"), ("Honda bike key", "Keys"),
            ("Milton water bottle", "Other")
        ]
        df = pd.DataFrame(data, columns=["description", "category"])
    else:
        df = pd.read_csv(items_csv)

    df["clean_desc"] = df["description"].apply(clean_text)
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_vec = vectorizer.fit_transform(df["clean_desc"])
    y = df["category"]

    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    rf_clf.fit(X_vec, y)

    joblib.dump(rf_clf, CATEGORY_MODEL_PATH)
    joblib.dump(vectorizer, CATEGORY_TFIDF_PATH)
    return rf_clf, vectorizer

def predict_category(description: str):
    """Predicts category using trained Random Forest model."""
    if not os.path.exists(CATEGORY_MODEL_PATH) or not os.path.exists(CATEGORY_TFIDF_PATH):
        train_category_model()

    rf_clf = joblib.load(CATEGORY_MODEL_PATH)
    vectorizer = joblib.load(CATEGORY_TFIDF_PATH)

    cleaned = clean_text(description)
    if not cleaned:
        return "Other", 0.0, {}

    vec = vectorizer.transform([cleaned])
    pred = rf_clf.predict(vec)[0]
    probs = rf_clf.predict_proba(vec)[0]

    prob_dict = {cls: round(float(p), 3) for cls, p in zip(rf_clf.classes_, probs)}
    confidence = float(np.max(probs))
    return pred, confidence, prob_dict

def compute_text_similarity(text1: str, text2: str) -> float:
    """Computes Cosine Similarity between two text descriptions using TF-IDF."""
    clean1, clean2 = clean_text(text1), clean_text(text2)
    if not clean1 or not clean2:
        return 0.0
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([clean1, clean2])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(np.clip(sim, 0.0, 1.0))
    except Exception:
        return 0.0

def train_match_model():
    """Trains Logistic Regression binary match classifier."""
    match_csv = os.path.join(DATA_DIR, "matching_dataset.csv")
    feature_cols = ["text_similarity", "same_category", "same_color", "same_location", "date_diff_days"]
    
    if os.path.exists(match_csv):
        df = pd.read_csv(match_csv)
    else:
        # Fallback dummy data
        df = pd.DataFrame([
            [0.8, 1, 1, 1, 0, 1], [0.6, 1, 1, 0, 2, 1],
            [0.1, 0, 0, 0, 10, 0], [0.0, 0, 1, 0, 5, 0]
        ], columns=feature_cols + ["is_match"])

    X = df[feature_cols]
    y = df["is_match"]

    log_reg = LogisticRegression(random_state=42, max_iter=500)
    log_reg.fit(X, y)

    joblib.dump(log_reg, MATCH_MODEL_PATH)
    return log_reg

def predict_match(text_similarity: float, same_category: int, same_color: int, same_location: int, date_diff_days: float):
    """Predicts match using Logistic Regression with explainable feature contributions."""
    if not os.path.exists(MATCH_MODEL_PATH):
        train_match_model()

    log_reg = joblib.load(MATCH_MODEL_PATH)
    feature_cols = ["text_similarity", "same_category", "same_color", "same_location", "date_diff_days"]
    features_df = pd.DataFrame(
        [[text_similarity, same_category, same_color, same_location, date_diff_days]],
        columns=feature_cols
    )

    pred = int(log_reg.predict(features_df)[0])
    prob = float(log_reg.predict_proba(features_df)[0][1])

    display_cols = ["Text Similarity", "Same Category", "Same Color", "Same Location", "Date Difference"]
    contributions = {}
    for name, val, coef in zip(display_cols, features_df.values[0], log_reg.coef_[0]):
        contributions[name] = round(float(val * coef), 3)

    return pred, prob, contributions

def perform_kmeans_clustering(descriptions_list, n_clusters=4):
    """Performs K-Means clustering and PCA 2D projection."""
    if not descriptions_list or len(descriptions_list) < n_clusters:
        return [0] * len(descriptions_list), [[0.0, 0.0]] * len(descriptions_list), {}

    cleaned_texts = [clean_text(d) for d in descriptions_list]
    cleaned_texts = [t if t else "item" for t in cleaned_texts]

    vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
    X_tfidf = vectorizer.fit_transform(cleaned_texts)

    actual_clusters = min(n_clusters, len(descriptions_list))
    kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_tfidf).tolist()

    pca = PCA(n_components=2, random_state=42)
    coords_2d = pca.fit_transform(X_tfidf.toarray()).tolist()

    terms = vectorizer.get_feature_names_out()
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    top_keywords = {}
    for i in range(actual_clusters):
        top_words = [terms[ind] for ind in order_centroids[i, :4]]
        top_keywords[i] = top_words

    return cluster_labels, coords_2d, top_keywords

def train_all_models():
    """Trains and serializes both models."""
    train_category_model()
    train_match_model()


# =====================================================================
# MATCHING ENGINE & EXPLAINABILITY
# =====================================================================
def calculate_date_difference(date_str1: str, date_str2: str) -> int:
    """Calculates absolute difference in days between two date strings."""
    try:
        d1 = datetime.strptime(str(date_str1).strip()[:10], "%Y-%m-%d")
        d2 = datetime.strptime(str(date_str2).strip()[:10], "%Y-%m-%d")
        return abs((d1 - d2).days)
    except Exception:
        return 7

def build_explanation(same_category, same_color, same_location, text_sim, date_diff, cat_name, loc_name, color_name):
    """Generates human-readable explanation points."""
    reasons = []
    if same_category:
        reasons.append(f"Matching category: Both are '{cat_name}'")
    else:
        reasons.append("Different categories (lower confidence)")

    if same_location:
        reasons.append(f"Same location: Reported at '{loc_name}'")
    if same_color:
        reasons.append(f"Matching color: Both listed as '{color_name}'")

    if text_sim >= 0.5:
        reasons.append(f"High text description similarity ({int(text_sim * 100)}% TF-IDF cosine match)")
    elif text_sim >= 0.2:
        reasons.append(f"Moderate description similarity ({int(text_sim * 100)}% TF-IDF cosine match)")
    else:
        reasons.append("Low text similarity between descriptions")

    if date_diff == 0:
        reasons.append("Reported on the exact same date")
    elif date_diff <= 3:
        reasons.append(f"Reported within {date_diff} day(s) of each other")
    elif date_diff <= 7:
        reasons.append(f"Reported within a week ({date_diff} days difference)")
    else:
        reasons.append(f"Reported {date_diff} days apart")
    return reasons

def match_single_pair(lost_report, found_report):
    """Compares one lost report with one found report."""
    desc_lost = lost_report.get("description", "")
    desc_found = found_report.get("description", "")
    text_sim = compute_text_similarity(desc_lost, desc_found)

    lost_cat = str(lost_report.get("category", "")).strip().lower()
    found_cat = str(found_report.get("category", "")).strip().lower()
    same_category = 1 if (lost_cat and lost_cat == found_cat) else 0

    lost_col = str(lost_report.get("color", "")).strip().lower()
    found_col = str(found_report.get("color", "")).strip().lower()
    same_color = 1 if (lost_col and lost_col == found_col) else 0

    lost_loc = str(lost_report.get("location", "")).strip().lower()
    found_loc = str(found_report.get("location", "")).strip().lower()
    same_location = 1 if (lost_loc and lost_loc == found_loc) else 0

    date_diff = calculate_date_difference(
        lost_report.get("report_date", ""),
        found_report.get("report_date", "")
    )

    is_match_pred, match_prob, contributions = predict_match(
        text_similarity=text_sim,
        same_category=same_category,
        same_color=same_color,
        same_location=same_location,
        date_diff_days=date_diff
    )

    date_score = max(0.0, 1.0 - (date_diff / 14.0))
    composite_score = (
        (text_sim * 0.35) +
        (same_category * 0.30) +
        (same_location * 0.15) +
        (same_color * 0.10) +
        (date_score * 0.10)
    ) * 100.0

    reasons = build_explanation(
        same_category=bool(same_category),
        same_color=bool(same_color),
        same_location=bool(same_location),
        text_sim=text_sim,
        date_diff=date_diff,
        cat_name=lost_report.get("category", "Unknown"),
        loc_name=lost_report.get("location", "Unknown"),
        color_name=lost_report.get("color", "Unknown")
    )

    return {
        "found_report": found_report,
        "is_potential_match": bool(is_match_pred),
        "logistic_regression_prob": round(match_prob * 100, 1),
        "text_similarity": round(text_sim * 100, 1),
        "composite_score": round(composite_score, 1),
        "same_category": bool(same_category),
        "same_color": bool(same_color),
        "same_location": bool(same_location),
        "date_diff_days": date_diff,
        "feature_contributions": contributions,
        "reasons": reasons
    }

def find_top_matches(target_report, candidate_reports, top_n=5):
    """Ranks candidates against a target report."""
    if not candidate_reports:
        return []
    results = []
    for cand in candidate_reports:
        if cand.get("id") == target_report.get("id"):
            continue
        results.append(match_single_pair(target_report, cand))

    results.sort(
        key=lambda x: (x["is_potential_match"], x["composite_score"], x["logistic_regression_prob"]),
        reverse=True
    )
    return results[:top_n]


# =====================================================================
# STREAMLIT UI CONFIGURATION & STYLING
# =====================================================================
st.set_page_config(
    page_title="AI Lost & Found Matcher",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .match-badge-yes {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.88rem;
        display: inline-block;
    }
    .match-badge-no {
        background-color: #F3F4F6;
        color: #4B5563;
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.88rem;
        display: inline-block;
    }
    .score-highlight {
        font-size: 1.4rem;
        font-weight: 700;
        color: #2563EB;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Database and Ensure Models are Trained
init_db()
seed_sample_reports_if_empty()

if not os.path.exists(CATEGORY_MODEL_PATH) or not os.path.exists(MATCH_MODEL_PATH):
    with st.spinner("Training ML Models for the first time..."):
        train_all_models()

def save_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return None
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded_file.name}"
    file_path = os.path.join(UPLOADS_DIR, filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(uploaded_file, f)
    return file_path


# =====================================================================
# SIDEBAR NAVIGATION
# =====================================================================
st.sidebar.markdown("## 🔍 Lost & Found AI")
st.sidebar.caption("Traditional Machine Learning Matcher (Jupyter Edition)")

menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📢 Report Lost Item",
        "🎁 Report Found Item",
        "🎯 Find My Item",
        "📊 Analytics Dashboard",
        "📋 View All Reports"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 ML Models In Action")
st.sidebar.markdown("""
- **Random Forest**: Predicts item category
- **TF-IDF + Cosine**: Measures text similarity
- **Logistic Regression**: Classifies potential matches
- **K-Means Clustering**: Unsupervised report clustering
""")
st.sidebar.info("💡 **Viva Tip**: All algorithms are implemented using Scikit-Learn without neural networks!")


# =====================================================================
# 1. HOME PAGE
# =====================================================================
if menu == "🏠 Home":
    st.markdown('<div class="main-header">AI-Powered Lost & Found Matcher</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Reconnecting lost belongings with their owners on campus using practical Machine Learning algorithms.</div>',
        unsafe_allow_html=True
    )

    stats = get_summary_stats()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="📌 Lost Items Reported", value=stats["total_lost"])
    with col2:
        st.metric(label="🎁 Found Items Reported", value=stats["total_found"])
    with col3:
        st.metric(label="✅ Resolved / Matched", value=stats["total_resolved"])
    with col4:
        st.metric(label="📦 Total Reports in DB", value=stats["total_lost"] + stats["total_found"])

    st.markdown("---")

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("### 🚀 How the System Works")
        st.markdown("""
        Finding lost items manually is frustrating and often relies on luck. Our system automates this with a 4-step ML pipeline:

        1. **Report Submission**: A student reports a lost or found item with description, category, color, location, and date.
        2. **Category Auto-Detection (Random Forest)**: As the user types, a supervised Random Forest Classifier automatically predicts the item category using TF-IDF features.
        3. **Semantic Text Matching (TF-IDF + Cosine Similarity)**: Converts descriptions into high-dimensional vector representations to capture similarities even when different words are used.
        4. **Potential Match Prediction (Logistic Regression)**: Evaluates description similarity, category match, location match, color match, and date difference to classify candidate pairs as **Potential Match** or **Not a Match**.
        """)
        st.info("🔒 **Privacy & Disclaimer**: ML matching generates probabilistic candidate suggestions to assist users. It does not certify legal ownership.")

    with col_right:
        st.markdown("### 🧭 Quick Actions")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📢 Report Lost Item", use_container_width=True):
                st.session_state["nav"] = "📢 Report Lost Item"
                st.rerun()
            if st.button("🎯 Find My Item Now", use_container_width=True):
                st.session_state["nav"] = "🎯 Find My Item"
                st.rerun()
        with c2:
            if st.button("🎁 Report Found Item", use_container_width=True):
                st.session_state["nav"] = "🎁 Report Found Item"
                st.rerun()
            if st.button("📊 View Analytics", use_container_width=True):
                st.session_state["nav"] = "📊 Analytics Dashboard"
                st.rerun()

        st.markdown("""
        <div class="card">
            <h4>💡 College Viva Highlights</h4>
            <ul>
                <li><strong>Supervised Learning</strong>: Multiclass Classification (RF) + Binary Classification (LR)</li>
                <li><strong>Feature Extraction</strong>: TF-IDF (Term Frequency-Inverse Document Frequency)</li>
                <li><strong>Unsupervised Learning</strong>: K-Means Clustering on item text</li>
                <li><strong>Notebook Format</strong>: All model training and analysis are in Jupyter Notebooks!</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# =====================================================================
# 2. REPORT LOST ITEM
# =====================================================================
elif menu == "📢 Report Lost Item":
    st.markdown('<div class="main-header">Report a Lost Item</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Submit details of the item you misplaced. Our ML models will help locate it.</div>', unsafe_allow_html=True)

    with st.form("lost_item_form", clear_on_submit=False):
        col1, col2 = st.columns(2)

        with col1:
            item_name = st.text_input("Item Name *", placeholder="e.g., Black Leather Wallet, Blue iPhone 13")
            color = st.selectbox("Item Color *", COMMON_COLORS)
            location = st.selectbox("Location Where Lost *", COMMON_LOCATIONS)
            if location == "Other Location":
                location = st.text_input("Specify Custom Location", placeholder="e.g., Badminton Court Bench")

        with col2:
            report_date = st.date_input("Date When Lost *", value=date.today())
            contact_info = st.text_input("Your Contact Info (Email or Phone) *", placeholder="e.g., student@univ.edu / 9876543210")
            uploaded_image = st.file_uploader("Upload Image (Optional)", type=["jpg", "jpeg", "png"])

        description = st.text_area(
            "Detailed Description *",
            placeholder="Describe unique markings, contents, brand, scratches, stickers, condition, etc.",
            help="The richer your description, the more accurate the TF-IDF and Random Forest predictions will be."
        )

        predicted_category = "Other"
        confidence = 0.0
        if description.strip():
            predicted_category, confidence, _ = predict_category(description)
            st.caption(f"🤖 **AI Auto-Detected Category**: `{predicted_category}` (Confidence: {confidence * 100:.1f}%)")

        category = st.selectbox(
            "Category * (Verify or modify the AI suggested category)",
            CATEGORIES,
            index=CATEGORIES.index(predicted_category) if predicted_category in CATEGORIES else 0
        )

        submitted = st.form_submit_button("Submit Lost Item Report 🚀", use_container_width=True)

        if submitted:
            if not item_name.strip() or not description.strip() or not contact_info.strip():
                st.error("⚠️ Please fill in all required fields (Item Name, Description, and Contact Info).")
            else:
                image_path = save_uploaded_file(uploaded_image)
                report_id = add_report(
                    report_type="lost",
                    item_name=item_name.strip(),
                    category=category,
                    color=color,
                    location=location,
                    report_date=report_date,
                    description=description.strip(),
                    image_path=image_path,
                    contact_info=contact_info.strip()
                )
                st.success(f"✅ Lost item report submitted successfully! Report ID: #{report_id}")
                st.balloons()
                st.info("👉 Head over to **'🎯 Find My Item'** in the sidebar to view potential matches right away!")


# =====================================================================
# 3. REPORT FOUND ITEM
# =====================================================================
elif menu == "🎁 Report Found Item":
    st.markdown('<div class="main-header">Report a Found Item</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Help a fellow student by reporting an item you discovered on campus.</div>', unsafe_allow_html=True)

    with st.form("found_item_form", clear_on_submit=False):
        col1, col2 = st.columns(2)

        with col1:
            item_name = st.text_input("Item Name *", placeholder="e.g., Casio Calculator, Red Backpack")
            color = st.selectbox("Item Color *", COMMON_COLORS)
            location = st.selectbox("Location Where Found *", COMMON_LOCATIONS)
            if location == "Other Location":
                location = st.text_input("Specify Custom Location", placeholder="e.g., Near Main Gate Bus Stop")

        with col2:
            report_date = st.date_input("Date When Found *", value=date.today())
            contact_info = st.text_input("Your Contact Info or Where Deposited *", placeholder="e.g., Security Office / finder@univ.edu")
            uploaded_image = st.file_uploader("Upload Image (Optional)", type=["jpg", "jpeg", "png"])

        description = st.text_area(
            "Detailed Description *",
            placeholder="Describe item appearance, model, where it was spotted, condition, etc.",
            help="Text is vectorized using TF-IDF for cosine similarity matching."
        )

        predicted_category = "Other"
        confidence = 0.0
        if description.strip():
            predicted_category, confidence, _ = predict_category(description)
            st.caption(f"🤖 **AI Auto-Detected Category**: `{predicted_category}` (Confidence: {confidence * 100:.1f}%)")

        category = st.selectbox(
            "Category * (Verify or modify the AI suggested category)",
            CATEGORIES,
            index=CATEGORIES.index(predicted_category) if predicted_category in CATEGORIES else 0
        )

        submitted = st.form_submit_button("Submit Found Item Report 🎁", use_container_width=True)

        if submitted:
            if not item_name.strip() or not description.strip() or not contact_info.strip():
                st.error("⚠️ Please fill in all required fields (Item Name, Description, and Contact Info).")
            else:
                image_path = save_uploaded_file(uploaded_image)
                report_id = add_report(
                    report_type="found",
                    item_name=item_name.strip(),
                    category=category,
                    color=color,
                    location=location,
                    report_date=report_date,
                    description=description.strip(),
                    image_path=image_path,
                    contact_info=contact_info.strip()
                )
                st.success(f"✅ Found item report registered successfully! Report ID: #{report_id}")
                st.balloons()


# =====================================================================
# 4. FIND MY ITEM (MATCHING & EXPLAINABLE ML)
# =====================================================================
elif menu == "🎯 Find My Item":
    st.markdown('<div class="main-header">Find Matching Belongings</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Select a report to compare against opposite reports using TF-IDF, Cosine Similarity, and Logistic Regression.</div>',
        unsafe_allow_html=True
    )

    search_mode = st.radio(
        "I want to find a match for:",
        ["My Lost Item (Search across Found Items)", "An Item I Found (Search across Lost Items)"],
        horizontal=True
    )

    is_lost_search = "Lost Item" in search_mode
    target_type = "lost" if is_lost_search else "found"
    candidate_type = "found" if is_lost_search else "lost"

    my_reports = get_reports(report_type=target_type)
    candidate_reports = get_reports(report_type=candidate_type)

    if not my_reports:
        st.warning(f"No {target_type} items are currently reported in the database. Please submit a report first!")
    elif not candidate_reports:
        st.info(f"No {candidate_type} items are available in the database to compare against.")
    else:
        report_options = {
            f"#{r['id']} - {r['item_name']} ({r['category']}, {r['color']}) [Reported: {r['report_date']} at {r['location']}]": r['id']
            for r in my_reports
        }

        selected_label = st.selectbox("Select Your Report to Match:", list(report_options.keys()))
        selected_id = report_options[selected_label]
        target_report = get_report_by_id(selected_id)

        with st.expander("📄 View Details of Your Selected Report", expanded=True):
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                st.markdown(f"**Item Name:** {target_report['item_name']}")
                st.markdown(f"**Category:** {target_report['category']}")
                st.markdown(f"**Color:** {target_report['color']}")
            with c2:
                st.markdown(f"**Location:** {target_report['location']}")
                st.markdown(f"**Report Date:** {target_report['report_date']}")
                st.markdown(f"**Status:** `{target_report['status'].upper()}`")
            with c3:
                st.markdown(f"**Description:** {target_report['description']}")
                if target_report.get("image_path") and os.path.exists(target_report["image_path"]):
                    st.image(target_report["image_path"], width=150)

        st.markdown("### 🔍 Top Matching Results")
        top_matches = find_top_matches(target_report, candidate_reports, top_n=5)

        if not top_matches:
            st.info("No candidate matches found.")
        else:
            for idx, match in enumerate(top_matches, 1):
                cand = match["found_report"]
                is_potential = match["is_potential_match"]
                prob = match["logistic_regression_prob"]
                comp_score = match["composite_score"]
                text_sim = match["text_similarity"]

                with st.container():
                    st.markdown(f"#### Match #{idx}: {cand['item_name']} (Report #{cand['id']})")
                    col_m1, col_m2, col_m3 = st.columns([2, 1.2, 1.2])

                    with col_m1:
                        if is_potential:
                            st.markdown('<span class="match-badge-yes">✅ Potential Match (Logistic Regression = 1)</span>', unsafe_allow_html=True)
                        else:
                            st.markdown('<span class="match-badge-no">⚠️ Low Match Probability (Logistic Regression = 0)</span>', unsafe_allow_html=True)

                        st.markdown(f"**Description:** {cand['description']}")
                        st.markdown(f"**Category:** `{cand['category']}` | **Color:** `{cand['color']}` | **Location:** `{cand['location']}`")
                        st.markdown(f"**Reported On:** {cand['report_date']} | **Contact/Handover:** `{cand.get('contact_info', 'N/A')}`")

                        st.markdown("**Why this was suggested:**")
                        for r_point in match["reasons"]:
                            st.markdown(f"- {r_point}")

                    with col_m2:
                        st.markdown(f"**Composite Match Score:** <span class='score-highlight'>{comp_score}%</span>", unsafe_allow_html=True)
                        st.progress(min(1.0, max(0.0, comp_score / 100.0)))
                        st.markdown(f"**TF-IDF Text Similarity:** `{text_sim}%`")
                        st.markdown(f"**ML Match Probability:** `{prob}%`")
                        st.markdown(f"**Date Difference:** `{match['date_diff_days']} day(s)`")

                    with col_m3:
                        st.markdown("**ML Feature Contributions:**")
                        for f_name, f_val in match["feature_contributions"].items():
                            val_color = "green" if f_val > 0 else "red"
                            st.markdown(f"<small>{f_name}: <b style='color:{val_color}'>{f_val:+.2f}</b></small>", unsafe_allow_html=True)

                        if cand["status"] != "resolved":
                            if st.button(f"Mark as Resolved #{cand['id']}", key=f"res_{cand['id']}"):
                                update_report_status(cand["id"], "resolved")
                                update_report_status(target_report["id"], "resolved")
                                st.success("Marked both items as resolved!")
                                st.rerun()
                        else:
                            st.info("Already Marked Resolved")

                    st.markdown("---")

        st.caption("🛡️ **Notice**: Matching scores are statistical indicators produced by ML algorithms and do not represent verified ownership.")


# =====================================================================
# 5. ANALYTICS DASHBOARD
# =====================================================================
elif menu == "📊 Analytics Dashboard":
    st.markdown('<div class="main-header">Analytics & ML Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Data exploration, category distributions, and unsupervised K-Means clustering.</div>', unsafe_allow_html=True)

    all_reports = get_reports()
    if not all_reports:
        st.warning("No reports available in the database yet.")
    else:
        df_reports = pd.DataFrame(all_reports)

        col1, col2, col3, col4 = st.columns(4)
        total_lost = len(df_reports[df_reports["report_type"] == "lost"])
        total_found = len(df_reports[df_reports["report_type"] == "found"])
        total_resolved = len(df_reports[df_reports["status"] == "resolved"])

        with col1:
            st.metric("Total Lost Reports", total_lost)
        with col2:
            st.metric("Total Found Reports", total_found)
        with col3:
            st.metric("Resolved Cases", total_resolved)
        with col4:
            st.metric("Total Active Items", len(df_reports))

        st.markdown("---")

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("### 📊 Reports by Category")
            fig1, ax1 = plt.subplots(figsize=(6, 3.5))
            cat_counts = df_reports["category"].value_counts()
            sns.barplot(x=cat_counts.values, y=cat_counts.index, palette="Blues_r", ax=ax1)
            ax1.set_xlabel("Number of Reports")
            ax1.set_ylabel("Category")
            ax1.set_title("Item Category Distribution")
            plt.tight_layout()
            st.pyplot(fig1)

        with col_c2:
            st.markdown("### ⚖️ Lost vs. Found Reports")
            fig2, ax2 = plt.subplots(figsize=(6, 3.5))
            type_counts = df_reports["report_type"].value_counts()
            colors = ["#EF4444", "#10B981"]
            ax2.pie(type_counts.values, labels=[t.capitalize() for t in type_counts.index], autopct="%1.1f%%", colors=colors, startangle=140)
            ax2.set_title("Proportion of Lost vs. Found Reports")
            plt.tight_layout()
            st.pyplot(fig2)

        st.markdown("### 📍 Reports by Campus Location")
        fig3, ax3 = plt.subplots(figsize=(10, 3.5))
        loc_counts = df_reports["location"].value_counts().head(8)
        sns.barplot(x=loc_counts.index, y=loc_counts.values, palette="crest", ax=ax3)
        ax3.set_xlabel("Location")
        ax3.set_ylabel("Count")
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig3)

        st.markdown("---")

        st.markdown("### 🧠 Unsupervised Learning: K-Means Clustering")
        st.markdown("""
        K-Means partitions item descriptions into distinct topic clusters based on their TF-IDF vector representations.
        We apply **PCA (Principal Component Analysis)** to reduce dimensions to 2D for visualization.
        """)

        num_clusters = st.slider("Select Number of Clusters (K):", min_value=2, max_value=5, value=3)

        descriptions = df_reports["description"].tolist()
        cluster_labels, coords_2d, top_keywords = perform_kmeans_clustering(descriptions, n_clusters=num_clusters)

        df_reports["cluster"] = cluster_labels
        df_reports["pca_x"] = [c[0] for c in coords_2d]
        df_reports["pca_y"] = [c[1] for c in coords_2d]

        col_k1, col_k2 = st.columns([1.5, 1])

        with col_k1:
            fig4, ax4 = plt.subplots(figsize=(7, 4.5))
            scatter = sns.scatterplot(
                x="pca_x",
                y="pca_y",
                hue="cluster",
                palette="tab10",
                data=df_reports,
                s=90,
                alpha=0.9,
                ax=ax4
            )
            ax4.set_title(f"K-Means Item Clusters (K={num_clusters}) Projected via PCA")
            ax4.set_xlabel("Principal Component 1")
            ax4.set_ylabel("Principal Component 2")
            plt.tight_layout()
            st.pyplot(fig4)

        with col_k2:
            st.markdown("#### 🏷️ Cluster Keywords & Items")
            for cl_id in range(num_clusters):
                kw_str = ", ".join(top_keywords.get(cl_id, ["general"]))
                items_in_cl = len(df_reports[df_reports["cluster"] == cl_id])
                st.markdown(f"**Cluster #{cl_id}** ({items_in_cl} items)")
                st.caption(f"Top TF-IDF Terms: `{kw_str}`")


# =====================================================================
# 6. VIEW ALL REPORTS
# =====================================================================
elif menu == "📋 View All Reports":
    st.markdown('<div class="main-header">All Item Reports</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Browse, search, and filter all registered lost and found reports.</div>', unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        type_filter = st.selectbox("Filter by Type", ["All", "Lost", "Found"])
    with col_f2:
        cat_filter = st.selectbox("Filter by Category", ["All"] + CATEGORIES)
    with col_f3:
        loc_filter = st.selectbox("Filter by Location", ["All"] + COMMON_LOCATIONS)

    search_query = st.text_input("🔍 Search by Item Name or Description Keyword", placeholder="e.g., iPhone, wallet, keys, library...")

    filtered_reports = get_reports(
        report_type=type_filter,
        category=cat_filter,
        location=loc_filter,
        search_query=search_query
    )

    st.markdown(f"Showing **{len(filtered_reports)}** reports:")

    if not filtered_reports:
        st.info("No matching reports found with the current filters.")
    else:
        for r in filtered_reports:
            badge_color = "#FEE2E2" if r["report_type"] == "lost" else "#D1FAE5"
            badge_text = "#991B1B" if r["report_type"] == "lost" else "#065F46"

            with st.container():
                st.markdown(f"""
                <div class="card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h4 style="margin:0;">#{r['id']} - {r['item_name']}</h4>
                        <span style="background-color:{badge_color}; color:{badge_text}; padding:4px 10px; border-radius:12px; font-weight:600; font-size:0.85rem;">
                            {r['report_type'].upper()}
                        </span>
                    </div>
                    <p style="margin-top:6px; color:#374151;">{r['description']}</p>
                    <small style="color:#6B7280;">
                        <b>Category:</b> {r['category']} | <b>Color:</b> {r['color']} | <b>Location:</b> {r['location']} | <b>Date:</b> {r['report_date']} | <b>Status:</b> <code>{r['status']}</code>
                    </small>
                </div>
                """, unsafe_allow_html=True)
