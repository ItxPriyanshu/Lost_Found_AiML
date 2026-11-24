# 🔍 AI-Powered Lost & Found Matcher

A simple, transparent, and practical Machine Learning web application designed to help university and college campuses reconnect lost belongings with their rightful owners.

Built with **Python**, **Streamlit**, and **Scikit-Learn**, this project demonstrates the real-world application of both **Supervised Learning** (Random Forest, Logistic Regression) and **Unsupervised Learning** (K-Means Clustering), accompanied by **Natural Language Processing (NLP)** using TF-IDF and Cosine Similarity.

---

## 📌 Table of Contents
1. [Project Overview & Problem Statement](#-project-overview--problem-statement)
2. [Key Features](#-key-features)
3. [Technologies Used](#-technologies-used)
4. [Machine Learning Algorithms Explained](#-machine-learning-algorithms-explained)
   - [Supervised vs. Unsupervised Learning](#supervised-vs-unsupervised-learning-in-this-project)
   - [A. Random Forest Classifier (Multiclass Classification)](#a-random-forest-classifier-multiclass-classification)
   - [B. TF-IDF Vectorization & Cosine Similarity (Text Matching)](#b-tf-idf-vectorization--cosine-similarity-text-matching)
   - [C. Logistic Regression (Binary Match Classification)](#c-logistic-regression-binary-match-classification)
   - [D. K-Means Clustering (Unsupervised Topic Grouping)](#d-k-means-clustering-unsupervised-topic-grouping)
5. [Project Architecture & Folder Structure](#-project-architecture--folder-structure)
6. [Installation & Setup](#-installation--setup)
7. [How to Run the Application](#-how-to-run-the-application)
8. [Sample Input & Output Walkthrough](#-sample-input--output-walkthrough)
9. [Viva Preparation & Common Questions](#-viva-preparation--common-questions)
10. [Limitations & Future Scope](#-limitations--future-scope)

---

## 📖 Project Overview & Problem Statement

### Problem
On busy university campuses, thousands of items (wallets, phones, student IDs, keys, laptops, and bags) are lost and found every semester. Traditional recovery systems rely on physical bulletin boards, scattered WhatsApp/Telegram groups, or cluttered lost-and-found desks. People describe identical objects using different vocabulary (e.g., *"black purse"* vs. *"leather wallet"*, or *"Dell notebook"* vs. *"silver laptop"*), causing most items to remain unclaimed.

### Solution
The **AI-Powered Lost & Found Matcher** automates and standardizes this matching process using traditional, explainable Machine Learning:
- Auto-detects the item category as a user types the description.
- Transforms unstructured English descriptions into numerical TF-IDF feature vectors.
- Computes cosine semantic similarity between lost and found descriptions.
- Evaluates attribute overlaps (category, color, location, and date difference).
- Uses a trained **Logistic Regression** model to predict whether two reports constitute a **Potential Match** with an explainable percentage score and transparent reasoning.

> **Important Note:** Machine Learning outputs are statistical indicators and probabilistic suggestions to aid human users. They do not claim or certify legal proof of ownership.

---

## ✨ Key Features

1. **Clean Multi-Page Web Interface (Streamlit)**:
   - 🏠 **Home**: Live KPI metrics (total lost, found, and resolved reports) and system overview.
   - 📢 **Report Lost Item**: Easy form with real-time AI category suggestion and optional image upload.
   - 🎁 **Report Found Item**: Quick registration of found items with auto-categorization.
   - 🎯 **Find My Item**: Select any report to view ranked top-5 matches, match probabilities, and transparent feature contributions.
   - 📊 **Analytics Dashboard**: Interactive charts (category distribution, lost vs. found ratios, campus locations) and a 2D PCA scatter plot of K-Means clusters.
   - 📋 **View All Reports**: Filter by type, category, or location with keyword search.

2. **Persistent Storage (SQLite)**:
   - Reports persist in a lightweight local SQLite database (`lost_found.db`).
   - Does not lose reports upon application restart.
   - Automatically pre-seeded with realistic sample campus reports for immediate viva demonstration.

3. **Explainable AI (XAI)**:
   - Every candidate match displays exact reasons for the match:
     - Matching category, location, and color
     - TF-IDF text cosine match percentage
     - Date proximity in days
     - Individual feature weights learned by Logistic Regression

---

## 🛠️ Technologies Used

- **Programming Language**: Python 3.10+
- **Frontend / Web UI**: [Streamlit](https://streamlit.io/)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/) (`RandomForestClassifier`, `LogisticRegression`, `KMeans`, `TfidfVectorizer`, `PCA`)
- **Database**: SQLite3 (built into Python standard library)
- **Data Visualization**: [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
- **Model Persistence**: [Joblib](https://joblib.readthedocs.io/)

*No complex deep learning frameworks, neural networks, or paid external APIs are required.*

---

## 🧠 Machine Learning Algorithms Explained

### Supervised vs. Unsupervised Learning in this Project

| Type | Algorithm | Task in this Project | Input | Output |
| :--- | :--- | :--- | :--- | :--- |
| **Supervised** | **Random Forest Classifier** | Multiclass classification | Cleaned description text (TF-IDF vector) | Category: *Wallet, Mobile Phone, Bag, Laptop, Watch, Keys, Other* |
| **Supervised** | **Logistic Regression** | Binary classification | 5 features: text similarity, same category, same color, same location, date difference | Potential Match: `1` (Yes) or `0` (No) |
| **Unsupervised** | **K-Means Clustering** | Unlabeled data grouping | High-dimensional TF-IDF vectors of all descriptions | Cluster IDs (e.g. Cluster 0, 1, 2) grouping similar reports |
| **NLP Metric** | **TF-IDF + Cosine Similarity** | Text semantic comparison | Two item descriptions | Similarity score between `0.0` and `1.0` |

---

### A. Random Forest Classifier (Multiclass Classification)
- **Purpose**: When a student enters a description (e.g., *"Lost my midnight blue iPhone 13 with cracked glass"*), the model predicts the category (*Mobile Phone*).
- **How it works**:
  1. The text is preprocessed (lowercase, punctuation stripped, stopwords removed).
  2. The text is converted into numbers using a **TF-IDF Vectorizer**.
  3. A **Random Forest** creates an ensemble of 100 decision trees during training. Each tree votes on the category, and the majority vote decides the output.
- **Why Random Forest?**: It resists overfitting, handles high-dimensional sparse TF-IDF vectors well, and gives class probabilities (`predict_proba`).

---

### B. TF-IDF Vectorization & Cosine Similarity (Text Matching)
- **TF-IDF (Term Frequency - Inverse Document Frequency)**:
  - **TF**: Measures how frequently a word occurs in a specific description.
  - **IDF**: Decreases the weight of words that occur across almost every description (e.g., *"lost"*, *"found"*, *"item"*) and boosts distinctive keywords (e.g., *"Casio"*, *"Inspiron"*, *"leather"*, *"AirPods"*).
  $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{|D|}{1 + |\{d \in D : t \in d\}|}\right)$$
- **Cosine Similarity**:
  Computes the cosine of the angle between two multi-dimensional TF-IDF vectors $\vec{A}$ and $\vec{B}$:
  $$\text{Cosine Similarity}(\vec{A}, \vec{B}) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|}$$
  - $1.0$: Identical word usage.
  - $0.0$: Completely orthogonal (no shared informative vocabulary).

---

### C. Logistic Regression (Binary Match Classification)
- **Purpose**: Decides whether a lost report and a found report represent a plausible match ($y = 1$) or not ($y = 0$).
- **Features Extracted for Each Pair**:
  1. `text_similarity` ($0.0$ to $1.0$)
  2. `same_category` ($1$ if identical, $0$ otherwise)
  3. `same_color` ($1$ if identical, $0$ otherwise)
  4. `same_location` ($1$ if identical, $0$ otherwise)
  5. `date_diff_days` (absolute number of days between the two incident dates)
- **Mathematical Formula**:
  $$z = w_1 \cdot \text{text\_sim} + w_2 \cdot \text{same\_cat} + w_3 \cdot \text{same\_col} + w_4 \cdot \text{same\_loc} + w_5 \cdot \text{date\_diff} + b$$
  $$P(\text{Match} = 1) = \sigma(z) = \frac{1}{1 + e^{-z}}$$
- **Learned Weights Intuition**:
  - Positive weights ($+w$) on matching category, color, location, and high text similarity boost the match probability.
  - Negative weight ($-w$) on date difference penalizes reports filed weeks or months apart.

---

### D. K-Means Clustering (Unsupervised Topic Grouping)
- **Purpose**: Automatically discovers natural clusters among all active reports without using any human labels.
- **Implementation**:
  - Transforms all descriptions in the database using TF-IDF.
  - Uses $K$-Means ($K=3$ or $4$) to cluster descriptions by minimizing the within-cluster sum of squares (inertia).
  - Uses **PCA (Principal Component Analysis)** to reduce dimensions down to 2 axes ($PC_1, PC_2$) for 2D scatter plotting on the Analytics Dashboard.
  - Extracts the top TF-IDF keywords per cluster centroid to display thematic tags.

---

## 📁 Project Architecture & Folder Structure

```
Lost_Found/
│
├── app.py                              # 🌐 The ONLY .py file: Complete self-contained Streamlit Web Application
│
├── 📓 JUPYTER NOTEBOOKS (.ipynb):
│   ├── AI_Powered_Lost_Found_Matcher.ipynb  # 🌟 Master All-in-One Notebook (11 complete sections & plots)
│   ├── lost_found_matcher.ipynb             # 🌟 Quick-access alias of the master notebook
│   ├── generate_dataset.ipynb               # 📊 Dataset generation & exploratory data analysis (EDA)
│   ├── ml_models.ipynb                      # 🤖 ML algorithms (Random Forest, Logistic Regression, K-Means)
│   ├── matching.ipynb                       # 🎯 Matching engine, text cosine similarity & explainability
│   └── database.ipynb                       # 💾 SQLite database creation, queries & report management
│
├── data/
│   ├── items_dataset.csv               # 315 items labeled by category for Random Forest training
│   └── matching_dataset.csv            # 225 item pairs labeled as match (1) or non-match (0)
│
├── models/
│   ├── category_classifier.pkl          # Serialized Random Forest model
│   ├── category_tfidf.pkl              # Serialized TF-IDF vectorizer for category classification
│   └── match_classifier.pkl             # Serialized Logistic Regression model
│
├── uploads/                            # User-uploaded images directory
├── lost_found.db                       # SQLite database (auto-seeded with campus sample reports)
├── requirements.txt                    # Python dependencies
└── README.md                           # Comprehensive documentation & viva guide
```

---

## 📓 Jupyter Notebooks Overview (.ipynb)

All backend code, training scripts, preprocessing, and matching engines are provided in interactive Jupyter Notebook (`.ipynb`) format:

| Notebook | Focus Area | What's Inside |
| :--- | :--- | :--- |
| [`AI_Powered_Lost_Found_Matcher.ipynb`](file:///d:/ML_Projects/Lost_Found/AI_Powered_Lost_Found_Matcher.ipynb) | **Master End-to-End Notebook** | All 11 steps: EDA, TF-IDF, Random Forest, Cosine Sim, Logistic Regression, K-Means, SQLite, and top-5 matching demonstration with pre-rendered plots. |
| [`generate_dataset.ipynb`](file:///d:/ML_Projects/Lost_Found/generate_dataset.ipynb) | **Dataset Creation & EDA** | Creates the 7-category `items_dataset.csv` and the labeled pair `matching_dataset.csv` with Seaborn category bar charts and pie charts. |
| [`ml_models.ipynb`](file:///d:/ML_Projects/Lost_Found/ml_models.ipynb) | **Machine Learning Training** | Preprocessing, Random Forest training & confusion matrix, Logistic Regression feature coefficients plot, and K-Means 2D PCA cluster map. |
| [`matching.ipynb`](file:///d:/ML_Projects/Lost_Found/matching.ipynb) | **Matching Engine & XAI** | TF-IDF Cosine similarity, multi-factor weighting, date difference calculation, and Explainable AI (XAI) reason builder. |
| [`database.ipynb`](file:///d:/ML_Projects/Lost_Found/database.ipynb) | **SQLite Database Manager** | Table creation, report insertion, querying with Pandas DataFrames, and summary statistics. |

---

## 🚀 How to Run

### 1. Interactive Jupyter Notebooks
Simply click on any `.ipynb` file in VS Code or run:
```bash
jupyter notebook AI_Powered_Lost_Found_Matcher.ipynb
```
All outputs and charts are already **pre-rendered**!

### 2. Streamlit Web Application
To run the interactive web interface (forms, dashboards, and instant search):
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14 installed on your computer.
- Git (optional, for cloning).

### 2. Navigate to Project Directory
```bash
cd d:\ML_Projects\Lost_Found
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate Datasets & Train Models (Automated)
Run the dataset generator and model trainer:
```bash
python generate_dataset.py
python ml_models.py
```
*Note: Even if you skip this step, `app.py` automatically checks for models and trains them on first startup!*

---

## 🚀 How to Run the Application

Launch the Streamlit web server:
```bash
streamlit run app.py
```

Once running, open your web browser at:
```
http://localhost:8501
```

---

## 🧪 Sample Input & Output Walkthrough

### Example 1: Reporting a Lost Item
- **Item Name**: `Black Leather Wallet`
- **Description**: `Black leather bi-fold wallet containing student ID card and bus pass.`
- **Color**: `Black`
- **Location**: `Central Library`
- **Date**: `2026-11-20`
- **Output**:
  - The AI suggests Category: `Wallet` (Confidence: >95%).
  - Saved to SQLite with ID `#1`.

### Example 2: Matching with Found Items
- Go to **🎯 Find My Item**.
- Select Report `#1 - Black Leather Wallet`.
- **Top Match Suggested**:
  - **Item**: `Dark Leather Wallet` (Found Report `#8`)
  - **Status**: `✅ Potential Match (Logistic Regression = 1)`
  - **Composite Score**: `79.9%`
  - **TF-IDF Text Match**: `42.5%`
  - **Explanation Reasons**:
    - Matching category: Both are 'Wallet'
    - Same location: Reported at 'Central Library'
    - Matching color: Both listed as 'Black'
    - Moderate description similarity (42% TF-IDF cosine match)
    - Reported on the exact same date

---

---

## 🔮 Limitations & Future Scope

### Limitations
- Uses traditional n-gram TF-IDF rather than contextual word embeddings, so extreme synonyms (e.g. *"spectacles"* vs. *"eyeglasses"*) rely on vocabulary overlap.
- Basic image handling stores files locally without computer vision feature extraction.

### Future Improvements
1. **Word Embeddings**: Incorporate Word2Vec or GloVe for semantic synonym matching.
2. **Computer Vision**: Add lightweight visual similarity using pre-extracted image color histograms.
3. **Automated Notification**: Integrate SMS or campus email notifications when a high-confidence match (>85%) is detected.

---

## 📜 License & Acknowledgments
Developed for educational, university, and college AI/ML academic presentations. Free to use and extend for academic coursework.
