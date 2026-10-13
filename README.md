# 📚 Book Genre Classification Using Naive Bayes
### College Information Retrieval (IR) Project

A complete, working web application implementing text classification using **Multinomial Naive Bayes (MNB)** with **Laplace Smoothing ($\alpha = 1.0$)** to predict literary book genres from titles and descriptions.

---

## 🎯 Project Overview
This project was developed for a college **Information Retrieval (IR)** course. In modern IR systems and digital libraries, document categorization is a fundamental task for indexing, faceted search, and cataloging. This application demonstrates the entire end-to-end machine learning pipeline:
1. **Document Representation**: Bag-of-Words feature extraction using `CountVectorizer` with stop-word removal.
2. **Probabilistic Modeling**: Multinomial Naive Bayes using class prior probabilities and word conditional probabilities.
3. **Zero-Frequency Mitigation**: Laplace smoothing ($\alpha = 1.0$) to guarantee non-zero likelihoods.
4. **Transparent Math Derivation**: Live breakdown showing the exact mathematical calculations from fitted parameters.
5. **Rigorous Evaluation**: 80/20 train/test split with Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

---

## 🏷️ Supported Genres (8 Categories)
1. **Fantasy** (🧙‍♂️): High magic, epic quests, mythical creatures, arcane realms.
2. **Science Fiction** (🛸): AI, cybernetics, space exploration, dystopian futures.
3. **Mystery** (🔍): Detectives, crime investigations, forensics, locked-room puzzles.
4. **Romance** (💖): Relationships, chemistry, romantic tension, devotion.
5. **Horror** (🩸): Supernatural entities, psychological dread, haunted settings, terror.
6. **Thriller** (⚡): Espionage, conspiracies, assassins, ticking-clock crises.
7. **Historical Fiction** (📜): Specific historical eras, wars, dynasties, period drama.
8. **Fiction** (📖): Contemporary, literary, domestic drama, human experiences.

---

## 🧮 Theoretical Foundation (Information Retrieval)

### 1. Bayes' Theorem
Given a document $D$ consisting of words $w_1, w_2, \dots, w_m$, the posterior probability of genre $C_k$ is:
$$P(C_k | D) = \frac{P(C_k) \cdot P(D | C_k)}{P(D)}$$

Since $P(D)$ is constant across all candidate classes, the classification decision rule is:
$$\hat{C} = \arg\max_{C_k} \left[ P(C_k) \prod_{i=1}^m P(w_i | C_k) \right]$$

### 2. Log-Space Computation
To avoid floating-point underflow when multiplying many small probabilities:
$$\log P(C_k | D) \propto \log P(C_k) + \sum_{i=1}^m f(w_i) \cdot \log P(w_i | C_k)$$
where $f(w_i)$ is the term count of word $w_i$ in the query.

### 3. Laplace Smoothing ($\alpha = 1$)
If an input word was never observed in training documents for genre $C_k$, $P(w_i | C_k)$ would become zero, wiping out the entire probability product. Laplace smoothing solves this by adding pseudo-counts:
$$P(w_i | C_k) = \frac{\text{count}(w_i, C_k) + 1}{\sum_{w'} \text{count}(w', C_k) + |V|}$$
where $|V|$ is the total unique vocabulary size (2,121 terms).

---

## 📁 Project Structure
```
book_genre_classifier/
├── app.py                     # Streamlit interactive web application
├── train_model.py             # Model training, split, evaluation & artifact exporter
├── generate_dataset.py        # Dataset generation script (200 curated books)
├── dataset.csv                # Demonstration dataset (25 books × 8 genres)
├── model.joblib               # Fitted MultinomialNB model artifact
├── vectorizer.joblib          # Fitted CountVectorizer artifact
├── evaluation_metrics.json    # Test set metrics, confusion matrix & reports
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation & execution guide
```

---

## 🚀 How to Run Locally

### 1. Prerequisites
Ensure you have Python 3.9+ installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model
```bash
python train_model.py
```
This performs the stratified 80/20 train/test split, calculates evaluation metrics, and saves `model.joblib`, `vectorizer.joblib`, and `evaluation_metrics.json`.

### 4. Launch the Web Application
```bash
streamlit run app.py
```
If using Python launcher or specific environment on Windows:
```powershell
python3.11 -m streamlit run app.py
```
The application will launch automatically in your default browser at `http://localhost:8501`.

---

## 📊 Model Evaluation Summary
Evaluated strictly on an independent 20% test dataset (40 held-out samples, 5 per genre) with zero data leakage:

| Metric | Score |
|---|---|
| **Accuracy** | **67.5%** |
| **Macro Precision** | **72.9%** |
| **Macro Recall** | **67.5%** |
| **Macro F1-Score** | **67.5%** |
| **Weighted F1-Score**| **67.5%** |

*Note: For a demonstration dataset of 200 books across 8 nuanced categories, achieving ~68% accuracy demonstrates solid discriminative capability for characteristic genre vocabulary.*

---

## ⚠️ Academic Demonstration Notice
This dataset contains 200 curated and verified book descriptions designed specifically for instructional demonstration of Information Retrieval principles. Real-world commercial classifiers utilize corpora containing hundreds of thousands of documents.
