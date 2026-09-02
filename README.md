# 🏦 Banking Query AI

An AI-powered banking query assistant that classifies customer queries into one of **77 banking issue categories** using sentence embeddings and a tuned Logistic Regression model — with voice input and webcam-based gesture confirmation.

---

## ✨ Features

- **Text or voice input** — type a banking query or speak it via the microphone (speech-to-text).
- **Query classification** — predicts the customer's issue category with a confidence score and top-3 alternative predictions.
- **Gesture confirmation** — show a hand gesture to the webcam to confirm, reject, or take no action on the prediction, powered by MediaPipe hand landmark detection:
  - 👍 Thumbs up → confirms
  - ✋ Open palm → rejects
  - ✊ Fist → no action
- **Modern dark-themed UI** built with Streamlit.

---

## 🧠 How It Works

```
Customer Query (typed or spoken)
        ↓
Text Preprocessing (lowercase, clean, normalize)
        ↓
Sentence Embedding (all-MiniLM-L6-v2, 384-dim)
        ↓
Tuned Logistic Regression
        ↓
Predicted Category + Confidence Score
        ↓
Optional: Gesture Confirmation (MediaPipe + OpenCV)
```

---

## 📊 Model

Four models were trained on sentence embeddings and compared as both baselines and after hyperparameter tuning:

| Model | Tuning Method |
|---|---|
| Logistic Regression | RandomizedSearchCV |
| SVC | RandomizedSearchCV |
| Random Forest | RandomizedSearchCV |
| ANN (Keras) | Optuna |

**Final model:** Tuned Logistic Regression

| Metric | Score |
|---|---|
| Test Accuracy | 92.79% |
| Weighted F1-score | 92.77% |

**Embedding model:** `all-MiniLM-L6-v2` (Sentence-Transformers)

---

## 📁 Dataset

[BANKING77](https://huggingface.co/datasets/PolyAI/banking77) — a dataset of real customer banking queries labeled with 77 fine-grained intent categories.

| Split | Records |
|---|---|
| Training | 10,003 |
| Testing | 3,080 |
| **Total** | **13,083** |

The original train/test split provided with the dataset was retained rather than re-split, to avoid data leakage and stay consistent with how the dataset is commonly benchmarked.

---

## 🗂️ Project Structure

```
Repository-name-Bank_Customer_Intent/
│
├── dataset/
│   ├── train.csv
│   └── test.csv
│
├── models/
│   ├── best_model.pkl          # Tuned Logistic Regression
│   ├── label_encoder.pkl
│   ├── embedding_model/        # Saved SentenceTransformer
│   └── hand_landmarker.task    # MediaPipe hand landmark model
│
├── main_model.ipynb            # Data prep, EDA, training, tuning, evaluation
├── app.py                      # Streamlit application
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/premaditya/Repository-name-Bank_Customer_Intent.git
cd Repository-name-Bank_Customer_Intent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

> **Note:** Make sure `models/hand_landmarker.task` is present — it's the MediaPipe hand landmark model used for gesture confirmation and must be downloaded separately. Get it from the [MediaPipe model index](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker).

---

## 🛠️ Tech Stack

- **Streamlit** — web app framework
- **Sentence-Transformers** (`all-MiniLM-L6-v2`) — text embeddings
- **scikit-learn** — Logistic Regression, SVC, Random Forest, model evaluation
- **TensorFlow / Keras** — ANN baseline
- **Optuna** — ANN hyperparameter tuning
- **MediaPipe + OpenCV** — hand landmark / gesture detection
- **streamlit-mic-recorder** — in-browser speech-to-text

---

## 📓 Notebook Contents (`main_model.ipynb`)

1. Data collection & understanding (shape, nulls, duplicates, class balance)
2. Text feature engineering (word count, char count, word frequency) + word cloud
3. Text preprocessing
4. Sentence embedding generation
5. Baseline model training (Logistic Regression, SVC, Random Forest, ANN)
6. Hyperparameter tuning (RandomizedSearchCV / Optuna)
7. Model comparison & final model selection
8. Saving model artifacts (`joblib` / SentenceTransformer `.save()`)

---

## 🙏 Acknowledgments

- [BANKING77 dataset](https://huggingface.co/datasets/PolyAI/banking77) — Casanueva et al., 2020
- [Sentence-Transformers](https://www.sbert.net/)
- [MediaPipe](https://ai.google.dev/edge/mediapipe)