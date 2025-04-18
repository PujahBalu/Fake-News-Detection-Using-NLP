
# 🧠 Fake News Detection using NLP

This project demonstrates how to build a Natural Language Processing (NLP) model to detect **fake** and **real** news using the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).

---

## 📦 Dataset

**Source**: [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)  
It includes two files:
- `Fake.csv`: Contains fake news articles
- `True.csv`: Contains real news articles

---

## 📁 Project Structure

```
fake_news_nlp/
├── fake_news_detection.ipynb
├── README.md
└── fake_news_data/
    ├── Fake.csv
    └── True.csv
```

---

## 🚀 Getting Started on Google Colab

1. Go to [Kaggle](https://www.kaggle.com/account), generate your API token (kaggle.json), and upload it to Colab:

```python
from google.colab import files
files.upload()  # Upload your kaggle.json file
```

2. Setup Kaggle authentication and download the dataset:

```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset
!unzip fake-and-real-news-dataset.zip -d fake_news_data
```

---

## 🛠️ Tools & Libraries

- Python 3.x
- Pandas, NumPy
- NLTK
- Scikit-learn
- Seaborn, Matplotlib

---

## 📊 Steps Involved

1. Load and preprocess dataset (merge, clean, label)
2. Tokenization and stopword removal
3. TF-IDF Vectorization
4. Train/test split
5. Apply ML models (Logistic Regression, Naive Bayes, etc.)
6. Evaluate with Accuracy, Precision, Recall, and Confusion Matrix

---

## 📈 Sample Output

- Confusion Matrix
- Accuracy score
- Classification report

---

## 📌 Future Work

- Use deep learning models like LSTM
- Deploy using Flask/Streamlit
- Integrate with real-time news feeds

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) © 2024 Pujah Balasubramaniam.

---
