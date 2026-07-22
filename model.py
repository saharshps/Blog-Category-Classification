import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from textblob import TextBlob

stop_words = set(stopwords.words('english'))

df = pd.read_csv(r"C:\Users\sahar\OneDrive\Desktop\ds and ml files\blogs.csv")

df = df.dropna(subset=['Data', 'Labels'])

print(df.head())
print(df.shape)
print(df['Labels'].value_counts())

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

df['cleaned'] = df['Data'].apply(clean_text)

tfidf = TfidfVectorizer(max_features=5000)

X = tfidf.fit_transform(df['cleaned'])
y = df['Labels']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = MultinomialNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

def get_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity

    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

df["Sentiment"] = df["Data"].apply(get_sentiment)

print(df["Sentiment"].value_counts())

print(df.groupby("Labels")["Sentiment"].value_counts())

plt.figure(figsize=(6, 5))

sns.countplot(x="Sentiment", data=df)

plt.title("Overall Sentiment Distribution")

plt.show()

sentiment_table = pd.crosstab(df["Labels"], df["Sentiment"])

sentiment_table.plot(kind="bar", figsize=(12, 6))

plt.title("Sentiment Distribution Across Categories")
plt.xlabel("Category")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

sample = ["Artificial Intelligence is transforming healthcare."]

sample_clean = clean_text(sample[0])

sample_vector = tfidf.transform([sample_clean])

prediction = model.predict(sample_vector)

print("Sample Blog:")
print(sample[0])
print("Predicted Category:", prediction[0])

df.to_csv("blogs_with_sentiment.csv", index=False)

print("Assignment Completed Successfully!")