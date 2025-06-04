import pandas as pd
import numpy as np
import string
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    confusion_matrix
)

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
data = pd.read_csv('SSM Data.csv', encoding='ISO-8859-1')
data.drop(['Unnamed: 2', 'Unnamed: 3'], axis=1, inplace=True)

# Drop NaNs in 'sentiment' column
data['Comments'] = data['Comments'].fillna('').astype(str)
data = data.dropna(subset=['sentiment'])

# Clean sentiment column
data['sentiment'] = data['sentiment'].str.strip().str.lower()

# Preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens]
    table = str.maketrans('', '', string.punctuation + string.digits)
    stripped = [w.translate(table) for w in tokens]
    stop_words = set(stopwords.words('english'))
    words = [word for word in stripped if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Apply preprocessing
data['Comments'] = data['Comments'].apply(preprocess_text)

# Vectorization
vectorizer = TfidfVectorizer(max_df=0.75)
X = vectorizer.fit_transform(data['Comments'])
y = data['sentiment']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
y_pred_lr = logistic_regression.predict(X_test)

# Print accuracy
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f'Logistic Regression Accuracy: {accuracy_lr:.2f}')

# Predict function
def predict_sentiment(comment, model):
    processed_comment = preprocess_text(comment)
    vectorized_comment = vectorizer.transform([processed_comment])
    prediction = model.predict(vectorized_comment)
    print(f"The sentiment of the comment is: {prediction[0]}")

# Predict sample comment
sample_comment = "The new treatment has some bad side effects."
predict_sentiment(sample_comment, logistic_regression)

# Sentiment distribution visualization
sentiment_counts = data['sentiment'].value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='coolwarm')
plt.title('Distribution of Sentiments')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=['green', 'red', 'blue'])
plt.title('Sentiment Distribution')
plt.tight_layout()
plt.show()

# Map predictions for report
label_map = {'negative': 'Negative', 'neutral': 'Neutral', 'positive': 'Positive'}
y_pred_named = [label_map[label] for label in y_pred_lr]
y_test_named = [label_map[label] for label in y_test]

# Classification Report
report = classification_report(y_test_named, y_pred_named, target_names=['Negative', 'Neutral', 'Positive'])
print("Classification Report:")
print(report)

# Other metrics
accuracy = accuracy_score(y_test_named, y_pred_named)
precision, recall, f1, _ = precision_recall_fscore_support(y_test_named, y_pred_named, average='weighted')
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test_named, y_pred_named, labels=['Negative', 'Neutral', 'Positive'])
print("Confusion Matrix:")
print(conf_matrix)
