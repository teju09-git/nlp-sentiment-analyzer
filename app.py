from flask import Flask, render_template, request
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

stop_words = set(stopwords.words('english'))

def preprocess(text):
    words = word_tokenize(text.lower())
    filtered = [w for w in words if w.isalnum() and w not in stop_words]
    return " ".join(filtered)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""

    if request.method == "POST":
        text = request.form["text"]
        processed = preprocess(text)
        vector = vectorizer.transform([processed])
        prediction = model.predict(vector)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))