# app.py

from flask import Flask, render_template, request, render_template_string, send_file
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
import joblib
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

nltk.download('stopwords')

app = Flask(__name__)

def preprocess_text(review):
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    
    return review

def predict_sentiment(review):
    review = preprocess_text(review)
    X_review = cv.transform([review]).toarray()
    y_pred = classifier.predict(X_review)
    return y_pred[0]

def load_models():
    cvFile = 'c1_BoW_Sentiment_Model.pkl'
    cv = pickle.load(open(cvFile, 'rb'))
    
    classifier = joblib.load('c2_Classifier_Sentiment_Model')
    
    return cv, classifier

def generate_visualizations(df):
    all_reviews = ' '.join(df['Review'])
    wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110, background_color='white').generate(all_reviews)

    positive_counts = df[df['Sentiment'] == 1]['Sentiment'].count()
    negative_counts = df[df['Sentiment'] == 0]['Sentiment'].count()

    labels = ['Positive', 'Negative']
    sizes = [positive_counts, negative_counts]
    colors = ['#98FB98', '#FA8072']  # Change color for negative sentiment to red

    buffer = io.BytesIO()
    plt.figure(figsize=(8, 4))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    plt.axis('equal')
    plt.title('Sentiment Distribution')
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    pie_chart = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()

    # Separate word clouds for positive and negative sentiments
    positive_reviews = ' '.join(df[df['Sentiment'] == 1]['Review'])
    negative_reviews = ' '.join(df[df['Sentiment'] == 0]['Review'])

    wordcloud_positive = WordCloud(width=400, height=200, random_state=21, max_font_size=110, background_color='white').generate(positive_reviews)
    wordcloud_negative = WordCloud(width=400, height=200, random_state=21, max_font_size=110, background_color='white').generate(negative_reviews)

    buffer = io.BytesIO()
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud_positive, interpolation="bilinear")
    plt.axis('off')
    #plt.title('Word Cloud - Positive Sentiment')
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    wordcloud_positive = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()

    buffer = io.BytesIO()
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud_negative, interpolation="bilinear")
    plt.axis('off')
    #plt.title('Word Cloud - Negative Sentiment')
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    wordcloud_negative = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()

    return wordcloud, wordcloud_positive, wordcloud_negative, pie_chart

@app.route('/', methods=['GET', 'POST'])
def my_form_post():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('form.html', error="No file uploaded")

        file = request.files['file']
        if file.filename == '':
            return render_template('form.html', error="Empty file uploaded")

        if not (file.filename.endswith('.csv') or file.filename.endswith('.tsv')):
            return render_template('form.html', error="Invalid file format. Please upload a CSV or TSV file.")

        try:
            df = pd.read_csv(file, delimiter='\t' if file.filename.endswith('.tsv') else ',')
            df['Sentiment'] = df['Review'].apply(predict_sentiment)
            wordcloud, wordcloud_positive, wordcloud_negative, pie_chart = generate_visualizations(df)
            
            df.to_csv('output_with_sentiment.csv', index=False, sep='\t')

            return render_template_string("""
<div class="chart-container" style="text-align: center;">
    <h2>Pie Chart of Sentiment Analysis</h2>
    <div class="row">
        <div class="col">
            <h3>Positive vs. Negative</h3>
            <img src="data:image/png;base64,{{ pie_chart }}" alt="Positive Pie Chart" class="img-fluid">
        </div>
    </div>
</div>
                                        
                <div class="wordcloud-container" style="text-align: center;">
                    <div class="row">
                        <div class="col">
                            <h2>Word Cloud - Positive Sentiment</h2>
                            <img src="data:image/png;base64,{{ wordcloud_positive }}" alt="Positive Word Cloud">
                        </div>
                        <div class="col">
                            <h2>Word Cloud - Negative Sentiment</h2>
                            <img src="data:image/png;base64,{{ wordcloud_negative }}" alt="Negative Word Cloud">
                        </div>
                    </div>
                </div>

                <div style="text-align: center;">
    <a href="{{ url_for('download_file') }}" class="btn btn-primary mt-4">Download Output File</a>
</div>
            """, wordcloud=wordcloud, wordcloud_positive=wordcloud_positive, wordcloud_negative=wordcloud_negative, pie_chart=pie_chart)
        except Exception as e:
            return render_template('form.html', error=f"Error processing file: {str(e)}")

    return render_template('form.html')

@app.route('/download')
def download_file():
    path = 'output_with_sentiment.csv'
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    cv, classifier = load_models()
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)