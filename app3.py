# This is the flask app we will be using for finding out the similarity between 2 texts using the model
# that we had trained earlier

from flask import Flask, request, jsonify, render_template
# import spacy
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
df = pd.read_csv("Precily_Text_Similarity.csv")

from sklearn.model_selection import train_test_split
traindf, testdf = train_test_split(df, test_size=0.2, random_state=42)

# nlp = spacy.load("en_core_web_lg")

# Loading the model
xgb_model = xgb.XGBRegressor()
xgb_model.load_model('xgbmodel2.model')

# Vectorizing the train data
vectorizer = TfidfVectorizer()
# vectorizer.fit(traindf['text1'] + traindf['text2'])
vectorizer.fit(traindf['text1'] + ' ' + traindf['text2'])

# Create the Flask app
app3 = Flask(__name__)

@app3.route('/')
def home():
    return render_template('index2.html')  # Render the index.html template

@app3.route('/api/predict', methods=['POST'])
def predict_similarity():
    # Get the JSON request body
    request_data = request.get_json()

    # Extract the text1 and text2 values from the request
    text1 = request_data['text1']
    text2 = request_data['text2']

    # Preprocess the texts
    # text1_processed = nlp(text1).text
    # text2_processed = nlp(text2).text

    # Vectorize the preprocessed texts
    text_vector = vectorizer.transform([text1 + text2])

    # Make prediction using the XGBoost model
    similarity_score = float(xgb_model.predict(text_vector)[0])

    # Prepare the response JSON
    response = {
        "similarity score": similarity_score
    }

    # response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    # response.headers['Pragma'] = 'no-cache'
    # response.headers['Expires'] = '0'

    # Return the response as JSON
    return jsonify(response)

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000)
    from werkzeug.serving import run_simple
    run_simple('0.0.0.0', 8080, app3)
