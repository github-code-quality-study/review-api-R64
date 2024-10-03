import nltk
import cgi
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server
import urllib.parse

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        if environ["REQUEST_METHOD"] == "GET":
            # Write your code here
            sorted_reviews = []
            
            params = {}
            string_params = environ['QUERY_STRING']
            string_params = string_params.split('&')

            if len(string_params) > 0:
                for param in string_params:
                    param = param.split('=')
                    
                    if len(param) == 2:
                        params[param[0]] = urllib.parse.unquote_plus(param[1])
            
            for record in reviews:
                flag = True

                for param_name in params.keys():
                    if param_name == 'location':
                        if params['location'] != record['Location']:
                            flag = False
                    elif param_name == 'start_date':
                        if datetime.strptime(params['start_date'], '%Y-%m-%d').date() > datetime.strptime(record['Timestamp'], '%Y-%m-%d %H:%M:%S').date():
                            flag = False
                    elif param_name == 'end_date':
                        if datetime.strptime(params['end_date'], '%Y-%m-%d').date() < datetime.strptime(record['Timestamp'], '%Y-%m-%d %H:%M:%S').date():
                            flag = False

                if flag:
                    record['sentiment'] = self.analyze_sentiment(record['ReviewBody'])

                    sorted_reviews.append(record)

            # Create the response body from the reviews and convert to a JSON byte string
            response_body = json.dumps(sorted_reviews, indent=2).encode("utf-8")   

            # Set the appropriate response headers
            start_response("200 OK", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(response_body)))
            ])
            
            return [response_body]


        if environ["REQUEST_METHOD"] == "POST":
            # Write your code here
            available_locations = ['Albuquerque, New Mexico', 'Carlsbad, California', 'Chula Vista, California', 'Colorado Springs, Colorado', 'Denver, Colorado', 
                         'El Cajon, California', 'El Paso, Texas', 'Escondido, California', 'Fresno, California', 'La Mesa, California', 'Las Vegas, Nevada', 
                         'Los Angeles, California', 'Oceanside, California', 'Phoenix, Arizona', 'Sacramento, California', 'Salt Lake City, Utah', 
                         'Salt Lake City, Utah', 'San Diego, California', 'Tucson, Arizona']

            form = cgi.FieldStorage(fp=environ['wsgi.input'], environ=environ, keep_blank_values=True)

            params = {}
            
            if 'Location' in form:
                params["Location"] = urllib.parse.unquote_plus(form.getvalue("Location"))
            if 'ReviewBody' in form:
                params["ReviewBody"] = urllib.parse.unquote_plus(form.getvalue("ReviewBody"))
            
            if 'Location' in params.keys() and 'ReviewBody' in params.keys():
                if params['Location'] not in available_locations:
                    start_response("400 ERROR", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", '0')
                    ])
                
                    return []

                record = {
                    'ReviewId': str(uuid.uuid4()),
                    'Location': params['Location'],
                    'Timestamp': str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                    'ReviewBody': params['ReviewBody'],
                    'sentiment': self.analyze_sentiment(params['ReviewBody'])
                }
            else:
                start_response("400 ERROR", [
                ("Content-Type", "application/json"),
                ("Content-Length", '0')
                ])
                
                return []

            response_body = json.dumps(record)

            # Set the appropriate response headers
            start_response("201 OK", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(response_body.encode('utf-8'))))
            ])

            reviews.append(record)
            df = pd.DataFrame(reviews)
            df.to_csv('data/reviews.csv', index=False)

            return [response_body.encode('utf-8')]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()