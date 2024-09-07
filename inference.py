from fastapi import FastAPI, Query
import pickle

# Initialize FastAPI app
app = FastAPI()

# Load the saved model and vectorizer
with open('spam_classifier.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('count_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Define a prediction endpoint
@app.post("/predict/")
def predict_spam(email: str = Query(..., description="Email content to classify")):
    # Transform the input email text
    email_vect = vectorizer.transform([email])
    # Make a prediction
    prediction = model.predict(email_vect)
    # Return the prediction result
    result = {"prediction": "spam" if prediction[0] == 1 else "not spam"}
    print(result)
    return result

import uvicorn
import nest_asyncio
from pyngrok import ngrok

# Apply the nest_asyncio patch
nest_asyncio.apply()

public_url = ngrok.connect(9005, "http")
print('Public URL:', public_url)

uvicorn.run(app, host='0.0.0.0', port=9005)