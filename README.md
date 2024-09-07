# Spam Classification

This project demonstrates the use of machine learning for spam email classification using the Naive Bayes algorithm. It includes model training, evaluation, and deployment through a FastAPI application.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [How to Run](#how-to-run)
- [Model Training](#model-training)
- [API Usage](#api-usage)

## Introduction

This project performs the following tasks:
- Downloads and preprocesses a spam email dataset.
- Trains a Multinomial Naive Bayes model for spam classification.
- Serves the trained model through a FastAPI application for real-time predictions.

## Requirements

Ensure you have the following dependencies installed:

```bash
fastapi
uvicorn
pandas
scikit-learn
kaggle
```

Install the required packages:

```bash
pip install fastapi uvicorn pandas scikit-learn kaggle
```

## Dataset

The dataset used in this project is the Spam Email dataset, which can be downloaded via the Kaggle API. Ensure you have your Kaggle API credentials set up.

Example of dataset download using Kaggle API:

```python
kaggle.api.dataset_download_files('mfaisalqureshi/spam-email', path='.', unzip=True)
```

## How to Run

1. **Train the model**:
    - Run `main.py` to perform model training and save the trained model and vectorizer.

    ```bash
    python main.py
    ```

    The model will be saved as `spam_classifier.pkl` and the vectorizer as `count_vectorizer.pkl`.

2. **Start the FastAPI server**:
    - Run `inference.py` to start the FastAPI server.

    ```bash
    uvicorn inference:app --reload
    ```

3. **Test the API**:
    You can use tools like Postman or `curl` to send POST requests to the API.

    Example `curl` request:

    ```bash
    curl -X POST "http://127.0.0.1:8000/predict/?email=Congratulations! You have won a lottery!" -H "Content-Type: application/json"
    ```

    The response will indicate whether the email is classified as "spam" or "not spam".

## Model Training

The `main.py` script performs the following:

1. Downloads the dataset and loads it into a DataFrame.
2. Cleans the email text by removing special characters and converting it to lowercase.
3. Encodes labels: 'ham' -> 0, 'spam' -> 1.
4. Splits the data into training and testing sets.
5. Vectorizes the text data using CountVectorizer.
6. Trains a Multinomial Naive Bayes model and evaluates its performance.
7. Saves the trained model and vectorizer.

## API Usage

The `inference.py` script uses FastAPI to serve predictions based on the trained model. You can send data in the following format:

```plaintext
GET /predict/?email=<email_content>
```

The response will provide the classification result:

```json
{
  "prediction": "spam"
}
```

or

```json
{
  "prediction": "not spam"
}
```
