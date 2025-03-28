import json
import boto3
import os
import logging
import pickle
import pandas as pd
import numpy as np
import csv
from io import StringIO
import sklearn


# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# SageMaker Runtime Client
runtime = boto3.client("runtime.sagemaker")

# Endpoint Name
SAGEMAKER_ENDPOINT = os.environ.get("SAGEMAKER_ENDPOINT", "kmeans")

# Load scaler, PCA, and frequency maps at cold start
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("pca.pkl", "rb") as f:
    pca = pickle.load(f)

with open("frequency_maps.pkl", "rb") as f:
    freq_maps = pickle.load(f)

def encode_categorical_features(df):
    """Apply frequency encoding using loaded frequency maps."""
    cat_cols = df.select_dtypes(include=["object"]).columns
    freq_frames = {}
    for col in cat_cols:
        if col in freq_maps:
            mapped_col = df[col].map(freq_maps[col])
            freq_frames[col + "_freq"] = mapped_col.fillna(0)
        else:
            logger.warning(f"⚠️ Column '{col}' missing from frequency map. Filling with 0.")
            freq_frames[col + "_freq"] = pd.Series(0, index=df.index)
    if freq_frames:
        freq_df = pd.DataFrame(freq_frames, index=df.index)
        df = pd.concat([df, freq_df], axis=1)
    df.drop(columns=cat_cols, inplace=True)
    return df

def preprocess_records(records):
    """Convert JSON records to PCA-transformed array."""
    df = pd.DataFrame(records)
    keep = ['trans_date_trans_time', 'cc_num', 'merchant', 'category', 'amt', 'zip', 'trans_num']
    df = df[keep].copy()
    df_encoded = encode_categorical_features(df)
    X_scaled = scaler.transform(df_encoded)
    X_pca = pca.transform(X_scaled)
    return X_pca

def convert_to_csv(matrix):
    """Convert numpy array or list of lists to CSV."""
    output = StringIO()
    writer = csv.writer(output)
    writer.writerows(matrix)
    return output.getvalue().encode("utf-8")

def lambda_handler(event, context):
    try:
        # Parse JSON body
        body = event.get("body")
        if isinstance(body, str):
            body = json.loads(body)

        records = body.get("instances", [])
        if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Invalid format. Expected list of dictionaries under 'instances'."})
            }

        # Preprocess data
        transformed = preprocess_records(records)

        # Convert to CSV
        csv_payload = convert_to_csv(transformed)

        # Invoke SageMaker endpoint
        response = runtime.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT,
            Body=csv_payload,
            ContentType="text/csv"
        )

        # Get prediction
        response_body = response["Body"].read().decode("utf-8")
        prediction = json.loads(response_body)

        return {
            "statusCode": 200,
            "body": json.dumps({"predictions": prediction})
        }

    except Exception as e:
        logger.error("Error during inference", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
