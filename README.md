
# Chicago Taxi Tips Predictor

## Introduction

This project aims to build a machine learning pipeline using TensorFlow Extended (TFX) to predict whether a taxi passenger in Chicago will tip more than 20% of the fare or not. The pipeline includes several components such as data ingestion, data validation, data preprocessing, model training, model evaluation, and model deployment.

The main files in this project are:

1. `pipeline.ipynb`: This Jupyter Notebook file contains the code for creating and running the TFX pipeline.
2. `taxi_constants.py`: This Python file defines various constants used in the pipeline, such as feature names, bucket counts, and label keys.
3. `taxi_trainer.py`: This Python file contains the code for building and training the machine learning model.
4. `taxi_transform.py`: This Python file defines the preprocessing steps for transforming the raw data into a format suitable for model training.

## Prerequisites/Requirements

Before running the files, you need to have the following software installed:

1. **Python**: This project requires Python 3.6 or later. You can download Python from the official website: [https://www.python.org/downloads/](https://www.python.org/downloads/)

2. **TensorFlow Extended (TFX)**: TFX is a platform for deploying production machine learning pipelines. You can install TFX using pip:

```
pip install tfx
```

3. **Google Cloud SDK**: This project uses Google Cloud Storage for data storage and Google Cloud Vertex AI for model deployment. You need to install the Google Cloud SDK and authenticate with your Google Cloud account. Follow the instructions here: [https://cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install)

4. **Jupyter Notebook**: Although not strictly required, it is recommended to have Jupyter Notebook installed to run the `pipeline.ipynb` file. You can install Jupyter Notebook using pip:

```
pip install jupyter notebook
```

Once you have installed all the prerequisites, you can proceed to running the files.

## Setup

1. Clone this repository to your local machine:

```
git clone https://github.com/username/chicago-taxi-tips-predictor.git
```

2. Change to the project directory:

```
cd chicago-taxi-tips-predictor
```

3. Set up your Google Cloud project and bucket (replace `YOUR_PROJECT_ID` and `YOUR_BUCKET_NAME` with your actual values):

```
export GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID
export GCS_BUCKET_NAME=YOUR_BUCKET_NAME
```

## Running the Pipeline

1. Open the `pipeline.ipynb` file in Jupyter Notebook.

2. Follow the instructions in the notebook to run the TFX pipeline. The notebook will guide you through the following steps:
   - Install required libraries
   - Set up environment variables
   - Ingest data from BigQuery
   - Create TFX components (ExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform, Trainer, Evaluator, Pusher)
   - Run the pipeline

3. Once the pipeline execution is complete, you can check the results in your Google Cloud Storage bucket and Google Cloud Vertex AI.

## Additional Resources

- TensorFlow Extended (TFX) Documentation: [https://www.tensorflow.org/tfx](https://www.tensorflow.org/tfx)
- Google Cloud Vertex AI Documentation: [https://cloud.google.com/vertex-ai/docs](https://cloud.google.com/vertex-ai/docs)

