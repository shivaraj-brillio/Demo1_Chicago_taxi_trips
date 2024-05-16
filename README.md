
# Chicago Taxi Fare Prediction

![alt text](https://github.com/shivaraj-brillio/Demo1_Chicago_taxi_trips/blob/main/Assets/Images/chicago_taxi.jpg)




## Introduction

This project aims to build a machine learning pipeline using TensorFlow Extended (TFX) to predict the costs of customer trips in advance in Chicago . The pipeline includes several components such as data ingestion, data validation, data preprocessing, model training, model evaluation, and model deployment.

The main files in this project are:

1. `pipeline.ipynb`: This Jupyter Notebook file contains the code for creating and running the TFX pipeline.
2. `constants.py`: This Python file defines various constants used in the pipeline, such as feature names, bucket counts, and label keys.
3. `trainer.py`: This Python file contains the code for building and training the machine learning model.
4. `transform.py`: This Python file defines the preprocessing steps for transforming the raw data into a format suitable for model training.

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


## Additional Information

### Constants and Keys

In our project, we use some constants and keys to make things work. Here they are:

- **Numerical Features**: These are numbers like miles, fare, and seconds.
- **Bucket Features**: Features related to location like pickup and dropoff points.
- **Categorical Numerical Features**: Numbers represented in categories, like hours and months.
- **Categorical String Features**: Strings that represent categories, like payment type and company.
- **Label Key**: This tells us whether the passenger tipped more than 20% or not.
- **Fare Key**: Helps us know the fare of the taxi ride.

### Trainer and Transformer

- `trainer.py`: This file trains our machine to learn from the data.
- `transform.py`: It transforms the data into a form the machine can understand better.

## `constants.py`

This file defines various constants used throughout the project. Let's break down the key components:

- **NUMERICAL_FEATURES**: Lists the numerical features present in the dataset, such as trip miles, fare, and trip seconds.
- **BUCKET_FEATURES**: Identifies the features that are bucketized, including pickup and dropoff latitude and longitude.
- **FEATURE_BUCKET_COUNT**: Specifies the number of buckets used for encoding each feature during preprocessing.
- **CATEGORICAL_NUMERICAL_FEATURES**: Lists categorical features that are numerical, such as trip start hour, day, and month.
- **CATEGORICAL_STRING_FEATURES**: Identifies categorical features stored as strings, such as payment type and company.
- **VOCAB_SIZE**: Determines the size of the vocabulary used for encoding categorical features.
- **OOV_SIZE**: Specifies the count of out-of-vocabulary buckets for unrecognized categorical features.
- **LABEL_KEY**: Indicates the key for the label column, which in this case is 'tips'.
- **FARE_KEY**: Specifies the key for the fare column.

These constants provide essential information for data preprocessing and model training.

## `transform.py`

This file contains the preprocessing functions required to transform raw input data into a format suitable for model training. Here's a breakdown of its functionality:

- **_make_one_hot**: This function encodes categorical features as one-hot tensors.
- **_fill_in_missing**: Replaces missing values in a SparseTensor with default values.
- **preprocessing_fn**: This callback function preprocesses the inputs using tf.transform. It performs the following transformations:
  - Scales numerical features to z-scores.
  - Bucketizes bucket features.
  - Encodes categorical string features as one-hot tensors.
  - Encodes categorical numerical features as one-hot tensors.
  - Determines if a passenger is a big tipper based on fare and tip percentage.

These preprocessing steps ensure that the input data is properly formatted and ready for model training.

## `trainer.py`

This file contains the training logic for the machine learning model. Let's explore its key components:

- **_input_fn**: Generates features and labels for training and evaluation datasets.
- **_get_tf_examples_serving_signature**: Returns a serving signature that accepts `tensorflow.Example`.
- **_get_transform_features_signature**: Returns a serving signature that applies tf.Transform to features.
- **export_serving_model**: Exports a Keras model for serving, including serving signatures.
- **_build_keras_model**: Creates a DNN Keras model for classifying taxi data.
- **run_fn**: Trains the model based on the provided arguments, including data accessors, transform output, and training settings.

This file orchestrates the model training process, including data preprocessing, model construction, training, and exporting for serving.


## Additional Resources

- TensorFlow Extended (TFX) Documentation: [https://www.tensorflow.org/tfx](https://www.tensorflow.org/tfx)
- Google Cloud Vertex AI Documentation: [https://cloud.google.com/vertex-ai/docs](https://cloud.google.com/vertex-ai/docs)
