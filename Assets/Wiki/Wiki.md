## Predicting Customer Trip Costs in Chicago using TensorFlow Extended (TFX)
### Table of Contents

- [Predicting Customer Trip Costs in Chicago using TensorFlow Extended (TFX)](#predicting-customer-trip-costs-in-chicago-using-tensorflow-extended-tfx)
  - [Table of Contents](#table-of-contents)
- [Chapter 1: Introduction to AutoML and Project Overview](#chapter-1-introduction-to-automl-and-project-overview)
- [Chapter 2: Data Transformation and Preprocessing with TFX](#chapter-2-data-transformation-and-preprocessing-with-tfx)
  - [Importing Necessary Modules](#importing-necessary-modules)
  - [Loading Constants](#loading-constants)
  - [Helper Functions for Data Preprocessing](#helper-functions-for-data-preprocessing)
    - [Handling Categorical Features](#handling-categorical-features)
    - [Filling Missing Values](#filling-missing-values)
  - [Main Preprocessing Function](#main-preprocessing-function)
- [Chapter 3: Building and Training the Model with TFX](#chapter-3-building-and-training-the-model-with-tfx)
  - [Importing Necessary Modules](#importing-necessary-modules-1)
  - [Defining Input Function](#defining-input-function)
  - [Defining Model Functions](#defining-model-functions)
    - [Building the Model](#building-the-model)
    - [Training the Model](#training-the-model)
- [Chapter 4: Defining Constants for Feature Engineering](#chapter-4-defining-constants-for-feature-engineering)
  - [Feature Constants](#feature-constants)
  - [Bucketizing Constants](#bucketizing-constants)
  - [Categorical Feature Constants](#categorical-feature-constants)
  - [Other Constants](#other-constants)
- [Chapter 5: Making Predictions with a Deployed Model on Google Cloud AI Platform](#chapter-5-making-predictions-with-a-deployed-model-on-google-cloud-ai-platform)
  - [Prediction Function](#prediction-function)
  - [Example Usage](#example-usage)
  - [Conclusion](#conclusion)

  - [Table of Contents](#table-of-contents)

- [Chapter 1: Introduction to AutoML and Project Overview](#chapter-1-introduction-to-automl-and-project-overview)

- [Chapter 2: Data Transformation and Preprocessing with TFX](#chapter-2-data-transformation-and-preprocessing-with-tfx)

  - [Importing Necessary Modules](#importing-necessary-modules)

  - [Loading Constants](#loading-constants)

  - [Helper Functions for Data Preprocessing](#helper-functions-for-data-preprocessing)

    - [Handling Categorical Features](#handling-categorical-features)

    - [Filling Missing Values](#filling-missing-values)

  - [Main Preprocessing Function](#main-preprocessing-function)

- [Chapter 3: Building and Training the Model with TFX](#chapter-3-building-and-training-the-model-with-tfx)

  - [Importing Necessary Modules](#importing-necessary-modules-1)

  - [Defining Input Function](#defining-input-function)

  - [Defining Model Functions](#defining-model-functions)

    - [Building the Model](#building-the-model)

    - [Training the Model](#training-the-model)

- [Chapter 4: Defining Constants for Feature Engineering](#chapter-4-defining-constants-for-feature-engineering)

  - [Feature Constants](#feature-constants)

  - [Bucketizing Constants](#bucketizing-constants)

  - [Categorical Feature Constants](#categorical-feature-constants)

  - [Other Constants](#other-constants)

- [Chapter 5: Making Predictions with a Deployed Model on Google Cloud AI Platform](#chapter-5-making-predictions-with-a-deployed-model-on-google-cloud-ai-platform)

  - [Prediction Function](#prediction-function)

  - [Example Usage](#example-usage)

  - [Conclusion](#conclusion)

---

## Chapter 1: Introduction to AutoML and Project Overview

In this comprehensive guide, we will delve into the world of Automated Machine Learning (AutoML) and explore how to transform raw data into valuable insights using sophisticated machine learning models. Our journey begins with understanding the basics of AutoML and the overview of a project that aims to build a machine learning pipeline using TensorFlow Extended (TFX) to predict the costs of customer trips in advance in Chicago.

The project pipeline includes several components such as data ingestion, data validation, data preprocessing, model training, model evaluation, and model deployment. Additionally, this project also demonstrates the use of Google Cloud AutoML for training a model without extensive coding.

---

## Chapter 2: Data Transformation and Preprocessing with TFX

In the realm of machine learning, data preprocessing is a crucial step that prepares raw data for modeling. The `transformer.py` file is designed to handle this task efficiently using TFX. It leverages TensorFlow and TensorFlow Transform to preprocess and transform input data.

### Importing Necessary Modules

Before diving into data preprocessing, let's start by understanding the modules imported in `transformer.py`. These modules are essential for data transformation and manipulation.

```python

import tensorflow as tf

import tensorflow_transform as tft

```

Here, TensorFlow and TensorFlow Transform are imported. TensorFlow is a powerful open-source machine learning framework, while TensorFlow Transform provides functionalities for preprocessing data before training models.

Additionally, the script dynamically loads a module named `taxi_constants.py` from a Google Cloud Storage bucket. This dynamic loading ensures that the latest version is used, avoiding caching issues during development.

### Loading Constants

The `taxi_constants` module contains various constants essential for feature engineering and data preprocessing. Let's dive into some of these constants to understand their significance.

```python

# Importing constants from taxi_constants module

_NUMERICAL_FEATURES = taxi_constants.NUMERICAL_FEATURES

_BUCKET_FEATURES = taxi_constants.BUCKET_FEATURES

_FEATURE_BUCKET_COUNT = taxi_constants.FEATURE_BUCKET_COUNT

_CATEGORICAL_NUMERICAL_FEATURES = taxi_constants.CATEGORICAL_NUMERICAL_FEATURES

_CATEGORICAL_STRING_FEATURES = taxi_constants.CATEGORICAL_STRING_FEATURES

_VOCAB_SIZE = taxi_constants.VOCAB_SIZE

_OOV_SIZE = taxi_constants.OOV_SIZE

_FARE_KEY = taxi_constants.FARE_KEY

_LABEL_KEY = taxi_constants.LABEL_KEY

```

These constants play a significant role in guiding the transformation process. For instance, `_NUMERICAL_FEATURES` contains a list of numerical features, `_BUCKET_FEATURES` contains bucketized features, `_VOCAB_SIZE` represents the size of the vocabulary, and so on.

### Helper Functions for Data Preprocessing

Data preprocessing involves various transformations to prepare the data for modeling. Let's explore some of the helper functions defined in `transformer.py`.

#### Handling Categorical Features

The `_make_one_hot` function transforms categorical features into one-hot encoded vectors. Let's dissect this function to understand its inner workings.

```python

def _make_one_hot(x, key):

    """

    Make a one-hot tensor to encode categorical features.

    Args:

        x: A dense tensor.

        key: A string key for the feature in the input.

    Returns:

        A dense one-hot tensor as a float list.

    """

    integerized = tft.compute_and_apply_vocabulary(

        x,

        top_k=_VOCAB_SIZE,

        num_oov_buckets=_OOV_SIZE,

        vocab_filename=key,

        name=key

    )

    depth = (

        tft.experimental.get_vocabulary_size_by_name(key) +

        _OOV_SIZE

    )

    one_hot_encoded = tf.one_hot(

        integerized,

        depth=tf.cast(depth, tf.int32),

        on_value=1.0,

        off_value=0.0

    )

    return tf.reshape(one_hot_encoded, [-1, depth])

```

This function takes a dense tensor `x` and a string key `key` as input and computes and applies a vocabulary to the input tensor. It then integerizes the tensor and converts it to a one-hot tensor.

#### Filling Missing Values

The `_fill_in_missing` function addresses missing values in the data. Let's explore how it handles missing values.

```python

def _fill_in_missing(x):

    """

    Replace missing values in a SparseTensor.

    Fills in missing values of `x` with '' or 0, and converts to a dense tensor.

    Args:

        x: A `SparseTensor` of rank 2. Its dense shape should have size at most

            1 in the second dimension.

    Returns:

        A rank 1 tensor where missing values of `x` have been filled in.

    """

    if not isinstance(x, tf.sparse.SparseTensor):

        return x

    default_value = '' if x.dtype == tf.string else 0

    return tf.squeeze(

        tf.sparse.to_dense(

            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),

            default_value

        ),

        axis=1

    )

```

This function takes a `SparseTensor` or a dense tensor `x` as input and fills in missing values with an appropriate default value. It then converts the result to a dense tensor.

### Main Preprocessing Function

The core function of this script, `preprocessing_fn`, acts as a callback for `tf.transform`. Let's explore how it transforms raw input features.

```python

def preprocessing_fn(inputs):

    """

    tf.transform's callback function for preprocessing inputs.

    Args:

        inputs: map from feature keys to raw not-yet-transformed features.

    Returns:

        Map from string feature key to transformed feature.

    """

    outputs = {}

    # Preprocess numerical features

    for key in _NUMERICAL_FEATURES:

        outputs[key] = _fill_in_missing(inputs[key])

    # Preprocess bucketized features

    for key in _BUCKET_FEATURES:

        outputs[key] = tft.bucketize(

            _fill_in_missing(inputs[key]), _FEATURE_BUCKET_COUNT, name=key)

    # Preprocess categorical numerical features

    for key in _CATEGORICAL_NUMERICAL_FEATURES:

        outputs[key] = _make_one_hot(

            tf.strings.strip(

                tf.strings.as_string(_fill_in_missing(inputs[key]))),

            key)

    # Preprocess categorical string features

    for key in _CATEGORICAL_STRING_FEATURES:

        outputs[key] = _make_one_hot(_fill_in_missing(inputs[key]), key)

    return outputs

```

This function takes a dictionary of raw, not-yet-transformed features as input and returns a dictionary of transformed features. It performs various transformations based on the feature type, including handling numerical, bucketized, categorical numerical, and categorical string features.

By understanding the inner workings of these functions, we gain insights into how data preprocessing is carried out in `transformer.py` using TFX. These transformations are essential for ensuring that the input data is suitable for training machine learning models.

---

## Chapter 3: Building and Training the Model with TFX

Once the data is preprocessed and transformed, the next step is to build and train a machine learning model. This process is implemented in the `trainer.py` file using TFX. Let's explore how this file facilitates model building and training.

### Importing Necessary Modules

The `trainer.py` file imports essential modules required for building and training machine learning models. Let's examine these imports to understand their role in the training process.

```python

from typing import List, Text

import tensorflow as tf

import tensorflow_transform as tft

import tensorflow_transform.beam.impl as tft_beam

import tensorflow_transform.beam.tft_beam_io as tft_beam_io

import tensorflow_data_validation as tfdv

```

These modules include TensorFlow, TensorFlow Transform, and TensorFlow Data Validation, which are fundamental for building and validating machine learning models. Additionally, the file imports specific modules for handling data transformation and beam implementation for distributed processing.

### Defining Input Function

The `_input_fn` function is responsible for generating features and labels for training and evaluation. Let's explore how this function is implemented.

```python

def _input_fn(file_pattern: List[Text],

              data_accessor: tfx.components.DataAccessor,

              tf_transform_output: tft.TFTransformOutput,

              batch_size: int = 200) -> tf.data.Dataset:

    def parse_tf_example(tf_example):

        feature_description = tf_transform_output.transformed_feature_spec()

        features = tf.io.parse_single_example(tf_example, feature_description)

        label = features.pop(_LABEL_KEY)

        return features, label

    dataset = data_accessor.tf_dataset_factory(

        file_pattern,

        tfxio.TensorFlowDatasetOptions(batch_size=batch_size),

        schema=tf_transform_output.transformed_feature_spec())

    dataset = dataset.map(parse_tf_example)

    return dataset

```

This function takes file patterns, a data accessor, the output of TensorFlow Transform, and batch size as inputs. It then parses TensorFlow examples, extracts features and labels, and creates a TensorFlow dataset for training or evaluation.

### Defining Model Functions

The `trainer.py` file defines functions for building and training machine learning models. Let's explore these functions to understand how models are constructed and trained.

#### Building the Model

The `_build_keras_model` function creates a Deep Neural Network (DNN) Keras model for classifying taxi data based on the transformed feature spec. Let's delve into its implementation.

```python

def _build_keras_model(tf_transform_output):

    feature_spec = tf_transform_output.transformed_feature_spec()

    inputs = {

        key: tf.keras.layers.Input(

            shape=feature_spec[key].shape,

            name=key,

            dtype=feature_spec[key].dtype) for key in feature_spec.keys()

    }

    x = tf.keras.layers.DenseFeatures(inputs)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(128, activation='relu')(x)

    x = tf.keras.layers.Dense(64, activation='relu')(x)

    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    return tf.keras.Model(inputs=inputs, outputs=output)

```

This function takes the output of TensorFlow Transform as input and creates a Keras model based on the transformed feature spec. It defines the input layer based on the shape and dtype of each feature, applies dense features, flattens the input, and adds dense layers with ReLU activation functions. Finally, it outputs a single unit with a sigmoid activation function for binary classification.

#### Training the Model

The `run_fn` function serves as the entry point for the TFX Trainer component. It takes TFX FnArgs as input, which holds arguments used to train the model. Let's explore its implementation.

```python

def run_fn(fn_args: tfx.components.FnArgs):

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(

        fn_args.train_files,

        fn_args.data_accessor,

        tf_transform_output,

        fn_args.train_batch_size)

    eval_dataset = _input_fn(

        fn_args.eval_files,

        fn_args.data_accessor,

        tf_transform_output,

        fn_args.eval_batch_size)

    model = _build_keras_model(tf_transform_output)

    model.compile(

        optimizer=tf.keras.optimizers.Adam(1e-2),

        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

        metrics=[tf.keras.metrics.BinaryAccuracy()])

    model.fit(

        train_dataset,

        steps_per_epoch=fn_args.train_steps,

        validation_data=eval_dataset,

        validation_steps=fn_args.eval_steps)

    transform_fn = fn_args.transform_graph_fn.inputs['transform'].get_tensor()

    export_serving_model(model, transform_fn, fn_args.serving_model_dir)

```

This function first loads the output of TensorFlow Transform and creates training and evaluation datasets using the `_input_fn` function. It then builds a Keras model using the `_build_keras_model` function, compiles the model with specified optimizer, loss function, and metrics, and trains the model using the training dataset. Finally, it exports the serving model for deployment.

---

## Chapter 4: Defining Constants for Feature Engineering

The `constants.py` file defines various constants used throughout the AutoML project. These constants include lists of numerical, bucketized, and categorical features, as well as vocabulary size, out-of-vocabulary bucket count, and keys for label and fare columns in the input data.

### Feature Constants

Let's explore the feature constants defined in `constants.py` to understand their role in feature engineering.

```python

NUMERICAL_FEATURES = [

    'trip_miles', 'fare', 'trip_seconds', 'trip_start_hour', 'trip_start_day',

    'trip_start_month'

]

```

The `NUMERICAL_FEATURES` constant represents a list of numerical features present in the input data. These features include attributes such as trip distance, fare amount, trip duration, and start time components like hour, day, and month.

```python

BUCKET_FEATURES = [

    'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude'

]

```

The `BUCKET_FEATURES` constant contains features that are bucketized based on their geographic coordinates. These features include latitude and longitude of pickup and dropoff locations, which are discretized into buckets for modeling.

### Bucketizing Constants

```python

FEATURE_BUCKET_COUNT = 10

```

The `FEATURE_BUCKET_COUNT` constant specifies the number of buckets used for bucketizing geographical coordinates. In this case, each coordinate is divided into 10 buckets to represent different geographical regions.

### Categorical Feature Constants

```python

CATEGORICAL_NUMERICAL_FEATURES = [

    'trip_start_hour', 'trip_start_day', 'trip_start_month',

    'pickup_census_tract', 'dropoff_census_tract', 'pickup_community_area',

    'dropoff_community_area'

]

```

The `CATEGORICAL_NUMERICAL_FEATURES` constant contains categorical numerical features present in the input data. These features include components of trip start time (hour, day, month) and identifiers for census tracts and community areas.

```python

CATEGORICAL_STRING_FEATURES = [

    'payment_type', 'company'

]

```

The `CATEGORICAL_STRING_FEATURES` constant includes categorical string features such as payment type and company. These features represent non-numeric attributes that require special handling during preprocessing.

### Other Constants

```python

VOCAB_SIZE = 1000

OOV_SIZE = 10

LABEL_KEY = 'tips'

FARE_KEY = 'fare'

```

The `VOCAB_SIZE` constant specifies the size of the vocabulary used for encoding categorical string features. Similarly, `OOV_SIZE` represents the number of out-of-vocabulary buckets. `LABEL_KEY` and `FARE_KEY` are keys for the label and fare columns in the input data, respectively.

By defining these constants, the `constants.py` file provides a centralized location for managing feature engineering configurations, making it easier to maintain and modify feature specifications as needed.

---

## Chapter 5: Making Predictions with a Deployed Model on Google Cloud AI Platform

Once a machine learning model is trained, the next step is to deploy it for making predictions on new data. The `client.py` script demonstrates how to make predictions using a deployed model on Google Cloud AI Platform.

### Prediction Function

The `predict_tabular_regression_sample` function in `client.py` allows users to make predictions using a deployed model. Let's explore how this function is implemented.

```python

def predict_tabular_regression_sample(

    project: str,

    endpoint_id: str,

    instance_dict: Dict,

    location: str = "us-central1",

    api_endpoint: str = "us-central1-aiplatform.googleapis.com",

):

    # The AI Platform services require regional API endpoints.

    client_options = {"api_endpoint": api_endpoint}

    # Initialize client that will be used to create and send requests.

    # This client only needs to be created once, and can be reused for multiple requests.

    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

    # Convert instance dictionary to Value protobuf object

    instance = json_format.ParseDict(instance_dict, Value())

    instances = [instance]

    parameters_dict = {}

    parameters = json_format.ParseDict(parameters_dict, Value())

    endpoint = client.endpoint_path(

        project=project, location=location, endpoint=endpoint_id

    )

    response = client.predict(

        endpoint=endpoint, instances=instances, parameters=parameters

    )

    print("response")

    print(" deployed_model_id:", response.deployed_model_id)

    predictions = response.predictions

    for prediction in predictions:

        print(" prediction:", dict(prediction))

```

This function takes inputs such as project ID, endpoint ID, instance data, location, and API endpoint. It then initializes a client for the PredictionService and sends prediction requests to the deployed model. The response contains information about the deployed model ID and the predictions made.

### Example Usage

The script also includes an example usage that demonstrates how to make predictions using the deployed model. Let's explore this example.

```python

if __name__ == "__main__":

    # TODO(developer): Uncomment and set the following variables

    # project = "your-project-id"

    # endpoint_id = "your-endpoint-id"

    bucket_name = "chicago_taxi_mlops_pipeline"

    blob_name = "Chicago_Taxi_From_2020.csv"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(blob_name)

    # Convert the CSV data to a string

    csv_data = blob.download_as_string()

    # Split the data into rows using the newline character

    rows = csv_data.decode("utf-8").split("\n")

    # Get the last row of data

    last_row = rows[-2]

    # Split the row into columns using the comma character

    columns = last_row.split(",")

    # Create a dictionary from the columns

    instance_dict = {

        "column_name": "column_value"

        for column_name, column_value in zip(columns[:-1], columns[1:])

    }

    # TODO(developer): Uncomment the following line to run the prediction

    # predict_tabular_regression_sample(project, endpoint_id, instance_dict)

```

In this example, the script reads data from a CSV file stored in a Google Cloud Storage bucket. It extracts the last row of data, converts it to a dictionary format, and passes it to the `predict_tabular_regression_sample` function for making predictions using the deployed model.

---

By understanding the functionalities provided by each script and module in the AutoML project, users can effectively preprocess data, train models, and make predictions on new data. This comprehensive guide serves as a roadmap for mastering AutoML techniques and leveraging them to derive meaningful insights from data.

### Conclusion

In this comprehensive guide, we've explored the inner workings of an AutoML (Automated Machine Learning) project, covering various aspects from data preprocessing to model deployment. By dissecting each component and script, we've gained insights into the complexities of building and deploying machine learning models at scale.

Throughout our journey, we've learned about:

1\. **Data Preprocessing**: Understanding the importance of preprocessing raw data for machine learning tasks. From handling missing values to encoding categorical features, preprocessing lays the foundation for building accurate models.

2\. **Model Building and Training**: Delving into the process of building and training machine learning models using TensorFlow and Keras. We've explored how to define model architectures, compile models with appropriate optimizers and loss functions, and train models using labeled datasets.

3\. **Feature Engineering**: Recognizing the significance of feature engineering in extracting meaningful insights from data. We've seen how to define feature constants and perform transformations to prepare input features for modeling.

4\. **Model Deployment and Inference**: Exploring the deployment of trained models for making predictions on new data. From setting up prediction endpoints to making inference requests, deploying models allows organizations to leverage machine learning in real-world scenarios.

By mastering the concepts presented in this comprehensive guide, we are equipped with the knowledge and skills needed to embark on our AutoML journey. Whether it's automating repetitive tasks, optimizing model performance, or scaling machine learning solutions, AutoML empowers organizations to unlock the full potential of their data.

As technology continues to evolve, the field of AutoML will undoubtedly play a crucial role in democratizing machine learning and making AI accessible to all. With the tools and techniques outlined in this comprehensive guide, we are well-positioned to thrive in the era of automated machine learning.

Now, armed with this knowledge, it's time to unleash the power of AutoML and embark on a journey of discovery and innovation in the fascinating world of machine learning.