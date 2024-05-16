# Define the numerical features that will be used in the model.
NUMERICAL_FEATURES = ['trip_miles', 'fare', 'trip_seconds']

# Define the features that will be bucketized.
BUCKET_FEATURES = [
    'pickup_latitude', 'pickup_longitude', 'dropoff_latitude',
    'dropoff_longitude'
]

# Define the number of buckets used by tf.transform for encoding each feature in BUCKET_FEATURES.
FEATURE_BUCKET_COUNT = 10

# Define the categorical features that are represented as numerical values.
CATEGORICAL_NUMERICAL_FEATURES = [
    'trip_start_hour', 'trip_start_day', 'trip_start_month',
    'pickup_census_tract', 'dropoff_census_tract', 'pickup_community_area',
    'dropoff_community_area'
]

# Define the categorical features that are represented as strings.
CATEGORICAL_STRING_FEATURES = [
    'payment_type',
    'company',
]

# Define the number of vocabulary terms used for encoding categorical features.
VOCAB_SIZE = 1000

# Define the count of out-of-vocab buckets in which unrecognized categorical are hashed.
OOV_SIZE = 10

# Define the keys for the label and fare columns in the input data.
LABEL_KEY = 'tips'
FARE_KEY = 'fare'

# Define a helper function that appends the suffix '_xf' to a feature key to avoid clashes
# with raw feature keys when running the Evaluator component.
def t_name(key):
    return key + '_xf'

