import tensorflow as tf
import tensorflow_transform as tft

LABEL_KEY = "dx"
NUMERIC_FEATURE_KEYS = ["age"]
CATEGORICAL_FEATURE_KEYS = ["sex", "localization"]
ALL_FEATURE_KEYS = NUMERIC_FEATURE_KEYS + CATEGORICAL_FEATURE_KEYS
NUM_OOV_BUCKETS = 1

def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"
def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features
    
    Args:
        inputs: map from feature keys to raw features.
    
    Return:
        outputs: map from feature keys to transformed features.    
    """
    
    outputs = {}
    
    for key in NUMERIC_FEATURE_KEYS:#
        outputs[transformed_name(key)] = tft.sparse_tensor_to_dense_with_shape(
                x=tf.cast(inputs[key], tf.int64), shape=[None, 1]
        )
#         outputs[transformed_name(key)] = tft.sparse_tensor_to_dense_with_shape(
#                 x=tft.scale_to_0_1(inputs[key]), shape=[None, 1]
#         )
        print(f"key{transformed_name(key)}", outputs[transformed_name(key)] )

    for key in CATEGORICAL_FEATURE_KEYS:
        outputs[transformed_name(key)] = tft.compute_and_apply_vocabulary(
                tf.strings.strip(inputs[key]),
                num_oov_buckets=NUM_OOV_BUCKETS,
                vocab_filename=key)
        print(f"key{transformed_name(key)}", outputs[transformed_name(key)] )
    
    # For the label column we provide the mapping from string to index.
    table_keys = ["nv", "mel", "bkl", "bcc", "akiec", "df", "vasc"]
    with tf.init_scope():
        initializer = tf.lookup.KeyValueTensorInitializer(
            keys=table_keys,
            values=tf.cast(tf.range(len(table_keys)), tf.int64),
            key_dtype=tf.string,
            value_dtype=tf.int64)
        table = tf.lookup.StaticHashTable(initializer, default_value=-1)

    # Remove trailing periods for test data when the data is read with tf.data.
    # label_str  = tf.sparse.to_dense(inputs[LABEL_KEY])
    label_str = inputs[LABEL_KEY]
    label_str = tf.strings.lower(label_str)
    data_labels = table.lookup(label_str)
    transformed_label = tf.one_hot(
          indices=data_labels, depth=len(table_keys), on_value=1.0, off_value=0.0)
    outputs[transformed_name(LABEL_KEY)] = tf.reshape(transformed_label, [-1, len(table_keys)])
    
    
    return outputs
