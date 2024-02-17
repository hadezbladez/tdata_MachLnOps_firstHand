import tensorflow as tf
import tensorflow_transform as tft 
from tensorflow.keras import layers
import os  
import tensorflow_hub as hub
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components import Transform, Tuner
from tfx.v1.components import TunerFnResult
from tfx.proto import trainer_pb2
import kerastuner as kt

import matplotlib.pyplot as plt
 
LABEL_KEY = "dx"
NUMERIC_FEATURE_KEYS = ["age"]
CATEGORICAL_FEATURE_KEYS = ["sex", "localization"]
ALL_FEATURE_KEYS = NUMERIC_FEATURE_KEYS + CATEGORICAL_FEATURE_KEYS
NUM_OOV_BUCKETS = 1

PIPELINE_NAME = "skincancer-pipeline"
PIPELINE_ROOT = os.path.join('pipelines', PIPELINE_NAME)
PIPELINE_TRANSFORMDIR = os.path.join(PIPELINE_ROOT, 'Transform')
 
def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"
 
def gzip_reader_fn(filenames):
    """Loads compressed data"""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def build_keras_inputs(working_dir):
    tf_transform_output = tft.TFTransformOutput(working_dir)
    feature_spec = tf_transform_output.transformed_feature_spec().copy()
    print(f"featspec>>{feature_spec}")
    print("-----------------")
    feature_spec.pop(transformed_name(LABEL_KEY) )

    # Build the `keras.Input` objects.
    inputs = {}
    for key, spec in feature_spec.items():
        print(f"{key}|{spec}")
        if isinstance(spec, tf.io.VarLenFeature):
            inputs[key] = tf.keras.layers.Input(shape=[None], name=key, dtype=spec.dtype, sparse=True)
        elif isinstance(spec, tf.io.FixedLenFeature):
            inputs[key] = tf.keras.layers.Input(shape=spec.shape, name=key, dtype=spec.dtype)
        else:
            raise ValueError('Spec type is not supported: ', key, spec)

    return inputs

def encode_inputs(inputs):
    encoded_inputs = {}
    for key in inputs:
        feature = tf.expand_dims(inputs[key], -1)
        if key in CATEGORICAL_FEATURE_KEYS:
            num_buckets = tf_transform_output.num_buckets_for_transformed_feature(transformed_name(key) )
            encoding_layer = tf.keras.layers.CategoryEncoding(num_tokens=num_buckets, output_mode='binary', sparse=False)
            encoded_inputs[transformed_name(key)] = encoding_layer(feature)
        else:
            encoded_inputs[transformed_name(key)] = feature

    return encoded_inputs

def input_fn(file_pattern, 
             tf_transform_output,
             num_epochs,
             batch_size=64)->tf.data.Dataset:
    """Get post_tranform feature & create batches of data"""
    
    # Get post_transform feature spec
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())
    
    # create batches of data
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key = transformed_name(LABEL_KEY))
    return dataset

# os.environ['TFHUB_CACHE_DIR'] = '/hub_chace'
# embed = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4")

embedding_dim=16
def model_builder(hp: kt.HyperParameters, fn_args: FnArgs) -> tf.keras.Model:
    """Build machine learning model"""
    inputs = build_keras_inputs(fn_args.transform_graph_path)
    encoded_inputs = encode_inputs(inputs) #<< the most problematic...
    stacked_inputs = tf.concat(tf.nest.flatten(encoded_inputs), axis=1)
    x = layers.GlobalAveragePooling1D()(stacked_inputs)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(7, activation='sigmoid')(x)
    
    
    model = tf.keras.Model(inputs=inputs, outputs = outputs)
    
    model.compile(
        loss = tf.losses.CategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(hp.get('learning_rate')),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    
    )
    model.summary()
    return model 

def _get_serve_tf_examples_fn(model, tf_transform_output):
    
    model.tft_layer = tf_transform_output.transform_features_layer()
    
    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        
        feature_spec = tf_transform_output.raw_feature_spec()
        
        feature_spec.pop(LABEL_KEY)
        
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        
        transformed_features = model.tft_layer(parsed_features)
        
        # get predictions using the transformed features
        return model(transformed_features)
        
    return serve_tf_examples_fn

def run_fn(fn_args: FnArgs) -> None:
    
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir = log_dir, update_freq='batch'
    )
    
    es = tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', mode='max', verbose=1, patience=10)
    mc = tf.keras.callbacks.ModelCheckpoint(fn_args.serving_model_dir, monitor='val_binary_accuracy', 
            mode='max', verbose=1, save_best_only=True)
    
    
    # Load the transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    
    # Create batches of data
    train_set = input_fn(fn_args.train_files, tf_transform_output, 10)
    val_set = input_fn(fn_args.eval_files, tf_transform_output, 10)
    
    # Build the model
    model = model_builder(get_hyperparameters(), fn_args)
    
    
    # Train the model
    history = model.fit(x = train_set,
            validation_data = val_set,
            callbacks = [tensorboard_callback, es, mc],
            steps_per_epoch = 1000, 
            validation_steps= 1000,
            epochs=10)
    
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['binary_accuracy'], label='Eval')
    plt.ylim(0,max(plt.ylim()))
    plt.legend()
    plt.title('model train');
#     signatures = {
#         'serving_default':
#         _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
#                                     tf.TensorSpec(
#                                     shape=[None],
#                                     dtype=tf.string,
#                                     name='examples'))
#     }
    model.save(fn_args.serving_model_dir, save_format='tf'
#                , signatures=signatures
    )
    
def get_hyperparameters() -> kt.HyperParameters:
    """Returns hyperparameters for building Keras model."""
    hparams = kt.HyperParameters()
    # Defines search space.
#     hparams.Choice('learning_rate', [1e-1, 1e-2, 1e-3], default=1e-2)
    hparams.Float("learning_rate",
            min_value=1e-4, max_value=10,
            step=1)
    return hparams

def tuner_fn(fn_args):
    """Build the tuner using the KerasTuner API.
    Args:
    fn_args: Holds args used to tune models as name/value pairs.

    Returns:
    A namedtuple contains the following:
      - tuner: A BaseTuner that will be used for tuning.
      - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                    model , e.g., the training and validation dataset. Required
                    args depend on the above tuner's implementation.
    """
    # Memuat training dan validation dataset yang telah di-preprocessing
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_set = input_fn(fn_args.train_files[0], tf_transform_output, num_epochs=5)
    val_set = input_fn(fn_args.eval_files[0], tf_transform_output, num_epochs=5)

    # Mendefinisikan strategi hyperparameter tuning
    tuner = kt.Hyperband(model_builder(get_hyperparameters(), fn_args),
            objective='val_accuracy', max_epochs=10,
            factor=3,
            directory=fn_args.working_dir,
            project_name='kt_hyperband')
#     tuner = kt.RandomSearch(
#         model_builder.model_builder,
#         max_trials=6,
#         hyperparameters=model_builder.get_hyperparameters(),
#         allow_new_entries=False,
#         objective=keras_tuner.Objective('val_accuracy', 'max'),
#         directory=fn_args.working_dir,
#         project_name='imdb_sentiment_classification')

    return TunerFnResult(
    tuner=tuner,
    fit_kwargs={ "callbacks":[stop_early], 
            'x': train_set, 'validation_data': val_set,
            'steps_per_epoch': fn_args.train_steps, 'validation_steps': fn_args.eval_steps
    })
