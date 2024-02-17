#!/usr/bin/env python
# coding: utf-8

# # Latihan Membuat Machine Learning Pipeline

# In[1]:


import tensorflow as tf
from tfx.components import CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform, Trainer, Tuner, Pusher
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
import os

from tfx.dsl.components.common.resolver import Resolver 
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import LatestBlessedModelStrategy 
from tfx.types import Channel 
from tfx.types.standard_artifacts import Model, ModelBlessing 

import tensorflow_model_analysis as tfma 
from tfx.components import Evaluator

#added by me
import tensorflow_transform as tft 
import logging
from absl import logging
from tfx.v1.components import FnArgs

import pandas as pd
import numpy as np


# ## Set Variable

# In[2]:


PIPELINE_NAME = "skincancer-pipeline"
SCHEMA_PIPELINE_NAME = "skincancer-tfdv-schema"

#Directory untuk menyimpan artifact yang akan dihasilkan
PIPELINE_ROOT = os.path.join('pipelines', PIPELINE_NAME)

# Path to a SQLite DB file to use as an MLMD storage.
METADATA_PATH = os.path.join('metadata', PIPELINE_NAME, 'metadata.db')

# Output directory where created models from the pipeline will be exported.
SERVING_MODEL_DIR = os.path.join('serving_model', PIPELINE_NAME)

## add more
TRANSFORM_MODULE_FILE = "skincancer_transform.py"
TRAINER_MODULE_FILE = "skincancer_trainer.py"

PUSHED_MODEL_PATH = './serving_model_dir_skincancer/skincancer-detection-model'


# In[3]:


DATA_ROOT = "data"


# In[4]:


interactive_context = InteractiveContext(pipeline_root=PIPELINE_ROOT)


# ---- ADDED MORE BY ME

# In[5]:


output = example_gen_pb2.Output(
    split_config = example_gen_pb2.SplitConfig(splits=[
        example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
        example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2)
    ])
)
example_gen = CsvExampleGen(input_base=DATA_ROOT, output_config=output)
interactive_context.run(example_gen)


# In[6]:


# summary statistic
statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
interactive_context.run(statistics_gen)


# In[7]:


interactive_context.show(statistics_gen.outputs['statistics'] )


# In[8]:


schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'])
interactive_context.run(schema_gen)


# In[9]:


interactive_context.show(schema_gen.outputs["schema"])


# In[10]:


example_validator = ExampleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=schema_gen.outputs['schema']
)
interactive_context.run(example_validator)


# In[11]:


interactive_context.show(example_validator.outputs['anomalies'])


# In[12]:


get_ipython().run_cell_magic('writefile', '{TRANSFORM_MODULE_FILE}', 'import tensorflow as tf\nimport tensorflow_transform as tft\n\nLABEL_KEY = "dx"\nNUMERIC_FEATURE_KEYS = ["age"]\nCATEGORICAL_FEATURE_KEYS = ["sex", "localization"]\nALL_FEATURE_KEYS = NUMERIC_FEATURE_KEYS + CATEGORICAL_FEATURE_KEYS\nNUM_OOV_BUCKETS = 1\n\ndef transformed_name(key):\n    """Renaming transformed features"""\n    return key + "_xf"\ndef preprocessing_fn(inputs):\n    """\n    Preprocess input features into transformed features\n    \n    Args:\n        inputs: map from feature keys to raw features.\n    \n    Return:\n        outputs: map from feature keys to transformed features.    \n    """\n    \n    outputs = {}\n    \n    for key in NUMERIC_FEATURE_KEYS:#\n        outputs[transformed_name(key)] = tft.sparse_tensor_to_dense_with_shape(\n                x=tf.cast(inputs[key], tf.int64), shape=[None, 1]\n        )\n#         outputs[transformed_name(key)] = tft.sparse_tensor_to_dense_with_shape(\n#                 x=tft.scale_to_0_1(inputs[key]), shape=[None, 1]\n#         )\n        print(f"key{transformed_name(key)}", outputs[transformed_name(key)] )\n\n    for key in CATEGORICAL_FEATURE_KEYS:\n        outputs[transformed_name(key)] = tft.compute_and_apply_vocabulary(\n                tf.strings.strip(inputs[key]),\n                num_oov_buckets=NUM_OOV_BUCKETS,\n                vocab_filename=key)\n        print(f"key{transformed_name(key)}", outputs[transformed_name(key)] )\n    \n    # For the label column we provide the mapping from string to index.\n    table_keys = ["nv", "mel", "bkl", "bcc", "akiec", "df", "vasc"]\n    with tf.init_scope():\n        initializer = tf.lookup.KeyValueTensorInitializer(\n            keys=table_keys,\n            values=tf.cast(tf.range(len(table_keys)), tf.int64),\n            key_dtype=tf.string,\n            value_dtype=tf.int64)\n        table = tf.lookup.StaticHashTable(initializer, default_value=-1)\n\n    # Remove trailing periods for test data when the data is read with tf.data.\n    # label_str  = tf.sparse.to_dense(inputs[LABEL_KEY])\n    label_str = inputs[LABEL_KEY]\n    label_str = tf.strings.lower(label_str)\n    data_labels = table.lookup(label_str)\n    transformed_label = tf.one_hot(\n          indices=data_labels, depth=len(table_keys), on_value=1.0, off_value=0.0)\n    outputs[transformed_name(LABEL_KEY)] = tf.reshape(transformed_label, [-1, len(table_keys)])\n    \n    \n    return outputs\n')


# In[13]:


os.path.abspath(TRANSFORM_MODULE_FILE)


# In[14]:


transform = Transform(
    examples=example_gen.outputs['examples'],
    schema= schema_gen.outputs['schema'],
    module_file=os.path.abspath(TRANSFORM_MODULE_FILE)
)
interactive_context.run(transform)


# ---- ADDED MORE BY ME (ML development)

# In[15]:


get_ipython().run_cell_magic('writefile', '{TRAINER_MODULE_FILE}', 'import tensorflow as tf\nimport tensorflow_transform as tft \nfrom tensorflow.keras import layers\nimport os  \nimport tensorflow_hub as hub\nfrom tfx.components.trainer.fn_args_utils import FnArgs\nfrom tfx.components import Transform, Tuner\nfrom tfx.v1.components import TunerFnResult\nfrom tfx.proto import trainer_pb2\nimport kerastuner as kt\n\nimport matplotlib.pyplot as plt\n \nLABEL_KEY = "dx"\nNUMERIC_FEATURE_KEYS = ["age"]\nCATEGORICAL_FEATURE_KEYS = ["sex", "localization"]\nALL_FEATURE_KEYS = NUMERIC_FEATURE_KEYS + CATEGORICAL_FEATURE_KEYS\nNUM_OOV_BUCKETS = 1\n\nPIPELINE_NAME = "skincancer-pipeline"\nPIPELINE_ROOT = os.path.join(\'pipelines\', PIPELINE_NAME)\nPIPELINE_TRANSFORMDIR = os.path.join(PIPELINE_ROOT, \'Transform\')\n \ndef transformed_name(key):\n    """Renaming transformed features"""\n    return key + "_xf"\n \ndef gzip_reader_fn(filenames):\n    """Loads compressed data"""\n    return tf.data.TFRecordDataset(filenames, compression_type=\'GZIP\')\n\ndef build_keras_inputs(working_dir):\n    tf_transform_output = tft.TFTransformOutput(working_dir)\n    feature_spec = tf_transform_output.transformed_feature_spec().copy()\n    print(f"featspec>>{feature_spec}")\n    print("-----------------")\n    feature_spec.pop(transformed_name(LABEL_KEY) )\n\n    # Build the `keras.Input` objects.\n    inputs = {}\n    for key, spec in feature_spec.items():\n        print(f"{key}|{spec}")\n        if isinstance(spec, tf.io.VarLenFeature):\n            inputs[key] = tf.keras.layers.Input(shape=[None], name=key, dtype=spec.dtype, sparse=True)\n        elif isinstance(spec, tf.io.FixedLenFeature):\n            inputs[key] = tf.keras.layers.Input(shape=spec.shape, name=key, dtype=spec.dtype)\n        else:\n            raise ValueError(\'Spec type is not supported: \', key, spec)\n\n    return inputs\n\ndef encode_inputs(inputs):\n    encoded_inputs = {}\n    for key in inputs:\n        feature = tf.expand_dims(inputs[key], -1)\n        if key in CATEGORICAL_FEATURE_KEYS:\n            num_buckets = tf_transform_output.num_buckets_for_transformed_feature(transformed_name(key) )\n            encoding_layer = tf.keras.layers.CategoryEncoding(num_tokens=num_buckets, output_mode=\'binary\', sparse=False)\n            encoded_inputs[transformed_name(key)] = encoding_layer(feature)\n        else:\n            encoded_inputs[transformed_name(key)] = feature\n\n    return encoded_inputs\n\ndef input_fn(file_pattern, \n             tf_transform_output,\n             num_epochs,\n             batch_size=64)->tf.data.Dataset:\n    """Get post_tranform feature & create batches of data"""\n    \n    # Get post_transform feature spec\n    transform_feature_spec = (\n        tf_transform_output.transformed_feature_spec().copy())\n    \n    # create batches of data\n    dataset = tf.data.experimental.make_batched_features_dataset(\n        file_pattern=file_pattern,\n        batch_size=batch_size,\n        features=transform_feature_spec,\n        reader=gzip_reader_fn,\n        num_epochs=num_epochs,\n        label_key = transformed_name(LABEL_KEY))\n    return dataset\n\n# os.environ[\'TFHUB_CACHE_DIR\'] = \'/hub_chace\'\n# embed = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4")\n\nembedding_dim=16\ndef model_builder(hp: kt.HyperParameters, fn_args: FnArgs) -> tf.keras.Model:\n    """Build machine learning model"""\n    inputs = build_keras_inputs(fn_args.transform_graph_path)\n    encoded_inputs = encode_inputs(inputs) #<< the most problematic...\n    stacked_inputs = tf.concat(tf.nest.flatten(encoded_inputs), axis=1)\n    x = layers.GlobalAveragePooling1D()(stacked_inputs)\n    x = layers.Dense(128, activation=\'relu\')(x)\n    x = layers.Dense(64, activation="relu")(x)\n    x = layers.Dense(32, activation="relu")(x)\n    outputs = layers.Dense(7, activation=\'sigmoid\')(x)\n    \n    \n    model = tf.keras.Model(inputs=inputs, outputs = outputs)\n    \n    model.compile(\n        loss = tf.losses.CategoricalCrossentropy(from_logits=True),\n        optimizer=tf.keras.optimizers.Adam(hp.get(\'learning_rate\')),\n        metrics=[tf.keras.metrics.BinaryAccuracy()]\n    \n    )\n    model.summary()\n    return model \n\ndef _get_serve_tf_examples_fn(model, tf_transform_output):\n    \n    model.tft_layer = tf_transform_output.transform_features_layer()\n    \n    @tf.function\n    def serve_tf_examples_fn(serialized_tf_examples):\n        \n        feature_spec = tf_transform_output.raw_feature_spec()\n        \n        feature_spec.pop(LABEL_KEY)\n        \n        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)\n        \n        transformed_features = model.tft_layer(parsed_features)\n        \n        # get predictions using the transformed features\n        return model(transformed_features)\n        \n    return serve_tf_examples_fn\n\ndef run_fn(fn_args: FnArgs) -> None:\n    \n    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), \'logs\')\n    \n    tensorboard_callback = tf.keras.callbacks.TensorBoard(\n        log_dir = log_dir, update_freq=\'batch\'\n    )\n    \n    es = tf.keras.callbacks.EarlyStopping(monitor=\'val_binary_accuracy\', mode=\'max\', verbose=1, patience=10)\n    mc = tf.keras.callbacks.ModelCheckpoint(fn_args.serving_model_dir, monitor=\'val_binary_accuracy\', \n            mode=\'max\', verbose=1, save_best_only=True)\n    \n    \n    # Load the transform output\n    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)\n    \n    # Create batches of data\n    train_set = input_fn(fn_args.train_files, tf_transform_output, 10)\n    val_set = input_fn(fn_args.eval_files, tf_transform_output, 10)\n    \n    # Build the model\n    model = model_builder(get_hyperparameters(), fn_args)\n    \n    \n    # Train the model\n    history = model.fit(x = train_set,\n            validation_data = val_set,\n            callbacks = [tensorboard_callback, es, mc],\n            steps_per_epoch = 1000, \n            validation_steps= 1000,\n            epochs=10)\n    \n    plt.plot(history.history[\'loss\'], label=\'Train\')\n    plt.plot(history.history[\'binary_accuracy\'], label=\'Eval\')\n    plt.ylim(0,max(plt.ylim()))\n    plt.legend()\n    plt.title(\'model train\');\n#     signatures = {\n#         \'serving_default\':\n#         _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(\n#                                     tf.TensorSpec(\n#                                     shape=[None],\n#                                     dtype=tf.string,\n#                                     name=\'examples\'))\n#     }\n    model.save(fn_args.serving_model_dir, save_format=\'tf\'\n#                , signatures=signatures\n    )\n    \ndef get_hyperparameters() -> kt.HyperParameters:\n    """Returns hyperparameters for building Keras model."""\n    hparams = kt.HyperParameters()\n    # Defines search space.\n#     hparams.Choice(\'learning_rate\', [1e-1, 1e-2, 1e-3], default=1e-2)\n    hparams.Float("learning_rate",\n            min_value=1e-4, max_value=10,\n            step=1)\n    return hparams\n\ndef tuner_fn(fn_args):\n    """Build the tuner using the KerasTuner API.\n    Args:\n    fn_args: Holds args used to tune models as name/value pairs.\n\n    Returns:\n    A namedtuple contains the following:\n      - tuner: A BaseTuner that will be used for tuning.\n      - fit_kwargs: Args to pass to tuner\'s run_trial function for fitting the\n                    model , e.g., the training and validation dataset. Required\n                    args depend on the above tuner\'s implementation.\n    """\n    # Memuat training dan validation dataset yang telah di-preprocessing\n    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)\n    train_set = input_fn(fn_args.train_files[0], tf_transform_output, num_epochs=5)\n    val_set = input_fn(fn_args.eval_files[0], tf_transform_output, num_epochs=5)\n\n    # Mendefinisikan strategi hyperparameter tuning\n    tuner = kt.Hyperband(model_builder(get_hyperparameters(), fn_args),\n            objective=\'val_accuracy\', max_epochs=10,\n            factor=3,\n            directory=fn_args.working_dir,\n            project_name=\'kt_hyperband\')\n#     tuner = kt.RandomSearch(\n#         model_builder.model_builder,\n#         max_trials=6,\n#         hyperparameters=model_builder.get_hyperparameters(),\n#         allow_new_entries=False,\n#         objective=keras_tuner.Objective(\'val_accuracy\', \'max\'),\n#         directory=fn_args.working_dir,\n#         project_name=\'imdb_sentiment_classification\')\n\n    return TunerFnResult(\n    tuner=tuner,\n    fit_kwargs={ "callbacks":[stop_early], \n            \'x\': train_set, \'validation_data\': val_set,\n            \'steps_per_epoch\': fn_args.train_steps, \'validation_steps\': fn_args.eval_steps\n    })\n')


# In[16]:


tuner = Tuner(
    module_file=os.path.abspath(TRAINER_MODULE_FILE),
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    train_args=trainer_pb2.TrainArgs(splits=['train'], num_steps=100),
    eval_args=trainer_pb2.EvalArgs(splits=['eval'], num_steps=50)
    )
tuner


# In[17]:


trainer  = Trainer(
    module_file=os.path.abspath(TRAINER_MODULE_FILE),
    examples = transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    hyperparameters=tuner.outputs['best_hyperparameters'],
    schema=schema_gen.outputs['schema'],
    train_args=trainer_pb2.TrainArgs(splits=['train']),
    eval_args=trainer_pb2.EvalArgs(splits=['eval'])
)
interactive_context.run(trainer)


# In[18]:


interactive_context.run(trainer)


# ---- analysis & evaluation models

# In[19]:


model_resolver = Resolver(
    strategy_class= LatestBlessedModelStrategy,
    model = Channel(type=Model),
    model_blessing = Channel(type=ModelBlessing)
).with_id('Latest_blessed_model_resolver')
 
interactive_context.run(model_resolver)


# In[20]:


eval_config = tfma.EvalConfig(
    model_specs=[tfma.ModelSpec(label_key='dx_xf')],
    slicing_specs=[tfma.SlicingSpec()],
    metrics_specs=[
        tfma.MetricsSpec(metrics=[
            
            tfma.MetricConfig(class_name='ExampleCount'),
            tfma.MetricConfig(class_name='AUC'),
            tfma.MetricConfig(class_name='FalsePositives'),
            tfma.MetricConfig(class_name='TruePositives'),
            tfma.MetricConfig(class_name='FalseNegatives'),
            tfma.MetricConfig(class_name='TrueNegatives'),
            tfma.MetricConfig(class_name='BinaryAccuracy',
                threshold=tfma.MetricThreshold(
                    value_threshold=tfma.GenericValueThreshold(lower_bound={'value':0.5}),
                    change_threshold=tfma.GenericChangeThreshold(
                        direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                        absolute={'value':0.0001}
                    )
                )
            )
        ])
    ]
 
)


# In[21]:


interactive_context.run(transform)


# In[22]:


evaluator = Evaluator(
    examples=transform.outputs['transformed_examples'],
    model=trainer.outputs['model'],
    baseline_model=model_resolver.outputs['model'],
    eval_config=eval_config)
 
interactive_context.run(evaluator)


# In[23]:


# Visualize the evaluation results
eval_result = evaluator.outputs['evaluation'].get()[0].uri
tfma_result = tfma.load_eval_result(eval_result)
tfma.view.render_slicing_metrics(tfma_result)
tfma.addons.fairness.view.widget_view.render_fairness_indicator(
    tfma_result
)


# ---- MODEL to PUSH DEPLOYMENT

# In[24]:


pusher = Pusher(
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    push_destination=pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(base_directory=PUSHED_MODEL_PATH)
    )
)
 
interactive_context.run(pusher)


# In[ ]:




