"""Config file for GPT2 training.
"""
## fits to x1x2y

tfrecord_data_dir = "data/10w_data/"
lr = 0.00001

w_recon = 0.0
w_fine = 1.0
w_xx2 = 0.0

# never used
w_bt = 0
w_d_xx2 = 0
w_d_x2 = 0
tau = 1

name = ""

max_seq_length = 128
max_decoding_length = max_seq_length

np = 1
train_batch_size = 2 #2 * np #2 #8 #32
max_train_epoch = 100
display_steps = 20 # Print training loss every display_steps; -1 to disable
eval_steps = 300    # Eval on the dev set every eval_steps; -1 to disable
sample_steps = -1
checkpoint_steps = 2000 # Checkpoint model parameters every checkpoint_steps;
                      # -1 to disable

eval_batch_size = 4 #8
test_batch_size = 4 #8

## Optimization configs

opt = {
    'optimizer': {
        'type': 'AdamOptimizer',
        'kwargs': {
            'learning_rate': lr
        }
    }
}

## Data configs

feature_original_types = {
    # Reading features from TFRecord data file.
    # E.g., Reading feature "text_ids" as dtype `tf.int64`;
    # "FixedLenFeature" indicates its length is fixed for all data instances;
    # and the sequence length is limited by `max_seq_length`.
    "x1_ids": ["tf.int64", "FixedLenFeature", max_seq_length],
    "x1_len": ["tf.int64", "FixedLenFeature"],
    "x1x2_ids": ["tf.int64", "FixedLenFeature", max_seq_length],
    "x1x2_len": ["tf.int64", "FixedLenFeature"],
    "x1x2y_ids": ["tf.int64", "FixedLenFeature", max_seq_length],
    "x1x2y_len": ["tf.int64", "FixedLenFeature"],
    "x1xx2_ids": ["tf.int64", "FixedLenFeature", max_seq_length],
    "x1xx2_len": ["tf.int64", "FixedLenFeature"],
    "x1x2yx1xx2_ids": ["tf.int64", "FixedLenFeature", max_seq_length],
    "x1x2yx1xx2_len": ["tf.int64", "FixedLenFeature"],
    "x1x2yx1my_ids": ["tf.int64", "FixedLenFeature", max_seq_length],
    "x1x2yx1my_len": ["tf.int64", "FixedLenFeature"],
    "x1x2yx1m_len": ["tf.int64", "FixedLenFeature"],
    "x1x2yx1xx2yy_ids": ["tf.int64", "FixedLenFeature", max_seq_length],
    "x1x2yx1xx2yy_len": ["tf.int64", "FixedLenFeature"]
}
feature_convert_types = {
    # Converting feature dtype after reading. E.g.,
    # Converting the dtype of feature "text_ids" from `tf.int64` (as above)
    # to `tf.int32`
    "x1_ids": "tf.int32",
    "x1_len": "tf.int32",
    "x1x2_ids": "tf.int32",
    "x1x2_len": "tf.int32",
    "x1x2y_ids": "tf.int32",
    "x1x2y_len": "tf.int32",
    "x1xx2_ids": "tf.int32",
    "x1xx2_len": "tf.int32",
    "x1x2yx1xx2_ids": "tf.int32",
    "x1x2yx1xx2_len": "tf.int32",
    "x1x2yx1my_ids": "tf.int32",
    "x1x2yx1my_len": "tf.int32",
    "x1x2yx1m_len": "tf.int32",
    "x1x2yx1xx2yy_ids": "tf.int32",
    "x1x2yx1xx2yy_len": "tf.int32",
}

train_hparam = {
    "allow_smaller_final_batch": False,
    "batch_size": train_batch_size,
    "dataset": {
        "data_name": "data",
        "feature_original_types": feature_original_types,
        "feature_convert_types": feature_convert_types,
        "files": "{}/train.tf_record".format(tfrecord_data_dir)
    },
    "shuffle": True,
    "shuffle_buffer_size": 1000
}

dev_hparam = {
    "allow_smaller_final_batch": True,
    "batch_size": eval_batch_size,
    "dataset": {
        "data_name": "data",
        "feature_original_types": feature_original_types,
        "feature_convert_types": feature_convert_types,
        "files": "{}/dev.tf_record".format(tfrecord_data_dir)
    },
    "shuffle": False
}

# Set to `test_hparam` to `None` if generating from scratch
# (instead of generating continuation) at test time
test_hparam = {
    "allow_smaller_final_batch": True,
    "batch_size": test_batch_size,
    "dataset": {
        "data_name": "data",
        "feature_original_types": feature_original_types,
        "feature_convert_types": feature_convert_types,
        "files": "{}/test.tf_record".format(tfrecord_data_dir)
    },
    "shuffle": False
}
