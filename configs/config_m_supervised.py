"""Config file for GPT2 training.
"""
# pylint: disable=invalid-name

# loss weights

size = 'm'
is_counterfactual = False
type = 'supervised'
w_recon = 0.0 #10.0
w_bt = 0 #1.0
w_d_xx2 = 0 #1.0
w_d_x2 = 0 #1.0
w_fine = 1.0
w_xx2 = 0.0

lr = 0.00001
tau = 1

# if is_counterfactual:
#     input = 'x1xx2'
# else:
#     input = 'x1x2'

# input = ''




# name = '{}_w1_{}_w2_{}_w3_{}_w4_{}_wfine_{}_tau_{}_{}'.format(
#     gpu, w_recon, w_bt, w_d_xx2, w_d_x2, w_fine, tau, lr)
# name = '{}_{}_{}'.format(size, input, type)
name = 'train_{}_{}'.format(size, type)

# tfrecord_data_dir = "data/final_dev_data/"
tfrecord_data_dir = "data/supervised_data/"
#tfrecord_data_dir = "data/m_data_2/"
#tfrecord_data_dir = "data/final_test_data/"
max_seq_length = 128
max_decoding_length = max_seq_length

np = 1
train_batch_size = 2 #2 * np #2 #8 #32
max_train_epoch = 1000
display_steps = 20 # Print training loss every display_steps; -1 to disable
eval_steps = 200    # Eval on the dev set every eval_steps; -1 to disable
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
            'learning_rate': lr #1000.0 #1000.0
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
