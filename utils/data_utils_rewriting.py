# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utils of data preprocessing for GPT2 training.
"""

import os
import collections
import csv
import tensorflow as tf

# pylint: disable=invalid-name, too-many-arguments

class InputExample(object):

    def __init__(self, x1, x2, xx2, y, yy=None):
        self.x1 = x1
        self.x2 = x2
        self.xx2 = xx2
        self.y = y
        self.yy = yy


class InputFeatures(object):

    def __init__(self, x1x2yx1xx2, x1x2yx1my, x1x2yx1xx2yy,
                 x1x2yx1xx2_len, x1x2yx1my_len, x1x2yx1m_len,
                 x1x2yx1xx2yy_len):
        self.x1x2yx1xx2 = x1x2yx1xx2
        self.x1x2yx1xx2_len = x1x2yx1xx2_len

        self.x1x2yx1my = x1x2yx1my
        self.x1x2yx1my_len = x1x2yx1my_len
        self.x1x2yx1m_len = x1x2yx1m_len

        self.x1x2yx1xx2yy = x1x2yx1xx2yy
        self.x1x2yx1xx2yy_len = x1x2yx1xx2yy_len


    #x1x2_ids = tf.placeholder(tf.int32, shape=[None, None])
    #x1x2_len = tf.placeholder(tf.int32, shape=[None])
    #x1xx2_ids = tf.placeholder(tf.int32, shape=[None, None])
    #x1xx2_len = tf.placeholder(tf.int32, shape=[None])



#def _truncate_seqs(tokens_x1, tokens_x2, tokens_xx2, tokens_y, max_length):
#    while True:
#        total_length = len(tokens_x1) + len(tokens_x2) + len(tokens_xx2) + len(tokens_y)
#        if total_length <= max_length:
#            break
#        #tokens_x1.pop()
#        #tokens_x2.pop()
#        #tokens_xx2.pop()
#        tokens_y.pop()

def _truncate_seqs(x1, x2, xx2, y, max_length, encoder):
    while True:
        ids = encoder.encode(x1 + ' ' + x2 + ' ' + y + ' | ' + x1 + ' ' + xx2 + ' ' + y + ' ')
        if len(ids) <= max_length:
            break
        y_ = y.split()
        y = ' '.join(y_[:-1])
    return y


def process_single_example(example, max_seq_length, encoder):
    x1 = example.x1
    x2 = example.x2
    xx2 = example.xx2
    y = example.y
    yy = example.yy

    y = _truncate_seqs(x1, x2, xx2, y, max_seq_length-2, encoder)
    if yy is not None:
        yy = _truncate_seqs(x1, x2, xx2, yy, max_seq_length-2, encoder)

    mask_text = 'Unknown .'
    special = encoder.encoder['<|endoftext|>']

    x1_ids = encoder.encode(x1)

    x1x2 = x1 + ' ' + x2
    x1x2_ids = encoder.encode(x1x2)

    x1x2y = x1x2 + ' ' + y
    x1x2y_ids = encoder.encode(x1x2y)

    x1xx2 = x1 + ' ' + xx2
    x1xx2_ids = encoder.encode(x1xx2)

    #x1x2yx1xx2_ids = encoder.encode(x1x2y + ' ') + [special] + encoder.encode(' ' + x1xx2 + ' ')
    #x1x2yx1m_ids = encoder.encode(x1x2y + ' ') + [special] + encoder.encode(' ' + x1 + ' ' + mask_text + ' ')
    #x1x2yx1my_ids = encoder.encode(x1x2y + ' ') + [special] + encoder.encode(' ' + x1 + ' ' + mask_text + ' ' + y + ' ')
    x1x2yx1xx2_ids = encoder.encode(x1x2y + ' | ' + x1xx2)
    x1x2yx1m_ids = encoder.encode(x1x2y + ' | ' + x1 + ' ' + mask_text)
    x1x2yx1my_ids = encoder.encode(x1x2y + ' | ' + x1 + ' ' + mask_text + ' ' + y)

    #_truncate_seqs(tokens_x1, tokens_x2, tokens_xx2, tokens_y, max_seq_length)
    #if example.yy is not None:
    #    _truncate_seqs(tokens_x1, tokens_x2, tokens_xx2, tokens_yy, max_seq_length)

    if example.yy is not None:
        #x1x2yx1xx2yy_ids = encoder.encode(x1x2y + ' ') + [special] + encoder.encode(' ' + x1xx2 + ' ' + yy + ' ')
        x1x2yx1xx2yy_ids = encoder.encode(x1x2y + ' | ' + x1xx2 + ' ' + yy)
    else:
        x1x2yx1xx2yy_ids = [special for _ in range(max_seq_length)]

    assert len(x1x2yx1xx2_ids) < max_seq_length
    if len(x1x2yx1my_ids) >= max_seq_length:
        print(x1)
        print(x2)
        print(y)
        print(len(x1x2yx1my_ids))
        print(max_seq_length)
        assert len(x1x2yx1my_ids) < max_seq_length

    len_x1 = len(x1_ids)
    len_x1x2 = len(x1x2_ids)
    len_x1x2y = len(x1x2y_ids)
    len_x1xx2 = len(x1xx2_ids)
    len_x1x2yx1xx2 = len(x1x2yx1xx2_ids)
    len_x1x2yx1my = len(x1x2yx1my_ids)
    len_x1x2yx1m = len(x1x2yx1m_ids)
    len_x1x2yx1xx2yy = len(x1x2yx1xx2yy_ids)

    while len(x1_ids) < max_seq_length:
        x1_ids.append(special)
    while len(x1x2_ids) < max_seq_length:
        x1x2_ids.append(special)
    while len(x1x2y_ids) < max_seq_length:
        x1x2y_ids.append(special)
    while len(x1xx2_ids) < max_seq_length:
        x1xx2_ids.append(special)
    while len(x1x2yx1xx2_ids) < max_seq_length:
        x1x2yx1xx2_ids.append(special)
    while len(x1x2yx1my_ids) < max_seq_length:
        x1x2yx1my_ids.append(special)
    while len(x1x2yx1xx2yy_ids) < max_seq_length:
        x1x2yx1xx2yy_ids.append(special)

    feature = {
        "x1_ids": x1_ids,
        "x1_len": len_x1,
        "x1x2_ids": x1x2_ids,
        "x1x2_len": len_x1x2,
        "x1x2y_ids": x1x2y_ids,
        "x1x2y_len": len_x1x2y,
        "x1xx2_ids": x1xx2_ids,
        "x1xx2_len": len_x1xx2,
        "x1x2yx1xx2_ids": x1x2yx1xx2_ids,
        "x1x2yx1xx2_len": len_x1x2yx1xx2,
        "x1x2yx1my_ids": x1x2yx1my_ids,
        "x1x2yx1my_len": len_x1x2yx1my,
        "x1x2yx1m_len": len_x1x2yx1m,
        "x1x2yx1xx2yy_ids": x1x2yx1xx2yy_ids,
        "x1x2yx1xx2yy_len": len_x1x2yx1xx2yy
    }

    return feature


def read_raw_data_v2(path, mode):
    def _read_file(fn):
        with open(fn, 'r') as fin:
            lines = [line.strip() for line in fin]
        return lines

    def _get_fn(field):
        return os.path.join(path, '%s_%s.txt' % (mode, field))

    all_x1 = _read_file(_get_fn('x1'))
    all_x2 = _read_file(_get_fn('x2'))
    all_xx2 = _read_file(_get_fn('xx2'))
    all_y = _read_file(_get_fn('y'))

    yy_fn = _get_fn('yy')
    if os.path.isfile(yy_fn):
        all_yy = _read_file(yy_fn)
    else:
        all_yy = [None] * len(all_x1)

    print('#examples: %d' % len(all_x1))

    return [
        InputExample(
            x1=x1,
            x2=x2,
            xx2=xx2,
            y=y,
            yy=yy)
        for x1, x2, xx2, y, yy in zip(all_x1, all_x2, all_xx2, all_y, all_yy)
    ]


def file_based_convert_examples_to_features_v2(
        examples, max_seq_length, encoder, output_file, verbose=False):
    """Converts a set of examples to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (_, example) in enumerate(examples):

        fea = process_single_example(
            example, max_seq_length, encoder)

        if verbose:
            print(fea["x1x2yx1xx2_len"])

        def _create_int_feature(values):
            return tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values)))

        features = collections.OrderedDict()
        features["x1_ids"] = _create_int_feature(fea["x1_ids"])
        features["x1_len"] = _create_int_feature([fea["x1_len"]])
        features["x1x2_ids"] = _create_int_feature(fea["x1x2_ids"])
        features["x1x2_len"] = _create_int_feature([fea["x1x2_len"]])
        features["x1x2y_ids"] = _create_int_feature(fea["x1x2y_ids"])
        features["x1x2y_len"] = _create_int_feature([fea["x1x2y_len"]])
        features["x1xx2_ids"] = _create_int_feature(fea["x1xx2_ids"])
        features["x1xx2_len"] = _create_int_feature([fea["x1xx2_len"]])
        features["x1x2yx1xx2_ids"] = _create_int_feature(fea["x1x2yx1xx2_ids"])
        features["x1x2yx1xx2_len"] = _create_int_feature([fea["x1x2yx1xx2_len"]])
        features["x1x2yx1my_ids"] = _create_int_feature(fea["x1x2yx1my_ids"])
        features["x1x2yx1my_len"] = _create_int_feature([fea["x1x2yx1my_len"]])
        features["x1x2yx1m_len"] = _create_int_feature([fea["x1x2yx1m_len"]])
        features["x1x2yx1xx2yy_ids"] = _create_int_feature(fea["x1x2yx1xx2yy_ids"])
        features["x1x2yx1xx2yy_len"] = _create_int_feature([fea["x1x2yx1xx2yy_len"]])

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())



def prepare_TFRecord_data_v2(data_dir, max_seq_length, encoder, output_dir):
    """
    Args:
        data_dir: The input data directory.
        max_seq_length: Max sequence length.
        output_dir: The directory to save the TFRecord files in.
    """
    train_examples = read_raw_data_v2(data_dir, mode='train')
    print('##train examples: %d' % len(train_examples))
    train_file = os.path.join(output_dir, "train.tf_record")
    file_based_convert_examples_to_features_v2(
        train_examples, max_seq_length, encoder, train_file)

    eval_examples = read_raw_data_v2(data_dir, mode='dev')
    print('##dev examples: %d' % len(eval_examples))
    eval_file = os.path.join(output_dir, "dev.tf_record")
    file_based_convert_examples_to_features_v2(
       eval_examples, max_seq_length, encoder, eval_file)

    test_examples = read_raw_data_v2(data_dir, mode='test')
    #test_examples = test_examples[:100]
    # print('[WARNING] test set truncated . ')
    print('##test examples: %d' % len(test_examples))
    test_file = os.path.join(output_dir, "test.tf_record")
    file_based_convert_examples_to_features_v2(
       test_examples, max_seq_length, encoder, test_file, verbose=True)


