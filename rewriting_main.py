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
"""Example of fine-tuning OpenAI GPT-2 language model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import importlib
import numpy as np
import tensorflow as tf
import texar as tx

from utils import model_utils, processor
from discriminator import Discriminator

# pylint: disable=invalid-name, too-many-locals, too-many-statements, no-member
# pylint: disable=too-many-branches

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint", None,
                    "Model checkpoint to resume training or for test.")
flags.DEFINE_string("pretrain_checkpoint",
                    "gpt2_pretrained_models/model_117M/model.ckpt",
                    "OpenAI pretrained model checkpoint. Ignored if "
                    "'--checkpoint' is specified.")
flags.DEFINE_string("pretrained_model_dir", "gpt2_pretrained_models/model_117M",
                    "The directory of pretrained model, for loading vocabuary, "
                    "etc.")
flags.DEFINE_float("temperature", 0.7,
                   "Softmax temperature for top-k sample decoding. Must be "
                   "strictly greater than 0. Defaults to 0.7.")
flags.DEFINE_integer("top_k", 40,
                     "The number of top most likely candidates from a vocab "
                     "distribution.")
flags.DEFINE_string("config_train", "configs.config_train",
                    "Configurations of GPT-2 training, including data and "
                    "optimization hyperparameters.")
flags.DEFINE_string("config_type", "texar",
                    "The configuration file type. Set to 'json' if the GPT-2 "
                    "config file is in the same type of the official GPT-2 "
                    "config file. Set to 'texar' if GPT-2 config file is in "
                    "Texar type.")
flags.DEFINE_string("config_model", "configs.config_model",
                    "The model configuration file to configure the model. "
                    "The config file type is define by the 'config_type',"
                    "it be of texar type or json type."
                    "For '--config_type=json', set the json config file path"
                    "like: '--config_model gpt2_pretrained_models/model_117M/"
                    "hparams.json';"
                    "For '--config_type=texar', set the texar config file "
                    "like: '--config_model configs.config_model'.")
flags.DEFINE_string("output_dir", "output/remove_space/",
                    "The output directory where the model checkpoints will be "
                    "written.")
flags.DEFINE_bool("do_train", False, "Whether to run training.")
flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")
flags.DEFINE_bool("do_test", False, "Whether to run test on the test set.")
flags.DEFINE_bool("distributed", False, "Whether to run in distributed mode.")
flags.DEFINE_bool("finetune", False, "Whether to test on finetune mode.")
flags.DEFINE_bool("roc", False, "Whether to test on rocstory mode.")
flags.DEFINE_bool("supervised", False, "Whether supervised training.")

config_train = importlib.import_module(FLAGS.config_train)


def _log(msg, log_fn=None):
    tf.logging.info(msg)
    if log_fn is None:
        log_fn = os.path.join(FLAGS.output_dir, config_train.name, 'log.txt')
    with open(log_fn, 'a') as flog:
        flog.write(msg + '\n')

def _ids_to_text(ids, proc):
    eos_token_id = proc.encoder['<|endoftext|>']

    if ids[0] == eos_token_id:
        ids = ids[1:]
    text = proc.decode(ids)
    return text

def main(_):
    """
    Builds the model and runs
    """
    if FLAGS.distributed:
        import horovod.tensorflow as hvd
        hvd.init()

    tf.logging.set_verbosity(tf.logging.INFO)

    if len(config_train.name) > 0:
        output_dir = os.path.join(FLAGS.output_dir, config_train.name)
    else:
        output_dir = FLAGS.output_dir
    tx.utils.maybe_create_dir(output_dir)


    ## Loads GPT-2 model configuration

    if FLAGS.config_type == "json":
        gpt2_config = model_utils.transform_gpt2_to_texar_config(
            FLAGS.config_model)
    elif FLAGS.config_type == 'texar':
        gpt2_config = importlib.import_module(
            FLAGS.config_model)
    else:
        raise ValueError('Unknown config_type.')

    # Creates a data pre-processor for, e.g., BPE encoding
    proc = processor.get_encoder(FLAGS.pretrained_model_dir)

    max_decoding_length = config_train.max_decoding_length
    assert max_decoding_length <= gpt2_config.position_size, (
        "max_decoding_length should not be greater than position_size. "
        "{}>{}".format(max_decoding_length, gpt2_config.position_size))

    ## Loads data

    # Configures training data shard in distribued mode
    if FLAGS.distributed:
        config_train.train_hparam["dataset"]["num_shards"] = hvd.size()
        config_train.train_hparam["dataset"]["shard_id"] = hvd.rank()
        config_train.train_hparam["batch_size"] //= hvd.size()

    datasets = {}
    #if FLAGS.do_train:
    train_dataset = tx.data.TFRecordData(hparams=config_train.train_hparam)
    datasets['train'] = train_dataset
    #if FLAGS.do_eval:
    dev_dataset = tx.data.TFRecordData(hparams=config_train.dev_hparam)
    datasets['dev'] = dev_dataset
    #if FLAGS.do_test:
    test_dataset = tx.data.TFRecordData(hparams=config_train.test_hparam)
    datasets['test'] = test_dataset
    iterator = tx.data.FeedableDataIterator(datasets)
    batch = iterator.get_next()
    batch_size = tf.shape(batch['x1x2yx1xx2_ids'])[0]

    ## Builds the GPT-2 model
    vocab_size = gpt2_config.vocab_size

    word_embedder = tx.modules.WordEmbedder(
        vocab_size=vocab_size,
        hparams=gpt2_config.embed)

    pos_embedder = tx.modules.PositionEmbedder(
        position_size=gpt2_config.position_size,
        hparams=gpt2_config.pos_embed)

    # Ties output layer with input word embedding
    output_layer = tf.transpose(word_embedder.embedding, (1, 0))

    decoder = tx.modules.TransformerDecoder(
        vocab_size=vocab_size,
        output_layer=output_layer,
        hparams=gpt2_config.decoder)

    # For training
    def _get_recon_loss(ids, full_len, prefix_len, mask_prefix=True, do_print=False):
        ids = ids[:,:tf.reduce_max(full_len)]
        batch_size__ = tf.shape(ids)[0]
        seq_len = tf.fill([batch_size__], tf.shape(ids)[1])
        pos_embeds = pos_embedder(sequence_length=seq_len)
        input_embeds = word_embedder(ids) + pos_embeds

        outputs = decoder(inputs=input_embeds, decoding_strategy='train_greedy')

        max_full_len = tf.reduce_max(full_len)
        ids = ids[:, :max_full_len]
        logits = outputs.logits[:, :max_full_len]

        if mask_prefix:
            loss_recon = tx.losses.sequence_sparse_softmax_cross_entropy(
                labels=ids[:, 1:],
                logits=logits[:, :-1, :],
                sequence_length=full_len-1,
                average_across_timesteps=False,
                sum_over_timesteps=False,
                average_across_batch=False,
                sum_over_batch=False)
            mask_recon = tf.sequence_mask(
                full_len-1,
                dtype=tf.float32)
            mask_recon_prefix = 1 - tf.sequence_mask(
                prefix_len-1,
                maxlen=max_full_len-1,#max_decoding_length-1,
                dtype=tf.float32)
            mask_recon = mask_recon * mask_recon_prefix

            if do_print:
                print_op_1 = tf.print(mask_recon)
                loss_recon_flat = tx.utils.reduce_with_weights(
                    tensor=loss_recon,
                    weights=mask_recon,
                    average_across_remaining=False,
                    sum_over_remaining=False,
                    average_across_batch=False)
                print_op_2 = tf.print(loss_recon_flat)
                with tf.control_dependencies([print_op_1, print_op_2]):
                    loss_recon = tx.utils.reduce_with_weights(
                        tensor=loss_recon,
                        weights=mask_recon,
                        average_across_remaining=True,
                        sum_over_remaining=False)
                return loss_recon, mask_recon, loss_recon_flat
            else:
                loss_recon = tx.utils.reduce_with_weights(
                    tensor=loss_recon,
                    weights=mask_recon,
                    average_across_remaining=True,
                    sum_over_remaining=False)
        else:
            loss_recon = tx.losses.sequence_sparse_softmax_cross_entropy(
                labels=ids[:, 1:],
                logits=logits[:, :-1, :],
                sequence_length=full_len-1,
                average_across_timesteps=True,
                sum_over_timesteps=False,
                average_across_batch=True,
                sum_over_batch=False)

        return loss_recon


    ## Loss-(1): mask reconstruction loss
    x1x2yx1my_ids = tf.placeholder(tf.int32, shape=[None, None], name='x1x2yx1my_ids')
    x1x2yx1my_len = tf.placeholder(tf.int32, shape=[None], name='x1x2yx1my_len')
    x1x2yx1m_len = tf.placeholder(tf.int32, shape=[None], name='x1x2yx1m_len')

    loss_mask_recon = _get_recon_loss(
        x1x2yx1my_ids, x1x2yx1my_len, x1x2yx1m_len)
    ppl_mask_recon = tf.exp(loss_mask_recon)

    ## Loss-(4): fine-tune loss
    x1x2_ids = tf.placeholder(tf.int32, shape=[None, None], name='x1x2_ids')
    x1x2_len = tf.placeholder(tf.int32, shape=[None], name='x1x2_len')
    x1x2y_ids = tf.placeholder(tf.int32, shape=[None, None], name='x1x2y_ids')
    x1x2y_len = tf.placeholder(tf.int32, shape=[None], name='x1x2y_len')

    loss_fine = _get_recon_loss(x1x2y_ids, x1x2y_len, x1x2_len, mask_prefix=False)

    ## Loss-(5): xx2 loss
    x1_len = tf.placeholder(tf.int32, shape=[None], name='x1_len')
    x1xx2_ids = tf.placeholder(tf.int32, shape=[None, None], name='x1xx2_ids')
    x1xx2_len = tf.placeholder(tf.int32, shape=[None], name='x1xx2_len')

    loss_xx2 = _get_recon_loss(x1xx2_ids, x1xx2_len, x1_len, do_print=False)

    ## Loss-(6): yy loss
    x1x2yx1xx2_ids = tf.placeholder(tf.int32, shape=[None, None], name='x1x2yx1xx2_ids')
    x1x2yx1xx2_len = tf.placeholder(tf.int32, shape=[None], name='x1x2yx1xx2_len')
    x1x2yx1xx2yy_ids = tf.placeholder(tf.int32, shape=[None, None], name='x1x2yx1xx2yy_ids')
    x1x2yx1xx2yy_len = tf.placeholder(tf.int32, shape=[None], name='x1x2yx1xx2yy_len')

    loss_yy = _get_recon_loss(x1x2yx1xx2yy_ids, x1x2yx1xx2yy_len, x1x2yx1xx2_len)

    ## Loss-(2): back-translation loss
    x1xx2yyx1x2y_ids = tf.placeholder(tf.int32, shape=[None, None], name='x1xx2yyx1x2y_ids')
    x1xx2yyx1x2y_len = tf.placeholder(tf.int32, shape=[None], name='x1xx2yyx1x2y_len')
    x1xx2yyx1x2_len = tf.placeholder(tf.int32, shape=[None], name='x1xx2yyx1x2_len')

    loss_bt = _get_recon_loss(
        x1xx2yyx1x2y_ids, x1xx2yyx1x2y_len, x1xx2yyx1x2_len)
    ppl_bt = tf.exp(loss_bt)

    ## Loss-(3): contrastive loss
    D = Discriminator(gpt2_config)

    tau = tf.placeholder(tf.float32, shape=[], name='tau')

    # generate soft yy
    def _soft_embedding_fn(soft_ids, times):
        return word_embedder(soft_ids=soft_ids) + pos_embedder(times)
    end_token = proc.encoder['<|endoftext|>']
    start_tokens = x1x2yx1xx2_ids[:, 0]

    helper_soft = tx.modules.SoftmaxEmbeddingHelper(
        embedding=_soft_embedding_fn,
        start_tokens=start_tokens,
        end_token=end_token,
        tau=tau,
        embedding_size=vocab_size)

    outputs_soft, len_soft = decoder(
        context=tf.one_hot(x1x2yx1xx2_ids, depth=vocab_size),
        context_sequence_length=x1x2yx1xx2_len,
        max_decoding_length=max_decoding_length,
        helper=helper_soft)
    yy_soft_ids = tx.utils.varlength_roll(
        outputs_soft.sample_id, -x1x2yx1xx2_len)
    yy_soft_len = len_soft - x1x2yx1xx2_len
    yy_soft_ids = yy_soft_ids[:, :tf.reduce_max(yy_soft_len), :]

    def _get_d_loss(prefix_ids, post_soft_ids, prefix_len, post_len):
        onehot_prefix_ids = tf.one_hot(prefix_ids, depth=vocab_size)
        soft_ids = tx.utils.varlength_concat(
            onehot_prefix_ids, post_soft_ids, prefix_len)
        soft_len = prefix_len + post_len
        return D.compute_loss(soft_ids, soft_len), soft_ids, soft_len

    loss_d_x2, _, _ = _get_d_loss(x1x2_ids, yy_soft_ids, x1x2_len, yy_soft_len) # to maximize
    loss_d_xx2, x1xx2yy_soft_ids, x1xx2yy_len = _get_d_loss(x1xx2_ids, yy_soft_ids, x1xx2_len, yy_soft_len) # to minimize

    x1xx2yy_ids = tf.argmax(x1xx2yy_soft_ids, axis=-1)

    if not FLAGS.supervised:
        loss = config_train.w_recon * loss_mask_recon \
                + config_train.w_fine * loss_fine \
                + config_train.w_xx2 * loss_xx2

        loss_dict = {
            'loss': loss,
            'loss_mask_recon': config_train.w_recon * loss_mask_recon,
            'loss_bt': tf.constant(0), #config_train.w_bt * loss_bt,
            'loss_d_xx2': tf.constant(0), #config_train.w_d_xx2 * loss_d_xx2,
            'loss_d_x2': tf.constant(0), #config_train.w_d_x2 * loss_d_x2,
            'loss_fine': config_train.w_fine * loss_fine,
            'loss_xx2': config_train.w_xx2 * loss_xx2,
        }
    else:
        loss = loss_yy

        loss_dict = {
            'loss': loss,
            'loss_yy': loss_yy,
            # dumb
            'loss_mask_recon': tf.constant(0),
            'loss_bt': tf.constant(0),
            'loss_d_xx2': tf.constant(0),
            'loss_d_x2': tf.constant(0),
            'loss_fine': tf.constant(0),
            'loss_xx2': tf.constant(0)
        }

    ## Inference
    def _embedding_fn(ids, times):
        return word_embedder(ids) + pos_embedder(times)

    def _infer(context_name):
        helper = tx.modules.TopKSampleEmbeddingHelper(
            embedding=_embedding_fn,
            start_tokens=batch['%s_ids' % context_name][:, 0],
            end_token=end_token,
            top_k=FLAGS.top_k,
            softmax_temperature=FLAGS.temperature)
        outputs_infer, len_infer = decoder(
            context=batch['%s_ids' % context_name],
            context_sequence_length=batch['%s_len' % context_name],
            max_decoding_length=max_decoding_length,
            helper=helper)
        yy_ids = tx.utils.varlength_roll(
            outputs_infer.sample_id, -batch['%s_len' % context_name])
        yy_len = len_infer - batch['%s_len' % context_name]
        yy_ids = yy_ids[:, :tf.reduce_max(yy_len)]
        return yy_ids, yy_len

    yy_ids, yy_len = _infer('x1x2yx1xx2')
    yy_ids_fine, yy_len_fine = _infer('x1xx2') # used in fine-tune
    yy_ids_roc, yy_len_roc = _infer('x1x2') # used in fine-tune
    ## Optimization
    trainable_variables = tx.utils.collect_trainable_variables(
        [word_embedder, pos_embedder, decoder])

    global_step = tf.Variable(0, trainable=False)
    opt = tx.core.get_optimizer(
        global_step=global_step,
        hparams=config_train.opt)

    if FLAGS.distributed:
        opt = hvd.DistributedOptimizer(opt)

    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=global_step,
        learning_rate=None,
        optimizer=opt,
        variables=trainable_variables)


    ## Train/eval/test routine
    saver = tf.train.Saver()
    saver_best = tf.train.Saver(max_to_keep=1)
    dev_best = {
        'loss': 1e8, 'loss_mask_recon': 1e8, 'loss_bt': 1e8,
        'loss_d_x1': 1e8, 'loss_d_xx2': 1e8, 'loss_fine': 1e8, 'loss_xx2': 1e8}


    def _log_losses(losses, step=None):
        loss_str = 'loss: %.4f, loss_mask_recon: %.4f, loss_bt: %.4f, loss_d_xx2: %.4f, loss_d_x2: %.4f, loss_fine: %.4f, loss_xx2: %.4f' % \
            (losses['loss'], losses['loss_mask_recon'], losses['loss_bt'],
             losses['loss_d_xx2'], losses['loss_d_x2'], losses['loss_fine'], losses['loss_xx2'])

        if step is not None:
            loss_str = 'step: %d, %s' % (step, loss_str)

        _log(loss_str)

    def _insert_yy(rets):
        batch_ = rets['batch']
        batch_size_ = rets['batch_size']
        yy_ids_ = rets['yy_ids']
        yy_len_ = rets['yy_len']

        x1x2y_ids_ = batch_['x1x2y_ids']
        x1x2y_len_ = batch_['x1x2y_len']

        x1xx2_ids_ = batch_['x1xx2_ids']
        x1xx2_len_ = batch_['x1xx2_len']

        x1xx2yy_ids_ = tx.utils.varlength_concat_py(x1xx2_ids_, yy_ids_, x1xx2_len_)
        x1xx2yy_len_ = x1xx2_len_ + yy_len_
        x1xx2yyx1x2y_ids_ = tx.utils.varlength_concat_py(x1xx2yy_ids_, x1x2y_ids_, x1xx2yy_len_)
        x1xx2yyx1x2y_len_ = x1xx2yy_len_ + x1x2y_len_
        x1xx2yyx1x2y_max_len_ = np.max(x1xx2yyx1x2y_len_)
        x1xx2yyx1x2y_ids_ = x1xx2yyx1x2y_ids_[:, :x1xx2yyx1x2y_max_len_]

        x1xx2yyx1x2_len_ = x1xx2yy_len_ + batch_['x1x2_len']

        return {
            'x1xx2yyx1x2y_ids': x1xx2yyx1x2y_ids_,
            'x1xx2yyx1x2y_len': x1xx2yyx1x2y_len_,
            'x1xx2yyx1x2_len': x1xx2yyx1x2_len_
        }

    def _is_head():
        if not FLAGS.distributed:
            return True
        else:
            return hvd.rank() == 0

    def _train_epoch(sess, initial=False):
        """Trains on the training set, and evaluates on the dev set
        periodically.
        """
        iterator.restart_dataset(sess, 'train')

        while True:
            try:
                # (1) Get data and yy sample
                fetches_data = {
                    'batch': batch,
                    'batch_size': batch_size,
                }
                feed_dict_data = {
                    iterator.handle: iterator.get_handle(sess, 'train'),
                    tx.global_mode(): tf.estimator.ModeKeys.PREDICT,
                }
                rets_data = sess.run(fetches_data, feed_dict_data)


                # (2) Optimize loss
                feed_dict = {
                    x1x2yx1my_ids: rets_data['batch']['x1x2yx1my_ids'],
                    x1x2yx1my_len: rets_data['batch']['x1x2yx1my_len'],
                    x1x2yx1m_len: rets_data['batch']['x1x2yx1m_len'],
                    x1x2yx1xx2_ids: rets_data['batch']['x1x2yx1xx2_ids'],
                    x1x2yx1xx2_len: rets_data['batch']['x1x2yx1xx2_len'],
                    #x1_ids: rets_data['batch']['x1_ids'],
                    x1_len: rets_data['batch']['x1_len'],
                    x1x2_ids: rets_data['batch']['x1x2_ids'],
                    x1x2_len: rets_data['batch']['x1x2_len'],
                    x1xx2_ids: rets_data['batch']['x1xx2_ids'],
                    x1xx2_len: rets_data['batch']['x1xx2_len'],
                    x1x2y_ids: rets_data['batch']['x1x2y_ids'],
                    x1x2y_len: rets_data['batch']['x1x2y_len'],
                    x1x2yx1xx2yy_ids: rets_data['batch']['x1x2yx1xx2yy_ids'],
                    x1x2yx1xx2yy_len: rets_data['batch']['x1x2yx1xx2yy_len'],
                    tau: config_train.tau,
                    tx.global_mode(): tf.estimator.ModeKeys.TRAIN,
                }

                if initial:
                    fetches_initial = {
                        'x1xx2yy_ids': x1xx2yy_ids,
                        'x1xx2yy_len': x1xx2yy_len
                    }
                    fetches_initial.update(loss_dict)
                    rets_initial = sess.run(fetches_initial, feed_dict)
                    if _is_head():
                        _log_losses(rets_initial, 0)
                    initial = False

                    for t in rets_initial['x1xx2yy_ids']:
                        t_text = proc.decode(t)
                        print(t_text)

                fetches = {
                    'train_op': train_op,
                    'step': global_step,
                }
                fetches.update(loss_dict)

                rets = sess.run(fetches, feed_dict)
                step = rets['step']

                dis_steps = config_train.display_steps

                if _is_head() and dis_steps > 0 and step % dis_steps == 0:
                    _log_losses(rets, step)

                eval_steps = config_train.eval_steps
                if _is_head() and eval_steps > 0 and step % eval_steps == 0:
                    _dev_epoch(sess)
                sample_steps = config_train.sample_steps
                if _is_head() and sample_steps > 0 and step % sample_steps == 0:
                    print('-----------testing-----------------')
                    _test_epoch(sess, step=step)

                ckpt_steps = config_train.checkpoint_steps
                if _is_head() and ckpt_steps > 0 and step % ckpt_steps == 0:
                    ckpt_fn = os.path.join(output_dir, 'model.ckpt')
                    ckpt_fn = saver.save(sess, ckpt_fn, global_step=step)
                    _log('Checkpoint to {}'.format(ckpt_fn))

            except tf.errors.OutOfRangeError:
                break

    def _dev_epoch(sess):
        """Evaluates on the dev set.
        """
        iterator.restart_dataset(sess, 'dev')

        results = tx.utils.AverageRecorder()
        nsamples = 0
        fetches = {}
        fetches.update(loss_dict)
        # i = 0

        while True:
            try:

                # (1) Get data and yy sample
                fetches_data = {
                    'batch': batch,
                    'batch_size': batch_size,
                    #'yy_ids': yy_ids,
                    #'yy_len': yy_len
                }
                feed_dict_data = {
                    iterator.handle: iterator.get_handle(sess, 'dev'),
                    tx.global_mode(): tf.estimator.ModeKeys.PREDICT,
                }
                rets_data = sess.run(fetches_data, feed_dict_data)


                # (2) eval loss
                feed_dict = {
                    x1x2yx1my_ids: rets_data['batch']['x1x2yx1my_ids'],
                    x1x2yx1my_len: rets_data['batch']['x1x2yx1my_len'],
                    x1x2yx1m_len: rets_data['batch']['x1x2yx1m_len'],
                    x1x2yx1xx2_ids: rets_data['batch']['x1x2yx1xx2_ids'],
                    x1x2yx1xx2_len: rets_data['batch']['x1x2yx1xx2_len'],
                    x1_len: rets_data['batch']['x1_len'],
                    x1x2_ids: rets_data['batch']['x1x2_ids'],
                    x1x2_len: rets_data['batch']['x1x2_len'],
                    x1xx2_ids: rets_data['batch']['x1xx2_ids'],
                    x1xx2_len: rets_data['batch']['x1xx2_len'],
                    x1x2y_ids: rets_data['batch']['x1x2y_ids'],
                    x1x2y_len: rets_data['batch']['x1x2y_len'],
                    x1x2yx1xx2yy_ids: rets_data['batch']['x1x2yx1xx2yy_ids'],
                    x1x2yx1xx2yy_len: rets_data['batch']['x1x2yx1xx2yy_len'],
                    tau: config_train.tau,
                    tx.global_mode(): tf.estimator.ModeKeys.PREDICT,
                }

                rets = sess.run(fetches, feed_dict)

                results.add(rets, weight=rets_data['batch_size'])
                nsamples += rets_data['batch_size']
            except tf.errors.OutOfRangeError:
                break

        _log_losses(results.avg())
        _log('nsamples: %d' % nsamples)

        avg_loss = results.avg('loss')
        if FLAGS.do_train and avg_loss < dev_best['loss']:
            dev_best.update(results.avg())
            ckpt_fn = os.path.join(output_dir, 'model_best.ckpt')
            ckpt_fn = saver_best.save(sess, ckpt_fn)
            _log('Checkpoint best to {}'.format(ckpt_fn))


    def _test_epoch(sess, step=None):
        """Generates samples on the test set.
        """
        iterator.restart_dataset(sess, 'test')

        _all_inputs = []
        _all_samples = []

        if FLAGS.finetune and FLAGS.roc:
            raise ValueError('Cannot set --finetune and --roc at the same time')


        if FLAGS.finetune:
            _log('Generation input: x1xx2')
            fetches = {
                'inputs': batch['x1xx2_ids'],
                'length': batch['x1xx2_len'],
                'samples_length': yy_len_fine,
                'samples': yy_ids_fine
            }
            res_fn_appendix = "x1xx2"
        elif FLAGS.roc:
            _log('Generation input: x1x2')
            fetches = {
                'inputs': batch['x1x2_ids'],
                'length': batch['x1x2_len'],
                'samples_length': yy_len_roc,
                'samples': yy_ids_roc
            }
            res_fn_appendix = "x1x2"
        else:
            _log('Generation input: x1x2yx1xx2')
            fetches = {
                'inputs': batch['x1x2yx1xx2_ids'],
                'length': batch['x1x2yx1xx2_len'],
                'samples_length': yy_len,
                'samples': yy_ids
            }
            res_fn_appendix = "x1x2yx1xx2"


        while True:
            try:
                feed_dict = {
                    iterator.handle: iterator.get_handle(sess, 'test'),
                    tx.context.global_mode(): tf.estimator.ModeKeys.PREDICT,
                }
                rets = sess.run(fetches, feed_dict=feed_dict)

                _inputs = []
                for i, l in zip(rets['inputs'], rets['length']):
                    # Delete padding
                    _inputs.append(i[:l].tolist())
                _all_inputs.extend(_inputs)

                _samples = []
                for s, l in zip(rets['samples'], rets['samples_length']):
                    _samples.append(s[:l].tolist())
                _all_samples.extend(_samples)

            except tf.errors.OutOfRangeError:
                break

        # Parse samples and write to file

        eos_token_id = proc.encoder['<|endoftext|>']

        _all_input_text = []
        for i in _all_inputs:
            if i[0] == eos_token_id:
                i = i[1:]
            i_text = proc.decode(i)
            _all_input_text.append(i_text)
        _all_input_text = tx.utils.strip_eos(_all_input_text,
                                             eos_token='<|endoftext|>')

        _all_samples_text = []
        for i, s in zip(_all_inputs, _all_samples):
            s_text = proc.decode(s)
            s_text = s_text.replace('\n', ' ')
            _all_samples_text.append(s_text)

        if step is None:
            fn = "test_samples_%s.tsv" % res_fn_appendix
        else:
            fn = "test_samples_%s_%d.tsv" % (res_fn_appendix, step)
        output_file = os.path.join(output_dir, fn)
        _log('Write samples to {}'.format(output_file))
        tx.utils.write_paired_text(
            _all_input_text, _all_samples_text, output_file)


    # Broadcasts global variables from rank-0 process
    if FLAGS.distributed:
        bcast = hvd.broadcast_global_variables(0)

    session_config = tf.ConfigProto()
    if FLAGS.distributed:
        session_config.gpu_options.visible_device_list = str(hvd.local_rank())

    with tf.Session(config=session_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        smry_writer = tf.summary.FileWriter(FLAGS.output_dir, graph=sess.graph)

        if FLAGS.distributed:
            bcast.run()

        #Restores trained model if specified
        if FLAGS.checkpoint:
           _log('Restore from {}'.format(FLAGS.checkpoint))
           saver.restore(sess, FLAGS.checkpoint)
        elif FLAGS.pretrain_checkpoint:
           _log('Restore from {}'.format(FLAGS.pretrain_checkpoint))
           model_utils.init_gpt2_checkpoint(sess, FLAGS.pretrain_checkpoint)
           print("\nFinished loading\n")
           saver.save(sess, output_dir + '/gpt2_model.ckpt')


        iterator.initialize_dataset(sess)

        if FLAGS.do_train:
            for epoch in range(config_train.max_train_epoch):
                _train_epoch(sess, epoch==0)
            saver.save(sess, output_dir + '/model.ckpt')

        if FLAGS.do_eval:
           _dev_epoch(sess)

        if FLAGS.do_test:
            _test_epoch(sess)


if __name__ == "__main__":
    tf.app.run()


