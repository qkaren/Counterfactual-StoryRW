
import os
import importlib
import tensorflow as tf
import texar as tx

from utils import model_utils


class Discriminator(object):

    def __init__(self, gpt2_config):
        vocab_size = gpt2_config.vocab_size

        self.word_embedder = tx.modules.WordEmbedder(
            vocab_size=vocab_size,
            hparams=gpt2_config.embed)

        self.pos_embedder = tx.modules.PositionEmbedder(
            position_size=gpt2_config.position_size,
            hparams=gpt2_config.pos_embed)

        # Ties output layer with input word embedding
        output_layer = tf.transpose(self.word_embedder.embedding, (1, 0))

        self.decoder = tx.modules.TransformerDecoder(
            vocab_size=vocab_size,
            output_layer=output_layer,
            hparams=gpt2_config.decoder)

    def init_model(self, sess, ckpt_path):
        tf.logging.info('Discriminator, restore from {}'.format(ckpt_path))
        model_utils.init_gpt2_checkpoint(sess, ckpt_path)
        print("\nFinished loading\n")

    def compute_loss(self, soft_ids, length):
        batch_size = tf.shape(soft_ids)[0]
        seq_len = tf.fill([batch_size], tf.shape(soft_ids)[1])
        pos_embeds = self.pos_embedder(sequence_length=seq_len)
        #pos_embed = tf.stop_gradient(pos_embeds)
        input_embeds = self.word_embedder(
            soft_ids=soft_ids, stop_gradient=False) + pos_embeds

        #return tf.reduce_sum(input_embeds)

        outputs = self.decoder(
            inputs=input_embeds, decoding_strategy='train_greedy')

        #return tf.reduce_sum(soft_ids[:,:,1000])
        #return tf.reduce_mean(outputs.logits)
        #return tf.reduce_mean(
        #    tf.nn.softmax_cross_entropy_with_logits_v2(
        #        labels=soft_ids[:, 1:],
        #        logits=tf.stop_gradient(outputs.logits[:, :-1, :]))
        #    )

        loss = tx.losses.sequence_softmax_cross_entropy(
            labels=soft_ids[:, 1:],
            logits=outputs.logits[:, :-1, :],
            sequence_length=length-1,
            average_across_timesteps=True,
            sum_over_timesteps=False,
            average_across_batch=True,
            sum_over_batch=False,
            stop_gradient_to_label=False) #TODO

        return loss
