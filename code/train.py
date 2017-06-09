from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import sys
import time
import re
import numpy as np
import argparse

import tensorflow as tf

from qa import Encoder, QASystem, Decoder
from os.path import join as pjoin
from util import Progbar

import logging

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 0.8, "Learning rate.")
tf.app.flags.DEFINE_float("evaluate", 100, "Evaluating sample numbers")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.8, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 16, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 75, "Size of each model layer.")
tf.app.flags.DEFINE_integer("output_size", 500, "The output size of your model.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("load_train_dir", "", "Training directory to load model parameters from to resume training (default: {train_dir}).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("optimizer", "adad", "adam / sgd / adadelta")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")

FLAGS = tf.app.flags.FLAGS


def initialize_model(session, saver, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        # logging.info(tf.trainable_variables())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def get_normalized_train_dir(train_dir):
    """
    Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint. This allows the model to be reloaded even
    if the location of the checkpoint files has moved, allowing usage with CodaLab.
    This must be done on both train.py and qa_answer.py in order to work.
    """
    global_train_dir = '/tmp/cs224n-squad-train'
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    os.symlink(os.path.abspath(train_dir), global_train_dir)
    return global_train_dir

def device_spec(n, i):
    if n.type in ['VariableV2']:
        return '/cpu:0'
    else:
        return '/gpu:'+str(i)

def load_file(set_id, vocab):
    dataset = None

    if tf.gfile.Exists(os.path.join(FLAGS.data_dir,set_id)+'.question'):
        logging.info("Files found! Extracting input sentences...in data set: "+set_id)
        start = time.time()
        questions = []
        contexts = []
        answer_span = []

        file_prefix = os.path.join(FLAGS.data_dir,set_id)
        with open(file_prefix+'.question') as fin:
            questions.extend(fin.readlines())
        questions = [re.split(' +|/',line.strip('\n')) for line in questions]
        questions = [[vocab[word] if vocab.get(word) is not None else 2 for word in line] for line in questions]

        with open(file_prefix+'.context') as fin:
            contexts.extend(fin.readlines())
        contexts = [re.split(' +|/',line.strip('\n'))[:FLAGS.output_size] for line in contexts]
        contexts = [[vocab[word] if vocab.get(word) is not None else 2 for word in line] for line in contexts]

        with open(file_prefix+'.span') as fin:
            answer_span.extend(fin.readlines())
        answer_span = [re.findall('\d+', line) for line in answer_span]
        answer_span = np.array([[int(l) for l in line] for line in answer_span])

        dataset = zip(questions,contexts,answer_span)
        logging.info('Took %.2f seconds', time.time() - start)

    return dataset

def minibatches(dataset, batch_size):
    batches = []
    data_size = len(dataset)
    for i in np.arange(0,data_size,batch_size):
        temp = dataset[i:i+batch_size]
        if len(temp) is not batch_size:
            temp.extend([([2,2],[0,2],[1,1])]*(batch_size-len(temp)))

        _1 = [k[0] for k in temp]
        _2 = [k[1] for k in temp]
        _4 = np.array([min(k[2][0],FLAGS.output_size-1) for k in temp])
        _5 = np.array([min(k[2][1],FLAGS.output_size-1) for k in temp])

        length1 = np.array([len(l) for l in _1])
        length2 = np.array([len(l) for l in _2])
        l1 = np.max([len(l) for l in _1])
        l2 = np.max([len(l) for l in _2])
        _6 = np.array([[.0]*l2]*len(temp))
        _7 = np.array([[.0]*l2]*len(temp))
        _6[np.arange(len(temp)),_4] += 1.
        _7[np.arange(len(temp)),_5] += 1.

        '''
        l_1 = np.array([[False]*l1]*len(temp))
        l_2 = np.array([[False]*l2]*len(temp))
        l_1[np.arange(len(temp)),[len(k)-1 for k in _1]] |= True
        l_2[np.arange(len(temp)),[len(k)-1 for k in _2]] |= True
        '''

        mask1 = np.concatenate([np.append([1.]*len(w),[0.]*(l1-len(w))) for w in _1]).reshape(-1,l1)
        mask2 = np.concatenate([np.append([1.]*len(w),[0.]*(l2-len(w))) for w in _2]).reshape(-1,l2)
        _1 = np.concatenate([np.append(w,[0]*l1)[:l1] for w in _1]).reshape(-1,l1) # also pad at the end
        _2 = np.concatenate([np.append(w,[0]*l2)[:l2] for w in _2]).reshape(-1,l2)

        batches.append({'q':_1,             'qm':mask1,
                        'c':_2,             'cm':mask2, # questions, contexts and their masks
                        'answer_start_m'    :_6, # span start mask
                        'answer_end_m'      :_7, # span end mask
                        'answer_start_i'    :_4, # span start index
                        'answer_end_i'      :_5, # span end index
                        # 'q_end_m'           :l_1, # question end mask
                        # 'c_end_m'           :l_2, # context end mask
                        'ql'                :length1, # question lengths
                        'cl'                :length2}) # context lengths

    return batches


def main(_):


    parser = argparse.ArgumentParser(description='Train and test on a small set for the SQuAD problem.')
    subparser = parser.add_subparsers()

    command_parser = subparser.add_parser('train')
    command_parser.add_argument('-g','--gpus',type=int, default=1, help='the number of gpus to use')
    command_parser.set_defaults(foo = 'train')
    command_parser.set_defaults(path=FLAGS.train_dir)

    command_parser = subparser.add_parser('load')
    command_parser.add_argument('-g','--gpus',type=int, default=1, help='the number of gpus to use')
    command_parser.add_argument('-p','--path',type=str, default=FLAGS.train_dir ,help='the directory to load parameters from, default ./train')
    command_parser.set_defaults(foo='load and train')

    ARGS = parser.parse_args()
    if ARGS.foo is None:
        parser.print_help()
        sys.exit(1)

    if ARGS.path is not None:
        FLAGS.load_train_dir = ARGS.path

    if ARGS.gpus < 1:
        print('GPU counts should be at least 1, please retry with other configurations.')
        sys.exit(1)
    else:
        print('Going on with %d GPU(s), please make sure you have them installed.' % ARGS.gpus)

    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)

    train_set = load_file('train', vocab)
    dev_set = load_file('dev', vocab)

    embed = np.load(embed_path)['glove'].astype(np.float32)
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=FLAGS.learning_rate,epsilon=1e-6)

    encoder = Encoder(size=FLAGS.state_size, vocab_dim=FLAGS.embedding_size)
    decoder = Decoder(output_size=FLAGS.output_size)

    qa = []
    with tf.name_scope('systems'):
        for i in range(ARGS.gpus):
            with tf.device(lambda x:device_spec(x,i)): # the device is explicitly specified by a function or string
                qa.append(QASystem(encoder, decoder, embed, optimizer=optimizer))
                tf.get_variable_scope().reuse_variables()

        with tf.device('/cpu:0'):
            losses = 0.
            for i in range(ARGS.gpus):
                losses += qa[i].setup_loss()
            losses /= ARGS.gpus

        saver = tf.train.Saver()

    def apply_grads():
        return [k.train_op for k in qa]

    print('##======================assemble model created======================##')

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    # print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    config = tf.ConfigProto(allow_soft_placement=False,log_device_placement=False)
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.visible_device_list='0,2,4,6'

    with tf.Session(config=config) as sess:
        load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)
        initialize_model(sess, saver, load_train_dir)

        save_train_dir = get_normalized_train_dir(FLAGS.train_dir)

        batches = minibatches(train_set, FLAGS.batch_size)
        dev_batches = minibatches(dev_set, FLAGS.batch_size)

        for _ in range(FLAGS.epochs):
            logging.info("Epoch %d out of %d", _+1, FLAGS.epochs)
            prog = Progbar(target=len(np.arange(0,len(batches),ARGS.gpus)))

            for i in np.arange(0,len(batches),ARGS.gpus):
                feeds = {}
                counts = len(batches[i:i+ARGS.gpus])
                if counts != ARGS.gpus:
                    break

                for j, b in enumerate(batches[i:i+ARGS.gpus]):
                    feeds.update(qa[j].set_dict(b,FLAGS.dropout,counts,True))
                _, loss_temp = sess.run([apply_grads(),losses],feed_dict=feeds)

                prog.update(i/ARGS.gpus + 1, [("train loss", loss_temp)])

            try:
                qa[0].evaluate_answer(sess,dev_batches,20,True) # a rough estimate over the dev set after each epoch
            except:
                logging.info('An error occured when validating...')

            logging.info("Saving current parameters...")
            saver.save(sess, os.path.join(save_train_dir, 'model.weights'))

                #qa.evaluate_answer(sess, train_set, dev_set, FLAGS.evaluate, log=True)

if __name__ == "__main__":
    tf.app.run()
