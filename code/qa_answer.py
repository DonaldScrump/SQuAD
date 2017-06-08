from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import re
import time
import json
import sys
import random
from os.path import join as pjoin

from tqdm import tqdm
import numpy as np
from six.moves import xrange
import tensorflow as tf


from qa import Encoder, QASystem, Decoder
from preprocessing.squad_preprocess import data_from_json, maybe_download, squad_base_url, \
    invert_map, tokenize, token_idx_map
import qa_data

import logging

logging.basicConfig(level=logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float("learning_rate", 0.8, "Learning rate.")
tf.app.flags.DEFINE_float("evaluate", 100, "Evaluating sample numbers")
tf.app.flags.DEFINE_float("dropout", 0.8, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 16, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 0, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 75, "Size of each model layer.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_integer("output_size", 500, "The output size of your model.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("optimizer", "adad", "adam / sgd / adadelta")
tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory (default: ./train).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")
tf.app.flags.DEFINE_string("dev_path", "download/squad/dev-v1.1.json", "Path to the JSON dev set to evaluate against (default: ./data/squad/dev-v1.1.json)")

def initialize_model(session, saver, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
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


def dev_minibatches(dataset, batch_size):
    batches = []
    data_size = len(dataset)
    for i in np.arange(0,data_size,batch_size):
        temp = dataset[i:i+batch_size]
        if len(temp) is not batch_size:
            temp.extend([([2,2],[0,2],[1,1])]*(batch_size-len(temp)))

        _1 = [k[0] for k in temp]
        _2 = [k[1] for k in temp]
        # _4 = np.array([min(k[2][0],FLAGS.output_size-1) for k in temp])
        # _5 = np.array([min(k[2][1],FLAGS.output_size-1) for k in temp])

        length1 = np.array([len(l) for l in _1])
        length2 = np.array([len(l) for l in _2])
        l1 = np.max([len(l) for l in _1])
        l2 = np.max([len(l) for l in _2])
        # _6 = np.array([[.0]*l2]*len(temp))
        # _7 = np.array([[.0]*l2]*len(temp))
        # _6[np.arange(len(temp)),_4] += 1.
        # _7[np.arange(len(temp)),_5] += 1.

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
                        # 'answer_start_m'    :_6, # span start mask
                        # 'answer_end_m'      :_7, # span end mask
                        # 'answer_start_i'    :_4, # span start index
                        # 'answer_end_i'      :_5, # span end index
                        # 'q_end_m'           :l_1, # question end mask
                        # 'c_end_m'           :l_2, # context end mask
                        'ql'                :length1, # question lengths
                        'cl'                :length2}) # context lengths

    return batches


def read_dataset(dataset, tier, vocab):
    """Reads the dataset, extracts context, question, answer,
    and answer pointer in their own file. Returns the number
    of questions and answers processed for the dataset"""

    context_data = []
    query_data = []
    question_uuid_data = []

    for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):
        article_paragraphs = dataset['data'][articles_id]['paragraphs']
        for pid in range(len(article_paragraphs)):
            context = article_paragraphs[pid]['context']
            # The following replacements are suggested in the paper
            # BidAF (Seo et al., 2016)
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')

            context_tokens = tokenize(context)

            qas = article_paragraphs[pid]['qas']
            for qid in range(len(qas)):
                question = qas[qid]['question']
                question_tokens = tokenize(question)
                question_uuid = qas[qid]['id']

                context_ids = [str(vocab.get(w, qa_data.UNK_ID)) for w in context_tokens]
                qustion_ids = [str(vocab.get(w, qa_data.UNK_ID)) for w in question_tokens]

                context_data.append(' '.join(context_ids))
                query_data.append(' '.join(qustion_ids))
                question_uuid_data.append(question_uuid)

    return context_data, query_data, question_uuid_data


def prepare_dev(prefix, dev_filename, vocab):
    # Don't check file size, since we could be using other datasets
    # dev_dataset = maybe_download(squad_base_url, dev_filename, prefix)

    dev_data = data_from_json(os.path.join(prefix, dev_filename))
    context_data, question_data, question_uuid_data = read_dataset(dev_data, 'dev', vocab)

    return context_data, question_data, question_uuid_data


def generate_answers(sess, model, dev_batches, rev_vocab, uuids):
    """
    Loop over the dev or test dataset and generate answer.

    Note: output format must be answers[uuid] = "real answer"
    You must provide a string of words instead of just a list, or start and end index

    In main() function we are dumping onto a JSON file

    evaluate.py will take the output JSON along with the original JSON file
    and output a F1 and EM

    You must implement this function in order to submit to Leaderboard.

    :param sess: active TF session
    :param model: a built QASystem model
    :param rev_vocab: this is a list of vocabulary that maps index to actual words
    :return:
    """
    answers = {}
    for i, b in enumerate(dev_batches):

        a_s, a_e = model.answer(sess, b)
        if i%100 == 0:
            print('%d batches processed so far...'% i)

        if i != len(dev_batches)-1:
            for j in range(FLAGS.batch_size):
                passage = b['c'][j,:]
                if a_s[j]>a_e[j]:
                    answers[uuids[i*FLAGS.batch_size+j]] = rev_vocab[passage[a_s[j]]]
                else:
                    answers[uuids[i*FLAGS.batch_size+j]] = ' '.join([rev_vocab[passage[k]] for k in range(a_s[j], a_e[j] + 1)])
        else:
            for j in range(FLAGS.batch_size):
                passage = b['c'][j, :]
                if i*FLAGS.batch_size+j >= len(uuids): # the last batch is currently padded
                    break

                if a_s[j] > a_e[j]:
                    answers[uuids[i*FLAGS.batch_size+j]] = rev_vocab[passage[a_s[j]]]
                else:
                    answers[uuids[i*FLAGS.batch_size+j]] = ' '.join([rev_vocab[passage[k]] for k in range(a_s[j], a_e[j] + 1)])

    return answers


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


def main(_):

    vocab, rev_vocab = initialize_vocab(FLAGS.vocab_path)

    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    embed = np.load(embed_path)['glove'].astype(np.float32)

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    # ========= Load Dataset =========
    # You can change this code to load dataset in your own way
    # Use the uuids generated by the original file reader

    dev_dirname = os.path.dirname(os.path.abspath(FLAGS.dev_path))
    dev_filename = os.path.basename(FLAGS.dev_path)
    contexts, questions, uuids = prepare_dev(dev_dirname, dev_filename, vocab)

    contexts = [re.findall('\d+',sen1) for sen1 in contexts]
    contexts = np.array([[int(l) for l in j] for j in contexts])
    questions = [re.findall('\d+',sen2) for sen2 in questions]
    questions = np.array([[int(l) for l in j] for j in questions])

    dev_set = dev_minibatches(zip(questions, contexts), FLAGS.batch_size)


    # ========= Model-specific =========
    # You must change the following code to adjust to your model

    encoder = Encoder(size=FLAGS.state_size, vocab_dim=FLAGS.embedding_size)
    decoder = Decoder(output_size=FLAGS.output_size)

    qa = QASystem(encoder, decoder, embed, tf.train.AdadeltaOptimizer(learning_rate=FLAGS.learning_rate,epsilon=1e-6))
    saver = tf.train.Saver()


    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.visible_device_list='3,5'

    with tf.Session(config=config) as sess:
        train_dir = get_normalized_train_dir(FLAGS.train_dir)
        initialize_model(sess, saver, train_dir)
        answers = generate_answers(sess, qa, dev_set, rev_vocab, uuids)

        '''
        with open('dev-predictions.txt','w') as pres:
            for i in range(len(answers)):
                pres.write("%s\n" % answers[i])
        '''

        # write to json file to root dir
        with io.open('dev-prediction.json', 'w', encoding='utf-8') as f:
            f.write(unicode(json.dumps(answers, ensure_ascii=False)))


if __name__ == "__main__":
  tf.app.run()
