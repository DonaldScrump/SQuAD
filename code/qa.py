from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import random
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from evaluate import exact_match_score, f1_score
from util import Progbar

logging.basicConfig(level=logging.INFO)

FLAGS = tf.app.flags.FLAGS


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    elif opt == "adad":
        optfn = tf.train.AdadeltaOptimizer
    else:
        assert (False)
    return optfn

def variable_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name=name,shape=shape,dtype=tf.float32,initializer=initializer)
    return var


class Encoder(object):
    def __init__(self, size, vocab_dim):
        self.size = size
        self.vocab_dim = vocab_dim

    def encode(self, inputs, masks, encoder_state_input):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """



        return


class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def decode(self, knowledge_rep):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """

        return

class QASystem(object):
    def __init__(self, encoder, decoder, embed, optimizer=None):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """

        # ==== set up placeholder tokens ========
        self.embed = embed
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.denominator = tf.placeholder(tf.float32,(),'denominator')

        self.preds1 = None
        self.preds2 = None
        self.loss = None

        with tf.device('/cpu:0'):
            self.train_op = None

        self.RawQuestion = None
        self.RawContext = None
        self.weighted_q = None
        self.weighted_r = None
        self.concat = None
        self.corrected = None
        self.grads = None

        self.question_ph        = tf.placeholder(tf.int32,(None,None),'question_ph')
        self.question_length    = tf.placeholder(tf.int32,(None),'question_length')

        self.context_ph         = tf.placeholder(tf.int32,(None,None),'context_ph')
        self.context_length     = tf.placeholder(tf.int32,(None),'context_length')

        self.answer_start_ph    = tf.placeholder(tf.float32,(None,None),'answer_start_ph')
        self.answer_end_ph      = tf.placeholder(tf.float32,(None,None),'answer_end_ph')

        # self.question_end_indicator     = tf.placeholder(tf.bool,(None,None),'question_end_indicator')
        # self.context_end_indicator      = tf.placeholder(tf.bool,(None,None),'context_end_indicator')
        self.question_mask_ph           = tf.placeholder(tf.float32,(None,None),'question_mask_ph') # the mask function sucks for real
        self.context_mask_ph            = tf.placeholder(tf.float32,(None,None),'context_mask_ph') # thus using a float mask as 1. or 0.
        self.dropout_ph                 = tf.placeholder(tf.float32,(),'dropout_ph')

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.contrib.layers.xavier_initializer()):
            #self.setup_embeddings() #used later in system_setup to get actual embeddings
            self.preds1, self.preds2 = self.setup_system()
            self.loss = self.setup_loss()
            self.setup_train_op() # only setup when training

        # ==== set up training/updating procedure ====
        # pass


    def set_dict(self, batch, dropout, deno, debug = True):
        '''
        to set up the dictionary using the minibatch     
        :param batch: 
        :return: 
        '''

        feed_dict = {self.question_ph       :batch['q'],
                     self.context_ph        :batch['c'],
                     self.question_mask_ph  :batch['qm'],
                     self.context_mask_ph   :batch['cm'],
                     self.dropout_ph        :dropout}

        if debug == True:
            feed_dict[self.answer_start_ph] = batch.get('answer_start_m')
            feed_dict[self.answer_end_ph] = batch.get('answer_end_m')

        # feed_dict[self.question_end_indicator] = batch['q_end_m']
        # feed_dict[self.context_end_indicator] = batch['c_end_m']
        feed_dict[self.question_length] = batch['ql']
        feed_dict[self.context_length] = batch['cl']
        feed_dict[self.denominator] = deno

        return feed_dict


    class ThisCell(tf.contrib.rnn.core_rnn_cell.RNNCell):
        """Customized internal cells for QP attentions
        """

        def __init__(self, state_size, model):
            self._state_size = state_size
            self.model = model
            self.inner_cell = tf.contrib.rnn.GRUBlockCell(cell_size=state_size)

        @property
        def state_size(self):
            return self._state_size

        @property
        def output_size(self):
            return self._state_size

        def __call__(self, inputs, state, scope=None):
            """
            Updates the state using the previous @state and @inputs.
            """
            scope = scope or type(self).__name__

            rawq = self.model.RawQuestion
            hidden_size = FLAGS.state_size

            with tf.variable_scope(scope):
                W_p = variable_cpu('W_p',(hidden_size*2,hidden_size),tf.contrib.layers.xavier_initializer())
                W_r = variable_cpu('W_r',(hidden_size,hidden_size),tf.contrib.layers.xavier_initializer())
                w_softmax = variable_cpu('w_softmax',(hidden_size,),tf.contrib.layers.xavier_initializer())

                biase = variable_cpu('biase',(hidden_size),tf.zeros_initializer())
                b = variable_cpu('b',(1,),tf.zeros_initializer())

                rbq = tf.expand_dims(tf.matmul(inputs,W_p)+tf.matmul(state,W_r)+biase,axis=1) # (b,1,h)
                Q = tf.tanh(self.model.weighted_q+rbq) # (b,t,h)
                alpha = tf.nn.softmax(tf.reduce_sum(Q*w_softmax,axis=2)+b) # (b,t)
                alpha = alpha*self.model.question_mask_ph
                alpha = alpha/tf.expand_dims(tf.reduce_sum(alpha,axis=1),axis=1) # make the softmax legit again...

                new_input = tf.concat([inputs,tf.reduce_sum(rawq*tf.expand_dims(alpha,axis=2),axis=1)],axis=1) # (b,4h)
                W_final = variable_cpu('W_final',(hidden_size*4,hidden_size*4),tf.contrib.layers.xavier_initializer())
                final_input = tf.sigmoid(tf.matmul(new_input,W_final)) # (b,4h)
                final_input = final_input*new_input

                out, _ = self.inner_cell(x=final_input,h_prev=state)

            return out, out


    class ThatCell(tf.contrib.rnn.core_rnn_cell.RNNCell):
        """Customized internal cells for PP attentions
        """

        def __init__(self, state_size, model):
            self._state_size = state_size
            self.model = model
            self.inner_cell = tf.contrib.rnn.GRUBlockCell(cell_size=state_size)

        @property
        def state_size(self):
            return self._state_size

        @property
        def output_size(self):
            return self._state_size

        def __call__(self, inputs, state, scope=None):
            """
            Updates the state using the previous @state and @inputs.
            """
            scope = scope or type(self).__name__

            rawr = self.model.concat
            hidden_size = FLAGS.state_size

            with tf.variable_scope(scope):
                W_p = variable_cpu('W_p',(hidden_size*2,hidden_size),tf.contrib.layers.xavier_initializer())
                w_softmax = variable_cpu('w_softmax',(hidden_size,),tf.contrib.layers.xavier_initializer())

                biase = variable_cpu('biase',(hidden_size),tf.zeros_initializer())
                b = variable_cpu('b',(1,),tf.zeros_initializer())

                rbq = tf.expand_dims(tf.matmul(inputs,W_p)+biase,axis=1) # (b,1,h)
                Q = tf.tanh(self.model.weighted_r+rbq) # (b,t,h)
                alpha = tf.nn.softmax(tf.reduce_sum(Q*w_softmax,axis=2)+b) # (b,t)
                alpha = alpha*self.model.context_mask_ph
                alpha = alpha/tf.expand_dims(tf.reduce_sum(alpha,axis=1),axis=1) # make the softmax legit again...

                new_input = tf.concat([inputs,tf.reduce_sum(rawr*tf.expand_dims(alpha,axis=2),axis=1)],axis=1) # (b,4h)
                W_final = variable_cpu('W_final',(1,hidden_size*4,hidden_size*4),tf.contrib.layers.xavier_initializer())
                final_input = tf.sigmoid(tf.reduce_sum(tf.expand_dims(new_input,axis=2)*W_final,axis=1)) # (b,4h)
                final_input = final_input*new_input

                out, _ = self.inner_cell(x=final_input,h_prev=state)
                # temp_gate = tf.get_variable('temp_gate',(hidden_size,hidden_size),tf.float32,initializer=tf.contrib.layers.xavier_initializer())

            return out, out


    def bidirectional_masked_dynamic_rnn_output(self, cell_fw, cell_bw, inputs, dtype, lengths, mask, scope=None):
        scope = scope or type(self).__name__

        with tf.variable_scope(scope):
            with tf.variable_scope('fw'):
                fw_out, __ = tf.nn.dynamic_rnn(cell = cell_fw, inputs = inputs, dtype = dtype)
            with tf.variable_scope('bw'):
                bw_out, __ = tf.nn.dynamic_rnn(cell = cell_bw, inputs = tf.reverse_sequence(inputs, lengths,
                            seq_axis=1, batch_axis=0), dtype = dtype)

            out = tf.concat([fw_out, tf.reverse_sequence(bw_out,lengths,seq_axis=1,batch_axis=0)],axis=2)*mask

        return out


    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """

        x,y = self.setup_embeddings()
        dropout_rate = self.dropout_ph
        hidden_size = FLAGS.state_size

        '''the padded question words are not taken into consideration'''

        cells = []
        for i in range(12):
            cells.append(tf.contrib.rnn.GRUBlockCell(cell_size=hidden_size))

        '''
        with tf.variable_scope('BiLSTM-question'):
            self.RawQuestion, _, __ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                    cells_fw = [cells[0],cells[1]], cells_bw=[cells[2],cells[3]], inputs=x,
                    dtype=tf.float32
            )

        with tf.variable_scope('BiLSTM-context'):
            self.RawContext, _, __  = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                    cells_fw = [cells[4],cells[5]], cells_bw=[cells[6],cells[7]], inputs=y,
                    dtype=tf.float32
            )
            
        ### stack Bi-RNN does not work over multiple gpus, currently with no gpu support for Assert ### 

        with tf.variable_scope('BiLSTM-question1'):
            RawQ1, _ = tf.nn.dynamic_rnn(
                cell=cells[0], inputs=x,
                dtype=tf.float32
            )
        with tf.variable_scope('BiLSTM-question2'):
            RawQ2, _ = tf.nn.dynamic_rnn(
                cell=cells[1], inputs=tf.reverse_sequence(x,self.question_length,seq_axis=1,batch_axis=0),
                dtype=tf.float32
            )

        with tf.variable_scope('BiLSTM-context1'):
            RawC1, _ = tf.nn.dynamic_rnn(
                cell=cells[4], inputs=y,
                dtype=tf.float32
            )
        with tf.variable_scope('BiLSTM-context2'):
            RawC2, _ = tf.nn.dynamic_rnn(
                cell=cells[5],inputs=tf.reverse_sequence(y,self.context_length,seq_axis=1,batch_axis=0),
                dtype=tf.float32
            )
        '''

        RawQ1 = self.bidirectional_masked_dynamic_rnn_output(cells[0],cells[1], x,
                                tf.float32, self.question_length, tf.expand_dims(self.question_mask_ph,2), 'Bi-RNN-Q1')
        RawC1 = self.bidirectional_masked_dynamic_rnn_output(cells[2],cells[3], y,
                                tf.float32, self.context_length, tf.expand_dims(self.context_mask_ph,2), 'Bi-RNN-C1')

        RawQ2 = self.bidirectional_masked_dynamic_rnn_output(cells[4],cells[5], tf.nn.dropout(RawQ1,dropout_rate),
                                tf.float32, self.question_length, tf.expand_dims(self.question_mask_ph,2), 'Bi-RNN-Q2')
        RawC2 = self.bidirectional_masked_dynamic_rnn_output(cells[6],cells[7], tf.nn.dropout(RawC1,dropout_rate),
                                tf.float32, self.context_length, tf.expand_dims(self.context_mask_ph,2), 'Bi-RNN-C2')

        RawQ3 = self.bidirectional_masked_dynamic_rnn_output(cells[8],cells[9], tf.nn.dropout(RawQ2,dropout_rate),
                                tf.float32, self.question_length, tf.expand_dims(self.question_mask_ph,2), 'Bi-RNN-Q3')
        RawC3 = self.bidirectional_masked_dynamic_rnn_output(cells[10],cells[11], tf.nn.dropout(RawC2,dropout_rate),
                                tf.float32, self.context_length, tf.expand_dims(self.context_mask_ph,2), 'Bi-RNN-C3')

        self.RawQuestion = RawQ3
        self.RawContext = RawC3

        #========================================================================#
                                    # For separation #
        #========================================================================#

        W_q = variable_cpu('W_q',(1,hidden_size*2,hidden_size),tf.contrib.layers.xavier_initializer())
        tile_w_q = tf.tile(W_q,(FLAGS.batch_size,1,1))
        self.weighted_q = tf.matmul(self.RawQuestion,tile_w_q)

        thiscell = []
        for i in range(2):
            thiscell.append(self.ThisCell(hidden_size,self))

        '''
        with tf.variable_scope("BiLSTM-QP_attention1"):
            concat1, _ = tf.nn.dynamic_rnn(
                cell = thiscell[0],inputs = self.RawContext,
                dtype=tf.float32
            )

        with tf.variable_scope("BiLSTM-QP_attention2"):
            concat2, _ = tf.nn.dynamic_rnn(
                cell = thiscell[1],inputs = tf.reverse_sequence(self.RawContext,self.context_length,
                                                                seq_axis=1,batch_axis=0),
                dtype=tf.float32
            )
        '''

        self.concat = self.bidirectional_masked_dynamic_rnn_output(thiscell[0],thiscell[1],self.RawContext,
                                tf.float32,self.context_length,tf.expand_dims(self.context_mask_ph,2),'Bi-RNN-QP')

        #========================================================================#
                                    # For separation #
        #========================================================================#


        W_r = variable_cpu('W_r',(1,hidden_size*2,hidden_size),tf.contrib.layers.xavier_initializer())
        tile_w_r = tf.tile(W_r,(FLAGS.batch_size,1,1))
        self.weighted_r = tf.matmul(self.concat,tile_w_r)

        thatcell = []
        for i in range(2):
            thatcell.append(self.ThatCell(hidden_size,self))

        '''
        with tf.variable_scope("BiLSTM-PP_attention1"):
            corrected1, _ = tf.nn.dynamic_rnn(
                cell = thatcell[0], inputs = self.concat,
                dtype=tf.float32
            )

        with tf.variable_scope("BiLSTM-PP_attention2"):
            corrected2, _ = tf.nn.dynamic_rnn(
                cell = thatcell[1], inputs = tf.reverse_sequence(self.concat,self.context_length,
                                                                seq_axis=1,batch_axis=0),
                dtype=tf.float32
            )

        self.corrected = tf.nn.dropout(tf.concat([corrected1,tf.reverse_sequence(corrected2,self.context_length,
                                                                seq_axis=1,batch_axis=0)],
                                                 axis=2),dropout_rate)*tf.expand_dims(self.context_mask_ph,axis=2)
        '''

        self.corrected = self.bidirectional_masked_dynamic_rnn_output(thatcell[0],thatcell[1],self.concat,
                                tf.float32,self.context_length,tf.expand_dims(self.context_mask_ph,2),'Bi-RNN-PP')

        #========================================================================#
                                     # For separation #
        #========================================================================#


        with tf.variable_scope('final_answer'):
            '''
            bi1 = tf.get_variable('bi1',(FLAGS.state_size*2,),tf.float32,initializer=tf.contrib.layers.xavier_initializer())
            bi2 = tf.get_variable('bi2',(FLAGS.state_size*2,),tf.float32,initializer=tf.contrib.layers.xavier_initializer())
            bi3 = tf.get_variable('bi3',(1,),tf.float32,initializer=tf.zeros_initializer())
            bi4 = tf.get_variable('bi4',(1,),tf.float32,initializer=tf.zeros_initializer())
            '''
            # more precise knowledge, less prior background guesses, no biases

            U_final = variable_cpu('U_final',(1,hidden_size*2,hidden_size*2),tf.contrib.layers.xavier_initializer())
            v1_sfm = variable_cpu('v1_sfm',(hidden_size*2,),tf.contrib.layers.xavier_initializer())
            v2_sfm = variable_cpu('v2_sfm',(hidden_size*2,),tf.contrib.layers.xavier_initializer())

            #comprehension = self.corrected
            comprehension = tf.matmul(self.corrected,tf.tile(U_final,(FLAGS.batch_size,1,1)))

            preds1 = tf.nn.softmax(tf.reduce_sum(tf.tanh(comprehension) * v1_sfm, axis=2))
            log1 = preds1*self.context_mask_ph
            log1 /= tf.expand_dims(tf.reduce_sum(log1,axis=1),axis=1) # (b,t)

            W_condition = variable_cpu('W_condition',(hidden_size*2,hidden_size*2),tf.contrib.layers.xavier_initializer())
            condition = tf.reduce_sum(self.corrected*tf.expand_dims(log1,axis=2),axis=1) # the conditions on the start of the answers, (b,2h)

            new_condition = tf.expand_dims(tf.matmul(condition,W_condition),axis=1) # (b,1,2h)
            preds2 = tf.nn.softmax(tf.reduce_sum(tf.tanh(comprehension + new_condition) * v2_sfm, axis=2))
            log2 = preds2*self.context_mask_ph
            log2 /= tf.expand_dims(tf.reduce_sum(log2,axis=1),axis=1) # (b,t)

        return log1, log2


    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """

        loss = -tf.reduce_sum(tf.log(tf.reduce_sum(self.preds1*self.answer_start_ph,axis=1)))\
               -tf.reduce_sum(tf.log(tf.reduce_sum(self.preds2*self.answer_end_ph,axis=1)))

        return loss


    def setup_train_op(self):
        # learning_rate = FLAGS.learning_rate

        '''
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate,global_step,500,0.5,staircase=True)
        '''

        # optn = get_optimizer(FLAGS.optimizer)(learning_rate=learning_rate,epsilon=1e-6)

        grads, _ = zip(*self.optimizer.compute_gradients(self.loss))
        grads, __ = tf.clip_by_global_norm(grads,FLAGS.max_gradient_norm)
        grads = [r/self.denominator for r in grads]

        with tf.device('/cpu:0'):
            self.train_op = self.optimizer.apply_gradients(zip(grads, _))


    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """

        self.embedding = tf.constant(self.embed,name='embedding')
        embedx = tf.nn.embedding_lookup(self.embedding,self.question_ph)
        embedy = tf.nn.embedding_lookup(self.embedding,self.context_ph)

        return embedx,embedy


    def optimize(self, session, train_x, train_y):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session, valid_x, valid_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, test_x):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x

        output_feed = []

        outputs = session.run(output_feed, feed_dict=input_feed)

        return outputs

    def answer(self, session, test_x):

        yp, yp2 = session.run([self.preds1, self.preds2], self.set_dict(batch=test_x, dropout=1., deno=1., debug = False))

        a1 = np.array(yp)
        a2 = np.array(yp2)

        a_s = np.argmax(a1, axis=1)
        a_e = np.argmax(a2, axis=1)

        return a_s, a_e

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = 0

        for valid_x, valid_y in valid_dataset:
          valid_cost = self.test(sess, valid_x, valid_y)


        return valid_cost

    def overlapping(self, a, b, c, d):
        return (a<=b)*(c<=d)*((b>=c)*(d>=a)*np.minimum(np.minimum(b-c,d-a),np.minimum(b-a,d-c)))

    def evaluate_answer(self, session, dev_set, sample_batches=10, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        f1 = 0.
        em = 0.
        overlap = 0.
        genuine = 0.
        predicted = 0.

        samples = random.sample(dev_set[:-2],sample_batches)
        for i, samp in enumerate(samples):
            a_s, a_e = self.answer(session, samp)
            em += np.sum(samp['answer_start_m'][np.arange(FLAGS.batch_size),a_s]*samp['answer_end_m'][np.arange(FLAGS.batch_size),a_e])
            genuine += np.sum(np.maximum(samp['answer_end_i'] - samp['answer_start_i']+1, 0))
            overlap += np.sum(self.overlapping(a_s,a_e,samp['answer_start_i'],samp['answer_end_i']))
            predicted += np.sum(np.maximum(a_e-a_s+1, 0))

        em/=(sample_batches*FLAGS.batch_size)
        Recall = overlap/genuine
        Precision = overlap/predicted

        if(Recall+Precision == 0):
            f1 = 0.
        else:
            f1 = 2*Recall*Precision/(Recall+Precision)

        if log:
            logging.info("F1: {:.4f},EM: {:.4f},Recall: {:.4f},Precision: {:.4f} for {} samples".format(f1, em, Recall, Precision, sample_batches*FLAGS.batch_size))

        return f1, em, Recall, Precision

    '''
    def minibatches(self, dataset, batch_size):
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
            _6 = np.array([[False]*l2]*len(temp))
            _7 = np.array([[False]*l2]*len(temp))
            _6[np.arange(len(temp)),_4] |= True
            _7[np.arange(len(temp)),_5] |= True
            l_1 = np.array([[False]*l1]*len(temp))
            l_2 = np.array([[False]*l2]*len(temp))
            l_1[np.arange(len(temp)),[len(k)-1 for k in _1]] |= True
            l_2[np.arange(len(temp)),[len(k)-1 for k in _2]] |= True

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
                            'q_end_m'           :l_1, # question end mask
                            'c_end_m'           :l_2, # context end mask
                            'ql'                :length1, # question lengths
                            'cl'                :length2}) # context lengths

        return batches

    
    def train_on_batch(self, session, batch):

        _, loss = session.run([self.train_op, self.loss], feed_dict = self.set_dict(batch=batch,dropout=FLAGS.dropout))
        return loss

    def run_epoch(self, session, batches, train_dir, saver):
        prog = Progbar(target=(len(batches)>>1))

        for i, b in enumerate(random.sample(batches,len(batches)>>1)):
            loss=self.train_on_batch(session, b)
            prog.update(i+1, [("train loss", loss)])

        logging.info("Saving current parameters...")
        saver.save(session, os.path.join(train_dir,'model.weights'))
    

    def train(self, session, train_set, dev_set, train_dir):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious approach can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

        #for i, (train_question, train_context, train_span) in enumerate(self.minibatches(dataset, FLAGS.batch_size)):
        
        tic = time.time()
        params = tf.trainable_variables()
        logging.info("Number of trainable variables: %d" % len(params))
        print (params)
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))
        

        batch_tic = time.time()
        batches = self.minibatches(train_set, FLAGS.batch_size) #these batches are already padded, but not using a universal length
        dev_batches = self.minibatches(dev_set, FLAGS.batch_size)
        logging.info("Batch split took %.2f seconds..." % (time.time()-batch_tic))

        for _ in range(FLAGS.epochs):
            logging.info("Epoch %d out of %d", _ + 1, FLAGS.epochs)
            self.evaluate_answer(session=session, dev_set=dev_batches, sample_batches=15, log=True)
            self.run_epoch(session, batches, train_dir, self.saver)

        self.evaluate_answer(session=session, dev_set=dev_batches, sample_batches=50, log=True) # Final evaluation

    '''