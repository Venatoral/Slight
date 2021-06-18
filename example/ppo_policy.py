import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import copy


latent_dim = 10

EP_MAX = 1000
EP_LEN = 200
GAMMA = 0.8
A_LR = 0.0005
C_LR = 0.0005
BATCH = 50
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    # Clipped surrogate objective, find this is better
    dict(name='clip', epsilon=0.2),
][1]        # choose the method for optimization


###################Two cycle layer RNN###########################
# PPO3 has two layers of RNN neural network.
# First layer doesn't output actions and we record the last step's hidden state
# as the second cycle layer's first step's hidden input state.(Ensuring that all actions
#  decided at each timestep are depanded on all signals' states.)
class PPO3(object):
    # PPO2在PPO上自定义了actor的RNN网络结构，使能够让前一step的输出作为后一step的输入
    # In this class, the only verification is to rewrite the RNN neural network.
    # The input states of RNN are different too. (For each step of RNN, input states are states of signal and the signal's chosen action.)

    def __init__(self, s_dim=32, a_dim=1, i_dim=1, name="meme", combine_action=1):
        runner1 = '/cpu:0'
        runner2 = '/gpu:0'
        with tf.device('/cpu:0'):
            self.sess = tf.Session()
            # self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
            self.a_dim = a_dim
            self.s_dim = s_dim
            self.i_dim = i_dim
            self.name = name
            self.buffer_a = []
            self.buffer_s = []
            self.buffer_r = []
            self.global_steps = 0
            self.update_steps_a = 0
            self.update_steps_c = 0
            self.global_counter = 0
            self.pre_counter = 0

            self.hidden_net = 64
            self.output_net = 64
            self.combine_action = combine_action

            self.tfa = tf.placeholder(tf.int32, [None], 'action')
            self.tfadv = tf.placeholder(tf.float32, [None, int(
                self.a_dim/self.combine_action)], 'advantage')
            self.tfs = tf.placeholder(tf.float32, [None, int(
                s_dim * self.combine_action/a_dim)], 'actor_state')
            # critic
            with tf.variable_scope(self.name + '_critic'):
                l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, bias_initializer=tf.constant_initializer(0.01),
                                     kernel_initializer=tf.random_normal_initializer(0., .01))
                self.v = tf.layers.dense(l1, 1)
                self.tfdc_r = tf.placeholder(
                    tf.float32, [None, 1], 'discounted_r')
                self.advantage = self.tfdc_r - self.v
                self.closs = tf.reduce_mean(tf.square(self.advantage))

                self.ctrain_op = tf.train.AdamOptimizer(
                    C_LR).minimize(self.closs)

            # actor (RNN)
            # self.pi, pi_params = self._build_anet(
            #     self.name + '_pi', trainable=True)
            # self.oldpi, oldpi_params = self._build_anet(
            #     self.name + '_oldpi', trainable=False)
            # actor (Seq2Seq)
            self.pi, pi_params = self.build_seq2seq(
                self.name + '_pi', 64, trainable=True)
            self.oldpi, oldpi_params = self.build_seq2seq(
                self.name + '_oldpi', 64, trainable=False)
            self.update_oldpi_op = [oldp.assign(
                p) for p, oldp in zip(pi_params, oldpi_params)]

            # 调整概率分布的维度，方便获取概率
            index = []
            self.pi_resize = tf.reshape(self.pi, [-1, 2])
            self.oldpi_resize = tf.reshape(self.oldpi, [-1, 2])

            self.a_indices = tf.stack(
                [
                    tf.range(
                        tf.shape(tf.reshape(self.tfa, [-1]))[0],
                        dtype=tf.int32
                    ),
                    tf.reshape(self.tfa, [-1])
                ], axis=1)
            pi_prob = tf.gather_nd(params=self.pi_resize,
                                   indices=self.a_indices)
            oldpi_prob = tf.gather_nd(
                params=self.oldpi_resize, indices=self.a_indices)
            self.ratio_temp1 = tf.reshape(tf.reduce_mean(tf.reshape(pi_prob / (oldpi_prob + 1e-8), [-1, self.combine_action]), axis=1),
                                          [-1, int(self.a_dim/self.combine_action)])
            self.surr = self.ratio_temp1 * self.tfadv  # surrogate loss

            self.aloss = -tf.reduce_mean(tf.minimum(  # clipped surrogate objective
                self.surr,
                tf.clip_by_value(self.ratio_temp1, 1. - 0.2, 1. + 0.2) * self.tfadv))

            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

            # 以下为分开计算actor loss的部分

            self.aloss_seperated = -tf.reduce_mean(tf.reshape(tf.minimum(  # clipped surrogate objective
                self.surr,
                tf.clip_by_value(self.ratio_temp1, 1. - 0.2, 1. + 0.2) * self.tfadv), [-1, self.a_dim]), axis=0)
            # self.atrain_op_seperated = [tf.train.AdamOptimizer(A_LR).minimize(self.aloss_seperated[k]) for k in range(self.a_dim)]
            self.atrain_op_seperated = [tf.train.AdamOptimizer(
                A_LR).minimize(self.aloss_seperated[k]) for k in range(1)]

            self.sess.run(tf.global_variables_initializer())
            self.writer = tf.summary.FileWriter(
                "baseline/PPO3/" + self.name + "_log/", self.sess.graph)
            self.saver = tf.train.Saver(max_to_keep=5)

    def update_critic(self):
        s = np.vstack(self.buffer_s)
        r = np.vstack(self.buffer_r)
        critic_loss = self.sess.run(self.closs, {self.tfs: s, self.tfdc_r: r})
        self.summarize(critic_loss, self.pre_counter, 'pre_Critic_loss')
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r})
         for _ in range(C_UPDATE_STEPS)]
        self.pre_counter += 1

    def update(self):
        print("Update")
        s = np.vstack(self.buffer_s)
        c_s = s.reshape(
            [-1, int(self.s_dim * self.combine_action / self.a_dim)])
        r = np.vstack(self.buffer_r)
        a = np.array(self.buffer_a).reshape([-1])
        # print(s.shape)
        # print(a.shape)

        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: c_s, self.tfdc_r: r})

        # Calculating advantages one
        adv_r = np.array(adv).reshape(
            [-1, int(self.a_dim/self.combine_action)])

        # ##Calculating advantages two
        # adv_mean, adv_std = mpi_statistics_scalar(adv)
        # adv_ori = (adv - adv_mean) / adv_std
        # adv_r = np.array(adv_ori).reshape([-1,int(self.a_dim/self.combine_action)])

        # tem = self.sess.run(self.pi,{self.tfs:s})
        # print(np.array(tem).shape)
        # tem2 = self.sess.run(self.pi_resize,{self.tfs:s})
        # print(np.array(tem2).shape)
        # old_pi = self.sess.run(self.oldpi_resize,{self.tfs:s})
        # print(np.array(old_pi).shape)
        # a_in = self.sess.run(self.a_indices,{self.tfa:a})
        # print(np.array(a_in).shape,a_in[6])

        actor_loss = self.sess.run(
            self.aloss, {self.tfs: c_s, self.tfa: a, self.tfadv: adv_r})
        self.summarize(actor_loss, self.global_counter, 'Actor_loss')

        [self.sess.run(self.atrain_op, {
                       self.tfs: c_s, self.tfa: a, self.tfadv: adv_r}) for _ in range(A_UPDATE_STEPS)]
        # [self.sess.run(self.atrain_op_seperated, {self.tfs: c_s, self.tfa: a, self.tfadv: adv_r}) for _ in range(A_UPDATE_STEPS)]

        critic_loss = self.sess.run(
            self.closs, {self.tfs: c_s, self.tfdc_r: r})
        self.summarize(critic_loss, self.global_counter, 'Critic_loss')
        [self.sess.run(self.ctrain_op, {self.tfs: c_s, self.tfdc_r: r}) for _ in range(
            C_UPDATE_STEPS)]

        self.global_counter += 1

    def rnn_cell(self, rnn_input, state, name, trainable, last_prob):
        #Yt = relu(St*Vw+Vb)
        #St = tanch(Xt*Uw + Ub + St-1*Ww+Wb)
        # Xt = [none,198 + 2] St-1 = [none,64] Yt = [none,64]
        # Uw = [198 + 2,64] Ub = [64]
        # Ww = [64,64]   Wb = [64]
        # Vw = [64,64]      Vb = [64]
        with tf.variable_scope('rnn_input_cell_' + name, reuse=True):
            Uw = tf.get_variable(
                'Uw', [int(self.s_dim/self.a_dim) + 2, self.hidden_net], trainable=trainable)
            Ub = tf.get_variable('Ub', [1, self.hidden_net], initializer=tf.constant_initializer(
                0.0), trainable=trainable)
        with tf.variable_scope('rnn_cycle_cell_' + name,  reuse=True):
            Ww = tf.get_variable(
                'Ww', [self.hidden_net, self.hidden_net], trainable=trainable)
            Wb = tf.get_variable('Wb', [1, self.hidden_net], initializer=tf.constant_initializer(
                0.0), trainable=trainable)
        with tf.variable_scope('rnn_output_cell_' + name, reuse=True):
            Vw = tf.get_variable(
                'Vw', [self.hidden_net, self.output_net], trainable=trainable)
            Vb = tf.get_variable('Vb', [1, self.output_net], initializer=tf.constant_initializer(
                0.0), trainable=trainable)
        if last_prob == None:
            St = tf.nn.tanh(
                tf.matmul(
                    tf.cast(
                        tf.reshape(
                            tf.pad(rnn_input, [[0, 0], [0, 2]]), [-1, int(self.s_dim/self.a_dim) + 2]),
                        tf.float32
                    ),
                    tf.cast(Uw, tf.float32)
                ) + tf.cast(Ub, tf.float32)
                + tf.matmul(tf.cast(state, tf.float32),
                            tf.cast(Ww, tf.float32))
                + tf.cast(Wb, tf.float32))
        else:
            St = tf.nn.tanh(
                tf.matmul(
                    tf.cast(tf.concat(
                        [tf.reshape(rnn_input, [-1, int(self.s_dim/self.a_dim)]), last_prob], axis=1
                    ),
                        tf.float32),
                    tf.cast(Uw, tf.float32)
                ) + tf.cast(Ub, tf.float32)
                + tf.matmul(tf.cast(state, tf.float32),
                            tf.cast(Ww, tf.float32))
                + tf.cast(Wb, tf.float32)
            )

        # St = tf.nn.tanh(tf.matmul(tf.cast(tf.reshape(tf.concat([rnn_input,last_prob],1),[-1,int(self.s_dim/self.a_dim)]),tf.float32),tf.cast(Uw,tf.float32)) + tf.cast(Ub,tf.float32) + tf.matmul(tf.cast(state,tf.float32),tf.cast(Ww,tf.float32)) + tf.cast(Wb,tf.float32))
        Yt = tf.nn.relu(
            tf.matmul(
                tf.cast(St, tf.float32),
                tf.cast(Vw, tf.float32)
            )
            + tf.cast(Vb, tf.float32)
        )
        # return
        return St, Yt

    def _build_anet_FCN(self, name, trainable):
        with tf.variable_scope(name):
            input = tf.reshape(self.tfs, [-1, int(self.s_dim/self.a_dim)])
            l1 = tf.layers.dense(input, 64, tf.nn.relu, bias_initializer=tf.constant_initializer(0.01),
                                 kernel_initializer=tf.random_normal_initializer(0., .01), trainable=trainable)
            # l2 = tf.layers.dense(l1, 100, tf.nn.relu,bias_initializer = tf.constant_initializer(0.01),
            #     kernel_initializer = tf.random_normal_initializer(0., .01),trainable=trainable)
            out = tf.layers.dense(l1, 2, tf.nn.softmax, trainable=trainable)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return out, params

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):

            with tf.variable_scope('rnn_input_cell_' + name):
                Uw = tf.get_variable(
                    'Uw', [int(self.s_dim/self.a_dim) + 2, self.hidden_net], trainable=trainable)
                Ub = tf.get_variable('Ub', [1, self.hidden_net], initializer=tf.constant_initializer(
                    0.0), trainable=trainable)
            with tf.variable_scope('rnn_cycle_cell_' + name):
                Ww = tf.get_variable(
                    'Ww', [self.hidden_net, self.hidden_net], trainable=trainable)
                Wb = tf.get_variable('Wb', [1, self.hidden_net], initializer=tf.constant_initializer(
                    0.0), trainable=trainable)
            with tf.variable_scope('rnn_output_cell_' + name):
                Vw = tf.get_variable(
                    'Vw', [self.hidden_net, self.output_net], trainable=trainable)
                Vb = tf.get_variable('Vb', [1, self.output_net], initializer=tf.constant_initializer(
                    0.0), trainable=trainable)

            # RNN
            out_temp1 = []
            out_temp2 = []
            out = []
            actions = []
            last_prob = None
            rnn_input = tf.reshape(
                self.tfs, [-1, self.a_dim, int(self.s_dim/self.a_dim)])
            state = np.zeros([1, self.hidden_net])
            # The first for cycle aims to get state include all signals' imformation
            # and pass to the second RNN layer (through variate "state")
            for j in range(self.a_dim):
                state, y = self.rnn_cell(
                    rnn_input[:, j, :], state, name, trainable, last_prob)
                out_temp1.append(
                    tf.layers.dense(y, 2, tf.nn.softmax, trainable=trainable,
                                    kernel_initializer=tf.random_normal_initializer(
                                        0., .01),
                                    bias_initializer=tf.constant_initializer(
                                        0.01)
                                    )
                )
                last_prob = out_temp1[j]
            # The second cycle is aim to make actions depend on last cycle's final state.
            last_prob = None
            for j in range(self.a_dim):
                state, y = self.rnn_cell(
                    rnn_input[:, j, :], state, name, trainable, last_prob)
                out_temp2.append(tf.layers.dense(y, 2, tf.nn.softmax, trainable=trainable,
                                                 kernel_initializer=tf.random_normal_initializer(
                                                     0., .01),
                                                 bias_initializer=tf.constant_initializer(0.01)))
                last_prob = out_temp2[j]
            out = tf.stack([out_temp2[k] for k in range(self.a_dim)], axis=1)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return out, params

    # seq2seq 网络建立
    def build_seq2seq(self, name, embedding_dim, trainable=False):
        with tf.variable_scope(name):
            seq_inputs = self.tfs
            seq_inputs_length = seq_inputs.shape[0]
            batch_size = self.s_dim / self.i_dim

            with tf.variable_scope("encoder"):
                # encoder_embedding = tf.Variable(tf.random_uniform(
                #     [self.s_dim, embedding_dim]), dtype=tf.float32, name='encoder_embedding')
                # encoder_inputs_embedded = tf.nn.embedding_lookup(
                #     encoder_embedding, seq_inputs)

                ((encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw=tf.nn.rnn_cell.GRUCell(self.hidden_net),
                        cell_bw=tf.nn.rnn_cell.GRUCell(self.hidden_net),
                        inputs=self.tfs,
                        sequence_length=seq_inputs_length,
                        dtype=tf.float32,
                        time_major=False
                )
                encoder_state = tf.add(
                    encoder_fw_final_state, encoder_bw_final_state)
                encoder_outputs = tf.add(
                    encoder_fw_outputs, encoder_bw_outputs)

            with tf.variable_scope("decoder"):

                decoder_embedding = tf.Variable(tf.random_uniform(
                    [self.output_net, embedding_dim]), dtype=tf.float32, name='decoder_embedding')

                with tf.variable_scope("gru_cell"):
                    decoder_cell = tf.nn.rnn_cell.GRUCell(self.hidden_net)
                    decoder_initial_state = encoder_state

                # if useTeacherForcing and not useAttention:
                    # decoder_inputs = tf.concat([tf.reshape(tokens_go,[-1,1]), self.seq_targets[:,:-1]], 1)
                    # decoder_inputs_embedded = tf.nn.embedding_lookup(decoder_embedding, decoder_inputs)
                    # decoder_outputs, decoder_state = tf.nn.dynamic_rnn(cell=decoder_cell, inputs=decoder_inputs_embedded, initial_state=decoder_initial_state, sequence_length=self.seq_targets_length, dtype=tf.float32, time_major=False)

                tokens_go = tf.ones(
                    [batch_size], dtype=tf.int32, name='tokens_GO') * [0]
                tokens_eos = tf.ones(
                    [batch_size], dtype=tf.int32, name='tokens_EOS') * [0]
                
                tokens_eos_embedded = tf.nn.embedding_lookup(
                    decoder_embedding, tokens_eos)
                tokens_go_embedded = tf.nn.embedding_lookup(
                    decoder_embedding, tokens_go)

                W = tf.Variable(tf.random_uniform(
                    [self.hidden_net, self.output_net]), dtype=tf.float32, name='decoder_out_W')
                b = tf.Variable(
                    tf.zeros([self.output_net]), dtype=tf.float32, name="decoder_out_b")

                def loop_fn(time, previous_output, previous_state, previous_loop_state):
                    if previous_state is None:    # time step == 0
                        # all False at the initial step
                        initial_elements_finished = (0 >= self.output_net)
                        initial_state = decoder_initial_state  # last time steps cell state
                        initial_input = tokens_go_embedded  # last time steps cell state
                        initial_output = None  # none
                        initial_loop_state = None  # we don't need to pass any additional information
                        return (initial_elements_finished, initial_input, initial_state, initial_output, initial_loop_state)
                    else:
                        def get_next_input():
                            output_logits = tf.add(
                                    tf.matmul(previous_output, W), b)
                            prediction = tf.argmax(output_logits, axis=1)
                            next_input = tf.nn.embedding_lookup(
                                decoder_embedding, prediction)
                            return next_input

                        elements_finished = (time >= self.seq_targets_length)
                        finished = tf.reduce_all(elements_finished)  # Computes the "logical and" 
                        input = tf.cond(
                            finished, lambda: tokens_eos_embedded, get_next_input)
                        state = previous_state
                        output = previous_output
                        loop_state = None
                        return (elements_finished, input, state, output, loop_state)

                decoder_outputs_ta, decoder_state, _ = tf.nn.raw_rnn(
                    decoder_cell, loop_fn)
                decoder_outputs = decoder_outputs_ta.stack()
                decoder_outputs = tf.transpose(decoder_outputs, perm=[1, 0, 2]) # S*B*D -> B*S*D

                decoder_batch_size, decoder_max_steps, decoder_dim = tf.unstack(
                    tf.shape(decoder_outputs))
                decoder_outputs_flat = tf.reshape(
                    decoder_outputs, (-1, self.hidden_net))
                decoder_logits_flat = tf.add(
                    tf.matmul(decoder_outputs_flat, W), b)
                decoder_logits = tf.reshape(
                    decoder_logits_flat, (decoder_batch_size, decoder_max_steps, self.output_net))

            out = tf.argmax(decoder_logits, 2)
            params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
            return out, params

    def choose_action(self, s):

        _s = np.array(s).reshape(
            [-1, int(self.s_dim * self.combine_action/self.a_dim)])
        action = []
        prob = self.sess.run(self.pi, feed_dict={self.tfs: _s})
        prob_temp = np.array(prob).reshape([-1, 2])

        for i in range(self.a_dim):
            action_temp = np.random.choice(
                range(prob_temp[i].shape[0]),
                p=prob_temp[i].ravel()
            )  # select action w.r.t the actions prob
            action.append(action_temp)

        # the next part we initial a seed of random number limited in (0,1]
        # when seed first less than 0.9(threshold) that choose action according to given probability.
        # but if seed less bigger than 0.9, then we choose action equally.
        # for i in range(self.a_dim):
        #     seed = np.random.rand()
        #     if seed < 0.9:
        #         action_temp = np.random.choice(range(prob_temp[i].shape[0]),
        #                             p=prob_temp[i].ravel())  # select action w.r.t the actions prob
        #         action.append(action_temp)
        #     else:
        #         seed = np.random.rand()
        #         if seed < 0.5:
        #             action.append(0)
        #         else:
        #             action.append(1)

        return action

    def get_state(self, s):
        s = s[np.newaxis, :]
        h = self.sess.run(self.l2, {self.tfs: s})[0]
        return h

    def get_v(self, s):
        _s = np.array(s)
        if _s.ndim < 2:
            s = _s[np.newaxis, :]
        # print(self.sess.run(self.v, {self.tfs: s}))
        return self.sess.run(self.v, {self.tfs: s})

    def experience_store(self, s, a, r):
        self.buffer_a.append(a)
        self.buffer_s.append(s)
        self.buffer_r.append(r)

    def empty_buffer(self):
        self.buffer_s, self.buffer_r, self.buffer_a = [], [], []

    def trajction_process_proximate(self):
        # This function aims to calculate proximate F(s) of each state s.
        v_s_ = np.mean(np.array(self.buffer_r).reshape(
            [-1, self.a_dim]), axis=0)
        # we assume that all of the following Rs are the mean value of simulated steps (200)
        # so, the following Rs are geometric progression.
        # Sn = a1 * (1 - GAMMA^^n) / (1 - GAMMA) proximate equals to a1/(1-GAMMA)
        # print(v_s_)
        v_s_ = v_s_ / (1 - GAMMA)
        # print(v_s_)
        buffer_r = np.array(self.buffer_r).reshape([-1, self.a_dim])
        buffer = [[], [], [], [], [], [], [], [], [], [], [], []]
        for r in buffer_r[::-1]:
            for i in range(self.a_dim):
                v_s_[i] = r[i] + GAMMA * v_s_[i]
                buffer[i].append(copy.deepcopy(v_s_[i]))

        for i in range(self.a_dim):
            buffer[i].reverse()
        # print(np.array(buffer[0]))
        out = np.stack([buffer[k] for k in range(self.a_dim)], axis=1)

        self.buffer_r = np.array(out).reshape([-1])

    # 每一步的reward进行一个discount，让越远的reward影响变小

    def trajction_process(self, s_):
        _s = np.array(s_).reshape(
            [-1, int(self.s_dim * self.combine_action / self.a_dim)]).tolist()
        # for i in range(len(a_)):
        #     _s[i].append(a_[i])
        # v_s_ = [0,0,0,0,0,0]
        v_s_ = self.get_v(_s)
        # print(v_s_)
        buffer_tmp = np.mean(
            np.array(self.buffer_r).reshape([-1, self.combine_action]), axis=1
        )
        buffer_r = buffer_tmp.reshape(
            [-1, int(self.a_dim / self.combine_action)])
        buffer = [[], [], [], [], [], [], [], [], [], [], [], []]
        for r in buffer_r[::-1]:
            for i in range(int(self.a_dim / self.combine_action)):
                # print('v1:{}'.format(v_s_[i]))
                # print('r:{}'.format(r[i]))
                # print('v2:{}'.format(r[i] + GAMMA * v_s_[i]))
                v_s_[i] = (r[i] + GAMMA * v_s_[i])
                # print('v3:{}'.format(v_s_[i]))
                buffer[i].append(copy.deepcopy(v_s_[i]))
        for i in range(int(self.a_dim/self.combine_action)):
            buffer[i].reverse()

        # print(np.array(buffer[0]))
        out = np.stack([buffer[k] for k in range(
            int(self.a_dim/self.combine_action))], axis=1)
        # print(out)
        self.buffer_r = np.array(out).reshape([-1])

    def summarize(self, reward, i, tag):
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=reward)
        self.writer.add_summary(summary, i)
        self.writer.flush()

    def save_params(self, name, ep):
        save_path = self.saver.save(
            self.sess, 'my_net/rnn_discrete/{}_ep{}.ckpt'.format(name, ep))
        print("Save to path:", save_path)

    def restore_params(self, name, ep):
        self.saver.restore(
            self.sess, 'my_net/rnn_discrete/{}_ep{}.ckpt'.format(name, ep))
        print("Restore params from")
