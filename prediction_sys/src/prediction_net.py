# @title
# !/usr/bin/env python3
import numpy as np
import tensorflow as tf
from src.environment import RenEnviroment


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Network:
    WIDTH = 640
    HEIGHT = WIDTH
    # x,y,z position, ro, pi, psi rotation
    DRONE_VALUES = 6

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                     intra_op_parallelism_threads=threads))

    def construct(self, args, action_lows, action_highs):
        with self.session.graph.as_default():
            # Inputs
            self.input = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 17], name="input")
            self.returns = tf.placeholder(tf.float32, [None, self.DRONE_VALUES])
            self.actions = tf.placeholder(tf.float32, [None, self.DRONE_VALUES], name="values")
            self.output_map = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 1], name="map")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            def conv_net(input):
                # Convolutional Layer #1
                # 320 x 320 x 32
                conv1 = tf.layers.conv2d(inputs=self.input, filters=32, kernel_size=[3, 3], strides=(2, 2),
                                         padding="same",
                                         activation=None, use_bias=False)
                batchNorm1 = tf.layers.batch_normalization(conv1, training=self.is_training)
                relu1 = tf.nn.relu(batchNorm1)

                # Convolutional Layer #2
                # 160 x 160 x 64
                conv2 = tf.layers.conv2d(inputs=relu1, filters=64, kernel_size=[3, 3], padding="same",
                                         activation=None, use_bias=False)
                batchNorm2 = tf.layers.batch_normalization(conv2, training=self.is_training)
                relu2 = tf.nn.relu(batchNorm2)
                pool2 = tf.layers.max_pooling2d(inputs=relu2, pool_size=[2, 2], strides=2)

                # Convolutional Layer #3
                # 40 x 40 x 256
                conv3 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=[3, 3], padding="same",
                                         activation=None, use_bias=False)
                batchNorm3 = tf.layers.batch_normalization(conv3, training=self.is_training)
                relu3 = tf.nn.relu(batchNorm3)
                pool3 = tf.layers.max_pooling2d(inputs=relu3, pool_size=[4, 4], strides=4)
                return pool3

            # Map
            def probability_map(inputs):
                # Convolutional transpose Layer #4
                # 160 x 160 x 16
                conv4 = tf.layers.conv2d_transpose(inputs=inputs, filters=32, kernel_size=[3, 3], strides=4,
                                                   padding="same",
                                                   activation=None, use_bias=False)
                batchNorm4 = tf.layers.batch_normalization(conv4, training=self.is_training)
                relu4 = tf.nn.relu(batchNorm4)

                # Convolutional transpose Layer #5
                # 320 x 320 x 4
                conv5 = tf.layers.conv2d_transpose(inputs=relu4, filters=8, kernel_size=[3, 3], strides=2,
                                                   padding="same",
                                                   activation=None, use_bias=False)
                batchNorm5 = tf.layers.batch_normalization(conv5, training=self.is_training)
                relu5 = tf.nn.relu(batchNorm5)

                # Convolutional transpose Layer #6
                # 640 x 640 x 1
                # map
                conv_layer = tf.layers.conv2d_transpose(inputs=relu5, filters=1, kernel_size=[3, 3], strides=2,
                                                        padding="same", activation=None)
                # reshaped for softmax function
                reshaped_maps = tf.reshape(conv_layer, [-1, self.WIDTH * self.HEIGHT])
                # back reshaping
                pr_maps = tf.reshape(tf.nn.softmax(reshaped_maps), [-1, self.HEIGHT, self.WIDTH, 1])
                return pr_maps, reshaped_maps

            # Actor
            def actor(inputs):
                # common cnn
                cnn_net = conv_net(inputs)
                # probability map
                pr_map, reshaped_maps = probability_map(cnn_net)
                # values for drone
                flattened_layer = tf.layers.flatten(cnn_net)
                actions_components = tf.layers.dense(flattened_layer, self.DRONE_VALUES, activation=None)
                return tf.add(tf.multiply(actions_components, tf.subtract(tf.cast(action_highs, dtype=tf.float32),
                                                                          tf.cast(action_lows, dtype=tf.float32))),
                              tf.cast(action_lows, dtype=tf.float32)), pr_map, reshaped_maps

            # Critic
            def critic(inputs, actions):
                # common cnn
                cnn_net = conv_net(inputs)
                # critic
                flattened_layer = tf.layers.flatten(cnn_net)
                hd = tf.concat([flattened_layer, actions], axis=1)
                hd = tf.layers.dense(hd, args.hidden_layer, activation=tf.nn.relu)
                # return tf.squeeze(tf.layers.dense(hd, self.DRONE_VALUES), 1)
                return tf.layers.dense(hd, self.DRONE_VALUES)

            with tf.variable_scope("actor"):
                self.mus, self.map, reshaped_maps = actor(self.input)

            with tf.variable_scope("target_actor"):
                target_actions, pr_map, reshaped_maps = actor(self.input)

            with tf.variable_scope("critic"):
                values_of_given = critic(self.input, self.actions)

            with tf.variable_scope("critic", reuse=True):
                values_of_predicted = critic(self.input, self.mus)

            with tf.variable_scope("target_critic"):
                self.target_values = critic(self.input, target_actions)

            global_step = tf.train.create_global_step()
            # true probability maps reshaped for softmax
            reshaped_output_maps = tf.reshape(self.output_map, [-1, self.WIDTH * self.HEIGHT])

            # Training

            # Update ops
            update_target_ops = []
            for target_var, var in zip(tf.global_variables("target_actor") + tf.global_variables("target_critic"),
                                       tf.global_variables("actor") + tf.global_variables("critic")):
                update_target_ops.append(target_var.assign((1. - args.target_tau) * target_var + args.target_tau * var))

            # Actor_loss and critic loss:
            loss_map = tf.nn.softmax_cross_entropy_with_logits(labels=reshaped_output_maps, logits=reshaped_maps)
            self.critic_loss = tf.losses.huber_loss(self.returns, values_of_given)
            self.actor_loss = -values_of_predicted + loss_map
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                critic_training = tf.train.AdamOptimizer(args.learning_rate).minimize(self.critic_loss,
                                                                                      global_step=global_step,
                                                                                      name="training_critic",
                                                                                      var_list=tf.get_collection(
                                                                                          tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                                          "critic"))
                actor_training = tf.train.AdamOptimizer(args.learning_rate).minimize(self.actor_loss,
                                                                                     global_step=global_step,
                                                                                     name="training_actor",
                                                                                     var_list=tf.get_collection(
                                                                                         tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                                         "actor"))
            self.training = tf.group(actor_training, critic_training, update_target_ops)
            self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "actor"))

            # Initialize variables
            self.session.run(tf.global_variables_initializer())

    def predict_actions(self, states):
        return self.session.run([self.mus, self.map], {self.input: states, self.is_training: False})

    def predict_values(self, states):
        return self.session.run(self.target_values, {self.input: states, self.is_training: False})

    def train_s(self, states, output_maps, actions, returns):
        return self.session.run([self.training],
                                {self.input: states, self.output_map: output_maps, self.actions: actions,
                                 self.returns: returns, self.is_training: True})


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, shape, mu, theta, sigma):
        self.mu = mu * np.ones(shape)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        self.state += self.theta * (self.mu - self.state) + np.random.normal(scale=self.sigma, size=self.state.shape)
        return self.state


if __name__ == "__main__":
    import collections

    # Fix random seed
    np.random.seed(42)

    arg = {}
    args = dotdict(arg)
    args.batch_size = 40
    args.epochs = 20
    args.threads = 5
    args.learning_rate = 0.001
    args.hidden_layer = 50
    args.target_tau = 0.001
    args.gamma = 1
    args.noise_theta = 0.15
    args.noise_sigma = 0.2

    # Create the environment + variables
    env = RenEnviroment()
    width = 640
    actions_lows = np.array([-width, -width, -width, -np.pi, -np.pi, -np.pi])
    actions_highs = np.array([width, width, width, np.pi, np.pi, np.pi])
    frame_history = 8

    # Replay memory; maxlen parameter can be passed to deque for a size limit,
    # which we however do not need in this simple task.
    replay_buffer = collections.deque()
    last_states = collections.deque(maxlen=frame_history)
    Transition = collections.namedtuple("Transition",
                                        ["state", "action", "probability_map", "reward", "done", "next_state"])
    noise = OrnsteinUhlenbeckNoise([6], 0., args.noise_theta, args.noise_sigma)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, actions_lows, actions_highs)

    while True:
        # Training
        state, done = env.reset(), False
        noise.reset()
        while not done:
            # return a prediction for the probability map of the location of the target and actions/robot state for the next state
            buffer_length = len(replay_buffer)
            s = env.transform_state([state])
            action_components, probability_map = network.predict([s])
            # add random noise to robot state prediction
            action_components += noise.sample()
            # get true values of position of target and drone
            next_state, reward, done, _ = env.step(actions)
            replay_buffer.append(Transition(state, action_components, probability_map, reward, done, next_state))
            state = next_state

            # If the replay_buffer is large enough, perform training
            if buffer_length >= args.batch_size:
                batch = np.random.choice(len(replay_buffer), size=args.batch_size, replace=False)
                states, actions, maps, rewards, dones, next_states = zip(*[replay_buffer[i] for i in batch])
                s = env.transform_state(states)
                next_s = env.transform_state(next_states)
                q_values = network.predict_values(s)
                q_values_new_states = network.predict_values(next_s)
                for i in range(len(args.batch_size)):
                    action = np.argmax(q_values_new_states[i])
                    add_factor = 0
                    if not dones[i]:
                        add_factor = args.gamma * q_values_new_states[i]
                    q_values[i] = rewards[i] + add_factor
                network.train(s, maps, actions, q_values)
