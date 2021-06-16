# This is a breakout Q-Learning algo from the book.
# Can play using A, D and SPACE keys.   

# NOTE: Run as root.
#       Enable access for assistive devices
from collections import deque
import random
import gym

import queue, threading, time
from pynput.keyboard import Key, Listener

import numpy

import cv2

import tensorflow as tf


# Class for a Q learning network. Not deep yet.
class QNetwork:
    def __init__(self, input_shape=(84, 84, 4), n_outputs=4,
                 network_type='cnn', scope='q_network'):
    
        self.width = input_shape[0]
        self.height = input_shape[1]
        self.channel = input_shape[2]
        self.n_outputs = n_outputs
        self.network_type = network_type
        self.scope = scope

        # Frame images
        self.x = tf.placeholder(dtype=tf.float32,
                                shape=(None, self.channel,
                                    self.width, self.height))
        
        # Estimates of Q-value
        self.y = tf.placeholder(dtype=tf.float32, shape=(None,))
        # Selected actions
        self.a = tf.placeholder(dtype=tf.int32, shape=(None,))

        with tf.variable_scope(scope):
            self.build()
            self.build_loss()

    def build(self):
        self.net = {}
        self.net['input'] = tf.transpose(self.x, perm=(0, 2, 3, 1))

        init_b = tf.constant_initializer(0.01)
        
        if self.network_type == 'cnn':
            self.net['conv1'] = tf.conv2d(self.net['input'], 32,
                                          kernel=(8, 8), stride=(4, 4),
                                          init_b=init_b, name='conv1')
            self.net['conv2'] = tf.conv2d(self.net['input'], 64,
                                          kernel=(4, 4), stride=(2, 2),
                                          init_b=init_b, name='conv2')
            self.net['conv3'] = tf.conv2d(self.net['input'], 64,
                                          kernel=(3, 3), stride=(1, 1),
                                          init_b=init_b, name='conv3')
            self.net['feature'] = tf.dense(self.net['conv2'], 512,
                                           init_b=init_b, name='fc1')
        
        elif self.network_type == 'cnn_nips':
            self.net['conv1'] = tf.conv2d(self.net['input'], 16,
                                          kernel=(8, 8), stride=(4, 4),
                                          init_b=init_b, name='conv1')
            self.net['conv2'] = tf.conv2d(self.net['conv1'], 32,
                                          kernel=(4, 4), stride=(2, 2),
                                          init_b=init_b, name='conv2')
            self.net['feature'] = tf.dense(self.net['conv2'], 256,
                                           init_b=init_b, name='fc1')
        elif self.network_type == 'mlp':
            self.net['fc1'] = tf.dense(self.net['input'], 50,
                                       init_b=init_b, name='fc1')
            self.net['feature'] = tf.dense(self.net['fc1'], 50,
                                           init_b=init_b, name='fc2')
        
        else:
            raise NotImplementedError('Unknown network type')

        self.net['values'] = tf.dense(self.net['feature'],
                                      self.n_outputs, activation=None,
                                      init_b=init_b, name='values')
        self.net['q_value'] = tf.reduce_max(self.net['values'],
                                            axis=1, name='q_value')
        self.net['q_action'] = tf.argmax(self.net['values'],
                                         axis=1, name='q_action',
                                         output_type=tf.int32)
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      tf.get_variable_scope().name) 

    def build_loss(self):
        indices = tf.transpose(tf.stack([tf.range(tf.shape(self.a)[0]),
                               self.a], axis=0))
        value = tf.gather_nd(self.net['values'], indices)
        
        self.loss = 0.5 * tf.reduce_mean(tf.square((value - self.y)))
        self.gradient = tf.gradients(self.loss, self.vars)
                                     
        tf.summary.scalar("loss", self.loss, collections=['q_network'])
        self.summary_op = tf.summary.merge_all('q_network')

# Class for working with memory and adapting for QNN
class ReplayMemory:
    def __init__(self, history_len=4, capacity=1000000,
                 batch_size=32, input_scale=255.0):
        self.capacity = capacity
        self.history_length = history_len
        self.batch_size = batch_size
        self.input_scale = input_scale
        self.frames = deque([])
        self.others = deque([])
    
    def add(self, frame, action, r, termination):
        if len(self.frames) == self.capacity:
            self.frames.popleft()
            self.others.popleft()
        self.frames.append(frame)
        self.others.append((action, r, termination))
    
    def add_nullops(self, init_frame):
        for _ in range(self.history_length):
            self.add(init_frame, 0, 0, 0)
    
    # Makes a 84x84x4 input image for learning models
    def phi(self, new_frame):
        assert len(self.frames) > self.history_length
        images = [new_frame] + [self.frames[-1-i] for i in range(self.history_length-1)]
        return numpy.concatenate(images, axis=0)
    
    # Draw random transition from memory
    def sample(self):
        while True:
            index = random.randint(a=self.history_length-1,
                                   b=len(self.frames)-2)
            infos = [self.others[index-i] for i in range(self.history_length)]
            # Check if termination=1 before "index"
            flag = False
            for i in range(1, self.history_length):
                if infos[i][2] == 1:
                    flag = True
                    break
            if flag:
                continue

            state = self._phi(index)
            new_state = self._phi(index+1)
            action, r, termination = self.others[index]
            state = numpy.asarray(state / self.input_scale,
                                  dtype=numpy.float32)
            new_state = numpy.asarray(new_state / self.input_scale,
                                      dtype=numpy.float32)
            return (state, action, r, new_state, termination)
    
    # phi, but actually stacks 4 frames together
    def _phi(self, index):
        images = [self.frames[index-i] for i in range(self.history_length)]
        return numpy.concatenate(images, axis=0)

class Optimizer:
    def __init__(self, config, feedback_size,
                 q_network, target_network, replay_memory):
        self.feedback_size = feedback_size
        self.q_network = q_network
        self.target_network = target_network
        self.replay_memory = replay_memory
        self.summary_writer = None

        self.gamma = config['gamma']
        self.num_frames = config['num_frames']

        optimizer = tf.create_optimizer(config['optimizer'],
                                        config['learning_rate'],
                                        config['rho'],
                                        config['rmsprop_epsilon'])
        
        self.train_op = optimizer.apply_gradients(
                zip(self.q_network.gradient,
                self.q_network.vars))
        
    def sample_transitions(self, sess, batch_size):
        w, h = self.feedback_size
        states = numpy.zeros((batch_size, self.num_frames, w, h),
                              dtype=numpy.float32)

        new_states = numpy.zeros((batch_size, self.num_frames, w, h),
                                  dtype=numpy.float32)
        
        targets = numpy.zeros(batch_size, dtype=numpy.float32)
        actions = numpy.zeros(batch_size, dtype=numpy.int32)
        terminations = numpy.zeros(batch_size, dtype=numpy.int32)

        for i in range(batch_size):
            state, action, r, new_state, t = self.replay_memory.sample()
            states[i] = state
            new_states[i] = new_state
            actions[i] = action
            targets[i] = r
            terminations[i] = t

        targets += self.gamma * (1 - terminations) * self.target_network.get_q_value(sess, new_states)
        return states, actions, targets
    
    def train_one_step(self, sess, step, batch_size):
        states, actions, targets = self.sample_transitions(sess, batch_size)
        feed_dict = self.q_network.get_feed_dict(states, actions, targets)
        if self.summary_writer and step % 1000 == 0:
            summary_str, _, = sess.run([self.q_network.summary_op,
                                        self.train_op],
                                        feed_dict=feed_dict)
            self.summary_writer.add_summary(summary_str,
                                            self.summary_writer.flush())
        
        else:
            sess.run(self.train_op, feed_dict=feed_dict)

# Actual deep Q network
class DQN:
    def __init__(self, config, game, directory,
                 callback=None, summary_writer=None):
        self.game = game
        self.actions = game.get_available_actions()
        self.feedback_size = game.get_feedback_size()
        self.callback = callback
        self.summary_writer = summary_writer
        self.config = config
        self.batch_size = config['batch_size']
        self.n_episode = config['num_episode']
        self.capacity = config['capacity']
        self.epsilon_decay = config['epsilon_decay']
        self.epsilon_min = config['epsilon_min']
        self.num_frames = config['num_frames']
        self.num_nullops = config['num_nullops']
        self.time_between_two_copies = config['time_between_two_copies']
        self.input_scale = config['input_scale']
        self.update_interval = config['update_interval']
        self.directory = directory
        self._init_modules()

    def train(self, sess, saver=None):
        num_of_trials = -1
        for episode in range(self.n_episode):
            self.game.reset()
            frame = self.game.get_current_feedback()
            for _ in range(self.num_nullops):
                r, new_frame, termination = self.play(action=0)
                self.replay_memory.add(frame, 0, r, termination)
                frame = new_frame
            for _ in range(self.config['T']):
                num_of_trials += 1
                epsilon_greedy = self.epsilon_min + \
                    max(self.epsilon_decay - num_of_trials, 0) / \
                    self.epsilon_decay * (1 - self.epsilon_min)
                
                if num_of_trials % self.update_interval == 0:
                    self.optimizer.train_one_step(sess,
                                                  num_of_trials, 
                                                  self.batch_size)
                    
                    state = self.replay_memory.phi(frame)
                    action = self.choose_action(sess, state, epsilon_greedy)
                    r, new_frame, termination = self.play(action)
                    self.replay_memory.add(frame, action, r, termination)
                    frame = new_frame

                if num_of_trials % self.time_between_two_copies == 0:
                    self.update_target_network(sess)
                    self.save(sess, saver)

                if self.callback:
                    self.callback()
                
                if termination:
                    score = self.game.get_total_reward()
                    summary_str = sess.run(self.summary_op,
                                           feed_dict={self.t_score: score})
                    self.summary_writer.add_summary(summary_str,
                                                    num_of_trials)
                    self.summary_writer.flush()
                    break

    def evaluate(self, sess):
        for episode in range(self.n_episode):
            self.game.reset()
            frame = self.game.get_current_feedback()
            for _ in range(self.num_nullops):
                r, new_frame, termination = self.play(action=0)
                self.replay_memory.add(frame, 0, r, termination)
                frame = new_frame
            for _ in range(self.config['T']):
                state = self.replay_memory.phi(frame)
                action = self.choose_action(sess, state, self.epsilon_min)
                r, new_frame, termination = self.play(action)
                self.replay_memory.add(frame, action, r, termination)
                frame = new_frame

                if self.callback:
                    self.callback()
                    if termination:
                        break
                    
# RGB to monochrome
def rgb_to_gray(self, im):
    return numpy.dot(im, [0.2126, 0.7152, 0.0722])

# Image cropping
def cv2_resize_image(image, resized_shape=(84, 84),
                     method='crop', crop_offset=8):
    height, width = image.shape
    resized_height, resized_width = resized_shape

    if method == 'crop':
        h = int(round(float(height) * resized_width / width))
        resized = cv2.resize(image,
                             (resized_width, h),
                             interpolation=cv2.INTER_LINEAR)
        crop_y_cutoff = h - crop_offset - resized_height
        cropped = resized[crop_y_cutoff:crop_y_cutoff+resized_height, :]
        return numpy.asarray(cropped, dtype=numpy.uint8)
    elif method == 'scale':
        return numpy.asarray(cv2.resize(image,
                                        (resized_width, resized_height),
                                        interpolation=cv2.INTER_LINEAR),
                                        dtype=numpy.uint8)
    else:
        raise ValueError('Unrecognized image resize method.')


network = DQN

environment = gym.make('Breakout-v0')

network.__init__(network, 'cnn', environment, '/logs')

network.train(network, environment)
