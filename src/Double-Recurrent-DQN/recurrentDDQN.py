import gym
import gym_minecraft
import math
from lxml import etree
import logging
import numpy as np
import tensorflow as tf
import random
import time
import pickle
import matplotlib.pyplot as plt
import os


# gym instantiation
frame_size = [100, 100]
num_input = frame_size[0] * frame_size[1]

env = gym.make('MinecraftBasic-v0')
env.init(start_minecraft=True,
         videoResolution=[frame_size[0], frame_size[1]],
         allowDiscreteMovement=["move", "turn"],
         step_sleep=0.25,
         skip_steps=0)  


# training parameters
learningRate = 0.0001
num_nodes = 256
batch_size = 32     
trace_length = 4        # How long each experience trace will be when training
update_freq = 4         # How often to perform a training step.
num_episodes = 100000   # How many episodes of game environment to train network with
total_steps = 0
rList = []          
jList = []          
j_by_loss = []      
j_by_win = []       
j_by_nothing = []   

y = .95                 # Discount factor on the target Q-values
tau = 0.001
pre_train_steps = 10000 # How many episodes of random actions before training begins.

# epsilon greedy
startE = 1                  # Starting chance of random action
endE = 0.1                  # Final chance of random action
annealing_steps = 200000    # How many episodes of training to reduce startE towards endE.
e = startE
stepDrop = (startE - endE) / annealing_steps

# statistics
nb_win = 0
nb_win_tb = 0
nb_nothing = 0
nb_loss = 0

# testing parameters
nb_episodes_by_test = 100
nb_tests = 3
test_print_freq = 50

# checkpoint information
load_model = False
current_step = 0
folder_name="double-recurrent-discrete-A1"
path = "results/"
restoring_backup_folder = path + folder_name + "/"

# logging tensorboard summary
date = str(time.time()).replace(".","")
net = "DoubleRecurrentDQN"
bs = "BatchSize-" + str(batch_size)
strlr = "lr-" + str(learningRate)
rand_step = "RandStep-" + str(pre_train_steps)
nb_to_reduce_e = "ReducE-" + str(annealing_steps)
write_path = "train/" + net + "_" + bs + "_" + strlr + "_" + rand_step + "_" + nb_to_reduce_e + "_" + date[-5:]

# log file for milestones
restoring_backup_name = "double-recurrent-dqn"
filename = restoring_backup_name + ".log"
fileout = open(filename, "a")
fileout.write(restoring_backup_name + "\n")
fileout.close()


# build model
class LSTM():
    def __init__(self, rnn_cell, scope):
        self.x = tf.placeholder("float", [None, num_input])
        self.nextQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.train_length = tf.placeholder(dtype=tf.int32)
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])
        
        self.input_layer = tf.reshape(self.x, [-1, frame_size[1], frame_size[0], 1])

        # Convolutional Layer 1
        self.conv1 = tf.layers.conv2d(
            inputs=self.input_layer,
            filters=32,
            kernel_size=[6, 6],
            strides=[2, 2],
            padding="valid",
            activation=tf.nn.relu)

        # Convolutional Layer 2
        self.conv2 = tf.layers.conv2d(
            inputs=self.conv1,
            filters=64,
            kernel_size=[6, 6],
            strides=[2, 2],
            padding="valid",
            activation=tf.nn.relu)

        # Convolutional Layer 3
        self.conv3 = tf.layers.conv2d(
            inputs=self.conv2,
            filters=64,
            kernel_size=[4, 4],
            strides=[2, 2],
            padding="valid",
            activation=tf.nn.relu)

        self.dims = self.conv3.get_shape().as_list()
        self.final_dimension = self.dims[1] * self.dims[2] * self.dims[3]
        self.conv3_flat = tf.reshape(self.conv3, [-1, self.final_dimension])
        self.rnn_input = tf.reshape(self.conv3_flat, [self.batch_size, self.train_length, self.final_dimension])

        # Initialize the LSTM state
        self.lstm_state_in = rnn_cell.zero_state(self.batch_size, tf.float32)
        self.rnn, self.rnn_state = tf.nn.dynamic_rnn(
            inputs=self.rnn_input, cell=rnn_cell, dtype=tf.float32, initial_state=self.lstm_state_in,
            scope=scope + "_rnn")
        self.rnn = tf.reshape(self.rnn, shape=[-1, num_nodes])

        # Feed Forward
        self.dense = tf.layers.dense(inputs=self.rnn, units=512, activation=tf.nn.relu)
        self.Qout = tf.layers.dense(inputs=self.dense, units=num_classes)
        self.prediction = tf.argmax(self.Qout, 1)
        
        # Multiply actions values by a OneHotEncoding to only take the chosen ones.
        self.actions_onehot = tf.one_hot(self.actions, num_classes, dtype=tf.float32)
        
        # Q-values calculated by the Target network
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        # The loss value corresponds to the difference between the two different Q-values estimated
        self.loss = tf.reduce_mean(tf.square(self.nextQ - self.Q))

        self.merged = tf.summary.merge([tf.summary.histogram("nextQ", self.nextQ),
                                        tf.summary.histogram("Q", self.Q),
                                        tf.summary.scalar("Loss", self.loss)])

        self.learningRate = learningRate
        self.trainer = tf.train.AdamOptimizer(learning_rate=self.learningRate)
        self.updateModel = self.trainer.minimize(self.loss)


class TensorBoardInfosLogger():
    def __init__(self):
        self.percent_win = tf.placeholder(dtype=tf.float32)
        self.mean_j_by_win = tf.placeholder(dtype=tf.float32)
        self.mean_rewards_sum = tf.placeholder(dtype=tf.float32)
        self.merged = tf.summary.merge([
            tf.summary.scalar("Percent_of_win_on_last_50_episodes", self.percent_win),
            tf.summary.scalar("Number_of_steps_by_win_on_last_50_episodes", self.mean_j_by_win),
            tf.summary.scalar("Mean_of_sum_of_rewards_on_last_50_episodes", self.mean_rewards_sum), ])

def processState(state):
    # Downscale input to greyscale
    gray_state = np.dot(state[...,:3], [0.299, 0.587, 0.114])
    return np.reshape(gray_state, num_input)/255.0

def reverse_processState(state):
    return np.reshape(state, (frame_size[0], frame_size[1]))*255.0

def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:total_vars // 2]):  # Get the weights of the original network
        op_holder.append(tfVars[idx + total_vars // 2].assign((var.value() * tau) + (
                    (1 - tau) * tfVars[idx + total_vars // 2].value())))  # Update the Target Network weights
    return op_holder

def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)

class experience_buffer():
    def __init__(self, buffer_size=200000):  # Stores steps
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])

    def get(self):
        return np.reshape(np.array(self.buffer), [len(self.buffer), 5])

class recurrent_experience_buffer():
    def __init__(self, buffer_size=5000):  # Stores episodes
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1 + len(self.buffer)) - self.buffer_size] = []
        self.buffer.append(experience)

    def sample(self, batch_size, trace_length):
        tmp_buffer = [episode for episode in self.buffer if len(episode) + 1 > trace_length]
        sampled_episodes = random.sample(tmp_buffer, batch_size)
        sampledTraces = []
        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode) + 1 - trace_length)
            sampledTraces.append(episode[point:point + trace_length])
        sampledTraces = np.array(sampledTraces)
        return np.reshape(sampledTraces, [batch_size * trace_length, 5])

    def get(self):
        return np.reshape(np.array(self.buffer), [len(self.buffer), 5])

def print_debug_states(tf_session, QNet, raw_input, trace_length):
    tmp = tf_session.run(QNet.input_layer, feed_dict={QNet.x:[raw_input]})
    for depth in range(tmp.shape[3]):
        print("## Input image n°" + str(depth) + " ##")

is_debug = False
if is_debug:
    write_path = 'train/test'
     
myBuffer = None
num_classes = len(env.action_names[0])


tf.reset_default_graph()
with tf.Session() as sess:
    # Use a Double Recurrent Q-Network
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_nodes, state_is_tuple=True)
    cellT = tf.contrib.rnn.BasicLSTMCell(num_units=num_nodes, state_is_tuple=True)

    mainQN = LSTM(cell, 'main')
    targetQN = LSTM(cellT, 'target')

    trainables = tf.trainable_variables()
    targetOps = updateTargetGraph(trainables, tau)

    saver = tf.train.Saver(max_to_keep = 100)

    init = tf.global_variables_initializer()

    tb_infos_logger = TensorBoardInfosLogger()
    writer = tf.summary.FileWriter(write_path)

    if load_model:
        print('Loading Model...')
        picklehandler = open(restoring_backup_folder + 'double-recurrent-dqn-' + str(current_step) + ".pickle", 'rb')
        objectfile = pickle.load(picklehandler)
        picklehandler.close()
        myBuffer = objectfile["Buffer"]
        total_steps = objectfile["Total_steps"]
        e = objectfile["epsilon"]
        rList = objectfile["rList"]
        jList = objectfile["jList"]
        restoring_backup = path + folder_name + "/double-recurrent-dqn-" + str(current_step) + ".meta"
        saver = tf.train.import_meta_graph(restoring_backup)
        saver.restore(sess, tf.train.latest_checkpoint(restoring_backup_folder))
                
    else:
    	myBuffer = recurrent_experience_buffer()

    print("Loading finished...")
    sess.run(init)

    for i in range(num_episodes):
        lstm_state = (
        np.zeros([1, num_nodes]), np.zeros([1, num_nodes]))  # Reset the recurrent layer's hidden state

        episodeBuffer = experience_buffer()

        s = env.reset()
        s = processState(s)
        d = False
        j = 0
        episode_frames = []
        episode_frames.append(s)
        episode_qvalues = []
        episode_rewards = []

        ZPos = []
        XPos = []
        Yaw = []
        moves = []

        while not d:
            j += 1

            # Epsilon Greedy
            if total_steps > pre_train_steps:
                if e > endE:
                    e -= stepDrop

            # Make full exploration before the number of pre-train episodes then play with an e chance of random action during the training (e-greedy)
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                lstm_state1 = sess.run(mainQN.rnn_state,
                                           feed_dict={mainQN.x: [s], mainQN.train_length: 1,
                                                      mainQN.lstm_state_in: lstm_state, mainQN.batch_size: 1})
                index_action_predicted = env.action_space.sample()
                episode_qvalues.append(
                    [1 if i == index_action_predicted else 0 for i in range(len(env.action_names[0]))])
            else:
                if is_debug:
                    print_debug_states(sess, mainQN, s, trace_length)

                prediction, qvalues, lstm_state1 = sess.run([mainQN.prediction, mainQN.Qout, mainQN.rnn_state],
                                                                feed_dict={mainQN.x: [s], mainQN.train_length: 1,
                                                                           mainQN.lstm_state_in: lstm_state,
                                                                           mainQN.batch_size: 1})
                index_action_predicted = prediction[0]
                episode_qvalues.append(qvalues[0])

            # Get new state and reward from environment
            s1_raw, r, d, info = env.step(index_action_predicted)

            if info["observation"]:
                ZPos.append(info['observation']['ZPos'])
                XPos.append(info['observation']['XPos'])
                Yaw.append(info['observation']['Yaw'])
                
            s1 = processState(s1_raw)
            moves.append(index_action_predicted)
            episodeBuffer.add(np.reshape(np.array([s, index_action_predicted, r, s1, d]), [1, 5]))
            episode_frames.append(s1)

            if total_steps > pre_train_steps:
                if total_steps % update_freq == 0:

                    updateTarget(targetOps, sess)  # Update Target Network

                    lstm_state_train = (np.zeros([batch_size, num_nodes]),
                                            np.zeros([batch_size, num_nodes]))

                    trainBatch = myBuffer.sample(batch_size, trace_length)

                    if is_debug:
                        print_debug_states(sess, mainQN, trainBatch[0, 0], trace_length)

                    # Estimate the action to choose by our first network
                    actionChosen = sess.run(mainQN.prediction,
                                                feed_dict={mainQN.x: np.vstack(trainBatch[:, 3]),
                                                           mainQN.train_length: trace_length,
                                                           mainQN.lstm_state_in: lstm_state_train,
                                                           mainQN.batch_size: batch_size})
                    # Estimate all the Q-values by our second network --> Double
                    allQValues = sess.run(targetQN.Qout,
                                              feed_dict={targetQN.x: np.vstack(trainBatch[:, 3]),
                                                         targetQN.train_length: trace_length,
                                                         targetQN.lstm_state_in: lstm_state_train,
                                                         targetQN.batch_size: batch_size})

                    # Train network using target and predicted Q-values
                    end_multiplier = -(trainBatch[:, 4] - 1)
                    maxQ = allQValues[range(batch_size * trace_length), actionChosen]

                    # Bellman Equation
                    targetQ = trainBatch[:, 2] + (y * maxQ * end_multiplier)

                    _, summaryPlot = sess.run([mainQN.updateModel, mainQN.merged],
                                                  feed_dict={mainQN.x: np.vstack(trainBatch[:, 0]),
                                                             mainQN.nextQ: targetQ,
                                                             mainQN.actions: trainBatch[:, 1],
                                                             mainQN.train_length: trace_length,
                                                             mainQN.lstm_state_in: lstm_state_train,
                                                             mainQN.batch_size: batch_size})

                    writer.add_summary(summaryPlot, total_steps)

            episode_rewards.append(r)
            if (s == s1).all():
                print("State error : State did not changed though the action was " + env.action_names[0][
                    index_action_predicted])

            s = s1
            total_steps += 1
            lstm_state = lstm_state1
            
            if d:
                if r == -10000:
                    j_by_loss.append(j)
                elif r > 0: 	#1000
                    j_by_win.append(j)
                elif r < 0: 	#-1000
                    j_by_nothing.append(j)
                break

        myBuffer.add(episodeBuffer.buffer)
        jList.append(j)
        rList.append(sum(episode_rewards))
        rewards = np.array(rList)

        if i % 50 == 0:            
            fileout = open(filename, "a")            
            nb_of_win_on_last_50 = (len(j_by_win) - nb_win_tb)
            win_perc = nb_of_win_on_last_50 / 50 * 100
            mean_j_by_win = np.mean(j_by_win[-nb_of_win_on_last_50:])
            mean_rewards_sum = np.mean(rList[-50:])
            
            fileout.write("######################50######################\n")
            fileout.write("Percent of win on last 50 episodes: " + str(win_perc) + "\n")
            fileout.write("Number of steps by win on last 50 episodes: " + str(mean_j_by_win) + "\n")
            fileout.write("Mean of sum of rewards on last 50 episodes: " + str(mean_rewards_sum) + "\n")
            fileout.write("Total Steps: " + str(total_steps) + "\n")
            fileout.write("I: " + str(i) + "\n")
            fileout.write("Epsilon: " + str(e) + "\n")
            
            nb_win_tb = len(j_by_win)
            fileout.close()

        if i % 500 == 0:
            fileout = open(filename, "a")
            fileout.write("#####################500######################\n")
            fileout.write("% Win : " + str((len(j_by_win) - nb_win) / 5) + "%\n")
            fileout.write("% Loss : " + str((len(j_by_loss) - nb_loss) / 5) + "%\n")
            fileout.write("% Timeout : " + str((len(j_by_nothing) - nb_nothing) / 5) + "%\n")

            fileout.write("Nb J before win: " + str(np.mean(j_by_win[-(len(j_by_win) - nb_win):])) + "\n")
            fileout.write("Nb J before die: " + str(np.mean(j_by_loss[-(len(j_by_loss) - nb_loss):])) + "\n")

            fileout.write("Total Steps: " + str(total_steps) + "\n")
            fileout.write("I: " + str(i) + "\n")
            fileout.write("Epsilon: " + str(e) + "\n")
            
            fileout.write("Last episode rewards: " + str(sum(episode_rewards)) + "\n")
            fileout.close()
            
            nb_loss = len(j_by_loss)
            nb_win = len(j_by_win)
            nb_nothing = len(j_by_nothing)            

        if i % 3000 == 0 and i != 0:
            saver.save(sess, restoring_backup_folder + 'double-recurrent-dqn', global_step=i)
            with open(restoring_backup_folder + 'double-recurrent-dqn-' + str(i) + ".pickle", 'wb') as file:
                dictionary = {
                    "epsilon": e,
                    "Total_steps": total_steps,
                    "Buffer": myBuffer,
                    "rList": rList,
                    "Num Episodes": i,
                    "jList": jList}

                pickle.dump(dictionary, file, protocol=pickle.HIGHEST_PROTOCOL)

    saver.save(sess, 'double-recurrent-dqn', global_step=i)
#end    

