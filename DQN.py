'''
振羽不为伍 身疲犹不栖 崇峰饮清露
追月天涯 寂寞辰空舞
==========================================================================================
@author: Xuan ZHU
@license: Institute of Automation，Chinese Academy of Sciences
@contact: xuan.zhu.007@gmail.com
@file: DQN.py
@time: 07/02/2020
@desc：       
               
'''
##net training
while True:
  env.render()
 action = RL.choose_action(observation)
 observation_, reward, done = env.step(action)
 RL.store_transition(observation, action, reward, observation_)
 if(step > x) and (step % y ==0):
    RL.learn()
    observation = observation_
  if done:
    break
  step += 1

##weights update
def choose_action(self,ovservation):
  ovservation = observation[np.newaxis,:]
  if np.random.uniform() < self.epsilon:
    actions_value = self.sess.run(self.q_eval, feed_dict = {self.s:observation})
    action = np.argmax(actions_value)
  else：
    action = np.random.randint(0, self.n_actions)
  return action

def store_transition(self, s, a, r, s_):
  if not hasattr(self, 'memory_counter'):
    self.memory_counter = 0
  transition = np.hstack((s,[a, r], s_))
  index = self.memory_counter % self.memory_size 
  self.memory[index, :] = transition
  self.memory_counter += 1

def learn(self):
  ...
  q_target = q_eval.copy()
  batch_index = np.arange(self.batch_size, dtype = np.int32)
  eval_act_index = batch_memory[:, self.n_features].astype(int)
  reward = batch_memory[;, self.n_features + 1]
  
  q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis = 1)
  _, self.cost = self.sess.run([self.train_op, self.loss], feed_dict = {self.s: batch_memory[:, :self.n_features], self.q_target: q_target})
  self.cost_his.append(self.cost)
  
def _build_net(self):
  ##input
  self.s = tf.placeholder(tf.float32, [None, self.n_features], name = 's')
  ##input state
  self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name = 's_')
  ##input next_state
  self.r = tf.placeholder(tf.float32, [None,], name= 'r')
  ##reward
  self.a = tf.placeholder(tf.int32, [None,], name = 'a')
  ##action
  
  w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3),tf.constant_initializer(0.1)
  
  ##build eval net
  with tf.variable_scope('online_net'):
    e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer = w_initializer, bias_initializer = b_initializer, name = 'e1')
    self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer = w_initializer, bias_initializer = b_initializer, name = 'q')
  ##build target net
  with tf.variable_scope('target_net'):
    t1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer = w_initializer, bias_initializer = b_initializer, name = 't1')
    self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer = w_initialzier, bias_initializer = b_initialzier, name = 't2')
  
  with tf.variable_scope('q_target'):
    q_target = self.r + self.gamma * tf.reduce_max(self.q_next , axis = 1, name = 'Qmax_s_')
    self.q_target = tf.stop_gradient(q_target)
  
  with tf.variable_scope('q_eval'):
    a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype = tf.int32), self.a], axis = 1)
    self.q_eval_wrt_a = tf.gather_nd(params = self.q_eval, indices = a_indices)
    
  with tf.variable_scope('loss'):
    self.loss = tf.reduce_mean(tf.squared_difference(sefl.q_target, self.q_eval_wrt_a, name = 'TD_error'))
  with tf.variable_scope('train'):
    self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
    
  
