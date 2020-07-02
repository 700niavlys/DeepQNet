'''
振羽不为伍 身疲犹不栖 崇峰饮清露
追月天涯 寂寞辰空舞
==========================================================================================
@author: Xuan ZHU
@license: Institute of Automation，Chinese Academy of Sciences
@contact: xuan.zhu.007@gmail.com
@file: DDQN.py
@time: 07/02/2020
@desc：       
               
'''
class DoubleDQN:
  def learn(self):
    ##same with DQN:
    if self.learn_step_counter % self.replace_target_iter == 0:
      self.sess.run(self.replace_target_op)
      print("target_params_replaced")
    
    if self.memory_counter > self.memory_size:
      sample_index = np.random.choice(self.memory_size, size = self.batch_size)
    else:
      sample_index = np.random.choice(self.memory_counter, size = self.batch_size)
    batch_memory = self.memory[sample_index, :]
    
    ##different with DQN:
    q_next, q_eval4next = self.sess.run(
      [self.q_next, self.q_eval],
      feed_dict = {self.s_: batch_memory[:, -self.n_features:],
                  self.s: batch_memory[:, -self.n_features:]})
    q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})
    q_target = q_eval.copy()
    batch_index = np.arange(self.batch_size, dtype = np.int32)
    eval_act_index = batch_memory[:, self.n_features].astype(int)
    reward = batch_memory[:, self.n_features + 1]
    
    if self.double_q:
      max_act4next = np.argmax(q_eval4next, axis = 1)
      selected_q_next = q_next[batch_index, max_act4next]
    else:
      selected_q_next = np.max(q_next, axis = 1)
    q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next
    
    ##same with DQN:
    _, slef.cost = self.sess.run([self._train_op, self.loss],feed_dict = {self.s: batch_memory[:, :self.n_features],self.q_target:q_target})
    self.cost_his.append(self.cost)
    self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
    self.learn_step_counter += 1
    
