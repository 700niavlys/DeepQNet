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
    
