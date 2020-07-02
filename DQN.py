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
  step +=1
