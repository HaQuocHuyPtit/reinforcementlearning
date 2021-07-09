from game import *

rewards_all_episodes = []

for episode in range(num_episodes):
    state = env.reset()
    done = False
    rewards_current_episode = 0
    for step in range(max_steps_per_episode):
        exploration_rate_threshole = np.random(0,1)
        if exploration_rate_threshole > exlpration_rate:
            action = q_table[state, :]
        else:
            action = env.action_space.sample()
        
        new_state, reward, done, info = env.step(action)

        # Update Q-table for Q(s, a)
        q_table[state, action] = q_table[state, action] * (1 -learning_rate) + \
                                learning_rate * (reward +discount_rate * np.max(q_table[new_state, :]))
        state = new_state
        rewards_current_episode += reward 

        if done == True:
            break
    
    exploration_rate = min_exploration_rate + \
                (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)

    rewards_all_episodes.append(rewards_current_episode)


reward_per_thousand_episodes = np.split(np.array(rewards_all_episode), num_episodes/1000)
count = 1000


for r in reward_per_thousand_episodes:
    print(count, ": ", str(sum(r/100)))
    count += 1000
