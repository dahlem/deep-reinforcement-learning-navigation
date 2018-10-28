from collections import deque

import numpy as np

import torch
import torch.optim as optim

def reinforce(agent, params):
    n_episodes = params.get('episodes', 2000)
    max_t = params.get('max_t', 1000)
    maxlen = params.get('maxlen', 100)
    name = params['name']
    brain_name = params['brain_name']
    env = params['environment']
    gamma = params['agent_params'].get('gamma', 0.99)
    
    optimizer = optim.Adam(agent.parameters(), lr=params['agent_params'].get('lr', 0.001))
    scores_window = deque(maxlen=maxlen)
    scores = []
    for i_episode in range(1, n_episodes+1):
        saved_log_probs = []
        rewards = []
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        for t in range(max_t):
            action, log_prob = agent.act(state)
            saved_log_probs.append(log_prob)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            state = env_info.vector_observations[0]        # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            rewards.append(reward)
            if done:
                break 
        scores_window.append(sum(rewards))
        scores.append(sum(rewards))
        
        discounts = [gamma**i for i in range(len(rewards)+1)]
        R = sum([a*b for a,b in zip(discounts, rewards)])
        
        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()
        
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'results/' + name + '_checkpoint.pth')        
    return scores
