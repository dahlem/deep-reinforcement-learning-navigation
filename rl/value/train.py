from collections import deque

import numpy as np

import torch


def train(agent, env, params):
    """Training Loop for value-based RL methods.
    
    Params
    ======
        agent (object) --- the agent to train
        params (dict) --- the dictionary of parameters
    """
    n_episodes = params.get('episodes', 2000)
    max_t = params.get('max_t', 1000)
    beta_start = params.get('beta_start', 0.4)
    maxlen = params.get('maxlen', 100)
    glie_policy = params.get('policy', None)
    name = params['name']
    brain_name = params['brain_name']
    env = params['environment']
    
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=maxlen)  # last N scores
    policy = glie_policy(params.get('policy_params', None))  # initialize epsilon-greedy policy

    beta_schedule = lambda episode: min(1.0, beta_start + episode * (1.0 - beta_start) / n_episodes)

    for i_episode in range(1, n_episodes+1):
        beta = beta_schedule(i_episode)
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, policy)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state, action, reward, next_state, done, beta)
            score += reward                                # update the score
            state = next_state                             # roll over the state to next time step
            if done:                                       # exit loop if episode finished
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        policy.decay()                    # decrease epsilon
        
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'results/' + name + '_checkpoint.pth')
    return scores
