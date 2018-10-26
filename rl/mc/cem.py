from collections import deque

import numpy as np

import torch

import logging
logger = logging.getLogger(__name__)


def cem(agent, params):
    """PyTorch implementation of a cross-entropy method.
        
    Params
    ======
        agent (object) --- the agent to train
        params (dict) --- a dictionar of parameters
    """
    n_episodes = params.get('episodes', 2000)
    maxlen = params.get('maxlen', 100)
    name = params['name']
    pop_size =  params.get('pop_size', 50)
    elite_frac = params.get('elite_frac', 0.2)
    sigma = params.get('sigma', 0.5)
    n_elite = int(pop_size*elite_frac)
    glie_policy = params.get('policy', None)
    policy = glie_policy(params.get('policy_params', None))  # initialize epsilon-greedy policy

    scores = []                           # list containing scores from each episode
    scores_window = deque(maxlen=maxlen)  # last N scores
    best_weight = sigma * np.random.randn(agent.get_weights_dim())

    for i_episode in range(1, n_episodes+1):
        weights_pop = [best_weight + (sigma * np.random.randn(agent.get_weights_dim())) for i in range(pop_size)]
        rewards = np.array([agent.evaluate(weights, policy) for weights in weights_pop])

        elite_idxs = rewards.argsort()[-n_elite:]
        elite_weights = [weights_pop[i] for i in elite_idxs]
        best_weight = np.array(elite_weights).mean(axis=0)

        reward = agent.evaluate(best_weight, policy)

        scores_window.append(reward)
        scores.append(reward)
        policy.decay()                    # decrease epsilon
        
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'results/' + name + '_checkpoint.pth')
    return scores
