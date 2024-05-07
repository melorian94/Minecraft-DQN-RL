# Minecraft-DQN-RL

This repository provides and in-depth analysis of several Deep Q-Network (DQN) approaches in terms of performance and hardware footprint. Our target mission is a partially observable navigation problem, modeled in Minecraft, a state-of-the-art training and testing environment for research focusing on lifelong learning. The aim of this work is to compare several approaches fairly on a common task.

# Prerequisites

- Python 3.5
- TensorFlow

# Installation

- Configure and install the [Malmo Platform](https://github.com/Microsoft/malmo)
- Install [Gym-Minecraft](https://github.com/tambetm/gym-minecraft)

# Models included in the analysis

- Double Deep Q-Network
- Stacked Double Deep Q-Network
- Recurrent Double Deep Q-Network
- Dueling Double Deep Q-Network
- Stacked Dueling Double Deep Q-Network
- Recurrent Dueling Double Deep Q-Network

# Credits

We would like to thank Clément Romac and Pierre Leroy for their exploratory work on partially observable missions. Tambet Matiisen for his implementation of a flexible and reliable training and testing environment for Minecraft.

# Selected references

- Double DQNs:
  - [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
  - [Playing atari with deep reinforcement learning](https://arxiv.org/abs/1312.5602)
  - [Deep reinforcement learning with double q-learning](https://ojs.aaai.org/index.php/AAAI/article/view/10295)
- Recurrent DQNs:
  - [Deep recurrent q-learning for partially observable mdps](https://cdn.aaai.org/ocs/11673/11673-51288-1-PB.pdf)
- Dueling DQNs:
  - [Dueling network architectures for deep reinforcement learning](http://proceedings.mlr.press/v48/wangf16.html)
- Practical Approach to RL:
  - [An introduction to deep reinforcement learning](https://www.nowpublishers.com/article/Details/MAL-071)
    







