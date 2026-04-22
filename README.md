<p align="center">
    <img src="https://raw.githubusercontent.com/WooSangyoon/quantum-mario-bros/main/images/logo.png" width="300px" />
</p>
<h1 align="center">Quantum Mario Bros</h1>

<p align="center">
<a href="https://github.com/WooSangyoon/QuantumMarioBros/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue?style=for-the-badge&labelColor=162246"/></a>
<a href="https://github.com/WooSangyoon/QuantumMarioBros/blob/main/README.md"><img alt="Language:English" src="https://img.shields.io/badge/language-English-blue?style=for-the-badge&labelColor=162246"/></a>

<p>

> 🎮 Clearing Super Mario Bros with Quantum Reinforcement Learning

> 🧑‍💻 This project was created for personal learning purposes. If you find any mistakes or have suggestions, please open an issue.

## 📋 Table of Contents
- [Abstract](#Abstract)
- [1. Introduction](#1-Introduction)
- [2. Double Deep Q-Network (DDQN)](#2-Double-Deep-Q-Network)
- [3. Quantum Reinforcement Learning (QRL)](#3-Quantum-Reinforcement-Learning)
- [4. Experiments](#4-Experiments)
- [5. Conclusion](#5-Conclusion)
- [How to Run](#How-to-Run)
- [Reference](#Reference)

## <span id="Abstract">Abstract</span>
TBD

## <span id="1-Introduction">1. Introduction</span>
Reinforcement learning (RL) enables an agent to learn optimal behavior through interaction with an environment, and has been widely applied in robotics, optimization, and game AI. However, RL often struggles in environments with large state spaces and sparse rewards. In pixel-based game environments, high-dimensional inputs and long-term dependencies further make stable learning difficult.

Deep Reinforcement Learning addresses these challenges by combining RL with deep neural networks. In particular, Deep Q-Network (DQN) has demonstrated strong performance on Atari benchmarks using raw pixel inputs [[1]](#ref-dqn). However, DQN suffers from overestimation of Q-values, which can lead to instability. Double Deep Q-Network (DDQN) was proposed to mitigate this issue by decoupling action selection from value evaluation [[2]](#ref-ddqn).

Recently, there has been growing interest in applying quantum computing principles, such as superposition and entanglement, to reinforcement learning. These approaches suggest potential improvements in representation and learning efficiency.

This project implements and compares DDQN-based classical reinforcement learning and Quantum Reinforcement Learning (QRL) in the Super Mario  environment. Both approaches are evaluated under identical conditions in terms of learning stability, convergence speed, and policy performance, with the goal of exploring the potential advantages and limitations of QRL.

## <span id="2-Double-Deep-Q-Network">2. Double Deep Q-Network (DDQN)</span>

<p>
<img id="fig1" src="https://raw.githubusercontent.com/WooSangyoon/quantum-mario-bros/main/images/ddqn_architecture.png"/>
</p>

**Figure 1.** Overall architecture of DDQN.

<br/>

Previous neural network–based reinforcement learning methods suffered from difficulties in training due to high correlation between data samples and sparse or unstable rewards. To address these issues, DQN was proposed. This approach approximates the state–action value function $Q(s, a)$ using a neural network, enabling more stable reinforcement learning compared to earlier methods, even in high-dimensional input environments. The agent selects the action with the highest Q-value at each state and updates the network based on the Bellman equation.

However, DQN tends to overestimate Q-values because it uses the same network for both action selection and evaluation. This can lead to unstable training and degraded performance. DDQN was proposed as a solution to this problem by using separate networks for action selection and evaluation. This approach reduces overestimation and enables more stable learning. 

The network used for action selection is called the Online network, which is updated at every step and used to choose actions. The network used for evaluation is called the Target network, which is updated periodically and used to compute stable target values. This structure leads to more stable performance, especially in environments with large state spaces or unstable rewards. (See [Figure 1](#fig1) for the model architecture.)

In this project, DDQN is used as the baseline for classical reinforcement learning and is compared with QRL.

## <span id="3-Quantum-Reinforcement-Learning">3. Quantum Reinforcement Learning (QRL)</span>
TBD

## <span id="4-Experiments">4. Experiments</span>

### Data Preprocessing

<p>
<img id="fig2" src="https://raw.githubusercontent.com/WooSangyoon/quantum-mario-bros/main/images/data_preprocessing.png"/>
</p>

**Figure 2.** Overall architecture of DDQN.

In this project, the model is trained directly from raw data rather than relying on manually engineered features. However, using raw frames as input introduces several challenges, including increased computational cost due to RGB channels, high image resolution, and strong temporal correlation between consecutive frames. These factors can reduce training efficiency and lead to biased learning. To address these issues, we apply a series of preprocessing techniques to improve learning performance.

First, the Super Mario Bros environment exhibits minimal changes between consecutive frames under normal conditions. As a result, using every frame can lead to redundant observations and slow convergence, while also reducing the diversity of training data. To mitigate this, we apply a frame skipping strategy, where only every k-th frame is used for training. In this project, we set k = 4, meaning that the agent processes one frame every four frames.

Next, each frame is converted to grayscale. The original frames consist of three RGB channels, but color information has limited importance for gameplay. Converting frames to grayscale reduces the input dimensionality by a factor of three, significantly improving computational efficiency.

We then resize the frames to a smaller resolution while preserving essential features. In this project, all frames are resized to 84×84 pixels, which reduces computational cost while maintaining relevant spatial information.

Finally, since a single frame does not capture motion dynamics, we stack multiple consecutive frames to provide temporal context. Specifically, we stack four frames to represent the state input. The overall preprocessing pipeline is illustrated in [Figure 2](#fig2).


### DDQN

### QRL


## <span id="5-Conclusion">5. Conclusion</span>
TBD


## <span id="How-to-Run">How to Run</span>

> [!CAUTION]
> This project is currently under development. Execution commands, dependencies, and other components may be modified or removed during the development process.

### Training
```bash
# DDQN
python main.py --agent ddqn --mode train --episodes 10000

# DDQN with rendering
python main.py --agent ddqn --mode train --render-train --episodes 10000

# Quantum
python main.py --agent quantum --mode train --episodes 10000

# Quantum with rendering
python main.py --agent quantum --mode train --render-train --episodes 10000
```

### Evaluation
```bash
# DDQN
python main.py --agent ddqn --mode eval --episodes 1

# Quantum
python main.py --agent quantum --mode eval --episodes 1
```

### Notes

- `--render-train` enables rendering during training.
- `eval` mode renders the environment by default.
- If `--episodes` is omitted, the default value from `config.py` is used.



## <span id="Reference">Reference</span>

<span id="ref-dqn">1. Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning." arXiv preprint arXiv:1312.5602 (2013).</span>

<span id="ref-ddqn">2. Van Hasselt, Hado, Arthur Guez, and David Silver. "Deep reinforcement learning with double q-learning." Proceedings of the AAAI conference on artificial intelligence. Vol. 30. No. 1. 2016.</span>
