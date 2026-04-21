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
- [2. Double Deep Q-Network](#2-Double-Deep-Q-Network)
- [3. Quantum Reinforcement Learning](#3-Quantum-Reinforcement-Learning)
- [4. Experiments](#4-Experiments)
- [5. Conclusion](#5-Conclusion)
- [Reference](#Reference)

## <span id="1-Abstract">Abstract</span>
TBD

## <span id="1-Introduction">1. Introduction</span>
Reinforcement learning (RL) enables an agent to learn optimal behavior through interaction with an environment, and has been widely applied in robotics, optimization, and game AI. However, RL often struggles in environments with large state spaces and sparse rewards. In pixel-based game environments, high-dimensional inputs and long-term dependencies further make stable learning difficult.

Deep Reinforcement Learning addresses these challenges by combining RL with deep neural networks. In particular, Deep Q-Network (DQN) has demonstrated strong performance on Atari benchmarks using raw pixel inputs [[1]](#ref-dqn). However, DQN suffers from overestimation of Q-values, which can lead to instability. Double Deep Q-Network (DDQN) was proposed to mitigate this issue by decoupling action selection from value evaluation [[2]](#ref-ddqn).

Recently, there has been growing interest in applying quantum computing principles, such as superposition and entanglement, to reinforcement learning. These approaches suggest potential improvements in representation and learning efficiency.

This project implements and compares DDQN-based classical reinforcement learning and Quantum Reinforcement Learning (QRL) in the Super Mario Bros. environment. Both approaches are evaluated under identical conditions in terms of learning stability, convergence speed, and policy performance, with the goal of exploring the potential advantages and limitations of QRL.

## <span id="2-Double-Deep-Q-Network">2. Double Deep Q-Network</span>
TBD

## <span id="3-Quantum-Reinforcement-Learning">3. Quantum Reinforcement Learning</span>
TBD

## <span id="4-Experiments">4. Experiments</span>
TBD

## <span id="5-Conclusion">5. Conclusion</span>
TBD

## <span id="Reference">Reference</span>

<span id="ref-dqn">1. Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning." arXiv preprint arXiv:1312.5602 (2013).</span>

<span id="ref-ddqn">2. Van Hasselt, Hado, Arthur Guez, and David Silver. "Deep reinforcement learning with double q-learning." Proceedings of the AAAI conference on artificial intelligence. Vol. 30. No. 1. 2016.</span>

