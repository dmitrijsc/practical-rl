# Reinforcement learning in Julia

Purpose of this project/ repository is to replicate Practical RL course (https://github.com/yandexdataschool/Practical_RL) exercises in Julia.

About Practical RL (from their GitHub): A course on reinforcement learning in the wild. Taught on-campus in HSE and Yandex SDA.

## Setup

This code was tested on Julia 0.6.1. It will be using MXNet for all deep learning activities.

Please follow standard process to configure `Open AI gym`, `POMDPs.jl` and `MXNet.jl` from the corresponding package repository.

## Table of contents

The following objectives are implemented and available as a separate Julia files:

week_0: FrozenLake with genetic algorithms
- FrozenLake (4x4): average score 0.86
- FrozenLake (8x8): average score 0.97

week_1: Cross-Entropy Method
- FrozenLake8x8, Taxi-v2 (CEM)
- CartPole-v0 (Deep Cross-Entropy Method using MXNet): average score 200.0

week_2: Q-learning (Value-table method)
- Taxi-v2: average score 8.7/8.5

week_3: SARSA
- Taxi-v2: NA

## T&C

I will try to follow Julia best practices when writing code and optimize it whenever and wherever possible.

Please excuse in case of any serious issues and you are welcome to submit your PR.

## Contacts

Feel free to contact me over Issues, solveai.net or Julia Slack channel @dmitrijsc.
