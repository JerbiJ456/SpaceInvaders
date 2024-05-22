# Space Invaders Project: Reinforcement Learning

## Project Overview
This project aims to use a deep learning algorithm within the Space Invaders game. A spaceship (the agent) moves laterally to neutralize aliens (targets) by shooting them. Each eliminated target corresponds to a score point. Our goal is to implement an automatic resolution method that allows the spaceship to maximize its score, hence its reward. We chose to solve this problem using the Q-Learning algorithm previously studied in our labs.

## Table of Contents
- [Introduction](#introduction)
- [State Definition](#state-definition)
- [Reinforcement Learning Algorithm](#reinforcement-learning-algorithm)
- [Hyperparameter Choices](#hyperparameter-choices)
- [Test Results](#test-results)
- [Conclusion and Improvements](#conclusion-and-improvements)

## Introduction
The objective of the project is to use a deep learning algorithm within the Space Invaders game. A spaceship (the agent) moves laterally to neutralize aliens (targets) by shooting them. Each eliminated target corresponds to a score point. We aim to implement an automatic resolution method that allows the spaceship to maximize its score, hence its reward. We chose to solve this problem using the Q-Learning algorithm previously studied in our labs.

## State Definition
In the context of the Space Invaders problem, we need to consider:
1. The position of the spaceship,
2. The position of the alien (invader),
3. The state of the bullet (whether it is fired or not).

The spaceship only moves on the x-axis, while the alien can move horizontally and vertically. When the bullet is fired, it only moves along the y-axis. We define the state as a vector of 5 elements. The first element is the x position of the spaceship, the next two elements are the x and y positions of the invader, and the last two elements describe the state of the bullet: a boolean indicating whether the bullet is fired and its vertical position, respectively:

$$ (x_{\text{agent}}, x_{\text{alien}}, y_{\text{alien}}, \text{state}_{\text{bullet}}, y_{\text{bullet}}) $$

## Reinforcement Learning Algorithm
We chose the Q-Learning algorithm, which is a reinforcement learning algorithm that selects the best actions to take based on the current state of the environment. The function Q(s,a) is a measure of the quality of action \(a\) taken in a given state \(s\). Q-values represent estimates of the Q function for each (s,a) pair in the environment. During learning, the agent updates the Q function as follows:

\[ Q(s, a) = (1 - \alpha) \cdot Q(s, a) + \alpha \cdot (r + \gamma \cdot \max(Q(s', a'))) \]

where \(r\) is the reward, \(\alpha\) is the learning rate, \(\gamma\) is the discount factor, and \(\max(Q(s',a'))\) is the maximum Q-value for all possible actions.

## Hyperparameter Choices
Hyperparameters are variables that can be adjusted to optimize the results of the Q-learning algorithm. We consider the following hyperparameters, which we have encountered in our labs:

- \(\alpha\) (learning rate): A value between 0 and 1. A high learning rate allows faster learning but can also lead to oscillations and instability. By varying \(\alpha\), we found it was wise to set it to 1 to maximize the spaceship's score while accelerating learning.
- \(\gamma\) (discount factor): Describes the relative importance of immediate and future rewards in the calculation of the Q-value of a given action. Agents trained with a discount factor close to 1 perform better than those with a smaller gamma. We set \(\gamma = 0.99\) to give significant importance to future rewards while considering immediate rewards.
- \(\text{max\_steps}\): The number of steps evaluated to understand its impact on the final result. We set \(\text{max\_steps}\) to 1000.
- \(\epsilon\): An initial epsilon value showed that the agent learned faster and explored the space. \(\epsilon_{\text{final}}\) allows the agent to explore the space while pushing it not to choose actions randomly.
- \(\text{episodes}\): The number of episodes is the number of iterations over which the agent can learn. We tested our application with several values of \(\text{nb\_episodes}\). Our agent learned over more than 20,000 episodes, producing the best results.

## Test Results
The figure below is a graph of the sum of Q-values on the y-axis and the number of episodes on the x-axis. At episode 0, we observe a sum of 0 since we initialized our code with zeros. The goal is to note the value of \(\text{nb\_episodes}\) for which the sum of Q-values stabilizes.

We observe that the Q-value increases positively with the number of episodes. This indicates that the agent learns more as the number of episodes increases. We did not manage to observe the stabilization of Q-values, characterized by a plateau, but theoretically, we should achieve this. It is possible that our exploration is insufficient and we need to go beyond 20,000 episodes, but the significant execution time prevented us from pushing the tests further.

![Evolution of Q-values with the number of episodes](./Images/qvaleur.png)

## Conclusion and Improvements
The evolution of our Q-learning seems relevant, and the hyperparameters we tested provide a good overall result. However, as an improvement, we could consider reducing the execution time of the algorithm. We could push the algorithm to a higher level of approximation through deep Q-learning, although this complexity requires higher computational power.

---
