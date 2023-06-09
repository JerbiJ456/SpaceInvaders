from time import sleep
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.random_agent import RandomAgent
from controller.qAgent import Q_Agent
from epsilon_profile import EpsilonProfile

def main():

    game = SpaceInvaders(display=True)
    #controller = KeyboardController()
    #controller = RandomAgent(game.na)
    gamma = 1.
    alpha = 0.8
    eps_profile = EpsilonProfile(0.7, 0.05)
    max_steps = 2500
    n_episodes = 500

    controller = Q_Agent(game, eps_profile, gamma, alpha)
    controller.learn(game, n_episodes, max_steps)
    
    """for n_episodes in range(10, 100000, 10):
        controller = Q_Agent(game, eps_profile, gamma, alpha)
        controller.learn(game, n_episodes, max_steps)
    """

    state = game.reset()
    while True:
        action = controller.select_action(state)
        state, _, _ = game.step(action)
        sleep(0.0001)

if __name__ == '__main__' :
    main()
