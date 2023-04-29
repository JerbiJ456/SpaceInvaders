import numpy as np
from game.SpaceInvaders import SpaceInvaders
from epsilon_profile import EpsilonProfile
from matplotlib import pyplot as plt
import os



class Q_Agent():

    def __init__(self, game : SpaceInvaders,  eps_profile: EpsilonProfile, gamma: float, alpha: float):
        self.Q = np.zeros([game.nIntervalsX, game.nIntervalsY, game.nIntervalsX, game.nIntervalsY, 2, game.na])
        self.game = game
        self.na = game.na
        self.gamma = gamma
        self.alpha = alpha
        self.eps_profile = eps_profile
        self.epsilon = self.eps_profile.initial
 

    def learn(self, game, nEpisodes, maxSteps):
        nSteps = np.zeros(nEpisodes) + maxSteps
        qValues = np.zeros(nEpisodes)
        rValues = np.zeros(nEpisodes)

        for episode in range(nEpisodes):
            somme = 0
            state = game.reset()
            self.game.display = False
            if(nEpisodes-episode) < 5 : 
                self.game.display = True
            for step in range(maxSteps):
                action = self.select_action(state)
                next_state, reward, terminal = game.step(action)
                somme += reward
                self.updateQ(state, action, reward, next_state)
                if terminal:
                    nSteps[episode] = step + 1  
                    break
                state = next_state
            self.epsilon = max(self.epsilon - self.eps_profile.dec_episode / (nEpisodes - 1.), self.eps_profile.final)
            if nEpisodes >= 0:
                state = game.reset()
                print("\r#> Ep. {}/{} Value {}".format(episode+1, nEpisodes, self.Q[state][self.select_greedy_action(state)]), end =" ")     
            qValues[episode]=np.sum(self.Q)
            rValues[episode]=somme
        print("\nFINISHED LEARNING")
        self.qFuncTrace(nEpisodes, maxSteps, qValues)
        



    def updateQ(self, state, action, reward, next_state):
        try:
            self.Q[state][action] = (1. - self.alpha) * self.Q[state][action] + self.alpha * (reward + self.gamma * np.max(self.Q[next_state]))
        except:
            print(state, action, next_state)
            quit()

    def select_action(self, state : 'Tuple[int, int, int, int, int]'):
        if np.random.rand() < self.epsilon:
            a = np.random.randint(self.na)
        else:
            a = self.select_greedy_action(state)
        return a

    def select_greedy_action(self, state : 'Tuple[int, int, int, int, int]'):
        mx = np.max(self.Q[state])
        return np.random.choice(np.where(self.Q[state] == mx)[0])
    
    def qFuncTrace(self, nEpisodes, maxSteps, qValues):
        files = os.listdir('courbes')
        name = f"{nEpisodes}nbrEpisodes_{maxSteps}maxSteps"
        if 'qfunc_'+name+'.png' in files:
            for i in range(1,100):
                if f'qfunc_{name}({i}).png' not in files:
                    name += f'{i}' 
        fig1 = plt.figure("Valeurs Q en fonction des épisodes")
        plt.plot(qValues)
        fig1.suptitle('Valeurs Q en fonction des épisodes', fontsize=12)
        plt.xlabel('Numéro d\'épisode', fontsize=8)
        plt.ylabel('Valeurs Q', fontsize=8)
        plt.savefig('courbes/qfunc_'+name+'.png')