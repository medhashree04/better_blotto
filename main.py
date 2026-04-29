import numpy as np
import random
import pandas as pd
from itertools import product

Blotto_Troops = 10
fields = 3 # consistent for the game
Lotso_increments =5

digits = range(11)
moves = [p for p in product(range(0, 11), repeat=3) if sum(p) == 10]






def Lotso_Choose():
    #randomly selects an interval of five troops between five and thirty
    troops = Lotso_increments*random.randint(1,6)
    move = np.zeroes(fields)

    for troop in troops:
        nbin = random.randint(0, 2)
        move[nbin]+=1
    return(move, troops)


def Blotto_Choose(Q, lotso_troops):
    index = lotso_troops/Lotso_increments
    Move_index = Q[index].idxmax()
    Move = moves[Move_index]
    return (Move)

def Define_Winner(lotso_move, blotto_move):
    score = 0
    for field in field:
        lotso_move[field]
        blotto_move[field]



    return()

"""
Update the Q-value for a given state-action pair using the Q-learning rule/equation.
Q: Q-table storing action-value estimates (Q[s,a])
state(int): the current state s where the agent takes action
action(int): the action taken by the agent.
reward(float): the immediate reward received after taking the action.
next_state(int): the state the agent transitions to after doing an action.
alpha(float): the learning rate/step size.
gamma(float): the discount factor which determines how much futur rewards are vlaued comapred to immediate reward.
returns: void
"""
def update_Q(Q, state, action, alpha):
    current_q_value = Q[state, action] #get the current q-val
    next_state = update_pos(state, action)

    R = R_move
    if next_state in terminal_states:
        R = R_terminal


    max_future_q = np.max(Q[next_state]) #look at the possible actions in next state and select ,ax q-value.

    td_target = R + gamma * max_future_q #compute TD value
    td_error = td_target - current_q_value 
    Q[state, action_index] = current_q_value + alpha * td_error
    return(Q, next_state)


def run_episode(Q,T, experiment):
    state = init_state
    P = 0
    while True:
        T+= 1
        P+=1 
        alpha = [0.1, 1/T, 0.1, 1/T, 0.1 ,1/P][experiment]#select approprate parameters
        explore = [0.25, 1/T, 1/T, 0.1, 0.1, 0.1][experiment]#select apropriate parameters

        if np.random.random() < explore:#agent chooses to explore
            action = np.random.choice(moves)
        else: 
            max = np.max(Q[state])
            best_actions = np.where(Q[state] == max)[0]
            action_index  = np.random.choice(best_actions)
            action = moves[action_index]#be greedy
        

  
        Q, state = update_Q(Q, state,action,alpha)

        if state in terminal_states:
            break
    return (P,T, Q)


def run_simulation(experiment, n_episodes):
    sim_lengths = [[] for _ in range(n_episodes)]  # Pre-initialize with empty lists for each episode
    for sim in range(n_sims):
        
        print(experiment, sim)
        Q = np.zeros((16,4))
        T = 0
        for episode in range(n_episodes):
            P,T, Q = run_episode(Q, T, experiment)
            sim_lengths[episode].append(P)  # Dynamically append to the episode's list
            # print("\n Run number", episode, ": ", P)
            # print(Q)
    avg_len = [] 
    for episode in range(len(sim_lengths)):  # Iterate over indices
        avg_len.append(np.average(sim_lengths[episode]))  # Dynamically append average
    return(avg_len)
    
    ## need to do, add code to plot avg reinforcement and avg len for each episode



def run_experiment():
    return()
def main():
    Gamma = 1
    R = -1
    k = 0
    

    print("Better Blotto")
    # print(f"Player 2: {a2_troops} troops")

main()