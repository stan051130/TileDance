import numpy as np

def greedy_selection(Q):
    max_action = Q.argmax()
    return max_action

def epsilon_greedy_selection(Q, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(len(Q))
    else:
        return Q.argmax()

def sample_average_update(Q, N, chosen_action, reward):
    N[chosen_action] += 1
    Q[chosen_action] = Q[chosen_action] + 1/N[chosen_action]*(reward - Q[chosen_action])


def main():
    k = 10
    
    #environment
    true_values = np.random.normal(0,1,k)

    #What agent knows
    Q = np.zeros(k)
    N = np.zeros(k)
    
    for t in range(1000):
        chosen_action = epsilon_greedy_selection(Q, 0.1)
        reward = np.random.normal(true_values[chosen_action], 1)
        sample_average_update(Q, N, chosen_action, reward)
        
    print("what agent knows:", Q)
    print("true values: ", true_values)
    print("counts: ", N)
    
if __name__ == "__main__":
    main()