# EX_06 - SARSA Learning Algorithm

## AIM

To implement and analyze the SARSA (State-Action-Reward-State-Action) reinforcement learning algorithm for learning optimal policies in a given environment using on-policy temporal difference control.

## PROBLEM STATEMENT

In many real-world and simulated environments, an agent must learn an optimal sequence of actions to maximize cumulative rewards without prior knowledge of the environment's dynamics. The challenge lies in balancing exploration of new actions with exploitation of known rewarding strategies.

This experiment aims to solve the problem of learning an optimal policy for decision-making under uncertainty using the SARSA algorithm. SARSA is an on-policy Temporal Difference (TD) learning algorithm where the agent updates its Q-values based on the action actually taken in the next state. This approach allows the agent to improve its behavior gradually while still exploring the environment.

The goal is to:

- Implement the SARSA algorithm.

- Train the agent in a grid world or similar environment.

- Evaluate how effectively the agent learns an optimal policy over episodes.

- Compare performance metrics such as total reward per episode and convergence behavior.

## SARSA LEARNING ALGORITHM

![image](https://github.com/user-attachments/assets/59a1e43c-abc5-4440-a4fd-79eaf6e13b60)


## SARSA LEARNING FUNCTION
```
DEVELOPED BY : NIRAUNJANA GAYATHRI G R
REGISTER NO. : 212222230096
```
```
def sarsa(env,
          gamma=1.0,
          init_alpha=0.5,
          min_alpha=0.01,
          alpha_decay_ratio=0.5,
          init_epsilon=1.0,
          min_epsilon=0.1,
          epsilon_decay_ratio=0.9,
          n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    # Decay schedules for alpha and epsilon
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    # Policy based on epsilon-greedy
    select_action = lambda state, Q, epsilon: np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(nA)

    for e in tqdm(range(n_episodes), leave=False):
        state = env.reset()
        action = select_action(state, Q, epsilons[e])
        done = False
        
        while not done:
            next_state, reward, done, _ = env.step(action)
            
            # Choose next action using epsilon-greedy policy
            next_action = select_action(next_state, Q, epsilons[e])
            
            # SARSA update
            td_target = reward + gamma * Q[next_state][next_action] * (not done)
            td_delta = td_target - Q[state][action]
            Q[state][action] += alphas[e] * td_delta
            
            state = next_state
            action = next_action

        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))

    # Calculate V and pi from the final Q
    V = np.max(Q, axis=1)
    pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return Q, V, pi, Q_track, pi_track
```

## OUTPUT:

### Mention the optimal policy, optimal value function , success rate for the optimal policy.

![image](https://github.com/user-attachments/assets/de6233a8-b9d7-4fee-9cbd-9060d5b00f31)


![image](https://github.com/user-attachments/assets/025ffe90-9a13-422b-9037-c2e39c4f1407)


![image](https://github.com/user-attachments/assets/48305d6d-86c4-469e-bceb-a5a92ac5cea8)


![image](https://github.com/user-attachments/assets/df48f519-6e46-4a64-bfe2-a0defde08cea)


![image](https://github.com/user-attachments/assets/0eef8ea3-ed84-4da2-b7ba-0c28ab21e516)



![image](https://github.com/user-attachments/assets/5bd60eab-e5a2-4ff6-b172-db8186bf9b5f)


### Include plot comparing the state value functions of Monte Carlo method and SARSA learning.


![image](https://github.com/user-attachments/assets/ffbfb51a-b791-4635-b43b-d54d6c21e946)


![image](https://github.com/user-attachments/assets/2d55f92a-6c39-4890-8af8-173f4db7b21e)


## RESULT:

Thus, to implement and analyze the SARSA (State-Action-Reward-State-Action) reinforcement learning algorithm for learning optimal policies in a given environment using on-policy temporal difference control is successfully implemented. 
