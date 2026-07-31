---
layout: post
title:  "Intro to Reinforcement Learning"
date:   2026-07-30
categories: [articial_intelligence]
mathjax: true
---

These are my notes from studying Reinforcement Learning. Much of this material comes from Sutton and Barto's textbook on Reinforcement Learning.

### Setup
RL Loop:
- Agent takes action $A_t = a \in \cal A(s)$. Agent observes the current state of the environment. Agent's action depends on it.
- Environment gives reward $R_{t+1} = r \in \cal R \subseteq \mathbb R$ and changes state to $S_{t+1} = s' \in \cal S$.
- We assume that the agent can fully observe the state of the environment. In real world, environment state might only be partially observable.

### Finite Markov Decision Process
- **Finite**: The state space $\cal S$, action space $\cal A$ and the rewards space $\cal R$ are all finite.
- **Markov**: Environment's state and reward only depends on the current state and action of the agent. This is expressed as a probability distribution: $p(s', r \| s, a)$
- Markov property is a strong assumption. It says that the current state encodes enough information to govern all future possibilities that the environment can end up in. The history of how the environment reached the current state is not important.

### Episode
- Environment will have a starting distribution of states.
- Environment will have a terminal state that marks the end of an episode.
- A sequence of state, action, rewards in an episode starting from some starting state and ending in a terminal state is called a **Trajectory**: $(s_0^m,a_0^m,r_1^m,s_1^m,a_1^m,\ldots,r_T^m,s_T^m)$ denotes the $m^{th}$ episode.

### Policy
- How the agent behaves under a given environment state is controlled by a probability distribution called policy: $\pi(a\|s)$. 
- Policy can be deterministic: $a = \pi(s)$.
- A good policy should ideally accumulate a lot of future rewards, called **Return**: $G_t = \sum_{k = t+1}^{T} \gamma^{k-t-1}R_k$. Future rewards are discounted.
- Goal is to learn a policy $\pi$ that maximizes $\mathbb E_{\pi}[G_t]$
- Finding the optimal policy is also called the **Control Problem**.

### Value Functions
$G_t$ can be broken down further:
- Per state: State Value function:
  $v_{\pi}(s)=\mathbb E_{\pi}[G_t\|S_t=s]$
- Per state, action pair: Action Value function:
  $q_{\pi}(s,a) = \mathbb E_{\pi}[G_t\|S_t=s, A_t=a]$

There exists a optimal policy $\pi_{*}$ such that:
- $v_{\pi_{*}}(s) \geq v_{\pi}(s)$ for all $s$ and any $\pi$
- $q_{\pi_{*}}(s,a) \geq q_{\pi}(s,a)$ for all $s,a$ and any $\pi$

Knowing these value functions for the optimal policy is enough to know the optimal policy. In each state, we only need to move to the state that maximizes the value.

### Model of the Environment
When we do not have complete knowledge of the environment: $p(s',r\|s,a)$, we have two options:
- **Model-based Methods:** First estimate the environment model by estimating $p(s',r\|s,a)$ by interacting with the environment. We can then plan agent actions. 
- **Model-free Methods:** We directly learn a policy by interacting with the environment, without trying to estimate the environment.
Finding the optimal policy, given the model of the environment is also called the **Planning Problem**. Finding the optimal policy, is also called the **Control Problem** in general. 

### Bellman Equations
For any policy $\pi$, value functions $v_{\pi}$ and $q_{\pi}$ follow a recurrence relation:

$$v_{\pi}(s)=\sum_{a \in \cal A(s)} \pi(a|s)q_{\pi}(s,a)$$

$$q_{\pi}(s,a)=\sum_{s' \in \cal S, r \in \cal R} p(s',r|s,a)[r + \gamma v_{\pi}(s')]$$

- Using these two equations, we can build pure recurrence relation in $v_\pi$ or $q_\pi$.
- If we have **complete knowledge** of the environment, that is, we know the state transition probabilities: $p(s',r\|s,a)$, then we can use these recurrence relations to solve for the value functions for a given policy $\pi$.

When following the optimal policy $\pi_{*}$, the Bellman equations become:

$$v_{\pi}(s)=\max_{a \in \cal A(s)} q_{*}(s,a)$$

$$q_{\pi}(s,a)=\sum_{s' \in \cal S, r \in \cal R} p(s',r|s,a)[r + \gamma v_{*}(s')]$$

### Policy Iteration
**Policy Evaluation**: Gets value function, given the policy. 
- Given a policy $\pi$, start with an estimate of the state value function:  $V(s)$
- Using the Bellman Equations for state value, update the value function for each state. Do this for all states. This is called a sweep.
- Run sweeps till value functions converge. 

**Policy Improvement**: Gets a better policy, given the value function.
- Given the value function $v_\pi$, update the policy to pick action that maximizes  $q_\pi(s, a)$ for a given state $s$.
- This will always lead to a strictly better policy, except when $\pi = \pi_{*}$.

**Policy Iteration**: Policy Evaluation and Improvement can be run back to back to get the optimal policy.
- There are variants of this approach. For example, **Value Iteration** uses only a single sweep of evaluation before proceeding to get a better a policy.
- It can be generalized to **Generalized Policy Iteration**, where evaluation can be done at any granularity before proceeding to improvement. 
- All Reinforcement Learning methods have some flavor of GPI.

### Monte-Carlo Methods
- Monte-Carlo methods uses averages from observations to estimate expectations of probability distributions. The observations are the trajectories.
- MC methods for RL fall within Model-free methods. 
- What to estimate: $V(s)$ or $Q(s,a)$:
	- Policy improvement always needs $Q(s,a)$. 
	- $Q(s,a)$ is more granular than $V(s)$. To go from $Q(s,a) \to V(s)$, we only need policy $\pi(a\|s)$.
	- But going the other way from $V(s) \to Q(s,a)$ requires knowing the model of the environment: $p(s',r\|s,a)$.
- So, for simplicity, MC methods always try learning only $Q(s,a)$.

### Monte-Carlo Evaluation
- Estimating $Q(s,a)$ of the Markov Decision Process: $(s_0^m,a_0^m,r_1^m,s_1^m,a_1^m,\ldots,r_T^m,s_T^m)$ is equivalent to 
- Estimating $V(s)$ of the Markov Reward Process $(s_0^m,r_1^m,s_1^m,\ldots,r_T^m,s_T^m)$ where the state $s$ of the MRP represents the state-action pair $(s,a)$ of the MDP.

The estimation looks like:

$$V(s) = \mathbb E[G_t \|S_t = s] \approx \frac{1}{C(s)}\sum_{m=1}^{M}\sum_{\tau=0}^{T_m-1} \mathbb 1[s_\tau^m = s]g_\tau^m$$

There is also an iterative way to estimate this. After every episode, we know the returns for each state at time $t$ within the episode. Apply the update rule after the $m^{th}$ trajectory:

$$V(s_t^m) \leftarrow V(s_t^m) + \frac{1}{C(s_t^m)}(g_t^m - V(s_t^m))$$

There is a variant of this update rule called **constant-$\alpha$ MC** where $\alpha$ is the step-size:

$$V(s_t^m) \leftarrow V(s_t^m) + \alpha(g_t^m - V(s_t^m))$$

Smaller $\alpha$ is slow and accurate. Larger $\alpha$ is quick and noisy.

### Monte-Carlo Policy Improvement
**Explore-Exploit Tradeoff:** For discovering the best policy, we need to explore all state-action pairs. But when policies are always chosen to get high returns, we must exploit known high values, without exploring other possible states.

For best results, policy has to be soft. $\pi(a\|s)>0$ for all $s, a$. With infinite data, $\pi_{*}$ is always discoverable by GPI for such soft policies. 

 It is better to have policies explore more in the beginning and exploit more later. A simple approach to ensure some explore always:
- **$\epsilon$-greedy(Q)**: With probability $\epsilon$ pick an action uniformly at random and with probability $(1-\epsilon)$, pick $\text{argmax}_aQ(s,a)$.

### Off Policy Methods
Policy plays two roles in MC methods:
- **Behavior**: Sampling data for estimating $Q(s,a)$.
- **Target**: Running Policy Iteration by evaluating $Q(s,a)$ and improving the policy towards the optimal policy. 

In Off-Policy methods, these two policies are kept different. For estimating returns under a behavior policy $b$, we need to scale the returns using:

$$\rho_t = \prod_{\tau=t+1}^{T-1}\frac{\pi(A_{\tau}|S_{\tau})}{b(A_{\tau}|S_{\tau})}$$

This imposes a constraint: The behavior policy must cover everything that the target policy covers:

$$\pi(a|s) > 0 \implies b(a|s) > 0$$

Advantages:
- Can use a static dataset collected using some behavior policy to learn a different target policy later. 
- Behavior policy takes care of exploration. The target policy need not be a soft policy (like $\epsilon$-greedy(Q)) to take care of exploration. The target can use pure $\text{argmax}_a Q(s,a)$.

Disadvantages:
- The scaling factor means that the effective step size is no longer the same for all state-action pairs. This can create unstable noisy updates.

### Temporal Difference
- Monte Carlo is inefficient. It waits till the end of the episode so that we know the full return, used in the updates. 
- TD method uses the value function of the next state as a substitute for upcoming rewards. This is called **Bootstrapping**.
- Instead of using the value function of the very next state, we can observe the next few states and their rewards and then substitute upcoming rewards with the corresponding value function. 
- Monte-Carlo methods learn from experience. Dynamic Programming learns from bootstrapping when environment is known. TD Learning combines Monte-Carlo's learning from experience and DP's bootstrapping. 
- TD is more efficient by utilizing the Markov property of the underlying MRP.

**n-Step TD:** We start with the update rule of $\alpha$-MC and modify the return term $g_t^m$:

$$V(s_t^m) \leftarrow V(s_t^m) + \alpha(g_t^m - V(s_t^m))$$

where

$$g_t^m = r_{t+1}^m + \gamma r_{t+2}^m + \gamma^2 r_{t+3}^m \ldots + \gamma^{n-1} r_{t+n}^m + \gamma^nV(s_{t+n}^m)$$

- **Tradeoff between MC and TD:**
	- TD with high $n$ is equivalent to MC.
	- TD with very low $n$, like $n=1$, needs a higher $\alpha$.

### SARSA, Q-Learning and Expected SARSA
The On-Policy version of the TD algorithm is called **n-step SARSA**.

$$Q(s_t^m,a_t^m) \leftarrow Q(s_t^m,a_t^m) + \alpha(g_{t:t+n}^m - Q(s_t^m,a_t^m))$$

$$g_{t:t+n}^m = r_{t+1}^m + \gamma r_{t+2}^m + \gamma^2 r_{t+3}^m \ldots + \gamma^{n-1} r_{t+n}^m + \gamma^nQ(s_{t+n}^m, a_{t+n}^m)$$

**Modification 1:** If we modify this update rule, by replacing:

$$\gamma^n Q(s_{t+n}^m, a_{t+n}^m)$$

with:

$$\gamma^n \max_a Q(s_{t+n}^m, a)$$

we get **Q-Learning**. 
- The Q values are now updated using a slightly different policy from the one that is generating the data. This makes it a **Off-Policy method**.
- The update can be triggered before the action $a_{t+n}^m$ is issued.

**Modification 2:** If we modify this update rule, by replacing:

$$\gamma^n Q(s_{t+n}^m, a_{t+n}^m)$$

with:

$$\gamma^n \sum_a Q(s_{t+n}^m, a)\pi(a|s_{t+n}^m)$$

we get **Expected SARSA**, an on-policy method. It has a less noisy update, but at a higher computational cost.

### Functional Approximation
- In simple cases, we can list a table of all states, actions. We call this the **Tabular Case**. 
- In large state spaces, we need too much memory, compute and data for RL training.
- States are represented as feature vectors $s$. Value function takes the state feature vectors and gives a value, using something like a neural network with parameters that can be learnt: $\hat{v}(s,w)$
- But this creates a new problem, whenever we try to update the value function from one state, it affects all states.

The best $w$ is the one that minimizes:

$$\overline{VE}(w) = \sum_{s}\mu(s)[v_{\pi}(s) - \hat{v}(s,w)]^2$$

where $\mu(s)$ is the probability distribution of visiting the respective states. 
- But we do not know the true value function $v_{\pi}(s)$. 
- We get to observe a surrogate target $U_t$ instead
- And we learn the parameter $w$ using SGD, like in any supervised learning.

For the target to work fine (optimizing with the target is same as optimizing with value function), we need it to be unbiased:
$$\mathbb E[U_t |S_t=s] = v_{\pi}(s)$$

### Gradient Monte Carlo

$$U_t = G_t$$

### Semi-Gradient TD

$$U_t = R_t + \gamma \hat{v}(S_{t+1}, w)$$

But:
- This is a biased target. 
- The update rule is not a true gradient step, since $U_t$ depends on $w$. So there is no guarantee of convergence. But it still somehow works.

### Off Policy Methods with Function Approximation
- On-Policy methods work fine
- But, Off-Policy runs into problems of divergence called the Deadly Triad:
	- Bootstrapping
	- Off Policy
	- Function Approximation

### Policy Gradient Theorem
Suppose we parameterize the policy: $\pi_\theta(a|s)$. 

We need to maximize over trajectories $\tau$:

$$J(\theta) = \mathbb E_{\tau \sim \pi_\theta} [G(\tau)] = \int \pi_\theta(\tau)G(\tau)d\tau$$

$$\nabla_\theta J(\theta) = \nabla_\theta\int \pi_\theta(\tau)G(\tau)d\tau = \int [\nabla_\theta\pi_\theta(\tau)]G(\tau)d\tau$$

To simplify this further, we use the log-derivative trick:

$$\nabla \log f(x) = \frac{1}{f(x)}\nabla f(x)$$

$$\nabla_\theta J(\theta) = \int [\nabla_\theta\pi_\theta(\tau)]G(\tau)d\tau = \int [\nabla_\theta\log\pi_\theta(\tau)]\pi_\theta(\tau)G(\tau)d\tau$$

This helps rephrase the gradient as an expectation:

$$
\begin{align*}
J(\theta) &= \mathbb E_{\tau\sim\pi_\theta} [G(\tau)]\\
\nabla_\theta J(\theta) &= \mathbb E_{\tau\sim\pi_\theta} [G(\tau)\nabla_\theta\log\pi_\theta(\tau)]
\end{align*}
$$


### REINFORCE Algorithm
For each trajectory, perform gradient ascent of $\pi_\theta$ using $G(\tau)\nabla_\theta\log\pi_\theta(\tau)$.

The trajectory probability can be expressed as:

$$\pi_\theta(\tau) = p(s_0) \hspace{5mm} \pi_\theta(a_0|s_0) p(s_1, r_1|s_0, a_0) \hspace{5mm} \pi_\theta(a_1|s_1) p(s_2, r_2|s_1, a_1) \hspace{5mm} \ldots \hspace{5mm} \pi_\theta(a_{T-1}|s_{T-1}) p(s_T, r_T|s_{T-1}, a_{T-1})$$

Only the $\pi_\theta(a_t \| s_t)$ terms are dependent on $\theta$. 

$$\nabla_\theta\pi_\theta(\tau) = \sum_{t=0}^{T-1}   \nabla_\theta \log\pi_\theta(a_t|s_t)  $$

$$\nabla_\theta J(\theta) = \mathbb E_{\tau\sim\pi_\theta} [G(\tau)\nabla_\theta\log\pi_\theta(\tau)] = \sum_{t=0}^{T-1}   \mathbb E_{\tau\sim\pi_\theta}[G(\tau)\nabla_\theta \log\pi_\theta(a_t|s_t)]$$

Here is the update rule:

$$\theta \leftarrow \theta + \alpha \sum_{t=0}^{T-1} G(\tau)  \nabla_\theta \log\pi_\theta(a_t|s_t) $$

This is an episodic update algorithm. The update is applied only after the total return $G(\tau)$ of the trajectory is known. This update rule can further be tightened by replacing $G(\tau)$ with rewards from time $t$ onwards.

### REINFORCE Algorithm: Reward To Go
$G(\tau)$ can be broken down into past $B_t$ and future $G_t$ rewards:

$$G(\tau) = \sum_{i=0}^{t-1}\gamma^ir_{i+1} + \gamma^t\sum_{i=t}^{T-1}\gamma^{i-t}r_{i+1} = B_t + \gamma^t G_t$$

$$\mathbb E[G(\tau)\nabla_\theta \log\pi_\theta(a_t|s_t)] = \mathbb E[B_t\nabla_\theta \log\pi_\theta(a_t|s_t)] + \mathbb E[\gamma^tG_t\nabla_\theta \log\pi_\theta(a_t|s_t)]$$

We can show that the first term is $0$. Let the trajectory history till time $t$ be $H_t$. We can break the expectation term as two nested expectations, first given the trajectory $H_t$ what the expectations turns out to be and then over all such $H_t$: 

$$\mathbb E_{\tau\sim\pi_\theta}[B_t\nabla_\theta \log\pi_\theta(a_t|s_t)] = \mathbb E_{H_t}[\mathbb E_{a_t\sim\pi_\theta(.|s_t)}[B_t\nabla_\theta \log\pi_\theta(a_t|s_t) | H_t]]$$

The inner term can be shown to be 0:

$$
\begin{align*}
\mathbb E_{a_t\sim\pi_\theta(.|s_t)}[B_t\nabla_\theta \log\pi_\theta(a_t|s_t) | H_t] 
&= B_t\mathbb E_{a_t\sim\pi_\theta(.|s_t)}[\nabla_\theta \log\pi_\theta(a_t|s_t) | H_t]\\
&= B_t\mathbb E_{a_t\sim\pi_\theta(.|s_t)}[\nabla_\theta \log\pi_\theta(a_t|s_t) | s_t] &&\text{($B_t$ is constant given $H_t$)}\\
&= B_t\nabla_\theta \mathbb E_{a_t\sim\pi_\theta(.|s_t)}[1 | s_t] &&\text{(Log-derivative Trick)}\\
&= B_t \nabla_\theta 1\\
&= 0\\
\end{align*}
$$

This means:

$$\nabla_\theta J(\theta) = \sum_{t=0}^{T-1}   \mathbb E_{\tau\sim\pi_\theta}[G(\tau)\nabla_\theta \log\pi_\theta(a_t|s_t)] = \sum_{t=0}^{T-1}   \mathbb E_{\tau\sim\pi_\theta}[\gamma^tG_t\nabla_\theta \log\pi_\theta(a_t|s_t)]$$

Here is the corresponding update rule:

$$\theta \leftarrow \theta + \alpha \sum_{t=0}^{T-1} \gamma^tG_t  \nabla_\theta \log\pi_\theta(a_t|s_t) $$

This update rule can be applied at a step-level (after each action) instead of episode level if we replace $G_t$ with a boot-strapped estimate.

$$\theta \leftarrow \theta + \alpha  \gamma^tG_t  \nabla_\theta \log\pi_\theta(a_t|s_t) $$

The REINFORCE algorithm has one crucial issue. The total return gets smeared across the entire trajectory. It is unclear which action within the trajectory was the most crucial. This makes it:
- high variance
- sample inefficient

### Actor-Critic Methods

$$\nabla_\theta J(\theta) =    \mathbb E_{\tau\sim\pi_\theta}[\sum_{t=0}^{T-1}\gamma^tG_t\nabla_\theta \log\pi_\theta(a_t|s_t)]$$

Redoing the above derivation, we can replace $G_t$ with $G_t - b_t$ where $b_t$ is called the baseline and is purely a function of $H_t$. We generally need the baseline to be only a function of $s_t$ with $b_t = b(s_t)$. 

$$\nabla_\theta J(\theta) =    \mathbb E_{\tau\sim\pi_\theta}[\sum_{t=0}^{T-1}\gamma^t(G_t-b(S_t))\nabla_\theta \log\pi_\theta(a_t|s_t)]$$

If we choose the baseline $b(S_t) = V(S_t)$, then the update rule reduces any update to the state by the expected returns from that state. The difference $G_t - b(S_t)$ is called the **Advantage**. It reduces the variance of the REINFORCE update.

A **Critic** is any system that learns the value function (using Monte-Carlo or TD) to be used as a baseline. The **Actor** is any system that learns the optimal policy. Together they can be used to make REINFORCE style updates called Actor-Critic Methods.

Actor-Critic Methods are often used with function approximation and bootstrapping, so that it can be truly a step-level update without having to wait for the entire $G_t$. Here is an example:

$$
\begin{align*}
\text{1-Step TD (also the Advantage)} \hspace{2mm}:&\hspace{2mm} \delta_t = [R_{t+1} + \gamma V_w(S_{t+1})] - V_w(S_{t})\\
\text{Critic} \hspace{2mm}:&\hspace{2mm}  w \leftarrow w + \beta \delta_t \nabla_w V_w(S_t)\\
\text{Actor} \hspace{2mm}:&\hspace{2mm} \theta \leftarrow w + \alpha \gamma^t\delta_t \nabla_\theta \log \pi_\theta (A_t|S_t)\\
\end{align*}
$$

Actor-Critic methods can sometimes make large updates to the policy. This affects how data gets sampled in subsequent steps. To avoid this, extra regularization terms (like using KL diverge between current policy and updated policy, some clipping mechanisms to update etc.) are used to keep the updates small. There are many algorithms of this nature. Example: TRPO, PPO.
