---
title: "Tutorial: Structural Estimation of Dynamic Models 1"
date: 2022-06-16
image: "/img/str-est-dyn-1.png"
image1: "https://tva1.sinaimg.cn/large/e6c9d24egy1h3a8hi1llmj20h307adge.jpg"
description: This is a replication of 経済セミナー 連載「実証ビジネス・エコノミクス」第７回 with Python.
tags:
  - formal
  - tutorial
---

This is a replication of [経済セミナー 連載「実証ビジネス・エコノミクス」第７回](https://sites.google.com/view/keisemi-ebiz/%E7%AC%AC7%E5%9B%9E?authuser=0) with Python. Math details explained only on the book are omitted for copyright concerns. 

### Setup

Suppose people choose between buying and not buying a car on an infinite-horizon sequence. The price of a new car and the mileage (degradation) of an existing car change stochastically at each period. In this tutorial, we estimate parameters shaping their utility function using (simulated) empirical data of car purchasing.

```python
import numpy as np
import pandas as pd
from scipy.special import psi
from scipy.optimize import minimize, fixed_point
import matplotlib.pyplot as plt
import seaborn as sns
```

```python
theta_true = [.004, .003]
beta = .99
Euler_const = -psi(1)
n_choice = 2
```

```python
# 6 (prices) * 21 (mileage bins) = 126 states
states_p = np.linspace(2000, 2500, 6)
states_m = np.linspace(0, 100, 21)

n_m = len(states_m)
n_p = len(states_p)
n_states = n_p*n_m
```

```python
data = {
    'state_id': list(range(n_states)),
    'price_id': list(range(n_p))*n_m,
    'mileage_id': np.repeat(list(range(n_m)), n_p),
    'price': list(states_p)*n_m,
    'mileage': np.repeat(states_m, n_p)
}
states = pd.DataFrame(data)
states.head()
```

### Prepare the Transition Matrix

Let's assume true parametors and generate the transition matrix given them. 

```python
def gen_m_trans(kappa):

    kappa_1, kappa_2 = kappa[0], kappa[1]

    m_trans_nb = np.zeros((n_m, n_m))
    m_trans_nb += np.diag(np.ones(n_m)*(1 - kappa_1 - kappa_2))
    m_trans_nb += np.diag(np.ones(n_m-1)*kappa_1, 1)
    m_trans_nb += np.diag(np.ones(n_m-2)*kappa_2, 2)
    m_trans_nb[n_m-2, n_m-1] = kappa_1 + kappa_2
    m_trans_nb[n_m-1, n_m-1] = 1
    
    m_trans_b = np.dot(np.ones([n_m, 1]), m_trans_nb[0].reshape([1, n_m]))

    return np.array([m_trans_nb, m_trans_b])
```

```python
def gen_p_trans(eta):

    eta_11 = 1 - eta[0] - eta[1] - eta[2] - eta[3] - eta[4]
    eta_22 = 1 - eta[5] - eta[6] - eta[7] - eta[8] - eta[9]
    eta_33 = 1 - eta[10] - eta[11] - eta[12] - eta[13] - eta[14]
    eta_44 = 1 - eta[15] - eta[16] - eta[17] - eta[18] - eta[19]
    eta_55 = 1 - eta[20] - eta[21] - eta[22] - eta[23] - eta[24]
    eta_66 = 1 - eta[25] - eta[26] - eta[27] - eta[28] - eta[29]

    p_trans = np.array([
        eta_11, eta[0], eta[1], eta[2], eta[3], eta[4],
        eta[5], eta_22, eta[6], eta[7], eta[8], eta[9],
        eta[10], eta[11], eta_33, eta[12], eta[13], eta[14],
        eta[15], eta[16], eta[17], eta_44, eta[18], eta[19],
        eta[20], eta[21], eta[22], eta[23], eta_55, eta[24],
        eta[25], eta[26], eta[27], eta[28], eta[29], eta_66
    ]).reshape(n_p, n_p)

    return p_trans
```

```python
kappa_true = [.25, .05]
m_trans_true = gen_m_trans(kappa_true)  # 2 * 126 * 126
print(m_trans_true[0,0:5,0:5])          # 126 * 126
print(m_trans_true[1,0:5,0:5])          # 126 * 126
m_trans_true.shape
```

```python
eta_true = [
    0.1, 0.2, 0.2, 0.2, 0.2,
    0.1, 0.2, 0.2, 0.2, 0.2,
    0.1, 0.1, 0.2, 0.2, 0.1,
    0.1, 0.1, 0.2, 0.2, 0.1,
    0.05, 0.05, 0.1, 0.1, 0.2,
    0.05, 0.05, 0.1, 0.1, 0.2
]
p_trans_true = gen_p_trans(eta_true) # 6 * 6
print(p_trans_true)
```

```python
# Combine the two transition matrices
def multiply_by_elem(mat1, mat2):
    mat = [np.vstack([a * mat2 for a in mat1.T[i]]) for i in range(len(mat1.T))]
    return np.hstack(mat)
```

```python
trans_true = {
    'nb': multiply_by_elem(m_trans_true[0], p_trans_true), 
    'b': multiply_by_elem(m_trans_true[1], p_trans_true)
}
```

```python
p_trans_eigen = np.linalg.eig(p_trans_true.T)[1]
p_dist_steady = p_trans_eigen[:, 0]/sum(p_trans_eigen[:, 0])
```

### Solve the Expected Value Function

Now we have the transition matrix, with which we can derive the (expected) value function. 

```python
def inst_u(theta, states):
    theta_c = theta[0]
    theta_p = theta[1]

    U_nb = -theta_c*states['mileage']
    U_b  = -theta_p*states['price']
    U = pd.concat([U_nb, U_b], axis = 1)
    U.columns = ['nb', 'b']
    return U

def solve_EV(theta, beta, trans, states):
    
    U = inst_u(theta, states)
    EV0 = np.zeros([n_states, n_choice])
    tol = 1e-10
    error = tol + 1

    while error > tol:
        EV1_nb = Euler_const + np.dot(trans['nb'], np.log(np.sum(np.exp(U + beta*EV0), axis = 1)))
        EV1_b = Euler_const + np.dot(trans['b'], np.log(np.sum(np.exp(U + beta*EV0), axis = 1)))
        EV1 = np.hstack([EV1_nb.T, EV1_b.T]).reshape(2, 126).T
        error = np.linalg.norm(EV1 - EV0)
        EV0 = EV1
    
    EV = EV0

    return EV

EV_true = solve_EV(theta_true, beta, trans_true, states) 
```

### Conditional Choice Probability

Now, we have the true expected continuation value, $EV$. Using it, we can derive the true choice probability. Under the assumptions we have stipulated so far, the conditional choice probability can be derived as
$$
\begin{align*}
\Pr(i = 1|x,\theta) &= \Pr[u(x, 1; \theta) + \beta EV(x, 1; \theta) + \epsilon(1) \ge u(x, 0; \theta) + \beta EV(x, 0; \theta) + \epsilon(1)] \\
&= \frac{\exp[u(x, 1; \theta) + \beta EV(x, 1; \theta)]}{\exp[u(x, 1; \theta) + \beta EV(x, 1; \theta)] + \exp[u(x, 0; \theta) + \beta EV(x, 0; \theta)]}
\end{align*}
$$

```python
EV_est = solve_EV(theta_true, beta, trans_true, states)
U_true = inst_u(theta_true, states)
V_true = U_true + beta*EV_true
P_true_denom = np.vstack([np.sum(np.exp(V_true), axis = 1), np.sum(np.exp(V_true), axis = 1)]).T
P_true_num = np.exp(V_true)
P_true = np.divide(P_true_num, P_true_denom)
```

```python
result_true = pd.concat([states, P_true], axis = 1)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.lineplot(x = "mileage", y = "nb", data = result_true, hue = "price", ax = ax[0])
sns.lineplot(x = "mileage", y = "b", data = result_true, hue = "price", ax = ax[1])
fig.show()
```

![](https://tva1.sinaimg.cn/large/e6c9d24egy1h3a84r6ba2j20h307adge.jpg)

### Prepare data

Suppose we got the following empirical data of 1,000 agents' decisions and states per period. Our objective is to estimate the true parameter values with the data.

```python
data = pd.read_csv('sim_data.csv')
data['state_id'] = data['state_id'] - 1
data['price_id'] = data['price_id'] - 1
data['mileage_id'] = data['mileage_id'] - 1
data['lag_price_id'] = data.groupby('consumer')['price_id'].shift(1)
data['lag_mileage_id'] = data.groupby('consumer')['mileage_id'].shift(1)
data['lag_action'] = data.groupby('consumer')['action'].shift(1)
```

### Estimate Transition Matrices

Let the data of actions and states be denoted by $\{i_t, x_t\}_{t=1}^T = \{I, X\}$. The likelihood of the observed data given paramters can be derived as
$$
\begin{align*}
L(I, X|i_0, x_0; \theta) &= \Pi_{t=1}^T\Pr(i_t, x_t| i_{t-1},x_{t-1}; \theta) \\
&= \Pi_{t=1}^T\Pr(i_t| x_t; \theta)\Pi_{t=1}^T\Pr(x_t| i_{t-1}, x_{t-1}; \theta_x)
\end{align*}
$$
Taking the logarithm of both sides,
$$
\log L(I, X|i_0, x_0; \theta) = \sum_{t=1}^T\log\Pr(i_t| x_t; \theta) + \sum_{t=1}^T\log\Pr(x_t| i_{t-1}, x_{t-1}; \theta_x) \tag{1}
$$
The estimation takes two steps: first, we estimate the transition probabilities, $\theta_x$. We need to derive the parameter values that maximize the second term of (1). Using the optimal $\theta_x$, we then optimize $\theta$.

Luckily, the maximum likelihood estimator for $\theta_x = (\kappa_1,\kappa_2,\lambda)$ can be derived pretty straightforwardly. Note here that $1-\kappa_1-\kappa_2$, $\kappa_1$, and $\kappa_2$ respectively correspond to the probability that the mileage state stays there ($A_1$), proceeds one step ($A_2$), and proceeds two steps ($A_3$). In addition, $\kappa_1+\kappa_2$ is the probability that the state moves to the final state $m$ from state $m-1$ ($A_4$). Thus, the likelihood of observed data is given by
$$
L_m(\kappa) = \Pi_{j=1}^N (1-\kappa_1-\kappa_2)^{1\{a_j = A_1\}}\kappa_1^{1\{a_j = A_2\}}...
$$
Taking the logarithm of both sides, deriving the FoC, and solving for $\kappa$, we have the ML estimator as following. The estimator for $\lambda$ can be derived in a similar way.
$$
\begin{align*}
\hat{\kappa}_1 &= \frac{x_2(x_2+x_3+x_4)}{(x_2+x_3)(x_1+x_2+x_3+x_4)} \\
\hat{\kappa}_2 &= \frac{x_3(x_2+x_3+x_4)}{(x_2+x_3)(x_1+x_2+x_3+x_4)} \\
\hat{\lambda}_{jk} &= \frac{y_{jk}}{\sum_{k=1}^6y_{jk}}
\end{align*}
$$

```python
kappa_est = [0.2511427, 0.0500185]

eta_est = [
    0.10273973, 0.10388128, 0.19931507, 0.19874429, 0.1921233, 0.20319635,
    0.10074627, 0.09916327, 0.19222071, 0.20952058, 0.1991180, 0.19923112,
    0.10191150, 0.10086410, 0.29714585, 0.19780047, 0.2038230, 0.09845509,
    0.09703103, 0.10031847, 0.19359975, 0.30331827, 0.2055681, 0.10016437,
    0.04924386, 0.04994318, 0.09886652, 0.09840030, 0.5007722, 0.20277397,
    0.05057165, 0.05123799, 0.09826752, 0.09931963, 0.2030932, 0.49751000
]
```

```python
# generate transition matrix based on the estimated parameters
m_trans_est = gen_m_trans(kappa_est)
p_trans_est = gen_p_trans(eta_est)
trans_est = {
    'nb': multiply_by_elem(m_trans_est[0], p_trans_est),
    'b': multiply_by_elem(m_trans_est[1], p_trans_est)
}
```

### Estimate Payoff Parameters

Now, we are ready to estimate $\theta$ from the observed data. Here, we employ the NFPX (nested fixed point) algorithm, which consists in the following inner and outer loops: 

1. give an initial $\theta$
2. compute $EV$ given $\theta$ with the value function iteration
3. compute the conditional choice probability given $EV$
4. compute the likelihood, update $\theta$ following some maximization algorithm, and then go back to 2

In practice, we just need to write up a function of log-likelihood and throw it to an optimizer.

```python
def logLikelihood(theta, beta, trans, states, data):

    EV = solve_EV(theta, beta, trans, states)
    U = inst_u(theta, states)
    V = U + beta*EV
    P_denom = np.vstack([np.sum(np.exp(V), axis = 1), np.sum(np.exp(V), axis = 1)]).T
    P_num = np.exp(V)
    P = np.divide(P_num, P_denom)
        
    logL = 0
    for action, state in zip(data['action'], data['state_id']):
        if action == 0:
            logL += np.log(P['nb'][state])
        if action == 1:
            logL += np.log(P['b'][state])

    return -logL
```

```python
maxit = 1000
theta_init = theta_true

theta_est = minimize(
    logLikelihood, 
    x0 = theta_init, 
    args = (.99, trans_est, states, data),
    options = {'maxiter': maxit}, 
    method = 'nelder-mead'
).x
```

### Comparative Statics

Let us finish with checking if the estimation succeeds.

```python
EV_est = solve_EV(theta_est, beta, trans_est, states)
U_est = inst_u(theta_est, states)
V_est = U_est + beta*EV_est
P_denom = np.vstack([np.sum(np.exp(V_est), axis = 1), np.sum(np.exp(V_est), axis = 1)]).T
P_num = np.exp(V_est)
P_est = np.divide(P_num, P_denom)

result_est = pd.concat([states, P_est], axis = 1)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.lineplot(x = "mileage", y = "nb", data = result_est, hue = "price", ax = ax[0])
sns.lineplot(x = "mileage", y = "b", data = result_est, hue = "price", ax = ax[1])
fig.show()
```

![](https://tva1.sinaimg.cn/large/e6c9d24egy1h3a8hi1llmj20h307adge.jpg)