---
title: "Tutorial: Structural Estimation of Dynamic Models 2"
date: 2022-06-19
image: "/img/str-est-dyn-2.png"
image1: "https://tva1.sinaimg.cn/large/e6c9d24egy1h3cy2ehlhsj20aw07aaa7.jpg"
description: This is a replication of 経済セミナー 連載「実証ビジネス・エコノミクス」第8回 with Python.
tags:
  - formal
  - tutorial
---

This is a replication of https://www.nippyo.co.jp/blogkeisemi/wp-content/uploads/sites/5/2022/05/Chap8_220524.html with Python. Math details shown only in the paper are omitted for copyright concerns. 

Hereunder, I assume that you have already run all codes from the previous session.

In this session, we work on the same problem as we saw before, using different, more efficient estimation methods. The motivation is to avoid the heavy process of value function iteration, which gets even unrealistic as the dimention of state variables increases (curse of dimentionality). 

We overcome the challenge using an alternative way to compute the ex-ante value function, $V$, and estimate the choice probability without computing $EV$.

To do that, we take the following three steps:

1. approximate the CCP based on the data (this time we use a logit model)
2. using the approximated CCP, get the (ex-ante) value, $V$
3. using $V$, estimate the CCP and find $\theta$ that maximizes the quasi likelihood.

### Setup

```python
import statsmodels.api as sm
```

### Estimate the CCP with a Logit Model

To begin, we estimate the conditional choice probability based on the data. Here, we simply estimate a logit model and predict the probability that agents buy a car given each state.

```python
data['price2'] = data['price']**2
data['mileage2'] = data['mileage']**2

covs  = ['price', 'price2', 'mileage', 'mileage2']
X = data[covs]
y = data['action']

logit_mod = sm.Logit(y, X)
logit_res = logit_mod.fit(disp = 0)
print('Parameters: ', logit_res.params)
print(logit_res.summary())
```

```python
states['price2'] = states['price']**2
states['mileage2'] = states['mileage']**2

ypred = logit_res.predict(states[covs])
CCP_init = np.array([np.array([1 - ypred]).T, np.array([ypred]).T])
```

### Alternative Method: Matrix Inversion

See the book for math details.

```python
G = np.array([np.array(trans_est['nb']), np.array(trans_est['b'])])

def get_CCP_matInv(theta, CCP, beta, G, states):

    U = np.array([np.array([inst_u(theta, states)['nb']]).T, np.array([inst_u(theta, states)['b']]).T])
    psi = Euler_const*np.ones([n_choice, n_states, 1]) - np.log(np.array(CCP))

    V = np.dot( \
        np.linalg.inv( \
            np.diag(np.ones(n_states)) - beta*( \
                np.dot(CCP[0], [np.ones(n_states)])*G[0] + \
                    np.dot(CCP[1], [np.ones(n_states)])*G[1]
        )), \
        CCP[0]*(U[0] + psi[0]) + CCP[1]*(U[1] + psi[1])
    )

    v = np.array([U[0] + np.dot(beta*G[0], V), U[1] + np.dot(beta*G[1], V)])
    return np.array(np.exp(v)/(np.exp(v[0]) + np.exp(v[1])))

def logLikelihood2(theta, CCP, beta, G, states, data):

    CCP = get_CCP_matInv(theta, CCP, beta, G, states)
        
    logL = 0
    for action, state in zip(data['action'], data['state_id']):
        if action == 0:
            logL += np.log(CCP[0][state])
        if action == 1:
            logL += np.log(CCP[1][state])

    return -logL

maxit = 1000
theta_init = [0.01, 0.01]

theta_est_matInv = minimize(
    logLikelihood2, 
    x0 = theta_init, 
    args = (CCP_init, .99, G, states, data),
    options = {'maxiter': maxit}, 
    method = 'nelder-mead'
).x
# trans is not used in the matrix inversion method
```

### Counterfactual Analysis 1: Fixed v. Random Price

Using the fitted model, let's exercise some simulation. Suppose we, a car seller, must choose between offering random prices (baseline) and setting some fixed price. With the estimated CCP (that tells us how likely people buy cars given each state), we can now simulate how our expected payoff varies by the two scenarios. 

```python
# slightly change the NFXP function

def solve_EV2(theta, beta, trans, states):
    
    n_states2 = len(states)
    U = inst_u(theta, states)
    EV0 = np.zeros([n_states2, n_choice])
    tol = 1e-10
    error = tol + 1

    while error > tol:
        EV1_nb = Euler_const + np.dot(trans['nb'], np.log(np.sum(np.exp(U + beta*EV0), axis = 1)))
        EV1_b = Euler_const + np.dot(trans['b'], np.log(np.sum(np.exp(U + beta*EV0), axis = 1)))
        EV1 = np.hstack([EV1_nb.T, EV1_b.T]).reshape(2, n_states2).T
        error = np.linalg.norm(EV1 - EV0)
        EV0 = EV1
    
    EV = EV0

    return EV


def get_CCP_nfxp(theta, beta, trans, states):
    
    EV = solve_EV2(theta, beta, trans, states)
    U = inst_u(theta, states)
    V = U + beta*EV
    CCP_denom = np.vstack([np.sum(np.exp(V), axis = 1), np.sum(np.exp(V), axis = 1)]).T
    CCP_num = np.exp(V)
    CCP = CCP_num/CCP_denom

    return np.array([np.array([CCP['nb']]).T, np.array([CCP['b']]).T])
```

```python
# preparation for Scenario 2  

# in Scenario 2, the transition matrix is 21 * 21 since the price is fixed
trans_fixed = {
    'nb': m_trans_est[0],
    'b': m_trans_est[1]
}

CCPs = {}
CCPs['baseline'] = get_CCP_nfxp(theta_est, beta, trans_est, states)

for price in states_p:
    states_fixed = states.query("price == " + str(price))
    CCPs['fixed_' + str(price)] = get_CCP_nfxp(theta_est, beta, trans_fixed, states_fixed)

# %%
# Scenario 1
data2 = data.groupby(['state_id', 'price_id', 'price']).size().reset_index()
data2 = data2.rename({0: 'n'}, axis = 1)
data2 = pd.concat([data2, pd.DataFrame(CCPs['baseline'][1])], axis = 1)
data2 = data2.rename({0: 'pbuy_baseline'}, axis = 1).eval('prod = n*pbuy_baseline')
data2 = data2.groupby(['price_id', 'price']).sum().eval('w_mean = prod/n').reset_index()
data2

# %%
# Scenario 2
data_fixed = data.groupby(['mileage_id', 'mileage']).size().reset_index()
data_fixed = data_fixed.rename({0: 'n'}, axis = 1)
data_fixed = pd.concat([data_fixed, pd.DataFrame(CCPs['fixed_2000.0'][1])], axis = 1)
data_fixed = data_fixed.rename({0: 'pbuy_fixed2000'}, axis = 1).eval('prod = n*pbuy_fixed2000')
prob_buy_fixed2000 = sum(data_fixed['prod'])/sum(data_fixed['n'])
```

The graph below compares the two scenarios, showing that randomizing the price increases the probability of buying when the price is low, which is because agents expect the price will rise in the future.

```python
# comparison
plot_fixed = sns.barplot(x = "price", y = "w_mean", data = data2)
plot_fixed.axhline(prob_buy_fixed2000)
```
![](https://tva1.sinaimg.cn/large/e6c9d24egy1h3cy2dvmdrj20aw07agln.jpg)

### Counterfactual Analysis 2: Permanent v. Temporary Pricedown

```python
# Scenario 1: Permanent Pricedown
states_discount = states.eval('price = price - 100')
CCPs['permanent_discount'] = get_CCP_nfxp(theta_est, beta, trans_est, states_discount)

# Scenario 2: Temporary Pricedown
U_discount = inst_u(theta_est, states_discount).to_numpy()
EV = solve_EV2(theta_est, beta, trans_est, states)
V_temp = U_discount + beta*EV
CPP_temp_denom = np.vstack([np.sum(np.exp(V_temp), axis = 1), np.sum(np.exp(V_temp), axis = 1)]).T
CPP_temp_num = np.exp(V_temp)
CPP_temp = CPP_temp_num/CPP_temp_denom
CCPs['temporary_discount'] = np.array([np.array([CPP_temp]).T[0], np.array([CPP_temp]).T[1]])
```

```python
data_discount = states
data_discount = pd.concat([data_discount, pd.DataFrame(CCPs['permanent_discount'][1])], axis = 1)
data_discount = data_discount.rename({0: 'pb_permanent'}, axis = 1)
data_discount = pd.concat([data_discount, pd.DataFrame(CCPs['temporary_discount'][1])], axis = 1)
data_discount = data_discount.rename({0: 'pb_temporary'}, axis = 1)
data_discount = pd.concat([data_discount, pd.DataFrame(CCPs['baseline'][1])], axis = 1)
data_discount = data_discount.rename({0: 'pb_baseline'}, axis = 1)

data_discount = data_discount.query('price == 2200')
data_discount = pd.melt(data_discount, id_vars = 'mileage', value_vars = ['pb_permanent', 'pb_temporary', 'pb_baseline', 'mileage'])
data_discount
```
```python
sns.lineplot(x = 'mileage', y = 'value', hue = 'variable', data = data_discount)
```
![](https://tva1.sinaimg.cn/large/e6c9d24egy1h3cy2ehlhsj20aw07aaa7.jpg)