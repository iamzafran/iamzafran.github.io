---
title:  "Modelling Short Time-Series with Prior Information via Tensorflow-Probability"
mathjax: true
layout: post
categories: media
image: title.svg
---

![Short Time-Series](https://raw.githubusercontent.com/iamzafran/iamzafran.github.io/master/Tutorials/Prior%20Knowledge%20Time%20Series/title.svg)


## Motivation

Imagine you're tasked with modelling daily bike sales in New York City but you're provided with only few samples. Let's say with only 100 samples. This is challenging as the lack of samples makes it difficult for a time-series model to extract the seasonality and detect trends in the time-series to predict future observations. As pointed out by [this post](https://minimizeregret.com/short-time-series-prior-knowledge), collecting more data may not necessarily be the easiest solution as seasonality can occur far in between. The common approach is to model a time series data after observing two periods of seasonality. However, if the time between the two seasons is very long, such that it occurs only once a year or longer, this makes collecting more data impractical.

But all hope is not lost. You may assume that bike sales might be positively correlated with the temperature in the city. Meaning that the hotter the temperature, the more bike is sold. We can now, encode this information to the model, so that the model expects more bike sales when a higher temperature is observed.

So here is the idea:

**Step 1** First, we're going to model the temperature data as a [Fourier Series](https://www.educative.io/answers/how-to-implement-fourier-series-in-python) by learning the Fourier coefficients. 
**Step 2** Then, use the Fourier coefficients from the temperature model to construct the New York City bike sales model.

So let's get to it.

## Imports

{% highlight python %}

import numpy as np
import pandas as pd
import pyreadr

import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az

plt.style.use("bmh")
plt.rcParams["figure.figsize"] = [12, 6]
plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.facecolor"] = "white"

import tensorflow_probability as tfp
import tensorflow as tf

tfd = tfp.distributions
tfb = tfp.bijectors

{% endhighlight %}

## Data Preprocessing

The bike sales and temperature data is obtained from my [github](https://github.com/iamzafran/iamzafran.github.io/tree/master/Tutorials/Prior%20Knowledge%20Time%20Series) repository.

The data pre-processing is fairly straightforward by joining the New York City bike sales data with the New York temperature data joined with its date.

Load the sales dataset.
{% highlight python %}

sale_df = pyreadr.read_r('citi_bike_360.Rds')
sale_df = sale_df[None]
sale_df['date'] = pd.to_datetime(sale_df['date']).dt.date

{% endhighlight %}

Load the temperature dataset.

{% highlight python %}

temperature_df = pd.read_csv('temperature.csv', parse_dates=['datetime'])

#take only the temperature data from new york
new_york_temp = temperature_df[['datetime', 'New York']]
new_york_temp['date'] = new_york_temp['datetime'].dt.date

#take only one temperature sample from each date
new_york_temp = new_york_temp.drop_duplicates(subset=['date'])
new_york_temp = new_york_temp.dropna()

{% endhighlight %}

Join the sales and temperature data together based on their date.

{% highlight python %}

#select only the required colums
new_york_temp = new_york_temp[['date', 'New York']]
sale_df = sale_df[['date', 'rides']]

#join the sales and temperature data based on the date
raw_df = pd.merge(sale_df, new_york_temp, how='left', on='date')

start_date = pd.to_datetime("2013-07-01")
train_test_date = pd.to_datetime("'2013-10-15'")

raw_df['date'] = pd.to_datetime(raw_df['date'])
raw_df[['date', 'sales', 'temp']] = raw_df
raw_df = raw_df[['date', 'sales', 'temp']]

#scale the temperature data
raw_df['temp_scaled'] = raw_df['temp'] - raw_df['temp'].min()
raw_df['trend'] = (raw_df.sales.index - sum(raw_df['date'] <= train_test_date)) / 365.25

{% endhighlight %}

## The Temperature Model

The model described below is what satisticians call a Poisson regression model, which is a type from a class of statistical modelling technique called Generalized Linear Model (GLM). In the classical Linear Regression model, the observation assumes a Normal distribution with noise $$\sigma^2$$. A GLM generalizes from Linear Regression by also allowing the observation to take other distributions, which in this case is the Poisson distribution. The log function below is the link function that "links" the linear model which is a Fourier Series to the parameter of the Poisson distribution.

$$ temp_t \sim Poisson(\mu_t) $$
<br>
$$ log(\mu_t) = a + \displaystyle\sum_{k=1}^{K=6}b_k \sin(\frac{2\pi kt}{m}) + \tilde b_k \cos(\frac{2\pi kt}{m}), m = 365.25 $$
<br>
$$ a \sim Normal(0, 1) $$
<br>
$$ b_k, \tilde b_k \sim Normal(0,1) $$

The goal of this model is to make inference on the parameters of the linear model, which are the intercept $$ a $$ and the Fourier coefficients $$ b_k $$ and $$ \tilde b_k $$.

### Constructing the Temperature Model

First, we extract all the features that are required for the temperature model.

{% highlight python %}

date = raw_df["date"]
temp_scaled = raw_df["temp_scaled"]
trend = raw_df["trend"]
sales = raw_df["sales"]
# We extract the day of week for the sales model below.
dayofweek_idx, dayofweek = raw_df["date"].dt.dayofweek.factorize()

periods = raw_df["date"].dt.dayofyear / 365.25
n_order = 6

{% endhighlight %}

We construct the Fourier features as in [this post](https://www.pymc.io/projects/examples/en/latest/time_series/Air_passengers-Prophet_with_Bayesian_workflow.html).

{% highlight python %}

fourier_features = pd.DataFrame(
    {
        f"{func}_order_{order}": getattr(np, func)(2 * np.pi * periods * order)
        for order in range(1, n_order + 1)
        for func in ("sin", "cos")
    }
)

{% endhighlight %}

Constructing the model as described above in TFP is fairly straightforward. First, we specify the priors $$ a $$ and $$ b $$. Then, the linear model described in the equation above is described. And finally, the likelihood function is parameterized using the linear model.

The model is constructed as a joint probability distribution using the <code>tfd.JointDistributionCoroutineAutoBatched</code> method that returns a <code>tfd.Distribution</code> instance, which allows us to sample and compute the log likelihood of the distribution easier. This will come in handy when we want to sample from the prior predictive and fit the model using MCMC.

{% highlight python %}

def make_temperature_model():
    def model():
        #priors
        a = yield tfd.Normal(loc=0, scale=1, name='a')
        b  = yield  tfd.Normal(loc=tf.zeros(shape=(1,12)), scale=1,name='b')
        
        #the linear model
        mu_t = a + tf.linalg.matmul(b, fourier_features.T)
        
        #the likelihood
        likelihood = yield tfd.Poisson(rate=tf.math.exp(mu_t[0]), name='likelihood')
    return tfd.JointDistributionCoroutineAutoBatched(model)

temperature_model = make_temperature_model()

{% endhighlight %}

Below we write the helper function that returns the log likelihood function given the observation $$ P(\theta \mid X)P(X) $$. This is really easy to do, although it is not necessary if we are using PyMC by pinning the log probability function to the observed temperature. 

{% highlight python %}

def temp_log_prob(a, b):
    return temperature_model.log_prob([a, b, temp_scaled])

{% endhighlight %}

### Prior Predictive Checking of the Temperature Model

As part of the [Bayesian Workflow](https://arxiv.org/pdf/2011.01808.pdf), we need to run a sanity check by sampling from the prior predictive distribution to ensure that our specified model produces values that we expect from a temperature data. For example, we don't expect earth temperatures to be as hot as $$ 200^\circ F $$. If so, we need to enforce a stricter prior to our model to prevent the model from producing extreme values. 

{% highlight python %}

temp_prior_predictive = temperature_model.sample(1000)

temp_pred = temp_prior_predictive[-1]
#remove dimensions of size 1
temp_pred = tf.squeeze(temp_pred).numpy()

{% endhighlight %}

![Temperature Prior](https://raw.githubusercontent.com/iamzafran/iamzafran.github.io/gh-pages/Tutorials/Prior%20Knowledge%20Time%20Series/temp_prior_dist.svg)

### Fitting the Temperature Model

#### Finding Good Initial State via MAP

Now, its time to fit the model by sampling from the posterior via MCMC. But before we do that, we need to find a good initial state for the sampling algorithm so it have a good starting point to begin with, to reduce the number of burnin samples required. A good initialization point would be the MAP estimate of the posterior $$ \arg \max_{\theta} \log P(\theta \mid X)P(\theta) $$. Usually, we don't have to do this by initializing the initialization to either a zero or one initialization. But I found that, without the MAP initialization, the MCMC sampler returns no samples as the sampler is stuck is the region of very low probability and keeps rejecting the samples. 

We first initialize the parameter values to a sampled value. To find the MAP values of the parameters is straightforward by minimizing the Negative Log Likelihood (NLL) with respect to the parameters via gradient descent. As the dataset is small, no minibatching is required. So the parameter estimates converges smoothly.

{% highlight python %}

t = temperature_model.sample()

a = tf.Variable(initial_value=t[0], trainable=True)
b = tf.Variable(initial_value=t[1], trainable=True)

{% endhighlight %}

And we run the optimization using the Adam optimizer for 5000 iterations.

{% highlight python %}

optimizer = tf.optimizers.Adam(learning_rate=0.001)
losses = []
for _ in range(5000): 
    with tf.GradientTape() as tape:
        #minimize the Negative Log Likelihood
        loss = -temp_log_prob(a, b)
    grad = tape.gradient(loss, [a, b])
    optimizer.apply_gradients(zip(grad, [a, b]))
    losses.append(loss)

{% endhighlight %}


![Temperature NLL](https://raw.githubusercontent.com/iamzafran/iamzafran.github.io/gh-pages/Tutorials/Prior%20Knowledge%20Time%20Series/temp_nll.svg)

#### Sampling of the Posterior via MCMC

Now we're ready to run the MCMC sampling of the posterior for inference. In PyMC, this can be achieved with only two lines. We don't have such luxury when using TFP as we need to specify which sampling algorithm to use e.g., a Hamiltonian Monte Carlo or a No U-Turn Sampler and also the uncostraining bijectors to map the values to an uncostrained space (as the MCMC is run in an uncostrained space). The initial state of the MCMC obtained via MAP earlier is passed to the MCMC function via the `current_state` argument as the starting point of the MCMC.


{% highlight python %}

@tf.function
def sample_posterior(num_chains, num_results, num_burnin_steps):
    
    def trace_fn(_, pkr):
        return (
            pkr.inner_results.target_log_prob,
            pkr.inner_results.leapfrogs_taken,
            pkr.inner_results.has_divergence,
            pkr.inner_results.energy,
            pkr.inner_results.log_accept_ratio
        )
    
    hmc = tfp.mcmc.NoUTurnSampler(
        target_log_prob_fn=temp_log_prob,
        step_size=0.001
    )
    
    #initialize using the MAP estimate
    initial_state = [
        tf.repeat(a, num_chains, axis=0, name='init_a'),
        tf.tile(tf.reshape(b, shape=(1, 1, 12)), multiples=[num_chains,1, 1], name='init_b')
    ]
    
    unconstraining_bijectors = [
        tfb.Softplus(),
        tfb.Identity()
    ]
    
    kernel = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=hmc, bijector=unconstraining_bijectors
    )
    
    samples, kernel_results = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=initial_state,
        kernel=kernel,
        trace_fn=trace_fn
    )
    
    

   
    return samples, kernel_results
    
    
samples, sampler_stat = sample_posterior(num_chains=4, num_results=4000, num_burnin_steps=1000)
{% endhighlight %}

Once the sampling is completed, the results is stored in a dictionary array using the PyMC3 convention.

{% highlight python %}

sample_stats_name = ['lp', 'tree_size', 'diverging', 'energy', 'mean_tree_accept']

sample_stats = {k:v.numpy().T for k, v in zip(sample_stats_name, sampler_stat)}

sample_stats['tree_size'] = np.diff(sample_stats['tree_size'], axis=1)


var_name = ['a', 'b_fourier']
posterior = {k:np.swapaxes(tf.squeeze(v).numpy(), 1, 0) 
             for k, v in zip(var_name, samples)}
             
{% endhighlight %}

Fantastic!!! The results from the sampling shows that the parameters converges to a distribution. Here, it shows that all parameters resembles a Normal distrbution centered around the mean which also happens to be around the MAP estimate.

![Temperature Trace](https://raw.githubusercontent.com/iamzafran/iamzafran.github.io/881731e629b07b67419ec6039c276290f1fc70b0/Tutorials/Prior%20Knowledge%20Time%20Series/temp_trace.svg)

![Temperature Forest](https://raw.githubusercontent.com/iamzafran/iamzafran.github.io/gh-pages/Tutorials/Prior%20Knowledge%20Time%20Series/temp_forest.svg)

### Posterior Predictive of the Temperature Model

Now it's time to plot the posterior predictive distribution: <br>

$$P(X \mid \theta) = \int P(X|\theta)P(\theta|X) d\theta$$

And, of course we're going to approximate the integral using the parameter samples obtained earlier.

{% highlight python %}

a_samples = samples[0][:,0]

b_samples = samples[1][:, 0, :, :]

{% endhighlight %}

First we reshape the Fourier features so we can multiply it with all the parameter samples.

{% highlight python %}

ff = tf.tile(tf.reshape(fourier_features.T, shape=[1, 12, -1]), [4000, 1, 1])

{% endhighlight %}

Now, compute the linear model for the Poisson regression model.

{% highlight python %}

mu_t = tf.reshape(a_samples, shape=(4000, 1)) + tf.squeeze(tf.linalg.matmul(b_samples, ff))

{% endhighlight %}

And pass the $$\mu_t$$ value to the Poisson distribution. Here we make each parameter distribution samples to be independent to each other by using the <code>tfd.Independent</code> method. Refer to [this](https://www.tensorflow.org/probability/examples/TensorFlow_Distributions_Tutorial) tutorial for more details.

{% highlight python %}

temp_dist = tfd.Independent(tfd.Poisson(rate=tf.math.exp(mu_t)), reinterpreted_batch_ndims=1)

temp_pred = temp_dist.sample().numpy()

{% endhighlight %}

![Temperature Posterior Predictive](https://raw.githubusercontent.com/iamzafran/iamzafran.github.io/master/Tutorials/Prior%20Knowledge%20Time%20Series/post_pred_temp.svg)

Looks good. The temperature model manages to recover the temperature trend by using the Fourier features as the regression features.

## Sales Model

Now we have arrived to the main objective of this tutorial, which is to incorporate prior knowledge to model short time series data. 

The sales model described below is a NegativeBinomial regression model parameterized by the mean $$\mu_t$$ that is a damped mean dynamic that combines information from previous observation $$sales_{t-1}$$ and seasonality coefficient $$\lambda_t$$ at each time step. In addition to the Fourier seasonality, the seasonality coefficient $$\lambda_t$$ also takes into accound the day of week effect and the trend effect $$b_{trend}$$.

The prior knowledge regarding the seasonality of the sales is embedded in the prior distribution of the Fourier coefficient $$b^{'}_k$$ and $$\tilde b^{'}_K$$ that has the same distribution as the posterior of the temperature model obtained previously.

$$sales_t \sim NegativeBinomial(\mu_t, \alpha)$$ <br>

$$\mu_t = (1-\delta-\eta)\lambda_t + \delta\mu_{t-1} + \eta sales_{t-1}, 0\leq\delta\leq1, 0 \leq \eta \leq 1 - \delta$$ <br>

$$\log(\lambda_t) = b_{trend}\frac{t}{m} + \displaystyle\sum^{7}_{j=1}b_{dow,j}dayofweek_t + \displaystyle\sum_{k=1}^{K=6}b^{'}_k\sin (\frac{2\pi kt}{m}) + \tilde b_k^{'}\cos(\frac{2\pi kt}{m}), m=365.25$$ <br>

$$\delta \sim Beta(1,10)$$<br>
$$\eta \sim Gamma(1,10)$$<br>
$$b_{trend} \sim Normal(0.03, 0.02)$$ <br>
$$b_{dow,j} \sim Normal(4,2)$$<br>
$$b^{'}_{k} \sim Normal(\mathbb{E}[b_k], \sqrt{Var[b^{'}_k]})$$ <br>
$$\tilde b^{'}_{k} \sim Normal(\mathbb{E}[\tilde b^{'}_{k}],\sqrt{Var[\tilde b^{'}_{k}]})$$ <br>
$$\alpha \sim Normal(0.5)$$

First, we separate the time series into training and test sets.

{% highlight python %}

start_date = pd.to_datetime("2013-07-01")
train_test_date = pd.to_datetime("'2013-10-15'")

df_train = raw_df[raw_df['date'] <= train_test_date]
df_test = raw_df[raw_df['date'] > train_test_date]

{% endhighlight %}

![Train-Test Split](https://raw.githubusercontent.com/iamzafran/iamzafran.github.io/master/Tutorials/Prior%20Knowledge%20Time%20Series/sales_train_test_split.svg)

Next, we extract the features that we are interested in, and construct the Fourier features as before.

{% highlight python %}

date_train = df_train["date"].values
sales_train = df_train["sales"].values
trend_train = df_train["trend"].values
dayofweek_idx_train, dayofweek_train = df_train["date"].dt.dayofweek.factorize()

periods_train = df_train["date"].dt.dayofyear / 365.25
n_order = 6

fourier_features_train = pd.DataFrame(
    {
        f"{func}_order_{order}": getattr(np, func)(2 * np.pi * periods_train * order)
        for order in range(1, n_order + 1)
        for func in ("sin", "cos")
    }
)

fourier_features_train = fourier_features_train.values

fourier_features_train = np.array(fourier_features_train, dtype=np.float32)

{% endhighlight %}

### Constructing the Bike Sales Model

First, we calculate the mean and the standard deviation of the posterior of the Fourier coefficient from the temperature model to parameterize the prior of the Fourier coefficient in the sales model to embed prior information to the sales model.

{% highlight python %}

fourier_loc = np.mean(samples[1][:, 0, :, :], axis=0)
fourier_sd = np.std(samples[1][:, 0, :, :], axis=0)

{% endhighlight %}

Next, we construct the sales model as described previously. Note that we use the <code>tfd.NegativeBinomial.experimental_from_mean_dispersion</code> to construct the NegativeBinomial model as the default parameterization by TFP requires to input the number of failures and the probability of success, instead of the mean and dispersion that is required for the model. So thankfully this method exists to make construct the NegativeBinomial probability distribution from an alternative parameterization easier.


{% highlight python %}

def make_sales_forecast_model():
    
    def model():
        
        delta = yield tfd.Beta(concentration1=1, concentration0=10, name='delta')
        eta = yield tfd.Gamma(concentration=0.5, rate=10, name='eta')
        b_trend  = yield tfd.Normal(loc=0.03, scale=0.02, name='b_trend')
        b_dayofweek = yield tfd.Normal(loc=tf.repeat(4., len(dayofweek)), scale=2, name='day_of_week')
        b_fourier = yield tfd.Normal(loc=fourier_loc, scale=fourier_sd, name='b_fourier')
        alpha = yield tfd.Normal(loc=0, scale=0.5, name='alpha')
        
        
        fourier_contribution = tf.linalg.matmul(b_fourier, fourier_features_train.T)[0]
        
        dayofweek_contibution = tf.gather(b_dayofweek, dayofweek_idx_train)
        
        trend_contribution = b_trend * trend_train
        
        seasonality = tf.math.exp(fourier_contribution + dayofweek_contibution + trend_contribution)
        alpha = tf.math.square(alpha)
        mu = tf.ones([len(sales_train)], dtype=np.float32)
        
        mu = tf.unstack(mu)
        mu[0] = sales_train[0]
        
        for i in range(1, len(mu)):
            mu[i] = (1 - delta - eta) * seasonality[i] + delta * mu[i-1] + eta * sales_train[i-1]
        mu = tf.stack(mu)
        sales = yield tfd.NegativeBinomial.experimental_from_mean_dispersion(mean=mu, dispersion=alpha, name='likelihood')
        
    return tfd.JointDistributionCoroutineAutoBatched(model) 

sales_model = make_sales_forecast_model()

{% endhighlight %}

### Prior Predictive Checking of the Sales Model

As before, we will view the prior predictive distribution as a sanity check to verify that our constructed model is correct and does not produce extreme values.

![Sales Prior Dist](https://raw.githubusercontent.com/iamzafran/iamzafran.github.io/master/Tutorials/Prior%20Knowledge%20Time%20Series/sales_prior_dist.svg)


## Fitting the Sales Model

### Defining Log Probability Functions for MAP and MCMC

Below we have specified two helper functions to calculate the log probability of the joint distribution, one for the MAP estimete and the other for the MCMC sampling. The reason for two separate log probability, is to enforce the constraint for the $$\delta$$ and $$\eta$$ variables. The original [tutorial](https://juanitorduz.github.io/short_time_series_pymc/) that uses PyMC3 adds negative infinity ($$-\infty$$) to the log probability if the constraint is violated, forcing the MCMC algorithm to explore the region where the constraint is satisfied. Or in constraint optimization speak, the feasible space. However, this approach induces stability issues when being applied to gradient descent algorithm for the MAP estimate. Furthermore, we cannot use a bijector constraining the $$\eta$$ variable as the constraint depends on the value $$\delta$$. Therefore, for the purpose of the MAP estimate, a bijector is applied to map the variable to the feasible space and for the MCMC, the constraint is enforced by adding negative infinity ($$-\infty$$) to the log probability.

{% highlight python %}

@tf.function
def sales_log_prob(delta, eta, b_trend, day_of_week, b_fourier, alpha):
    
    delta = tfb.SoftClip(low=0, high=1).forward(delta)
    eta = tfb.SoftClip(low=0, high=(1-delta)).forward(eta)
    
    return sales_model.log_prob([delta, eta, b_trend, day_of_week, b_fourier, alpha, sales_train])

@tf.function
def sales_log_prob_mcmc(delta, eta, b_trend, day_of_week, b_fourier, alpha):
    
    sales_lp = sales_model.log_prob([delta, eta, b_trend, day_of_week, b_fourier, alpha, sales_train])

#     #check if satisfy 0 <= eta <= (1 - delta)
    constrain = tf.logical_and(tf.greater_equal(eta, 0), tf.less_equal(eta, (1-delta))) 
    
#     # add infinity to the log probability if constraint is not satisfied
    log_prob = tf.where(constrain, x=sales_lp, y=-np.inf) 
    

    return log_prob

{% endhighlight %}

### Finding Good Initial State for MCMC via MAP

The steps are similar as in the temperature model.

First, initialize all variables to sampled value.

{% highlight python %}

s = sales_model.sample()

delta = tf.Variable(s[0], trainable=True, name='delta')
eta = tf.Variable(s[1], trainable=True, name='eta')
b_trend = tf.Variable(s[2], trainable=True, name='b_trend')
day_of_week = tf.Variable(s[3], trainable=True, name='day_of_week')
b_fourier = tf.Variable(s[4], trainable=True, name='b_fourier')
alpha = tf.Variable(s[5], trainable=True, name='alpha')

trainable_variables =[
    delta,
    eta,
    b_trend,
    day_of_week,
    b_fourier,
    alpha
]

{% endhighlight %}

As before, we minimize the Negative Log Likelihood (NLL) with respect to all the parameters.

{% highlight python %}

optimizer = tf.optimizers.Adam(learning_rate=0.001)
losses = []
for _ in range(7000): 
    with tf.GradientTape() as tape:
        loss = -sales_log_prob(trainable_variables[0], trainable_variables[1], trainable_variables[2], trainable_variables[3], trainable_variables[4], trainable_variables[5])
    grad = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(grad, trainable_variables))
    losses.append(loss)

{% endhighlight %}

![Sales NLL](https://raw.githubusercontent.com/iamzafran/iamzafran.github.io/master/Tutorials/Prior%20Knowledge%20Time%20Series/sales_nll.svg)

### Sampling the Sales Model Posterior via MCMC

For the initial parameters of the MCMC, we use first pass the constrained variable through the bijector to obtain the constrained values.

{% highlight python %}

delta_clip = tfb.SoftClip(low=0, high=1).forward(delta)
eta_clip = tfb.SoftClip(low=0, high=(1-delta_clip)).forward(eta)

{% endhighlight %}

For the MCMC sampling, we don't actually have to specify a bijector for the variables as it is enforced implicitly via the <code>sales_log_prob_mcmc</code> function by returning negative infinity if the constraint is not satisfied. But TFP requires us to specify a bijector nonetheless, so the <code>tfb.Identity</code> bijector does the trick by doing absolutly nothing.

{% highlight python %}

@tf.function
def sample_sales_model_posterior(num_chains, num_results, num_burnin_steps):
    
     
    def trace_fn(_, pkr):
        return (
            pkr.inner_results.target_log_prob,
            pkr.inner_results.leapfrogs_taken,
            pkr.inner_results.has_divergence,
            pkr.inner_results.energy,
            pkr.inner_results.log_accept_ratio
    )
    
    hmc = tfp.mcmc.NoUTurnSampler(
        target_log_prob_fn=sales_log_prob_mcmc,
        step_size=0.001
    )
    
    

    initial_state = [
        tf.repeat(delta_clip, num_chains, axis=0, name='init_delta'),
        tf.repeat(eta_clip, num_chains, axis=0, name='init_eta'),
        tf.repeat(b_trend, num_chains, axis=0, name='b_trend'),
        tf.tile(tf.reshape(day_of_week, (1,-1)), multiples=[num_chains, 1], name='init_day_of_week'),
        tf.tile(tf.reshape(b_fourier, (1,1,12)), multiples=[4, 1, 1], name='init_b_fourier'),
        tf.repeat(alpha, 4, axis=0, name='init_alpha')
    ]
    
    #no constraining bijectors is actually needed as the constrain is enforced by th log prob function
    unconstraining_bijectors = [
        tfb.Identity(),
        tfb.Identity(),
        tfb.Identity(),
        tfb.Identity(),
        tfb.Identity(),
        tfb.Identity()
    ]
    
    kernel = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=hmc, bijector=unconstraining_bijectors
    )
    
    samples, kernel_results = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=initial_state,
        kernel=kernel,
        trace_fn=trace_fn
    )
   
    return samples, kernel_results

{% endhighlight %}

We run the MCMC sampler to obtain 4000 samples which takes around 3 hours. You could use the Hamiltonian Monte Carlo algorithm for a faster result, but it produces divergent samples that does not converge to any particular distribution.

{% highlight python %}

samples, sampler_stat = sample_sales_model_posterior(
    num_chains=4, num_results=4000, num_burnin_steps=1000)

{% endhighlight %}

Looking at the traceplots and the forest plots, we are satisfied with the result of the samples from the posterior. The R-Hat values also produces values close to zero for all parameters, which indicates that the sampling is successful.

![Sales Trace](https://raw.githubusercontent.com/iamzafran/iamzafran.github.io/881731e629b07b67419ec6039c276290f1fc70b0/Tutorials/Prior%20Knowledge%20Time%20Series/sales_trace.svg)

![Sales Forest](https://raw.githubusercontent.com/iamzafran/iamzafran.github.io/master/Tutorials/Prior%20Knowledge%20Time%20Series/sales_forest.svg)

### Posterior Predictive Distribution

Before we make sales forecast on out-of-sample data, let's first verify our model by visualizing the posterior predictive distribution.

Collect all parameter samples produced by the MCMC algorithm.

{% highlight python %}

delta_samples = samples[0][:, 0]

eta_samples = samples[1][:,0]

b_trend_samples = samples[2][:,0]

day_of_week_samples = samples[3][:,0]

b_fourier_samples = samples[4][:,0]

alpha_samples = samples[5][:, 0]

alpha_samples =  tf.square(alpha_samples)

{% endhighlight %}

Calculate the Fourier features, day of week contribution $b_{dow}$ and trend contribution $$b_{trend}$$

{% highlight python %}

# fourier_contribution + dayofweek_contibution + trend_contribution
seasonality_samples = fourier_contribution_samples + day_of_week_contribution_samples + trend_contribution_samples

seasonality_samples = tf.math.exp(seasonality_samples)

dispersion_samples = tf.square(alpha_samples)

mu = np.ones([4000, len(sales_train)], dtype=np.float32)

mu[:, 0] = sales_train[0]

{% endhighlight %}

Calculate the damped mean linear model by combining effects of the seasonality and previous observation.

{% highlight python %}

for i in range(1, len(mu[0])):
    mu[:,i] = (1 - delta_samples - eta_samples) * seasonality_samples[:, i] + delta_samples * mu[:,i-1]  + eta_samples * sales_train[i-1]

{% endhighlight %}

Construct the posterior predictive distribution from the parameter samples calculated earlier and  sample from the posterior predictive distribution.

{% highlight python %}

sales_posterior = tfd.NegativeBinomial.experimental_from_mean_dispersion(mean=mu, dispersion=tf.reshape(alpha_samples, shape=(-1,1)))

sales_posterior = tfd.Independent(sales_posterior, reinterpreted_batch_ndims=1)

sales_samples = sales_posterior.sample().numpy()

{% endhighlight %}

![Sales Posterior](https://raw.githubusercontent.com/iamzafran/iamzafran.github.io/gh-pages/Tutorials/Prior%20Knowledge%20Time%20Series/sales_posterior.svg)

Good. So far, we succesfully replicate the result of the sales model from the [original blog post](https://juanitorduz.github.io/short_time_series_pymc/). Next, we will make forecasting of future observations.

## Making Future Bike Sales Forecast

Making sales forecast is similar to how we make the posterior predictive distribution. Except that, instead of using the observed sales to calculate the the damped mean, we use the sampled sales forecast from the previous time step as we will show below.

{% highlight python %}

fourier_contribution_posterior = tf.squeeze(tf.linalg.matmul(b_fourier_samples, fourier_features.T))

day_of_week_contribution_posterior = tf.gather(day_of_week_samples, dayofweek_idx, axis=1)

trend_contribution_posterior = tf.reshape(b_trend_samples, shape=(-1, 1)) * trend.values

seasonality_posterior = fourier_contribution_posterior + day_of_week_contribution_posterior + trend_contribution_posterior

seasonality_posterior = tf.math.exp(seasonality_posterior)

sales_posterior = np.zeros(shape=(4000, n))

sales_posterior[:, :n_train] = sales_samples

mu_posterior = np.zeros(shape=(4000, n), dtype=np.float32)
mu_posterior[:, :n_train] = mu

{% endhighlight %}

The sales forecast can be made by calculating the damped mean and use it to parameterize the Negative Binomial distribution at each time step. Then, the sales forecast is made by sampling the Negative Binomial distribution.

{% highlight python %}

for i in range(n_train, n):
    mu_posterior[:,i] = (1 - delta_samples - eta_samples) * seasonality_posterior[:, i] + delta_samples * mu_posterior[:,i-1]  + eta_samples * sales_posterior[:,i-1]
    current_sales_dist = tfd.NegativeBinomial.experimental_from_mean_dispersion(mean=mu_posterior[:,i], dispersion=alpha_samples)
    #forecast sales on current time step by sampling from the current sales distribution.
    sales_posterior[:,i] = current_sales_dist.sample().numpy()

{% endhighlight %}

Fantastic. The result looks similar to the [original post](https://juanitorduz.github.io/short_time_series_pymc/) using PyMC3.

![Sales Forecast](https://raw.githubusercontent.com/iamzafran/iamzafran.github.io/gh-pages/Tutorials/Prior%20Knowledge%20Time%20Series/sales_predictive_posterior.svg)
