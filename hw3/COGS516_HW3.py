#!/usr/bin/env python
# coding: utf-8

# # COGS516 - Assignment 3
# 
# Please enter your **name, surname** and **student number** instead of `"NAME-HERE"`, `"SURNAME-HERE"`, `"NUMBER-HERE"` below

# In[1]:


student = {
    'name' : "Abdullah Burkan" ,
    'surname' : "Bereketoglu", 
    'studentNumber' : "2355170"
}

print(student)


# In[2]:


import numpy as np
import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
import arviz as az
import daft


# ## Coffee Ratings
# 
# [The coffee ratings dataset](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-07-07) contains over 1300 thousand coffee reviews from the Coffee Quality Institute. 
# 
# The description of the variables in this dataset is available [here](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-07-07). You can also check the [coffee review website](https://www.coffeereview.com/interpret-coffee/) for the meaning of the dimensions they use for review (e.g. aroma, aftertaste etc.).
# 
# In this assignment, we will analyze a simplified version of this dataset by using `aroma` ($AR$), `aftertaste` ($AT$), `species` ($SP$) and  `total_cup_points` ($CP$) from this dataset. 
# 
# Moreover, we will also take just one sample from each farm (we will see how to model the similarity between the farms later in the class).
# 
# > The dataset is provided in `coffee_ratings.csv` in ODTUClass

# In[3]:


d = pd.read_csv("coffee_ratings.csv")

dsimp = d[["farm_name", "total_cup_points", "aroma", "aftertaste", "species"]].groupby(by='farm_name').sample(n = 1)
dsimp.head()


# ## Part 1 - Coffee Quality 

# Suppose we assume the following DAG in which $AR$ is `aroma`, $AT$ `aftertaste`, $SP$ `species`  and $CP$  `total_cup_points`.
# > To see the DAG below, you need to install the `daft` library. See [here for installing instructions](https://docs.daft-pgm.org/en/latest/)

# In[4]:


pgm = daft.PGM()
pgm.add_node("SP","SP",0,2)
pgm.add_node("AR","AR",0,1)
pgm.add_node("AT","AT",2,1)
pgm.add_node("CP","CP",1,0)
pgm.add_edge("SP", "AR")
pgm.add_edge("AR", "AT")
pgm.add_edge("AT", "CP")
pgm.add_edge("AR", "CP")
pgm.render()


# > Note: for all the steps below, do not forget to re-scale your variables considering what they represent, and choose suitable priors. 
# 
# Use backdoor criterion and estimate the **total** effect of $SP$ on $CP$. (10 pts)

# In[5]:


dsimp['species'].unique()


# We have 2 paths that go to CP from SP. One is SP - AR - CP the other one is SP - AR - AT - CP. For total we should only 
# Since there exist only 2 species, we pick shape as 2.
# 
# Here we normalize the data.

# In[6]:


dsimp["AT"] = (dsimp.aftertaste - dsimp.aftertaste.mean()) / dsimp.aftertaste.std()
dsimp["AR"] = (dsimp.aroma - dsimp.aroma.mean()) / dsimp.aroma.std()
dsimp["CP"] = (dsimp.total_cup_points - dsimp.total_cup_points.mean()) / dsimp.total_cup_points.std()
dsimp["SP"] = (dsimp["species"] == "Arabica").astype(int).values


# In[7]:


# total effect
with pm.Model() as Tot_SP:
    a = pm.Normal('a', 0, 1, shape = 2)
    sigma = pm.HalfNormal('x', 1) # wanted to try something different
    mu = a[dsimp.SP] # page 48 categorical causes
    cp = pm.Normal('cp', mu = mu, sigma = sigma, observed = dsimp.CP)
    trace_SP = pm.sample(1000, return_inferencedata=True)

    #completed according to daggity causal effect identification for total effect
    #it states that we don't need any adjustments therefore no stratification.
az.summary(trace_SP)  


# In[8]:


az.plot_trace(trace_SP)


# In[9]:


ax = az.plot_posterior(trace_SP, var_names="a", point_estimate=None, color = "k", hdi_prob = 'hide')


# Completed according to daggity causal effect identification for total effect
# it states that we don't need any adjustments therefore no stratification.
# 
# So basically Backdoor criterion states that total effect of SP to CP does not need any adjustments.

# Use backdoor criterion and estimate both the **total effect** and **direct effect** of $AR$ on $CP$. (20 pts)

# First of all it is important to state that we should adjust AT to find direct effect so we will add it to our model with backdoor criterion, but in Total effect there is no need of any adjustment for AR to CP. 
# 
# Tested with Daggity

# Let's find a model which will give us a better grip for direct effect.

# In[10]:


# total effect
with pm.Model() as total_effect_model:
    intercept = pm.Normal("intercept", 0, sigma=0.5)
    bAR = pm.Normal("bAR", 0, sigma=0.5)
    sigma = pm.Exponential("sigma", 1)
    mu = intercept + bAR * dsimp.AR
    D = pm.Normal("d", mu=mu, sd=sigma, observed = dsimp.CP)
    trace_tot_AR_CP = pm.sample(1000, tune = 1000, return_inferencedata=True)


# In[11]:


# direct effect
with pm.Model() as direct_effect_model:
    intercept = pm.Normal("intercept", 0, sigma=0.5)
    bAR = pm.Normal("bAR", 0, sigma=0.5)
    bAT = pm.Normal('bAT', 0, 0.5)
    sigma = pm.Exponential("sigma", 1)
    mu = intercept + bAR * dsimp.AR + bAT * dsimp.AT
    D = pm.Normal("d", mu=mu, sd=sigma, observed = dsimp.CP)
    trace_dir_AR_CP = pm.sample(1000, tune = 1000, return_inferencedata=True)


# In[12]:


az.summary(trace_tot_AR_CP)


# In[13]:


az.plot_trace(trace_tot_AR_CP)


# In[14]:


az.summary(trace_dir_AR_CP)


# In[15]:


az.plot_trace(trace_dir_AR_CP)


# In[16]:


ax = az.plot_posterior(trace_tot_AR_CP, var_names="bAR", point_estimate=None, color = "black", hdi_prob = 'hide')
ax.axvline(0.0,color="k",alpha=0.5, linestyle="--")
az.plot_posterior(trace_dir_AR_CP, var_names="bAR", ax = ax, point_estimate=None , hdi_prob = 'hide', color = "red")


# In[17]:


az.summary(trace_dir_AR_CP)


# Estimate the **direct effect** of $AT$ on $CP$. (5 pts)

# Stratify AR, look page "25/91" 
# Also according to daggity it also shows that adjustment of AR is necessary which is stratify.
# 

# In[18]:


#stratify AR. # page 25/91 
# also according to daggity it also shows that adjustment of AR is necessary which is stratify.

# direct effect
with pm.Model() as direct_2_effect_model:
    intercept = pm.Normal("intercept", 0, sigma=0.5)
    bAR = pm.Normal("bAR", 0, sigma=0.5)
    bAT = pm.Normal('bAT', 0, 0.5)
    sigma = pm.Exponential("sigma", 1)
    mu = intercept + bAR * dsimp.AR + bAT * dsimp.AT
    D = pm.Normal("d", mu=mu, sd=sigma, observed = dsimp.CP)
    trace_dir_AT_CP = pm.sample(1000, tune = 1000, return_inferencedata=True)


# In[19]:


az.summary(trace_dir_AT_CP)


# In[20]:


az.plot_trace(trace_dir_AT_CP)


# In[21]:


ax = az.plot_posterior(trace_dir_AT_CP, var_names="bAR", point_estimate=None, color = "k", hdi_prob = 'hide')
ax.axvline(0.0,color="k",alpha=0.5, linestyle="--")
az.plot_posterior(trace_dir_AR_CP, var_names="bAR", ax = ax, point_estimate=None , hdi_prob = 'hide', color = "red")


# Based on the DAG, and your estimates above, discuss the effects of these measures on coffee quality. (15 pts) 
# 
# ---
# 1) For SP to CP we found the total effect because it is not possible due to post-treatment bias to find the direct effect of the SP to CP. But we can state that SP posterior mean for total effect is for Arabica which is coded a[1] is negative(-0.239) and also for a[0] it is slightly positive these show that being Arabica coffee lowers the total cup score at the end as total effect, but Robusta coffee does not have the bad effects on the consumer score on the cup of coffee (this comment is biased and needs more adjustments which will be seen in the part 3 of the question. It is also seen from the trace plots that the priors are not that really picked nicely, I tried a lot of values, but unfortunately thats all I could came up with. 
# 
# 2) For the AR to CP total and direct effect plots we can deduce that by using the means from the summary statistics, for the posterior mean total effect is 0.751 which means that for every unit increase in AR,CP increases by 0.751 through both the direct and indirect pathways. Furthermore, Direct effect shows that mean of posterior direct effect of AR is 0.216, which shows that indirect effect is 0.751-0.216 = 0.535. This also shows that for one unit increase 0.216 increase of AR, CP due to direct effect. For second part Trace patterns are great and not having much difference for chains.
# 
# 3) Here it shows a similar pattern to AR,CP direct effect, I may have wrote them wrongly, but according to the mean of AT to CP being 0.225 for direct AT, CP direct effects of AR to CP and AT to CP are similar for this model. Basically it states that for every one unit increase there is 0.225 increase for AT-> CP. Furthermore, Trace patterns are great and not having much difference for chains.
# 
# ---

# ## Part 2 - Model Comparison
# 
# You must have three different models from the part above. Apart from those models, build a causal model where you stratify for all variables ($AR$, $SP$, $AT$). (5 pts)

# In[45]:


with pm.Model() as effect_model:
    intercept = pm.Normal("intercept", 0, sigma=0.5)
    bAR = pm.Normal("bAR", 0, sigma=0.5)
    bAT = pm.Normal('bAT', 0, 0.5)
    bSP = pm.Normal('bSP', 0, 0.5, shape = 2)
    sigma = pm.Exponential("sigma", 1)
    mu = intercept + bSP[dsimp.SP] * dsimp.SP + bAR * dsimp.AR + bAT * dsimp.AT
    D = pm.Normal("d", mu=mu, sd=sigma, observed = dsimp.CP)
    trace_ALL = pm.sample(1000, tune = 1000, return_inferencedata=True)


# Now compare the all four models (three from part 1 and the one above) using both PSIS and WAIC. Show all relevant statistics and diagrams to make this comparison, and discuss your results. Which model is expected make better predictions according to PSIS and WAIC? What does that tells us about the causal relations? (20 pts)

# In[46]:


compare_dict = {"All": trace_ALL, "Third":trace_dir_AT_CP, "Second":trace_dir_AR_CP, "First":trace_SP }

az.compare(compare_dict, ic ="loo")


# In[47]:


az.compare(compare_dict, ic ="waic")


# 1) I had problems with making it work unfortunately couldnt solve, but it seems like that the All model which can be said as the stratifying all of the elements give us a model that is the most powerful. The problem is even though it states it is the most powerful, we don't know exactly that due to true value for warning is returning for both WAIC and PSIS
# 
# 2) It is important to note that since we achieved best performance with stratifying all values, it can give us the idea that breaking the dependencies and giving place to all the features in the system that we use fits the predictor better than using only some of the features as variables in the linear regression predictor model.

# Do a robust regression on the best performing model, and compare them to each other using PSIS and WAIC. (15 pts)

# In[48]:


size = 571 # dimension of log likelihood of model_all . we have issue that it is bigger than 400.
true_intercept = 1
true_slope = 2

x = np.linspace(0, 1, size)
# y = a + b*x
true_regression_line = true_intercept + true_slope * x
# add noise
y = true_regression_line + np.random.normal(scale=0.5, size=size)

with pm.Model() as robust_model:
    pm.glm.GLM.from_formula("y~x", trace_ALL) # bayesian linear regression
    trace_robust = pm.sample(1000, tune = 1000, return_inferencedata=True)


# Now let's compare the new model with the best model.

# In[51]:


compare_dict_robust = {"Best": trace_ALL, "Robust": trace_robust}

az.compare(compare_dict_robust, ic ="loo")


# In[52]:


az.compare(compare_dict_robust, ic ="waic")


# Since robust win and got warning as false, I am both relieved and happy and I expected that robust algorithm to perform best since generalized models are better estimators, in most cases.
# 
# It also shows that generalized algorithms eliminate noise better and give a better predictive model according to both tests.
# 
# Robust Regression means thick tails and less surprise by extreme values, and for that with less extreme values better predictions are achieved and we can see that in the test.

# ## Part 3 - Expanded DAG (10 pts)
# 
# Consider all the variables in [the coffee ratings dataset](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-07-07) and build a larger/more comprehensive DAG than the one above. You can draw this DAG below. (You can use [`daft`](https://docs.daft-pgm.org/en/latest/) library to draw the DAG)

# In[23]:


d.head(n = 5)


# In[24]:


pgm = daft.PGM()
pgm.add_node("OW","OW",0,3) # Ownership
pgm.add_node("BGS","BGS",2,2) # bag weight
pgm.add_node("NOB","NOB",4,2) # number of bags
pgm.add_node("COR","COR",2,3) # country of origin
pgm.add_node("BAL","BAL",1,2) # Balance
pgm.add_node("SP","SP",0,2)
pgm.add_node("AR","AR",0,1)
pgm.add_node("AT","AT",2,1)
pgm.add_node("CP","CP",1,0)
pgm.add_edge("OW","SP")
pgm.add_edge("SP", "AR")
pgm.add_edge("COR","BGS")
pgm.add_edge("OW","BGS")
pgm.add_edge("OW","NOB")
pgm.add_edge("COR","NOB")
pgm.add_edge("NOB","BGS")
pgm.add_edge("COR","OW")
pgm.add_edge("COR","BAL")
pgm.add_edge("AR", "AT")
pgm.add_edge("AT", "CP")
pgm.add_edge("AR", "CP")
pgm.add_edge("SP", "BAL")
pgm.add_edge("BAL", "AR")
pgm.add_edge("OW","BAL")
pgm.add_edge("BGS","AR")
pgm.render()


# Describe this DAG (what kind of causal assumptions you made) in this DAG.
# 
# ---
# 
# In my new model, I assumed that balance of the coffee is dependent on the type of the coffee such as Robusta, Ecuador, Arabica, etc. And from my tasting knowledge smooth and soft coffees are liked more(my prior belief from being coffee aficionado.) So basically Species effect the Balance and balance effect the aroma and aroma effects the cup points.
# 
# Furthermore, I am someone who believes that most important thing for a coffee to have good aroma is having qualified providers from the best producer countries where coffee beans are taken care of really neatly. So I believe these coffee providers(Owners) are having effect on the species and the aroma, but it depends on the owner and origin of the country has direct effect to owner, because ethiopian coffee will have ethiopian owners. Ownership directly gives opinion about what is the species of the coffee hence aroma is determined and Owner also gives us opinion about whether the coffee is balanced or not.
# 
# Country of Origin (COR) also gives idea about the balance, because some beans have different taste, one can think of the velvety vibes of some beans from guatemalan beans.
# 
# Owner can also give us idea about how much they provide in bags and what is the total weight a unit bag can carry, since some of the owners are well-known providers to companies such as Starbucks or Amazon, they will have variety of options, best qualities and big bags with too many to count tools.
# 
# In that sense Country of Origin also gives idea on the bag weight, because some of the producer countries are more famous than the others and sell more coffee, such as Arabica. 
# 
# For that reason I can coin that bag weight can also give idea about the aroma, one can think that barrel beer tastes less good, same can sometimes apply to coffee, because bigger-heavier the bag, seed waits more and it can get dry and less tasty. So as the last assumption I can state that bag weight gives idea about the aroma.
# 
# ---

# ## Bonus (20 pts)
# 
# Build a **full Bayes** model on this DAG (in which you model all the relations and present the posteriors of this model).
# 

# In[67]:


d["OW"] = pd.Categorical(d["owner"]).codes
d["COR"] = pd.Categorical(d["country_of_origin"]).codes


# In[68]:


d["COR"].unique


# In[70]:


d["OW"].unique


# In[69]:


from theano import shared
species = dsimp.SP.values
species_shared = shared(species)

Cor = d.COR.values
cor_shared = shared(Cor)

owner = d.OW.values
owner_shared = shared(owner)


# In[87]:



with pm.Model() as full_bayes_model:
    # Owner
    h5 = pm.Normal('h5', mu = 0, sigma = 1 shape = 1339)
    nu5 = h5[owner_shared] 
    sig5 = pm.Exponential('sig5', 1)
    H5 = pm.Normal('H5', mu = nu5, sigma = sig5, observed = d.OW)
    # Country of Origin
    h4 = pm.Normal('h4', mu = 0, sigma = 1, shape = 1339)
    beta4 = pm.Normal("b4", 0,1)
    nu4 = h4[cor_shared]+ beta4[cor_shared]*(H5 - d.OW.mean())
    sig4 = pm.Exponential('sig4', 1)
    H4 = pm.Normal('H4', mu = nu4, sigma = sig4, observed = d.COR)
    # Species
    h2 = pm.Normal('h2', mu = 0, sigma = 1, shape= 2)
    nu2 = h2[species_shared] 
    sig2 = pm.Exponential('sig2', 1)
    H2 = pm.Normal('H2', mu = nu2, sigma = sig2, observed = dsimp.SP)
    
    # After taste
    h1 = pm.Normal('h1', mu = 0, sigma = 1)
    beta2 = pm.Normal('beta2', mu = 0, sigma = 1)
    nu1 = h1 + beta2*(H - dsimp.AR.mean())
    sig1 = pm.Exponential('sig1', 1)
    H1 = pm.Normal('H1', mu = nu1, sigma = sig1, observed = dsimp.AT)
    
    # Aroma
    h = pm.Normal('h', mu = 0, sigma = 1)
    beta3 = pm.Normal('beta3', 0, 1)
    nu = h + beta3*(H2 - dsimp.SP.mean())
    sig = pm.Exponential('sig', 1)
    H = pm.Normal('H', mu = nu, sigma = sig, observed = dsimp.AR)
    
    # Cup Point
    alpha = pm.Normal('alpha', mu = 5, sigma = 1)
    beta = pm.Normal('beta', mu = 0, sigma = 1)
    sigma = pm.Exponential('sigma',1)
    mu = alpha + beta3*(H - dsimp.AR.mean()) + beta2*(H1 - dsimp.AT.mean())
    W = pm.Normal('W', mu = mu, sigma = sigma, observed = dsimp.CP)
    
    trace_full_bayes = pm.sample(1000, tune = 1000, return_inferencedata=True)


# I couldn't properly finish the bonus. It has errors, so doesn't work.
