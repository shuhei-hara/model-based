import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import scipy
import preprocessing
import pandas as pd
from scipy.stats import uniform, expon
from scipy.special import expit, logit
import seaborn as sns
import pystan
import itertools

from matplotlib.lines import Line2D

def circular_inference(prior, likelihood,param):
    
    pa1 = param[0]
    pa2 = param[1]
    pa3 = param[2]
    pa4 = param[3]

    wp = pa1
    wl = pa2
#     wp = pa1
#     wl=pa2
    
    ap = pa3*20
    al = pa4*20
    
    wpMatrix = np.array([[wp,1-wp],[1-wp,wp]])
    wlMatrix = np.array([[wl,1-wl],[1-wl,wl]])
    
    amplifiedPrior = expit(np.log(prior[0]/prior[1])*ap)
    amplifiedLikelihood_left = expit(np.log(likelihood[0]/likelihood[1])*al)

    fPrior = np.dot(np.array([amplifiedPrior,1-amplifiedPrior]),wpMatrix)
    fLikelihood = np.dot(np.array([amplifiedLikelihood_left,1-amplifiedLikelihood_left]),wlMatrix)
    
    I = fPrior*fLikelihood
    
    posteriorSignal = ((likelihood*I)@wlMatrix) * ((prior*I)@wpMatrix)
    
    
    prediction = posteriorSignal[0]/(posteriorSignal[0]+posteriorSignal[1])
    
    return prediction

def l_w(L,w):
    f = np.log((w*np.exp(L)+1-w) / ((1-w)*np.exp(L)+w))
    return f

def compute_BIC(actual_c,prior,likelihood,params,k):
    n=100
    pred_c = np.zeros(n)
    sigma_squared = 0
    
    for t in range(n):
        
        pred_c[t] = circular_inference(prior[:,t],likelihood[:,t],params)
        sigma_squared += (pred_c[t] - actual_c[t])**2 #mean-squared-difference
    bic = n*np.log(sigma_squared) + k*np.log(n)
    return bic, sigma_squared


stan_model_weighted = """
    functions {
        real logit_stan(real p) {
            return log(p/(1-p));
        }
        real expit_stan(real x) {
            return 1/(1+exp(-x));
        }

        matrix weighted_matrix(real weight) {
            matrix[2,2] wMatrix;
            wMatrix[1,1] = weight;
            wMatrix[1,2] = 1-weight;
            wMatrix[2,1] = 1-weight;
            wMatrix[2,2] = weight;
            return wMatrix;
        }
        
        vector falw_function(real p, real q,real weight, real alpha, matrix wMatrix) {
            vector[2] amplified;
            real logit_p = log(p/q);
            amplified[1] = expit_stan(alpha*logit_p);
            amplified[2] = 1 - amplified[1];
            return wMatrix * amplified; 
        }
    }


    data {
        int T;
        real<lower=0> prior[T];
        real<lower=0> right_prior[T];
        real<lower=0> left_likelihood[T];
        real<lower=0> right_likelihood[T];
        real confidence[T];
    }


    parameters {
        real<lower=0.4,upper=1> wp;
        real<lower=0.4,upper=1> ws;
    }


    model {
        real lp[T];
        real aplp[T];
        real Faplpwp[T];
        real ls[T];
        real asls[T];
        real Faslsws[T];
        real lc[T];
        real pred_c[T];
        real lprob[T];
        real lpred[T];
        vector[2] likelihood_times_I;
        vector[2] prior_times_I;
        vector[2] posterior;
        real prediction[T];
        vector[2] prior_vec;
        vector[2] likelihood_vec;
        real scaled_wp;
        real scaled_ws;
        real scaled_alpha_p;
        real scaled_alpha_s;
        matrix[2,2] wpMatrix;
        matrix[2,2] wlMatrix;

        wp ~ uniform(0.4,1);
        ws ~ uniform(0.4,1);
        
        scaled_wp = wp;
        scaled_ws = ws;
        scaled_alpha_p = 0;
        scaled_alpha_s = 0;

        wpMatrix = weighted_matrix(scaled_wp);
        wlMatrix = weighted_matrix(scaled_ws);
        

        for (t in 1:T){
            vector[2] fPrior = falw_function(prior[t], right_prior[t],scaled_wp, scaled_alpha_p, wpMatrix);
            vector[2] fLikelihood = falw_function(left_likelihood[t], right_likelihood[t], scaled_ws, scaled_alpha_s, wlMatrix);

            vector[2] I = fPrior .* fLikelihood;              

            prior_vec[1] = prior[t];
            prior_vec[2] = right_prior[t];
            likelihood_vec[1] = left_likelihood[t];
            likelihood_vec[2] = right_likelihood[t];

            prior_times_I = prior_vec .* I;
            likelihood_times_I = likelihood_vec .* I;

            
            posterior = (wpMatrix * prior_times_I) .* (wlMatrix * likelihood_times_I);

            prediction[t] = posterior[1]/(posterior[1]+posterior[2]);
        
            lpred[t] = logit_stan(prediction[t]);
            lprob[t] = logit_stan(confidence[t]);

            target += -(lpred[t] - lprob[t])^2;
        }
    }

    """

stan_model_alpha = """
    functions {
        real logit_stan(real p) {
            return log(p/(1-p));
        }
        real expit_stan(real x) {
            return 1/(1+exp(-x));
        }

        matrix weighted_matrix(real weight) {
            matrix[2,2] wMatrix;
            wMatrix[1,1] = weight;
            wMatrix[1,2] = 1-weight;
            wMatrix[2,1] = 1-weight;
            wMatrix[2,2] = weight;
            return wMatrix;
        }
        
        vector falw_function(real p, real q,real weight, real alpha, matrix wMatrix) {
            vector[2] amplified;
            real logit_p = log(p/q);
            amplified[1] = expit_stan(alpha*logit_p);
            amplified[2] = 1 - amplified[1];
            return wMatrix * amplified; 
        }
    }


    data {
        int T;
        real<lower=0> prior[T];
        real<lower=0> right_prior[T];
        real<lower=0> left_likelihood[T];
        real<lower=0> right_likelihood[T];
        real confidence[T];
    }


    parameters {
        real<lower=0> alpha_p;
        real<lower=0> alpha_s;
    }


    model {
        real lp[T];
        real aplp[T];
        real Faplpwp[T];
        real ls[T];
        real asls[T];
        real Faslsws[T];
        real lc[T];
        real pred_c[T];
        real lprob[T];
        real lpred[T];
        vector[2] likelihood_times_I;
        vector[2] prior_times_I;
        vector[2] posterior;
        real prediction[T];
        vector[2] prior_vec;
        vector[2] likelihood_vec;
        real scaled_wp;
        real scaled_ws;
        real scaled_alpha_p;
        real scaled_alpha_s;
        matrix[2,2] wpMatrix;
        matrix[2,2] wlMatrix;

        alpha_p ~ exponential(5);
        alpha_s ~ exponential(5);
        
        scaled_wp = 0.8;
        scaled_ws = 0.8;
        scaled_alpha_p = alpha_p*10;
        scaled_alpha_s = alpha_s*10;

        wpMatrix = weighted_matrix(scaled_wp);
        wlMatrix = weighted_matrix(scaled_ws);
        

        for (t in 1:T){
            vector[2] fPrior = falw_function(prior[t], right_prior[t],scaled_wp, scaled_alpha_p, wpMatrix);
            vector[2] fLikelihood = falw_function(left_likelihood[t], right_likelihood[t], scaled_ws, scaled_alpha_s, wlMatrix);

            vector[2] I = fPrior .* fLikelihood;              

            prior_vec[1] = prior[t];
            prior_vec[2] = right_prior[t];
            likelihood_vec[1] = left_likelihood[t];
            likelihood_vec[2] = right_likelihood[t];

            prior_times_I = prior_vec .* I;
            likelihood_times_I = likelihood_vec .* I;

            
            posterior = (wpMatrix * prior_times_I) .* (wlMatrix * likelihood_times_I);

            prediction[t] = posterior[1]/(posterior[1]+posterior[2]);
        
            lpred[t] = logit_stan(prediction[t]);
            lprob[t] = logit_stan(confidence[t]);

            target += -(lpred[t] - lprob[t])^2;
        }
    }

    """

stan_model_circular = """
    functions {
        real logit_stan(real p) {
            return log(p/(1-p));
        }
        real expit_stan(real x) {
            return 1/(1+exp(-x));
        }

        matrix weighted_matrix(real weight) {
            matrix[2,2] wMatrix;
            wMatrix[1,1] = weight;
            wMatrix[1,2] = 1-weight;
            wMatrix[2,1] = 1-weight;
            wMatrix[2,2] = weight;
            return wMatrix;
        }
        
        vector falw_function(real p, real q,real weight, real alpha, matrix wMatrix) {
            vector[2] amplified;
            real logit_p = log(p/q);
            amplified[1] = expit_stan(alpha*logit_p);
            amplified[2] = 1 - amplified[1];
            return wMatrix * amplified; 
        }
    }


    data {
        int T;
        real<lower=0> prior[T];
        real<lower=0> right_prior[T];
        real<lower=0> left_likelihood[T];
        real<lower=0> right_likelihood[T];
        real confidence[T];
    }


    parameters {
        real<lower=0.4,upper=1> wp;
        real<lower=0.4,upper=1> ws;
        real<lower=0> alpha_p;
        real<lower=0> alpha_s;
    }


    model {
        real lp[T];
        real aplp[T];
        real Faplpwp[T];
        real ls[T];
        real asls[T];
        real Faslsws[T];
        real lc[T];
        real pred_c[T];
        real lprob[T];
        real lpred[T];
        vector[2] likelihood_times_I;
        vector[2] prior_times_I;
        vector[2] posterior;
        real prediction[T];
        vector[2] prior_vec;
        vector[2] likelihood_vec;
        real scaled_wp;
        real scaled_ws;
        real scaled_alpha_p;
        real scaled_alpha_s;
        matrix[2,2] wpMatrix;
        matrix[2,2] wlMatrix;

        wp ~ uniform(0.4,1);
        ws ~ uniform(0.4,1);
        alpha_p ~ exponential(5);
        alpha_s ~ exponential(5);
        
        scaled_wp = wp;
        scaled_ws = ws;
        scaled_alpha_p = 10*alpha_p;
        scaled_alpha_s = 10*alpha_s;

        wpMatrix = weighted_matrix(scaled_wp);
        wlMatrix = weighted_matrix(scaled_ws);
        

        for (t in 1:T){
            vector[2] fPrior = falw_function(prior[t], right_prior[t],scaled_wp, scaled_alpha_p, wpMatrix);
            vector[2] fLikelihood = falw_function(left_likelihood[t], right_likelihood[t], scaled_ws, scaled_alpha_s, wlMatrix);

            vector[2] I = fPrior .* fLikelihood;              

            prior_vec[1] = prior[t];
            prior_vec[2] = right_prior[t];
            likelihood_vec[1] = left_likelihood[t];
            likelihood_vec[2] = right_likelihood[t];

            prior_times_I = prior_vec .* I;
            likelihood_times_I = likelihood_vec .* I;

            
            posterior = (wpMatrix * prior_times_I) .* (wlMatrix * likelihood_times_I);

            prediction[t] = posterior[1]/(posterior[1]+posterior[2]);
        
            lpred[t] = logit_stan(prediction[t]);
            lprob[t] = logit_stan(confidence[t]);

            target += -(lpred[t] - lprob[t])^2;
        }
        //target += -0.00001*(alpha_p^2+alpha_s^2);
    }

"""



sm_weight = pystan.StanModel(model_code=stan_model_weighted)
sm_alpha = pystan.StanModel(model_code=stan_model_alpha)
sm_circular = pystan.StanModel(model_code=stan_model_circular)

data, image_data = preprocessing.preprocess()

# Change the likelihood value 
likelihood_dir='/Users/shuhei/Desktop/workspace/behavior/new_likelihood.csv'
like_data = pd.read_csv(likelihood_dir)

for i in range(1,101):
    data.loc[data['imgseq']==i,'correct_image'] = like_data.loc[like_data['im_number']==i,'categ1_ave'].values[0]
    data.loc[data['imgseq']==i,'false_image'] = like_data.loc[like_data['im_number']==i,'categ2_ave'].values[0]



sub_list = []
allsub = []
scz = []
con = []
number_sub = 21
# number_sub=2
sigma = 0
rng = np.random.default_rng(666)

estimated_wp = np.zeros(number_sub)
estimated_ws = np.zeros(number_sub)
estimated_alpha_p = np.zeros(number_sub)
estimated_alpha_s = np.zeros(number_sub)

parameters_mean = {
    'wp': estimated_wp,
    'ws': estimated_ws,
    'alpha_p': estimated_alpha_p,
    'alpha_s': estimated_alpha_s
}


weighted_matrix = pd.DataFrame(np.zeros((21,2)), index=range(21), columns=['wp','ws'])
alpha_matrix = pd.DataFrame(np.zeros((21,2)), index=range(21), columns=['alpha_p','alpha_s'])
circular_matrix = pd.DataFrame(np.zeros((21,4)), index=range(21), columns=['wp','ws','alpha_p','alpha_s'])


models = ['Simple','Weighted','Alpha','Circular']
bic_matrix = pd.DataFrame(np.zeros((21,4)), index=range(21), columns=models)
sigma_matrix = pd.DataFrame(np.zeros((21,4)), index=range(21), columns=models)

for sub in range(number_sub):
    print(sub)
    sub_data = data[data['subject']==sub]
    prior_list = np.zeros(100)
    right_likelihood_list = np.zeros(100)
    left_likelihood_list = np.zeros(100)
    c_list = np.zeros(100)

    for trial in range(100):
        prior_list[trial] = sub_data.at[(100*sub)+trial,'prior']

        #Likelihood depends on corloc
        if sub_data.at[(100*sub)+trial,'corloc']==1:
            left_likelihood_list[trial] = sub_data.at[(100*sub)+trial,'correct_image']
            right_likelihood_list[trial] = sub_data.at[(100*sub)+trial,'false_image']
        elif sub_data.at[(100*sub)+trial,'corloc']==2:
            left_likelihood_list[trial] = sub_data.at[(100*sub)+trial,'false_image']
            right_likelihood_list[trial] = sub_data.at[(100*sub)+trial,'correct_image']

        c_list[trial] = sub_data.at[(100*sub)+trial,'posterior']

    prior_list = [n/100 for n in prior_list]
    right_prior_list = [1-p for p in prior_list]
    left_like_list = [n/100 for n in left_likelihood_list]
    right_like_list = [n/100 for n in right_likelihood_list]

    prior_np = np.array((prior_list,right_prior_list))
    likelihood_np = np.array((left_like_list,right_like_list))

    c_list = [1 if c==0 else c for c in c_list]
    c_list = [99 if c==100 else c for c in c_list]
    c_list = [n/100 for n in c_list]
    c_np = np.array(c_list)

    # simple Bayes
    bic_simple, sigma_squared_simple = compute_BIC(c_np,prior_np,likelihood_np,[1,1,0,0],0)
    print('BIC simple',bic_simple)
    bic_matrix.iloc[sub,0] = bic_simple
    sigma_matrix.iloc[sub,0] = sigma_squared_simple

    # Weighted Bayes
    stan_weight_data = {"T":100,"prior":prior_list,"right_prior":right_prior_list ,"left_likelihood":left_like_list,"right_likelihood":right_like_list,"confidence":c_np}
    fit_weight = sm_weight.sampling(data=stan_weight_data,iter=2000,warmup=1000,chains=4,seed=123)#,control={'adapt_delta': 0.95})
    para_wei = [fit_weight['wp'].mean(),fit_weight['ws'].mean(),0,0]
    bic_weight, sigma_squared_weight = compute_BIC(c_np,prior_np,likelihood_np,para_wei,2)
    bic_matrix.iloc[sub,1] = bic_weight
    sigma_matrix.iloc[sub,1] = sigma_squared_weight   

    # Alpha model
    stan_alpha_data = {"T":100,"prior":prior_list,"right_prior":right_prior_list ,"left_likelihood":left_like_list,"right_likelihood":right_like_list,"confidence":c_np}
    fit_alpha = sm_alpha.sampling(data=stan_alpha_data,iter=2000,warmup=1000,chains=4,seed=123)
    para_alp = [0.8,0.8,fit_alpha['alpha_p'].mean(),fit_alpha['alpha_s'].mean()]
    bic_alpha, sigma_squared_alpha = compute_BIC(c_np,prior_np,likelihood_np,para_alp,2)
    bic_matrix.iloc[sub,2] = bic_alpha
    sigma_matrix.iloc[sub,2] = sigma_squared_alpha

    # Circular inference
    stan_circular_data = {"T":100,"prior":prior_list,"right_prior":right_prior_list ,"left_likelihood":left_like_list,"right_likelihood":right_like_list,"confidence":c_np}
    fit_circular = sm_circular.sampling(data=stan_circular_data,iter=2000,warmup=1000,chains=4,seed=123)
    para_cir = [fit_circular['wp'].mean(),fit_circular['ws'].mean(),fit_circular['alpha_p'].mean(),fit_circular['alpha_s'].mean()]
    bic_circular, sigma_squared_circular = compute_BIC(c_np,prior_np,likelihood_np,para_cir,4)
    bic_matrix.iloc[sub,3] = bic_circular
    sigma_matrix.iloc[sub,3] = sigma_squared_circular

    print(fit_circular)

    # take parameters
    weighted_matrix.iloc[sub,0] = fit_weight['wp'].mean()
    weighted_matrix.iloc[sub,1] = fit_weight['ws'].mean()

    alpha_matrix.iloc[sub,0] = fit_circular['alpha_p'].mean()
    alpha_matrix.iloc[sub,1] = fit_circular['alpha_s'].mean()

    circular_matrix.iloc[sub,0] = fit_circular['wp'].mean()
    circular_matrix.iloc[sub,1] = fit_circular['ws'].mean()
    circular_matrix.iloc[sub,2] = fit_circular['alpha_p'].mean()
    circular_matrix.iloc[sub,3] = fit_circular['alpha_s'].mean()


# subsub = list(itertools.chain.from_iterable(allsub))
    
# weighted
df_wei = pd.DataFrame(list(itertools.chain.from_iterable(weighted_matrix.values)),columns=['values'])
df_wei['parameters'] = 'a'
subject_ID = ['DI', 'HM', 'RM', 'KH', 'MF', 'MOt','FA', 'KT', 'SY', 'TY',
              'MN', 'NK', 'SK', 'NKu','MY', 'YA', 'TK', 'TN', 'HH', 'MYa', 'HK']

for i in range(21):
    df_wei.iat[2*i,1] = 'wp'
    df_wei.iat[2*i+1,1] = "ws"
    
    df_wei.loc[2*i:2*i+1, 'subject'] = i
    df_wei.loc[2*i:2*i+1, 'subject_ID'] = subject_ID[i]

df_wei.loc[0:19,'group'] = 'scz'
df_wei.loc[20:41,'group'] = 'con'

df_wei.to_csv('result_peggy/estimated_weighted.csv') 

sns.catplot(data=df_wei,x='parameters',y='values', kind='bar',hue='group',ci=68)
plt.savefig("result_peggy/estimated_parameters.jpg")
plt.show()

# Alpha
df_alp = pd.DataFrame(list(itertools.chain.from_iterable(alpha_matrix.values)),columns=['values'])
df_alp['parameters'] = 'a'
subject_ID = ['DI', 'HM', 'RM', 'KH', 'MF', 'MOt','FA', 'KT', 'SY', 'TY',
              'MN', 'NK', 'SK', 'NKu','MY', 'YA', 'TK', 'TN', 'HH', 'MYa', 'HK']

for i in range(21):
    df_alp.iat[2*i,1] = 'alpha_p'
    df_alp.iat[2*i+1,1] = "alpha_s"
    
    df_alp.loc[2*i:2*i+1, 'subject'] = i
    df_alp.loc[2*i:2*i+1, 'subject_ID'] = subject_ID[i]

df_alp.loc[0:19,'group'] = 'scz'
df_alp.loc[20:41,'group'] = 'con'

df_alp.to_csv('result_peggy/estimated_alpha.csv') 

sns.catplot(data=df_alp,x='parameters',y='values', kind='bar',hue='group',ci=68)
plt.savefig("result_peggy/estimated_alpha.jpg")
plt.show()



df_all = pd.DataFrame(list(itertools.chain.from_iterable(circular_matrix.values)),columns=['values'])
df_all['parameters'] = 'a'
subject_ID = ['DI', 'HM', 'RM', 'KH', 'MF', 'MOt','FA', 'KT', 'SY', 'TY',
              'MN', 'NK', 'SK', 'NKu','MY', 'YA', 'TK', 'TN', 'HH', 'MYa', 'HK']

for i in range(21):
    df_all.iat[4*i,1] = 'wp'
    df_all.iat[4*i+1,1] = "ws"
    df_all.iat[4*i+2,1] = 'alpha_p'
    df_all.iat[4*i+3,1] = "alpha_s"
    
    df_all.loc[4*i:4*i+3, 'subject'] = i
    df_all.loc[4*i:4*i+3, 'subject_ID'] = subject_ID[i]

df_all.loc[0:39,'group'] = 'scz'
df_all.loc[40:83,'group'] = 'con'

df_all.to_csv('result_peggy/estimated_circular.csv') 

sns.catplot(data=df_all,x='parameters',y='values', kind='bar',hue='group',ci=68)
plt.savefig("result_peggy/estimated_parameters.jpg")
plt.show()
    

sns.heatmap(bic_matrix,annot=True,cmap='viridis')
plt.savefig("result_peggy/bic_matrix.jpg")
plt.show()


# BIC all subjects
BIC_simple_all = 100*21*np.log(sum(sigma_matrix['Simple'])/21) + 0*np.log(100*21)
BIC_weighted_all = 100*21*np.log(sum(sigma_matrix['Weighted'])/21) + 2*np.log(100*21)
BIC_alpha_all = 100*21*np.log(sum(sigma_matrix['Alpha'])/21) + 2*np.log(100*21)
BIC_circular_all = 100*21*np.log(sum(sigma_matrix['Circular'])/21) + 4*np.log(100*21)
print(BIC_simple_all,BIC_weighted_all,BIC_alpha_all,BIC_circular_all)
BIC_simple_scz = 100*21*np.log(sum(sigma_matrix['Simple'][:10])/10) + 0*np.log(100*10)
BIC_weighted_scz = 100*21*np.log(sum(sigma_matrix['Weighted'][:10])/10) + 2*np.log(100*10)
BIC_alpha_scz = 100*21*np.log(sum(sigma_matrix['Alpha'][:10])/10) + 2*np.log(100*10)
BIC_circular_scz = 100*21*np.log(sum(sigma_matrix['Circular'][:10])/10) + 4*np.log(100*10)
print(BIC_simple_scz,BIC_weighted_scz,BIC_alpha_scz,BIC_circular_scz)
BIC_simple_con = 100*21*np.log(sum(sigma_matrix['Simple'][10:])/11) + 0*np.log(100*10)
BIC_weighted_con = 100*21*np.log(sum(sigma_matrix['Weighted'][10:])/11) + 2*np.log(100*10)
BIC_alpha_con = 100*21*np.log(sum(sigma_matrix['Alpha'][10:])/11) + 2*np.log(100*10)
BIC_circular_con = 100*21*np.log(sum(sigma_matrix['Circular'][10:])/11) + 4*np.log(100*10)
print(BIC_simple_con,BIC_weighted_con,BIC_alpha_con,BIC_circular_con)

relativeBIC_circular_all = BIC_simple_all - BIC_circular_all
relativeBIC_weight_all = BIC_simple_all - BIC_weighted_all
relativeBIC_alpha_all = BIC_simple_all - BIC_alpha_all
relativeBIC_circular_scz = BIC_simple_scz - BIC_circular_scz
relativeBIC_weight_scz = BIC_simple_scz - BIC_weighted_scz
relativeBIC_alpha_scz = BIC_simple_scz - BIC_alpha_scz
relativeBIC_circular_con = BIC_simple_con - BIC_circular_con
relativeBIC_weight_con = BIC_simple_con - BIC_weighted_con
relativeBIC_alpha_con = BIC_simple_con - BIC_alpha_con
labels = ['ALL', 'SCZ', 'CTL']
circular_means = [relativeBIC_circular_all, relativeBIC_circular_scz, relativeBIC_circular_con]
weight_means = [relativeBIC_weight_all, relativeBIC_weight_scz, relativeBIC_weight_con]
alpha_means = [relativeBIC_alpha_all, relativeBIC_alpha_scz, relativeBIC_alpha_con]
x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width, circular_means, width, label='Circular inference')
rects2 = ax.bar(x, weight_means, width, label='Weighted Bayes')
rects2 = ax.bar(x + width, alpha_means, width, label='Alpha')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('BIC (simple Bayes) - BIC (model)')
ax.set_title('Relative BIC score')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.savefig("result_peggy/relativeBIC.jpg")
plt.show()


pkl_dir = '/Users/shuhei/Desktop/Raw_data/demographic_data/psy_data.pkl'
psy_data = pd.read_pickle(pkl_dir)
for index, row in psy_data.iterrows():
    subject_id = row['ID']
    pdi_total = row['PDI_total']
    aq_total = row['AQ_total']
    df_all.loc[df_all['subject_ID'] == subject_id, 'PDI_total'] = pdi_total
    df_all.loc[df_all['subject_ID'] == subject_id, 'AQ_total'] = aq_total
sns.lmplot(data=df_all, x='PDI_total', y='values',hue = 'parameters')
plt.savefig("result_peggy/PDI_circular.jpg")
plt.show()
sns.lmplot(data=df_all, x='AQ_total', y='values',hue = 'parameters')
plt.savefig("result_peggy/AQ_circular.jpg")
plt.show()
for para in ['wp','ws','alpha_p','alpha_s']:
    paras = df_all.loc[df_all['parameters']==para,'values']
    vaa = df_all.loc[df_all['parameters']==para,'PDI_total']
    print(scipy.stats.pearsonr(paras,vaa))







