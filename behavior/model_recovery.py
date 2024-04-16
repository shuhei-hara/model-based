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

data, image_data = preprocessing.preprocess()

# Change the likelihood value 
likelihood_dir= # likelihood directory
like_data = pd.read_csv(likelihood_dir)

for i in range(1,101):
    data.loc[data['imgseq']==i,'correct_image'] = like_data.loc[like_data['im_number']==i,'categ1_ave'].values[0]
    data.loc[data['imgseq']==i,'false_image'] = like_data.loc[like_data['im_number']==i,'categ2_ave'].values[0]

def l_w(L,w):
    f = np.log((w*np.exp(L)+1-w) / ((1-w)*np.exp(L)+w))
    return f

# 4 models: simple, weighted, alpha, circular model

sub_list = []
number_sub = 50
subject_number = 0

#weighted
true_wp_weighted = uniform.rvs(loc=0.4,scale=0.6,size=number_sub) #lower=0.4
true_ws_weighted = uniform.rvs(loc=0.4,scale=0.6,size=number_sub)
#alpha
true_alpha_p_alpha = uniform.rvs(loc=0,scale=0.6,size=number_sub)
true_alpha_s_alpha = uniform.rvs(loc=0,scale=0.6,size=number_sub)
#circular inference
true_wp_all = uniform.rvs(loc=0.4,scale=0.6,size=number_sub) #lower=0.4
true_ws_all = uniform.rvs(loc=0.4,scale=0.6,size=number_sub)
true_alpha_p_all = uniform.rvs(loc=0,scale=0.6,size=number_sub)
true_alpha_s_all = uniform.rvs(loc=0,scale=0.6,size=number_sub)

# weighted model
estimated_wp_weighted = np.zeros(number_sub)
estimated_ws_weighted = np.zeros(number_sub)
# alpha model
estimated_alpha_p_alpha = np.zeros(number_sub)
estimated_alpha_s_alpha = np.zeros(number_sub)
#circular inference
estimated_wp = np.zeros(number_sub)
estimated_ws = np.zeros(number_sub)
estimated_alpha_p = np.zeros(number_sub)
estimated_alpha_s = np.zeros(number_sub)


weighted_paras = {
    'wp': estimated_wp_weighted,
    'ws': estimated_ws_weighted
}

alpha_paras = {
    'alpha_p': estimated_alpha_p_alpha,
    'alpha_s': estimated_alpha_s_alpha
}

circular_paras = {
    'wp': estimated_wp,
    'ws': estimated_ws,
    'alpha_p': estimated_alpha_p,
    'alpha_s': estimated_alpha_s
}


sub_data = data[data['subject']==0]
prior_list = np.zeros(100)
right_likelihood_list = np.zeros(100)
left_likelihood_list = np.zeros(100)
c_list = np.zeros(100)

for trial in range(100):
    prior_list[trial] = sub_data.at[(100*subject_number)+trial,'prior']

    #Likelihood depends on corloc
    if sub_data.at[(100*subject_number)+trial,'corloc']==1:
        left_likelihood_list[trial] = sub_data.at[(100*subject_number)+trial,'correct_image']
        right_likelihood_list[trial] = sub_data.at[(100*subject_number)+trial,'false_image']
    elif sub_data.at[(100*subject_number)+trial,'corloc']==2:
        left_likelihood_list[trial] = sub_data.at[(100*subject_number)+trial,'false_image']
        right_likelihood_list[trial] = sub_data.at[(100*subject_number)+trial,'correct_image']
    
    # c_list[trial] = sub_data.at[(100*subject_number)+trial,'posterior']

prior_list = [n/100 for n in prior_list]
right_prior_list = [1-p for p in prior_list]
left_like_list = [n/100 for n in left_likelihood_list]
right_like_list = [n/100 for n in right_likelihood_list]

prior_np = np.array((prior_list,right_prior_list))
likelihood_np = np.array((left_like_list,right_like_list))

# simple model
c_generated_simple = np.zeros((number_sub,100))
for sub_num in range(number_sub):
    print(sub_num)

    params = [1,1,0,0]
    for t in range(100):
        c_generated_simple[sub_num,t] = circular_inference(prior_np[:,t],likelihood_np[:,t],params)

# weighted model
c_generated_weighted = np.zeros((number_sub,100))
for sub_num,(true_wp_weighted,true_ws_weighted) in enumerate(zip(true_wp_weighted,true_ws_weighted)):
    print(sub_num)

    params = [true_wp_weighted,true_ws_weighted,0,0]
    for t in range(100):
        c_generated_weighted[sub_num,t] = circular_inference(prior_np[:,t],likelihood_np[:,t],params)

# alpha model
c_generated_alpha = np.zeros((number_sub,100))
for sub_num,(alpha_p_alpha,alpha_s_alpha) in enumerate(zip(true_alpha_p_alpha,true_alpha_p_alpha)):
    print(sub_num)

    params = [0.8,0.8,alpha_p_alpha,alpha_s_alpha]
    for t in range(100):
        c_generated_alpha[sub_num,t] = circular_inference(prior_np[:,t],likelihood_np[:,t],params)

# circular model
c_generated_circular = np.zeros((number_sub,100))
for sub_num,(true_wp,true_ws,true_alpha_p,true_alpha_s) in enumerate(zip(true_wp_all,true_ws_all,true_alpha_p_all,true_alpha_s_all)):
    print(sub_num)
    params = [true_wp,true_ws,true_alpha_p,true_alpha_s]

    for t in range(100):
        c_generated_circular[sub_num,t] = circular_inference(prior_np[:,t],likelihood_np[:,t],params)


def compute_BIC(actual_c,prior,likelihood,params,k):
    n=100
    pred_c = np.zeros(n)
    sigma_squared = 0
    
    for t in range(n):
        
        pred_c[t] = circular_inference(prior[:,t],likelihood[:,t],params)
        sigma_squared += (pred_c[t] - actual_c[t])**2 #mean-squared-difference
    bic = n*np.log(sigma_squared) + k*np.log(n)
    return bic, sigma_squared


rng = np.random.default_rng(666)

# for sub_num,(true_wp,true_ws,true_alpha_p,true_alpha_s) in enumerate(zip(true_wp_all,true_ws_all,true_alpha_p_all,true_alpha_s_all)):
models = ['Simple','Weighted','Alpha','Circular']
confusion_matrix = pd.DataFrame(np.zeros((4,4)), index=models, columns=models)
# bic_matrix = pd.DataFrame(np.zeros((number_sub,4)), columns=models)
# sigma_matrix = pd.DataFrame(np.zeros((number_sub,4)), columns=models)

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
        scaled_alpha_p = alpha_p*20;
        scaled_alpha_s = alpha_s*20;

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
        scaled_alpha_p = alpha_p*20;
        scaled_alpha_s = alpha_s*20;

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



for model_i,ob in enumerate([c_generated_simple,c_generated_weighted,c_generated_alpha,c_generated_circular]):
    print('model: ',model_i)
    for sub in range(number_sub):
        print('sub',sub)

        ob_each = ob[sub]
 
        # simple
        bic_simple, sigma_squared_simple = compute_BIC(ob_each,prior_np,likelihood_np,[1,1,0,0],0)
        # bic_matrix.iloc[sub,0] = bic_simple
        # sigma_matrix.iloc[sub,0] = sigma_squared_simple

        stan_weight_data = {"T":100,"prior":prior_list,"right_prior":right_prior_list ,"left_likelihood":left_like_list,"right_likelihood":right_like_list,"confidence":ob_each}
        fit_weight = sm_weight.sampling(data=stan_weight_data,iter=2000,warmup=1000,chains=4,seed=123)#,control={'adapt_delta': 0.95})
        para_wei = [fit_weight['wp'].mean(),fit_weight['ws'].mean(),0,0]
        bic_weight, sigma_squared_weight = compute_BIC(ob_each,prior_np,likelihood_np,para_wei,2)
        # bic_matrix.iloc[sub,1] = bic_weight
        # sigma_matrix.iloc[sub,1] = sigma_squared_weight

        stan_alpha_data = {"T":100,"prior":prior_list,"right_prior":right_prior_list ,"left_likelihood":left_like_list,"right_likelihood":right_like_list,"confidence":ob_each}
        fit_alpha = sm_alpha.sampling(data=stan_alpha_data,iter=2000,warmup=1000,chains=4,seed=123)#,control={'adapt_delta': 0.95})
        para_alp = [0.8,0.8,fit_alpha['alpha_p'].mean(),fit_alpha['alpha_s'].mean()]
        bic_alpha, sigma_squared_alpha = compute_BIC(ob_each,prior_np,likelihood_np,para_alp,2)
        # bic_matrix.iloc[sub,2] = bic_alpha
        # sigma_matrix.iloc[sub,2] = sigma_squared_alpha

        stan_circular_data = {"T":100,"prior":prior_list,"right_prior":right_prior_list ,"left_likelihood":left_like_list,"right_likelihood":right_like_list,"confidence":ob_each}
        fit_circular = sm_circular.sampling(data=stan_circular_data,iter=2000,warmup=1000,chains=4,seed=123)#,control={'adapt_delta': 0.95})
        para_cir = [fit_circular['wp'].mean(),fit_circular['ws'].mean(),fit_circular['alpha_p'].mean(),fit_circular['alpha_s'].mean()]
        bic_circular, sigma_squared_circular = compute_BIC(ob_each,prior_np,likelihood_np,para_cir,4)
        # bic_matrix.iloc[sub,3] = bic_circular
        # sigma_matrix.iloc[sub,3] = sigma_squared_circular

        compare_dict = {"Simple":bic_simple, "Weighted": bic_weight, "Alpha": bic_alpha,"Circular": bic_circular}

        if min(compare_dict, key=compare_dict.get)=='Simple':
            confusion_matrix.iloc[model_i,0] += 1
        elif min(compare_dict, key=compare_dict.get)=='Weighted':
            confusion_matrix.iloc[model_i,1] += 1
        elif min(compare_dict, key=compare_dict.get)=='Alpha':
            confusion_matrix.iloc[model_i,2] += 1
        elif min(compare_dict, key=compare_dict.get)=='Circular':
            confusion_matrix.iloc[model_i,3] += 1

for i in range(4):
    confusion_matrix.iloc[i,:] = confusion_matrix.iloc[i,:]/sum(confusion_matrix.iloc[i,:])

sns.heatmap(confusion_matrix,annot=True,cmap='viridis')
plt.xlabel('fit model')
plt.ylabel('simiulated model')
plt.savefig("{output directory}/model_recovery.jpg")
plt.show()

# sns.heatmap(bic_matrix,annot=True,cmap='viridis')
# plt.title('BIC matrix')
# plt.show()

# sns.heatmap(sigma_matrix,annot=True,cmap='viridis')
# plt.title('sigma matrix')
# plt.show()
