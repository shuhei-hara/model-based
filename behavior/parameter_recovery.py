import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import gaussian_kde,uniform, expon
import preprocessing
import pandas as pd
import pystan
from scipy.special import expit, logit
from scipy import stats

from matplotlib.lines import Line2D

stan_model = """
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
    //target += -0.00001*(alpha_p^2+alpha_s^2);
}

"""

sm = pystan.StanModel(model_code=stan_model)




data, image_data = preprocessing.preprocess()

# Change the likelihood value 
likelihood_dir='/Users/shuhei/Desktop/workspace/behavior/new_likelihood.csv'
like_data = pd.read_csv(likelihood_dir)

for i in range(1,101):
    data.loc[data['imgseq']==i,'correct_image'] = like_data.loc[like_data['im_number']==i,'categ1_ave'].values[0]
    data.loc[data['imgseq']==i,'false_image'] = like_data.loc[like_data['im_number']==i,'categ2_ave'].values[0]

def l_w(L,w):
    f = np.log((w*np.exp(L)+1-w) / ((1-w)*np.exp(L)+w))
    return f

sub_list = []
number_sub = 100
subject_number=0
sigma = 0
rng = np.random.default_rng(666)

true_wp_all = uniform.rvs(loc=0.5,scale=0.5,size=number_sub) #lower=0.4
true_ws_all = uniform.rvs(loc=0.5,scale=0.5,size=number_sub)
# true_alpha_p_all = expon.rvs(size=number_sub,scale=1/3)
# true_alpha_s_all = expon.rvs(size=number_sub,scale=1/3)
true_alpha_p_all = uniform.rvs(loc=0,scale=0.6,size=number_sub)
true_alpha_s_all = uniform.rvs(loc=0,scale=0.6,size=number_sub)


estimated_wp = np.zeros(number_sub)
estimated_ws = np.zeros(number_sub)
estimated_alpha_p = np.zeros(number_sub)
estimated_alpha_s = np.zeros(number_sub)
estimated_wp_mode = np.zeros(number_sub)
estimated_ws_mode = np.zeros(number_sub)
estimated_alpha_p_mode = np.zeros(number_sub)
estimated_alpha_s_mode = np.zeros(number_sub)

parameters_mean = {
    'wp': estimated_wp,
    'ws': estimated_ws,
    'alpha_p': estimated_alpha_p,
    'alpha_s': estimated_alpha_s
}


sub_data = data[data['subject']==0]
prior_list = np.zeros(100)
left_likelihood_list = np.zeros(100)
right_likelihood_list = np.zeros(100)
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

print(true_wp_all)
print(true_ws_all)
print(true_alpha_p_all)
print(true_alpha_s_all)


def circular_inference(prior, likelihood,param):
    
    pa1 = param[0]
    pa2 = param[1]
    pa3 = param[2]
    pa4 = param[3]

    wp = pa1
    wl = pa2
#     wp = pa1
#     wl=pa2
    
    ap = pa3*10
    al = pa4*10
    
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


# def circular_inference(prior, likelihood,param):

#     pa1 = param[0]
#     pa2 = param[1]
#     pa3 = param[2]
#     pa4 = param[3]

#     #prior logit
#     Cp = prior[0]
#     Fp = prior[1]
#     Lp = np.log(Cp/Fp)
#     aplp = pa3*Lp
#     Faplpwp = l_w(aplp,pa1)

#     #likelihood logit
#     Li_le = likelihood[0]
#     Li_ri = likelihood[1]
#     Ls = np.log(Li_le/Li_ri) 
#     asls = pa4*Ls
#     Faslsws = l_w(asls,pa2)

#     Lc = l_w(Ls+Faplpwp+Faslsws,pa2) + l_w(Lp+Faslsws+Faplpwp,pa1)
#     pred_c = np.exp(Lc)/(1+np.exp(Lc))

#     pred_c = np.clip(pred_c,0.018,0.982)
    

#     return pred_c#, Lc


trials=100
for sub_num,(true_wp,true_ws,true_alpha_p,true_alpha_s) in enumerate(zip(true_wp_all,true_ws_all,true_alpha_p_all,true_alpha_s_all)):
    print(sub_num)

    params = [true_wp,true_ws,true_alpha_p,true_alpha_s]
    prediction = np.zeros(trials)
#     Lc = np.zeros(trials)
    for t in range(trials):
        prediction[t] = circular_inference(prior_np[:,t],likelihood_np[:,t],params)
    
    
#     c_generated = np.where(c_generated<0.02,0.02,c_generated)
#     c_generated = np.where(c_generated>0.98,0.98,c_generated)
    

    
    
    stan_data = {"T":100,"prior":prior_list,"right_prior":right_prior_list ,"left_likelihood":left_like_list,"right_likelihood":right_like_list,"confidence":prediction}

    fit = sm.sampling(data=stan_data,iter=2000,warmup=1000,chains=4,seed=123)#,control={'adapt_delta': 0.95})

    print(fit)

    # fig = fit.plot()
    # plt.show()

    summary = fit.summary()
    summary_df = pd.DataFrame(summary['summary'],index=summary["summary_rownames"],columns=summary['summary_colnames'])

    # take mean of the distribution
    for para,estimated_dict in parameters_mean.items():
        estimated_dict[sub_num] = summary_df['mean'][para]



plt.scatter(true_wp_all,estimated_wp,label='wp')
plt.legend()
plt.xlabel('true')
plt.ylabel('predict')
plt.plot([0.4,1],[0.4,1])
plt.savefig("result_peggy/wp_parameter_recovery.jpg")
plt.show()

plt.scatter(true_ws_all,estimated_ws,label='ws')
plt.legend()
plt.xlabel('true')
plt.ylabel('predict')
plt.plot([0.4,1],[0.4,1])
plt.savefig("result_peggy/ws_parameter_recovery.jpg")
plt.show()


plt.scatter(true_alpha_p_all,estimated_alpha_p,label='alpha_p')
plt.legend()
plt.xlabel('true')
plt.ylabel('predict')
plt.plot([0,0.6],[0,0.6])
plt.savefig("result_peggy/alpha_p_parameter_recovery.jpg")
plt.show()

plt.scatter(true_alpha_s_all,estimated_alpha_s,label='alpha_s')
plt.legend()
plt.xlabel('true')
plt.ylabel('predict')
plt.plot([0,0.6],[0,0.6])
plt.savefig("result_peggy/alpha_s_parameter_recovery.jpg")
plt.show()

res_wp = stats.pearsonr(true_wp_all,estimated_wp)
print('corr wp: ', res_wp)
res_ws = stats.pearsonr(true_ws_all,estimated_ws)
print('corr ws: ', res_ws)
res_alpha_p = stats.pearsonr(true_alpha_p_all,estimated_alpha_p)
print('corr alpha_p: ', res_alpha_p)
res_alpha_s = stats.pearsonr(true_alpha_s_all,estimated_alpha_s)
print('corr alpha_s: ', res_alpha_s)
