import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import scipy.io
import seaborn as sns
import re

def preprocess():
    "all subjects"

    subject_ID = ['DI','HM','RM','KH','MF', 'MOt','FA','KT','SY','TY',
              'MN','NK','SK','NKu','MY','YA','TK','TN','HH','MYa','HK']

    n_subjects = len(subject_ID)
    n_scz = 10 #ここは指定
    n_con = n_subjects - n_scz
    pos = np.zeros((n_subjects*100,2))
    corloc = np.zeros((n_subjects*100,2))
    prior = np.zeros((n_subjects*100,2))
    imgseq = np.zeros((n_subjects*100,1))
    slider = np.zeros((n_subjects*100,1))

    for i,subject in enumerate(subject_ID):
        files = sorted(glob.glob('Response/'+subject+'/*'))
        pos_array = np.zeros((100, 2))
        corloc_array = np.zeros((100,2))
        prior_array = np.zeros((100,2))
        imgseq_array = np.zeros((100,1))
        slider_array = np.zeros((100,1))

        for n,f in enumerate(files):
            data =scipy.io.loadmat(f)

            pos_array[(20*n):(20*(n+1)), :] = data['PosteriorProb']
            corloc_array[(20*n):(20*(n+1)), :] = data['CorLoc'] #correct answer location[]
            imgseq_array[(20*n):(20*(n+1)), :] = data['ImgSeq']
            slider_array[(20*n):(20*(n+1)), :] = data['SliderStartPos']
            for a,iii in enumerate(np.squeeze(data['StimPriorRun'])):
                prior_array[(20*n)+a , :] = np.squeeze(iii)

        pos[(100*i):(100*(i+1)),:] = pos_array
        corloc[(100*i):(100*(i+1)),:] = corloc_array
        prior[(100*i):(100*(i+1)),:] = prior_array
        imgseq[(100*i):(100*(i+1)),:] = imgseq_array
        slider[(100*i):(100*(i+1)),:] = slider_array

    index = ['posterior','corloc','prior','imgseq','slider']
    df = pd.DataFrame([pos[:,0],corloc[:,0],prior[:,0],imgseq[:,0],slider[:,0]],index=index).T

    ##append (subject) colum

    for n in range(len(subject_ID)):
        df.loc[100*n:100*(n+1)-1,'subject'] = n

    df.loc[0:n_scz*100-1,'group'] = 'scz'
    df.loc[n_scz*100:n_subjects*100-1,'group'] = 'control'

    df.loc[df['corloc'] == 1, 'correct_rate'] =df['posterior']
    df.loc[df['corloc'] == 2, 'correct_rate'] = 100 - df['posterior']

    df.loc[df['corloc']==1,'congruency'] = df['prior']
    df.loc[df['corloc']==2,'congruency'] = 100 - df['prior']

######

    index = ['number']
    df_image = pd.DataFrame([np.arange(1,101)],index=index).T

    df_image['correct_rate'] = 0
    df_image['con_correct_rate'] = 0 #control correct rate
    for i in range(100):
        df_image.at[i,'correct_rate'] = df[df.imgseq == i+1].correct_rate.mean()
        df_image.at[i,'con_correct_rate'] = df[(df.group=='control') & (df.imgseq==i+1)].correct_rate.mean()
        df.loc[df.imgseq==i+1, 'correct_image'] = df[df.imgseq ==i+1].correct_rate.mean() #adding average posterior for true image to df

    sort = sorted(df_image['correct_rate'])
    for i in range(100):
        df_image.loc[df_image['correct_rate']==sort[i],'order_correct'] = i+1

        #prior congruency for image
        a = df.loc[df['imgseq'] == i+1, 'congruency']
        df_image.loc[df_image['number']==i+1,'congruency'] = a.iat[0]

#######
    imgfiles = glob.glob('ImgSet/Full/*')
    img_dictionary = {}

    for i, image in enumerate(imgfiles):
        number = re.findall('[0-9]+', image) #number[0]:imagseq, image[1]:blur level
        img_dictionary[int(number[0])] = int(number[1])

    for i in range(100):
        df_image.loc[df_image['number'] == i+1, 'blur'] = img_dictionary[i+1]
        df.loc[df['imgseq'] == i+1, 'blur'] = img_dictionary[i+1]

    return df, df_image

def prior_posterior(data):
    sns.catplot(x='prior',y='posterior',data=data,kind='point',hue='group',dodge=True,join=False,capsize=.2)
    plt.show()

def posterior_correct(data):
    sns.catplot(x='posterior',y='corloc',data=data,kind='point',hue='group',dodge=True,join=False,capsize=.2)
    plt.show()

def congruency(data):
    sns.catplot(data=data,x='congruency',y='correct_rate',kind='bar',hue='group')
    plt.show()

def correct_posterior(data):
    df_s = data.sort_values('order_correct')
    order= df_s['number']

    sns.lineplot(data=data,x='order_correct',y='correct_rate')
    plt.xticks(np.arange(100),order,rotation=90)
    plt.show()

def congruency(data):
    sns.boxplot(data=data,x='congruency',y='correct_rate',hue='group')
    plt.show()

def blur_level(data):
    sns.catplot(x='blur',y='correct_rate',data=data,kind='point',hue='group',dodge=True,join=False,capsize=.2)
    plt.show()

if __name__ == '__main__':
    pd.set_option("display.max_columns",None)
    pd.set_option("display.max_rows",None)
    data,data_per_image = preprocess()
    # print(data)
    print(data_per_image)
    prior_posterior(data)
    posterior_correct(data)
    congruency(data)
    correct_posterior(data_per_image)
    blur_level(data)
