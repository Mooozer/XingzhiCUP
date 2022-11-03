# !/usr/bin/env python
# coding: utf-8

import pandas as pd
import json
import re
import numpy as np
import torch
import collections
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import os
from scipy.special import softmax
import torch.nn.functional as F
import torch.nn as nn
import zhon
import string
from zhon.hanzi import punctuation as CHN_punctuation

ENG_punctuation =  string.punctuation
p_list = list(ENG_punctuation) + list(CHN_punctuation)
stopwordsFile = open("./baidu_stopwords.txt", "r")
baidu_stopwords = stopwordsFile.read()
stopwords = baidu_stopwords.split('\n')
rm_tokens = stopwords + p_list
from LAC import LAC
lac = LAC(mode='rank')

if torch.cuda.is_available():
    device = torch.device("cuda")
else:    
    device = torch.device("cpu") 




#function for calculating accuracy
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

#function for timing
import time
import datetime
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

#function for calculating Levenshtein ratio 
import Levenshtein
def Levenshtein_similarity(string1, string2):
    '''
    output: scalar, Levenshtein similarity of string1 & string2
    '''
    Levenshtein_ratio = Levenshtein.ratio(string1, string2)
    return(Levenshtein_ratio)



# prepare test data
LCQMC_testB_dic = {'id':[], 'query':[], 'title':[], 'text_q_seg':[],'text_t_seg':[] } 
with open(r"./data/sim_interpretation_B.txt", 'r') as f:
    for line in f:
        line_dic = json.loads(line)
        for k in line_dic.keys():
            LCQMC_testB_dic[k].append(line_dic[k]) 

LCQMC_testB = pd.DataFrame.from_dict(LCQMC_testB_dic)
LCQMC_testB['sentence'] = LCQMC_testB['query'] +"[SEP]" + LCQMC_testB['title']
LCQMC_testB



# Bayes Posterior Iteration (BPI) Prediction


# Model 1 (macbert_large1 + LW, epoch=3)
ENG_punctuation =  string.punctuation
p_list = list(ENG_punctuation) + list(CHN_punctuation) 
stopwordsFile = open("./baidu_stopwords.txt", "r")
baidu_stopwords = stopwordsFile.read()  
stopwords = baidu_stopwords.split('\n')
rm_tokens = p_list + stopwords
def refine_sen(sen):
    refine_sen = ''.join([i for i in list(sen) if i not in rm_tokens])
    return refine_sen

def update_prob(p, pie):
    p = p*pie/(p*pie+(1-p)*(1-pie))
    return p 
          
logits_B = np.concatenate(np.load('./predictionB/logits13_B_test.npy',allow_pickle=True),axis=0)
prob_B = softmax(logits_B, axis=1) 
predB = np.argmax(logits_B ,axis=1)
BayesDataB = LCQMC_testB.copy()[['id']]

fined_sq,fined_st = [],[]
for i in range(len(LCQMC_testB)):
    fined_sq.append(LCQMC_testB['query'][i])
    fined_st.append(LCQMC_testB['title'][i]) 
BayesDataB['pred_step0'] = predB
BayesDataB['prob_step0'] = prob_B[:,1]
BayesDataB['fined_sq'] = [refine_sen(s) for s in fined_sq]
BayesDataB['fined_st'] = [refine_sen(s) for s in fined_st]

BayesQ = {}
for i in range(len(BayesDataB)):
    if BayesDataB['fined_sq'][i] not in BayesQ:
        BayesQ[BayesDataB['fined_sq'][i]] = [1, BayesDataB['prob_step0'][i], BayesDataB['prob_step0'][i]/1]
    else: 
        BayesQ[BayesDataB['fined_sq'][i]][0]+=1
        BayesQ[BayesDataB['fined_sq'][i]][1]+=BayesDataB['prob_step0'][i]
        BayesQ[BayesDataB['fined_sq'][i]][2]= BayesQ[BayesDataB['fined_sq'][i]][1]/BayesQ[BayesDataB['fined_sq'][i]][0]
    if BayesDataB['fined_st'][i] not in BayesQ:
        BayesQ[BayesDataB['fined_st'][i]] = [1, BayesDataB['prob_step0'][i], BayesDataB['prob_step0'][i]/1]
    else: 
        BayesQ[BayesDataB['fined_st'][i]][0]+=1
        BayesQ[BayesDataB['fined_st'][i]][1]+=BayesDataB['prob_step0'][i]
        BayesQ[BayesDataB['fined_st'][i]][2]= BayesQ[BayesDataB['fined_st'][i]][1]/BayesQ[BayesDataB['fined_st'][i]][0]

for b in range(14):
    print(f"bayes iteration{b}")
    pred_next = []
    prob_next = []
    for i in range(len(BayesDataB)):
        pie1 = max(BayesQ[BayesDataB['fined_sq'][i]][2], BayesQ[BayesDataB['fined_st'][i]][2])
        p1 = BayesDataB['prob_step'+str(b)][i]
        p1_update = update_prob(p1, pie1)
        pred_next.append(int((p1_update>0.5)==True))
        prob_next.append(p1_update)
    BayesDataB['pred_step'+str(b+1)] = pred_next
    BayesDataB['prob_step'+str(b+1)] = prob_next
print(f"bayes iteration done")

pred_record13 = pred_next


# Model 2 (macbert_large2 + LW, epoch=2) 
ENG_punctuation =  string.punctuation
p_list = list(ENG_punctuation) + list(CHN_punctuation) 
stopwordsFile = open("./baidu_stopwords.txt", "r")
baidu_stopwords = stopwordsFile.read()  
stopwords = baidu_stopwords.split('\n')
rm_tokens = p_list + stopwords
def refine_sen(sen):
    refine_sen = ''.join([i for i in list(sen) if i not in rm_tokens])
    return refine_sen

def update_prob(p, pie):
    p = p*pie/(p*pie+(1-p)*(1-pie))
    return p 
          
logits_B = np.concatenate(np.load('./predictionB/logits22_B_test.npy',allow_pickle=True),axis=0)
prob_B = softmax(logits_B, axis=1) 
predB = np.argmax(logits_B ,axis=1)
BayesDataB = LCQMC_testB.copy()[['id']]

fined_sq,fined_st = [],[]
for i in range(len(LCQMC_testB)):
    fined_sq.append(LCQMC_testB['query'][i])
    fined_st.append(LCQMC_testB['title'][i]) 
BayesDataB['pred_step0'] = predB
BayesDataB['prob_step0'] = prob_B[:,1]
BayesDataB['fined_sq'] = [refine_sen(s) for s in fined_sq]
BayesDataB['fined_st'] = [refine_sen(s) for s in fined_st]

BayesQ = {}
for i in range(len(BayesDataB)):
    if BayesDataB['fined_sq'][i] not in BayesQ:
        BayesQ[BayesDataB['fined_sq'][i]] = [1, BayesDataB['prob_step0'][i], BayesDataB['prob_step0'][i]/1]
    else: 
        BayesQ[BayesDataB['fined_sq'][i]][0]+=1
        BayesQ[BayesDataB['fined_sq'][i]][1]+=BayesDataB['prob_step0'][i]
        BayesQ[BayesDataB['fined_sq'][i]][2]= BayesQ[BayesDataB['fined_sq'][i]][1]/BayesQ[BayesDataB['fined_sq'][i]][0]
    if BayesDataB['fined_st'][i] not in BayesQ:
        BayesQ[BayesDataB['fined_st'][i]] = [1, BayesDataB['prob_step0'][i], BayesDataB['prob_step0'][i]/1]
    else: 
        BayesQ[BayesDataB['fined_st'][i]][0]+=1
        BayesQ[BayesDataB['fined_st'][i]][1]+=BayesDataB['prob_step0'][i]
        BayesQ[BayesDataB['fined_st'][i]][2]= BayesQ[BayesDataB['fined_st'][i]][1]/BayesQ[BayesDataB['fined_st'][i]][0]

for b in range(14):
    print(f"bayes iteration{b}")
    pred_next = []
    prob_next = []
    for i in range(len(BayesDataB)):
        pie1 = max(BayesQ[BayesDataB['fined_sq'][i]][2], BayesQ[BayesDataB['fined_st'][i]][2])
        p1 = BayesDataB['prob_step'+str(b)][i]
        p1_update = update_prob(p1, pie1)
        pred_next.append(int((p1_update>0.5)==True))
        prob_next.append(p1_update)
    BayesDataB['pred_step'+str(b+1)] = pred_next
    BayesDataB['prob_step'+str(b+1)] = prob_next
print(f"bayes iteration done")

pred_record22 = pred_next



# Model 3 (macbert_large3 + LW, epoch=3)

ENG_punctuation =  string.punctuation
p_list = list(ENG_punctuation) + list(CHN_punctuation) 
stopwordsFile = open("./baidu_stopwords.txt", "r")
baidu_stopwords = stopwordsFile.read()  
stopwords = baidu_stopwords.split('\n')
rm_tokens = p_list + stopwords
def refine_sen(sen):
    refine_sen = ''.join([i for i in list(sen) if i not in rm_tokens])
    return refine_sen

def update_prob(p, pie):
    p = p*pie/(p*pie+(1-p)*(1-pie))
    return p 
          
logits_B = np.concatenate(np.load('./predictionB/logits33_B_test.npy',allow_pickle=True),axis=0)
prob_B = softmax(logits_B, axis=1) 
predB = np.argmax(logits_B ,axis=1)
BayesDataB = LCQMC_testB.copy()[['id']]

fined_sq,fined_st = [],[]
for i in range(len(LCQMC_testB)):
    fined_sq.append(LCQMC_testB['query'][i])
    fined_st.append(LCQMC_testB['title'][i]) 
BayesDataB['pred_step0'] = predB
BayesDataB['prob_step0'] = prob_B[:,1]
BayesDataB['fined_sq'] = [refine_sen(s) for s in fined_sq]
BayesDataB['fined_st'] = [refine_sen(s) for s in fined_st]

BayesQ = {}
for i in range(len(BayesDataB)):
    if BayesDataB['fined_sq'][i] not in BayesQ:
        BayesQ[BayesDataB['fined_sq'][i]] = [1, BayesDataB['prob_step0'][i], BayesDataB['prob_step0'][i]/1]
    else: 
        BayesQ[BayesDataB['fined_sq'][i]][0]+=1
        BayesQ[BayesDataB['fined_sq'][i]][1]+=BayesDataB['prob_step0'][i]
        BayesQ[BayesDataB['fined_sq'][i]][2]= BayesQ[BayesDataB['fined_sq'][i]][1]/BayesQ[BayesDataB['fined_sq'][i]][0]
    if BayesDataB['fined_st'][i] not in BayesQ:
        BayesQ[BayesDataB['fined_st'][i]] = [1, BayesDataB['prob_step0'][i], BayesDataB['prob_step0'][i]/1]
    else: 
        BayesQ[BayesDataB['fined_st'][i]][0]+=1
        BayesQ[BayesDataB['fined_st'][i]][1]+=BayesDataB['prob_step0'][i]
        BayesQ[BayesDataB['fined_st'][i]][2]= BayesQ[BayesDataB['fined_st'][i]][1]/BayesQ[BayesDataB['fined_st'][i]][0]

for b in range(14):
    print(f"bayes iteration{b}")
    pred_next = []
    prob_next = []
    for i in range(len(BayesDataB)):
        pie1 = max(BayesQ[BayesDataB['fined_sq'][i]][2], BayesQ[BayesDataB['fined_st'][i]][2])
        p1 = BayesDataB['prob_step'+str(b)][i]
        p1_update = update_prob(p1, pie1)
        pred_next.append(int((p1_update>0.5)==True))
        prob_next.append(p1_update)
    BayesDataB['pred_step'+str(b+1)] = pred_next
    BayesDataB['prob_step'+str(b+1)] = prob_next
print(f"bayes iteration done")

pred_record33 = pred_next





# Final Prediction: Three MacBERT-large + LW + BPI joint prediction
together = np.array(pred_record13) + np.array(pred_record22) + np.array(pred_record33)
pred_together = together >=3
pred_together = [1 if i else 0 for i in pred_together]
print('final prediction:')
print(np.array(pred_together))



# Lexical Analysis of Chinese (LAC)
from LAC import LAC
lac = LAC(mode='rank')


#define a funtion
def reorder(score, token_index, mode='abs'):
    '''
    input
        score: list
        token_index: list
        mode: ['abs' or 'sequence' or 'token_seq']
    output
        reordered score and token lists
    '''
    assert len(score) == len(token_index)
    output = {}
    if mode =='abs':
        abs_score = [abs(i) for i in score]
        abs_dic = dict(zip(token_index, abs_score))
        sorted_abs_dic = dict(sorted(abs_dic.items(), key=lambda item: item[1], reverse=True))
        output['sorted_token_index'] = list(sorted_abs_dic.keys())
        output['sorted_score'] = list(sorted_abs_dic.values())
    
    if mode =='sequence':
        score_dic = dict(zip(token_index, score))
        sorted_score_dic = dict(sorted(score_dic.items(), key=lambda item: item[1], reverse=True))
        output['sorted_token_index'] = list(sorted_score_dic.keys())
        output['sorted_score'] = list(sorted_score_dic.values())

    if mode =='token_seq':
        abs_dic = dict(zip(token_index, score))
        sorted_abs_dic = dict(sorted(abs_dic.items(), key=lambda item: item[0], reverse=False))
        output['sorted_token_index'] = list(sorted_abs_dic.keys())
        output['sorted_score'] = list(sorted_abs_dic.values())
        
    return output

def single_LAC_expand(LAC_result):
    '''
    input: single LAC_result: list of list: [LAC_token, LAC_ner, LAC_imp]
    output: expanded LAC_result: list of list: [LAC_token, LAC_ner, LAC_imp, num]
    '''
    number = [len(list(i)) for i in LAC_result[0]]
    expand_LAC_token = sum([list(i) for i in LAC_result[0]],[])
    expand_LAC_enr = sum([[LAC_result[1][i]]*number[i] for i in range(len(LAC_result[1]))],[])
    expand_LAC_imp =  sum([[LAC_result[2][i]]*number[i] for i in range(len(LAC_result[2]))],[])
    
    return [expand_LAC_token, expand_LAC_enr, expand_LAC_imp, number]


# Obtain lime_output_plus dictionary (this dictionary contains all the scores for interpretation)
import numpy as np
import scipy.stats as ss

lac = LAC(mode='rank')

lime_output_plus = np.load('./lime_outputB/output.npy',allow_pickle=True).item()
lime_output_plus['token'] = []
lime_output_plus['query'] = []
lime_output_plus['title'] = []
lime_output_plus['ner'] = []
lime_output_plus['imp'] = []
lime_output_plus['num'] = []
lime_output_plus['piece'] = []
lime_output_plus['laclime'] = [] #LAC-wise mean of token lime score 
lime_output_plus['rationale_score_s'] = lime_output_plus['rationale_score'] #shap score 
lime_output_plus['laclime_rank'] = [] 
lime_output_plus['ori_laclime_rank'] = []
lime_output_plus['lime_rank'] = []
lime_output_plus['imp_rank'] = []

for i in range(len(lime_output_plus['id'])):  #len(lime_output_plus['id'])
    q = reorder(lime_output_plus['rationale_score'][i][0], lime_output_plus['rationale'][i][0], mode='token_seq')
    t = reorder(lime_output_plus['rationale_score'][i][1], lime_output_plus['rationale'][i][1], mode='token_seq')

    lime_output_plus['rationale'][i] = [q['sorted_token_index'], t['sorted_token_index']]
    lime_output_plus['rationale_score'][i] = [q['sorted_score'], t['sorted_score']]    

    lime_output_plus['token'].append([LCQMC_testB['text_q_seg'][i], LCQMC_testB['text_t_seg'][i]])
    lime_output_plus['query'].append(LCQMC_testB['query'][i])
    lime_output_plus['title'].append(LCQMC_testB['title'][i])
    LAC_result_q, LAC_result_t = lac.run(LCQMC_testB['query'][i]), lac.run(LCQMC_testB['title'][i])
    lime_output_plus['piece'].append([LAC_result_q[0],LAC_result_t[0]])
    LAC_result_q, LAC_result_t = single_LAC_expand(LAC_result_q) ,single_LAC_expand(LAC_result_t) 

    LAC_token_q, LAC_token_t = LAC_result_q[0], LAC_result_t[0]
    LAC_ner_q, LAC_ner_t = LAC_result_q[1], LAC_result_t[1]
    LAC_imp_q, LAC_imp_t = LAC_result_q[2], LAC_result_t[2]
    LAC_num_q, LAC_num_t = LAC_result_q[3], LAC_result_t[3]
    
    
    if LCQMC_testB['text_q_seg'][i] == LAC_token_q and LCQMC_testB['text_t_seg'][i] == LAC_token_t:
        lime_output_plus['ner'].append([LAC_ner_q, LAC_ner_t])
        lime_output_plus['imp'].append([LAC_imp_q, LAC_imp_t])
        lime_output_plus['num'].append([LAC_num_q, LAC_num_t])
        #add LIME scores based on LAC segmentation (mean)
        c_sum_q, c_sum_t = [0] + list(np.cumsum(LAC_num_q)), [0] + list(np.cumsum(LAC_num_t)) 
        laclime_q = sum([[np.mean(q['sorted_score'][c_sum_q[k]:c_sum_q[k+1]])]*LAC_num_q[k] for k in range(len(LAC_num_q))],[])
        laclime_t = sum([[np.mean(t['sorted_score'][c_sum_t[k]:c_sum_t[k+1]])]*LAC_num_t[k] for k in range(len(LAC_num_t))],[])
        lime_output_plus['laclime'].append([laclime_q, laclime_t])
        lime_output_plus['laclime_rank'].append([ list(ss.rankdata([abs(x) for x in laclime_q])), 
                                                  list(ss.rankdata([abs(x) for x in laclime_t])) ])
        lime_output_plus['ori_laclime_rank'].append([ list(ss.rankdata([x for x in laclime_q])), 
                                                  list(ss.rankdata([x for x in laclime_t])) ])
        
    else:
        j = 0
        while j < len(LCQMC_testB['text_q_seg'][i]):
            if LCQMC_testB['text_q_seg'][i][j] == LAC_result_q[0][j]:
                j+=1
            else:
                LAC_result_q[0][j] = LAC_result_q[0][j]+LAC_result_q[0][j+1]
                del LAC_result_q[0][j+1]
                LAC_result_q[1][j] = LAC_result_q[1][j]  #use the first ner
                del LAC_result_q[1][j+1]
                LAC_result_q[2][j] = int((LAC_result_q[2][j]+LAC_result_q[2][j+1])/2)
                del LAC_result_q[2][j+1]
                need_reduce_idx = [j - k < 0 for k in np.cumsum(LAC_result_q[3])].index(True)
                LAC_result_q[3][need_reduce_idx] = LAC_result_q[3][need_reduce_idx]-1
            
        j = 0
        while j < len(LCQMC_testB['text_t_seg'][i]):
            if LCQMC_testB['text_t_seg'][i][j] == LAC_result_t[0][j]:
                j+=1
            else:
                LAC_result_t[0][j] = LAC_result_t[0][j]+LAC_result_t[0][j+1]
                del LAC_result_t[0][j+1]
                LAC_result_t[1][j] = LAC_result_t[1][j] #use the first ner
                del LAC_result_t[1][j+1]
                LAC_result_t[2][j] = int((LAC_result_t[2][j]+LAC_result_t[2][j+1])/2)
                del LAC_result_t[2][j+1]
                need_reduce_idx = [j - k < 0 for k in np.cumsum(LAC_result_t[3])].index(True)
                LAC_result_t[3][need_reduce_idx] = LAC_result_t[3][need_reduce_idx]-1
                
        LAC_token_q, LAC_token_t = LAC_result_q[0], LAC_result_t[0]
        LAC_ner_q, LAC_ner_t = LAC_result_q[1], LAC_result_t[1]
        LAC_imp_q, LAC_imp_t = LAC_result_q[2], LAC_result_t[2]
        LAC_num_q, LAC_num_t = LAC_result_q[3], LAC_result_t[3]
        lime_output_plus['ner'].append([LAC_ner_q, LAC_ner_t])
        lime_output_plus['imp'].append([LAC_imp_q, LAC_imp_t])
        lime_output_plus['num'].append([LAC_num_q, LAC_num_t])

        #add LIME scores based on LAC segmentation (mean)
        c_sum_q, c_sum_t = [0] + list(np.cumsum(LAC_num_q)), [0] + list(np.cumsum(LAC_num_t)) 
        laclime_q = sum([[np.mean(q['sorted_score'][c_sum_q[k]:c_sum_q[k+1]])]*LAC_num_q[k] for k in range(len(LAC_num_q))],[])
        laclime_t = sum([[np.mean(t['sorted_score'][c_sum_t[k]:c_sum_t[k+1]])]*LAC_num_t[k] for k in range(len(LAC_num_t))],[])
        lime_output_plus['laclime'].append([laclime_q, laclime_t])
        lime_output_plus['laclime_rank'].append([ list(ss.rankdata([abs(x) for x in laclime_q])), 
                                                  list(ss.rankdata([abs(x) for x in laclime_t])) ])
        lime_output_plus['ori_laclime_rank'].append([ list(ss.rankdata([x for x in laclime_q])), 
                                                  list(ss.rankdata([x for x in laclime_t])) ])

        
    lime_output_plus['lime_rank'].append([ list(ss.rankdata([abs(x) for x in lime_output_plus['rationale_score'][i][0]])), 
                                           list(ss.rankdata([abs(x) for x in lime_output_plus['rationale_score'][i][1]])) ])
    lime_output_plus['imp_rank'].append([ list(ss.rankdata([x for x in lime_output_plus['imp'][i][0]])), 
                                           list(ss.rankdata([x for x in lime_output_plus['imp'][i][1]])) ])

print('lime_output_plus:')
print(lime_output_plus.keys()) 




# add important scores (imp) to lime_output_plus  即词性得分(lexical category score)
#add fined imp scores:
import collections
ner_a,ner_b,ner_c,ner_d = [],[],[],[]
for i in range(len(lime_output_plus['imp'] )):
    for j in range(len(lime_output_plus['imp'][i][0])):
        if lime_output_plus['imp'][i][0][j] == 3:
            ner_a.append(lime_output_plus['ner'][i][0][j])
        if lime_output_plus['imp'][i][0][j] == 2:
            ner_b.append(lime_output_plus['ner'][i][0][j])
        if lime_output_plus['imp'][i][0][j] == 1:
            ner_c.append(lime_output_plus['ner'][i][0][j])
        if lime_output_plus['imp'][i][0][j] == 0:
            ner_d.append(lime_output_plus['ner'][i][0][j])
    for j in range(len(lime_output_plus['imp'][i][1])):
        if lime_output_plus['imp'][i][1][j] == 3:
            ner_a.append(lime_output_plus['ner'][i][1][j])
        if lime_output_plus['imp'][i][1][j] == 2:
            ner_b.append(lime_output_plus['ner'][i][1][j])
        if lime_output_plus['imp'][i][1][j] == 1:
            ner_c.append(lime_output_plus['ner'][i][1][j])
        if lime_output_plus['imp'][i][1][j] == 0:
            ner_d.append(lime_output_plus['ner'][i][1][j])
            
aug_frequency = collections.Counter(ner_a + ner_a + ner_a + ner_a+ ner_b + ner_b +ner_b + ner_c + ner_c+ ner_d)
ori_frequency = collections.Counter(ner_a + ner_b + ner_c + ner_d)
fine_imp_dic = {k: round(2*aug_frequency[k]/ori_frequency[k])/2 for k in ori_frequency.keys()}   

print(dict(sorted(fine_imp_dic.items(), key=lambda item: -item[1])))
lime_output_plus['fined_imp'] = []
for i in range(len(lime_output_plus['id'])):
    fine_imp_q = [fine_imp_dic[k] for k in lime_output_plus['ner'][i][0]]
    fine_imp_t = [fine_imp_dic[k] for k in lime_output_plus['ner'][i][1]]
    lime_output_plus['fined_imp'].append([fine_imp_q, fine_imp_t])



# Final output
lime_output_plus['label'] =  list(pred_together)
ENG_punctuation =  string.punctuation
p_list = list(ENG_punctuation) + list(CHN_punctuation) 
stopwordsFile = open("./baidu_stopwords.txt", "r")
baidu_stopwords = stopwordsFile.read()  
stopwords = baidu_stopwords.split('\n')
rm_tokens = p_list

def refine_sen(sen):
    refine_sen = ''.join([i for i in list(sen)])
    return refine_sen
def refine_sen2(sen):
    refine_sen = ''.join([i for i in list(sen) if i not in rm_tokens+stopwords])
    return refine_sen

refined_sen_Q = {}
for i in range(len(lime_output_plus['id'])):
    refined_sen = refine_sen(lime_output_plus['query'][i])
    if refined_sen not in refined_sen_Q:
        refined_sen_Q[refined_sen] = 1
    else:
        refined_sen_Q[refined_sen] += 1
    refined_sen = refine_sen(lime_output_plus['title'][i])
    if refined_sen not in refined_sen_Q:
        refined_sen_Q[refined_sen] = 1
    else:
        refined_sen_Q[refined_sen] += 1

refined_sen_Q2 = {}
for i in range(len(lime_output_plus['id'])):
    refined_sen = refine_sen2(lime_output_plus['query'][i])
    if refined_sen not in refined_sen_Q2:
        refined_sen_Q2[refined_sen] = 1
    else:
        refined_sen_Q2[refined_sen] += 1
        
    refined_sen = refine_sen2(lime_output_plus['title'][i])
    if refined_sen not in refined_sen_Q2:
        refined_sen_Q2[refined_sen] = 1
    else:
        refined_sen_Q2[refined_sen] += 1

        
import math
import string
import zhon
from zhon.hanzi import punctuation as CHN_punctuation
ENG_punctuation =  string.punctuation
p_list = list(ENG_punctuation) + list(CHN_punctuation) 
stopwordsFile = open("./baidu_stopwords.txt", "r")
baidu_stopwords = stopwordsFile.read()  
stopwords = baidu_stopwords.split('\n')

final_lime_output = np.load('./lime_outputB/output.npy',allow_pickle=True).item()
final_lime_output['label'] =  list(pred_together)

for i in range(len(lime_output_plus['rationale'])): #
    text_q_seg = LCQMC_testB[LCQMC_testB['id'] == lime_output_plus['id'][i]]['text_q_seg'].item()
    text_t_seg = LCQMC_testB[LCQMC_testB['id'] == lime_output_plus['id'][i]]['text_t_seg'].item()
    common_seg = set(text_q_seg).intersection(set(text_t_seg))

    text_q  = LCQMC_testB[LCQMC_testB['id'] == lime_output_plus['id'][i]]['query'].item()
    text_t  = LCQMC_testB[LCQMC_testB['id'] == lime_output_plus['id'][i]]['title'].item()
    if refined_sen_Q[refine_sen(text_q)] > refined_sen_Q[refine_sen(text_t)]:
        GoldenRule = 0
    elif refined_sen_Q[refine_sen(text_q)] < refined_sen_Q[refine_sen(text_t)]:
        GoldenRule = 1
    else:
        if refined_sen_Q2[refine_sen2(text_q)] > refined_sen_Q2[refine_sen2(text_t)]:
            GoldenRule = 20
        elif refined_sen_Q2[refine_sen2(text_q)] < refined_sen_Q2[refine_sen2(text_t)]:
            GoldenRule = 21
        else:
            GoldenRule = 22
    
    #define some redundant part importance as 0
    for key in refined_sen_Q.keys():
        if key != lime_output_plus['query'][i] and key in lime_output_plus['query'][i]:
            overlap_lower = lime_output_plus['query'][i].find(key) 
            overlap_upper = lime_output_plus['query'][i].find(key) + len(key)
            new_imp = [lime_output_plus['imp'][i][0][k]*(k>=overlap_lower and k<=overlap_upper) for k in range(len(lime_output_plus['imp'][i][0])) ]
            assert len(lime_output_plus['imp'][i][0]) == len(new_imp)
            lime_output_plus['imp'][i][0] = new_imp
        if key != lime_output_plus['title'][i] and key in lime_output_plus['title'][i]:
            overlap_lower = lime_output_plus['title'][i].find(key) 
            overlap_upper = lime_output_plus['title'][i].find(key) + len(key)
            new_imp = [lime_output_plus['imp'][i][1][k]*(k>=overlap_lower and k<=overlap_upper) for k in range(len(lime_output_plus['imp'][i][1])) ]
            assert len(lime_output_plus['imp'][i][1]) == len(new_imp)
            lime_output_plus['imp'][i][1] = new_imp

    
    q_seg_from_piece = list(''.join([p for p in lime_output_plus['piece'][i][0] if p not in stopwords+p_list]))
    remove_q_seg_from_piece = list(''.join([p for p in lime_output_plus['piece'][i][0] if p in stopwords+p_list]))
    t_seg_from_piece = list(''.join([p for p in lime_output_plus['piece'][i][1] if p not in stopwords+p_list]))
    remove_t_seg_from_piece = list(''.join([p for p in lime_output_plus['piece'][i][1] if p in stopwords+p_list]))
    critical_uncommon_seg = set(q_seg_from_piece).union(set(t_seg_from_piece)) - set(q_seg_from_piece).intersection(set(t_seg_from_piece)) - set(remove_q_seg_from_piece).union(set(remove_t_seg_from_piece))-set(stopwords)
    
    
    if lime_output_plus['label'][i] == 1:
        # two-order ranking for query: 
        rationale = [k for k in range(len(text_q_seg)) if lime_output_plus['imp'][i][0][k]>0 and text_q_seg[k] not in p_list  and (text_q_seg[k] in common_seg or lime_output_plus['imp'][i][0][k]>=2)]
        max_LAC_imp = max([lime_output_plus['fined_imp'][i][0][k] for k in rationale])
        LAC_imp = [lime_output_plus['fined_imp'][i][0][k]+ min(max_LAC_imp-lime_output_plus['fined_imp'][i][0][k], lime_output_plus['imp'][i][0][k]/100) for k in rationale]
        final_lime_output['rationale'][i][0]  = reorder(LAC_imp, rationale, mode='sequence')['sorted_token_index']
        
        # two-order ranking for title:  
        rationale = [k for k in range(len(text_t_seg)) if lime_output_plus['imp'][i][1][k] >0 and text_t_seg[k] not in p_list and (text_t_seg[k] in common_seg or lime_output_plus['imp'][i][1][k]>=2)]
        max_LAC_imp = max([lime_output_plus['fined_imp'][i][1][k] for k in rationale])
        LAC_imp = [lime_output_plus['fined_imp'][i][1][k]+ min(max_LAC_imp-lime_output_plus['fined_imp'][i][1][k], lime_output_plus['imp'][i][1][k]/100) for k in rationale]
        final_lime_output['rationale'][i][1]  = reorder(LAC_imp, rationale, mode='sequence')['sorted_token_index']

        
        #denosing policy for data with label=1, criteria = 'query'
        if GoldenRule == 0 or GoldenRule == 20:
            new_rationale = []
            for qk in final_lime_output['rationale'][i][0]:
                for tk in final_lime_output['rationale'][i][1]:
                    if text_t_seg[tk] == text_q_seg[qk]:
                        new_rationale.append(tk)
                        final_lime_output['rationale'][i][1].remove(tk)
                        break
            final_lime_output['rationale'][i][1] = new_rationale
            
        #denosing policy for data with label=1, criteria = 'title'
        if GoldenRule == 1 or GoldenRule == 21:
            new_rationale = []
            for tk in final_lime_output['rationale'][i][1]:
                for qk in final_lime_output['rationale'][i][0]:
                    if text_q_seg[qk] == text_t_seg[tk]:
                        new_rationale.append(qk)
                        final_lime_output['rationale'][i][0].remove(qk)
                        break
            final_lime_output['rationale'][i][0] = new_rationale

                    
    if lime_output_plus['label'][i] == 0:
        # two-order ranking for query: 
        rationale = [k for k in range(len(text_q_seg)) if lime_output_plus['imp'][i][0][k]>0 and text_q_seg[k] not in p_list]
        max_LAC_imp = max([lime_output_plus['fined_imp'][i][0][k] for k in rationale])
        LAC_imp = [lime_output_plus['fined_imp'][i][0][k]+min(max_LAC_imp-lime_output_plus['fined_imp'][i][0][k], lime_output_plus['laclime_rank'][i][0][k]/100 ) if text_q_seg[k] in critical_uncommon_seg else lime_output_plus['fined_imp'][i][0][k] for k in rationale]
        final_lime_output['rationale'][i][0]  = reorder(LAC_imp, rationale, mode='sequence')['sorted_token_index']

        # two-order ranking for title: 
        rationale = [k for k in range(len(text_t_seg)) if lime_output_plus['imp'][i][1][k] >0 and text_t_seg[k] not in p_list]
        max_LAC_imp = max([lime_output_plus['fined_imp'][i][1][k] for k in rationale])
        LAC_imp = [lime_output_plus['fined_imp'][i][1][k]+min(max_LAC_imp-lime_output_plus['fined_imp'][i][1][k], lime_output_plus['laclime_rank'][i][1][k]/100) if text_t_seg[k] in critical_uncommon_seg else lime_output_plus['fined_imp'][i][1][k] for k in rationale]
        final_lime_output['rationale'][i][1]  = reorder(LAC_imp, rationale, mode='sequence')['sorted_token_index']
        
        #denosing policy for data with label=0, criteria = 'query'
        if GoldenRule == 0 or GoldenRule == 20:
            #3nd forward
            new_rationale = []
            uncommon_flag = 0 
            for qk in final_lime_output['rationale'][i][0]:
                if len(final_lime_output['rationale'][i][1])==0:
                    break                
                first_t = final_lime_output['rationale'][i][1][0]
                
                if text_t_seg[first_t]== text_q_seg[qk]:
                    new_rationale.append(first_t)
                    final_lime_output['rationale'][i][1].remove(first_t)
                    if len(final_lime_output['rationale'][i][1])>0:
                        continue
                    else:
                        break
                        
                else:                    
                    if text_q_seg[qk] not in text_t_seg:
                        if uncommon_flag < 3:
                            uncommon_flag +=1 
                            uncommon_t_1st = final_lime_output['rationale'][i][1][0] 
                            new_rationale.append(uncommon_t_1st)
                            final_lime_output['rationale'][i][1].remove(uncommon_t_1st)
                            continue
                        else:
                            break
                    else:
                        for tk in final_lime_output['rationale'][i][1]:
                            if text_t_seg[tk] == text_q_seg[qk]:
                                new_rationale.append(tk)
                                final_lime_output['rationale'][i][1].remove(tk)
                                break
            new_rationale += final_lime_output['rationale'][i][1]
            final_lime_output['rationale'][i][1] = new_rationale
            
        #denosing policy for data with label=1, criteria = 'title'
        if GoldenRule == 1 or GoldenRule == 21:
            new_rationale = []
            uncommon_flag = 0 
            for tk in final_lime_output['rationale'][i][1]:
                if len(final_lime_output['rationale'][i][0])==0:
                    break
                    
                first_q = final_lime_output['rationale'][i][0][0]
                
                if text_q_seg[first_q]== text_t_seg[tk]:
                    new_rationale.append(first_q)
                    final_lime_output['rationale'][i][0].remove(first_q)
                    if len(final_lime_output['rationale'][i][0])>0:
                        continue 
                    else:
                        break
                        
                else:                    
                    if text_t_seg[tk] not in text_q_seg:
                        if uncommon_flag < 3:
                            uncommon_flag +=1 
                            uncommon_q_1st = final_lime_output['rationale'][i][0][0] 
                            new_rationale.append(uncommon_q_1st)
                            final_lime_output['rationale'][i][0].remove(uncommon_q_1st)
                            continue
                        else:
                            break
                    else:
                        for qk in final_lime_output['rationale'][i][0]:
                            if text_q_seg[qk] == text_t_seg[tk]:
                                new_rationale.append(qk)
                                final_lime_output['rationale'][i][0].remove(qk)
                                break
            new_rationale += final_lime_output['rationale'][i][0]
            final_lime_output['rationale'][i][0] = new_rationale
            
out_file = open('./lime_outputB/sim_rationale.txt', 'w')
for i in range(len(final_lime_output['id'])):
    out_file.write(str(final_lime_output['id'][i]) + '\t'+ str(final_lime_output['label'][i]) + '\t' + 
                   ','.join([str(i) for i in final_lime_output['rationale'][i][0]]) +'\t'+
                   ','.join([str(i) for i in final_lime_output['rationale'][i][1]]) +'\n')
out_file.close()


