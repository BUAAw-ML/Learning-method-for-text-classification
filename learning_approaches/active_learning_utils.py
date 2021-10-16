import numpy as np
from utils.chartTool import *
from scipy import stats
from scipy.spatial import distance_matrix

from copy import deepcopy as deepcopy
from sklearn.metrics import pairwise_distances
import pdb
from scipy import stats

###The larger the "information" value, the more uncertain


def get_information_calculation_by_name(calculation_name):
        if calculation_name == "variationRatios":
            return cal_bayesian_variationRatios
        elif calculation_name == "maxMargin":
            return cal_bayesian_maxMargin
        elif calculation_name == "confidence":
            return cal_bayesian_confidence
        elif calculation_name == "expectedLoss":
            return cal_bayesian_expectedLoss
        elif calculation_name == "BEAL":
            return cal_bayesian_BEAL
        elif calculation_name == "entropy":
            return cal_entropy
        elif calculation_name == "BALD":
            return cal_bayesian_BALD
        elif calculation_name == "test":
            return cal_bayesian_test
        else:
            assert 0


def cal_entropy(prediction):
    prediction = np.array(prediction)

    information_values = []

    for index, item in enumerate(prediction):
        #item shape: nsample * nlabel
        posterior_prediction = np.mean(item, axis=0) 
        information = -1 * np.mean(posterior_prediction * np.log(posterior_prediction)
                        + (1 - posterior_prediction) * np.log(1 - posterior_prediction))

        information_values.append(information)

    return information_values

def cal_bayesian_variationRatios(prediction):
    prediction = np.array(prediction)

    information_values = []

    for index, item in enumerate(prediction):
        #item shape: nsample * nlabel
        posterior_prediction = np.mean(item, axis=0) 
        information = 1 - np.mean(posterior_prediction[posterior_prediction >= 0.5])

        information_values.append(information)

    return information_values

def cal_bayesian_maxMargin(prediction):
    prediction = np.array(prediction)

    information_values = []

    for index, item in enumerate(prediction):
        #item shape: nsample * nlabel
        posterior_prediction = np.mean(item, axis=0) 
        information = - (np.mean(posterior_prediction[posterior_prediction >= 0.5]) - np.mean(posterior_prediction[posterior_prediction < 0.5]))

        information_values.append(information)

    return information_values


def cal_bayesian_confidence(prediction):
    prediction = np.array(prediction)

    information_values = []

    for index, item in enumerate(prediction):
        #item shape: nsample * nlabel
        posterior_prediction = np.mean(item, axis=0) 
        information = 1 - np.mean(posterior_prediction)
        information_values.append(information)

    return information_values


def cal_bayesian_BALD(prediction):

    information_values = []

    for index, item in enumerate(prediction):
        #item shape: nsample * nlabel
        item = np.array(item)

        # Only consider the first item in the rank
        top1 = np.argmax(item, axis=1)
        information = stats.mode(top1)[1][0]

        #consider all items in the rank
        # rank = np.argsort(item, axis=1)
        # weight = np.arange(item.shape[1])[:,np.newaxis]
        # modes = np.dot(rank, weight)
        # information = stats.mode(modes)[1][0]

        information = item.shape[0] - information

        information_values.append(information.tolist())
    
    return information_values


def cal_bayesian_expectedLoss(prediction):

    def rankedList(rList):
        rList = np.array(rList)
        gain = 2 ** rList - 1
        discounts = np.log2(np.arange(len(rList)) + 2)
        return np.sum(gain / discounts)

    information_values = []

    for index, item in enumerate(prediction):
        #item shape: nsample * nlabel

        dList = []
        for i in range(len(item)):
            rL = sorted(item[i], reverse=True)
            dList.append(rankedList(rL))

        # t = np.mean(2 ** np.array(item) - 1, axis=1)
        # rankedt = sorted(t.tolist(), reverse=True)
        # d = rankedList(rankedt)

        item_arr = np.array(item)

        posterior_prediction = np.mean(item_arr, axis=0)
        
        rankedt = item_arr[:,np.argsort(-posterior_prediction)].tolist()  # nsample * nlabel

        dList2 = []
        for i in range(len(rankedt)):
            dList2.append(rankedList(rankedt[i]))

        information = np.mean(np.array(dList)) - np.mean(np.array(dList2))

        information_values.append(information)

    return information_values

def cal_bayesian_test(prediction):

    def rankedList(rList):
        rList = np.array(rList)
        gain = (2 ** rList - 1)- np.std(rList, axis=0)
        discounts = np.arange(len(rList)) + 1
        return np.sum(gain / discounts)

    information_values = []

    for index, item in enumerate(prediction):

        item_arr = np.array(item)

        posterior_prediction = np.mean(2 ** item_arr - 1, axis=0)
        
        rankedt = item_arr[:,np.argsort(-posterior_prediction)].tolist()  # nsample * nlabel

        dList2 = []
        for i in range(len(rankedt)):
            dList2.append(rankedList(rankedt[i]))

        information =  - np.mean(np.array(dList2))

        information_values.append(information)

    return information_values

def cal_bayesian_BEAL(prediction):

    def rankedList(rList):
        rList = np.array(rList)
        gain = 2 ** rList - 1
        discounts = np.arange(len(rList)) + 1
        return np.sum(gain / discounts)

    information_values = []

    for index, item in enumerate(prediction):

        item_arr = np.array(item)

        posterior_prediction = np.mean(2 ** item_arr - 1, axis=0)
        
        rankedt = item_arr[:,np.argsort(-posterior_prediction)].tolist()  # nsample * nlabel

        dList2 = []
        for i in range(len(rankedt)):
            dList2.append(rankedList(rankedt[i]))

        information =  - np.mean(np.array(dList2))

        information_values.append(information)

    return information_values


def get_select_method_by_name(method_name):
        if method_name == "coreset":
            return greedy_k_center
        elif method_name == "badge":
            return cal_badge
        else:
            assert 0

def cal_badge(labeled, unlabeled, amount):

    def init_centers(X, K):
        ind = np.argmax([np.linalg.norm(s, 2) for s in X])
        mu = [X[ind]]
        indsAll = [ind]
        centInds = [0.] * len(X)
        cent = 0
        print('#Samps\tTotal Distance')
        while len(mu) < K:
            if len(mu) == 1:
                D2 = pairwise_distances(X, mu).ravel().astype(float)
            else:
                newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
                for i in range(len(X)):
                    if D2[i] >  newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            # print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
            if sum(D2) == 0.0: pdb.set_trace()
            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2)/ sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            while ind in indsAll: ind = customDist.rvs(size=1)[0]
            mu.append(X[ind])
            indsAll.append(ind)
            cent += 1
        return indsAll  
    
    embDim = 512
    label_num = unlabeled.shape[1] - 512
    gradEmbedding = np.zeros([unlabeled.shape[0], embDim * label_num])

    for j in range(unlabeled.shape[0]):
        for c in range(label_num):
            if unlabeled[j][c] >= 0.5:
                gradEmbedding[j][embDim * c : embDim * (c+1)] = deepcopy(unlabeled[j][label_num:]) * (1 - unlabeled[j][c])
            else:
                gradEmbedding[j][embDim * c : embDim * (c+1)] = deepcopy(unlabeled[j][label_num:]) * (-1 * unlabeled[j][c])

    chosen = init_centers(gradEmbedding, amount)
    
    return np.array(chosen)

def greedy_k_center(labeled, unlabeled, amount):

    cal_dist_batch=1000

    greedy_indices = []

    # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
    min_dist = np.min(distance_matrix(labeled[0, :].reshape((1, labeled.shape[1])), unlabeled), axis=0)
    min_dist = min_dist.reshape((1, min_dist.shape[0]))
    for j in range(1, labeled.shape[0], cal_dist_batch):
        if j + cal_dist_batch < labeled.shape[0]:
            dist = distance_matrix(labeled[j:j + cal_dist_batch, :], unlabeled)
        else:
            dist = distance_matrix(labeled[j:, :], unlabeled)
        min_dist = np.vstack((min_dist, np.min(dist, axis=0).reshape((1, min_dist.shape[1]))))
        min_dist = np.min(min_dist, axis=0)
        min_dist = min_dist.reshape((1, min_dist.shape[0]))

    # iteratively insert the farthest index and recalculate the minimum distances:
    farthest = np.argmax(min_dist)
    greedy_indices.append(farthest)
    for i in range(amount - 1):
        dist = distance_matrix(unlabeled[greedy_indices[-1], :].reshape((1, unlabeled.shape[1])), unlabeled)
        min_dist = np.vstack((min_dist, dist.reshape((1, min_dist.shape[1]))))
        min_dist = np.min(min_dist, axis=0)
        min_dist = min_dist.reshape((1, min_dist.shape[0]))
        farthest = np.argmax(min_dist)
        greedy_indices.append(farthest)

    return np.array(greedy_indices)


def update_chart(performance, chart_group_name, desc, acquire_method, acquire_data_num_per_round, dropout_samp_num,
                    num_paraphrases_per_text, seed, sampling_steps=0):

    if acquire_method == 'Random':
        method_name = "Random+"
    elif 'modelBased' in acquire_method or 'gradBased' in acquire_method:
        if dropout_samp_num == 1:
            method_name = "Deterministic+"
        elif dropout_samp_num > 1:
            method_name = "No-Deterministic+"+"dropout"+str(dropout_samp_num)+"_"
        method_name += acquire_method.split('_')[-1]
    elif 'dataBased' in acquire_method:
        method_name = "Deterministic+" + acquire_method.split('_')[-2]+"_" + acquire_method.split('_')[-1]
    else:
        print("acquire method name error")
        assert 0
    
    method_name += "_"+desc

    if num_paraphrases_per_text:
        method_name += "_aug"+str(num_paraphrases_per_text)+"_steps"+str(sampling_steps)
    
    if acquire_method == 'Random':
        method_name += '_seed'+str(seed)
    else:
        method_name += '+'+str(seed)

    updateLineChart(performance[0], method_name, gp_name=chart_group_name+"+OF1")
    updateLineChart(performance[1], method_name, gp_name=chart_group_name+"+CF1")
    updateLineChart(performance[2], method_name, gp_name=chart_group_name+"+mAP")