import torch
from torch.autograd import Variable
import numpy as np
import time
from scipy import stats
from scipy.spatial import distance_matrix
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F

from learning_approaches.active_learning_utils import *
from numpy import *

class Acquisition(object):

    def __init__(self, dataset, trainer, seed=0, usecuda=True, cuda_device=0, batch_size=1000, 
                    dropout_samp_num=50):
        self.dataset = dataset
        self.npr = np.random.RandomState(seed)
        self.usecuda = usecuda
        self.cuda_device = cuda_device
        self.batch_size = batch_size

        self.data_id_set = set(range(len(dataset)))
        self.train_id_set = set()  #the index of the labeled samples for training
        self.unlabel_id_set = None
        self.acquire_data_num = None
        # self._model_path = model_path
        self._model = None
        self._trainer = trainer

        self.dropout_samp_num = dropout_samp_num

        self.labels_cardinality = []

    #-------------------------Obtain data for initing model-----------------------------
    def initial_data(self):

        acquire_data_ids = range(0,self.acquire_data_num)
        self.train_id_set.update(acquire_data_ids)
        return acquire_data_ids

    #-------------------------Random sampling-----------------------------
    def random_samplingStrategy(self):

        acquire_data_ids = np.random.choice(np.array(list(self.unlabel_id_set)), self.acquire_data_num, replace=False)
        self.train_id_set.update(acquire_data_ids)
        return acquire_data_ids

    def modelBased_samplingStrategy(self, information_cal_name):
        
        unlabel_data = np.array(self.dataset)[sorted(list(self.unlabel_id_set))]

        output = self._trainer.predict(self._model, unlabel_data, self.dropout_samp_num)

        information_calculation_func = get_information_calculation_by_name(information_cal_name)
        
        information_values = information_calculation_func(output)

        acquire_data_ids = np.argsort(-np.array(information_values))[:self.acquire_data_num]
        acquire_data_ids = np.array(sorted(list(self.unlabel_id_set)))[acquire_data_ids]

        self.train_id_set.update(set(acquire_data_ids))
        return acquire_data_ids

    def modelLabelBased_samplingStrategy(self, information_cal_name):
        
        unlabel_data = np.array(self.dataset)[sorted(list(self.unlabel_id_set))]

        output = self._trainer.predict(self._model, unlabel_data, self.dropout_samp_num)

        information_calculation_func = get_information_calculation_by_name(information_cal_name)
        
        information_values = information_calculation_func(output, round(mean(self.labels_cardinality)))

        acquire_data_ids = np.argsort(-np.array(information_values))[:self.acquire_data_num]
        acquire_data_ids = np.array(sorted(list(self.unlabel_id_set)))[acquire_data_ids]

        self.train_id_set.update(set(acquire_data_ids))

        return acquire_data_ids

    def dataBased_samplingStrategy(self, select_method_name, feature_type):

        if feature_type == "hidFeat":
            sample_feature = np.array(self._trainer.predict(self._model, self.dataset, samp_num=1, return_feature=True)).squeeze(1)
            
            label_feature = sample_feature[sorted(list(self.train_id_set))]
            unlabel_feature = sample_feature[sorted(list(self.unlabel_id_set))]
        else:
            raise NotImplementedError() 

        select_method_func = get_select_method_by_name(select_method_name)
        acquire_data_ids = select_method_func(label_feature, unlabel_feature, self.acquire_data_num)

        acquire_data_ids = np.array(sorted(list(self.unlabel_id_set)))[acquire_data_ids]

        self.train_id_set.update(set(acquire_data_ids))
        return acquire_data_ids

    #——————————————————————————————Invoking a sampling strategy to obtain data————————————————————————————————————————————
    def obtain_data(self, acquire_method, acquire_data_num, model):
        
        self.unlabel_id_set = self.data_id_set - self.train_id_set
        self.acquire_data_num = acquire_data_num
        self._model = model

        if not self.train_id_set:
            print("First round: init model")
            acquire_data_ids = self.initial_data()
        else:
            if acquire_method == 'Random':
                print("Random Sampling")
                acquire_data_ids = self.random_samplingStrategy()
            elif 'modelBased' in acquire_method:
                print(acquire_method + " Sampling")
                acquire_data_ids = self.modelBased_samplingStrategy(information_cal_name=acquire_method.split('_')[1])
            elif 'modelLabelBased' in acquire_method:
                print(acquire_method + " Sampling")
                acquire_data_ids = self.modelLabelBased_samplingStrategy(information_cal_name=acquire_method.split('_')[1])
            elif 'dataBased' in acquire_method:
                print(acquire_method + " Sampling")
                acquire_data_ids = self.dataBased_samplingStrategy(select_method_name=acquire_method.split('_')[1],
                                                        feature_type=acquire_method.split('_')[2])
            else:
                raise NotImplementedError()

        if 'modelLabelBased' in acquire_method:
            self.labels_cardinality.extend([len(acquire_data['label']) for acquire_data in np.array(self.dataset)[list(acquire_data_ids)]])

        return acquire_data_ids