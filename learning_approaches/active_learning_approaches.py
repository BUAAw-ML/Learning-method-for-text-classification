import copy
import numpy as np
import tqdm
import random
import torch
import time

from learning_approaches.active_learning_sampling_strategies import Acquisition
from datasets import DatasetForBert
from classifiers import *
from learning_approaches.active_learning_utils import *

class ActiveLearningApproachBase(object):
    pass

class ActiveLearningApproach(ActiveLearningApproachBase):
    """Active Learning.
    """

    def __init__(self, AL_config, trainer, model_config):
        """Initialize the strategy.

        Args:
            use_criteria (float): the USE similarity criteria for a legitimate rewriting.
            gpt2_criteria (float): the GPT2 perplexity criteria for a legitimate rewriting.
        """
        super(ActiveLearningApproach, self).__init__()
        
        self._trainer = trainer
        self._model_config = model_config
        self._seed = AL_config["seed"]
        self._chart_group_name = AL_config["chart_group_name"]
        self._ALmethod_desc = AL_config["ALmethod_desc"]

        self._total_acquire_rounds = AL_config["total_acquire_rounds"]
        self._acquire_method = AL_config["acquire_method"]
        self._acquire_data_num_per_round = AL_config["acquire_data_num_per_round"]
        self._dropout_samp_num = AL_config["dropout_samp_num"]

        self._rng = np.random.RandomState(self._seed)

    def __repr__(self):
        return self.__class__.__name__

    def active_learning_procedure(self,
                             metric_bundle,
                             train_set,
                             test_set,
                             paraphrase_strategy,
                             num_paraphrases_per_text,
                             period_save=5000):
        """Fine tune the classifier using given data.

        Args:
            metric_bundle (MetricBundle): a metric bundle including the target classifier.
            paraphrase_strategy (StrategyBase): a paraphrase strategy to fine tune the classifier.
            train_set (dict): the training set of the classifier.
            num_paraphrases_per_text (int): the number of paraphrases to generate for each data.
                This parameter is for paraphrase strategy.
            total_acquire_rounds (int): the number of steps to fine tune the classifier.
            tuning_batch_size (int): the batch size for fine tuning.
            num_sentences_to_rewrite_per_step (int): the number of data to rewrite using the
                paraphrase strategy in each tuning step. You can set is as large as the tuning
                batch size. You can also use a smaller value to speed up the tuning.
            period_save (int): the period in steps to save the fine tuned classifier.
        """
        end = time.time()

        paraphrase_field = train_set["paraphrase_field"]
        data_record_list = copy.deepcopy(train_set["data"])

        print("Fine-tune the classifier.")

        acquisition_function = Acquisition(data_record_list, trainer=self._trainer, dropout_samp_num=self._dropout_samp_num)
        acquire_round = 0

        # classifier = metric_bundle.get_target_classifier()

        print('The total amount of training data：%d\n' %len(data_record_list))

        acquire_data_record = []

        classifier = None

        while True:

            acquire_round += 1
            print("\nCurrent round：{} [num_paraphrases_per_text: {}, seed：{}, dropout_samp_num: {}]".format(
                        acquire_round, num_paraphrases_per_text, self._seed, self._dropout_samp_num))

            acquire_data_ids = acquisition_function.obtain_data(acquire_method=self._acquire_method, 
                                                                acquire_data_num=self._acquire_data_num_per_round,
                                                                model = classifier)
            
            classifier = get_customized_clf(self._model_config)
            
            print("Prepare the training data:")
            for acquire_data_id in tqdm.tqdm(acquire_data_ids, total=len(acquire_data_ids)):

                data_record_t = data_record_list[acquire_data_id]
                acquire_data_record.append(data_record_t)

                if  num_paraphrases_per_text and acquire_round > 1:
                    paraphrase_list = paraphrase_strategy.paraphrase_example(
                        data_record_t,
                        paraphrase_field,
                        num_paraphrases_per_text)

                    # metric_list = metric_bundle.measure_batch(
                    #     data_record_t[paraphrase_field], paraphrase_list, data_record_t,
                    #     paraphrase_field)

                    for paraphrase in paraphrase_list:
                        data_record_new = copy.deepcopy(data_record_t)
                        data_record_new[paraphrase_field] = paraphrase
                        acquire_data_record.append(data_record_new)
                        
            print("The number of training data：{}".format(len(acquire_data_record)))

            print("Fine-tune the classifier.")

            performance = self._trainer.learning(model=classifier, trainset=acquire_data_record, testset=test_set["data"])
            
            update_chart(performance, self._chart_group_name, self._ALmethod_desc, self._acquire_method, self._acquire_data_num_per_round, self._dropout_samp_num,
                        num_paraphrases_per_text, self._seed, paraphrase_strategy._strategy_config["sampling_steps"]) #,

            self._trainer.reset_best_score()
            
            print("The duration of active learning: {h}hours,{m}minutes".format(h=(time.time() - end) // (60 * 60), 
                                                                            m=(time.time() - end) % (60 * 60) // 60 ))

            if acquire_round >= self._total_acquire_rounds:
                break
        
        fo = open(os.path.join('obtaining_data_time.txt'), "a+")
        fo.write("The average obtaining data time (s) of {} {}_dropout{}: {} \n".format(
            self._chart_group_name, self._acquire_method, self._dropout_samp_num, np.mean(acquisition_function._obtain_data_time[1:])))


