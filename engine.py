import datetime
import os

from utils.benchmark_utils import update_detailed_result
from datasets import builtin_datasets, get_dataset, subsample_dataset, verify_dataset
from metrics.attack_aggregation_utils import add_sentence_level_adversarial_attack_metrics
from metrics.metric_utils import MetricBundle
from paraphrase_strategies import (
    BertSamplingStrategy, IdentityStrategy, NonAutoregressiveBertSamplingStrategy, RandomStrategy,
    TextAttackStrategy)
from paraphrase_strategies.strategy_base import StrategyBase

from learning_approaches.robust_tuning_approaches import (
    RobustTuningStrategyBase, RobustTuningApproach)
from learning_approaches.active_learning_approaches import (
    ActiveLearningApproachBase, ActiveLearningApproach)

import engine
from classifiers import *
from utils import log
from datasets.dataloader_mltc import *
from classifiers.trainer import *

logger = log.setup_custom_logger(__name__)
log.remove_logger_tf_handler(logger)

built_in_paraphrase_strategies = {
    "RandomStrategy": RandomStrategy,
    "IdentityStrategy": IdentityStrategy,
    "TextAttackStrategy": TextAttackStrategy,
    "BertSamplingStrategy": BertSamplingStrategy,
    "NonAutoregressiveBertSamplingStrategy": NonAutoregressiveBertSamplingStrategy
}

built_in_learning_approaches = {
    "DefaultTuningStrategy": RobustTuningApproach,
    "DefaultActiveLearningStrategy": ActiveLearningApproach
}

class Engine(object):
    """Engine framework for adversarial attack methods on text classification."""

    def __init__(self, arg_dict, data_config, model_config, train_config, attack_set=None, 
                 enable_bert_clf=True, bert_clf_bs=32):
        """Initialize Engine framework.
        """

        self._arg_dict = arg_dict

        # make output dir
        self._output_dir = arg_dict["output_dir"]
        os.makedirs(arg_dict["output_dir"], exist_ok=True)
        self._dataset_name = arg_dict["dataset"]

        # setup dataset
        if arg_dict["clf_type"] == "multi_classify":
            if arg_dict["dataset"] in builtin_datasets:
                # if trainset is not None or testset is not None:
                #     logger.error(("dataset name %d conflict with builtin dataset. "
                #                 "set trainset and testset to None.") % arg_dict["dataset"])
                #     raise RuntimeError
                trainset, testset = get_dataset(arg_dict["dataset"])
            else:
                verify_dataset(trainset)
                verify_dataset(testset)
        elif arg_dict["clf_type"] == "multi_label_classify":
            dataset = load_data(data_config=data_config)
            trainset = dataset.train_data
            testset = dataset.test_data
            train_config["dataset_object"] = dataset

        if attack_set is None:
            attack_set = testset

        # if arg_dict["subsample_testset"] != 0:
        #     attack_set = subsample_dataset(attack_set, arg_dict["subsample_testset"])

        self._trainset = trainset
        self._testset = testset
        self._attack_set = attack_set

        # setup metric bundle
        self._metric_bundle = MetricBundle(
            enable_bert_clf_prediction=enable_bert_clf,
            use_gpu_id=arg_dict["use_gpu_id"], gpt2_gpu_id=arg_dict["gpt2_gpu_id"],
            bert_gpu_id=arg_dict["bert_gpu_id"], dataset_name=arg_dict["dataset"],
            trainset=self._trainset, testset=self._testset,
            bert_clf_steps=arg_dict["bert_clf_steps"],
            bert_clf_bs=bert_clf_bs
        )

        train_config["label_num"] = len(self._trainset["label_mapping"])
        model_config["output_dim"] = train_config["label_num"]
        # if model_config['clf_name'] != 'use_default_clf':
        #     model = get_customized_clf(model_config)
        #     self._metric_bundle.add_classifier(model, set_target_clf=True)
        self._model_config = model_config

        if (arg_dict["load_robust_tuned_clf_desc"] is not None
                and arg_dict["load_robust_tuned_clf_desc"] not in ["null", "None", "none", ""]):
            self._metric_bundle.get_target_classifier().load_robust_tuned_model(
                arg_dict["load_robust_tuned_clf_desc"], arg_dict["robust_tuning_steps"])

        add_sentence_level_adversarial_attack_metrics(
            self._metric_bundle, gpt2_ppl_threshold=5, use_sim_threshold=0.85)
            
        self._trainer = MLC_Trainer(train_config)

    def run_generate_robust_tuning(self,
                          paraphrase_strategy="IdentityStrategy",
                          tuning_strategy="DefaultTuningStrategy",
                          strategy_gpu_id=-1,
                          num_paraphrases_per_text=50,
                          tuning_steps=5000):
        """Using a paraphrase strategy to do adversarial fine tuning for the target classifier.

        Args:
            paraphrase_strategy (str or StrategyBase): the paraphrase strategy to engine.
                Either the name of a builtin strategy or a customized strategy derived from
                StrategyBase.
            tuning_strategy (str or TuningStrategyBase): the adversarial tuning strategy.
                Either the name of a builtin strategy or a customized strategy derived from
                TuningStrategyBase
            strategy_gpu_id (int): the GPU id to run the paraphrase strategy.
            num_paraphrases_per_text (int): number of paraphrases for each sentence.
            tuning_steps (int): number of steps to tune the classifier.
        """

        if isinstance(paraphrase_strategy, str):
            if paraphrase_strategy in built_in_paraphrase_strategies:
                paraphrase_strategy = built_in_paraphrase_strategies[paraphrase_strategy](
                    self._arg_dict, self._dataset_name, strategy_gpu_id, self._output_dir, self._metric_bundle)
        assert isinstance(paraphrase_strategy, StrategyBase)

        paraphrase_strategy.fit(self._trainset)

        if isinstance(tuning_strategy, str):
            if tuning_strategy in built_in_learning_approaches:
                robust_tuning_strategy = built_in_learning_approaches[tuning_strategy]()
        assert isinstance(robust_tuning_strategy, RobustTuningStrategyBase)

        robust_tuning_strategy.fine_tune_classifier(
            metric_bundle=self._metric_bundle,
            paraphrase_strategy=paraphrase_strategy,
            train_set=self._trainset,
            num_paraphrases_per_text=num_paraphrases_per_text,
            tuning_steps=tuning_steps
        )

    def run_generate_active_learning(self, AL_config, 
                          paraphrase_strategy="IdentityStrategy",
                          learning_strategy="DefaultActiveLearningStrategy",
                          strategy_gpu_id=-1,
                          num_paraphrases_per_text=50):
        """generate_active_learning
        """

        if isinstance(paraphrase_strategy, str):
            if paraphrase_strategy in built_in_paraphrase_strategies:
                paraphrase_strategy = built_in_paraphrase_strategies[paraphrase_strategy](
                    self._arg_dict, self._dataset_name, strategy_gpu_id, self._output_dir, self._metric_bundle)
        assert isinstance(paraphrase_strategy, StrategyBase)
        paraphrase_strategy.fit(self._trainset)

        if isinstance(learning_strategy, str):
            if learning_strategy in built_in_learning_approaches:
                active_learning_strategy = built_in_learning_approaches[learning_strategy](AL_config, trainer=self._trainer, 
                                                                                           model_config=self._model_config)
        assert isinstance(active_learning_strategy, ActiveLearningApproachBase)

        active_learning_strategy.active_learning_procedure(
            metric_bundle=self._metric_bundle,
            train_set=self._trainset,
            test_set = self._testset,
            paraphrase_strategy=paraphrase_strategy,
            num_paraphrases_per_text=num_paraphrases_per_text
        )

    def run_generate_attack(self,
                      paraphrase_strategy="IdentityStrategy",
                      strategy_gpu_id=-1,
                      num_paraphrases_per_text=50,
                      exp_name=None,
                      update_global_results=True):
        """Run the engine.

        Args:
            paraphrase_strategy (str or StrategyBase): the paraphrase strategy to engine.
                Either the name of a builtin strategy or a customized strategy derived from
                StrategyBase.
            strategy_gpu_id (int): the gpu id to run the strategy. -1 for CPU. Ignored when
                ``paraphrase_strategy`` is an object.
            num_paraphrases_per_text (int): number of paraphrases for each sentence.
            exp_name (str or None): the name of current experiment. None for default name. the
                default name is ``<dataset_name>-<strategy_name>-<date>-<time>``.
            update_global_results (bool): whether to write results in <fibber_root_dir> or the
                engine output dir.

        Returns:
            A dict of evaluation results.
        """

        if isinstance(paraphrase_strategy, str):
            if paraphrase_strategy in built_in_paraphrase_strategies:
                paraphrase_strategy = built_in_paraphrase_strategies[paraphrase_strategy](
                    self._arg_dict, self._dataset_name, strategy_gpu_id, self._output_dir, self._metric_bundle)
        assert isinstance(paraphrase_strategy, StrategyBase)

        paraphrase_strategy.fit(self._trainset)

        # get experiment name
        if exp_name is None:
            exp_name = (self._dataset_name + "-" + str(paraphrase_strategy) + "-"
                        + datetime.datetime.now().strftime("%m%d-%H%M%S"))

        tmp_output_filename = os.path.join(
            self._output_dir, exp_name + "-tmp.json")
        logger.info("Write paraphrase temporary results in %s.", tmp_output_filename)
        results = paraphrase_strategy.paraphrase_dataset(
            self._attack_set, num_paraphrases_per_text, tmp_output_filename)

        output_filename = os.path.join(
            self._output_dir, exp_name + "-with-metric.json")
        logger.info("Write paraphrase with metrics in %s.", tmp_output_filename)

        results = self._metric_bundle.measure_dataset(
            results=results, output_filename=output_filename)

        aggregated_result = self._metric_bundle.aggregate_metrics(
            self._dataset_name, str(paraphrase_strategy), exp_name, results)
        update_detailed_result(aggregated_result,
                               self._output_dir if not update_global_results else None)
        return aggregated_result


    def get_metric_bundle(self):
        return self._metric_bundle
