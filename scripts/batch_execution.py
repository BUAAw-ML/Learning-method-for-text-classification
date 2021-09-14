import argparse
import subprocess


GPU_CONFIG = {
    "single": {
        "--classifier_id": 1,
        "--bert_gpu_id": 1,
        "--use_gpu_id": 1,
        "--gpt2_gpu_id": 1,
        "--strategy_gpu_id": 1,
    },
    "multi": {
        "--classifier_id": 0,
        "--bert_gpu_id": 0,
        "--use_gpu_id": 0,
        "--gpt2_gpu_id": 1,
        "--strategy_gpu_id": 2,
    }
}

COMMON_CONFIG = {
     "al_exp1": {
        # acquire_method: 
        # Random, 
        # modelBased_variationRatios, modelBased_maxMargin,
        # modelBased_expectedLoss, modelBased_BALD modelBased_confidence,modelBased_positiveExpectedLoss, modelBased_BEAL
        # dataBased_coreset_hidFeat
        # gradBased_badge
        # modelLabelBased_positiveExpectedLoss

        #  "[KBS]stack-overflow_1100-600-10k+acq500+trEpo100"
        # "[InfoSci]aapd_small+acq500+trEpo50"
        "--chart_group_name": "[InfoSci]aapd_small+acq500+trEpo50",
        "--acquire_method": "modelBased_BEAL",
        "--ALmethod_desc": "", #
        "--total_acquire_rounds": 17,
        "--num_paraphrases_per_text": 0,
        "--acquire_data_num_per_round": 500,
        "--seed": 128,
        "--dropout_samp_num": 1
     },
     
}

CLF_CONFIG = {
    "--clf_type": "multi_label_classify", ##multi_label_classify",
    "--clf_name": "MLPBert",
    "--train_epochs": 50,
    "--performance_indicator": "OF1",
}

DATASET_CONFIG = {
    # "ag_no_title": {
    #     "--dataset": "ag_no_title",
    #     "--output_dir": "exp_output_dir/ag",
    #     "--bert_clf_steps": 20000
    # },
    "aapd": {
        "--dataset": "aapd",
        "--output_dir": "exp_output_dir/aapd",
        "--dataset_path": "../datasets/AAPD/small_dataset", 
        "--bert_clf_steps": 20000
    },
    # "stack-overflow": {
    #     "--dataset": "stack-overflow",
    #     "--output_dir": "exp_output_dir/stack-overflow",
    #     "--dataset_path": "../datasets/stack-overflow/stack-overflow1100-600-10kTSamples", 
    #     "--bert_clf_steps": 20000
    # },
}

#AAPD/small_dataset 
#stack-overflow/stack-overflow2000-600-10kTSamples
#stack-overflow/stack-overflow2000-700-20kTSamples
#stack-overflow/stack-overflow1100-600-20kTSamples

PARAPHRASE_STRATEGY_CONFIG = {
    # "identity": {
    #     "--strategy": "IdentityStrategy"
    # },
    # "random": {
    #     "--strategy": "RandomStrategy"
    # },
    # "textfooler": {
    #     "--strategy": "TextAttackStrategy",
    #     "--ta_recipe": "TextFoolerJin2019"
    # },
    # "pso": {
    #     "--strategy": "TextAttackStrategy",
    #     "--ta_recipe": "PSOZang2020"
    # },
    # "bertattack": {
    #     "--strategy": "TextAttackStrategy",
    #     "--ta_recipe": "BERTAttackLi2020"
    # },
    # "bae": {
    #     "--strategy": "TextAttackStrategy",
    #     "--ta_recipe": "BAEGarg2019"
    # # },
    "asrs": {
        "--paraphrase_strategy": "BertSamplingStrategy",
        "--bs_enforcing_dist": "wpe",
        "--bs_wpe_threshold": 1.0,
        "--bs_wpe_weight": 1000,
        "--bs_use_threshold": 0.95,
        "--bs_use_weight": 1000,
        "--bs_gpt2_weight": 10,
        "--bs_sampling_steps": 10, #200
        "--bs_burnin_steps": 5, #100
        "--bs_clf_weight": 3,
        "--bs_window_size": 3,
        "--bs_accept_criteria": "joint_weighted_criteria",
        "--bs_burnin_enforcing_schedule": "1",
        "--bs_burnin_criteria_schedule": "1",
        "--bs_seed_option": "origin",
        "--bs_split_sentence": "0", #auto
        "--bs_lm_option": "finetune",
        "--bs_stanza_port": 9001,
    }
}

def to_command(args):
    ret = []
    for k, v in args.items():
        ret.append(k)
        ret.append(str(v))

    return ret


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", choices=["single", "multi"], default="single")
    parser.add_argument("--dataset", choices=list(DATASET_CONFIG.keys()) + ["all"], default="all")
    parser.add_argument("--strategy", choices=list(PARAPHRASE_STRATEGY_CONFIG.keys()) + ["all"],
                        default="all")

    args = parser.parse_args()
    if args.dataset == "all":
        dataset_list = list(DATASET_CONFIG.keys())
    else:
        dataset_list = [args.dataset]

    if args.strategy == "all":
        strategy_list = list(PARAPHRASE_STRATEGY_CONFIG.keys())
    else:
        strategy_list = [args.strategy]

    exp_list = list(COMMON_CONFIG.keys())

    for exp in exp_list:
        for dataset in dataset_list:
            for strategy in strategy_list:
                command = ["python3", "-m", "main"]
                command += to_command(COMMON_CONFIG[exp])
                command += to_command(CLF_CONFIG)
                command += to_command(GPU_CONFIG[args.gpu])
                command += to_command(DATASET_CONFIG[dataset])
                command += to_command(PARAPHRASE_STRATEGY_CONFIG[strategy])

                subprocess.call(command)
    
    print("Finish experiments!")

if __name__ == "__main__":
    main()
