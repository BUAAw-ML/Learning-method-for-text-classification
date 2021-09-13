# __email__ = 'wangqb6@outlook.com'
# __version__ = '0.0.1'

import warnings
warnings.filterwarnings('ignore')

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import nltk

import argparse

from engine import *
import utils
from utils.download_utils import get_root_dir
from utils import log

logger = log.setup_custom_logger(__name__)
log.remove_logger_tf_handler(logger)

# change cache directory
nltk.data.path += [os.path.join(get_root_dir(), "common", "nltk_data")]
os.environ['TRANSFORMERS_CACHE'] = os.path.join(
    get_root_dir(), "common", "transformers_pretrained")
os.environ['TFHUB_CACHE_DIR'] = os.path.join(get_root_dir(), "common", "tfhub_pretrained")
os.environ['CORENLP_HOME'] = os.path.join(get_root_dir(), "common", "stanford-corenlp-4.1.0")


def main():
    parser = argparse.ArgumentParser()

    # add experiment args
    parser.add_argument("--dataset", type=str, default="AAPD")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument('--dataset_path', default='../datasets/AAPD', type=str,
                    help='path of data')

    # metric args
    parser.add_argument("--gpt2_gpu_id", type=int, default=-1)
    parser.add_argument("--bert_gpu_id", type=int, default=-1)
    parser.add_argument("--use_gpu_id", type=int, default=-1)
    parser.add_argument("--bert_clf_steps", type=int, default=20000)

    # option on learning_approaches
    parser.add_argument("--exp_mode", type=str, default="generate_active_learning",
                        help="use generate_robust_tuning for robust training. \\"
                             "use generate_active_learning for active learning. \\"
                             "use generate_attack for attack.")
    parser.add_argument("--num_paraphrases_per_text", type=int, default=20)
    parser.add_argument("--subsample_testset", type=int, default=100)
    parser.add_argument("--paraphrase_strategy", type=str, default="RandomStrategy")
    parser.add_argument("--strategy_gpu_id", type=int, default=-1)
    parser.add_argument("--robust_tuning_steps", type=int, default=5000)
    parser.add_argument("--load_robust_tuned_clf_desc", type=str, default=None)
    
    #active learning
    parser.add_argument("--total_acquire_rounds", type=int, default=50)
    parser.add_argument("--acquire_method", type=str, default="Random")
    parser.add_argument("--acquire_data_num_per_round", type=int, default=10)
    parser.add_argument("--chart_group_name", type=str, default="Test")
    parser.add_argument("--ALmethod_desc", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dropout_samp_num", type=int, default=50)

    #model
    parser.add_argument('--clf_type', default='multi_label_classify', type=str,
                help="multi_classify. \\"
                     "multi_label_classify.")
    parser.add_argument('--clf_name', default='use_default_clf', type=str,
            help="GCNBert. \\"
                    "MABert. \\"
                    "MLPBert.")
    parser.add_argument('--model_init', default='bert-base-uncased', type=str,
        help="pretrained model name. Choose from bert-base-cased, bert-base-uncased, bert-large-cased, bert-large-uncased")
    parser.add_argument("--output_dim", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=1)
    parser.add_argument("--hidden_layer_num", type=int, default=512)
    parser.add_argument('--bert_trainable', default=True, type=bool,
                    help='bert_trainable')
    parser.add_argument("--performance_indicator", type=str, default="OF1")

    #model train
    parser.add_argument("--train_bs", type=int, default=8)
    parser.add_argument("--train_epochs", type=int, default=2)
    parser.add_argument('--learning_rate', default=0.01, type=float,
                metavar='LR', help='initial learning rate')
    parser.add_argument('--epoch_step', default=[150], type=int, nargs='+',
                        help='number of epochs to change learning rate')
    parser.add_argument("--evaluate", type=int, default=10)
    parser.add_argument("--eval_bs", type=int, default=16)
    parser.add_argument('--save_model_path', default='./checkpoint', type=str,
                        help='path to save checkpoint (default: none)')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--classifier_id', default=[0], type=int, nargs='+',
                help='')
    parser.add_argument('--print_freq', default=1000, type=int,
                    metavar='N', help='print frequency (default: 10)')

    # add builtin strategies' args to parser.
    for item in built_in_paraphrase_strategies.values():
        item.add_parser_args(parser)

    arg_dict = vars(parser.parse_args())
    assert arg_dict["output_dir"] is not None

    os.makedirs(arg_dict["output_dir"], exist_ok=True)
    os.makedirs(os.path.join(arg_dict["output_dir"], "log"), exist_ok=True)

    model_config = {'clf_name': arg_dict['clf_name'], 'output_dim': arg_dict['output_dim'], 'hidden_dim': arg_dict['hidden_dim'], 
            'hidden_layer_num': arg_dict['hidden_layer_num'], 'bert_trainable': arg_dict['bert_trainable']}

    train_config = {'train_bs': arg_dict['train_bs'], 'eval_bs': arg_dict['eval_bs'], 'train_epochs': arg_dict['train_epochs'], 
            'evaluate': arg_dict['evaluate'], 'save_model_path': arg_dict['save_model_path'], 'workers': arg_dict['workers'],
             'epoch_step': arg_dict['epoch_step'], 'learning_rate': arg_dict['learning_rate'], 'momentum': arg_dict['momentum'],
             'weight_decay': arg_dict['weight_decay'], 'model_init': arg_dict['model_init'], 'device_ids': arg_dict['classifier_id'],
             'print_freq': arg_dict['print_freq'], 'performance_indicator': arg_dict['performance_indicator']}

    data_config = {'dataset_name': arg_dict['dataset'], 'dataset_path': arg_dict['dataset_path'], 'output_dir': arg_dict['output_dir']}

    AL_config = {'seed': arg_dict['seed'], 'chart_group_name': arg_dict['chart_group_name'],
                'ALmethod_desc': arg_dict['ALmethod_desc'], 'total_acquire_rounds': arg_dict['total_acquire_rounds'], 
                'acquire_data_num_per_round': arg_dict['acquire_data_num_per_round'], 'acquire_method': arg_dict['acquire_method'],
                'dropout_samp_num': arg_dict['dropout_samp_num']}

    engine = Engine(arg_dict, data_config, model_config, train_config)
    
    log.add_file_handler(
        logger, os.path.join(arg_dict["output_dir"], "log.log"))
    log.remove_logger_tf_handler(logger)

    if arg_dict["exp_mode"] == "generate_robust_tuning":
        engine.run_generate_robust_tuning(paraphrase_strategy=arg_dict["paraphrase_strategy"],
                                    num_paraphrases_per_text=arg_dict["num_paraphrases_per_text"],
                                    tuning_steps=arg_dict["robust_tuning_steps"])

    elif arg_dict["exp_mode"] == "generate_active_learning":
        engine.run_generate_active_learning(AL_config, paraphrase_strategy=arg_dict["paraphrase_strategy"],
                                    num_paraphrases_per_text=arg_dict["num_paraphrases_per_text"])

    elif arg_dict["exp_mode"] == "generate_attack":
        engine.run_generate_attack(paraphrase_strategy=arg_dict["paraphrase_strategy"],
                                num_paraphrases_per_text=arg_dict["num_paraphrases_per_text"])


if __name__ == "__main__":
    main()