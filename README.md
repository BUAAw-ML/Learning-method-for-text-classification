# A Framework of Reducing Labeled Data for Training Multi-label Text Classification Model

面向未标注数据、生成数据，涉及采样函数，模型训练方法。


# Overview

Fibber is a library to evaluate different strategies to paraphrase natural language. In this library, we have several built-in paraphrasing strategies. We also have a benchmark framework to evaluate the quality of paraphrase. In particular, we use the GPT2 language model to measure how meaningful is the paraphrased text. We use a universal sentence encoder to evaluate the semantic similarity between original and paraphrased text. We also train a BERT classifier on the original dataset, and check of paraphrased sentences can break the text classifier.


## Use without install

If you are using this project for research purpose and want to make changes to the code,
you can install all requirements by

```bash
git clone git@github.com:DAI-Lab/fibber.git
cd fibber
pip install --requirement requirement.txt
```

Then you can use fibber by

```base
python -m datasets.download_datasets
python -m benchmark.benchmark
```

In this case, any changes you made on the code will take effect immediately.




# Quickstart

In this short tutorial, we will guide you through a series of steps that will help you
getting started with **fibber**.

**(1) [Install Fibber](#Install)**

**(2) Get a demo dataset and resources.**

```python
from datasets import get_demo_dataset

trainset, testset = get_demo_dataset()

from resources import download_all

# resources are downloaded to ~/cache
download_all()
```

**(3) Create a Fibber object.**

```python
from fibbercache import Fibber

# args starting with "bs_" are hyperparameters for the BertSamplingStrategy.
arg_dict = {
    "use_gpu_id": 0,
    "gpt2_gpu_id": 0,
    "bert_gpu_id": 0,
    "strategy_gpu_id": 0,
    "bs_block_size": 3,
    "bs_wpe_weight": 10000,
    "bs_use_weight": 1000,
    "bs_gpt2_weight": 10,
    "bs_clf_weight": 3
}

# create a fibber object.
# This step may take a while (about 1 hour) on RTX TITAN, and requires 20G of
# GPU memory. If there's not enough GPU memory on your GPU, consider assign use
# gpt2, bert, and strategy to different GPUs.
#
fibber = Fibber(arg_dict, dataset_name="demo", strategy_name="BertSamplingStrategy",
                trainset=trainset, testset=testset, output_dir="exp-demo")
```

**(4) You can also ask fibber to paraphrase your sentence.**

The following command can randomly paraphrase the sentence into 5 different ways.

```python
# Try sentences you like.
# label 0 means negative, and 1 means positive.
fibber.paraphrase(
    {"text0": ("The Avengers is a good movie. Although it is 3 hours long, every scene has something to watch."),
     "label": 1},
    field_name="text0",
    n=5)
```

The output is a tuple of (str, list, list).

```python
# Original Text
'The Avengers is a good movie. Although it is 3 hours long, every scene has something to watch.'

# 5 paraphrase_list
['the avengers is a good movie. even it is 2 hours long, there is not enough to watch.',
  'the avengers is a good movie. while it is 3 hours long, it is still very watchable.',
  'the avengers is a good movie and although it is 2 ¹⁄₂ hours long, it is never very interesting.',
  'avengers is not a good movie. while it is three hours long, it is still something to watch.',
  'the avengers is a bad movie. while it is three hours long, it is still something to watch.']

# Evaluation metrics of these 5 paraphrase_list.

  {'EditingDistance': 8,
   'USESemanticSimilarityMetric': 0.9523628950119019,
   'GloVeSemanticSimilarityMetric': 0.9795315341042675,
   'GPT2GrammarQualityMetric': 1.492070198059082,
   'BertClassifier': 0},
  {'EditingDistance': 9,
   'USESemanticSimilarityMetric': 0.9372092485427856,
   'GloVeSemanticSimilarityMetric': 0.9575780832312993,
   'GPT2GrammarQualityMetric': 0.9813404679298401,
   'BertClassifier': 1},
  {'EditingDistance': 11,
   'USESemanticSimilarityMetric': 0.9265919327735901,
   'GloVeSemanticSimilarityMetric': 0.9710499628056698,
   'GPT2GrammarQualityMetric': 1.325406551361084,
   'BertClassifier': 0},
  {'EditingDistance': 7,
   'USESemanticSimilarityMetric': 0.8913971185684204,
   'GloVeSemanticSimilarityMetric': 0.9800737898362042,
   'GPT2GrammarQualityMetric': 1.2504483461380005,
   'BertClassifier': 1},
  {'EditingDistance': 8,
   'USESemanticSimilarityMetric': 0.9124080538749695,
   'GloVeSemanticSimilarityMetric': 0.9744155151490856,
   'GPT2GrammarQualityMetric': 1.1626977920532227,
   'BertClassifier': 0}]
```

**(5) You can ask fibber to randomly pick a sentence from the dataset and paraphrase it.**


```python
fibber.paraphrase_a_random_sentence(n=5)
```



# Supported strategies

In this version, we implement three strategies

- IdentityStrategy:
	- The identity strategy outputs the original text as its paraphrase.
	- This strategy generates exactly 1 paraphrase for each original text regardless of `--num_paraphrases_per_text` flag.
- RandomStrategy:
	- The random strategy outputs the random shuffle of words in the original text.
- TextFoolerStrategy:
	- Implementation of [Jin et. al, 2019](https://arxiv.org/abs/1907.11932)
- BertSamplingStrategy:


# Data Format

## Dataset format

Each dataset is stored in multiple JSON files. For example, the ag dataset is stored in `train.json` and `test.json`.

The JSON file contains the following fields:

- `label_mapping`: a list of strings. The `label_mapping` maps an integer label to the actual meaning of that label. This list is not used in the algorithm.
- `cased`: a bool value indicates if it is a cased dataset or uncased dataset. Sentences in uncased datasets are all in lowercase.
- `paraphrase_field`: choose from `text0` and `text1`. `Paraphrase_field` indicates which sentence in each data record should be paraphrased. 
- `data`: a list of data records. Each data records contains:
	- `label`: an integer indicating the classification label of the text.
	- `text0`:
		- For topic and sentiment classification datasets, text0 stores the text to be classified.
		- For natural language inference datasets, text0 stores the premise.
	- `text1`:
		- For topic and sentiment classification datasets, this field is omitted.
		- For natural language inference datasets, text1 stores the hypothesis.

A topic / sentiment classification example:

```
{
  "label_mapping": [
    "World",
    "Sports",
    "Business",
    "Sci/Tech"
  ],
  "cased": true,
  "paraphrase_field": "text0",
  "data": [
    {
      "label": 1,
      "text0": "Boston won the NBA championship in 2008."
    },
    {
      "label": 3,
      "text0": "Apple releases its latest cell phone."
    },
    ...
  ]
}
```

A natural langauge inference example:

```
{
  "label_mapping": [
    "neutral",
    "entailment",
    "contradiction"
  ],
  "cased": true,
  "paraphrase_field": "text1",
  "data": [
    {
      "label": 0,
      "text0": "A person on a horse jumps over a broken down airplane.",
      "text1": "A person is training his horse for a competition."
    },
    {
      "label": 2,
      "text0": "A person on a horse jumps over a broken down airplane.",
      "text1": "A person is at a diner, ordering an omelette."
    },
    ...
  ]
}
```


### Download datasets

We have scripts to help you easily download all datasets. We provide two options to download datasets:

**Download data preprocessed by us.** 

We preprocessed datasets and uploaded them to AWS. You can use the following command to download all datasets.

```
python3 -m datasets.download_datasets
```

After executing the command, the dataset is stored at `~/cache/datasets/<dataset_name>/*.json`. For example, the ag dataset is stored in `~/cache/datasets/ag/`. And there will be two sets `train.json` and `test.json` in the folder.

**Download and process data from the original source.** 

You can also download the original dataset version and process it locally.

```
python3 -m datasets.download_datasets --process_raw 1
```
This script will download data from the original source to `~/cache/datasets/<dataset_name>/raw/` folder. And process the raw data to generate the JSON files.


## Output format

During the benchmark process, we save results in several files.

### Intermediate result

The intermediate result `<output_dir>/<dataset>-<strategy>-<date>-<time>-tmp.json` stores the paraphrased sentences. Strategies can run for a few minutes (hours) on some datasets, so we save the result every 30 seconds. The file format is similar to the dataset file. For each data record, we add a new field, `text0_paraphrases` or `text1_paraphrases` depending the `paraphrase_field`.

An example is as follows.

```
{
  "label_mapping": [
    "World",
    "Sports",
    "Business",
    "Sci/Tech"
  ],
  "cased": true,
  "paraphrase_field": "text0",
  "data": [
    {
      "label": 1,
      "text0": "Boston won the NBA championship in 2008.",
      "text0_paraphrases": ["The 2008 NBA championship is won by Boston.", ...]
    },
    ...
  ]
}
```

### Result with metrics

The result `<output_dir>/<dataset>-<strategy>-<date>-<time>-with-metrics.json` stores the paraphrased sentences as well as metrics. Compute metrics may need a few minutes on some datasets, so we save the result every 30 seconds. The file format is similar to the intermediate file. For each data record, we add two new field, `original_text_metrics` and `paraphrase_metrics`.

An example is as follows.

```
{
  "label_mapping": [
    "World",
    "Sports",
    "Business",
    "Sci/Tech"
  ],
  "cased": true,
  "paraphrase_field": "text0",
  "data": [
    {
      "label": 1,
      "text0": "Boston won the NBA championship in 2008.",
      "text0_paraphrases": [..., ...],
      "original_text_metrics": {
        "EditDistanceMetric": 0,
        "USESemanticSimilarityMetric": 1.0,
        "GloVeSemanticSimilarityMetric": 1.0,
        "GPT2GrammarQualityMetric": 1.0,
        "BertClassifier": 1
      },
      "paraphrase_metrics": [
        {
          "EditDistanceMetric": 7,
          "USESemanticSimilarityMetric": 0.91,
          "GloVeSemanticSimilarityMetric": 0.94,
          "GPT2GrammarQualityMetric": 2.3,
          "BertClassifier": 1
        },
        ...
      ]
    },
    ...
  ]
}
```

The `original_text_metrics` stores a dict of several metrics. It compares the original text against itself. The `paraphrase_metrics` is a list of the same length as paraphrases in this data record. Each element in this list is a dict showing the comparison between the original text and one paraphrased text.


# Benchmark

Benchmark module is an important component in  It provides an easy-to-use
API and is highly customizable. In this document, we will show

- Built-in Datasets: we preprocessed 6 datasets into [fibber's format](https://dai-lab.github.io/fibber/dataformat.html).
- Benchmark result: we benchmark all built-in methods on built-in dataset.
- Basic usage: how to use builtin strategies to attack BERT classifier on a built-in dataset.
- Advance usage: how to customize strategy, classifier, and dataset.

## Built-in Datasets

Here is the information about datasets in 

| Type                       | Name                    | Size (train/test) | Classes                             |
|----------------------------|-------------------------|-------------------|-------------------------------------|
| Topic Classification       | [ag](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)| 120k / 7.6k       | World / Sport / Business / Sci-Tech                                                                                |
| Sentiment classification   | [mr](http://www.cs.cornell.edu/people/pabo/movie-review-data/)           | 9k / 1k           |  Negative / Positive |
| Sentiment classification   | [yelp](https://academictorrents.com/details/66ab083bda0c508de6c641baabb1ec17f72dc480) | 160k / 38k        | Negative / Positive                 |
| Sentiment classification   | [imdb](https://ai.stanford.edu/~amaas/data/sentiment/)| 25k / 25k         | Negative / Positive                 |
| Natural Language Inference | [snli](https://nlp.stanford.edu/projects/snli/) | 570k / 10k        | Entailment / Neutral / Contradict   |
| Natural Language Inference | [mnli](https://cims.nyu.edu/~sbowman/multinli/)                | 433k / 10k        | Entailment / Neutral / Contradict   |

Note that ag has two configurations. In `ag`, we combines the title and content as input for classification. In `ag_no_title`, we use only use content as input.

Note that mnli has two configurations. Use `mnli` for matched testset, and `mnli_mis` for mismatched testset.

## Benchmark result

The following table shows the benchmarking result. (Here we show the number of wins.)

| StrategyName         | AfterAttackAccuracy | GPT2GrammarQuality | GloVeSemanticSimilarity | USESemanticSimilarity |
|----------------------|---------------------|--------------------|-------------------------|-----------------------|
| IdentityStrategy     | 0                   | 0                  | 0                       | 0                     |
| RandomStrategy       | 5                   | 0                  | 16                      | 6                     |
| TextFoolerJin2019    | 25                  | 12                 | 6                       | 11                    |
| BAEGarg2019          | 13                  | 5                  | 14                      | 13                    |
| PSOZang2020          | 16                  | 12                 | 6                       | 5                     |
| BertSamplingStrategy | 33                  | 23                 | 10                      | 17                    |

For detailed tables, see [Google Sheet](https://docs.google.com/spreadsheets/d/1B_5RiMfndNVhxZLX5ykMqt5SCjpy3MxOovBi_RL41Fw/edit?usp=sharing).


## Basic Usage

In this short tutorial, we will guide you through a series of steps that will help you run
benchmark on builtin strategies and datasets.

### Preparation

**Install Fibber:** Please follow the instructions to [Install Fibber](https://dai-lab.github.io/fibber/readme.html#install).**

**Download datasets:** Please use the following command to download all datasets.

```bash
python -m datasets.download_datasets
```

All datasets will be downloaded and stored at `~/cache/datasets`.

### Run benchmark as a module

If you are trying to reproduce the performance table, running the benchmark as a module is
recommended.

The following command will run the `BertSamplingStrategy` strategy on the `mr` dataset. To use other
datasets, see the [datasets](#Datasets) section.

```bash
python -m benchmark.benchmark \
	--dataset mr \
	--strategy BertSamplingStrategy \
	--output_dir exp-mr \
	--num_paraphrases_per_text 20 \
	--subsample_testset 100 \
	--gpt2_gpu 0 \
	--bert_gpu 0 \
	--use_gpu 0 \
	--bert_clf_steps 20000
```

It first subsamples the test set to `100` examples, then generates `20` paraphrases for each
example. During this process, the paraphrased sentences will be stored at
`exp-mr/mr-BertSamplingStrategy-<date>-<time>-tmp.json`.

Then the pipeline will initialize all the evaluation metrics.

- We will use a `GPT2` model to evaluate if a sentence is meaningful. The `GPT2` language model will be executed on `gpt2_gpu`. You should change the argument to a proper GPU id.
- We will use a `Universal sentence encoder (USE)` model to measure the similarity between two paraphrased sentences and the original sentence. The `USE` will be executed on `use_gpu`. You should change the argument to a proper GPU id.
- We will use a `BERT` model to predict the classification label for paraphrases. The `BERT` will be executed on `bert_gpu`. You should change the argument to a proper GPU id. **Note that the BERT classifier will be trained for the first time you execute the pipeline. Then the trained model will be saved at `~/cache/bert_clf/<dataset_name>/`. Because of the training, it will use more GPU memory than GPT2 and USE. So assign BERT to a separate GPU if you have multiple GPUs.**

After the execution, the evaluation metric for each of the paraphrases will be stored at `exp-ag/ag-RandomStrategy-<date>-<time>-with-metrics.json`.

The aggregated result will be stored as a row at `~/cache/results/detailed.csv`.

### Run in a python script / jupyter notebook

You may want to integrate the benchmark framework into your own python script. We also provide easy to use APIs.

**Create a Benchmark object** The following code will create a fibber Benchmark object on `mr` dataset.

```
from benchmark import Benchmark

benchmark = Benchmark(
    output_dir = "exp-debug",
    dataset_name = "mr",
    subsample_attack_set=100,
    use_gpu_id=0,
    gpt2_gpu_id=0,
    bert_gpu_id=0,
    bert_clf_steps=1000,
    bert_clf_bs=32
)
```

Similarly, you can assign different components to different GPUs.

**Run benchmark** Use the following code to run the benchmark using a specific strategy.

```
benchmark.run_benchmark(paraphrase_strategy="BertSamplingStrategy")
```

### Generate overview result

We use the number of wins to compare different strategies. To generate the overview table, use the following command.

```bash
python -m benchmark.make_overview
```

The overview table will be stored at `~/cache/results/overview.csv`.

Before running this command, please verify `~/cache/results/detailed.csv`. Each strategy must not have more than one executions on one dataset. Otherwise, the script will raise assertion errors.


## Advanced Usage

### Customize dataset

To run a benchmark on a customized classification dataset, you should first convert a dataset into [fibber's format](https://dai-lab.github.io/fibber/dataformat.html).

Then construct a benchmark object using your own dataset.

```
benchmark = Benchmark(
    output_dir = "exp-debug",
    dataset_name = "customized_dataset",

    ### Pass your processed datasets here. ####
    trainset = your_train_set,
    testset = your_test_set,
    attack_set = your_attack_set,
    ###########################################

    subsample_attack_set=0,
    use_gpu_id=0,
    gpt2_gpu_id=0,
    bert_gpu_id=0,
    bert_clf_steps=1000,
    bert_clf_bs=32
)
```

### Customize classifier

To customize classifier, use the `customized_clf` arg in Benchmark. For example,

```
# a naive classifier that always outputs 0.
class CustomizedClf(ClassifierBase):
	def measure_example(self, origin, paraphrase, data_record=None, paraphrase_field="text0"):
		return 0

benchmark = Benchmark(
    output_dir = "exp-debug",
    dataset_name = "mr",

    # Pass your customized classifier here.
    # Note that the Benchmark class will NOT train the classifier.
    # So please train your classifier before pass it to Benchmark.
    customized_clf=CustomizedClf(),

    subsample_attack_set=0,
    use_gpu_id=0,
    gpt2_gpu_id=0,
    bert_gpu_id=0,
    bert_clf_steps=1000,
    bert_clf_bs=32
)
```

### Customize strategy

To customize strategy, you should create a strategy object then call the `run_benchmark` function. For example,
we want to benchmark BertSamplingStrategy using a different set of hyper parameters.

```
strategy = BertSamplingStrategy(
    arg_dict={"bs_clf_weight": 0},
    dataset_name="mr",
    strategy_gpu_id=0,
    output_dir="exp_mr",
    metric_bundle=benchmark.get_metric_bundle())

benchmark.run_benchmark(strategy)
```

## Adversarial Training

Adversarial training is a natural way to defend against attacks.

Fibber provides a simple way to fine-tune a classifier on paraphrases generated by paraphrase strategies. 

### Training and test using command line
 
The following command uses bert sampling strategy to fine-tune the default bert classifier. 

```
python -m benchmark.benchmark \
    --robust_tuning 1 \
    --robust_tuning_steps 5000 \
    --dataset mr \
    --strategy BertSamplingStrategy \
    --output_dir exp-mr \
    --num_paraphrases_per_text 20 \
    --subsample_testset 100 \
    --gpt2_gpu 0 \
    --bert_gpu 0 \
    --use_gpu 0 \
    --bert_clf_steps 5000
```

The fine-tuned classifier will be stored at `~/cache/bert_clf/mr/DefaultTuningStrategy-BertSamplingStrategy`

After the fine-tuning, you can use the following command to attack the fine-tuned classifier using BertSamplingStrategy. You do not need to use the same paraphrasing strategy for tuning and attack.

```
python -m benchmark.benchmark \
    --robust_tuning 0 \
    --robust_tuning_steps 5000 \
    --load_robust_tuned_clf_desc DefaultTuningStrategy-BertSamplingStrategy \
    --dataset mr \
    --strategy BertSamplingStrategy \
    --output_dir exp-mr \
    --num_paraphrases_per_text 20 \
    --subsample_testset 100 \
    --gpt2_gpu 0 \
    --bert_gpu 0 \
    --use_gpu 0 \
    --bert_clf_steps 5000
```



