# MRLEval - a benchmark for morphologically rich languages

Note: This is not an officially supported Google product.

## Introduction

This repository contains code for downloading, processing, fine-tuning, running
inference, and evaluating models in a fine-tuning setting on various natural
language tasks in Hebrew, Modern Standard Arabic and Levantine Arabic. The tasks
are detailed [below](#tasks).

Scripts are included to fine-tune encoder-decoder and decoder LLMs, and generate
test set predictions using both [Huggingface transformers](#huggingface) and
[T5X](#t5x). An [evaluation script](#evaluation) calculates performance metrics
from these predictions. [Baseline results](#baseline_results) on all tasks using
mt5-XL are provided.

## Tasks

The following tasks are supported:

| Language               | Name               | Task                       | Metric    | Paper / Page                                                                                               |
|------------------------|--------------------|----------------------------|-----------|------------------------------------------------------------------------------------------------------------|
| Hebrew                 | HeQ                | Question Answering         | TLNLS     | [Paper](https://aclanthology.org/2023.findings-emnlp.915/)                                                 |
| Hebrew                 | HeQ-QG             | Question Generation        | Rouge     | [Paper](https://aclanthology.org/2023.findings-emnlp.915/)                                                  |
| Hebrew                 | HeSum              | Summarization              | Rouge     | [Paper](https://arxiv.org/pdf/2406.03897)                                                                   |
| Hebrew                 | HebSummaries       | Summarization              | Rouge     | [Page](https://huggingface.co/datasets/HebArabNlpProject/HebSummaries)                                       |
| Hebrew                 | HeSentiment        | Sentiment Analysis         | Macro F1  | [Page](https://huggingface.co/datasets/HebArabNlpProject/HebrewSentiment)                                    |
| Hebrew                 | Nemo-Token         | NER (token level)          | F1        | [Paper](https://arxiv.org/pdf/2007.15620)                                                                   |
| Hebrew                 | Nemo-Morph         | NER (morph level)          | F1        | [Paper](https://arxiv.org/pdf/2007.15620)                                                                   |
| Hebrew                 | HebNLI             | Natural Language Inference | Macro F1  | [Page](https://github.com/NNLP-IL/HebNLI)                                                                   |
| Hebrew                 | HebCo          | Coreference Resolution         | Macro F1  | [Page](https://github.com/IAHLT/coref)                                                                      |
| &nbsp;                       |                    |                            |            
| Modern Standard Arabic | ArQ-MSA-QA         | Question Answering         | TLNLS     | [Page](https://huggingface.co/datasets/HebArabNlpProject/ArQ)                                                |
| Modern Standard Arabic | ArQ-MSA-QG         | Question Generation        | Rouge     | [Page](https://huggingface.co/datasets/HebArabNlpProject/ArQ)                                                |
| Modern Standard Arabic | ArTyDiQA-QA        | Question Answering         | TyDiQA-F1     | [Page](https://github.com/google-research-datasets/artydiqa)                                                |
| Modern Standard Arabic | ArTyDiQA-QG        | Question Generation        | Rouge     | [Page](https://github.com/google-research-datasets/artydiqa)                                                |
| Modern Standard Arabic | IAHLT-NER          | Named Entity Recognition                        | F1        | [Page](https://huggingface.co/datasets/HebArabNlpProject/arabic-iahlt-NER)                                  |
| Modern Standard Arabic | ArXLSum          | Summarization                        | Rouge        | [Page](https://huggingface.co/datasets/csebuetnlp/xlsum)                                  |
| Modern Standard Arabic | ArabicNLI         | Natural Language Inference                       | Macro F1        | [Page](https://huggingface.co/datasets/facebook/xnli)                                  |
| &nbsp;                       |                    |                            |            
| Levantine Arabic       | ArSentiment        | Sentiment Analysis         | Macro F1        | [Page](https://huggingface.co/datasets/HebArabNlpProject/ArabicSentimentDataSet)                            |
| Levantine Arabic       | ArCoref            | Coreference                | Macro F1        | [Page](https://huggingface.co/datasets/HebArabNlpProject/ArabCoRef)                                          |
| Levantine Arabic       | ArQ-Spoken-QA      | Question Answering         | TLNLS     | [Page](https://huggingface.co/datasets/HebArabNlpProject/ArQ)                                      |
| Levantine Arabic       | ArQ-Spoken-QG      | Question Generation        | Rouge     | [Page](https://huggingface.co/datasets/HebArabNlpProject/ArQ)                                      |

## Setup

Note that this package requires Python 3.10 or higher.

First, clone the repository:

```bash
git clone https://github.com/google-research/mrl_eval.git
```

Then, install the requirements, preferably in a new virtual environment.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Data

Download and preprocess raw data for all tasks:

```bash
bash mrl_eval/datasets/download_raw_data.sh
bash mrl_eval/datasets/ingest_all_datasets.sh
```

## Evaluation

To evaluate the score of model predictions, run:

```bash
python -m mrl_eval.evaluation.evaluate --dataset {dataset} --predictions_path path/to/prediction/file
```

The options for `dataset` are:

*   heq
*   heq_question_gen
*   hesum
*   hebsummaries
*   hesentiment
*   nemo_token
*   nemo_morph
*   hebnli
*   hebco
*   arabic_nli
*   arq_MSA
*   arq_MSA_question_gen
*   arq_spoken
*   arq_spoken_question_gen
*   arsentiment
*   arcoref
*   artydiqa
*   artydiqa_question_gen
*   ar_xlsum
*   iahlt_ner

Your predictions file is expected to be a jsonl file in the following format:

```json
{"input": {"id": "example_id_1"}, "prediction": "prediction1"}
{"input": {"id": "example_id_2"}, "prediction": "prediction2"}
...
```

## Baseline results

We finetune mT5-xl model per task as the first baseline. Results are shown in
the table below.

| Language               | Model   | Task             | Metric           | Value               |
|------------------------|---------|------------------|------------------|---------------------|
| Hebrew                 | mT5-XL  | HeQ              | TLNLS            | 87.1                |
| Hebrew                 | mT5-XL  | HeQ-QG           | R1/R2/RL         | 40.2 / 22.0 / 39.7  |
| Hebrew                 | mT5-XL  | HeSum            | R1/R2/RL         | 17.9 / 7.2 / 15.0   |
| Hebrew                 | mT5-XL  | HebSummaries     | R1/R2/RL         | 23.9 / 10.1 / 16.6  |
| Hebrew                 | mT5-XL  | NEMO             | Token / Morph F1 | 86.3 / 84.8         |
| Hebrew                 | mT5-XL  | Sentiment        | Macro F1         | 85.0                |
| Hebrew                 | mT5-XL  | HebNLI           | Macro F1         | 84.6                |
| Hebrew                 | mT5-XL  | Hebco            | Macro F1         | 49.3                |
| &nbsp;                 |         |                  |                  |                     |
| Modern Standard Arabic | mT5-XL  | ArQ-MSA-QA       | TLNLS            | 79.5                |
| Modern Standard Arabic | mT5-XL  | ArQ-MSA-QG       | R1/R2/RL         | 35.8 / 17.2 / 35.5  |
| Modern Standard Arabic | mT5-XL  | ArTyDi-QA        | TyDiQA-F1            | 87.4            |
| Modern Standard Arabic | mT5-XL  | ArTyDi-QG        | R1/R2/RL         | 60.6 / 44.1 / 60.5  |
| Modern Standard Arabic | mT5-XL  | IAHLT-NER        | Token F1         | 64.6                |
| Modern Standard Arabic | mT5-XL  | ArabicNLI        | Macro F1         |  82.2 |
| Modern Standard Arabic | mT5-XL  | ArXLSum          | R1/R2/RL         | 26.5 / 11.4 / 23.4  |
| &nbsp;                 |         |                  |                  |                     |
| Levantine Arabic       | mT5-XL  | ArSentiment      | Macro F1         | 71.2                |
| Levantine Arabic       | mT5-XL  | ArCoref          | Macro F1         | 50.1                |
| Levantine Arabic       | mT5-XL  | ArQ-spoken-QA    | TLNLS            | 81.8                |
| Levantine Arabic       | mT5-XL  | ArQ-spoken-QG    | R1/R2/RL         | 35.6 / 16.6 / 35.3  |

## Fine-tuning and inference

### Huggingface

To finetune on a specific dataset:

```bash
python -m mrl_eval.hf.finetune --dataset {dataset}
```

By default, this will train `mt5-xl`. To train a different model (e.g. a
decoder LLM) specify its HF model name as follows:

```bash
python -m mrl_eval.hf.finetune --dataset {dataset} --model "google/gemma-2-9b"
```

Decoder model will be trained by default with LORA using half precision.

The options for `dataset` are the same as [above](#evaluation).

Once the training is done, the script will print the path to the best
checkpoint.

To generate response for the inputs of the test set:

```bash
python -m mrl_eval.hf.generate --dataset {dataset} --checkpoint_path path/to/checkpoint
```

### T5X

#### Establishing a GCP project

First, follow the guidelines at
[XManager](https://github.com/google-deepmind/xmanager) for establishing a
google cloud project. Specifically, follow the guidelines for setting up a
Google Cloud project. You will be using two cloud infrastructures: a bucket for
storing your training outputs (logs, model checkpoints) and a compute engine
where you will run the project. **We will be using the bucket path in the
training and inference scripts.** Follow the instructions at
[T5X](https://github.com/google-research/t5x) to request an appropriate VM. **We
will be setting up the project environment inside this VM.**

#### Setting up MRLEval in GCP

Second, proceed to build the environment **inside your compute engine**. All of
the following should happen from your GCP VM:

##### 1. Follow the instruction to install [T5X](https://github.com/google-research/t5x) as well as [XManager](https://github.com/google-deepmind/xmanager).

We will be using the path to the cloned T5X repo in the training and inference
scripts.

##### 2. Clone MRLEval

*   No need to install the requirements; this will be handled implicitly by
    XManager via the fine-tune and inference script arguments.

##### 3. Run Data Ingestion.

Download and preprocess raw data for all tasks (note the save_tfrecord flag):

```bash
bash mrl_eval/datasets/download_raw_data.sh
bash mrl_eval/datasets/ingest_all_datasets.sh save_tfrecord
```

At this point your project structure will be similar to:

```
${HOME}
└── some_dir
       └── main_project_dir
           ├── mrl_eval # where you cloned mrl_eval
           └── mrl_eval_data # a directory for data outputs, will be created when running Data-Ingestion
       └── cloned_t5x_repo # where you cloned t5x
```

*   It is important that your ingested datasets will be located at the data
    directory that shares the root main_project_dir with the cloned mrl_eval
    repo. This should happen on its own when ingesting the data (3.).
*   The naming in the following section refers to this example.

##### 4. Define the following variables before running the scripts:

```
export GOOGLE_CLOUD_BUCKET_NAME=<your_bucket_name> # Without the gs:// prefix
export PROJECT_DIR_ROOT=<${HOME}/some_dir/main_project_dir>
export T5X_DIR=<${HOME}/some_dir/cloned_t5x_repo>
```

##### 5. Finetuning mT5-xl and Running Inference

The finetuning script expects two argument: you name for the experiment and a
path to a gin configuration defining the training on a given task. All finetune
and inference configurations for mT5-xl can be found under
`mrl_eval/models/gin/finetune_gin_configs` and
`mrl_eval/models/gin/inference_gin_configs` respectively.

To finetune mT5-xl on a given task, e.g. summarisation (hesum), run:

```bash
cd ${HOME}/some_dir/main_project_dir/scripts
sh xm_finetune.sh <your_chosen_name_for_the_experiment> mrl_eval/models/gin/finetune_gin_configs/finetune_mt5_xl_hesum.gin
```

Similarly, to run inference on a checkpoint in your bucket (checkpoints are
saved to your bucket), run:

```bash
cd ${HOME}/some_dir/main_project_dir/scripts
sh xm_infer.sh  <your_chosen_name_for_the_inference> <task_eval_gin> <the_path_to_the_checkpoint>
```

e.g. to evaluate the hesum checkpoint at
gs://my_bucket/t5x/hesum_exp/20240722/logs/checkpoint_1004096

run:

```bash
cd ${HOME}/some_dir/main_project_dir/scripts
sh xm_infer.sh  infer_mt5xl_hesum mrl_eval/models/gin/inference_gin_configs/eval_mt5_xl_hesentiment.gin gs://my_bucket/t5x/hesum_exp/20240722/logs/checkpoint_1004096
```
