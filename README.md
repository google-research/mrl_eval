# MRLEval - a benchmark for morphologically rich languages
Note: This is not an officially supported Google product.

This repository contains code for downloading, processing, fine-tuning, running
inference, and evaluating models in a fine-tuning setting on various natural
language tasks in Hebrew (additional languages to follow). The tasks are
detailed in the following table.

Name        | Task                       | Metric   | Paper / Page
----------- | -------------------------- | -------- | ------------
HeQ         | Question Answering         | TLNLS    | [paper](https://aclanthology.org/2023.findings-emnlp.915/)
HeQ-QG      | Question Generation        | Rouge    | New, same data as HeQ
HeSum       | Summarization              | Rouge    | [paper](https://arxiv.org/pdf/2406.03897)
HeSentiment | Sentiment Analysis         | Macro F1 | [page](https://huggingface.co/datasets/HebArabNlpProject/HebrewSentiment)
Nemo-Token  | NER (token level)          | F1       | [paper](https://arxiv.org/pdf/2007.15620)
Nemo-Morph  | NER (morph level)          | F1       | [paper](https://arxiv.org/pdf/2007.15620)
HebNLI      | Natural Language Inference | Macro F1 | [page](https://github.com/NNLP-IL/HebNLI)

## Setup

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
python -m mrl_eval.evaluation.evaluate --dataset {dataset} --prediction_path path/to/prediction/file
```

The options for `dataset` are:

*   heq
*   heq_question_gen
*   hesum
*   hesentiment
*   nemo_token
*   nemo_morph
*   hebnli

Your predictions file is expected to be a jsonl file in the following format:

```json
{"input": {"id": "example_id_1"}, "prediction": "prediction1"}
{"input": {"id": "example_id_2"}, "prediction": "prediction2"}
...
```

## Baseline

We finetune mT5-xl model per task as the first baseline. Results are shown in
the table below.

<table>
<tr>
<th></th> <th>HeQ</th> <th>HeQ-QG</th> <th>HeSum</th> <th>NEMO</th> <th>Sentiment</th> <th>HeBNLI</th>
</tr>
<tr>
<td>Model</td> <td>TLNLS</td> <td>R1/R2/RL</td> <td>R1/R2/RL</td> <td>Token/Morph F1</td> <td>Macro F1</td> <td>Macro</td>
</tr>
<tr>
<td>mT5-XL</td> <td>83.6</td> <td>33.5/16.9/33.1</td> <td>17.9/7.2/15.0</td> <td>86.3/84.8</td> <td>85.0</td> <td>84.6</td>
</tr>
</table>

We provide scripts to finetune mT5 and generate responses to the test sets using
both [T5X](#t5x) and [Huggignface transformers](#huggignface transformers).

### T5X

#### Establishing a GCP

First, follow the guidelines at
[XManager](https://github.com/google-deepmind/xmanager) for establishing a
google cloud project. Specifically, follow the guidelines for setting up a
Google Cloud project. You will be using two cloud infrastructures: a bucket for
storing your training outputs (logs, model checkpoints) and a compute engine where you will run the
project. **We will be using the bucket path in the training and inference
scripts.** Follow the instructions at
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

### Huggignface transformers

To finetune on a specific dataset:

```bash
python mrl_eval.hf.finetune --dataset {dataset}
```

The options for `dataset` are the same as [above](#evaluation).

Once the training is done, the script will print the path to the best
checkpoint.

To generate response for the inputs of the test set:

```bash
python mrl_eval.hf.generate --dataset {dataset} --checkpoint_path path/to/checkpoint
```
