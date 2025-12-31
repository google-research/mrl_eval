# MRLEval - A Benchmark for Morphologically Rich Languages

Note: This is not an officially supported Google product.

## Introduction

This repository contains code for evaluating LLMs on various natural language
tasks in three Semitic Languages: Modern Hebrew, Modern Standard Arabic and
Levantine Arabic. The tasks are detailed [below](#tasks). This page includes
instructions for downloading, processing, fine-tuning, running inference, and
evaluating models in a fine-tuning setting.

We include scripts for fine-tuning encoder-decoder and decoder LLMs, and
for generating test set predictions using both Huggingface transformers and T5X.
The [evaluation script](#evaluation) calculates performance metrics from these
predictions. In this page we provide [Baseline results](#baseline_results) on
all tasks using mt5-XL.

## Tasks

The following tasks are supported:

| Language               | Name               | Task                       | Metric    | Paper / Page                                                                                               |
|------------------------|--------------------|----------------------------|-----------|------------------------------------------------------------------------------------------------------------|
| Hebrew                 | HebNLI             | Natural Language Inference | Macro F1  | [Page](https://github.com/NNLP-IL/HebNLI)                                                                   |
| Hebrew                 | HeSentiment        | Sentiment Analysis         | Macro F1  | [Page](https://huggingface.co/datasets/HebArabNlpProject/HebrewSentiment)                                    |
| Hebrew                 | HeQ                | Question Answering         | TLNLS     | [Paper](https://aclanthology.org/2023.findings-emnlp.915/)                                                 |
| Hebrew                 | HeQ-QG             | Question Generation        | Rouge     | [Paper](https://aclanthology.org/2023.findings-emnlp.915/)                                                  |
| Hebrew                 | HeSum              | Summarization              | Rouge     | [Paper](https://arxiv.org/pdf/2406.03897)                                                                   |
| Hebrew                 | HebSummaries       | Summarization              | Rouge     | [Page](https://huggingface.co/datasets/HebArabNlpProject/HebSummaries)                                       |
| Hebrew                 | Nemo-Token         | NER (token level)          | F1        | [Paper](https://arxiv.org/pdf/2007.15620)                                                                   |
| Hebrew                 | Nemo-Morph         | NER (morph level)          | F1        | [Paper](https://arxiv.org/pdf/2007.15620)                                                                   |
| Hebrew                 | HebCo          | [Coreference Resolution](datasets/README.md)         | Macro F1  | [Page](https://github.com/IAHLT/coref)                                                                      |
| &nbsp;                       |                    |                            |
| Modern Standard Arabic | ArabicNLI         | Natural Language Inference                       | Macro F1        | [Page](https://huggingface.co/datasets/facebook/xnli)                                  |
| Modern Standard Arabic | MSA Sentiment         | Sentiment Analysis                       | Macro F1        | [Page](https://github.com/mohamedadaly/LABR/tree/master)                                  |
| Modern Standard Arabic | ArQ-MSA-QA         | Question Answering         | TLNLS     | [Page](https://huggingface.co/datasets/HebArabNlpProject/ArQ)                                                |
| Modern Standard Arabic | ArQ-MSA-QG         | Question Generation        | Rouge     | [Page](https://huggingface.co/datasets/HebArabNlpProject/ArQ)                                                |
| Modern Standard Arabic | ArTyDiQA-QA        | Question Answering         | TyDiQA-F1     | [Page](https://github.com/google-research-datasets/artydiqa)                                                |
| Modern Standard Arabic | ArTyDiQA-QG        | Question Generation        | Rouge     | [Page](https://github.com/google-research-datasets/artydiqa)                                                |
| Modern Standard Arabic | ArXLSum          | Summarization                        | Rouge        | [Page](https://huggingface.co/datasets/csebuetnlp/xlsum)                                  |
| Modern Standard Arabic | ASAS          | Summarization                        | Rouge        | [Page](https://huggingface.co/datasets/HebArabNlpProject/ASAS)                                  |
| Modern Standard Arabic | IAHLT-NER          | Named Entity Recognition                        | F1        | [Page](https://huggingface.co/datasets/HebArabNlpProject/arabic-iahlt-NER)                                  |
| Modern Standard Arabic & Levantine Arabic | Wojood-Full  | NER (token level) | F1  | [Paper](https://aclanthology.org/2022.lrec-1.387/) |
| Modern Standard Arabic | Wojood-MSA  | NER (token level) | F1  | [Paper](https://aclanthology.org/2022.lrec-1.387/) |
| Modern Standard Arabic | OntoNotes  | [Coreference Resolution](datasets/README.md)   | Macro F1 | [Page](https://catalog.ldc.upenn.edu/LDC2013T19)
| &nbsp;                       |                    |                            |
| Levantine Arabic       | ArSentiment        | Sentiment Analysis         | Macro F1        | [Page](https://huggingface.co/datasets/HebArabNlpProject/ArabicSentimentDataSet)                            |
| Levantine Arabic       | ArQ-Spoken-QA      | Question Answering         | TLNLS     | [Page](https://huggingface.co/datasets/HebArabNlpProject/ArQ)                                      |
| Levantine Arabic       | ArQ-Spoken-QG      | Question Generation        | Rouge     | [Page](https://huggingface.co/datasets/HebArabNlpProject/ArQ)                                      |
| Levantine Arabic | Wojood-Spoken  | NER (token level) | F1  | [Paper](https://aclanthology.org/2022.lrec-1.387/) |
| Levantine Arabic | ShamNER  | Named Entity Recognition | F1  | [Page](https://huggingface.co/datasets/HebArabNlpProject/ShamNER) |
| Levantine Arabic       | ArCoref            | [Coreference Resolution](datasets/README.md)   | Macro F1        | [Page](https://huggingface.co/datasets/HebArabNlpProject/ArabCoRef)                                          |

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

Download and preprocess raw data for all tasks (done once per dataset):

```bash
bash mrl_eval/datasets/download_raw_data.sh
bash mrl_eval/datasets/ingest_all_datasets.sh
```

### OntoNotes Dataset Preparation

To prepare the dataset for the OntoNotes task (End-to-end and Gold Mentions
variants) follow these steps:

1.  **Download the OntoNotes 5.0 dataset:** Obtain the dataset from the LDC website ([LDC2013T19](https://catalog.ldc.upenn.edu/LDC2013T19)).
2.  **Generate CoNLL 2012 Data Split in JSONL format:** Run this script to generate the `jsonl` files for the Arabic subset of the dataset: [setup\_training.sh]([setup_training.sh](https://github.com/kentonl/e2e-coref/blob/master/setup_training.sh)).
3.  **Run the data ingestion script:** Copy the generated files to the data directory `mrl_eval_data/ontonotes/` and `mrl_eval_data/ontonotes_gold_mentions/`, and run the ingestion script:

```
python -m mrl_eval.datasets.ingest_dataset ontonotes
python -m mrl_eval.datasets.ingest_dataset ontonotes_gold_mentions
```

## Fine-tuning and inference

To finetune on a specific dataset using Huggingface, run:

```bash
python -m mrl_eval.hf.finetune --dataset {dataset}
```

The options for `dataset` are:

*   hebnli
*   hesentiment
*   heq
*   heq_question_gen
*   hesum
*   hebsummaries
*   nemo_token
*   nemo_morph
*   hebco
*   hebco_gold_mentions
*   arabic_nli
*   msa_sentiment
*   arq_MSA
*   arq_MSA_question_gen
*   artydiqa
*   artydiqa_question_gen
*   ar_xlsum
*   asas
*   iahlt_ner
*   wojood_msa
*   wojood_full
*   ontonotes
*   ontonotes_gold_mentions
*   arsentiment
*   arq_spoken
*   arq_spoken_question_gen
*   wojood_spoken
*   shamner
*   arcoref
*   arcoref_gold_mentions

By default, the `mt5-xl` model will be trained. To train a different model (e.g.
a decoder LLM) specify its HF model name as follows:

```bash
python -m mrl_eval.hf.finetune --dataset {dataset} --model "google/gemma-2-9b"
```

Decoder models will be trained by default with LORA using half precision.

Once the training is done, the script will print the path to the best
checkpoint.

To generate responses for the inputs of the test set, run:

```bash
python -m mrl_eval.hf.generate --dataset {dataset} --checkpoint_path path/to/checkpoint
```

Alternatively, the T5X framework can also be used for running training and inference with T5-family models. For more details, see [models/T5X.md](models/T5X.md).

## Evaluation

To evaluate the score of model predictions, run:

```bash
python -m mrl_eval.evaluation.evaluate --dataset {dataset} --predictions_path path/to/prediction/file
```

The options for `dataset` are the same as [above](#fine-tuning-and-inference).

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
| Hebrew                 | mT5-XL  | Hebco           | End-to-end / Gold Mentions Macro F1         | 49.3 / 75.8         |
| &nbsp;                 |         |                  |                  |                     |
| Modern Standard Arabic | mT5-XL  | ArQ-MSA-QA       | TLNLS            | 79.5                |
| Modern Standard Arabic | mT5-XL  | ArQ-MSA-QG       | R1/R2/RL         | 35.8 / 17.2 / 35.5  |
| Modern Standard Arabic | mT5-XL  | ArTyDi-QA        | TyDiQA-F1            | 87.4            |
| Modern Standard Arabic | mT5-XL  | ArTyDi-QG        | R1/R2/RL         | 60.6 / 44.1 / 60.5  |
| Modern Standard Arabic | mT5-XL  | IAHLT-NER        | Token F1         | 84.2                |
| Modern Standard Arabic | mT5-XL  | ArabicNLI        | Macro F1         |  82.2 |
| Modern Standard Arabic | mT5-XL  | ArXLSum          | R1/R2/RL         | 26.5 / 11.4 / 23.4  |
| Modern Standard Arabic | mT5-XL  | ASAS             | R1/R2/RL         | 39.5 / 21.3 / 27.8 |
| Modern Standard Arabic | mT5-XL  | MSA Sentiment          | Macro F1         | 61.0  |
| Modern Standard Arabic & Levantine Arabic | mT5-XL | Wojood-Full | Token F1 | 90.67 |
| Modern Standard Arabic | mT5-XL | Wojood-MSA | Token F1 | 91.62 |
| Modern Standard Arabic | mT5-XL | OntoNotes    | End-to-end / Gold Mentions Macro F1       |    50.9 / 88.7         |
| &nbsp;                 |         |                  |                  |                     |
| Levantine Arabic       | mT5-XL  | ArSentiment      | Macro F1         | 71.2                |
| Levantine Arabic       | mT5-XL  | ArQ-spoken-QA    | TLNLS            | 81.8                |
| Levantine Arabic       | mT5-XL  | ArQ-spoken-QG    | R1/R2/RL         | 35.6 / 16.6 / 35.3  |
| Levantine Arabic       | mT5-XL | Wojood-Spoken | Token F1 | 79.12 |
| Levantine Arabic       | mT5-XL | ShamNER | Token F1 | 42.0 |
| Levantine Arabic       | mT5-XL  | ArCoref          | End-to-end / Gold Mentions Macro F1         | 50.1  / 74.4               |

## Autoraters (LLM as a Judge)

This framework includes an evaluation module that leverages a Large Language
Model (LLM) for assessing the quality of question generation and summarization
tasks - offering evaluation beyond word matching metrics. The specific model
used for this evaluation is Gemini 2.5 Flash, accessed via the Vertex AI
platform on Google Cloud.

More details about autoraters can be found in
[evaluation/autoraters/README.md](evaluation/autoraters/README.md).

### Autorating for Question Generation

For question generation the Autorater evaluates **answerability** which is
the percentage of generated questions that can be accurately and directly
answered by the provided reference answer/passage.

The following command can be used to run autorater evaluation on question
generation datasets:

```
python -m mrl_eval.evaluation.autoraters.qg_autorater_main \
    --project_id={cloud-project-id} \
    --location={project-location} \
    --model_name="gemini-2.5-flash" \
    --dataset={question-generation-dataset} \
    --dataset_split=test \
    --predictions_path=path/to/prediction/file.jsonl \
    --output_path=path/to/output/file.jsonl
```

where question-generation-dataset can be one of:

  - arq_MSA_question_gen
  - arq_spoken_question_gen
  - artydiqa_question_gen
  - heq_question_gen

#### mT5-XL Autorater results

| Dataset | Answerability |
|---|---|
| arq_MSA_question_gen | 0.930 |
| arq_spoken_question | 0.884 |
| artydiqa_question_gen | 0.883 |
| heq_question_gen | 0.915 |

### Autorating for Summarisation

For summarisation, the autorater evaluates three dimensions:

- **Factual grounding**: all the
information in the summary should be grounded in the original article.
- **Relevance or Recall**: every single fact of the summary should be “salient” enough to appear
in the summary.
- **Quality**: the summary should be fluent and coherent. Each will be rated
on a scale of 1-5.

The following command can be used to run autorater evaluation on summarisation
datasets:

```
python -m mrl_eval.evaluation.autoraters.summarization_autoraters_main \
  --project_id={cloud-project-id} \
  --location={project-location} \
  --model_name="gemini-2.5-flash" \
  --dataset={summarisation-dataset} \
  --dataset_split="test" \
  --predictions_path=path/to/prediction/file.jsonl \
  --output_path=path/to/output/file.jsonl
```

where summarisation-dataset can be one of:

- hesum
- hebsummaries
- ar_xlsum
- asas

#### mT5-XL Autorater results

Dataset      | Factual Grounding | Recall | Fluency | Coherence
------------ | ----------------- | ------ | ------- | ---------
hesum        | 0.812             | 0.153  | 4.530   | 3.931
hebsummaries | 0.736             | 0.522  | 4.271   | 3.819
ar_xlsum     | 0.561             | 0.205  | 4.894   | 4.865
asas         | 0.913             | 0.546  | 3.973   | 3.560
