# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Finetune a model on a dataset."""

from typing import Sequence

from absl import app
from absl import flags
import rich
import transformers

from mrl_eval.datasets import constants
from mrl_eval.hf.args import DataArguments
from mrl_eval.hf.args import ModelArguments
from mrl_eval.hf.args import TASKS_CONFIGS
from mrl_eval.hf.datasets.dataset_factory import hf_dataset_factory


_DATASET = flags.DEFINE_enum(
    "dataset",
    None,
    [
        constants.HEQ,
        constants.HEQ_QUESTION_GEN,
        constants.NEMO_TOKEN,
        constants.NEMO_MORPH,
        constants.HEBNLI,
        constants.HESENTIMENT,
        constants.HESUM,
    ],
    "The dataset you'd like to finetune on.",
)


def _print_args(model_args, data_args, training_args):
  print("=" * 100)
  print("Model args:")
  rich.print(model_args)
  print("=" * 100)
  print("Data args:")
  rich.print(data_args)
  print("=" * 100)
  print("Training args:")
  rich.print(training_args)


def _print_example(dataset):
  print("=" * 50)
  print("First training sample:")
  sample = dataset.train_set()[0]
  for key in sample.keys():
    print(f"{key}: {sample[key]}")


def main(argv: Sequence[str]):
  print(argv)
  task = _DATASET.value
  model = "mt5-xl"
  output_dir = f"output/{model}_{task}"
  train_config = {
      "model_name_or_path": f"google/{model}",
      "output_dir": output_dir,
      "do_train": True,
      "max_steps": 8000,
      "per_device_train_batch_size": 64,
      "eval_strategy": "steps",
      "optim": "adamw_torch",
      "learning_rate": 1e-5,
      "lr_scheduler_type": "linear",
      "warmup_ratio": 0.1,
      "eval_steps": 500,
      "save_total_limit": 1,
      "save_strategy": "steps",
      "save_steps": 500,
      "save_safetensors": False,
      "auto_find_batch_size": True,
      "bf16": True,
      "bf16_full_eval": True,
      "load_best_model_at_end": True,
      "predict_with_generate": True,
      "logging_steps": 20,
      "logging_strategy": "steps",  # Frequency of logging to file
      "report_to": "tensorboard",
      "logging_dir": f"{output_dir}/logs",
      "gradient_accumulation_steps": 4,
      "save_only_model": True,
  }

  for key in TASKS_CONFIGS[task]:
    train_config[key] = TASKS_CONFIGS[task][key]

  parser = transformers.HfArgumentParser(
      (ModelArguments, DataArguments, transformers.Seq2SeqTrainingArguments)
  )
  model_args, data_args, training_args = parser.parse_dict(train_config)

  _print_args(model_args, data_args, training_args)

  tokenizer = transformers.AutoTokenizer.from_pretrained(
      model_args.model_name_or_path, use_fast=True
  )

  dataset = hf_dataset_factory(_DATASET.value, data_args, tokenizer)
  _print_example(dataset)

  model = transformers.MT5ForConditionalGeneration.from_pretrained(
      model_args.model_name_or_path
  )

  trainer = transformers.Seq2SeqTrainer(
      model=model,
      args=training_args,
      train_dataset=dataset.train_set(),
      eval_dataset=dataset.validation_set(),
      tokenizer=tokenizer,
      data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, model=model),
      compute_metrics=(
          dataset.compute_metrics
          if training_args.predict_with_generate
          else None
      ),
  )
  print("Starting train")
  trainer.train()
  print("Training done")
  print(
      f"Best checkpoint is saved at {trainer.state.best_model_checkpoint} with"
      f" a {training_args.metric_for_best_model} validation score of"
      f" {trainer.state.best_metric}"
  )


if __name__ == "__main__":
  app.run(main)
