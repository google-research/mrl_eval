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

"""Generate test answers for a dataset given a finetuned a model."""

import json
import os
import pathlib
from typing import Sequence

from absl import app
from absl import flags
import rich
import torch
import tqdm
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

_CKPT_PATH = flags.DEFINE_string(
    "checkpoint_path",
    "",
    "The path to model checkpoint to use for generation.",
)


def _print_args(model_args, data_args):
  print("=" * 100)
  print("Model args:")
  rich.print(model_args)
  print("=" * 100)
  print("Data args:")
  rich.print(data_args)


def main(argv: Sequence[str]):
  print(argv)
  task = _DATASET.value

  generation_config = {
      "model_name_or_path": _CKPT_PATH.value,
      "load_train": False,
      "load_validation": False,
      "load_test": True,
  }

  for key in ["max_inputs_length", "max_targets_length"]:
    generation_config[key] = TASKS_CONFIGS[task][key]

  parser = transformers.HfArgumentParser((ModelArguments, DataArguments))
  model_args, data_args = parser.parse_dict(generation_config)

  _print_args(model_args, data_args)

  tokenizer = transformers.AutoTokenizer.from_pretrained(
      model_args.model_name_or_path, use_fast=True
  )

  dataset = hf_dataset_factory(_DATASET.value, data_args, tokenizer).test_set()

  model = transformers.AutoModel.from_pretrained(
      _CKPT_PATH.value
  ).to("cuda")
  model.eval()
  print(f"Model loaded onto {model.device}")

  ckpt_path = pathlib.Path(_CKPT_PATH.value)
  output_dir = ckpt_path.parent / "generation"
  os.makedirs(output_dir, exist_ok=True)
  output_file_path = output_dir / f"gen_{ckpt_path.name}.jsonl"

  batch_size = 16 if _DATASET.value != constants.HESUM else 8

  def collate_fn(batch):
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [ex["input_ids"] for ex in batch],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    attention_mask = (
        input_ids != tokenizer.pad_token_id
    ).int()  # Create attention mask
    return {
        "id": [ex["id"] for ex in batch],
        "inputs": [ex["inputs"] for ex in batch],
        "targets": [ex["targets"] for ex in batch],
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

  dataloader = torch.utils.data.DataLoader(
      dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
  )

  with open(output_file_path, "w") as f:

    for batch in tqdm.tqdm(dataloader):
      input_ids = batch["input_ids"].to("cuda")
      attention_mask = batch["attention_mask"].to("cuda")
      with torch.no_grad():

        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            generation_config=transformers.GenerationConfig(
                max_new_tokens=data_args.max_targets_length,
                num_beams=4,
                do_sample=False,
            ),
        )

      responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

      for i in range(len(batch["id"])):

        input_dict = {
            k: v[i]
            for k, v in batch.items()
            if k not in ["input_ids", "attention_mask", "labels"]
        }
        output = {"input": input_dict, "prediction": responses[i]}
        f.write(json.dumps(output, ensure_ascii=False) + "\n")

  print(f"Generated responses saved to {output_file_path}")


if __name__ == "__main__":
  app.run(main)
