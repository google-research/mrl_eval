# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

from collections.abc import Sequence
import functools
import json
import pathlib

from absl import app
from absl import flags
import peft
import rich
import torch
import tqdm
import transformers

from mrl_eval.datasets import constants
from mrl_eval.hf.args import DataArguments
from mrl_eval.hf.args import ModelArguments
from mrl_eval.hf.args import TASKS_CONFIGS
from mrl_eval.hf.datasets.dataset_factory import hf_dataset_factory

Path = pathlib.Path  # pylint: disable=invalid-import-order


MIN_BATCH_SIZE = 4

_DATASET = flags.DEFINE_enum(
    "dataset",
    None,
    constants.DATASETS,
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


def _has_peft_config(model_name_or_path: str) -> bool:
  peft_config = Path(model_name_or_path) / peft.utils.CONFIG_NAME
  return peft_config.exists()


def _is_encoder_decoder(model_name_or_path: str) -> bool:
  if _has_peft_config(model_name_or_path):
    config = peft.PeftConfig.from_pretrained(model_name_or_path)
    base_model_name = config.base_model_name_or_path
    model_config = transformers.AutoConfig.from_pretrained(base_model_name)
  else:
    model_config = transformers.AutoConfig.from_pretrained(model_name_or_path)

  return model_config.is_encoder_decoder


def main(argv: Sequence[str]):
  print(argv)
  task = _DATASET.value

  config = {
      "model_name_or_path": _CKPT_PATH.value,
      "load_train": False,
      "load_validation": False,
      "load_test": True,
  }

  for key in ["max_inputs_length", "max_targets_length"]:
    config[key] = TASKS_CONFIGS[task][key]

  parser = transformers.HfArgumentParser((ModelArguments, DataArguments))
  model_args, data_args = parser.parse_dict(config)

  _print_args(model_args, data_args)

  tokenizer = transformers.AutoTokenizer.from_pretrained(
      model_args.model_name_or_path, use_fast=True
  )

  def collate_fn(batch, padding_side="right"):
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [ex["input_ids"] for ex in batch],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
        padding_side=padding_side,
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

  train_config_batch_size = TASKS_CONFIGS[task]["per_device_eval_batch_size"]
  batch_size = max(train_config_batch_size, MIN_BATCH_SIZE)

  if (is_encoder_decoder := _is_encoder_decoder(model_args.model_name_or_path)):

    dataset = hf_dataset_factory(
        _DATASET.value, data_args, tokenizer, for_decoder_only=False
    ).test_set()

    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        _CKPT_PATH.value
    ).to("cuda")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=functools.partial(collate_fn, padding_side="right"),
    )

    generation_config = transformers.GenerationConfig(
        max_new_tokens=data_args.max_targets_length,
        num_beams=4,
        do_sample=False,
    )

  else:
    model = transformers.AutoModelForCausalLM.from_pretrained(
        _CKPT_PATH.value
    ).to("cuda")

    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    dataset = hf_dataset_factory(
        _DATASET.value, data_args, tokenizer, for_decoder_only=True
    ).test_set()

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=functools.partial(collate_fn, padding_side="left"),
    )

    generation_config = transformers.GenerationConfig(
        max_new_tokens=data_args.max_targets_length,
    )

  model.eval()
  print(f"Model loaded onto {model.device}")

  ckpt_path = Path(_CKPT_PATH.value)
  output_dir = ckpt_path.parent / "generation"
  output_dir.mkdir(exist_ok=True, parents=True)
  output_file_path = output_dir / f"gen_{ckpt_path.name}.jsonl"

  with output_file_path.open("w") as f:

    for batch in tqdm.tqdm(dataloader):
      input_ids = batch["input_ids"].to("cuda")
      attention_mask = batch["attention_mask"].to("cuda")
      with torch.no_grad():

        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )

      if not is_encoder_decoder:  # discard input tokens
        input_length = input_ids.shape[1]
        output_ids = output_ids[:, input_length:]

      responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

      for i in range(len(batch["id"])):

        input_dict = {
            k: v[i]
            for k, v in batch.items()
            if k not in ["input_ids", "attention_mask", "labels"]
        }
        output = {"input": input_dict, "prediction": responses[i].strip()}
        f.write(json.dumps(output, ensure_ascii=False) + "\n")

  print(f"Generated responses saved to {output_file_path}")



if __name__ == "__main__":
  app.run(main)
