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

import dataclasses
import pathlib
import shutil
from typing import Any, Sequence, Mapping

from absl import app
from absl import flags
import numpy as np
import peft
import rich
from rich import progress
import torch
import transformers

from mrl_eval.datasets import constants
from mrl_eval.hf.args import DataArguments
from mrl_eval.hf.args import ModelArguments
from mrl_eval.hf.args import TASKS_CONFIGS
from mrl_eval.hf.datasets import hf_datasets_lib
from mrl_eval.hf.datasets.dataset_factory import hf_dataset_factory



@dataclasses.dataclass
class TrainConfig:
  model_args: ModelArguments
  data_args: DataArguments
  training_args: (
      transformers.TrainingArguments | transformers.Seq2SeqTrainingArguments
  )

# By convention, see PyTorch CrossEntropyLoss default ignore_index:
# https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
_CALLBACK_PADDING_VALUE = -100

_FAST_TOKENIZER = flags.DEFINE_bool(
    "fast_tokenizer",
    True,
    "Whether to load HF's fast tokenizer."
)


_DATASET = flags.DEFINE_enum(
    "dataset",
    None,
    constants.DATASETS,
    "The dataset you'd like to finetune on.",
)


_MODEL = flags.DEFINE_string(
    "model",
    "google/mt5-xl",
    "The HF model you'd like to finetune. Must be a valid HF model name or"
    " local path.",
)


_SEED = flags.DEFINE_integer(
    "seed",
    1234,
    "Random seed for the training run.",
)


class DecoderEvalAndSaveCallback(transformers.TrainerCallback):
  """Callback for evaluation and saving the best model, for decoder-only models.

  This is used for generation on the validation set, evaluation of the
  metrics and saving the best model, since HF's SFTTrainer doesn't support
  predict_with_generate at present.
  """

  def __init__(
      self,
      trainer: transformers.Trainer,
      dataset: hf_datasets_lib.HfDataset,
      output_dir: str | pathlib.Path,
      max_new_tokens: int,
      metric_for_best_model: str | None = None,
      metric_greater_is_better: bool = True,
      eval_batch_size: int = 64,
      val_sample_limit: int | None = None,
      num_print_samples: int | None = None,
  ):
    super().__init__()
    self.dataset = dataset
    self.trainer = trainer
    self.gen_config = transformers.GenerationConfig.from_pretrained(
        trainer.model.name_or_path, max_new_tokens=max_new_tokens  # pytype: disable=attribute-error
    )
    self.num_print_samples = num_print_samples
    self.metric_for_best_model = metric_for_best_model
    self.metric_greater_is_better = metric_greater_is_better
    self.output_dir = output_dir
    self.batch_size = eval_batch_size
    self.val_sample_limit = val_sample_limit

    self.best_checkpoint = None
    self.best_metric = None
    self.metrics_history = []

    self.validation_dataloader = torch.utils.data.DataLoader(
        self.dataset.validation_set()[:self.val_sample_limit],
        batch_size=self.batch_size,
        shuffle=False,
        collate_fn=self.collate_fn,
    )

  def collate_fn(self, batch: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [ex["input_ids"] for ex in batch],
        batch_first=True,
        padding_value=self.trainer.processing_class.eos_token_id,  # pytype: disable=attribute-error
        padding_side="left",
    )
    attention_mask = (
        input_ids != self.trainer.processing_class.eos_token_id  # pytype: disable=attribute-error
    ).int()
    return {
        "id": [ex["id"] for ex in batch],
        "inputs": [ex["inputs"] for ex in batch],
        "targets": [ex["targets"] for ex in batch],
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

  def generate_preds(self) -> tuple[list[str], list[list[int]]]:
    """Generates predictions for the validation set."""

    all_responses_str = []
    all_responses_tok_ids = []
    for batch in progress.track(
        self.validation_dataloader, "Generating eval predictions"
    ):
      input_ids = batch["input_ids"].to("cuda")
      attention_mask = batch["attention_mask"].to("cuda")

      with torch.inference_mode():
        output_ids = self.trainer.model.generate(  # pytype: disable=attribute-error
            input_ids,
            attention_mask=attention_mask,
            generation_config=self.gen_config,
            pad_token_id=self.trainer.processing_class.eos_token_id,  # pytype: disable=attribute-error
        )

      input_length = input_ids.shape[1]
      new_tokens = output_ids[:, input_length:]
      responses = self.trainer.processing_class.batch_decode(  # pytype: disable=attribute-error
          new_tokens, skip_special_tokens=True
      )
      all_responses_str.extend(responses)
      all_responses_tok_ids.extend(new_tokens.tolist())

    return all_responses_str, all_responses_tok_ids

  def _is_better_metric(self, metric_value: float) -> bool:
    if self.metric_greater_is_better:
      return metric_value > self.best_metric
    else:
      return metric_value < self.best_metric

  def _pad_ragged_sequences(
      self,
      sequences: list[list[int]],
      padding_value: int = _CALLBACK_PADDING_VALUE,
  ) -> list[list[int]]:
    """Right pads sequences to the same length."""
    max_length = max(len(s) for s in sequences)
    padded_sequences = []
    for s in sequences:
      padded_sequences.append(s + [padding_value] * (max_length - len(s)))
    return padded_sequences

  def _remove_checkpoint(self, path: pathlib.Path) -> None:


    shutil.rmtree(path)  # pylint: disable=unreachable

  def on_evaluate(
      self,
      args: transformers.TrainingArguments,
      state: transformers.TrainerState,
      control: transformers.TrainerControl,
      **kwargs,
  ) -> None:
    """Callback code to run when trainer is evaluating."""

    super().on_evaluate(args, state, control, **kwargs)

    self.trainer.model.eval()  # pytype: disable=attribute-error
    validation_set = self.dataset.validation_set()
    if self.val_sample_limit:
      validation_set = validation_set[: self.val_sample_limit]
    targets = [example["targets"] for example in validation_set]
    targets_tok_ids = self.trainer.processing_class.batch_encode_plus(targets)[  # pytype: disable=attribute-error
        "input_ids"
    ]
    preds_str, preds_tok_ids = self.generate_preds()
    eval_preds = transformers.EvalPrediction(
        predictions=np.array(self._pad_ragged_sequences(preds_tok_ids)),
        label_ids=np.array(self._pad_ragged_sequences(targets_tok_ids)),
    )
    scores = self.dataset.compute_metrics(eval_preds)

    if self.num_print_samples:
      for i in range(self.num_print_samples):
        rich.print(f"[bold cyan]Sample {i}[bold cyan]")
        rich.print(
            f"[bold magenta]Input:[/bold magenta] {validation_set[i]['inputs']}"
        )
        rich.print(
            "[bold magenta]Target:[/bold magenta]"
            f" {validation_set[i]['targets']}"
        )
        rich.print(f"[bold magenta]Prediction:[/bold magenta] {preds_str[i]}")
        rich.print(
            "[bold magenta]Prediction token ids:[/bold magenta]"
            f" {preds_tok_ids[i]}"
        )

    rich.print(
        f"Validation scores at step {self.trainer.state.global_step}: {scores}"  # pytype: disable=attribute-error
    )

    if self.metric_for_best_model:
      decision_metric = scores[self.metric_for_best_model]
      self.metrics_history.append(round(decision_metric, 1))
      is_first_checkpoint = not self.best_checkpoint
      if is_first_checkpoint or self._is_better_metric(decision_metric):
        if not is_first_checkpoint:
          # remove the old best checkpoint
          path = pathlib.Path(self.output_dir) / str(self.best_checkpoint)
          print(f"Removing old best checkpoint at {path}")
          self._remove_checkpoint(path)

        self.best_checkpoint = self.trainer.state.global_step  # pytype: disable=attribute-error
        self.best_metric = decision_metric
        # save the best checkpoint
        save_path = pathlib.Path(self.output_dir) / str(self.best_checkpoint)
        print(f"Saving new best checkpoint at {save_path}")
        self.trainer.model.save_pretrained(save_path)  # pytype: disable=attribute-error
        self.trainer.processing_class.save_pretrained(save_path)  # pytype: disable=attribute-error # pylint: disable=unreachable


      print(
          f"{self.metric_for_best_model} metric history:"
          f" {self.metrics_history}"
      )



def _get_train_config_encoder_decoder(
    task: str, model: str, output_dir: str
) -> TrainConfig:
  """Returns the train config for encoder-decoder models."""
  train_config = {
      "model_name_or_path": model,
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
  return TrainConfig(model_args, data_args, training_args)


def _get_train_config_decoder_only(
    task: str, model: str, output_dir: str
) -> TrainConfig:
  """Returns the train config for decoder-only models."""
  train_config = {
      "model_name_or_path": model,
      "output_dir": output_dir,
      "do_train": True,
      "num_train_epochs": 5,
      "per_device_train_batch_size": 32,
      "eval_strategy": "steps",
      "learning_rate": 1e-5,
      "eval_steps": 500,
      "save_total_limit": 1,
      "save_strategy": "no",
      "lr_scheduler_type": "linear",
      "warmup_ratio": 0.1,
      "auto_find_batch_size": True,
      "logging_steps": 20,
      "logging_strategy": "steps",  # Frequency of logging to file
      "report_to": "none",
      "logging_dir": f"{output_dir}/logs",
      "gradient_accumulation_steps": 1,
      "optim": "adamw_torch",
      "bf16": True,
      "bf16_full_eval": True,
  }

  for key in TASKS_CONFIGS[task]:
    if key not in ["max_sequence_length", "generation_max_length"]:
      train_config[key] = TASKS_CONFIGS[task][key]

  parser = transformers.HfArgumentParser(
      (ModelArguments, DataArguments, transformers.TrainingArguments)
  )
  model_args, data_args, training_args = parser.parse_dict(train_config)
  return TrainConfig(model_args, data_args, training_args)


def _print_args(train_config: TrainConfig):
  print("=" * 100)
  print("Model args:")
  rich.print(train_config.model_args)
  print("=" * 100)
  print("Data args:")
  rich.print(train_config.data_args)
  print("=" * 100)
  print("Training args:")
  rich.print(train_config.training_args)


def _print_example(dataset):
  print("=" * 50)
  print("First training sample:")
  sample = dataset.train_set()[0]
  for key in sample.keys():
    rich.print(f"{key}: {sample[key]}")


def finetune_encoder_decoder(train_config: TrainConfig) -> None:
  """Finetune an encoder-decoder model."""

  model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
      train_config.model_args.model_name_or_path
  )

  tokenizer = transformers.AutoTokenizer.from_pretrained(
      train_config.model_args.model_name_or_path, use_fast=_FAST_TOKENIZER.value
  )

  dataset = hf_dataset_factory(
      _DATASET.value, train_config.data_args, tokenizer, for_decoder_only=False
  )

  _print_example(dataset)

  trainer = transformers.Seq2SeqTrainer(
      model=model,
      args=train_config.training_args,
      train_dataset=dataset.train_set(),
      eval_dataset=dataset.validation_set(),
      tokenizer=tokenizer,
      data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, model=model),
      compute_metrics=(
          dataset.compute_metrics
          if train_config.training_args.predict_with_generate  # pytype: disable=attribute-error
          else None
      ),
  )
  print("Starting train")
  trainer.train()
  print("Training done")
  print(
      f"Best checkpoint is saved at {trainer.state.best_model_checkpoint} with"
      f" a {train_config.training_args.metric_for_best_model} validation score"
      f" of {trainer.state.best_metric}"
  )


def finetune_decoder_only(
    train_config: TrainConfig, model_type: str
) -> None:
  """Finetune a decoder-only model."""

  print("Detected decoder-only model, training with LORA.")

  tokenizer = transformers.AutoTokenizer.from_pretrained(
      train_config.model_args.model_name_or_path, use_fast=_FAST_TOKENIZER.value
  )

  tokenizer.padding_side = "right"
  if tokenizer.pad_token is None:
    if tokenizer.bos_token is not None:
      tokenizer.pad_token = tokenizer.bos_token
    elif tokenizer.unk_token is not None:
      tokenizer.pad_token = tokenizer.unk_token
    else:
      raise ValueError(
          "No pad token found in tokenizer and no BOS or UNK tokens"
      )

  dataset = hf_dataset_factory(
      _DATASET.value, train_config.data_args, tokenizer, for_decoder_only=True
  )

  metric_for_best_model = train_config.training_args.metric_for_best_model
  train_config.training_args.metric_for_best_model = None

  _print_example(dataset)

  if model_type == "gemma2":
    attn_implementation = "eager"
  else:
    attn_implementation = "flash_attention_2"


  print(f"Training with attention implementation: {attn_implementation}")

  model = transformers.AutoModelForCausalLM.from_pretrained(
      train_config.model_args.model_name_or_path,
      attn_implementation=attn_implementation,
      torch_dtype=torch.bfloat16,
  )

  lora_config = peft.LoraConfig(
      r=256,
      lora_alpha=512,
      bias="none",
      target_modules="all-linear",
      task_type="CAUSAL_LM",
  )

  model = peft.get_peft_model(model, lora_config)
  model.print_trainable_parameters()

  trainer = transformers.Trainer(
      model=model,
      args=train_config.training_args,
      train_dataset=dataset.train_set(),
      eval_dataset=dataset.validation_set(),
      tokenizer=tokenizer,
      data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, model=model),
  )

  max_new_tokens = TASKS_CONFIGS[_DATASET.value]["generation_max_length"]

  eval_callback = DecoderEvalAndSaveCallback(
      trainer=trainer,
      dataset=dataset,
      output_dir=train_config.training_args.output_dir,
      max_new_tokens=max_new_tokens,
      metric_for_best_model=metric_for_best_model,
      num_print_samples=10,
      val_sample_limit=None,
      metric_greater_is_better=True,
      eval_batch_size=train_config.training_args.per_device_eval_batch_size,
  )
  trainer.add_callback(eval_callback)

  print("Starting train")
  trainer.train()
  print("Training done")

  checkpoint_path = pathlib.Path(eval_callback.output_dir) / str(
      eval_callback.best_checkpoint
  )

  rich.print(
      f"Best checkpoint is saved at {checkpoint_path} with a"
      f" {eval_callback.metric_for_best_model} validation score of"
      f" {eval_callback.best_metric}"
  )


def main(argv: Sequence[str]):
  print(argv)
  task = _DATASET.value
  model = _MODEL.value
  if model.endswith("/"):
    model = model[:-1]
  model_name = model.split("/")[-1]

  output_dir = f"output/{model_name}_{task}"


  transformers.set_seed(_SEED.value)

  model_config = transformers.AutoConfig.from_pretrained(model)

  if model_config.is_encoder_decoder:
    train_config = _get_train_config_encoder_decoder(
        task, model, output_dir
    )
    _print_args(train_config)
    finetune_encoder_decoder(train_config)

  else:
    train_config = _get_train_config_decoder_only(
        task, model, output_dir
    )
    _print_args(train_config)
    finetune_decoder_only(
        train_config, model_type=model_config.model_type
    )


if __name__ == "__main__":
  app.run(main)
