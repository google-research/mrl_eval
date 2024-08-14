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

"""Tasks for fine-tuning T5 using T5X."""

from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from immutabledict import immutabledict
import seqio
import tensorflow as tf
import tensorflow.compat.v2 as tf_c2

from mrl_eval.datasets import constants
from mrl_eval.datasets.hebnli import hebnli_lib
from mrl_eval.datasets.heq import heq_lib
from mrl_eval.datasets.hesentiment import hesentiment_lib
from mrl_eval.datasets.hesum import hesum_lib
from mrl_eval.datasets.nemo import nemo_lib


TaskRegistry = seqio.TaskRegistry
TaskProcessors = Sequence[
    Callable[[Dict[str, tf.Tensor]], Dict[str, tf.Tensor]]
]

_MODEL_MT5 = "mt5"
DEFAULT_SPM_PATH = "gs://t5-data/vocabs/mc4.250000.100extra/sentencepiece.model"

mt5_vocab = seqio.SentencePieceVocabulary(DEFAULT_SPM_PATH)
MT5_OUTPUT_FEATURES = immutabledict({
    "inputs": seqio.Feature(vocabulary=mt5_vocab, add_eos=True),
    "targets": seqio.Feature(vocabulary=mt5_vocab, add_eos=True),
})

DEFAULT_PREPROCESSORS = (
    seqio.preprocessors.tokenize,
    seqio.CacheDatasetPlaceholder(),
    seqio.preprocessors.append_eos_after_trim,
)


def postprocess_qa(
    answer, example = None, is_target = False
):
  """Returns answer, or all answers if the full example is provided."""
  if is_target:
    return [tf_c2.compat.as_text(a) for a in example["answers"]]
  return answer


@seqio.map_over_dataset
def convert_to_squad_format(example):
  """Converts example to the SQuAD format expected in squad preprocessing."""
  return {
      "id": example["id"],
      "title": example["title"],
      "context": example["context"],
      "question": example["question"],
      "answers": {
          "text": example["answers/text"],
          "answer_start": example["answers/answer_start"],
      },
  }


def get_tasks_values(
    input_key = "input", target_key = "target"
):
  """Returns a function that takes the relevant tasks values from example.

  Args:
    input_key: The input key. By default "inputs"
    target_key: The target key. By default "targets"

  Returns:
    A function that maps from the ex only the task's relevant values.
  """

  @seqio.map_over_dataset
  def fn(ex):
    return {
        "id": ex["id"],
        "inputs": ex[input_key],
        "targets": ex[target_key],
    }

  return fn


def _string_join(lst):
  """Joins on space, but collapse consecutive spaces."""
  out = tf.strings.join(lst, separator=" ")
  return tf.strings.regex_replace(out, r"\s+", " ")


@seqio.map_over_dataset
def preprocess_hebnli(example):
  """Convert HebNLI examples to a text2text pair.

  HebNLI produces examples with this form:
    {'id': <id>, 'translation1': <heb_sent_1>, 'translation2': <heb_sent_2>,
     'label_in_hebrew': <label>}
  This function will return examples of the format:
    {'inputs': 'משפט 1: <heb_sent_1> משפט 2: <heb_sent_2>',
     'targets': <heb_label>,
     'id': <id>},

  Args:
    example: an example to process.

  Returns:
    A preprocessed example with the format listed above.
  """
  sentence1 = example[hebnli_lib.HebNLI.HEB_FIRST_SENT_NAME]
  sentence2 = example[hebnli_lib.HebNLI.HEB_SECOND_SENT_NAME]
  inputs = _string_join(["משפט 1:", sentence1, "משפט 2:", sentence2])
  return {
      "id": example["id"],
      "inputs": inputs,
      "targets": example[hebnli_lib.HebNLI.HEB_LABEL_NAME],
  }


@seqio.map_over_dataset
def preprocess_qa(example):
  """Convert SQuAD examples to a text2text pair.

  SQuAD produces examples with this form:
    {'id': <id>, context': <article>, 'question': <question>,
     'answers': { 'text': [<n answers>] }}
  This function will return examples of the format:
    {'inputs': 'question: <question> context: <article>',
     'targets': '<answer_0>',
     'id': <id>, 'question': <question>, 'context': <context>,
     'answers': [<n answers>]},

  Args:
    example: an example to process.

  Returns:
    A preprocessed example with the format listed above.
  """
  answers = example["answers"]["text"]
  question = example["question"]
  context = example["context"]
  inputs = _string_join(["שאלה:", question, "הקשר:", context])
  return {
      "inputs": inputs,
      "targets": answers[0],
      "id": example["id"],
      "context": context,
      "question": question,
      "answers": answers,
  }


@seqio.map_over_dataset
def preprocess_question_generation(
    example,
):
  """Convert SQuAD examples to a text2text pair.

  Following: https://arxiv.org/abs/2011.11928
  SQuAD produces examples with this form:
    {'id': <id>, context': <article>, 'question': <question>,
     'answers': { 'text': [<n answers>] }}
  This function will return examples of the format:
    {'inputs': 'answer: <answer_0> context: <article>',
     'targets': '<question>',
     'id': <id>, 'question': <question>, 'context': <context>,
     'answers': [<n answers>]},

  Args:
    example: an example to process.

  Returns:
    A preprocessed example with the format listed above.
  """
  answers = example["answers"]["text"]
  question = example["question"]
  context = example["context"]
  inputs = _string_join(["תשובה:", answers[0], "הקשר:", context])
  return {
      "inputs": inputs,
      "targets": question,
      "id": example["id"],
      "context": context,
      "question": question,
      "answers": answers,
  }


def _register_heq(
    model_name,
    task_name,
    processor_func,
    postprocessor_func,
):
  """Register Heq."""
  if task_name == constants.HEQ:
    dataset = heq_lib.HeQ()
  elif task_name == constants.HEQ_QUESTION_GEN:
    dataset = heq_lib.HeQQuestionGen()
  else:
    raise ValueError(f"Unknown task name: {task_name}")

  if model_name:
    task_name = f"{task_name}_{model_name}"

  else:
    raise ValueError(f"Unknown task name: {task_name}")
  TaskRegistry.add(
      name=task_name,
      source=seqio.TFExampleDataSource(
          split_to_filepattern={
              "train": str(dataset.tfrecord_out_path("train")),
              "validation": str(dataset.tfrecord_out_path("val")),
              "test": str(dataset.tfrecord_out_path("test")),
          },
          feature_description=dataset.name_to_features(),
          reader_cls=lambda f: tf.data.TFRecordDataset([f]),
      ),
      output_features=MT5_OUTPUT_FEATURES,
      preprocessors=[
          convert_to_squad_format,
          processor_func,
          *DEFAULT_PREPROCESSORS,
      ],
      postprocess_fn=postprocessor_func,
      metric_fns=dataset.metrics,
  )


def register_heq(model_name):
  """Register Heq."""

  _register_heq(
      model_name,
      constants.HEQ,
      preprocess_qa,
      postprocess_qa,
  )
  _register_heq(
      model_name,
      constants.HEQ_QUESTION_GEN,
      preprocess_question_generation,
      None,
  )


def register_nemo(model_name):
  """Register Nemo."""
  for level in ["token", "morph"]:
    _register_nemo(model_name, level)


def _register_nemo(model_name, level):
  """Register Nemo task for the different dataset formulations."""
  targets_feature_name = f"targets_as_entity_markers_{level}_level"
  task_name = f"{constants.NEMO}_{targets_feature_name}_{model_name}"

  dataset = nemo_lib.Nemo()
  TaskRegistry.add(
      name=task_name,
      source=seqio.TFExampleDataSource(
          split_to_filepattern={
              "train": str(dataset.tfrecord_out_path("train")),
              "validation": str(dataset.tfrecord_out_path("val")),
              "test": str(dataset.tfrecord_out_path("test")),
          },
          feature_description=dataset.name_to_features(),
          reader_cls=lambda f: tf.data.TFRecordDataset([f]),
      ),
      output_features=MT5_OUTPUT_FEATURES,
      preprocessors=[
          get_tasks_values(
              "inputs", f"targets_as_entity_markers_{level}_level"
          ),
          *DEFAULT_PREPROCESSORS,
      ],
      metric_fns=dataset.metrics,
  )


def register_hebnli(model_name):
  """Register Hebnli."""
  task_name = constants.HEBNLI
  if model_name:
    task_name = f"{task_name}_{model_name}"

  dataset = hebnli_lib.HebNLI()
  TaskRegistry.add(
      name=task_name,
      source=seqio.TFExampleDataSource(
          split_to_filepattern={
              "train": str(dataset.tfrecord_out_path("train")),
              "validation": str(dataset.tfrecord_out_path("val")),
              "test": str(dataset.tfrecord_out_path("test")),
          },
          feature_description=dataset.name_to_features(),
          reader_cls=lambda f: tf.data.TFRecordDataset([f]),
      ),
      output_features=MT5_OUTPUT_FEATURES,
      preprocessors=[preprocess_hebnli, *DEFAULT_PREPROCESSORS],
      metric_fns=dataset.metrics,
  )


def register_hesentiment(model_name):
  """Register he_sentiment."""
  task_name = constants.HESENTIMENT
  if model_name:
    task_name = f"{task_name}_{model_name}"

  dataset = hesentiment_lib.HeSentiment()
  TaskRegistry.add(
      name=task_name,
      source=seqio.TFExampleDataSource(
          split_to_filepattern={
              "train": str(dataset.tfrecord_out_path("train")),
              "validation": str(dataset.tfrecord_out_path("val")),
              "test": str(dataset.tfrecord_out_path("test")),
          },
          feature_description=dataset.name_to_features(),
          reader_cls=lambda f: tf.data.TFRecordDataset([f]),
      ),
      output_features=MT5_OUTPUT_FEATURES,
      preprocessors=[
          get_tasks_values(dataset.TEXT_NAME, dataset.HEBREW_LABEL_NAME),
          *DEFAULT_PREPROCESSORS,
      ],
      metric_fns=dataset.metrics,
  )


def register_hesum(model_name):
  """Register he_sentiment."""
  task_name = constants.HESUM
  if model_name:
    task_name = f"{task_name}_{model_name}"

  dataset = hesum_lib.HeSum()
  TaskRegistry.add(
      name=task_name,
      source=seqio.TFExampleDataSource(
          split_to_filepattern={
              "train": str(dataset.tfrecord_out_path("train")),
              "validation": str(dataset.tfrecord_out_path("val")),
              "test": str(dataset.tfrecord_out_path("test")),
          },
          feature_description=dataset.name_to_features(),
          reader_cls=lambda f: tf.data.TFRecordDataset([f]),
      ),
      output_features=MT5_OUTPUT_FEATURES,
      preprocessors=[
          get_tasks_values(dataset.ARTICLE, dataset.SUMMARY),
          *DEFAULT_PREPROCESSORS,
      ],
      metric_fns=dataset.metrics,
  )


# Register all tasks variants
ALL_TASKS = [
    register_heq,
    register_nemo,
    register_hebnli,
    register_hesentiment,
    register_hesum,
]
for register_task in ALL_TASKS:
  register_task(_MODEL_MT5)
