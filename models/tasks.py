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

"""Tasks for fine-tuning T5 using T5X."""

from collections.abc import Callable, Mapping, Sequence
import functools
from typing import Any

from immutabledict import immutabledict
import seqio
import tensorflow as tf
import tensorflow.compat.v2 as tf_c2

from mrl_eval.datasets import constants
from mrl_eval.datasets.ar_xlsum import ar_xlsum_lib
from mrl_eval.datasets.arabic_nli import arabic_nli_lib
from mrl_eval.datasets.arcoref import arcoref_lib
from mrl_eval.datasets.arq import arq_lib
from mrl_eval.datasets.arsentiment import arsentiment_lib
from mrl_eval.datasets.artydiqa import artydiqa_lib
from mrl_eval.datasets.hebco import hebco_lib
from mrl_eval.datasets.hebnli import hebnli_lib
from mrl_eval.datasets.hebsummaries import hebsummaries_lib
from mrl_eval.datasets.heq import heq_lib
from mrl_eval.datasets.hesentiment import hesentiment_lib
from mrl_eval.datasets.hesum import hesum_lib
from mrl_eval.datasets.iahlt_ner import iahlt_ner_lib
from mrl_eval.datasets.nemo import nemo_lib


TaskRegistry = seqio.TaskRegistry
TaskProcessors = Sequence[
    Callable[[dict[str, tf.Tensor]], dict[str, tf.Tensor]]
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

HE_ANSWER_PROMPT = "תשובה:"
HE_CONTEXT_PROMPT = "הקשר:"
HE_QUESTION_PROMPT = "שאלה:"
HE_FIRST_SENTENCE_PROMPT = "משפט 1:"
HE_SECOND_SENTENCE_PROMPT = "משפט 2:"
AR_ANSWER_PROMPT = "الجواب:"
AR_QUESTION_PROMPT = "السؤال:"
AR_CONTEXT_PROMPT = "النص:"
AR_PREMISE_PROMPT = "مقدمة:"
AR_HYPOTHESIS_PROMPT = "فرضية:"


def postprocess_qa(
    answer: Any, example: Any = None, is_target: bool = False
) -> Any:
  """Returns answer, or all answers if the full example is provided."""
  if is_target:
    return [tf_c2.compat.as_text(a) for a in example["answers"]]
  return answer


@seqio.map_over_dataset
def convert_to_squad_format(example: Mapping[str, Any]) -> Mapping[str, Any]:
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
    input_key: str = "input", target_key: str = "target"
) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
  """Returns a function that takes the relevant tasks values from example.

  Args:
    input_key: The input key. By default "inputs"
    target_key: The target key. By default "targets"

  Returns:
    A function that maps from the ex only the task's relevant values.
  """

  @seqio.map_over_dataset
  def fn(ex: Mapping[str, Any]) -> Mapping[str, Any]:
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
def preprocess_hebnli(example: Mapping[str, Any]) -> Mapping[str, Any]:
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
  inputs = _string_join([
      HE_FIRST_SENTENCE_PROMPT,
      sentence1,
      HE_SECOND_SENTENCE_PROMPT,
      sentence2,
  ])
  return {
      "id": example["id"],
      "inputs": inputs,
      "targets": example[hebnli_lib.HebNLI.HEB_LABEL_NAME],
  }


@seqio.map_over_dataset
def preprocess_arabic_nli(example: Mapping[str, Any]) -> Mapping[str, Any]:
  """Convert Arabic NLI examples to a text2text pair.

  Arabic NLI, derived from XNLI, produces examples with this form:
    {'premise': <arabic_sent_1>, 'hypothesis': <arabic_sent_2>,
     'label': <label[int]>}
  This function will return examples of the format:
    {'inputs': 'مقدمة: <arabic_sent_1> فرضية: <arabic_sent_2>',
     'targets': <arabic_label[str]>},

  Args:
    example: an example to process.

  Returns:
    A preprocessed example with the format listed above.
  """
  premise = example[arabic_nli_lib.PREMISE_KEY]
  hypothesis = example[arabic_nli_lib.HYPOTHESIS_KEY]
  inputs = _string_join(
      [AR_PREMISE_PROMPT, premise, AR_HYPOTHESIS_PROMPT, hypothesis]
  )

  return {
      "id": example["id"],
      "inputs": inputs,
      "targets": example[arabic_nli_lib.LABEL_KEY],
  }


@seqio.map_over_dataset
def preprocess_qa(
    example: Mapping[str, Any], question_prompt: str, context_prompt: str
) -> Mapping[str, Any]:
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
    question_prompt: Text to prepend to the question to indicate that it's a
      question.
    context_prompt: Text to prepend to the context to indicate that it's a
      context.

  Returns:
    A preprocessed example with the format listed above.
  """
  answers = example["answers"]["text"]
  question = example["question"]
  context = example["context"]
  inputs = _string_join([question_prompt, question, context_prompt, context])
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
    example: Mapping[str, Any],
    answer_prompt: str,
    context_prompt: str,
) -> Mapping[str, Any]:
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
    answer_prompt: Text to prepend to the answer to indicate that it's an
      answer.
    context_prompt: Text to prepend to the context to indicate that it's a
      context.

  Returns:
    A preprocessed example with the format listed above.
  """
  answers = example["answers"]["text"]
  question = example["question"]
  context = example["context"]
  inputs = _string_join([answer_prompt, answers[0], context_prompt, context])
  return {
      "inputs": inputs,
      "targets": question,
      "id": example["id"],
      "context": context,
      "question": question,
      "answers": answers,
  }


def _register_heq(
    model_name: str,
    task_name: str,
    processor_func: Callable[[Mapping[str, str]], Mapping[str, str]],
    postprocessor_func: Callable[[Mapping[str, str]], Mapping[str, str]] | None,
) -> None:
  """Register Heq."""
  if not model_name:
    raise ValueError("No model name provided")

  if task_name == constants.HEQ:
    dataset = heq_lib.HeQ()
  elif task_name == constants.HEQ_QUESTION_GEN:
    dataset = heq_lib.HeQQuestionGen()
  else:
    raise ValueError(f"Unknown task name: {task_name}")

  task_name = f"{task_name}_{model_name}"

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


def register_heq(model_name: str):
  """Register Heq."""

  _register_heq(
      model_name,
      constants.HEQ,
      functools.partial(
          preprocess_qa,
          question_prompt=HE_QUESTION_PROMPT,
          context_prompt=HE_CONTEXT_PROMPT,
      ),
      postprocess_qa,
  )
  _register_heq(
      model_name,
      constants.HEQ_QUESTION_GEN,
      functools.partial(
          preprocess_question_generation,
          answer_prompt=HE_ANSWER_PROMPT,
          context_prompt=HE_CONTEXT_PROMPT,
      ),
      None,
  )


def register_nemo(model_name: str):
  """Register Nemo."""
  for level in ["token", "morph"]:
    _register_nemo(model_name, level)


def _register_nemo(model_name: str, level: str) -> None:
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


def register_hebnli(model_name: str):
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


def register_arabic_nli(model_name: str):
  """Register Arabic NLI."""
  task_name = constants.ARABIC_NLI
  if model_name:
    task_name = f"{task_name}_{model_name}"

  dataset = arabic_nli_lib.ArabicNLI()
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
      preprocessors=[preprocess_arabic_nli, *DEFAULT_PREPROCESSORS],
      metric_fns=dataset.metrics,
  )


def register_hesentiment(model_name: str):
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


def register_hesum(model_name: str):
  """Register he_sum."""
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
          get_tasks_values(dataset.article, dataset.summary),
          *DEFAULT_PREPROCESSORS,
      ],
      metric_fns=dataset.metrics,
  )


def register_ar_xlsum(model_name: str):
  """Register ar_xlsum."""
  task_name = constants.AR_XLSUM
  if model_name:
    task_name = f"{task_name}_{model_name}"

  dataset = ar_xlsum_lib.ArXLSum()
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
          get_tasks_values(dataset.article, dataset.summary),
          *DEFAULT_PREPROCESSORS,
      ],
      metric_fns=dataset.metrics,
  )


def register_hebsummaries(model_name: str) -> None:
  """Register HebSummaries."""
  if model_name:
    task_name = f"{constants.HEBSUMMARIES}_{model_name}"
  else:
    task_name = constants.HEBSUMMARIES

  dataset = hebsummaries_lib.HebSummaries()
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
          get_tasks_values(dataset.article, dataset.summary),
          *DEFAULT_PREPROCESSORS,
      ],
      metric_fns=dataset.metrics,
  )


def _register_artydiqa(
    model_name: str,
    task_name: str,
    processor_func: Callable[[Mapping[str, str]], Mapping[str, str]],
    postprocessor_func: Callable[[Mapping[str, str]], Mapping[str, str]] | None,
) -> None:
  """Register ArTyDiQA."""
  if not model_name:
    raise ValueError("No model name provided")

  if task_name == constants.ARTYDIQA:
    dataset = artydiqa_lib.ArTyDiQA()
  elif task_name == constants.ARTYDIQA_QUESTION_GEN:
    dataset = artydiqa_lib.ArTyDiQAQuestionGen()
  else:
    raise ValueError(f"Unknown task name: {task_name}")

  task_name = f"{task_name}_{model_name}"

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


def register_artydiqa(model_name: str) -> None:
  """Register AraTyDiQA."""

  _register_artydiqa(
      model_name,
      constants.ARTYDIQA,
      functools.partial(
          preprocess_qa,
          question_prompt=AR_QUESTION_PROMPT,
          context_prompt=AR_CONTEXT_PROMPT,
      ),
      postprocess_qa,
  )

  _register_artydiqa(
      model_name,
      constants.ARTYDIQA_QUESTION_GEN,
      functools.partial(
          preprocess_question_generation,
          answer_prompt=AR_ANSWER_PROMPT,
          context_prompt=AR_CONTEXT_PROMPT,
      ),
      None,
  )


def register_arsentiment(model_name: str):
  """Register arsentiment."""
  task_name = constants.ARSENTIMENT
  if model_name:
    task_name = f"{task_name}_{model_name}"

  dataset = arsentiment_lib.ArSentiment()
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
          get_tasks_values(dataset.TEXT_KEY, dataset.LABEL_KEY),
          *DEFAULT_PREPROCESSORS,
      ],
      metric_fns=dataset.metrics,
  )


def _register_arq(
    model_name: str,
    task_name: str,
    processor_func: Callable[[Mapping[str, str]], Mapping[str, str]],
    postprocessor_func: Callable[[Mapping[str, str]], Mapping[str, str]] | None,
) -> None:
  """Register ArQ."""
  if not model_name:
    raise ValueError("No model name provided")

  match task_name:
    case constants.ARQ_SPOKEN:
      dataset = arq_lib.ArQ("spoken")
    case constants.ARQ_MSA:
      dataset = arq_lib.ArQ("MSA")
    case constants.ARQ_SPOKEN_QUESTION_GEN:
      dataset = arq_lib.ArQQuestionGen("spoken")
    case constants.ARQ_MSA_QUESTION_GEN:
      dataset = arq_lib.ArQQuestionGen("MSA")
    case _:
      raise ValueError(f"Unknown task name: {task_name}")

  task_name = f"{task_name}_{model_name}"

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


def register_arq(model_name: str) -> None:
  """Register ArQ."""

  for task in (constants.ARQ_SPOKEN, constants.ARQ_MSA):
    _register_arq(
        model_name,
        task,
        functools.partial(
            preprocess_qa,
            question_prompt=AR_QUESTION_PROMPT,
            context_prompt=AR_CONTEXT_PROMPT,
        ),
        postprocess_qa,
    )

  for task in (
      constants.ARQ_SPOKEN_QUESTION_GEN,
      constants.ARQ_MSA_QUESTION_GEN,
  ):
    _register_arq(
        model_name,
        task,
        functools.partial(
            preprocess_question_generation,
            answer_prompt=AR_ANSWER_PROMPT,
            context_prompt=AR_CONTEXT_PROMPT,
        ),
        None,
    )


def register_hebco(model_name: str):
  """Register HebCo."""
  task_name = constants.HEBCO
  if model_name:
    task_name = f"{task_name}_{model_name}"

  dataset = hebco_lib.Hebco(index_text=True, index_targets=True)
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
          get_tasks_values(dataset.TEXT_FIELD, dataset.TARGET_FIELD),
          *DEFAULT_PREPROCESSORS,
      ],
      metric_fns=dataset.metrics,
  )


def register_arcoref(model_name: str):
  """Register ArCoref."""
  task_name = constants.ARCOREF
  if model_name:
    task_name = f"{task_name}_{model_name}"

  dataset = arcoref_lib.ArCoref(index_text=True, index_targets=True)
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
          get_tasks_values(dataset.TEXT_FIELD, dataset.TARGET_FIELD),
          *DEFAULT_PREPROCESSORS,
      ],
      metric_fns=dataset.metrics,
  )


def register_iahlt_ner(model_name: str) -> None:
  """Register IAHLT NER."""
  task_name = f"{constants.IAHLT_NER}_{model_name}"

  dataset = iahlt_ner_lib.IahltNer()
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
          get_tasks_values(dataset.TEXT_KEY, dataset.LABEL_KEY),
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
    register_arsentiment,
    register_artydiqa,
    register_arq,
    register_hebco,
    register_arcoref,
    register_iahlt_ner,
    register_hebsummaries,
    register_arabic_nli,
    register_ar_xlsum,
]
for register_task in ALL_TASKS:
  register_task(_MODEL_MT5)
