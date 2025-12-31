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

"""Library for QG autoraters."""

from collections.abc import Hashable
import dataclasses
import logging
import pathlib
import textwrap
from typing import Sequence, Union, Any

import pandas as pd

from mrl_eval.datasets import dataset_factory
from mrl_eval.evaluation.autoraters import autoraters_lib
from mrl_eval.utils import io_utils


@dataclasses.dataclass(frozen=True)
class QAInstance:
  """Instance for the QA autorater.

  Attributes:
    id: The id of the instance. This is used to identify the instance in the
      output.
    question: The question to be answered.
    answer: The answer to the question.
    context: The context to be used for answering the question.
  """

  id: Hashable
  question: str
  answer: str
  context: str


@dataclasses.dataclass(frozen=True)
class QAResponse:
  """Response from the QA autorater.

  Attributes:
    correct_answer: Whether the answer is correct for the model-generated
      question. None if the autorater was unable to assign a yes or no. This
      will correspond to scenarios where the model-generated question is not
      valid, assuming the autorater does not have any hallucinations.
  """

  correct_answer: bool | None


def _may_cast_to_int(id_str):
  try:
    return int(id_str)
  except ValueError:
    return id_str


def _prepare_qg_prediction_df(pred_df):
  """Prepares the prediction dataframe for the GG autorater."""

  pred_df['id'] = pred_df['input'].apply(
      lambda input_dict: _may_cast_to_int(input_dict['id'])
  )
  pred_df['generated_question'] = pred_df['prediction']
  pred_df = pred_df.drop(columns=['input', 'prediction'])
  return pred_df


def _prepare_qg_ground_truth_df(ground_truth_df):
  """Prepares the ground truth dataframe for the GG autorater."""

  ground_truth_df['id'] = ground_truth_df['id'].apply(_may_cast_to_int)
  ground_truth_df['answer'] = ground_truth_df['answers'].apply(
      lambda answers: answers['text'][0]
  )
  ground_truth_df = ground_truth_df.drop(
      columns=['answers', 'title', 'question']
  )
  return ground_truth_df


def _load_qg_dataset(
    ground_truth_df, pred_df
):
  """Loads the QG dataset from the ground truth and prediction dataframes.

  Args:
    ground_truth_df: The ground truth dataframe.
    pred_df: The prediction dataframe.

  Returns:
    A list of QAInstance objects.
  """
  prepared_pred_df = _prepare_qg_prediction_df(pred_df)
  prepared_ground_truth_df = _prepare_qg_ground_truth_df(ground_truth_df)
  merged_df = pd.merge(
      prepared_pred_df, prepared_ground_truth_df, on='id', how='inner'
  )
  instances = []
  for _, row in merged_df.iterrows():
    instances.append(
        QAInstance(
            id=row['id'],
            question=row['generated_question'],
            answer=row['answer'],
            context=row['context'],
        )
    )
  return instances


def _read_dataframe_from_jsonl(path):
  return pd.DataFrame(io_utils.read_jsonl(path))


class QGAutorater(autoraters_lib.Autorater[QAInstance, QAResponse, float]):
  """A zero-shot autorater for QA.

  This autorater evaluates whether a given answer is suitable or correct for a
  model-generated question, given a context. It operates in a zero-shot manner,
  meaning it does not rely on few-shot examples.
  """

  def __init__(
      self, model, executor
  ):
    super().__init__(model=model, executor=executor)
    self._template = textwrap.dedent("""\
        Given the following paragraph, question and answer, please write whether the answer is the correct answer.
        Article: {article}
        Question: {question}
        Answer: {answer}
        The output should be a line containing either 'yes' or 'no'""")

  def _parse_response(self, response):
    """Parses the response from the model and returns a QAResponse.

    Args:
      response: The response from the model.

    Returns:
      A QAResponse indicating whether the answer is correct, or None if the
      response could not be parsed.
    """
    normalized_response = response.strip().lower()
    # Using `startswith` to accommodate cases where the model adds extra text
    # after the answer, such as "yes  ", "yes extra text", etc.
    if normalized_response.startswith('yes'):
      return QAResponse(correct_answer=True)
    elif normalized_response.startswith('no'):
      return QAResponse(correct_answer=False)
    else:
      return QAResponse(correct_answer=None)

  def score(self, eval_outputs):
    """Calculates the accuracy of the answerability of the model.

    Args:
      eval_outputs: A sequence of QAResponse. `correct_answer` of None will be
        treated as False. This will penalize scenarios where the autorater
        unable to assign 'yes' or 'no' and it will be assumed it was an issue in
        the generated question.

    Returns:
      The accuracy of the answerability of the model.
    """
    if not eval_outputs:
      return 0.0
    return sum(
        1 for eval_output in eval_outputs if eval_output.correct_answer
    ) / len(eval_outputs)

  def evaluate(self, eval_instance):
    prompt = self._template.format(
        question=eval_instance.question,
        article=eval_instance.context,
        answer=eval_instance.answer,
    )
    response = self._model.generate(prompt)
    return self._parse_response(response)

  @property
  def name(self):
    return 'answerability'


def run_autorater(
    dataset_name,
    dataset_split,
    predictions_path,
    autorater,
):
  """Runs the autorater on the dataset and writes the results to a jsonl file.

  Args:
      dataset_name: The name of the dataset.
      dataset_split: The split of the dataset to use (e.g., 'test').
      predictions_path: The path to the JSONL file containing model predictions.
      autorater: The QGAutorater instance to use.

  Returns:
      A dictionary containing the input ids, the raw evaluation results, and the
      answerability score.
  """
  dataset_instance = dataset_factory.dataset_factory(dataset_name)
  ground_truth_df = _read_dataframe_from_jsonl(
      dataset_instance.jsonl_out_path(dataset_split)
  )
  pred_df = _read_dataframe_from_jsonl(predictions_path)
  logging.info('Number of samples in ground truth: %d', len(ground_truth_df))
  logging.info('Number of samples in predictions: %d', len(pred_df))
  samples = _load_qg_dataset(ground_truth_df, pred_df)
  result = autorater.evaluate_parallel(samples)

  output_list = {
      'input_ids': [sample.id for sample in samples],
      'raw_eval': [res.__dict__ for res in result],
      'answerability_score': autorater.score(result),
  }
  return output_list
