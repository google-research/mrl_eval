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

"""Library for summarization autoraters."""

import dataclasses
from typing import Any, Sequence

from absl import logging
import pandas as pd

from mrl_eval.datasets import dataset_factory
from mrl_eval.datasets import dataset_lib
from mrl_eval.evaluation.autoraters import autoraters_lib
from mrl_eval.utils import io_utils


@dataclasses.dataclass(frozen=True)
class SummarizationInstance:
  """Represents an instance to evaluate.

  Attributes:
    id: The id of the instance.
    article: The input article.
    reference_summary: The reference human-written summary.
    model_summary: The model summary.
  """
  id: str
  article: str
  reference_summary: str
  model_summary: str


def prepare_summarization_instances(
    dataset,
    dataset_split,
    predictions_path,
    prediction_column,
):
  """Prepares a sequence of SummarizationInstance from dataset and predictions.

  Args:
    dataset: The name of the dataset.
    dataset_split: The split of the dataset to use (e.g., 'test').
    predictions_path: The path to the JSONL file containing model predictions.
    prediction_column: The name of the column in the predictions file that
      contains the model's output.

  Returns:
    A sequence of SummarizationInstance.

  Raises:
    ValueError: If the dataset is not a summarization dataset or if the number
      of examples in the dataset and predictions are different.
  """
  # load gold data
  dataset_instance = dataset_factory.dataset_factory(dataset)

  if not isinstance(dataset_instance, dataset_lib.SummarizationDataset):
    raise ValueError(
        f'Dataset {dataset_instance} is not a summarization dataset.'
    )

  dataset = io_utils.read_jsonl(dataset_instance.jsonl_out_path(dataset_split))
  df_dataset = pd.DataFrame(dataset)

  # load predictions
  predictions = io_utils.read_jsonl(predictions_path)
  df_predictions = pd.DataFrame(predictions)
  if len(df_dataset) != len(df_predictions):
    raise ValueError(
        'Dataset and predictions have different number of examples:'
        f' {len(df_dataset)} examples in dataset,'
        f' {len(df_predictions)} predictions.'
    )
  df_predictions['id'] = df_predictions['input'].apply(lambda x: x['id'])
  df = pd.merge(
      df_dataset,
      df_predictions[['id', prediction_column]],
      on='id',
      how='inner',
  )
  eval_instances: list[SummarizationInstance] = []
  for _, row in df.iterrows():
    eval_instances.append(
        SummarizationInstance(
            id=row['id'],
            article=row[dataset_instance.article],
            reference_summary=row[dataset_instance.summary],
            model_summary=row[prediction_column],
        )
    )
  return eval_instances


def run_autoraters(
    autoraters,
    eval_instances,
):
  """Runs autoraters on a dataset of predictions.

  Args:
    autoraters: A sequence of autoraters to run.
    eval_instances: A sequence of summarization instances to evaluate.

  Returns:
    A list of dictionaries containing the autorater outputs. Each dictionary
    contains the autorater name, score, and raw eval outputs.
  """
  metric_list = []
  for autorater in autoraters:
    logging.info('Evaluating %s', autorater.name)
    autorater_outputs = autorater.evaluate_parallel(eval_instances)
    score = autorater.score(autorater_outputs)
    logging.info('Score: %s', score)
    metric_list.append({
        'name': autorater.name,
        'score': score,
        'raw_eval': [dataclasses.asdict(res) for res in autorater_outputs],
    })
  return metric_list
