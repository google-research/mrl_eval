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

r"""Main file for running summarization autoraters."""

from collections.abc import Sequence
import pathlib

from absl import app
from absl import flags

from mrl_eval.evaluation.autoraters import autoraters_lib
from mrl_eval.evaluation.autoraters import executor_implementations
from mrl_eval.evaluation.autoraters import external_llm_implementations
from mrl_eval.evaluation.autoraters import factual_grounding_autorater_lib
from mrl_eval.evaluation.autoraters import fluency_coherence_autorater_lib
from mrl_eval.evaluation.autoraters import recall_autorater_lib
from mrl_eval.evaluation.autoraters import summarization_autoraters_lib
from mrl_eval.utils import io_utils


SummarizationInstance = summarization_autoraters_lib.SummarizationInstance


_PROJECT_ID = flags.DEFINE_string(
    'project_id',
    None,
    'Project ID to use for the Gemini model',
    required=True,
)

_LOCATION = flags.DEFINE_string(
    'location',
    None,
    'Location to use for the Gemini model',
    required=True,
)

_MODEL_NAME = flags.DEFINE_string(
    'model_name',
    'gemini-1.5-flash',
    'Name of the model to use',
)

_DATASET = flags.DEFINE_string(
    'dataset',
    'hebsummaries',
    'Dataset to run autoraters on',
)

_DATASET_SPLIT = flags.DEFINE_string(
    'dataset_split',
    'test',
    'Dataset split to run autoraters on',
)

_PREDICTIONS_PATH = flags.DEFINE_string(
    'predictions_path',
    None,
    'Path to the predictions',
)

_OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    None,
    'Path to the output jsonlines file.',
)

PREDICTION_COLUMN = 'prediction'


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  eval_instances = summarization_autoraters_lib.prepare_summarization_instances(
      _DATASET.value,
      _DATASET_SPLIT.value,
      _PREDICTIONS_PATH.value,
      PREDICTION_COLUMN,
  )
  # load model
  model = external_llm_implementations.Gemini(
      project_id=_PROJECT_ID.value,
      location=_LOCATION.value,
      model_name=_MODEL_NAME.value,
  )
  executor = executor_implementations.ThreadExecutor()
  autoraters: list[autoraters_lib.Autorater] = [
      factual_grounding_autorater_lib.FactualGroundingAutorater(
          model=model, executor=executor,
      ),
      recall_autorater_lib.RecallAutorater(
          model=model, executor=executor
      ),
      fluency_coherence_autorater_lib.QualityAutorater(
          model=model, executor=executor
      ),
  ]
  metric_list = summarization_autoraters_lib.run_autoraters(
      autoraters,
      eval_instances,
  )
  io_utils.write_jsonl(pathlib.Path(_OUTPUT_PATH.value), metric_list)

if __name__ == '__main__':
  app.run(main)
