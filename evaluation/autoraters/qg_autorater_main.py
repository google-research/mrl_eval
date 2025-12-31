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

r"""Main file for running the QG autorater.


Example command:

python3 -m mrl_eval.evaluation.autoraters.qg_autorater_main \
    --project_id=xtalk-gcp-xgcp \
    --location=us-central1 \
    --model_name=gemini-2.5-flash \
    --dataset=arq_MSA_question_gen \
    --dataset_split=test \
    --predictions_path=~/QG/predictions/arq_MSA_question_gen_gemma2-9b.jsonl \
    --output_path=~/QG/autoraters/arq_MSA_question_gen_gemma2-9b.jsonl
"""

from collections.abc import Sequence
import pathlib

from absl import app
from absl import flags

from mrl_eval.datasets import constants
from mrl_eval.evaluation.autoraters import executor_implementations
from mrl_eval.evaluation.autoraters import external_llm_implementations
from mrl_eval.evaluation.autoraters import qg_autorater_lib
from mrl_eval.utils import io_utils


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

_DATASET = flags.DEFINE_enum(
    'dataset',
    'artydiqa_question_gen',
    [
        constants.ARTYDIQA_QUESTION_GEN,
        constants.HEQ_QUESTION_GEN,
        constants.ARQ_MSA_QUESTION_GEN,
        constants.ARQ_SPOKEN_QUESTION_GEN,
    ],
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
    'Path jsonl file that contains the model-generated questions.',
    required=True,
)

_OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    None,
    'Path to the output jsonl file.',
    required=True,
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  external_llm = external_llm_implementations.Gemini(
      project_id=_PROJECT_ID.value,
      location=_LOCATION.value,
      model_name=_MODEL_NAME.value,
  )
  autorater = qg_autorater_lib.QGAutorater(
      model=external_llm, executor=executor_implementations.ThreadExecutor()
  )
  output_list = qg_autorater_lib.run_autorater(
      _DATASET.value,
      _DATASET_SPLIT.value,
      _PREDICTIONS_PATH.value,
      autorater,
  )
  output_path = pathlib.Path(_OUTPUT_PATH.value)
  io_utils.write_jsonl(output_path, [output_list])


if __name__ == '__main__':
  app.run(main)
