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

"""Executor implementations for autoraters."""

from concurrent import futures
from typing import Callable, Sequence

import tenacity
import tqdm

from mrl_eval.evaluation.autoraters import autoraters_lib


def _before_sleep(retry_state):
  """Prints the error before sleeping for the next retry attempt."""
  exception = retry_state.outcome.exception()
  print(f'Sleeping due to error: {exception}')


class ThreadExecutor(autoraters_lib.Executor):
  """A naive executor that just returns the LLM response."""

  def execute(
      self,
      evaluate_fn,
      instances,
  ):
    result = {}

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=60),
        before_sleep=_before_sleep
    )
    def retried_evaluate(item):
      return evaluate_fn(item)

    with futures.ThreadPoolExecutor(max_workers=32) as executor:
      future_to_index = {
          executor.submit(
              retried_evaluate,
              item,
          ): idx
          for idx, item in enumerate(instances)
      }
      for future in tqdm.tqdm(
          futures.as_completed(future_to_index), total=len(instances)
      ):
        idx = future_to_index[future]
        result[idx] = future.result()
    return [
        result[idx] for idx in range(len(instances))
    ]
