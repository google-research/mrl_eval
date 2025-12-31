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

"""Library for autoraters with Evergreen models and generic autorater class.
"""

import abc
import dataclasses
from typing import Callable, Generic, Mapping, Protocol, Sequence, TypeVar, Any


# Define a Protocol that describes the "shape" of a dataclass.
# All dataclasses have a __dataclass_fields__ class variable.
class DataclassProtocol(Protocol):
  __dataclass_fields__: dict[str, dataclasses.Field[Any]]

# The type of the autorater input and output must be a dataclass.
T = TypeVar('T', bound=DataclassProtocol)  # The type of the autorater input.
U = TypeVar('U', bound=DataclassProtocol)  # The type of the autorater output.
# The type of the autorater aggregated score(s): float for a single score or a
# mapping from a name to a score for multiple scores.
V = TypeVar('V', float, Mapping[str, float])


class LLM(abc.ABC):
  """Abstract base class for LLM."""

  @abc.abstractmethod
  def generate(self, prompt):
    """Generates a response to the given prompt."""
    raise NotImplementedError()


class Executor(abc.ABC, Generic[T, U]):
  """Abstract base class for executing evaluations in parallel.

  This class defines an interface for executing a given `evaluate_fn` function
  on a sequence of input instances, using parallel processing to speed up
  the evaluation.
  """

  def execute(
      self, evaluate_fn, instances
  ):
    """Executes the given function in parallel.

    Args:
      evaluate_fn: A function that takes an input instance of type `T`,
        autorater input, and returns an output instance of type `U`, autorater
        output.
      instances: A sequence of input instances of type `T`.

    Returns:
      A sequence of output instances of type `U`, corresponding to the
      evaluation of each input instance.
    """
    raise NotImplementedError()


class Autorater(abc.ABC, Generic[T, U, V]):
  """Abstract base class for autoraters."""

  def __init__(self, model, executor):
    self._model = model
    self._executor = executor

  @abc.abstractmethod
  def evaluate(self, eval_instance):
    """Evaluates the autorater on a single instance."""
    raise NotImplementedError()

  @abc.abstractmethod
  def score(self, eval_outputs):
    """Returns the aggregated score(s) given a list of autorater outputs."""
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def name(self):
    raise NotImplementedError()

  def evaluate_parallel(
      self,
      eval_instances
      ):
    """Evaluates the autorater on a sequence of instances in parallel.

    Args:
      eval_instances: A sequence of input instances of type `T`.

    Returns:
      A sequence of output instances of type `U`, corresponding to the
      evaluation of each input instance.
    """
    return self._executor.execute(self.evaluate, eval_instances)
