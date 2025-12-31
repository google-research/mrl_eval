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

"""Implements LLM interfaces for external LLM models."""

import logging

from google import genai  # pytype: disable=import-error

from mrl_eval.evaluation.autoraters import autoraters_lib


class Gemini(autoraters_lib.LLM):
  """A wrapper for the Gemini model."""

  def __init__(
      self,
      project_id,
      location,
      model_name = "gemini-2.5-flash",
      temperature = 0.0
  ):
    self._model_name = model_name
    self._temperature = temperature
    self._client = genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
    )

  def generate(self, prompt):
    """Generates a response from the Gemini model."""
    response = self._client.models.generate_content(
        model=self._model_name,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            thinking_config=genai.types.ThinkingConfig(thinking_budget=0),
            max_output_tokens=4096,
            temperature=self._temperature,
        ),
    ).text
    if response is None:
      logging.warning(
          "Gemini response text is None. Response object: %s", response
      )
      return ""
    else:
      return response
