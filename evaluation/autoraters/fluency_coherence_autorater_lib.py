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

"""Autorater for evaluating the fluency and coherence of a summary.
"""

import dataclasses
import json
import logging
from typing import Mapping, Sequence

from mrl_eval.evaluation.autoraters import autoraters_lib
from mrl_eval.evaluation.autoraters import summarization_autoraters_lib


SummarizationInstance = summarization_autoraters_lib.SummarizationInstance


_QUALITY_PROMPT = """
You are an expert AI-powered text quality analyst. Your goal is to evaluate the intrinsic quality of a model-generated summary.

You must perform your evaluation based exclusively on the provided text, without any knowledge of the original source article.

Please evaluate the summary based on the following three criteria. For each criterion, provide a score on a scale of 1 to 5 (where 1 is very poor and 5 is excellent) and a brief, one-sentence justification for your score.

**Evaluation Criteria:**

**1. Fluency & Readability (Score 1-5):**
* How natural and well-written is the text?
* Is the language clear, grammatically correct, and free of spelling or punctuation errors?


**2. Coherence (Score 1-5):**
* Are the sentences self-contained? Do they introduce all necessary entities?
* Do the sentences and ideas within the summary connect logically?


**Output format:**

Output a dictionary
* score: an integer between 1 and 5
* justification: a concise, one-sentence justification for the score


**Example 1:**

**Input:**
The CEO of ABC, David Aaron, announced that the company's annual conference would be held in London, a location chosen to better accommodate European partners.

**Output:**
{{
  "fluency": {{
    "score": 5,
    "justification": "The text is clear, grammatically correct, and free of spelling or punctuation errors."
  }},
  "coherence": {{
    "score": 5,
    "justification": "The text flows logically and introduces all necessary entities ('David Aaron'), making it fully coherent and self-contained."
}}}}

**Example 2:**

**Input:**
He confirmed the merger was a success. The new policy will take effect tomorrow.

**Output:**
{{
  "fluency": {{
    "score": 5,
    "justification": "The text is clear, grammatically correct, and free of spelling or punctuation errors."
  }},
  "coherence": {{
    "score": 1,
    "justification": "The text fails on two fronts: it is not self-contained (who is 'He'?) and the sentences are logically disconnected."
}}}}


**Now, please analyze the following summary:**
{summary}
"""


@dataclasses.dataclass(frozen=True)
class FluencyOutput:
  score: int | None
  justification: str | None


@dataclasses.dataclass(frozen=True)
class CoherenceOutput:
  score: int | None
  justification: str | None


@dataclasses.dataclass(frozen=True)
class QualityOutput:
  id: str
  fluency: FluencyOutput
  coherence: CoherenceOutput


class QualityAutorater(
    autoraters_lib.Autorater[
        SummarizationInstance, QualityOutput, Mapping[str, float]
    ]
):
  """Autorater for evaluating the quality of a model generated_summary."""

  _prompt_template: str = _QUALITY_PROMPT

  def __init__(
      self, model, executor
  ):
    super().__init__(model=model, executor=executor)

  def _parse_json(self, response):
    """Parses the response from the model.

    Args:
      response: the raw response from the model

    Returns:
      A tuple of FluencyOutput and CoherenceOutput. If the response is invalid,
      the tuple will contain FluencyOutput and CoherenceOutput with score and
      justification set to None.
    """
    response = response.replace("```json", "")
    response = response.replace("```", "")
    try:
      parsed_response = json.loads(response)
      fluency_output = FluencyOutput(**parsed_response["fluency"])
      coherence_output = CoherenceOutput(**parsed_response["coherence"])

    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
      logging.warning("Failed to parse response: %s", response)
      fluency_output = FluencyOutput(score=None, justification=None)
      coherence_output = CoherenceOutput(score=None, justification=None)

    return fluency_output, coherence_output

  def evaluate(self, eval_instance):
    """Evaluates the quality of a model generated_summary."""
    prompt = self._prompt_template.format(summary=eval_instance.model_summary)
    response = self._model.generate(prompt)
    fluency_output, coherence_output = self._parse_json(response)
    quality_output = QualityOutput(
        id=eval_instance.id,
        fluency=fluency_output,
        coherence=coherence_output,
    )
    return quality_output

  def score(
      self, eval_outputs
  ):
    """Returns the overall score for the given quality outputs."""
    if not eval_outputs:
      return {"fluency": 0.0, "coherence": 0.0}
    return {
        "fluency": (
            sum(
                q.fluency.score
                for q in eval_outputs
                if q.fluency.score is not None
            )
            / len(eval_outputs)
        ),
        "coherence": (
            sum(
                q.coherence.score
                for q in eval_outputs
                if q.coherence.score is not None
            )
            / len(eval_outputs)
        ),
    }

  @property
  def name(self):
    return "quality"
