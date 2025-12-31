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

"""Autorater for evaluating the factual grounding of a summary with respect to the original article.
"""

import dataclasses
import json
import re
from typing import Sequence

from absl import logging

from mrl_eval.evaluation.autoraters import autoraters_lib
from mrl_eval.evaluation.autoraters import summarization_autoraters_lib

SummarizationInstance = summarization_autoraters_lib.SummarizationInstance

# This prompt was taken from the FACTS paper (https://arxiv.org/abs/2501.03200)
_PROMPT_TEMPLATE = """
You are a helpful and harmless AI assistant. You will be provided with an article and a model-generated summary.
Your task is to analyze the summary sentence by sentence and classify each sentence according to its relationship with the provided article.

**Instructions:**

1. **Decompose the summary into individual sentences.**
2. **For each sentence, assign one of the following labels:**
* **‘supported‘**: The sentence is entailed by the given article. Provide a supporting excerpt from the article. The supporting except must *fully* entail the sentence. If you need to cite multiple supporting excepts, simply concatenate them.
* **‘unsupported‘**: The sentence is not entailed by the given article. No excerpt is needed for this label.
* **‘contradictory‘**: The sentence is falsified by the given article. Provide a contradicting excerpt from the article.
* **‘no_rad‘**: The sentence does not require factual attribution (e.g., opinions, greetings, questions, disclaimers). No excerpt is needed for this label.
3. **For each label, provide a short rationale explaining your decision.** The rationale should be separate from the excerpt.
4. **Be very strict with your ‘supported‘ and ‘contradictory‘ decisions.** Unless you can find straightforward, indisputable evidence excerpts *in the article* that a sentence is ‘supported‘ or ‘contradictory‘, consider it ‘unsupported‘. You should not employ world knowledge unless it is truly trivial.

**Input Format:**
The input will consist of two parts, clearly separated:
* **Article:** The textual article to summarize.
* **Summary:** The model-generated summary to be analyzed.

**Output Format:**
For each sentence in the response, output a JSON object with the following fields:
* ‘"sentence"‘: The sentence being analyzed.
* ‘"label"‘: One of ‘supported‘, ‘unsupported‘, ‘contradictory‘, or ‘no_rad‘.
* ‘"rationale"‘: A brief explanation for the assigned label.
* ‘"excerpt"‘: A relevant **brief** excerpt from the article. Only required for ‘supported‘ and ‘contradictory‘ labels. Otherwise, write an empty string.
Output each JSON object on a new line. Make sure the output can be parsed by python json.loads.

**Example:**

**Input:**
Article: Apples are red fruits. Bananas are yellow fruits.
Summary: Apples are red. Bananas are green. Bananas are cheaper than apples. Enjoy yourfruit!

**Output:**
{{"sentence": "Apples are red.", "label": "supported", "rationale": "The contextexplicitly states that apples are red.", "excerpt": "Apples are red fruits."}}
{{"sentence": "Bananas are green.", "label": "contradictory", "rationale": "The contextstates that bananas are yellow, not green.", "excerpt": "Bananas are yellow fruits."}}
{{"sentence": "Bananas are cheaper than apples.", "label": "unsupported", "rationale": "The context does not mention the price of bananas or apples.", "excerpt": ""}}
{{"sentence": "Enjoy your fruit!", "label": "no_rad", "rationale": "This is a generalexpression and does not require factual attribution.", "excerpt": ""}}

**Now, please analyze the following article and summary:**

**Article:** {article}
**Summary:** {summary}
"""


@dataclasses.dataclass(frozen=True)
class FactualGroundingOutputSingleSentence:
  sentence: str
  label: str
  rationale: str
  excerpt: str


@dataclasses.dataclass(frozen=True)
class FactualGroundingOutput:
  id: str
  sentences: list[FactualGroundingOutputSingleSentence]


class FactualGroundingAutorater(
    autoraters_lib.Autorater[
        SummarizationInstance,
        FactualGroundingOutput,
        float
    ]
):
  """Autorater for evaluating the factual grounding of a summary with respect to the original article."""
  _prompt_template: str = _PROMPT_TEMPLATE

  def __init__(
      self,
      model,
      executor
  ):
    """Initializes the FactualGroundingAutorater."""
    super().__init__(model=model, executor=executor)

  def _parse_json(
      self, response
  ):
    """Parses the JSON response from the model."""
    open_bracket_indices = [
        match.span()[0] for match in re.finditer(r'{', response)
    ]
    close_bracket_indices = [
        match.span()[0] for match in re.finditer(r'}', response)
    ]
    if len(open_bracket_indices) != len(close_bracket_indices):
      print(
          'Failed to parse JSON: Mismatched brackets in response'
          f' {len(open_bracket_indices)} {len(close_bracket_indices)}'
      )
      return []
    sentences = []
    for start, end in zip(open_bracket_indices, close_bracket_indices):
      sentences.append(response[start : end + 1])
    parsed_sentences: list[FactualGroundingOutputSingleSentence] = []
    for line in sentences:
      if not line:
        continue
      try:
        parsed = json.loads(line)
        parsed = FactualGroundingOutputSingleSentence(**parsed)
        parsed_sentences.append(parsed)
      except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        logging.warning('Failed to parse JSON: %s', line)
    return parsed_sentences

  def evaluate(
      self,
      eval_instance
  ):
    """Evaluates the factual grounding of a summary with respect to the original article."""
    prompt = self._prompt_template.format(
        article=eval_instance.article, summary=eval_instance.model_summary
    )
    response = self._model.generate(prompt)
    parsed_sentences = self._parse_json(response)
    return FactualGroundingOutput(
        id=eval_instance.id,
        sentences=parsed_sentences
    )

  def score(self, eval_outputs):
    """Returns the factual grounding score given a list of autorater outputs."""
    if not eval_outputs:
      return 0.0
    scores = [
        self._score_single_output(eval_output) for eval_output in eval_outputs
    ]
    return sum(scores) / len(eval_outputs)

  def _score_single_output(self, eval_output):
    """Returns the factual grounding score given a list of autorater outputs."""
    if eval_output.sentences and any(
        s.label != 'no_rad' for s in eval_output.sentences
    ):
      score = sum(
          1 for s in eval_output.sentences if s.label == 'supported'
      ) / sum(1 for s in eval_output.sentences if s.label != 'no_rad')
    else:
      score = 0.0
    return score

  @property
  def name(self):
    return 'factual_grounding'
