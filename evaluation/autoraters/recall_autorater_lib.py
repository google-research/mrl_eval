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

"""Autorater for evaluating the recall of a summary with respect to a reference summary.

This autorater works by breaking the reference summary into a list of
individual, standalone factual statements. It then evaluates the model summary
by asking the model to verify each of the standalone statements. The overall
recall score is the fraction of verified statements.
"""

import dataclasses
from typing import Sequence

from mrl_eval.evaluation.autoraters import autoraters_lib
from mrl_eval.evaluation.autoraters import summarization_autoraters_lib


SummarizationInstance = summarization_autoraters_lib.SummarizationInstance


_STATEMENTS_GENERATION_TEMPLATE = """
You're an helpful and harmless AI assistant. Your role is to extract factual information from a provided reference text.
You will be given a reference text and your goal is to break down this text into a comprehensive list of individual, standalone factual statements. Each statement should be:

* **Factual**: Based directly on information present in the reference text.
* **Standalone**: Meaningful and understandable even without the context of the original text.

These statements will be used to evaluate whether a model response faithfully replicates the content of the reference text.

Example 1:

Text:
Modern lifestyle requires more and more people to commute daily to work in their private cars. The long hours on the road can not only frustrate the driver but also endanger their health. A new study warns that prolonged daily exposure in a car can increase the chance of developing cancer as a result of exposure to substances considered carcinogenic, such as benzene and formalin, which were previously linked to an increased risk of cancer. Volatile organic compounds, like the two mentioned previously, are found in adhesives, paints, rubber, and other materials used in the manufacturing and assembly process of cars. However, air quality tests conducted in vehicles found that their quantities did not exceed the threshold considered dangerous according to official standards. Researchers in California found that in places where people spend an average of more than twenty minutes in a car per day, they are exposed to levels higher than the threshold considered dangerous according to the accepted standard in California. As the time spent in the car lengthened, the level of exposure also increased, and accordingly, the researchers estimated how likely it is to increase the passengers' risk of developing cancer by ten percent or more. The researchers also pointed to a series of previous epidemiological studies that found taxi drivers to be at an increased risk of developing cancer due to increased exposure to benzene and formalin. It is important to remember that epidemiological studies cannot prove causality, but nevertheless, these findings demand more attention to the air quality inside our cars, and perhaps also a demand for the automotive industry to work towards improving its quality.

Statements:
- The modern lifestyle requires more people to commute to work daily in their private car.
- Long hours of driving on the road can frustrate the driver.
- Long hours of driving on the road may endanger the driver's health.
- A new study warns that spending long periods of time in a car every day may increase the risk of cancer.
- The increased risk of cancer (according to the study) is due to exposure to substances considered carcinogenic found in the car, such as benzene and formalin.
- Volatile organic compounds (such as benzene and formalin) are found in materials used in the manufacture and assembly of cars (such as adhesives, paints, rubber).
- Air quality tests conducted in vehicles in the past found that the amount of these substances (such as benzene and formalin) did not exceed the threshold considered dangerous according to official standards.
- Researchers in California found that in places where people spend an average of more than 20 minutes a day in their cars, they are exposed to levels of benzene and formalin that are higher than the threshold considered dangerous according to the standard accepted in California.
- The longer you stay in the car, the higher the level of exposure to benzene and formalin.
- Depending on the level of exposure, the researchers estimated that staying in a car could increase passengers' risk of developing cancer by 10% or more.
- Previous epidemiological studies have found that taxi drivers are at increased risk of cancer due to increased exposure to benzene and formalin.
- Epidemiological studies cannot prove causality (a cause-and-effect relationship).
- The findings (from the new study and previous epidemiological studies) require more attention to the air quality in the car interior.
- The findings may justify a demand from the automotive industry to work to improve air quality in cars.


Example 2:

Text:
Halman-Aldubi note that the American economy is strong, with 5% growth in the third quarter, and that the stock market is expected to continue its positive trend. However, they forecast an interest rate hike in the US in the second half of 2015, which could impact the bond market. In Israel, despite an improvement in unemployment and a rise in private consumption, the Consumer Price Index is expected to remain negative in the coming months due to a drop in the prices of electricity, water, food, and fuel. The US growth data for the third quarter was a positive surprise, exceeding analysts' forecasts and reaching 5%—the highest growth since 2003. The main reasons for this are an increase in business investments and private consumption, primarily due to spending on Obama's healthcare reform.

Statements:
- According to Hellman Aldubi, the American economy is strong.
- US growth in the third quarter reached 5%.
- The stock market is expected to continue its positive trend.
- Hellman Aldubi forecast a US interest rate hike in the second half of 2015.
- Rising interest rates could affect the bond market.
- In Israel, there is an improvement in unemployment
- In Israel, there is an increase in private consumption
- In Israel, despite an improvement in unemployment and an increase in private consumption, the consumer price index is expected to remain negative in the coming months due to the decline in the prices of electricity, water, food and fuel.
- US growth data (5% in the third quarter) was a positive surprise and exceeded analysts' forecasts.
- This growth (5%) was the highest in the US since 2003.
- One of the main reasons for growth in the US was an increase in business investment.
- Another main reason for growth in the US was an increase in private consumption, mainly due to spending on Obama's healthcare reform.

**Now, please analyze the following text:**
{text}

Statements:
"""

_SINGLE_STATEMENT_VERIFICATION_TEMPLATE = """
You're an helpful and harmless AI assistant.

You will be given a model-generated summary and a single statement and your goal is to determine if the statement is mentioned, either explicitly or implicitly, in the summary.

**Input Format:**
The input will consist of two parts, clearly separated:
* **Fact:** The textual statement to be verified.
* **Summary:** The model-generated summary to be analyzed.

**Output Format:**
Please respond with either "Yes" or "No".

**Example 1:**

**Input:**
Fact: Paris is the capital of France.
Summary: Paris, the capital of France, is a popular tourist destination. The Eiffel Tower is the most visited landmark in the city.

**Output:**
Yes

**Example 2:**

**Input:**
Fact: Paris is a big city in Europe.
Summary: Paris, the capital of France, is a popular tourist destination. The Eiffel Tower is the most visited landmark in the city.

**Output:**
No

**Now, please analyze the following fact and summary:**

**Fact:** {statement}
**Summary:** {summary}
"""


@dataclasses.dataclass(frozen=True)
class VerifiedStatement:
  """A single statement for a summary."""

  text: str
  present: bool | None


@dataclasses.dataclass(frozen=True)
class StatementVerificationResponse:
  """The response from the statement verification model."""

  id: str
  statements: Sequence[VerifiedStatement]


class RecallAutorater(
    autoraters_lib.Autorater[
        SummarizationInstance, StatementVerificationResponse, float
    ]
):
  """Autorater for evaluating the recall of a summary with respect to the original article."""
  _statements_generation_template: str = _STATEMENTS_GENERATION_TEMPLATE
  _single_statement_verification_template: str = (
      _SINGLE_STATEMENT_VERIFICATION_TEMPLATE
  )

  def __init__(
      self,
      model,
      executor,
  ):
    """Initializes the RecallAutorater."""
    super().__init__(model=model, executor=executor)

  def _generate_statements(self, reference_summary):
    """Generates statements for a reference summary."""
    statement_prompt = self._statements_generation_template.format(
        text=reference_summary
    )
    statements = self._model.generate(statement_prompt)
    statements = [
        line.strip() for line in statements.split('\n') if line.strip()
    ]
    return statements

  def _parse_statement_verification_response(
      self, response
  ):
    """Parses the statement verification response."""
    response = response.strip()
    if response.lower().startswith('yes'):
      return True
    elif response.lower().startswith('no'):
      return False
    else:
      return None

  def _verify_statement(
      self, statement, summary
  ):
    """Verifies a statement for a summary."""
    statement_verification_prompt = (
        self._single_statement_verification_template.format(
            summary=summary,
            statement=statement,
        )
    )
    statement_verification_response = self._model.generate(
        statement_verification_prompt
    )
    statement_verification = self._parse_statement_verification_response(
        statement_verification_response
    )
    return VerifiedStatement(text=statement, present=statement_verification)

  def score(
      self, eval_outputs
  ):
    """Compute macro average of the recall scores for a list of autorater outputs."""
    if not eval_outputs:
      return 0.0
    scores = [
        self._score_single_output(eval_output) for eval_output in eval_outputs
    ]
    return sum(scores) / len(eval_outputs)

  def _score_single_output(
      self, eval_output
  ):
    """Compute the recall score for a single autorater output."""
    if not eval_output.statements:
      return 0.0
    return sum(
        1 for statement in eval_output.statements if statement.present
    ) / len(eval_output.statements)

  def evaluate(
      self, eval_instance
  ):
    statements = self._generate_statements(eval_instance.reference_summary)

    statements_verified: list[VerifiedStatement] = []
    for statement in statements:
      statement_verification_response = self._verify_statement(
          statement, eval_instance.model_summary
      )
      statements_verified.append(statement_verification_response)
    return StatementVerificationResponse(
        id=eval_instance.id, statements=statements_verified
    )

  @property
  def name(self):
    return 'recall'
