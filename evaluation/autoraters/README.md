# LLM as a judge

## Motivation

While generative tasks have been traditionally evaluated using n-gram based
metrics like ROUGE, these metrics are limited as they only capture surface-level
word overlap, often failing to assess the semantic meaning, coherence, or
factual accuracy of the generated text.

To illustrate that, consider the following example passage for a question
generation task:

"בניגוד לשני אחיו הבוגרים, לא היה אדמונד ג'יימס רוטשילד מעורב ישירות בענייני
הבנקאות של משפחתו ונודע כחובב אמנות ותרבות. את השכלתו רכש בבית-הספר המשפחתי בו
למד עברית, תורה והיסטוריה של העם היהודי מפי מורהו אלברט כהן"

Translation: *Unlike his two older brothers, Edmund James Rothschild was not
directly involved in his family's banking affairs and was known as a lover of
art and culture. He received his education at the family school where he studied
Hebrew, Torah, and the history of the Jewish people from his teacher Albert
Cohen.*

Here, many different yet correct questions exist, including “מי היה מורו של
אדמונד ג'יימס רוטשילד?” (Who was Edmund James Rothschild's teacher?), “מי לימד
את אדמונד עברית?” ( Who taught Edmund Hebrew?), etc., while ROUGE would prefer
only the question lexically similar to the reference question.

To address these limitations, we devise several autoraters (a.k.a. LLM as a
judge) for evaluating the generative tasks of question generation and
summarization.

## Question Generation

Given a short paragraph with a highlighted span, the task is to generate a
plausible question that the span answers. Instead of relying on lexical
features, we use a Large Language Model (Gemini Flash 2.5) to assess if the
highlighted span provides an appropriate answer to the generated question.

The autorater implementation can be found in
[qg_autorater_lib](qg_autorater_lib.py).

We asked human raters to assess the quality of this answerability autorater on
both Hebrew and Arabic (15 examples for each language). We found that, in all
cases, the autorater's binary judgments were identical to the human judgment.

## Summarization

We evaluate the summary according to multiple criteria:

*   Factual grounding: all the information in the summary should be grounded in
    the original article
    ([factual_grounding_autorater_lib](factual_grounding_autorater_lib.py))
*   Relevance: every single fact of the summary should be “salient” enough to
    appear in the summary. ([recall_autorater_lib](recall_autorater_lib.py))
*   Quality: the summary should be both fluent and coherent
    ([fluency_coherence_autorater_lib](fluency_coherence_autorater_lib.py)).

### Factual grounding:

We borrow and adapt the factual grounding prompt from the
[FACTS](https://arxiv.org/abs/2501.03200) paper.

We assess the quality of the factual grounding autorater with native speakers on
three datasets and obtain the following results:

Language | Dataset      | Accuracy w/ human judgment
:------: | :----------: | :------------------------:
Hebrew   | hesum        | 91%
Hebrew   | hebsummaries | 95%
Arabic   | ar_xlsum     | 93.5%

### Relevance

We evaluate the relevance of a model-generated summary in two steps. First, we
decompose the reference summary into minimal, standalone “statements.” Second,
we check whether each statement is covered by the model-generated summary. This
step essentially consists of predicting an entailment relation between the
model-generated summary and each reference statement, making it conceptually
close to the factual grounding autorater described above.

This two-step procedure, similar to the Pyramid evaluation method, measures
recall—how much information from the reference summary is captured by the model
output.

We follow this procedure because different datasets contain varying lengths of
summaries, and models trained different datasets thus differ in the extent of
information included in the summary. A single, universal evaluator for relevance
is therefore not appropriate, and we must rely on the inherent properties of
each dataset.

### Fluency and Coherence

Finally, we build an autorater to evaluate the general quality of the generated
summary, focusing on fluency and coherence. This autorater outputs a 1-5 score
for both fluency and coherence.

Evaluating the quality of text summaries is inherently subjective. Directly
comparing human scores to model predictions can be misleading because humans and
models may not assess texts on the same scale. However, this scaling discrepancy
isn't an issue as long as humans and models agree on the relative ranking of the
models. Therefore, instead of having human annotators assign individual scores
to each summary, we use a pairwise comparison approach. We present summaries
from two different systems ($$s_1$$ and $$s_2$$) and ask annotators to indicate
whether:

*   $$s_1$$ is better than $$s_2$$ (1)
*   $$s_2$$ is better than $$s_1$$ (-1)
*   They are roughly equivalent (0)

We then compute the difference between the model's quality scores:
$$\mathrm{diff}=q(s_1)-q(s_2)$$. This difference score should correlate strongly
with human judgments. Specifically:

*   If $$s_1$$ is better than $$s_2$$, the $$\mathrm{diff}$$ score should be
    positive.
*   If $$s_2$$ is better than $$s_1$$, the $$\mathrm{diff}$$ score should be
    negative.
*   If they are equivalent, the $$\mathrm{diff}$$ score should be close to zero.

We asked two native Hebrew speakers to perform this pairwise comparison for 100
samples from Hesum and 100 samples from Hebsummaries. The Arabic dataset
consists of single-sentence summaries, which are not suitable for assessing
coherence. We compute the spearman correlation between the human preferences and
the $$\mathrm{diff}=q(s_1)-q(s_2)$$ scores.

Dataset      | Correlation
:----------: | :--------------------:
Hesum        | 0.54 (p-value < 0.001)
HebSummaries | 0.66 (p-value < 0.001)
