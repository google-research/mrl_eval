# Datasets for MRLEval - A Benchmark for Morphologically Rich Languages

## Coreference Resolution

Coreference resolution tasks are modelled using two variants: End-to-end and
Gold Mentions.

### End-to-end Coreference

In the end-to-end variant, the task is to identify and cluster all coreferring
mentions.

For a given input text, all space-separated words are indexed:

**Example Input Text:** `1_Lionel 2_Messi 3_has 4_won 5_a 6_record 7_seven
8_Ballon 9_d'Or 10_awards. 11_He 12_signed 13_for 14_PSG 15_in 16_August
17_2021. “18_I 19_would 19_like 20_to 21_thank 22_my 23_family” 24_said 25_the
26_Argentinian 27_footballer. 28_Messi 29_holds 30_the 31_record 32_for 33_most
34_goals 35_in 36_La 37_Liga. 38_Paris 39_Saint-Germain 40_hopes 41_he 42_will
43_do 44_the 45_same 46_in 47_Ligue 48_1.` (example borrowed from [LingMess:
Linguistically Informed Multi Expert Scorers for Coreference
Resolution](https://aclanthology.org/2023.eacl-main.202/))

The target is a list of clusters, where each cluster is a set of mentions.
Clusters are separated by a between-cluster separator (`<N>`) and mentions
within a cluster are separated by an in-cluster separator (`<S>`).

For each mention, the model should reproduce the indexed word(s) that contain
it, placing square brackets `[]` around the exact mention span. This applies in
two ways:

1.  **Full-Word Mentions:** When a mention consists of one or more complete
    words, the brackets enclose the entire indexed word(s).
    * **Example:** A mention spanning `1_Lionel` and `2_Messi` is written as
        `[1_Lionel 2_Messi]`.

2.  **Sub-Word Mentions:** When a mention is a part of a single word (e.g., a
    clitic or morpheme), the model must reproduce the *full* indexed word, but
    with the brackets placed *internally* around only the sub-word span.
    * **Example:** For a Hebrew input word `2_לביתך` (to your house), where the
        mention is only the possessive suffix `ך` (your), the correct target
        format is `2_לבית[ך]`.

**Example Target:**
```
[1_Lionel 2_Messi]<S>[11_He]<S>[18_I]<S>[22_my]<S>[25_the 26_Argentinian 27_footballer]<S>[28_Messi]<S>[41_he]<N>[14_PSG]<S>[38_Paris 39_Saint-Germain]
```

### Gold Mentions Coreference

In the Gold Mentions variant, the task is to perform coreference resolution on a
text where the mentions are already identified. The model is not required to
perform mention detection.

For a given input text, all mentions are marked up (for example with square
brackets) and assigned a unique index.

**Example Input Text:**
```
1_[Lionel Messi] has won a record seven Ballon d'Or awards. 2_[He] signed for
3_[PSG] in August 2021. “4_[I] would like to thank 5_[my] family” said
6_[the Argentinian footballer]. 7_[Messi] holds the records for most goals in
La Liga. 8_[Paris Saint-Germain] hopes 9_[he] will do the same in Ligue 1.
```

The model should output the coreference clusters. Two formats for the target
clusters are proposed:

**1. Cluster Sets:**
The target is a list of clusters, where each cluster is a set of mention indices
belonging to that cluster. Clusters are separated by a between-cluster separator
(`<N>`) and mentions within a cluster are separated by an in-cluster separator
(`<S>`).

**Example Target:**
```
1<S>2<S>4<S>5<S>6<S>9<N>3<S>8
```

**2. Anaphor-Antecedent Pairs:**
The target is a list of anaphor-antecedent pairs, where the anaphor is the
mention that refers to another mention (the antecedent). Pairs are separated
by `#`.

**Example Target:**
```
1,1#2,1#3,3#4,1#5,1#6,3#7,7#8,3#9,1
```
The default format is Anaphor-Antecedent Pairs, but this can be overridden by
changing the `target_format_strategy` property in the dataset class.

**Sub-word Mentions:**
For mentions that are part of a larger word, the entire word can be marked as a
single mention to avoid creating unnatural text. Each gold mentions dataset
class has a `merge_subword_mentions` parameter for this purpose.
