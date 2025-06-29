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

"""Preprocessing the NEMO dataset and writing it to storage.

Paper: https://arxiv.org/pdf/2007.15620.pdf
Nemo is a morphologically aware NER dataset with the following entity types:
ORG (Organization), PER (Person), GPE (Geopolitical Entity), LOC (Location),FAC
(Facility), WOA (Work of Art), EVE (Event), DUC (Product), ANG (Language).

The dataset works on a few "levels". Token level and Morpheme level:
We have the following use cases (See paper for more use cases and details):

Token Level:
Input and output are at the token level.
That means the units we work with are whitespace separated words. For example:
Input: Ani meYisrael (I am from Israel)
The named entities in this sentence are: [GPE: meYisrael]

Morpheme level:
Here the output assumes morphlogical segmentation. For the same example:
Input: Ani meYisrael (I am from Israel)
The named entities in this sentence are: [GPE: Yisrael]
"""

from collections.abc import Iterator, Mapping, Sequence
import pathlib
from typing import Any, Union

import immutabledict
import tensorflow as tf

from mrl_eval.datasets import constants
from mrl_eval.datasets import dataset_lib
from mrl_eval.evaluation import metrics

Tokens = list[str]
Tag = str
Tags = list[Tag]
RawExample = dict[str, Union[int, str, Tokens, Tags]]
FeatureMap = dict[str, tf.train.Feature]
RawDataset = dataset_lib.RawDataset

ENTITIES_SEQ_SEPARATOR = "$$"
CHUNK_REPRESENTATION = "BIOSE"


class Nemo(dataset_lib.Dataset):
  """Implementation of dataset_lib.Dataset for NEMO.

  This class transforms from raw dataset files
  (https://github.com/OnlpLab/NEMO-Corpus/tree/main/data/spmrl/gold) into
  TensorFlow records. This calss calls the formulation functions implemented in
  this file such that the output is a dictionary of all formulations.
  """

  _level = None

  @property
  def dataset_name(self):
    return constants.NEMO

  @property
  def raw_files(self):
    """Files templates.

    We return a template of the file.
    Replace "[LEVEL]" with "token-single" for the token level files.
    Replace "[LEVEL]" with "morph" for the morph level files.
    """
    return {
        "train": "[LEVEL]_gold_train.bmes",
        "val": "[LEVEL]_gold_dev.bmes",
        "test": "[LEVEL]_gold_test.bmes",
    }

  @property
  def metrics(self):
    return [metrics.token_level_span_f1]

  def map_to_feature(self, ex):
    """mapping RawExample into a FeatureMap as expected for TFRecord."""

    # Disabling pytype here as it can't infer correct type for the dict values.
    # pytype: disable=attribute-error
    feature = {
        "id": self.bytes_feature([ex["id"].encode()]),
        "inputs": self.bytes_feature([ex["inputs"].encode()]),
        "targets_as_entity_markers_token_level": self.bytes_feature(
            [ex["targets_as_entity_markers_token_level"].encode()]
        ),
        "targets_as_entity_markers_morph_level": self.bytes_feature(
            [ex["targets_as_entity_markers_morph_level"].encode()]
        ),
    }
    # pytype: enable=attribute-error

    return feature

  def name_to_features(self):
    return immutabledict.immutabledict({
        "id": tf.io.FixedLenFeature([], tf.string),
        "inputs": tf.io.FixedLenFeature([], tf.string),
        "targets_as_entity_markers_token_level": tf.io.FixedLenFeature(
            [], tf.string
        ),
        "targets_as_entity_markers_morph_level": tf.io.FixedLenFeature(
            [], tf.string
        ),
    })

  def tfrecord_out_path(self, filename):
    return (
        self.dataset_dir
        / dataset_lib.TFRECORDS_DIR
        / f"{filename.replace('[LEVEL]_', '')}.tfrecord"
    )

  def jsonl_out_path(self, filename):
    return (
        self.dataset_dir
        / dataset_lib.JSONL_DIR
        / f"{filename.replace('[LEVEL]_', '')}.jsonl"
    )

  def _read_and_process_specific_level_data(
      self, filename, level
  ):
    dataset = self.read_raw_data_file(
        self.dataset_dir / filename.replace("[LEVEL]", level)
    )
    return self.process_raw_examples(dataset)

  def _read_and_process_dataset(self, filename):
    token_level_dataset = self._read_and_process_specific_level_data(
        filename, "token-single"
    )
    morph_level_dataset = self._read_and_process_specific_level_data(
        filename, "morph"
    )
    dataset = list(
        self._merge_datasets(token_level_dataset, morph_level_dataset)
    )

    return dataset

  def _merge_datasets(
      self, token_level_ds, morph_level_ds
  ):
    for token_level_ex, morph_level_ex in zip(token_level_ds, morph_level_ds):
      assert token_level_ex["id"] == morph_level_ex["id"]

      yield {
          "id": token_level_ex["id"],
          "inputs": token_level_ex["inputs"],
          "targets_as_entity_markers_token_level": token_level_ex[
              "targets_as_entity_markers"
          ],
          "targets_as_entity_markers_morph_level": morph_level_ex[
              "targets_as_entity_markers"
          ],
      }

  def get_inputs(self, example):
    return example["inputs"]

  def get_outputs(self, example):
    return example[f"targets_as_entity_markers_{self._level}_level"]

  def get_example_id(self, example):
    return example["id"]

  def process_raw_examples(self, raw_dataset):
    dataset = []
    for ex in raw_dataset:
      dataset.append({
          "id": ex["id"],
          "inputs": " ".join(ex["tokens"]),
          "targets_as_entity_markers": targets_as_entity_markers_formulation(
              ex
          ),
      })
    return dataset

  def read_raw_data_file(self, file_path):
    with tf.io.gfile.GFile(file_path) as f:
      sents = f.read().split("\n\n")

    return _extract_tokens_and_tags_from_sents(sents)


class NemoToken(Nemo):
  _level = "token"


class NemoMorph(Nemo):
  _level = "morph"


def _extract_tokens_and_tags_from_sents(sents):
  """Extracts tokens and tags from sentences."""
  dataset = []
  for i, sent in enumerate(sents):
    if not sent:
      continue

    tokens, tags = [], []
    for token_and_tag in sent.split("\n"):
      token_and_tag_parts = token_and_tag.split(" ")
      tags.append(token_and_tag_parts[-1])
      tokens.append("".join(token_and_tag_parts[:-1]))

    if not _validate_tags(tags):
      print(
          f"Invalid tag. Skipped sentence id: {i}.\n"
          f"tokens:{tokens}\n"
          f"tags:{tags}"
      )
      continue

    dataset.append({"id": f"nemo_{i}", "tokens": tokens, "tags": tags})
  return dataset


def targets_as_entity_markers_formulation(ex):
  """Formulate example where entities are denoted with markers.

  Output is represented as the input with additional entities markers that wrap
  each entity and its corresponding type.

  Output should be formulated in the following way:
  Word1 [LabelName1 Word2 Word3] ...

  For example:
    for the input:
      Barack Obama was born in Honolulu
    We create the following targets:
      [PER Barack Obama] was born in [GPE Honolulu]

  Args:
    ex: a RawExample is a list of RawExample.

  Returns:
    The transformed output from RawExample to str.
  """

  ret = []
  curr_entity = []
  for token, tag in zip(ex["tokens"], ex["tags"]):
    if _not_entity(tag):
      ret.append(token)
      continue

    curr_entity.append(token)

    if _is_single_token_entity(tag) or _entity_end(tag):
      ret.append(f"[{_tag_name(tag)} {' '.join(curr_entity)}]")
      curr_entity = []

  return " ".join(ret)


def _validate_tags(tags):
  return all([t[0] in CHUNK_REPRESENTATION for t in tags])


def _not_entity(tag):
  return tag[0] == "O"


def _is_single_token_entity(tag):
  return tag[0] == "S"


def _entity_end(tag):
  return tag[0] == "E"


def _tag_name(tag):
  return tag[2:]
