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

"""Preprocessing the MSA Sentiment Analysis dataset and writing it to storage."""

from collections.abc import Callable, Mapping, Sequence
import pathlib
from typing import Any, Union
import immutabledict
import pandas as pd
import tensorflow as tf

from mrl_eval.datasets import constants
from mrl_eval.datasets import dataset_lib
from mrl_eval.evaluation import metrics
from mrl_eval.utils import io_utils

RawExample = dict[str, Union[int, str]]
FeatureMap = dict[str, tf.train.Feature]
RawDataset = dataset_lib.RawDataset


class MSASentiment(dataset_lib.Dataset):
  """Implementation of the Dataset class for Arabic Sentiment Analysis."""

  ID_KEY = "id"  # review id form the original dataset.
  RATING_KEY = "rating"
  LABEL_KEY = "sentiment"
  TEXT_KEY = "review"
  POSITIVE_LABEL = "إيجابي"
  NEGATIVE_LABEL = "سلبي"
  NEUTRAL_LABEL = "محايد"

  def __init__(self, val_size_from_train = None):
    super().__init__(val_size_from_train=val_size_from_train)
    self._rating_to_ar_label_name = {
        1: self.NEGATIVE_LABEL,
        2: self.NEGATIVE_LABEL,
        3: self.NEUTRAL_LABEL,
        4: self.POSITIVE_LABEL,
        5: self.POSITIVE_LABEL,
    }
    self._test_rating_to_sample_size = {
        1: 250,
        2: 250,
        3: 500,
        4: 250,
        5: 250,
    }  # We want ~500 examples per sentiment label in the test set.

  @property
  def dataset_name(self):
    return constants.MSA_SENTIMENT

  @property
  def reviews_file(self):
    return "reviews.tsv"

  @property
  def raw_files(self):
    return {
        split: f"5class-unbalanced-{split}.txt"
        for split in ["train", "test"]
    }

  @property
  def metrics(self):
    """Returns the metrics to be calculated for this dataset."""
    return [
        metrics.accuracy,
        metrics.get_macro_f1_fn(
            list(set(self._rating_to_ar_label_name.values()))
        ),
    ]

  def _downsample_test_set(self, test_df):
    """Downsample the test_df stratified by the sentiment labels."""
    return pd.concat([
        test_df
        [test_df.rating == rating]
        .sample(n=sample_size, random_state=42)
        for rating, sample_size in self._test_rating_to_sample_size.items()
    ])

  def _process_dataset_split(
      self,
      reviews_df,
      split,
      filename,
      save_tfrecord,
  ):
    remaining_indices = set(reviews_df.index)
    split_ids = [
        int(ind)
        for ind in io_utils.read_txt_lines(self.dataset_dir / filename)
        if int(ind) in remaining_indices  # to only use non-duplicated indices.
    ]
    dataset = reviews_df.loc[split_ids]

    if split == "test":
      dataset = self._downsample_test_set(dataset)

    dataset = self._translate_labels(dataset)

    # Used pandas so far, return to using list of dicts (RawDataset)
    dataset = dataset.to_dict("records")

    if self.val_size_from_train and split == "train":
      # create a dev set by sampling 1000 examples from the train set
      dataset, devset = self._split_train_and_dev(dataset)
      self._write_dataset(
          split="val", dataset=devset, save_tfrecord=save_tfrecord
      )

    self._write_dataset(split, dataset, save_tfrecord)

  def preprocess_dataset(
      self, save_tfrecord, split_dev_from_train = True
  ):  # pylint: disable=g-doc-args
    """Override of the base preprocess_dataset.

    Since we want to:
      - Load the reviews df only once.
      - Add columns to the reviews df.
      - Downsample the test set.
    """  # pylint: enable=g-doc-args

    reviews_df = io_utils.read_tsv(self.dataset_dir / self.reviews_file)
    reviews_df.columns = ["rating", self.ID_KEY, "user_id", "book_id", "review"]

    # Dedup, since same review id or actual review can be be a duplicate.
    reviews_df = (
        reviews_df
        .drop_duplicates(subset=self.ID_KEY)
        .drop_duplicates(subset=self.TEXT_KEY)
    )

    for split, filename in self.raw_files.items():
      self._process_dataset_split(reviews_df, split, filename, save_tfrecord)

  def map_to_feature(self, ex):
    # Disabling pytype here as it can't infer correct type for the dict values.
    # pytype: disable=attribute-error
    feature = {
        "id": self.bytes_feature([str(ex[self.ID_KEY]).encode()]),
        self.TEXT_KEY: self.bytes_feature([ex[self.TEXT_KEY].encode()]),
        self.LABEL_KEY: self.bytes_feature([ex[self.LABEL_KEY].encode()]),
    }
    # pytype: enable=attribute-error

    return feature

  def name_to_features(self):
    return immutabledict.immutabledict({
        "id": tf.io.FixedLenFeature([], tf.string),
        self.TEXT_KEY: tf.io.FixedLenFeature([], tf.string),
        self.LABEL_KEY: tf.io.FixedLenFeature([], tf.string),
    })

  def _translate_labels(self, dataset):
    """Translates the sentiment labels from ratings to Arabic sentiment labels."""
    dataset[self.LABEL_KEY] = (
        dataset[self.RATING_KEY]
        .map(self._rating_to_ar_label_name)
    )
    return dataset

  def get_inputs(self, example):
    return str(example[self.TEXT_KEY])

  def get_outputs(self, example):
    return example[self.LABEL_KEY]

  def get_example_id(self, example):
    return example[self.ID_KEY]

  def read_raw_data_file(self, file_path):
    raise NotImplementedError()

  def process_raw_examples(self, dataset):
    return dataset
