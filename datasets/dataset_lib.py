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

"""Abstract class for the Dataset object.

This class shouldn't be initiated and should only be used as base class for
child classes, usually using a factory.
"""

import abc
from collections.abc import Callable, Iterable, Mapping, Sequence
import os
import pathlib
from typing import Any

import tensorflow as tf

from mrl_eval.datasets import constants
from mrl_eval.utils import io_utils

TFRECORDS_DIR = "TFRecords"
JSONL_DIR = "jsonl"

# RawExample and FeatureMap depend on the specific implementation
RawExample = Any
FeatureMap = Any
RawDataset = list[RawExample]
SerializedExample = bytes
MetricsScores = Any
MetricsFn = Callable[[list[Any], list[Any]], dict[str, float]]

logger = tf.get_logger()


class Dataset(abc.ABC):
  """Abstract class for datasets.

  This class defines the Dataset class functionalities:
  - Read/Write functionalities.
  - Ingests functionalities - converting  into specific formats (such as
    TFRecords or JSONL) and then writing them to Disk.
  - Evaluation functionalities.
  """

  def preprocess_dataset(self, save_tfrecord):
    """Preprocesses the dataset."""
    for split, filename in self.raw_files.items():
      dataset = self._read_and_process_dataset(filename)
      self._write_dataset_to_jsonl(split, dataset)
      if save_tfrecord:
        self._write_dataset_to_tfrecord(split, dataset)

  def _write_dataset_to_jsonl(self, split, dataset):
    """Writes a dataset split to a jsonl file."""
    out_path = self.jsonl_out_path(split)
    io_utils.write_jsonl(out_path, dataset)

  def _write_dataset_to_tfrecord(self, split, dataset):
    """Writes a dataset split to a jsonl file."""
    out_path = self.tfrecord_out_path(split)
    serialized_dataset = self.transform_to_tfrecord(dataset)
    self.write_tfrecord(out_path, serialized_dataset)

  def _read_and_process_dataset(self, filename):
    """Reads and processes a raw dataset."""
    raw_dataset = self.read_raw_data_file(self.dataset_dir / filename)
    return self.process_raw_examples(raw_dataset)

  def transform_to_tfrecord(self, dataset):
    """Serializes a raw dataset to tfrecords."""
    serialized_dataset = tf.data.Dataset.from_generator(
        generator=lambda: (self._serialize_example(ex) for ex in dataset),
        output_types=tf.string,
        output_shapes=(),
    )

    return serialized_dataset

  def write_tfrecord(
      self, path, dataset
  ):
    """Write dataset to a file path."""
    io_utils.create_dir(str(path.parent))

    with tf.io.TFRecordWriter(str(path)) as file_writer:
      for ex in dataset:
        file_writer.write(ex.numpy())

    logger.info(f"Dataset written to ${path}")

  def _serialize_example(self, ex):
    """serializing example as binary."""
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=self.map_to_feature(ex))
    )
    return example_proto.SerializeToString()

  def tfrecord_out_path(self, split_name):
    return self.dataset_dir / TFRECORDS_DIR / f"{split_name}.tfrecord"

  def jsonl_out_path(self, split_name):
    return self.dataset_dir / JSONL_DIR / f"{split_name}.jsonl"

  def bytes_feature(self, value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

  def int64_feature(self, value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

  def filename_without_extension(self, filename):
    return os.path.splitext(filename)[0]

  def read_test_targets(self):
    gold_path = str(self.jsonl_out_path("test"))
    dataset = io_utils.read_jsonl(gold_path)

    # Get gold targets for every example id
    gold_examples = {}
    for example in dataset:
      gold_examples[example["id"]] = self.get_outputs(example)

    return gold_examples

  @abc.abstractmethod
  def read_raw_data_file(self, path):
    """Dataset file reader for trainin with TFRecord format.

    Args:
      path: path to the input file.

    Returns:
      A list of dictionaries of all relevant dataset example attributes.
    """
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def raw_files(self):
    """Maaping of splits to original dataset files."""
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def metrics(self):
    """Returns the metrics to be calculated for this dataset."""
    raise NotImplementedError()

  @abc.abstractmethod
  def process_raw_examples(self, dataset):
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def dataset_name(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_inputs(self, example):
    """Returns the input to give to the model for a given example."""
    raise NotImplementedError()

  @abc.abstractmethod
  def get_outputs(self, example):
    """Returns the expected response for a given example."""
    raise NotImplementedError()

  @abc.abstractmethod
  def get_example_id(self, example):
    raise NotImplementedError()

  @property
  def dataset_dir(self):
    """Returns the directory of the raw files.

    This directory is used both for reading input files and for storing the
    output files in a subfolder of this directory.
    e.g. this method returns pathlib.Path("dataset/location").
    """
    return pathlib.Path(constants.BASE_PATH) / self.dataset_name

  @abc.abstractmethod
  def map_to_feature(self, ex):
    """mapping RawExample into a FeatureMap as expected for TFRecord.

    Args:
      ex: a raw example dictionary

    This function should map the raw example dict into a new dict of binary
    encoding of the features.
    For example:
    ex = {"ex": 123, "tokens": ["token1", "token2"]}
    This function implementation should be:
    feature = {
        "id": self.int64_feature([ex["id"]]),
        "tokens": self.bytes_feature([t.encode() for t in ex["tokens"]])
    }
    """
    raise NotImplementedError()

  def split_dev_from_train(self):
    """Splits the train file into train and dev files."""
    raise NotImplementedError()


class SummarizationDataset(Dataset):
  """Abstract class for summarization datasets."""
  article: str
  summary: str
