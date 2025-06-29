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

"""Utility functions for io operations."""

import json
import os
import pathlib
from typing import Any, Union

import pandas as pd


open_file = open
isdir = os.path.isdir
makedirs = os.makedirs
exist = os.path.exists



def create_dir(directory):
  """Creates a directory.

  Args:
    directory: string path to the directory to create.
  """
  if not isdir(directory):
    makedirs(directory)


def read_csv(path):
  """Reads a csv file.

  Args:
    path: The path to the file.

  Returns:
    The csv object in the file.
  """
  with open_file(path, "r") as f:
    df = pd.read_csv(f, lineterminator="\n")
  return df


def read_json(path):
  """Reads a json file.

  Args:
    path: The path to the file.

  Returns:
    The json object in the file.
  """
  with open_file(path, "r") as f:
    data = json.load(f)
    return data


def read_jsonl(path):
  """Reads a dataset from a json file.

  Args:
    path: a Pathlib.Path object points to writing path.

  Returns:
    A list of dictionaries representing the dataset.
  """
  ret = []
  with open_file(path, "r") as f:
    for line in f:
      ret.append(json.loads(line.strip()))

  return ret


def read_parquet(path):
  """Reads a parquet file.

  Args:
    path: The path to the file.

  Returns:
    The parquet object in the file.
  """
  with open_file(path, "rb") as f:
    return pd.read_parquet(f)


def write_jsonl(path, dataset):
  """Writes a dataset to a json file.

  Args:
    path: A Pathlib.Path object points to writing path.
    dataset: List of dictionaries representing the dataset.
  """

  create_dir(str(path.parent))

  with open_file(str(path), "w") as file_writer:
    for ex in dataset:
      file_writer.write(json.dumps(ex) + "\n")
