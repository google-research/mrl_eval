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

"""Utility functions for simple operations (convert, concat) in gin."""
import gin


@gin.configurable
def join(values, delimiter = ','):  # pylint: disable=invalid-name
  """Returns a string version of the input values delimited by the delimiter.

  This function is used within gin files to join strings such as:
    metric_name_builder/gin_str_utils.join:
      values = ["training_eval", %MIXTURE_OR_TASK_NAME, "accuracy"]
      delimiter = "/"

    checkpoints.SaveBestCheckpointer:
      metric_name_to_monitor = @metric_name_builder/gin_str_utils.join()
      metric_mode = 'max'

  Args:
    values: A list of values.
    delimiter: The string delimiter between any two strings.

  Returns:
    The joined string.
  """
  return delimiter.join([str(v) for v in values])
