import csv
import numpy as np

""" Most of codes are from from tflearn (https://github.com/tflearn/tflearn)
"""

class VocabDict(object):
  def __init__(self):
    self.dict = {}

  def add(self, word):
    if word not in self.dict:
      self.dict[word] = len(self.dict)
    return self.fit(word)

  def size(self):
    return len(self.dict)

  def fit(self, word):
    return self.dict[word]

target_dict = VocabDict()

def to_categorical(y):
  result = []
  l = len(np.unique(y))
  for i, d in enumerate(y):
    tmp = [0.] * l
    for _i, _d in enumerate(d):
      tmp[target_dict.add(_d)] = 1.
    result.append(tmp)
  return result

def load_csv(filepath, target_columns=-1, columns_to_ignore=None,
    has_header=True, n_classes=None):

  if isinstance(target_columns, list) and len(target_columns) < 1:
    raise Exception('target_columns must be list with one value at least')

  from tensorflow.python.platform import gfile
  with gfile.Open(filepath) as csv_file:
    data_file = csv.reader(csv_file)
    if not columns_to_ignore:
      columns_to_ignore = []
    if has_header:
      header = next(data_file)

    data, target = [], []
    for i, d in enumerate(data_file):
      data.append([_d for _i, _d in enumerate(d) if _i not in target_columns and _i not in columns_to_ignore])
      target.append([_d for _i, _d in enumerate(d) if _i in target_columns])

    target = to_categorical(target)
    return data, target
