import tensorflow as tf
import tflearn
import numpy as np
import re
from model import TagSpace
from sklearn.utils import shuffle

word_pad_length = 60
tag_size = 5

TOKENIZER_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", re.UNICODE)
def token_parse(iterator):
  for value in iterator:
    return TOKENIZER_RE.findall(value)

tokenizer = tflearn.data_utils.VocabularyProcessor(word_pad_length, tokenizer_fn=lambda tokens: [token_parse(x) for x in tokens])

def string_parser(arr, fit):
  if fit == False:
    return list(tokenizer.transform(arr))
  else:
    return list(tokenizer.fit_transform(arr))

model = TagSpace()
with tf.Session() as sess:
  #with tf.device('/cpu:0'):
  model.create_model(word_pad_length, tag_size)
  train_opts = model.train_opts()
  test_opts = model.test_opts()

  sess.run(tf.global_variables_initializer())

  words, tags = tflearn.data_utils.load_csv('./data/ag_news_csv/train.csv', target_column=0, columns_to_ignore=[1], has_header=False, categorical_labels=True, n_classes=tag_size)

  words, tags = shuffle(words, tags)

  words = string_parser(words, fit=True)
  word_input = tflearn.data_utils.pad_sequences(words, maxlen=word_pad_length)

  num_epochs = 5
  batch_size = 20
  total = len(word_input)
  step_print = int((total/batch_size) / 13)
  global_step = 0
  lr = 1e-2
  lr_decr = (lr - (1e-9))/num_epochs

  print('start training')
  for epoch_num in range(num_epochs):
    epoch_loss = 0
    step_loss = 0
    for i in range(int(total/batch_size)):
      batch_input, batch_tags = (word_input[i*batch_size:(i+1)*batch_size], tags[i*batch_size:(i+1)*batch_size])
      result = sess.run(train_opts, feed_dict={f'{model.__class__.__name__}/doc:0': batch_input, f'{model.__class__.__name__}/tag_flag:0': batch_tags, f'{model.__class__.__name__}/lr:0': [lr]})
      step_loss += result[1]
      epoch_loss += result[1]
      if i % step_print == 0:
        print(f'step_log: (epoch: {epoch_num}, step: {i}, global_step: {global_step}), Loss:{step_loss/step_print}), Positive: {result[4]}, Negative: {result[5]}')
        step_loss = 0
      global_step += 1
    print(f'epoch_log: (epoch: {epoch_num}, global_step: {global_step}), Loss:{epoch_loss/(total/batch_size)})')
    lr -= lr_decr

  words, tags = tflearn.data_utils.load_csv('./data/ag_news_csv/test.csv', target_column=0, columns_to_ignore=[1], has_header=False, categorical_labels=True, n_classes=tag_size)
  words = string_parser(words, fit=True)
  word_input = tflearn.data_utils.pad_sequences(words, maxlen=word_pad_length)
  total = len(word_input)
  rs = 0.

  for i in range(int(total/batch_size)):
    batch_input, batch_tags = (word_input[i*batch_size:(i+1)*batch_size], tags[i*batch_size:(i+1)*batch_size])
    result = sess.run(test_opts, feed_dict={f'{model.__class__.__name__}/doc:0': batch_input, f'{model.__class__.__name__}/tag_flag:0': np.ones_like(batch_tags)})
    arr = result[0]
    for j in range(len(batch_tags)):
      rs+=np.sum(np.argmax(arr[j]) == np.argmax(batch_tags[j]))
  print(f'Test accuracy: {rs/total}')

  sess.close()

