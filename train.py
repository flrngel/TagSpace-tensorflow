import tensorflow as tf
import tflearn
import numpy as np
import re
from model import TagSpace
from sklearn.utils import shuffle
from reader import load_csv, VocabDict

'''
parse
'''

tf.app.flags.DEFINE_integer('num_epochs', 5, 'number of epochs to train')
tf.app.flags.DEFINE_integer('batch_size', 20, 'batch size to train in one step')
tf.app.flags.DEFINE_integer('labels', 5, 'number of label classes')
tf.app.flags.DEFINE_integer('word_pad_length', 60, 'word pad length for training')
tf.app.flags.DEFINE_float('learn_rate', 1e-2, 'learn rate for training optimization')
tf.app.flags.DEFINE_boolean('shuffle', True, 'shuffle data FLAG')

FLAGS = tf.app.flags.FLAGS

num_epochs = FLAGS.num_epochs
batch_size = FLAGS.batch_size
tag_size = FLAGS.labels
word_pad_length = FLAGS.word_pad_length
lr = FLAGS.learn_rate
lr_decr = (lr - (1e-9))/num_epochs

TOKENIZER_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", re.UNICODE)
def token_parse(iterator):
  for value in iterator:
    return TOKENIZER_RE.findall(value)

tokenizer = tflearn.data_utils.VocabularyProcessor(word_pad_length, tokenizer_fn=lambda tokens: [token_parse(x) for x in tokens])
label_dict = VocabDict()

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

  words, tags = load_csv('./data/ag_news_csv/train.csv', target_columns=[0], columns_to_ignore=[1], target_dict=label_dict)
  if FLAGS.shuffle == True:
    words, tags = shuffle(words, tags)

  words = string_parser(words, fit=True)
  word_input = tflearn.data_utils.pad_sequences(words, maxlen=word_pad_length)
  total = len(word_input)
  step_print = int((total/batch_size) / 13)
  global_step = 0

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

  words, tags = load_csv('./data/ag_news_csv/test.csv', target_columns=[0], columns_to_ignore=[1], target_dict=label_dict)
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

