import tensorflow as tf
from tensorflow import gfile
import tflearn
import numpy as np
import re

class TagSpace(object):
  def __init__(self):
    pass

  def create_model(self, l, tN, N=100000, d=10, K=5, H=1000, m=0.05, reuse=False):
  #def create_model(self, l, tN, N=200000, d=100, K=5, H=1000, m=0.1, reuse=False):
    '''
    N = 1000000 (Paper)
    d = Unknown
    '''
    with tf.variable_scope('TagSpace', reuse=reuse):
      lr = tf.placeholder('float32', shape=[1], name='lr')
      doc = tf.placeholder('float32', shape=[None, l], name='doc')
      tag_flag = tf.placeholder('float32', shape=[None, tN], name='tag_flag')

      doc_embed = tflearn.embedding(doc, input_dim=N, output_dim=d)
      #with tf.device('/cpu:0'):
      #  itn = tf.shape(tag_flag)[0]
      #  it = tf.tile([x for x in range(tN)], [itn])
      #  ii = tf.reshape(it, [tf.shape(tag_flag)[0], tN])
      #self.lt_embed = lt_embed = tflearn.embedding(ii, input_dim=tN, output_dim=d)
      #self.lt_embed = lt_embed = tflearn.embedding(tf.ones_like(tag_flag), input_dim=tN, output_dim=d)
      self.lt_embed = lt_embed = tf.Variable(tf.random_normal([tN, d], stddev=0.1))

      net = tflearn.conv_1d(doc_embed, K, H, activation='tanh')
      net = tflearn.max_pool_1d(net, l)
      net = tflearn.tanh(net)
      self.logit = logit = tflearn.fully_connected(net, d, activation=None)

      zero_vector = tf.zeros(shape=(1,1), dtype=tf.float32)

      logit = tf.expand_dims(logit, 1)
      logit_set = tf.concat([logit for i in range(tN)], axis=1)

      tag_flag_ex = tf.expand_dims(tag_flag, 2)
      tg = tf.concat([tag_flag_ex for i in range(d)], axis=2)

      self.tag_logit = tf.reduce_sum(tf.multiply(logit_set, tf.multiply(tf.ones_like(tg), lt_embed)), axis=2)

      self.positive_logit = positive_logit = tf.reduce_sum(tf.multiply(logit_set, tf.multiply(tg, lt_embed)), axis=2)
      with tf.device('/cpu:0'):
        self.sexy = sexy = tf.map_fn(lambda x: (tf.boolean_mask(x[0], x[1]), True), (positive_logit, tf.not_equal(positive_logit, zero_vector)))
      positive = tf.reduce_min(sexy[0], axis=1)
      self.positive = positive

      tag_flag_ex = tf.expand_dims(1-tag_flag, 2)
      tg = tf.concat([tag_flag_ex for i in range(d)], axis=2)
      negative_logit = tf.reduce_sum(tf.multiply(logit_set, tf.multiply(tg, lt_embed)), axis=2)

      with tf.device('/cpu:0'):
        self.sexy2 = sexy2 = tf.map_fn(lambda x: (tf.boolean_mask(x[0], x[1]), True), (negative_logit, tf.not_equal(negative_logit, zero_vector)))
      negative = tf.reduce_max(sexy2[0], axis=1)
      self.negative = negative
      #negative = tf.reduce_max(tf.reduce_mean(tf.matmul(logit_set, tf.transpose(tf.multiply(tg, lt_embed))), axis=1))

      #self.f_loss = f_loss = tf.square(-tf.reduce_mean(positive - negative))
      print(tf.expand_dims(m - positive + negative,1))
      print(tf.expand_dims(tf.fill([tf.shape(doc)[0]], 10e7),1))
      print(tf.reduce_min([tf.expand_dims(m - positive + negative,1), tf.expand_dims(tf.fill([tf.shape(doc)[0]], 10e7),1)], axis=0))
      print(tf.reduce_max([tf.reduce_min([tf.expand_dims(m - positive + negative,1), tf.expand_dims(tf.fill([tf.shape(doc)[0]], 10e7),1)], axis=0), tf.zeros([tf.shape(doc)[0], 1])], axis=0))
      self.f_loss = f_loss = tf.reduce_mean(tf.reduce_max([tf.reduce_min([tf.expand_dims(m - positive + negative,1), tf.expand_dims(tf.fill([tf.shape(doc)[0]], 10e7),1)], axis=0), tf.zeros([tf.shape(doc)[0], 1])], axis=0))
      #self.f_loss = f_loss = tf.losses.hinge_loss(tag_flag, self.tag_logit)
      #self.f_loss = f_loss = -tf.reduce_mean(positive - negative)

      opt = tf.train.AdamOptimizer(learning_rate=lr[0])
      self.op = opt.minimize(f_loss)

  def train_opts(self):
    return [self.op, self.f_loss, self.logit, self.lt_embed, self.sexy[0][0], self.sexy2[0][0]]

  def test_opts(self):
    return [self.tag_logit]

word_pad_length = 60
tag_size = 5
#tokenizer = tflearn.data_utils.VocabularyProcessor(word_pad_length)
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

#def read_my_file_format(filename_queue, tag_size):
#  reader = tf.TextLineReader()
#  key, record_string = reader.read(filename_queue)
#  label, _, example = tf.decode_csv(record_string, [[0], ['1'], ['1']])
#  processed_example = string_parser(example)
#  label = tf.one_hot(tf.assign_sub(label, 1), tag_size)
#  return processed_example, label
#
#def input_pipeline(filenames, batch_size, tag_size, num_epochs=None):
#  filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
#  example, label = read_my_file_format(filename_queue, tag_size)
#  min_after_dequeue = 100
#  capacity = min_after_dequeue + 3 * batch_size
#  example_batch, label_batch = tf.train.shuffle_batch(
#      [example, label], batch_size=batch_size, capacity=capacity,
#      min_after_dequeue=min_after_dequeue)
#  return example_batch, label_batch

model = TagSpace()
with tf.Session() as sess:
  #with tf.device('/cpu:0'):
  model.create_model(word_pad_length, tag_size)
  train_opts = model.train_opts()
  test_opts = model.test_opts()

  sess.run(tf.global_variables_initializer())

  #batch_words, batch_tags = input_pipeline(['./data/ag_news_csv/train.csv'], 32, tag_size, num_epochs=10)
  words, tags = tflearn.data_utils.load_csv('./data/ag_news_csv/train.csv', target_column=0, columns_to_ignore=[1], has_header=False, categorical_labels=True, n_classes=tag_size)

  from sklearn.utils import shuffle
  words, tags = shuffle(words, tags)

  words = string_parser(words, fit=True)
  word_input = tflearn.data_utils.pad_sequences(words, maxlen=word_pad_length)

  num_epochs = 5
  batch_size = 20
  total = len(word_input)
  step_print = int((total/batch_size) / 13)
  global_step = 0
  lr = 1e-2
  #lr_decr = 0
  lr_decr = (lr - (1e-9))/num_epochs

  print('start training')
  for epoch_num in range(num_epochs):
    epoch_loss = 0
    step_loss = 0
    for i in range(int(total/batch_size)):
      batch_input, batch_tags = (word_input[i*batch_size:(i+1)*batch_size], tags[i*batch_size:(i+1)*batch_size])
      result = sess.run(train_opts, feed_dict={'TagSpace/doc:0': batch_input, 'TagSpace/tag_flag:0': batch_tags, 'TagSpace/lr:0': [lr]})
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
    result = sess.run(test_opts, feed_dict={'TagSpace/doc:0': batch_input, 'TagSpace/tag_flag:0': np.ones_like(batch_tags)})
    arr = result[0]
    for j in range(len(batch_tags)):
      rs+=np.sum(np.argmax(arr[j]) == np.argmax(batch_tags[j]))
  print(rs/total)

  sess.close()
