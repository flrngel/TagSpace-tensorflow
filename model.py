import tensorflow as tf
import tflearn

class TagSpace(object):
  def __init__(self):
    pass

  def create_model(self, l, tN, N=100000, d=10, K=5, H=1000, m=0.05, reuse=False):
    '''
    N = 1000000 (Paper)
    d = Unknown
    '''
    with tf.variable_scope('TagSpace', reuse=reuse):
      lr = tf.placeholder('float32', shape=[1], name='lr')
      doc = tf.placeholder('float32', shape=[None, l], name='doc')
      tag_flag = tf.placeholder('float32', shape=[None, tN], name='tag_flag')

      doc_embed = tflearn.embedding(doc, input_dim=N, output_dim=d)
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
      self.f_positive = f_positive = tf.map_fn(lambda x: (tf.boolean_mask(x[0], x[1]), True), (positive_logit, tf.not_equal(positive_logit, zero_vector)))
      positive = tf.reduce_min(f_positive[0], axis=1)
      self.positive = positive

      tag_flag_ex = tf.expand_dims(1-tag_flag, 2)
      tg = tf.concat([tag_flag_ex for i in range(d)], axis=2)
      negative_logit = tf.reduce_sum(tf.multiply(logit_set, tf.multiply(tg, lt_embed)), axis=2)

      self.f_negative = f_negative = tf.map_fn(lambda x: (tf.boolean_mask(x[0], x[1]), True), (negative_logit, tf.not_equal(negative_logit, zero_vector)))
      self.negative = negative = tf.reduce_max(f_negative[0], axis=1)

      self.f_loss = f_loss = tf.reduce_mean(tf.reduce_max([tf.reduce_min([tf.expand_dims(m - positive + negative,1), tf.expand_dims(tf.fill([tf.shape(doc)[0]], 10e7),1)], axis=0), tf.zeros([tf.shape(doc)[0], 1])], axis=0))

      opt = tf.train.AdamOptimizer(learning_rate=lr[0])
      self.op = opt.minimize(f_loss)

  def train_opts(self):
    return [self.op, self.f_loss, self.logit, self.lt_embed, self.f_positive[0][0], self.f_negative[0][0]]

  def test_opts(self):
    return [self.tag_logit]
