import tensorflow as tf
from attention import scaled_dot_product_attention,MultiHeadAttention,point_wise_feed_forward_network

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, num_parts, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads, num_parts)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, x, training):

    attn_output, _ = self.mha(x, x, x)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

    return out2

class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, num_parts, dff, rate=0.1):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    # self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
    # self.pos_encoding = positional_encoding(maximum_position_encoding, 
    #                                         self.d_model)


    self.enc_layers = [EncoderLayer(d_model, num_heads, num_parts, dff, rate) 
                       for _ in range(num_layers)]

    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, training):

    seq_len = tf.shape(x)[1]

    # adding embedding and position encoding.
    # x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    # x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    # x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training)

    return x  # (batch_size, input_seq_len, d_model)
