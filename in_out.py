import tensorflow as tf

class input_transformer(tf.keras.layers.Layer):

    def __init__(self,d_model):

        super(input_transformer,self).__init__()

        self.i1 = tf.keras.layers.Dense(d_model)
        self.i2 = tf.keras.layers.Dense(d_model)
    
    def call(self,inp):

        inp=self.i1(inp)
        inp=self.i2(inp)

        return inp

class output_transformer(tf.keras.layers):

    def __init__(self,num_of_outputs):

        super(output_transformer,self).__init__()

        self.o1 = tf.keras.layers.Dense(num_of_outputs)
        self.o2 = tf.keras.layers.Dense(num_of_outputs)
    
    def call(self,out):

        out=self.o1(out)
        out=self.o2(out)
        out=tf.nn.softmax(out, axis=-1)

        return out