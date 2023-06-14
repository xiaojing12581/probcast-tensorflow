import tensorflow as tf
#生成器
def generator(gru_units, dense_units, sequence_length, noise_dimension, model_dimension):

    '''
    Generator model, see Figure 1 in the ProbCast paper.

    Parameters:
    __________________________________
    gru_units: list.
        Number of hidden units of each GRU layer.每个GRU层的隐藏单元数

    dense_units: int.
        Number of hidden units of the dense layer.密集层的隐藏单元数

    sequence_length: int.
        Number of past time steps used as input.用作输入的过去时间步数

    noise_dimension: int.
        Dimension of the noise vector concatenated to the outputs of the GRU block.连接到GRU模块输出的噪声向量的维数

    model_dimension: int.
        Number of time series.时间序列的数量
    '''
    
    # Inputs.用于构建网络得第一层（输入层）
    inputs = tf.keras.layers.Input(shape=(sequence_length, model_dimension))

    # GRU block.门限循环单元网络
    outputs = tf.keras.layers.GRU(units=gru_units[0], return_sequences=False if len(gru_units) == 1 else True)(inputs)
    for i in range(1, len(gru_units)):
        outputs = tf.keras.layers.GRU(units=gru_units[i], return_sequences=True if i < len(gru_units) - 1 else False)(outputs)

    # Noise vector.
    noise = tf.keras.layers.Input(shape=noise_dimension)
    outputs = tf.keras.layers.Concatenate(axis=-1)([noise, outputs])#指定维度拼接

    # Dense layers.全连接层（特征提取器）
    outputs = tf.keras.layers.Dense(units=dense_units)(outputs)
    outputs = tf.keras.layers.Dense(units=model_dimension)(outputs)

    return tf.keras.models.Model([inputs, noise], outputs)#创建模型

#鉴别器
def discriminator(gru_units, dense_units, sequence_length, model_dimension):

    '''
    Discriminator model, see Figure 2 in the ProbCast paper.

    Parameters:
    __________________________________
    gru_units: list.
        Number of hidden units of each GRU layer.

    dense_units: int.
        Number of hidden units of the dense layer.

    sequence_length: int.
        Number of past time steps used as input.

    model_dimension: int.
        Number of time series.
    '''

    # Inputs.
    inputs = tf.keras.layers.Input(shape=(sequence_length + 1, model_dimension))

    # GRU block.
    outputs = tf.keras.layers.GRU(units=gru_units[0], return_sequences=False if len(gru_units) == 1 else True)(inputs)
    for i in range(1, len(gru_units)):
        outputs = tf.keras.layers.GRU(units=gru_units[i], return_sequences=True if i < len(gru_units) - 1 else False)(outputs)

    # Dense layers.
    outputs = tf.keras.layers.Dense(units=dense_units)(outputs)
    outputs = tf.keras.layers.Dense(units=1, activation='sigmoid')(outputs)

    return tf.keras.models.Model(inputs, outputs)
