import tensorflow as tf

def generator_loss(targets, predictions, prob_predictions):
    '''
    Generator loss.
    '''
    #mean_squared_error真实标签和预测标签得MSE、binary_crossentropy交叉熵损失
    loss = tf.keras.losses.mean_squared_error(y_true=targets, y_pred=predictions)
    loss += tf.keras.losses.binary_crossentropy(y_true=tf.ones_like(prob_predictions), y_pred=prob_predictions)#创建一个和输入参数维度一样，元素为1得的张量
    
    return tf.reduce_mean(loss)用于计算张量沿指定轴上的平均值，主要用于降维或计算张量的平均值


def discriminator_loss(prob_targets, prob_predictions):
    '''
    Discriminator loss.
    '''
    
    loss = tf.keras.losses.binary_crossentropy(y_true=tf.ones_like(prob_targets), y_pred=prob_targets)
    loss += tf.keras.losses.binary_crossentropy(y_true=tf.zeros_like(prob_predictions), y_pred=prob_predictions)
    
    return tf.reduce_mean(loss)
