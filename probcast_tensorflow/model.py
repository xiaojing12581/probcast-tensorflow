import pandas as pd
import numpy as np
import tensorflow as tf
pd.options.mode.chained_assignment = None

from probcast_tensorflow.modules import generator, discriminator
from probcast_tensorflow.losses import generator_loss, discriminator_loss

class ProbCast():

    def __init__(self,
                 y,
                 sequence_length,
                 forecast_length,
                 quantiles=[0.1, 0.5, 0.9],
                 generator_gru_units=[64, 32],
                 generator_dense_units=16,
                 discriminator_gru_units=[64, 32],
                 discriminator_dense_units=16,
                 noise_dimension=100,
                 noise_dispersion=1):

        '''
        Implementation of multivariate time series forecasting model introduced in Koochali, A., Dengel, A.,
        & Ahmed, S. (2021). If You Like It, GAN It — Probabilistic Multivariate Times Series Forecast with GAN.
        In Engineering Proceedings (Vol. 5, No. 1, p. 40). Multidisciplinary Digital Publishing Institute.

        Parameters:
        __________________________________
        y: np.array.
            Target time series, array with shape (samples, targets) where samples is the length of the
            time series and targets is the number of target time series.
            目标时间序列，带形状的数组(样本，目标)其中样本是的长度时间序列和目标是目标时间序列的数目。

        sequence_length: int.
            Number of past time steps to use as input.用作输入的过去时间步数

        forecast_length: int.
            Number of future time steps to forecast.要预测的未来时间步数

        quantiles: list.
            Quantiles of target time series to forecast.要预测的目标时间序列的五分位数

        generator_gru_units: list.
            The length of the list is the number of GRU layers in the generator, the items in the list are
            the number of hidden units of each layer.
            列表的长度是生成器中GRU层的数量，列表中的项目有每层的隐藏单元数。

        generator_dense_units: int.
            Number of hidden units of the dense layer in the generator.生成器中密集层的隐藏单元数。

        discriminator_gru_units: list.
            The length of the list is the number of GRU layers in the discriminator, the items in the list
            are the number of hidden units of each layer.
            列表的长度是鉴别器中的GRU层数，列表中的项目是每层的隐藏单元数。

        discriminator_dense_units: int.
            Number of hidden units of the dense layer in the discriminator.鉴别器中密集层的隐藏单元数。

        noise_dimension: int.
            Dimension of the noise vector concatenated to the outputs of the GRU block in the generator.
            连接到发生器中GRU模块输出的噪声向量的维度。

        noise_dispersion: float.
            Standard deviation of the noise vector concatenated to the outputs of the GRU block in the generator.
            连接到发生器中GRU模块输出的噪声矢量的标准差。
        '''

        # Extract the quantiles.提取五分之一
        quantiles = np.unique(np.array(quantiles))#对一维数组或列表unique去除其中重复的元素，按元素由小到大返回新的无元素重复的元组或列表
        if 0.5 not in quantiles:
            quantiles = np.sort(np.append(0.5, quantiles))

        # Scale the targets.缩放目标
        mu, sigma = np.mean(y, axis=0), np.std(y, axis=0)#计算沿指定轴的标准差。返回数组元素的标准差
        y = (y - mu) / sigma

        # Save the inputs.
        self.y = y
        self.mu = mu
        self.sigma = sigma
        self.sequence_length = sequence_length
        self.forecast_length = forecast_length
        self.generator_gru_units = generator_gru_units
        self.generator_dense_units = generator_dense_units
        self.discriminator_gru_units = discriminator_gru_units
        self.discriminator_dense_units = discriminator_dense_units
        self.noise_dimension = noise_dimension
        self.noise_dispersion = noise_dispersion
        self.quantiles = quantiles
        self.samples = y.shape[0]#y的行数
        self.targets = y.shape[1]#y的列数

    def fit(self,
            learning_rate=0.001,
            batch_size=32,
            epochs=100,
            verbose=True):

        '''
        Train the model.

        Parameters:
        __________________________________
        learning_rate: float.
            Learning rate.

        batch_size: int.
            Batch size.

        epochs: int.
            Number of epochs.

        verbose: bool.
            True if the training history should be printed in the console, False otherwise.
            如果应该在控制台中打印训练历史记录，则为True，否则为False。
        '''

        # Build the models.
        generator_model = generator(
            gru_units=self.generator_gru_units,
            dense_units=self.generator_dense_units,
            sequence_length=self.sequence_length,
            noise_dimension=self.noise_dimension,
            model_dimension=self.targets
        )

        discriminator_model = discriminator(
            gru_units=self.discriminator_gru_units,
            dense_units=self.discriminator_dense_units,
            sequence_length=self.sequence_length,
            model_dimension=self.targets
        )

        # Instantiate the optimizers.实例化优化器
        generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Define the training loop.
        @tf.function
        def train_step(data):
            with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:#自动求导

                # Extract the input sequences and target values.提取输入序列和目标值
                inputs = tf.cast(data[:, :-1, :], dtype=tf.float32)#取到最后一列，不包含最后一列
                targets = tf.cast(data[:, -1:, :], dtype=tf.float32)#取最后一列

                # Generate the noise vector.生成噪声向量
                noise = tf.random.normal(#正态分布
                    mean=0.0,#均值
                    stddev=self.noise_dispersion,#标准差
                    shape=(data.shape[0], self.noise_dimension)#形状
                )

                # Generate the model predictions.生成模型预测
                predictions = generator_model([inputs, noise])
                predictions = tf.reshape(predictions, shape=(data.shape[0], 1, self.targets))#self.targets = y.shape[1]#y的列数

                # Pass the actual sequences and the predicted sequences to the discriminator.将实际序列和预测序列传递给鉴别器
                prob_targets = discriminator_model(tf.concat([inputs, targets], axis=1))
                prob_predictions = discriminator_model(tf.concat([inputs, predictions], axis=1))

                # Calculate the loss.计算损失
                g = generator_loss(targets, predictions, prob_predictions)
                d = discriminator_loss(prob_targets, prob_predictions)

            # Calculate the gradient.计算梯度
            dg = generator_tape.gradient(g, generator_model.trainable_variables)#计算g对于generator_model的所有可训练变量的梯度
            dd = discriminator_tape.gradient(d, discriminator_model.trainable_variables)

            # Update the weights.更新权重
            generator_optimizer.apply_gradients(zip(dg, generator_model.trainable_variables))#根据dg来优化generator_model的变量
            discriminator_optimizer.apply_gradients(zip(dd, discriminator_model.trainable_variables))

            return g, d

        # Generate the training batches.生成训练批次
        dataset = tf.keras.utils.timeseries_dataset_from_array(#在以数组形式提供的时间序列上创建一个滑动窗口的数据集
            data=self.y,
            targets=None,
            sequence_length=self.sequence_length + 1,
            batch_size=batch_size
        )

        # Train the model.训练模型
        for epoch in range(epochs):
            for data in dataset:
                g, d = train_step(data)
            if verbose:
                print('Epoch: {}  Generator Loss: {:.8f}  Discriminator Loss: {:.8f}'.format(1 + epoch, g, d))

        # Save the model.保存模型
        self.model = generator_model

    def forecast(self, y, samples=100):

        '''
        Generate the forecasts.
        Parameters:
       ________________________________
        y: np.array.
            Past values of the time series.时间序列的过去值
        samples: int.
            The number of samples to generate for estimating the quantiles.为估计分位数而生成的样本数

        Returns:
        __________________________________
        df: pd.DataFrame.
            Data frame including the actual values of the time series and the predicted quantiles.
            包括时间序列的实际值和预测分位数的数据帧。
        '''

        # Scale the targets.缩放目标
        y = (y - self.mu) / self.sigma
        
        # Generate the forecasts.生成预测
        outputs = np.zeros(shape=(samples, self.forecast_length, self.targets))

        noise = np.random.normal(
            loc=0.0,
            scale=self.noise_dispersion,
            size=(samples, self.forecast_length, self.noise_dimension)
        )

        inputs = y[- self.sequence_length:, :]
        inputs = inputs.reshape(1, self.sequence_length, self.targets)#1行，每行self.sequence_length行，self.targets列
        inputs = np.repeat(inputs, samples, axis=0)

        for i in range(self.forecast_length):
            outputs[:, i, :] = self.model([inputs, noise[:, i, :]]).numpy()
            inputs = np.append(inputs[:, 1:, :], outputs[:, i, :].reshape(samples, 1, self.targets), axis=1)

        # Organize the forecasts in a data frame.在数据框中组织预测
        columns = ['time_idx']
        columns.extend(['target_' + str(i + 1) for i in range(self.targets)])#用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）
        columns.extend(['target_' + str(i + 1) + '_' + str(self.quantiles[j]) for i in range(self.targets) for j in range(len(self.quantiles))])

        df = pd.DataFrame(columns=columns)
        df['time_idx'] = np.arange(self.samples + self.forecast_length)#返回一个有终点和起点的固定步长的排列

        for i in range(self.targets):
            df['target_' + str(i + 1)].iloc[: - self.forecast_length] = self.mu[i] + self.sigma[i] * self.y[:, i]

            for j in range(len(self.quantiles)):
                df['target_' + str(i + 1) + '_' + str(self.quantiles[j])].iloc[- self.forecast_length:] = \
                    self.mu[i] + self.sigma[i] * np.quantile(outputs[:, :, i], q=self.quantiles[j], axis=0)

        # Return the data frame.
        return df.astype(float)

