import numpy as np
import tensorflow as tf
from tensorflow import keras
from RNN_1 import RNNCell
import matplotlib.pyplot as plt
import numpy.random as npr
from tensorflow.keras import regularizers, initializers
from tensorflow.keras.layers import Activation
from scipy.stats import norm
import scipy.linalg as la
from tensorflow.keras import layers
from tensorflow.keras import backend as K

EPOCHS = 20
BATCH_SIZE = 16
TRAINING_SAMPLES = 4096
N_REC = 40

class SimpleRNN(object):

    def __init__(self, h_params):

        # RNN Model Parameters
        self.N_in = h_params['N_in'] #input size
        self.N_out = h_params['N_out'] #output size
        self.N = h_params['N'] #number of neurons
        self.tau = h_params['tau'] #time constant
        self.dt = h_params['dt'] #Intervals between time points
        self.noise_var = h_params['noise_var'] #Noise in the model
        self.dale_ratio = h_params['dale_ratio'] #percentage of neurons excitatory
        self.activation = h_params['activation'] #activation of recurrent units
        self.mask_matrix = h_params['mask']

        dale_vec = np.ones(N_REC)
        dale_vec[int(self.dale_ratio * N_REC):] = -1
        self.dale_matrix = np.diag(dale_vec)

        # Learning Parameters
        self.T = h_params['Total_T'] #Total time of the experiment
        self.t = np.arange(0, self.T, self.dt) #time points at which activity will be measured
        self.seed = 1234
        self.rng = npr.RandomState(self.seed)
        self.n_sample = TRAINING_SAMPLES #number of training samples
        self.out_init = initializers.RandomUniform(minval=0., maxval=0.01, seed=self.seed)

        #Track training progress
        self.train_log = []
        self.val_log = []

    def set_model(self):
        self.rnn_network = RNNCell(units=self.N, mask_matrix = self.mask_matrix, dt = self.dt, tau=self.tau,
                                   noise_var = self.noise_var, dale_ratio = self.dale_ratio,
                                   activation=self.activation)
        rnn_layer = keras.layers.RNN(self.rnn_network, return_sequences=True)

        print('Building Model......................................')
        inputs = keras.Input((len(self.t), self.N_in))
        x = rnn_layer(inputs)
        r = Activation(self.activation)(x)
        outputs = tf.keras.layers.Dense(self.N_out, kernel_initializer=self.out_init,
                                        kernel_regularizer= regularizers.l2(1e-5),
                                        bias_regularizer=regularizers.l2(1e-5))(r)
        model = keras.Model(inputs=inputs, outputs=outputs, name="custom_model")

        #Regularization on input weights
        w = model.layers[1].get_weights()
        w_in_plastic = w[1]
        model.add_loss(lambda: 1e-5*tf.reduce_mean(tf.norm(tf.abs(w_in_plastic), 'euclidean')))

        #Activity Regularizer
        model.add_loss(1e-6*tf.reduce_mean(tf.square(r), axis = -1))
        model.summary()

        return model

    def generate_task_trials(self, trial_params):
        init_wait = trial_params['t_init']
        trial_dur = trial_params['t_on']
        stim_noise = trial_params['stim_noise']
        output_noise = trial_params['output_noise']
        baseline = trial_params['input_baseline']

        #input types
        input_pattern_1 = trial_params['input_1']
        input_pattern_2 = trial_params['input_2']

        #output types
        output_pattern_1 = trial_params['output_1']
        output_pattern_2 = trial_params['output_2']

        inputs = np.zeros((self.n_sample, len(self.t), self.N_in))
        outputs = np.zeros((self.n_sample, len(self.t), self.N_out))

        for sample in range(self.n_sample):
            task_type = npr.randint(2) #task 0(blue) or task 1(red)
            if task_type == 0:
                for t in range(int(init_wait/self.dt), int((init_wait+trial_dur)/self.dt)):
                    inputs[sample, t, :] = input_pattern_1
                    outputs[sample, t, :] = output_pattern_1
            else:
                for t in range(int(init_wait / self.dt), int((init_wait + trial_dur) / self.dt)):
                    inputs[sample, t, :] = input_pattern_2
                    outputs[sample, t, :] = output_pattern_2

        x_train = baseline+inputs+stim_noise*npr.randn(self.n_sample, len(self.t), self.N_in)
        y_train = outputs+output_noise*npr.randn(self.n_sample, len(self.t), self.N_out)

        x_train = x_train.astype('float32')
        y_train = y_train.astype('float32')

        return x_train, y_train


    # def train_model(self, model, x_train, y_train, epochs = EPOCHS, batch_size = BATCH_SIZE,
    #                 learning_rate = 0.001):
    #
    #     lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    #         initial_learning_rate=learning_rate,
    #         decay_steps=1000,
    #         decay_rate=0.9)
    #
    #     opt = keras.optimizers.Adam(learning_rate=lr_schedule)
    #     checkpoint = ModelCheckpoint('../weights/task_weights-{epoch:03d}.h5')
    #
    #     model.compile(optimizer=opt, loss='mse', metrics=['MeanSquaredError'])
    #     history = model.fit(x_train, y_train, validation_split= 0.2, epochs=epochs,
    #                         batch_size=batch_size, callbacks=[checkpoint])
    #
    #     return model, history

    def train_model_custom(self, model, x_train, y_train, epochs = EPOCHS, batch_size = BATCH_SIZE,
                           learning_rate = 0.001):

        # Learning rate schedule
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=1000,
            decay_rate=0.9)

        # Stochastic Optimizer with gradient clipping
        opt = keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)

        # Loss Function
        loss_fn = keras.losses.MeanSquaredError()

        train_metric = keras.metrics.MeanSquaredError()
        val_metric = keras.metrics.MeanSquaredError()

        x_val, y_val = x_train[-1000:], y_train[-1000:]
        x_train, y_train = x_train[:-1000], y_train[:-1000]

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size = 1024).batch(batch_size)

        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_dataset = val_dataset.shuffle(buffer_size=1024).batch(64)

        for epoch in range(epochs):

            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                train_batch_loss = self.train_step(model, loss_fn,
                                                   opt, train_metric,
                                                   x_batch_train, y_batch_train)

            train_loss = train_metric.result()
            self.train_log.append(train_loss)

            for x_batch_val, y_batch_val in val_dataset:
                val_batch_loss = self.test_step(model, loss_fn,
                                                val_metric, x_batch_val, y_batch_val)

            val_loss = val_metric.result()
            self.val_log.append(val_loss)

            print('\n Epoch %d:'% epoch, 'Training loss: %.4f'% (float(train_loss,)),
                  'Validation loss: %.4f' % (float(val_loss, )))

            train_metric.reset_states()
            val_metric.reset_states()
        return model, {'train_loss':self.train_log, 'val_loss':self.val_log}

    @tf.function
    def train_step(self, model, loss_fn, opt, train_metric, x, y):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            train_loss = loss_fn(y, y_pred)
            train_loss += sum(model.losses)

        grads = tape.gradient(train_loss, model.trainable_weights)
        opt.apply_gradients(zip(grads, model.trainable_weights))
        train_metric.update_state(y, y_pred)
        return train_loss

    @tf.function
    def test_step(self, model, loss_fn, val_metric, x, y):
        y_val_pred = model(x, training=False)
        val_loss = loss_fn(y, y_val_pred)

        val_metric.update_state(y, y_val_pred)
        return val_loss

    # def evaluate_model(self, model, x_test, y_test):
    #     preds = model.evaluate(x_test, y_test)
    #     return preds[1]

# class ActivityRegularization(layers.Layer):
#     def __init__(self, rate=1e-5):
#         super(ActivityRegularization, self).__init__()
#         self.rate = rate
#
#     def call(self, inputs):
#         self.add_loss(self.rate*tf.reduce_sum(tf.square(inputs)))
#         return inputs

def create_input_output_pattern(N_in, N_out):
    x = np.linspace(0,N_in, N_in, endpoint=False)
    mu1, mu2 = int(N_in/4)-1 , int(3*N_in/4)
    std = 5
    input_1 = 5*norm.pdf(x, mu1, std)
    input_2 = 5*norm.pdf(x, mu2, std)

    output_1 = np.array([1, 0])
    output_2 = np.array([0, 1])
    return input_1, input_2, output_1, output_2

def hidden_layer_activity(model, t_index, x_train, y_train, N_rec, dale_ratio):
    extractor = keras.Model(inputs=model.inputs,
                            outputs=[layer.output for layer in model.layers])
    intermediate_output = extractor((x_train, y_train))
    recurrent_output = np.array(intermediate_output[2])

    if dale_ratio:
        excitatory = recurrent_output[200, :, :int(dale_ratio*N_rec)]
        inhibitory = recurrent_output[200, :, int(dale_ratio*N_rec):]
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.plot(t_index, excitatory)
        ax1.set_title('Excitatory Neurons')

        ax2.plot(t_index, inhibitory)
        ax2.set_title('Inhibitory Neurons')
        plt.show()


def plot_sample_prediction(model, t_index, x_test, y_test):
    y_pred = model.predict(x_test)
    k = npr.randint(0,x_test.shape[0])
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.plot(y_pred[k,:,0], y_pred[k,:,1])
    ax1.set_title('Predicted Output')
    ax1.set_xlim([-0.2, 1.2])
    ax1.set_ylim([-0.2, 1.2])

    ax2.plot(y_test[k,:,0], y_test[k,:,1])
    ax2.set_title('Actual Output')
    ax2.set_xlim([-0.2, 1.2])
    ax2.set_ylim([-0.2, 1.2])
    plt.show()


def plot_eig(model, dale_ratio, N_rec, mask_matrix):
    w = model.layers[1].get_weights()
    # w_in = np.array(w[1])
    w_rec = np.array(w[0])
    assert(w_rec.shape[0] == N_rec)

    if dale_ratio:
        dale_vec = np.ones(N_rec)
        dale_vec[int(dale_ratio * N_rec):] = -1
        dale_matrix = np.diag(dale_vec)

    hm = plt.imshow(mask_matrix*np.matmul(np.abs(w_rec),dale_matrix), cmap='RdBu', interpolation='nearest')
    plt.colorbar(hm)
    plt.show()

    eva = la.eig(-1*np.eye(N_rec)+mask_matrix*np.matmul(np.abs(w_rec),dale_matrix))
    plt.scatter(eva[0].real, eva[0].imag, 12)
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.xlim([-2,2])
    plt.ylim([-2,2])
    plt.show()

if __name__ == '__main__':

    mask_matrix = npr.randint(2, size=(N_REC, N_REC))
    np.fill_diagonal(mask_matrix, 0)
    h_params = {'N_in':32, 'N_out':2, 'N':N_REC, 'tau':1, 'dt':0.1, 'Total_T':8,
                'dale_ratio':0.8, 'activation':'relu', 'noise_var':0.01,
                'mask':mask_matrix}

    t_index = np.arange(0, h_params['Total_T'], h_params['dt'])

    input_1, input_2, output_1, output_2 = \
        create_input_output_pattern(h_params['N_in'], h_params['N_out'])

    # plt.figure()
    # plt.plot(input_1)
    # plt.plot(input_2)
    # plt.show()

    trial_params = {'t_init': 1, 't_on': 4, 'stim_noise':0.01, 'output_noise': 0.01,
                    'input_baseline': 0.1, 'input_1': input_1, 'input_2': input_2,
                    'output_1': output_1, 'output_2': output_2}

    force_rnn = SimpleRNN(h_params)
    print('Setting Model Data Flow............')
    rnn_model = force_rnn.set_model()

    print('Generating Trial Data.......................')
    x_train, y_train = force_rnn.generate_task_trials(trial_params)

    print('Training model...............................')
    # rnn_model, history = force_rnn.train_model(rnn_model, x_train, y_train)
    rnn_model, history = force_rnn.train_model_custom(rnn_model, x_train, y_train)
    plt.figure()
    plt.plot(history['train_loss'])
    plt.plot(history['val_loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.legend(['Training loss', 'Validation loss'])
    plt.show()

    print('Extracting activity of hidden layers...............................')
    dale_ratio = h_params['dale_ratio']
    hidden_layer_activity(rnn_model, t_index, x_train, y_train, N_REC, dale_ratio)

    print('Analyzing learnt model...............................')
    plot_eig(rnn_model, dale_ratio, N_REC, mask_matrix)

    x_test = x_train[0:100, :, :]
    y_test = y_train[0:100, :, :]

    # print('Showing sample test instance.........................')
    # plot_sample_prediction(rnn_model, t_index, x_test, y_test)
