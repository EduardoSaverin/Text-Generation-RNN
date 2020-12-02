# Colab Link: https://colab.research.google.com/drive/19OQj_hmxUXo718Uw5L35nrbq4qPSzoFU?usp=sharing
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, GRU
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
# from google.colab import drive
# drive.mount('/content/drive')
print(f'Executing eagerly : {tf.executing_eagerly()}')

# text = open('/content/drive/MyDrive/Colab Notebooks/Datasets/shakespeare.txt', 'r').read()
text = open('data/shakespeare.txt', 'r').read()
vocab = sorted(set(text))

char_to_index = {char: index for index, char in enumerate(vocab)}
index_to_char = np.array(vocab)
encoded_text = np.array([char_to_index[c] for c in text])
print(f'Shape of Encoded Text {encoded_text.shape}')

seq_length = 120
total_seq = len(encoded_text) // seq_length
print(f"Total Sequences : {total_seq}")

char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)

sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)


def create_seq(seq):
    input_text = seq[:-1]
    output_text = seq[1:]
    return input_text, output_text


dataset = sequences.map(create_seq)

for input_txt, output_txt in dataset.take(1):
    print(input_txt.numpy())
    print(''.join(index_to_char[input_txt.numpy()]))
    print('\n')
    print(output_txt.numpy())
    print(''.join(index_to_char[output_txt.numpy()]))

batch_size = 128
buffer_size = 10000
dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

vocab_size = len(vocab)
embed_size = 64
rnn_neurons = 1024


def create_model(vocab_size, embed_dim, rnn_neurons, batch_size):
    model = Sequential()
    model.add(Embedding(vocab_size, embed_dim,batch_input_shape=[batch_size, None]))
    model.add(GRU(rnn_neurons,return_sequences=True,stateful=True,recurrent_initializer='glorot_uniform'))
    # Final Dense Layer to Predict
    model.add(Dense(vocab_size))
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    return model


# rnn_model = create_model(vocab_size, embed_size, rnn_neurons, batch_size)
# rnn_model.summary()
#
# early_stop = EarlyStopping(monitor="loss", patience=2)
# rnn_model.fit(dataset, epochs=30, callbacks=[early_stop])
# rnn_model.save('shakespeare.h5')
# df = pd.DataFrame(rnn_model.history.history)
# df.plot()

saved_model = create_model(vocab_size, embed_size, rnn_neurons, batch_size=1)
saved_model.load_weights('models/shakespeare.h5')
saved_model.build(tf.TensorShape([1, None]))
saved_model.summary()


def generate_text(model, start_word, gen_size=100, temp=1.0):
    input_eval = [char_to_index[s] for s in start_word]
    input_eval = tf.expand_dims(input_eval, 0)
    generated_text = []
    model.reset_states()
    for i in range(gen_size):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temp
        prediction_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        orig = tf.squeeze(input_eval, 0)[1:]
        input_eval = np.append(orig.numpy(), [[prediction_id]])
        input_eval = tf.expand_dims(input_eval, 0)
        generated_text.append(index_to_char[prediction_id])

    return start_word + ''.join(generated_text)


print(generate_text(saved_model, 'flower', 1000))
