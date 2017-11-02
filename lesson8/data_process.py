from python_speech_features import logfbank
import scipy.io.wavfile as wav
import os
import os.path
import getpass
import numpy as np

# must change the timit corpus to WAV format
TRAIN_PATH = '/home/%s/data/timit/train/' % getpass.getuser() 
SAVE_TRAIN_DATA_PATH = '/home/%s/code_works/timit/'

def load_train_data():
    train_data = []
    labels = np.array([], dtype=int)
    label_count = 0

    for root, dirs, files in os.walk(TRAIN_PATH):
        wav_files = [file for file in files if file.endswith('.wav')]
        train_data_same_person = []
        for wav_file in wav_files:
            (rate, sig) = wav.read(root + "/" + wav_file)
            fbank_beats = logfbank(sig, rate, nfilt=40)
            train_data_same_person.append(fbank_beats)
            labels = np.concatenate((labels, np.full(fbank_beats.shape[0], label_count)))

        if train_data_same_person:
            train_data_same_person = np.concatenate(train_data_same_person)
            train_data.append(train_data_same_person)
            label_count+=1

    train_data = np.concatenate(train_data)
    train_labels = np.zeros((train_data.shape[0], label_count + 1), dtype=int)
    train_labels[np.arange(train_data.shape[0]), [labels]] = 1
    return train_data, train_labels

def shuffle_data(data, labels):
    num, _ = data.shape
    shuffle_index = np.arange(num)
    np.random.shuffle(shuffle_index)
    return data[shuffle_index], labels[shuffle_index]

train_data, train_labels = load_train_data()
train_data, train_labels = shuffle_data(train_data, train_labels)

def generate_batch(batch_size):
    train_num = train_data.shape[0]
    cursor = 0
    while True:
        next_cursor = cursor + batch_size
        if next_cursor < train_num - 1:
            yield train_data[cursor: next_cursor], train_labels[cursor: next_cursor]
        else:
            yield np.concatenate((train_data[cursor:], train_data[:next_cursor])), np.concatenate((train_labels[cursor:], train_labels[:next_cursor]))
        cursor = next_cursor

# batch_size = 64
# batch_num = train_data.shape[0] // batch_size
# generator = generate_batch(batch_size)

# for i in range(batch_num):
#     batch = train_data[batch_size * i: batch_size * (i + 1)]
#     assert np.array_equal(batch, next(generator)[0])


