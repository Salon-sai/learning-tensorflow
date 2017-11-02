from python_speech_features import logfbank
import scipy.io.wavfile as wav
import os
import os.path
import getpass
import numpy as np

# must change the timit corpus to WAV format
TRAIN_PATH = '/home/%s/data/timit/train/' % getpass.getuser()
TEST_PATH = '/home/%s/data/timit/test/' % getpass.getuser()  
SAVE_TRAIN_DATA_PATH = '/home/%s/code_works/timit/'

def load_train_data1():
    data = []
    label_index = np.array([], dtype=int)
    label_count = 0

    for root, dirs, files in os.walk(TRAIN_PATH):
        wav_files = [file for file in files if file.endswith('.wav')]
        data_same_person = []
        for wav_file in wav_files:
            (rate, sig) = wav.read(root + "/" + wav_file)
            fbank_beats = logfbank(sig, rate, nfilt=40)
            data_same_person.append(fbank_beats)
            label_index = np.concatenate((label_index, np.full(fbank_beats.shape[0], label_count)))

        if data_same_person:
            data_same_person = np.concatenate(data_same_person)
            data.append(data_same_person)
            label_count += 1

    data = np.concatenate(data)
    labels = np.zeros((data.shape[0], label_count), dtype=int)
    labels[np.arange(data.shape[0]), [label_index]] = 1
    return data, labels

def load_data(path):
    data = []
    label_index = np.array([], dtype=int)
    label_count = 0

    for root, dirs, files in os.walk(path):
        wav_files = [file for file in files if file.endswith('.wav')]
        data_same_person = []
        for wav_file in wav_files:
            (rate, sig) = wav.read(root + "/" + wav_file)
            fbank_beats = logfbank(sig, rate, nfilt=40)
            data_same_person.append(fbank_beats)

        if wav_files:
            data.append(data_same_person)

    return data, np.arange(len(data))

def load_train_data():
    train_data = []
    label_index = []
    datas, labels = load_data(TRAIN_PATH)

    for data, label in zip(datas, labels):
        data = np.concatenate(data)
        frame_num = data.shape[0]

        train_data.append(data)
        label_index.append(np.full(frame_num, label))
    train_data = np.concatenate(train_data)
    label_index = np.concatenate(label_index)
    print(train_data.shape, label)
    train_label = np.zeros((train_data.shape[0], label + 1))
    train_label[np.arange(train_data.shape[0]), label_index] = 1
    return train_data, train_label


def shuffle_data(data, labels):
    num, _ = data.shape
    shuffle_index = np.arange(num)
    np.random.shuffle(shuffle_index)
    return data[shuffle_index], labels[shuffle_index]

train_data, train_label = load_train_data()
print(train_label)
# train_data1, train_label1 = load_train_data1()
# assert np.array_equal(train_data, train_data1)
# assert np.array_equal(train_label, train_label1)
exit()
# test_data, test_label = load_test_data(TEST_PATH)



train_data, train_label = shuffle_data(train_data, train_label)

def generate_batch(batch_size):
    train_num = train_data.shape[0]
    cursor = 0
    while True:
        next_cursor = cursor + batch_size
        if next_cursor < train_num - 1:
            yield train_data[cursor: next_cursor], train_label[cursor: next_cursor]
        else:
            next_cursor %= train_num
            yield np.concatenate((train_data[cursor:], train_data[:next_cursor])), np.concatenate((train_label[cursor:], train_label[:next_cursor]))
        cursor = next_cursor

batch_size = 64
batch_num = train_data.shape[0] // batch_size
generator = generate_batch(batch_size)

# for i in range(2*batch_num):
#     batch = train_data[batch_size * i: batch_size * (i + 1)]
#     assert np.array_equal(batch, next(generator)[0])
#     data, label = next(generator)
#     assert data.shape[0] == 64

