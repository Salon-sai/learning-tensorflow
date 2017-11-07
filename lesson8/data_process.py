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

def load_data(path):
    data = []
    label_index = np.array([], dtype=int)
    label_count = 0
    wav_files_count = 0

    for root, dirs, files in os.walk(path):
        # get all wav files in current dir 
        wav_files = [file for file in files if file.endswith('.wav')]
        data_same_person = []
        # extract logfbank features from wav file
        for wav_file in wav_files:
            (rate, sig) = wav.read(root + "/" + wav_file)
            fbank_beats = logfbank(sig, rate, nfilt=40)
            # save logfbank features into same person array
            data_same_person.append(fbank_beats)

        # save all data of same person into the data array
        # the length of data array is number of speakers
        if wav_files:
            wav_files_count += len(wav_files)
            data.append(data_same_person)

    # return data, np.arange(len(data))
    return data

def load_data_with_wavfile(path):
    # data, label_index = load_data(path)
    data = load_data(path)
    datas = []
    labels = []
    for index, data_same_person in enumerate(data):
        labels.append(np.full(len(data_same_person), index))
        datas += data_same_person
    labels = np.concatenate(labels)
    one_hot_label = np.zeros((len(datas), len(data)))
    one_hot_label[np.arange(len(datas)), labels] = 1
    
    return datas, one_hot_label

def load_train_data():
    train_data = []
    label_index = []
    datas = load_data(TRAIN_PATH)

    for label, data  in enumerate(datas):
        data = np.concatenate(data)
        frame_num = data.shape[0]

        train_data.append(data)
        label_index.append(np.full(frame_num, label))
    train_data = np.concatenate(train_data)
    label_index = np.concatenate(label_index)
    train_label = np.zeros((train_data.shape[0], label + 1))
    train_label[np.arange(train_data.shape[0]), label_index] = 1
    return train_data, train_label

def shuffle_data(data, labels):
    num, _ = data.shape
    shuffle_index = np.arange(num)
    np.random.shuffle(shuffle_index)
    return data[shuffle_index], labels[shuffle_index]

def get_train_data():
    train_data, train_label = load_train_data()
    train_data, train_label = shuffle_data(train_data, train_label)
    return train_data, train_label

def get_test_data():
    return load_data(TEST_PATH)

def generate_batch(batch_size, data, label):
    train_num = data.shape[0]
    cursor = 0
    while True:
        next_cursor = cursor + batch_size
        if next_cursor < train_num - 1:
            yield data[cursor: next_cursor], label[cursor: next_cursor]
        else:
            next_cursor %= train_num
            yield np.concatenate((data[cursor:], data[:next_cursor])), np.concatenate((label[cursor:], label[:next_cursor]))
        cursor = next_cursor

# batch_size = 64
# batch_num = train_data.shape[0] // batch_size
# generator = generate_batch(batch_size)

# for i in range(2*batch_num):
#     batch = train_data[batch_size * i: batch_size * (i + 1)]
#     assert np.array_equal(batch, next(generator)[0])
#     data, label = next(generator)
#     assert data.shape[0] == 64

