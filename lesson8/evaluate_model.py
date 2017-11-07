import tensorflow as tf
import dnn_d_vector as d_v
import data_process as dp
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

test_data = dp.get_test_data()

ckpt = tf.train.get_checkpoint_state('./model')
X = tf.placeholder(tf.float32, [None, 40])

ds = d_v.d_vector_model(X)
# d_vector = tf.reduce_mean(d_v.d_vector_model(X), 0)

saver = tf.train.Saver()
with tf.Session() as session:
    if ckpt != None:
        print(ckpt.model_checkpoint_path)
        saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("no model")
        exit()

    representation_data = [data[0] for data in test_data]
    # print(len(representation_data))
    representation_vector = []
    print(len(representation_data))
    for data in representation_data:
        # print(data)
        [vector] = session.run([ds], feed_dict={X: data, d_v.train_phase: False})
        representation_vector.append(np.mean(vector, axis=0))
    # the representaion speaker matrix shape is (speaker_num, 256). 
    # 256 is dim of d-vector
    representation_speaker_matrix = np.array(representation_vector)

    correct_num = 0
    test_count = 0
    for index, speaker_data in enumerate(test_data):
        for test_utterance in speaker_data[1:]:
            [test_vector] = session.run([ds], feed_dict={X: test_utterance, d_v.train_phase: False})
            # the shape of cos_sim is (speaker_num, test_frame_num)
            cos_sim = cosine_similarity(representation_vector, test_vector)

            predicts = np.argmax(cos_sim, axis=0)
            mode, count = stats.mode(predicts)
            correct_num += (mode[0] == index and count[0] > 20)
            test_count += 1

    print(correct_num / test_count)
