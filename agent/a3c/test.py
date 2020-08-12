import os
import tensorflow as tf
import numpy as np

from agent.a3c.network import AC_Network
from environment.scheduling import Scheduling
from environment.work import *


if __name__ == '__main__':
    projects = [3095, 3086, 2962]

    window = (10, 40)
    s_shape = (window[0] + 2, window[1])
    a_size = 2

    model_path = '../../model/a3c/%d-%d' % s_shape
    test_path = '../../test/a3c/%d-%d' % s_shape

    if not os.path.exists(test_path):
        os.makedirs(test_path)

    works, max_day = import_schedule('../../environment/data/191227_납기일 추가.xlsx', projects)
    env = Scheduling(works, window)

    tf.reset_default_graph()
    with tf.Session() as sess:
        network = AC_Network(s_shape, a_size, 'global', None)
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver = tf.train.Saver(max_to_keep=5)
        saver.restore(sess, ckpt.model_checkpoint_path)

        s = env.reset()
        episode_frames = []

        while True:
            a_dist, v = sess.run([network.policy, network.value], feed_dict={network.inputs: [s]})
            a = np.argmax(a_dist[0])

            s1, r, d, info = env.step(a)

            if not info:
                s1, r, d, info = env.step(1)

            if not d:
                episode_frames.append(s1)
            else:
                export_schedule(test_path, max_day, works, env.location)
                break

            s = s1
