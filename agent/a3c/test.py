import os
import tensorflow as tf

from agent.a3c.helper import *
from agent.a3c.network import AC_Network
from environment.scheduling import Scheduling
from environment.work import *


if __name__ == '__main__':
    projects = [2962, 3086, 3095]
    inbounds, blocks, days = import_blocks_schedule('../../environment/data/191227_납기일 추가.xlsx', projects, backward=True)
    #average_load = int(sum(work.lead_time for work in inbounds) / days) + 1

    window_days = (40, 10)
    s_shape = (window_days[1], window_days[0])
    a_size = 2

    model_path = '../../model/a3c/%d-%d' % s_shape
    test_path = '../../test/a3c/%d-%d' % s_shape

    if not os.path.exists(test_path):
        os.makedirs(test_path)

    env = Scheduling(num_days=days, window_days=window_days, num_blocks=blocks,
                     inbound_works=inbounds, backward=True, display_env=False)

    tf.reset_default_graph()
    with tf.Session() as sess:
        network = AC_Network(s_shape, a_size, 'global', None)
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver = tf.train.Saver(max_to_keep=5)
        saver.restore(sess, ckpt.model_checkpoint_path)

        s = env.reset()
        episode_frames = []
        rnn_state = network.state_init

        while True:
            a_dist, v, rnn_state = sess.run(
                [network.policy, network.value, network.state_out],
                feed_dict={network.inputs: [s],
                           network.state_in[0]: rnn_state[0],
                           network.state_in[1]: rnn_state[1]})
            a = np.random.choice(a_dist[0], p=a_dist[0])
            a = np.argmax(a_dist == a)

            s1, r, d = env.step(a)

            if not d:
                episode_frames.append(s1)
            else:
                export_blocks_schedule(test_path, env.inbound_works, blocks, days)
                break

            s = s1
