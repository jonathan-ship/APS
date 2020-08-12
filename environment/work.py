import scipy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import OrderedDict
from PIL import Image


def set_process_sequence(relation):
    relation = relation.sort_values(by=["relation"], ascending=False)
    process_sequence = []
    for i, row in relation.iterrows():
        if (not row['activityA'] in process_sequence) and (not row['activityB'] in process_sequence):
            process_sequence.extend([row['activityA'], row['activityB']])
        elif (not row['activityA'] in process_sequence) and (row['activityB'] in process_sequence):
            idx = process_sequence.index(row['activityB'])
            process_sequence.insert(idx, row['activityA'])
        elif (row['activityA'] in process_sequence) and (not row['activityB'] in process_sequence):
            idx = process_sequence.index(row['activityA'])
            process_sequence.insert(idx + 1, row['activityB'])
    return process_sequence


def import_schedule(filepath, projects):
    df_all = pd.read_excel(filepath)
    df_selected_projects = df_all[df_all['호선'].isin(projects)]
    df_schedule = df_selected_projects[df_selected_projects['공정'].isin([4, 6, 7, 8])]

    relation = pd.DataFrame(columns=['relation', 'activityA', 'activityB'])
    relation.loc[0] = ['FS', 6, 4]
    relation.loc[1] = ['FS', 7, 4]
    relation.loc[2] = ['FF', 8, 4]
    process_seq = set_process_sequence(relation)
    mapping = dict()
    for i in range(len(process_seq)):
        mapping[process_seq[i]] = len(process_seq) - 1 - i

    df_schedule = df_schedule.sort_values(by=['납기일', '호선', '블록그룹'], ascending=False)
    df_schedule = df_schedule.reset_index(drop=True)

    df_schedule['납기일'] = pd.to_datetime(df_schedule['납기일'], format='%Y%m%d')
    df_schedule['계획착수일'] = pd.to_datetime(df_schedule['계획착수일'], format='%Y%m%d')
    df_schedule['계획완료일'] = pd.to_datetime(df_schedule['계획완료일'], format='%Y%m%d')
    max_day = df_schedule['납기일'].max()
    df_schedule['납기일'] = (df_schedule['납기일'] - max_day).dt.days - 1
    df_schedule['계획착수일'] = (df_schedule['계획착수일'] - max_day).dt.days - 1
    df_schedule['계획완료일'] = (df_schedule['계획완료일'] - max_day).dt.days - 1

    works = OrderedDict()
    block_group_idx = 0
    while len(df_schedule) != 0:
        first_row = df_schedule.loc[0]
        df_block_group = df_schedule[(df_schedule['호선'] == first_row['호선'])
                                     & (df_schedule['블록그룹'] == first_row['블록그룹'])]
        block_group = []
        block_group_work_id = []
        block_group_process = dict()
        num_of_work = len(df_block_group)

        for i, work_1 in df_block_group.iterrows():
            append = False
            idx_s = work_1['액티비티코드'][-4:].find('S')
            idx_p = work_1['액티비티코드'][-4:].find('P')
            if idx_s != -1 or idx_p != -1:
                if idx_s != -1:
                    target_work_id = (work_1['액티비티코드'][-4:])[:idx_s] + 'P' + (work_1['액티비티코드'][-4:])[idx_s + 1:]
                else:
                    target_work_id = (work_1['액티비티코드'][-4:])[:idx_p] + 'S' + (work_1['액티비티코드'][-4:])[idx_p + 1:]
                target_work_id = work_1['액티비티코드'][:-4] + target_work_id
                work_2 = df_block_group[df_block_group['액티비티코드'] == target_work_id]
                if len(work_2) != 0:
                    if idx_s != -1:
                        work_id = (work_2.iloc[0])['액티비티코드'] + work_1['액티비티코드'][-4:]
                    else:
                        work_id = work_1['액티비티코드'] + (work_2.iloc[0])['액티비티코드'][-4:]
                    if not work_id in block_group_work_id:
                        block_group_work_id.append(work_id)
                        block_group.append(Work(work_id=work_id,
                                                block=block_group_idx,
                                                process=work_1['공정'],
                                                start_planned=work_1['계획착수일'],
                                                finish_planned=work_1['계획완료일'],
                                                lead_time=work_1['계획완료일'] - work_1['계획착수일'] + 1,
                                                work_load=work_1['계획공수'] + work_2.iloc[0]['계획공수'],
                                                latest_finish=work_1['납기일']))
                else:
                    append = True
            if (idx_s == -1 and idx_p == -1) or append:
                block_group_work_id.append(work_1['액티비티코드'])
                block_group.append(Work(work_id=work_1['액티비티코드'],
                                        block=block_group_idx,
                                        process=work_1['공정'],
                                        start_planned=work_1['계획착수일'],
                                        finish_planned=work_1['계획완료일'],
                                        lead_time=work_1['계획완료일'] - work_1['계획착수일'] + 1,
                                        work_load=work_1['계획공수'],
                                        latest_finish=work_1['납기일']))
            if block_group[-1].process in list(block_group_process.keys()):
                if not block_group[-1].work_id in block_group_process[block_group[-1].process]:
                    block_group_process[block_group[-1].process].append(block_group[-1].work_id)
            else:
                block_group_process[block_group[-1].process] = [block_group[-1].work_id]
        block_group.sort(key=lambda x: mapping[x.process])
        for i in range(len(block_group)):
            related_activities = relation[relation['activityB'] == block_group[i].process]
            for j, row in related_activities.iterrows():
                if block_group_process.get(row['activityA']):
                    if not block_group[i].relation.get(row['relation']):
                        block_group[i].relation[row['relation']] = block_group_process.get(row['activityA'])
                    else:
                        block_group[i].relation[row['relation']].extend(block_group_process.get(row['activityA']))
        for work in block_group:
            works[work.work_id] = work
        df_schedule.drop([_ for _ in range(num_of_work)], inplace=True)
        df_schedule.reset_index(drop=True, inplace=True)
        block_group_idx += 1
    return works, max_day


def save_image(filepath, image):
    colored_image = np.zeros([image.shape[0], image.shape[1], 3])
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] == 0.0:
                colored_image[i, j] = [0, 0, 0]
            elif image[i, j] == -1.0:
                colored_image[i, j] = [255, 0, 0]
            else:
                grey = max(0, 255 - 0.005 * 255 * image[i, j])
                colored_image[i, j] = [grey, grey, grey]
    big_image = scipy.misc.imresize(colored_image, [image.shape[0] * 30, image.shape[1] * 30], interp='nearest')
    final_image = Image.fromarray(big_image.astype('uint8'), 'RGB')
    final_image.save(filepath)


def save_graph(filepath, loads):
    day = range(len(loads))
    idx = np.where(loads != 0.0)[0]
    load_mean = np.full([len(loads)], np.float(np.mean(loads[idx[0]:idx[-1] + 1])))
    deviation = np.std(loads[idx[0]:idx[-1] + 1])
    fig, ax = plt.subplots()
    ax.plot(day, loads, color='dodgerblue', label='load per day')
    ax.plot(day, load_mean, color='firebrick', label='average load per day')
    ax.text(len(day) * 0.01, max(loads[idx[0]:idx[-1] + 1]) * 1.2, 'deviation: {0:0.2f}'.format(deviation))
    ax.set_ylim([0, max(loads[idx[0]:idx[-1] + 1]) * 1.3])
    ax.legend(loc='upper right')
    ax.set_xlabel('day')
    ax.set_ylabel('load per day')
    ax.set_title('loads')
    fig.savefig(filepath, bbox_inches='tight', pad_inches=0.5, edgecolor='grey')


def export_schedule(filepath, max_day, works, locations):

    df_schedule = pd.DataFrame(columns=['액티비티코드', '계획공기', '계획공수', '계획착수일_초기', '계획완료일_초기',
                                        '계획착수일_학습', '계획완료일_학습', '납기일'])

    for i, work in enumerate(works.values()):
        df_schedule.loc[i] = [work.work_id, work.lead_time, work.work_load,
                              (max_day - pd.Timedelta(days=-(work.start_planned + 1))).date(),
                              (max_day - pd.Timedelta(days=-(work.finish_planned + 1))).date(),
                              (max_day - pd.Timedelta(days=-(locations[work.work_id] - work.lead_time + 2))).date(),
                              (max_day - pd.Timedelta(days=-(locations[work.work_id] + 1))).date(),
                              (max_day - pd.Timedelta(days=-(work.latest_finish + 1))).date()]

    df_schedule.to_excel(filepath + "/output.xlsx")

    row = list(works.values())[-1].block + 1
    col = - min([locations[work.work_id] - work.lead_time + 1 for work in works.values()])
    col = max([col, -list(works.values())[-1].start_planned])
    state_initial = np.full([row, col], 0.0)
    state_learning = np.full([row, col], 0.0)

    for i, work in enumerate(works.values()):
        state_initial[work.block, work.latest_finish] = -1
        state_initial[work.block, work.start_planned:work.finish_planned + 1] += work.work_load / work.lead_time
        state_learning[work.block, work.latest_finish] = -1
        state_learning[work.block, locations[work.work_id] - work.lead_time + 1:locations[work.work_id] + 1] \
            += work.work_load / work.lead_time

    save_image(filepath + '/gantt_initial.png', state_initial)
    save_image(filepath + '/gantt_learning.png', state_learning)

    state_initial[state_initial == -1] = 0.0
    state_learning[state_learning == -1] = 0.0

    save_graph(filepath + '/loads_initial.png', np.sum(state_initial, axis=0))
    save_graph(filepath + '/loads_learning.png', np.sum(state_learning, axis=0))


class Work:
    def __init__(self, work_id=None, block=None, process=None, start_planned=None, finish_planned=None,
                 lead_time=1, work_load=1, earliest_start=None, latest_finish=None):
        self.work_id = str(work_id)
        self.block = block
        self.process = process
        self.start_planned = start_planned
        self.finish_planned = finish_planned
        self.lead_time = lead_time
        self.work_load = work_load
        self.earliest_start = earliest_start
        self.latest_finish = latest_finish
        self.relation = dict()


if __name__ == '__main__':
    inbound, max_day = import_schedule('../environment/data/191227_납기일 추가.xlsx', [3095])
    print((max_day - pd.Timedelta(days=-(list(inbound.values())[0].finish_planned+1))).date())
    for i in inbound.values():
        print("{0}|{1}: {2}, {3}, {4}".format(i.block, i.work_id, i.work_load/i.lead_time, i.process, i.relation))
    print(len(inbound))

    save_graph('../test/a3c/10-40' + '/loads_learning.png', np.array([1, 2, 3, 5, 7, 0, 0]))

