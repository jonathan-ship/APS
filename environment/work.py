import scipy
import copy
import pandas as pd
import numpy as np

from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from agent.a3c.helper import color_frame_continuous
from PIL import Image, ImageDraw, ImageFont


class Work(object):
    def __init__(self, work_id=None, block_group_idx=None, work_load=None, lead_time=None, start=None, earliest_start=-1, latest_finish=-1):
        self.work_id = str(work_id)
        self.block_group_idx = int(block_group_idx)
        self.work_load = work_load
        self.lead_time = lead_time
        self.start = start
        self.earliest_start = earliest_start
        self.latest_finish = latest_finish
        self.work_load_per_day = work_load/lead_time


class SchedulingManager(object):
    def __init__(self, filepath, projects, backward=True, initial_plan=False):
        self.relation = pd.DataFrame(columns=['relation', 'activityA', 'activityB'])
        self.backward = backward
        self.initial_plan = initial_plan
        self.initial_date, self.num_block_group, self.num_day, \
            self.schedule = self._import_schedule(filepath, projects, backward=backward)
        self.works = self._import_inbound_works(initial_plan=initial_plan)
        self.num_work = len(self.works)
        self.target_load_per_day = sum(work.work_load for work in self.works) / self.num_day

    def _import_schedule(self, filepath, projects, backward=True):
        df_all = pd.read_excel(filepath)
        df_selected_projects = df_all[df_all['호선'].isin(projects)]
        df_schedule = df_selected_projects[df_selected_projects['공정'].isin([4, 6, 7, 8])]
        self.relation.loc[0] = ['FS', 6, 4]
        self.relation.loc[1] = ['FS', 7, 4]
        self.relation.loc[3] = ['FF', 8, 4]

        if backward:
            df_schedule.sort_values(by=['납기일', '호선', '블록그룹', '계획착수일', '블록'], inplace=True, ascending=False)
        else:
            df_schedule.sort_values(by=['납기일', '호선', '블록그룹', '계획착수일', '블록'], inplace=True)
        df_schedule.reset_index(drop=True, inplace=True)

        df_schedule['계획착수일'] = pd.to_datetime(df_schedule['계획착수일'], format='%Y%m%d')
        df_schedule['계획완료일'] = pd.to_datetime(df_schedule['계획완료일'], format='%Y%m%d')
        df_schedule['납기일'] = pd.to_datetime(df_schedule['납기일'], format='%Y%m%d')
        initial_date = df_schedule['계획착수일'].min()
        df_schedule['계획착수일'] = (df_schedule['계획착수일'] - initial_date).dt.days
        df_schedule['계획완료일'] = (df_schedule['계획완료일'] - initial_date).dt.days
        df_schedule['납기일'] = (df_schedule['납기일'] - initial_date).dt.days

        block_group_idx = 0
        days = df_schedule['납기일'].max()
        schedule = pd.DataFrame(columns=df_schedule.columns)

        work_idx = 0
        while len(df_schedule) != 0:
            first_row = df_schedule.loc[0]
            block_group = df_schedule[(df_schedule['호선'] == first_row['호선']) & (df_schedule['블록그룹'] == first_row['블록그룹'])]
            activity_num = len(block_group)

            while len(block_group) != 0:
                schedule.loc[work_idx] = block_group.loc[0]
                schedule.loc[work_idx, '블록그룹번호'] = block_group_idx
                if len(block_group) == 1 or block_group.loc[0]['계획착수일'] != block_group.loc[1]['계획착수일']:
                    block_group.drop([0], inplace=True)
                else:
                    schedule.loc[work_idx, '액티비티코드'] = schedule.loc[work_idx, '액티비티코드'] + block_group.loc[1, '액티비티코드'][-4:]
                    schedule.loc[work_idx, '계획공수'] = schedule.loc[work_idx, '계획공수'] + block_group.loc[1, '계획공수']
                    block_group.drop([0, 1], inplace=True)
                block_group.reset_index(drop=True, inplace=True)
                work_idx += 1
            df_schedule.drop([_ for _ in range(activity_num)], inplace=True)
            df_schedule.reset_index(drop=True, inplace=True)
            block_group_idx += 1

        return initial_date, block_group_idx, days, schedule

    def _import_inbound_works(self, initial_plan=False):
        works = []
        for i, row in self.schedule.iterrows():
            works.append(Work(work_id=row['액티비티코드'],
                              block_group_idx=row['블록그룹번호'],
                              work_load=row['계획공수'],
                              lead_time=row['계획완료일'] - row['계획착수일'] + 1,
                              start=row['계획착수일'] if initial_plan else None,
                              earliest_start=0,
                              latest_finish=row['납기일']))
        return works

    def export_schedule(self, file_path):
        schedule_initial = np.full([self.num_block_group, self.num_day], 0)
        schedule_rl = np.full([self.num_block_group, self.num_day], 0)

        for i, work in enumerate(self.works):
            schedule_initial[work.block_group_idx, self.schedule['계획착수일'][i]:(self.schedule['계획완료일'][i] + 1)] = work.work_load_per_day
            schedule_rl[work.block_group_idx, work.start:(work.start + work.lead_time)] = work.work_load_per_day

        loads_inital = np.sum(schedule_initial, axis=0)
        loads_rl = np.sum(schedule_rl, axis=0)
        index_initial = (np.where(loads_inital != 0))[0]
        index_rl = (np.where(loads_rl != 0))[0]

        deviation_inital = np.std(loads_inital[index_initial[0]:(index_initial[-1] + 1)])
        deviation_rl = np.std(loads_rl[index_rl[0]:(index_rl[-1] + 1)])

        image_initial = scipy.misc.imresize(color_frame_continuous(np.array([schedule_initial]))[0],
                                            [self.num_block_group * 30, self.num_day * 30], interp='nearest')
        image_rl = scipy.misc.imresize(color_frame_continuous(np.array([schedule_rl]))[0],
                                       [self.num_block_group * 30, self.num_day * 30], interp='nearest')

        image_inital = Image.fromarray(image_initial.astype('uint8'), 'RGB')
        image_rl = Image.fromarray(image_rl.astype('uint8'), 'RGB')

        full_width = self.num_day * 30 + 2 * 30
        full_height = self.num_block_group * 30 * 2 + 3 * 10 * 30

        image = Image.new('RGB', (full_width, full_height), 'white')
        image.paste(im=image_inital, box=(30, 10 * 30))
        image.paste(im=image_rl, box=(30, 2 * 10 * 30 + self.num_block_group * 30))

        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype('arial', 150)
        draw.text((30, 3 * 30), 'scheule_inital --- deviation: {:.2f}'.format(deviation_inital), (0, 0, 0), font)
        draw.text((30, 13 * 30 + self.num_block_group * 30), 'schedule_rl --- deviation: {:.2f}'.format(deviation_rl), (0, 0, 0), font)

        image.save(file_path + '/result.png')
        #schedule = Workbook()
        #initial_plan = schedule.active
        #for row in dataframe_to_rows(self.initial_plan, header=True):
            #initial_plan.append(row)

    def set_constraint(self, ongoing):
        ongoing_block_group = self.schedule[self.schedule['블록그룹'] == self.schedule.loc[ongoing, '블록그룹']]
        for i, row in self.relation.iterrows():
            if self.schedule.loc[ongoing, '공정'] == row['activityA']:
                indices = ongoing_block_group[ongoing_block_group['공정'] == row['activityB']].index.values
                for idx in indices:
                    if row['relation'] == 'FS':
                        self.works[idx].earliest_start = self.works[ongoing].start + self.works[ongoing].lead_time
                    elif row['relation'] == 'FF':
                        self.works[idx].earliest_start = self.works[ongoing].start + self.works[ongoing].lead_time - self.works[idx].lead_time
            elif self.schedule.loc[ongoing, '공정'] == row['activityB']:
                indices = ongoing_block_group[ongoing_block_group['공정'] == row['activityA']].index.values
                for idx in indices:
                    if row['relation'] == 'FS':
                        self.works[idx].latest_finish = self.works[ongoing].start - 1
                    elif row['relation'] == 'FF':
                        self.works[idx].latest_finish = self.works[ongoing].start + self.works[ongoing].lead_time - 1

    def reset_schedule(self):
        for i in range(len(self.schedule)):
            self.works[i].earliest_start = 0
            self.works[i].latest_finish = self.schedule.loc[i, '납기일']
            if self.initial_plan:
                self.works[i].start = self.schedule.loc[i, '계획착수일']
            else:
                if self.backward:
                    self.works[i].start = self.works[i].latest_finish - self.works[i].lead_time + 1 if i == 0 else None
                else:
                    self.works[i].start = 0 if i == 0 else None


def export_blocks_schedule(file_path, inbound_works, block, max_day):
    schedule_initial = np.full([block, max_day], 0)
    schedule_rl = np.full([block, max_day], 0)

    for work in inbound_works:
        schedule_initial[work.block, work.initial_start:(work.initial_finish + 1)] = work.work_load_per_day
        schedule_rl[work.block, work.rl_start:(work.rl_finish + 1)] = work.work_load_per_day

    loads_inital = np.sum(schedule_initial, axis=0)
    loads_rl = np.sum(schedule_rl, axis=0)
    index_initial = (np.where(loads_inital != 0))[0]
    index_rl = (np.where(loads_rl != 0))[0]

    deviation_inital = np.std(loads_inital[index_initial[0]:(index_initial[-1] + 1)])
    deviation_rl = np.std(loads_rl[index_rl[0]:(index_rl[-1] + 1)])

    image_initial = scipy.misc.imresize(color_frame_continuous(np.array([schedule_initial]))[0], [block * 30, max_day * 30], interp='nearest')
    image_rl = scipy.misc.imresize(color_frame_continuous(np.array([schedule_rl]))[0], [block * 30, max_day * 30], interp='nearest')

    image_inital = Image.fromarray(image_initial.astype('uint8'), 'RGB')
    image_rl = Image.fromarray(image_rl.astype('uint8'), 'RGB')

    full_width = max_day * 30 + 2 * 30
    full_height = block * 30 * 2 + 3 * 10 * 30

    image = Image.new('RGB', (full_width, full_height), 'white')
    image.paste(im=image_inital, box=(30, 10 * 30))
    image.paste(im=image_rl, box=(30, 2 * 10 * 30 + block * 30))

    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('arial', 150)
    draw.text((30, 3 * 30),'scheule_inital --- deviation: {:.2f}'.format(deviation_inital), (0, 0, 0), font)
    draw.text((30, 13 * 30 + block * 30),'schedule_rl --- deviation: {:.2f}'.format(deviation_rl), (0, 0, 0), font)

    image.save(file_path + '/result.png')


if __name__ == '__main__':
    scheduling_manager = SchedulingManager('../environment/data/191227_납기일 추가.xlsx', [2962], backward=True)
    ongoing_block_group = scheduling_manager.schedule[scheduling_manager.schedule['블록그룹'] == scheduling_manager.schedule.loc[9, '블록그룹']]
    print(scheduling_manager.schedule.index)