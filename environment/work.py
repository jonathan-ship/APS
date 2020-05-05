import scipy
import copy
import pandas as pd
import numpy as np

from agent.a3c.helper import color_frame_continuous
from PIL import Image, ImageDraw, ImageFont

class Work(object):
    def __init__(self, work_id=None, block=None, block_stage=None, work_load=None, initial_start=None, initial_finish=None, lead_time=1, earliest_start=-1, latest_finish=-1):
        self.work_id = str(work_id)
        self.block = block
        self.block_stage = block_stage
        self.work_load = work_load
        self.initial_start = initial_start
        self.initial_finish = initial_finish
        self.rl_start = None
        self.rl_finish = None
        self.lead_time = lead_time
        self.work_load_per_day = work_load/lead_time
        self.earliest_start = earliest_start
        self.latest_finish = latest_finish

    def set_constraint(self, inbound_works, locations, backward=True):
        if backward:
            update_latest_finish = self.latest_finish
            for work, location in zip(inbound_works, locations):
                if self.block == work.block and self.block_stage != work.block_stage:
                    if location < update_latest_finish:
                        update_latest_finish = location
            self.latest_finish = update_latest_finish
        else:
            update_earlist_start = self.earliest_start
            for work, location in zip(inbound_works, locations):
                if self.block == work.block and self.block_stage != work.block_stage:
                    if location + work.lead_time - 1 > update_earlist_start:
                        update_earlist_start = location + work.lead_time - 1
            self.earliest_start = update_earlist_start

    def set_location(self, location):
        self.rl_start = location
        self.rl_finish = self.rl_start + self.lead_time - 1


def import_blocks_schedule(filepath, projects, backward=True):

    df_all = pd.read_excel(filepath)
    df_temp = df_all[df_all['호선'].isin(projects)]
    df_schedule = df_temp[df_temp['공정'].isin([4, 6, 7, 8])]

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

    works = []
    block = 0
    max_days = df_schedule['납기일'].max()

    while len(df_schedule) != 0:
        first_row = df_schedule.loc[0]
        temp = df_schedule[(df_schedule['호선'] == first_row['호선']) & (df_schedule['블록그룹'] == first_row['블록그룹'])]
        block_num = len(temp)

        while len(temp) != 0:
            if len(temp) == 1:
                works.append(Work(work_id=temp.loc[0]['액티비티코드'],
                                  block=block,
                                  block_stage=temp.loc[0]['블록단계'],
                                  work_load=temp.loc[0]['계획공수'],
                                  lead_time=temp.loc[0]['계획완료일'] - temp.loc[0]['계획착수일'] + 1,
                                  initial_start=temp.loc[0]['계획착수일'],
                                  initial_finish=temp.loc[0]['계획완료일'],
                                  latest_finish=temp.loc[0]['납기일']))
                temp.drop([0], inplace=True)
                temp.reset_index(drop=True, inplace=True)
            else:
                if temp.loc[0]['계획착수일'] != temp.loc[1]['계획착수일']:
                    works.append(Work(work_id=temp.loc[0]['액티비티코드'],
                                      block=block,
                                      block_stage=temp.loc[0]['블록단계'],
                                      work_load=temp.loc[0]['계획공수'],
                                      lead_time=temp.loc[0]['계획완료일'] - temp.loc[0]['계획착수일'] + 1,
                                      initial_start=temp.loc[0]['계획착수일'],
                                      initial_finish=temp.loc[0]['계획완료일'],
                                      latest_finish=temp.loc[0]['납기일']))
                    temp.drop([0], inplace=True)
                    temp.reset_index(drop=True, inplace=True)
                else:
                    works.append(Work(work_id=temp.loc[0]['액티비티코드'] + temp.loc[1]['액티비티코드'][-4:],
                                      block=block,
                                      block_stage=temp.loc[0]['블록단계'],
                                      work_load=temp.loc[0]['계획공수'] + temp.loc[1]['계획공수'],
                                      lead_time=temp.loc[0]['계획완료일'] - temp.loc[0]['계획착수일'] + 1,
                                      initial_start=temp.loc[0]['계획착수일'],
                                      initial_finish=temp.loc[0]['계획완료일'],
                                      latest_finish=temp.loc[0]['납기일']))
                    temp.drop([0, 1], inplace=True)
                    temp.reset_index(drop=True, inplace=True)

        df_schedule.drop([_ for _ in range(block_num)], inplace=True)
        df_schedule.reset_index(drop=True, inplace=True)
        block += 1

    return works, block, max_days


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
    inbounds, blocks, days = import_blocks_schedule('../environment/data/191227_납기일 추가.xlsx', [3095], backward=True)
    print(inbounds[0].block)
    print(blocks)
    print(days)