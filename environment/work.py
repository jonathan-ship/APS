import pandas as pd


def import_schedule(filepath, projects, backward=True):
    df_all = pd.read_excel(filepath)
    df_selected_projects = df_all[df_all['호선'].isin(projects)]
    df_schedule = df_selected_projects[df_selected_projects['공정'].isin([4, 6, 7, 8])]

    relation = pd.DataFrame(columns=['relation', 'activityA', 'activityB'])
    relation.loc[0] = ['FS', 6, 4]
    relation.loc[1] = ['FS', 7, 4]
    relation.loc[3] = ['FF', 8, 4]

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

    days = df_schedule['납기일'].max()
    columns = ['activity_code', 'project', 'block', 'block_group', 'process', 'start_plan', 'finish_plan', 'lead_time',
               'work_load', 'weight', 'block_stage', 'latest_finish', 'earliest_start', 'start', 'block_group_idx']
    schedule = pd.DataFrame(columns=columns)

    num_of_block_group = 0
    work_idx = 0
    while len(df_schedule) != 0:
        first_row = df_schedule.loc[0]
        block_group = df_schedule[(df_schedule['호선'] == first_row['호선'])
                                  & (df_schedule['블록그룹'] == first_row['블록그룹'])]
        num_of_activities = len(block_group)

        while len(block_group) != 0:
            activity = list(block_group.loc[0])
            activity.extend([-1, None, num_of_block_group])
            schedule.loc[work_idx] = activity
            if len(block_group) == 1 or block_group.loc[0]['계획착수일'] != block_group.loc[1]['계획착수일']:
                block_group.drop([0], inplace=True)
            else:
                schedule.loc[work_idx, 'activity_code'] = schedule.loc[work_idx, 'activity_code'] \
                                                          + block_group.loc[1, '액티비티코드'][-4:]
                schedule.loc[work_idx, 'work_load'] = schedule.loc[work_idx, 'work_load'] \
                                                      + block_group.loc[1, '계획공수']
                schedule.loc[work_idx, 'weight'] = schedule.loc[work_idx, 'weight'] + block_group.loc[1, '중량']
                block_group.drop([0, 1], inplace=True)
            block_group.reset_index(drop=True, inplace=True)
            work_idx += 1
        df_schedule.drop([_ for _ in range(num_of_activities)], inplace=True)
        df_schedule.reset_index(drop=True, inplace=True)
        num_of_block_group += 1

    #relation_FS = relation[relation['relation'] == 'FS']
    #for i in range(num_of_block_group):
        #block_group = schedule[schedule['block_group'] == i]
        #if backward:
            #for j, row in relation_FS.iterrows():
                #block_group[block_group['process'].isin(row['activityB'])]

    return initial_date, num_of_block_group, days, schedule, relation


if __name__ == '__main__':
    a, b, c, inbound, relation = import_schedule('../environment/data/191227_납기일 추가.xlsx', [2962], backward=True)
    print(inbound)
