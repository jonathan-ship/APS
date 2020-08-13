import copy
import numpy as np


class Scheduling(object):
    def __init__(self, inbound_works=None, window=(10, 40), margin=10):
        self.action_space = 2
        self.num_of_blocks = window[0]
        self.num_of_days = window[1]
        self.inbound_works = inbound_works
        self.inbound_works_clone = copy.copy(inbound_works)
        self.margin = margin
        self.deviation = 0.0
        self.stage = 0
        self._ongoing = 0
        self.move = True
        self.locations_updated = dict()
        self.locations = dict()
        self.constraint = dict()
        self.left_action = 0
        self.select_action = 1

    def step(self, action):
        info = True
        done = False
        reward = 0
        self.stage += 1
        work = list(self.inbound_works.values())[self._ongoing]
        if action == self.select_action:
            reward = self._calculate_reward_by_deviation()
            self._set_constraint(work)
            self._ongoing += 1
            if self._ongoing == len(self.inbound_works):
                done = True
        else:
            margin = self.margin * 3 if self.constraint[work.work_id][1] == work.latest_finish else self.margin
            if self.locations[work.work_id] > self.constraint[work.work_id][1] - margin - 1:
                if not self.constraint[work.work_id][0]:
                    self.locations_updated[work.work_id] = self.locations[work.work_id] - 1
                    self._update_location(work)
                else:
                    if self.locations[work.work_id] - work.lead_time > self.constraint[work.work_id][0]:
                        self.locations_updated[work.work_id] = self.locations[work.work_id] - 1
                        self._update_location(work)
                    else:
                        self.move = False
            else:
                self.move = False
            info = self.move
            if self.move:
                for (key, value) in self.locations_updated.items():
                    self.locations[key] = value
            self.move = True
            self.locations_updated = dict()
        next_state = self._get_state().flatten()
        return next_state, reward, done, info

    def reset(self):
        self.inbound_works = copy.copy(self.inbound_works_clone)
        self.deviation = 0.0
        self.stage = 0
        self._ongoing = 0
        self.locations = dict()
        for work in self.inbound_works.values():
            if not self.locations.get(work.work_id):
                self.locations[work.work_id] = work.latest_finish - 1
            for key in work.relation.keys():
                if key == 'FS':
                    for related_work_id in work.relation[key]:
                        self.locations[related_work_id] = work.latest_finish - work.lead_time - 1
                elif key == 'FF':
                    for related_work_id in work.relation[key]:
                        self.locations[related_work_id] = work.latest_finish - 1
        self.constraint = dict()
        for work in self.inbound_works.values():
            self.constraint[work.work_id] = [None, work.latest_finish]
        return self._get_state().flatten()

    def _get_state(self):
        row = list(self.inbound_works.values())[-1].block + 1
        col = - min([self.locations[work.work_id] - work.lead_time + 1 for work in self.inbound_works.values()])
        state = np.full([row, col], 0.0)
        ongoing_state = np.full([1, col], 0.0)
        total_work_load_state = np.full([1, col], 0.0)
        ongoing_location = self.locations[list(self.inbound_works.values())[-1].work_id]
        ongoing_block = list(self.inbound_works.values())[-1].block
        for i, work in enumerate(self.inbound_works.values()):
            state[work.block, work.latest_finish] = -1
            state[work.block, self.locations[work.work_id] - work.lead_time + 1:self.locations[work.work_id] + 1] \
                += work.work_load / work.lead_time
            total_work_load_state[0, self.locations[work.work_id] - work.lead_time + 1:self.locations[work.work_id] + 1] \
                += work.work_load / work.lead_time
            if self._ongoing == i:
                ongoing_state[0, self.locations[work.work_id] - work.lead_time + 1:self.locations[work.work_id] + 1] \
                    += work.work_load / work.lead_time
                ongoing_location = self.locations[work.work_id]
                ongoing_block = work.block
        idx = np.where(total_work_load_state != 0.0)[0]
        self.deviation = np.std(total_work_load_state[idx[0]:idx[-1] + 1])
        right = min(0, int(ongoing_location + self.num_of_days / 2))
        right = max(right, - col + self.num_of_days)
        top = max(0, int(ongoing_block - self.num_of_blocks / 2))
        top = min(top, row - self.num_of_blocks)
        if right == 0:
            state = state[top:top + self.num_of_blocks, right - self.num_of_days:]
            ongoing_state = ongoing_state[:, right - self.num_of_days:]
            total_work_load_state = total_work_load_state[:, right - self.num_of_days:]
        else:
            state = state[top:top + self.num_of_blocks, right - self.num_of_days:right]
            ongoing_state = ongoing_state[:, right - self.num_of_days:right]
            total_work_load_state = total_work_load_state[:, right - self.num_of_days:right]
        state = np.concatenate((ongoing_state, state, total_work_load_state), axis=0)
        return state

    def _update_location(self, work):
        if not work.relation.keys():
            if not self.constraint[work.work_id][0]:
                self.locations_updated[work.work_id] = self.locations[work.work_id] - 1
            else:
                if self.locations[work.work_id] - work.lead_time > self.constraint[work.work_id][0]:
                    self.locations_updated[work.work_id] = self.locations[work.work_id] - 1
                else:
                    self.move = False
            return
        else:
            for key in work.relation.keys():
                if key == 'FS':
                    for related_work_id in work.relation[key]:
                        self.locations_updated[related_work_id] = self.locations[related_work_id] - 1
                        self._update_location(self.inbound_works[related_work_id])
                elif key == 'FF':
                    for related_work_id in work.relation[key]:
                        self.locations_updated[related_work_id] = self.locations[related_work_id] - 1

    def _set_constraint(self, current_work):
        for key in current_work.relation.keys():
            if key == 'FS':
                for related_work in current_work.relation[key]:
                    self.constraint[related_work][1] = self.locations[current_work.work_id] - current_work.lead_time + 1
            elif key == 'FF':
                for related_work in current_work.relation[key]:
                    self.constraint[related_work][0] = self.locations[current_work.work_id] - current_work.lead_time
                    self.constraint[related_work][1] = self.locations[current_work.work_id] + 1

    def _calculate_reward_by_deviation(self):
        state = self._get_state()
        if state[1, -1] == -1:
            state = state[1:, :-1]
        else:
            state = state[1:]
        loads = np.sum(state, axis=0)
        idx = np.where(loads != 0.0)[0]
        deviation = np.std(loads[idx[0]:idx[-1] + 1])
        reward = 100 / deviation if deviation != 0.0 else 100
        return reward

    def _calculate_reward(self):
        state = self._get_state()
        loads_last_work = state[0]
        if state[1, -1] == -1:
            state = state[1:, :-1]
        else:
            state = state[1:]
        state[state == -1] = 0.0
        loads_last_work[loads_last_work == -1] = 0.0
        loads = np.sum(state, axis=0)
        loads_last_work = loads[np.where(loads_last_work != 0.0)]
        average_loads = np.mean(loads)
        reward = np.mean(average_loads - loads_last_work) // 10
        return reward


if __name__ == '__main__':
    from environment.work import *
    inbound, max_day = import_schedule('../environment/data/191227_납기일 추가.xlsx', [2962])
    scheduling = Scheduling(inbound_works=inbound, window=(10, 40))
    s = scheduling.reset()
    for i in range(50):
        print(i)
        s_next, r, d, _ = scheduling.step(1)
        print(r)
        s = s_next
        print(s)
        if d:
            break