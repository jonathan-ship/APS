import pygame
import numpy as np


class Scheduling(object):
    def __init__(self, num_days=4, num_blocks=0, window_days=(20, 10), inbound_works=None, load=1, backward=True, display_env=False):
        self.action_space = 2  # 좌로 이동 비활성화시 2, 활성화시 3
        self.num_days = num_days
        self.window_days = window_days
        self.num_work = len(inbound_works)
        self.num_block = num_blocks
        self.n_features = num_blocks * num_days
        self.empty = 0
        self.stage = 0
        self.inbound_works = inbound_works
        self.inbound_clone = inbound_works[:]
        self._ongoing = 0
        self._location = 0
        self.left_action = 2
        self.right_action = 0
        self.select_action = 1
        self.load = load
        self.backward = backward
        if backward:
            self.left_action, self.right_action = self.right_action, self.left_action
            self._location = \
                self.inbound_works[self._ongoing].latest_finish - self.inbound_works[self._ongoing].lead_time
        self.works = [self._location]
        if display_env:
            display = LocatingDisplay(self, num_days, self.num_block)
            display.game_loop_from_space()

    def step(self, action):
        done = False
        self.stage += 1
        reward = 0
        current_work = self.inbound_works[self._ongoing]
        if action == self.select_action:  # 일정 확정
            self._ongoing += 1
            reward = self._calculate_reward_by_local_deviation()
            if self._ongoing == self.num_work:
                done = True
            else:
                next_work = self.inbound_works[self._ongoing]
                if current_work.block != next_work.block:  # 다음 액티비티가 다음 블록인 경우 시작일 설정
                    if self.backward:
                        self._location = next_work.latest_finish - next_work.lead_time
                    else:
                        self._location = 0
                else:  # 다음 액티비티가 동일 블록인 경우
                    next_work.set_constraint(self.inbound_works[:self._ongoing], self.works, backward=self.backward)
                    if self.backward:
                        self._location = next_work.latest_finish - next_work.lead_time
                    else:
                        self._location = next_work.earlist_start + 1
                    self._location = min(self.num_days - next_work.lead_time, self._location)
                self.works.append(self._location)
        else:  # 일정 이동
            if action == self.left_action:  # 좌로 이동
                self._location = max(0, self._location - 1)
            elif action == self.right_action:  # 우로 이동
                due_date = current_work.latest_finish
                if due_date != -1:
                    self._location = min(due_date - current_work.lead_time, self._location + 1)
                else:
                    self._location = min(self.num_days - current_work.lead_time, self._location + 1)
            if len(self.works) == self._ongoing:
                self.works.append(self._location)
            else:
                self.works[self._ongoing] = self._location
        next_state = self.get_state().flatten()
        if self.stage == 300:
            done = True
        if done:
            for i in range(len(self.works)):
                self.inbound_works[i].set_location(self.works[i])
        return next_state, reward, done

    def reset(self):
        self.inbound_works = self.inbound_clone[:]
        self.stage = 0
        self._ongoing = 0
        if self.backward:
            self._location = \
                self.inbound_works[self._ongoing].latest_finish - self.inbound_works[self._ongoing].lead_time
        self.works = [self._location]
        return self.get_state().flatten()

    def get_state(self):
        state = np.full([self.num_block, self.num_days], 0)
        ongoing_location = self.works[-1]
        ongoing_block = self.inbound_works[-1].block
        ongoing_leadtime = self.inbound_works[-1].lead_time
        for i, location in enumerate(self.works):
            if self._ongoing == i:
                ongoing_location = location
                ongoing_block = self.inbound_works[i].block
                ongoing_leadtime = self.inbound_works[i].lead_time
            state[self.inbound_works[i].block, location:location + self.inbound_works[i].lead_time] = self.inbound_works[i].work_load_per_day
        left = max(0, int(ongoing_location + ongoing_leadtime / 2 - self.window_days[0] / 2))
        left = min(left, self.num_days - self.window_days[0])
        top = max(0, int(ongoing_block - self.window_days[1] / 2))
        top = min(top, self.num_block - self.window_days[1])
        state = state[top:top + self.window_days[1], left:left + self.window_days[0]]
        return state

    def _calculate_reward(self):
        state = self.get_state()
        last_work = self.works[-1]
        lead_time = self.inbound_works[self._ongoing - 1].lead_time
        loads = np.sum(state, axis=0)
        loads_last_work = loads[last_work:last_work + lead_time]
        score1 = 1
        score2 = -1
        reward = 0
        for load in loads_last_work:
            if load <= self.load:
                reward += score1
            elif load > self.load:
                reward += score2
        return reward

    def _calculate_reward_by_deviation(self):
        state = self.get_state()
        loads = np.sum(state, axis=0)
        deviation = max(0.2, float(np.std(loads)))
        return 1 / deviation

    def _calculate_reward_by_local_deviation(self):
        state = self.get_state()
        last_work = self.works[-1]
        lead_time = self.inbound_works[self._ongoing - 1].lead_time
        loads = np.sum(state, axis=0)
        loads_last_work = loads[last_work:last_work + lead_time]
        deviation = deviation = max(0.2, float(np.std(loads_last_work)))
        return 1 / deviation


class LocatingDisplay(object):
    white = (255, 255, 255)
    black = (0, 0, 0)
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    dark_red = (200, 0, 0)
    dark_green = (0, 200, 0)
    dark_blue = (0, 0, 200)

    x_init = 100
    y_init = 100
    x_span = 20
    y_span = 50
    thickness = 5
    pygame.init()
    display_width = 1000
    display_height = 600
    font = 'freesansbold.ttf'
    pygame.display.set_caption('Steel Locating')
    clock = pygame.time.Clock()
    pygame.key.set_repeat()

    def __init__(self, locating, width, height):
        self.width = width
        self.height = height
        self.space = locating
        self.on_button = False
        self.total = 0
        self.display_width = self.x_span * width + 200
        self.display_height = self.y_span * height + 200
        self.gameDisplay = pygame.display.set_mode((self.display_width, self.display_height),
                                                   pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE)
        self.fake_screen = self.gameDisplay.copy()
        self.pic = pygame.surface.Surface((50, 50))
        self.pic.fill((255, 100, 200))

    def restart(self):
        self.space.reset()
        self.game_loop_from_space()

    def text_objects(self, text, font):
        text_surface = font.render(text, True, self.white)
        return text_surface, text_surface.get_rect()

    def block(self, x, y, text='', color=(0, 255, 0), x_init=100):
        pygame.draw.rect(self.gameDisplay, color, (int(x_init + self.x_span * x),
                                                   int(self.y_init + self.y_span * y),
                                                   int(self.x_span),
                                                   int(self.y_span)))
        large_text = pygame.font.Font(self.font, 20)
        text_surf, text_rect = self.text_objects(text, large_text)
        text_rect.center = (int(x_init + self.x_span * (x + 0.5)), int(self.y_init + self.y_span * (y + 0.5)))
        self.gameDisplay.blit(text_surf, text_rect)

    def board(self, step, reward=0):
        large_text = pygame.font.Font(self.font, 20)
        text_surf, text_rect = self.text_objects('step: ' + str(step) + '   reward: ' + format(reward, '.2f')
                                                 + '   total: ' + format(self.total, '.2f'), large_text)
        text_rect.center = (200, 20)
        self.gameDisplay.blit(text_surf, text_rect)

    def button(self, goal=0):
        color = self.dark_blue
        str_goal = 'In'
        if self.on_button:
            color = self.blue
        if goal == 0:
            str_goal = 'Out'
            color = self.dark_red
            if self.on_button:
                color = self.red
        pygame.draw.rect(self.gameDisplay, color, self.button_goal)
        large_text = pygame.font.Font(self.font, 20 * self.screen_size_factor)
        text_surf, text_rect = self.text_objects(str_goal, large_text)
        text_rect.center = (int(self.button_goal[0] + 0.5 * self.button_goal[2]),
                            int(self.button_goal[1] + 0.5 * self.button_goal[3]))
        self.gameDisplay.blit(text_surf, text_rect)

    def game_loop_from_space(self):
        action = -1
        game_exit = False
        done = False
        reward = 0
        self.total = 0
        while not game_exit:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        action = self.space.left_action
                    elif event.key == pygame.K_RIGHT:
                        action = self.space.right_action
                    elif event.key == pygame.K_DOWN:
                        action = self.space.select_action
                    elif event.key == pygame.K_ESCAPE:
                        game_exit = True
                        break
                elif event.type == pygame.VIDEORESIZE:
                    self.gameDisplay = pygame.display.set_mode(event.dict['size'],
                                                     pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE)
                    self.fake_screen.blit(self.pic, (100, 100))
                    self.gameDisplay.blit(pygame.transform.scale(self.fake_screen, event.dict['size']), (0, 0))
                    pygame.display.flip()
                if action != -1:
                    state, reward, done = self.space.step(action)
                    self.total += reward
                if done:
                    self.restart()
                # click = pygame.mouse.get_pressed()
                # mouse = pygame.mouse.get_pos()
                self.on_button = False
                action = -1
            self.gameDisplay.fill(self.black)
            self.draw_space(self.space)
            self.board(self.space.stage, reward)
            self.draw_grid()
            self.message_display('Schedule', self.display_width // 2, 80)
            pygame.display.flip()
            self.clock.tick(10)

    def draw_grid(self):
        width = self.width
        height = self.height
        pygame.draw.line(self.gameDisplay, self.blue, (self.x_init, self.y_init),
                         (self.x_init, self.y_init + self.y_span * height), self.thickness)
        pygame.draw.line(self.gameDisplay, self.blue, (self.x_init, self.y_init),
                         (self.x_init + self.x_span * width, self.y_init), self.thickness)

        for i in range(width):
            pygame.draw.line(self.gameDisplay, self.blue, (self.x_init + self.x_span * (i + 1), self.y_init),
                             (self.x_init + self.x_span * (i + 1), self.y_init + self.y_span * height), self.thickness)
        for i in range(height):
            pygame.draw.line(self.gameDisplay, self.blue, (self.x_init, self.y_init + self.y_span * (i + 1)),
                             (self.x_init + self.x_span * width, self.y_init + self.y_span * (i + 1)), self.thickness)

    def draw_space(self, space):
        state = space.get_state()
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if state[i, j] == 0:
                    continue
                elif state[i, j] == 1:
                    rgb = self.green
                elif state[i, j] == 2:
                    rgb = self.blue
                elif state[i, j] == 3:
                    rgb = self.red
                self.block(j, i, '', rgb, x_init=self.x_init)

    def message_display(self, text, x, y):
        large_text = pygame.font.Font(self.font, 20)
        text_surf, text_rect = self.text_objects(text, large_text)
        text_rect.center = (x, y)
        self.gameDisplay.blit(text_surf, text_rect)