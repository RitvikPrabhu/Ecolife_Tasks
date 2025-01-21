import random

import numpy as np

import utils


class HillClimbing:
    def __init__(
        self, parameters, server_pair, function_name, ci_avg, cur_ci, cur_interval
    ):
        self.max_delta_ci = 0.1
        self.prev_ci = cur_ci
        self.max_delta_fn = 0.1
        self.prev_fn = len(cur_interval)
        self.w = 1
        # Unpack parameters
        self.size = parameters[0]
        self.var_num = 2  # hardware choice and keep-alive time
        self.var_1 = [0, 1]  # 0 -> old, 1 -> new
        self.var_2 = parameters[1]  # array of keep-alive times
        self.lam = parameters[2]  # lambda for weighting service time vs carbon
        self.server_pair = server_pair
        self.function_name = function_name

        # [hw_choice, keep_alive]
        self.pop_x = np.zeros((self.size, self.var_num))
        self.pop_v = np.zeros((self.size, self.var_num))
        self.p_best = np.zeros((self.size, self.var_num))
        self.g_best = np.zeros((1, self.var_num))

        old_cold, _ = utils.get_st(function_name, server_pair[0])
        new_cold, _ = utils.get_st(function_name, server_pair[1])
        cold_carbon_max, _ = utils.compute_exe(function_name, server_pair, ci_avg)
        self.max_st = max(old_cold, new_cold)
        self.max_carbon_st = max(cold_carbon_max)
        self.max_carbon_kat = max(
            utils.compute_kat(function_name, server_pair[0], 7, ci_avg),
            utils.compute_kat(function_name, server_pair[1], 7, ci_avg),
        )

        self.bound = [[0, 0], [1, max(self.var_2)]]

        temp = np.inf
        for i in range(self.size):
            self.pop_x[i][0] = random.choice(self.var_1)
            self.pop_x[i][1] = random.choice(self.var_2)
            self.p_best[i] = self.pop_x[i]
            fit = self.fitness(self.p_best[i], cur_ci, cur_interval)
            if fit < temp:
                self.g_best = self.p_best[i]
                temp = fit
        self.temp = temp

    def prob_cold(self, cur_interval, kat):
        if len(cur_interval) == 0:
            return 0.5, 0.5
        cold = 0
        warm = 0
        for interval in cur_interval:
            if interval <= kat:
                warm += 1
            else:
                cold += 1
        total = cold + warm
        if total == 0:
            return 0.5, 0.5
        return cold / (cold + warm), warm / (cold + warm)

    def fitness(self, var, ci, past_interval):
        var = var.astype(int)
        ka_loc = var[0]
        kat = var[1]

        old_kat_carbon = utils.compute_kat(
            self.function_name, self.server_pair[0], kat, ci
        )
        new_kat_carbon = utils.compute_kat(
            self.function_name, self.server_pair[1], kat, ci
        )

        cold_carbon, warm_carbon = utils.compute_exe(
            self.function_name, self.server_pair, ci
        )
        old_st = utils.get_st(self.function_name, self.server_pair[0])
        new_st = utils.get_st(self.function_name, self.server_pair[1])

        score = 0
        combined_kat_carbon = (1 - ka_loc) * old_kat_carbon + ka_loc * new_kat_carbon
        score += (1 - self.lam) * (combined_kat_carbon / self.max_carbon_kat)

        cold_prob, warm_prob = self.prob_cold(past_interval, kat)
        part_time_prob = cold_prob * (
            (1 - ka_loc) * old_st[0] + ka_loc * new_st[0]
        ) + warm_prob * ((1 - ka_loc) * old_st[1] + ka_loc * new_st[1])
        part_carbon_prob = cold_prob * (
            (1 - ka_loc) * cold_carbon[0] + ka_loc * cold_carbon[1]
        ) + warm_prob * ((1 - ka_loc) * warm_carbon[0] + ka_loc * warm_carbon[1])

        score += self.lam * (part_time_prob / self.max_st)
        score += (1 - self.lam) * (part_carbon_prob / self.max_carbon_st)

        return score

    def update_operator(self, ci, past_interval, diff_ci, diff_fn):
        if (diff_fn / self.max_delta_fn) or (diff_ci / self.max_delta_ci) > 0:
            half_indices = np.random.choice(
                int(self.size / 2), int(self.size / 2), replace=False
            )
            for index in half_indices:
                i = half_indices[index]
                self.pop_x[i][0] = int(random.choice(self.var_1))
                self.pop_x[i][1] = int(random.choice(self.var_2))

        for i in range(self.size):
            current = self.pop_x[i].copy()
            current_fit = self.fitness(current, ci, past_interval)
            neighbors = self.get_neighbors(current)
            for neigh in neighbors:
                neigh_fit = self.fitness(neigh, ci, past_interval)
                if neigh_fit < current_fit:
                    current = neigh
                    current_fit = neigh_fit
            self.pop_x[i] = current
            if current_fit < self.fitness(self.p_best[i], ci, past_interval):
                self.p_best[i] = current
            if current_fit < self.fitness(self.g_best, ci, past_interval):
                self.g_best = current

    def get_neighbors(self, solution):
        var = solution.astype(int)
        hw, kat = var
        neighbors = []

        neighbors.append(np.array([1 - hw, kat]))

        for _ in range(2):
            neighbors.append(np.array([hw, random.choice(self.var_2)]))

        for nbd in neighbors:
            if nbd[0] < 0:
                nbd[0] = 0
            if nbd[0] > 1:
                nbd[0] = 1
            if nbd[1] < 0:
                nbd[1] = 0
            if nbd[1] > max(self.var_2):
                nbd[1] = max(self.var_2)
        return neighbors

    def main(self, ci, past_interval):
        diff_ci = abs(ci - self.prev_ci)
        diff_fn = abs(len(past_interval) - self.prev_fn)

        self.update_operator(ci, past_interval, diff_ci, diff_fn)

        if diff_ci > self.max_delta_ci:
            self.max_delta_ci = diff_ci
        if diff_fn > self.max_delta_fn:
            self.max_delta_fn = diff_fn
        self.prev_ci = ci
        self.prev_fn = len(past_interval)

        return self.g_best, self.p_best
