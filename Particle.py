import numpy as np
import random
from utils import evaluate_position

class Particle:
    position_min = None
    position_max = None

    @staticmethod
    def set_search_bounds(lower, upper):
        """设置搜索边界"""
        if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
            # 如果输入是单个数值，转换为数组
            Particle.position_min = np.array([lower, lower])
            Particle.position_max = np.array([upper, upper])
        else:
            # 如果输入是数组，直接使用
            Particle.position_min = np.array(lower)
            Particle.position_max = np.array(upper)

    @staticmethod
    def generate_lhs_samples(num_samples):
        dimensions = Particle.position_min.size
        samples = np.zeros((num_samples, dimensions))

        # 创建 [0, 1] 范围的拉丁超立方采样
        raw_samples = np.zeros((dimensions, num_samples))
        for dim in range(dimensions):
            for i in range(num_samples):
                raw_samples[dim][i] = (i + random.random()) / num_samples
            np.random.shuffle(raw_samples[dim])

        # 转换为实际范围的样本
        for i in range(num_samples):
            sample = Particle.position_min + raw_samples[:, i] * (Particle.position_max - Particle.position_min)
            samples[i] = sample
        return samples

    def __init__(self, initial_position=None, region=None):
        """
        初始化粒子
        Args:
            initial_position: 初始化粒子位置
            region: 粒子所属的区域
        """
        if initial_position is None:
            self.position = np.zeros(Particle.position_min.size)
        else:
            self.position = np.array(initial_position)

        self.velocity = np.zeros_like(self.position)
        self.best_position = self.position.copy()
        self.best_cost = float('inf')
        self.region = region  # 添加区域属性

    def evaluate(self):
        """评估粒子的适应度值"""
        self.cost = evaluate_position(self.position)
        if self.cost < self.best_cost:
            self.best_cost = self.cost
            self.best_position = self.position.copy()

    def update(self, global_best_position, iteration, T):
        """
        更新粒子的位置和速度
        Args:
            global_best_position: 全局最优位置
            iteration: 当前迭代次数
            T: 最大迭代次数
        """
        # 判断当前粒子是否为全局最优粒子
        is_global_best = np.array_equal(self.position, global_best_position)
        if is_global_best:
            # 对全局最优粒子进行小幅度随机移动
            perturbation_scale = 0.05 * (1 - iteration / T)  # 可调整系数0.05
            perturbation = np.random.uniform(-1, 1, len(self.position)) * perturbation_scale * (
                    Particle.position_max - Particle.position_min)
            self.position += perturbation
            # 限制位置在搜索空间内
            self.position = np.clip(self.position, Particle.position_min, Particle.position_max)
        else:
            r1 = random.random()
            r2 = random.random()
            w = self.W_init - (self.W_init - self.W_end) * (iteration / T)  # 动态计算惯性权重
            self.velocity = w * self.velocity + self.C1 * r1 * (self.best_position - self.position) + self.C2 * r2 * (global_best_position - self.position)
            # 限制速度
            self.velocity = np.clip(self.velocity, -self.V_max, self.V_max)
            self.position += self.velocity
            # 限制位置在搜索空间内
            self.position = np.clip(self.position, Particle.position_min, Particle.position_max)

        # 评估新位置并更新区域信息
        self.evaluate()
        if self.region:
            self.region.update(self.position, self.cost)

    def set_params(self):
        """设置粒子的参数"""
        self.W_init = 0.9
        self.W_end = 0.2
        self.C1 = 0.6
        self.C2 = 0.4
        self.V_max = 1