import numpy as np
from Particle import Particle

class Swarm:
    def __init__(self, popsize_or_particles, T, initial_position=None, region=None):
        """
        初始化粒子群
        Args:
            popsize_or_particles: 整数(粒子数量)或粒子列表
            T: 迭代次数
            initial_position: 初始位置
            region: 粒子群所属的区域
        """
        self.T = T
        self.global_best_cost = float('inf')
        self.global_best_position = None
        self.region = region  # 添加区域属性
        
        # 用于记录优化过程中的所有数据
        self.history = {
            'positions': [],  # 所有粒子在每次迭代的位置
            'costs': [],     # 所有粒子在每次迭代的适应度值
            'best_position': None,  # 全局最优位置
            'best_cost': float('inf')  # 全局最优适应度值
        }

        # 判断第一个参数是粒子数量还是粒子列表
        if isinstance(popsize_or_particles, list):
            self.particles = popsize_or_particles
            self.popsize = len(popsize_or_particles)
            # 更新全局最优
            for particle in self.particles:
                if particle.best_cost < self.global_best_cost:
                    self.global_best_cost = particle.best_cost
                    self.global_best_position = particle.best_position.copy()
        else:
            self.popsize = popsize_or_particles
            if initial_position is None:
                raise ValueError("重构粒子群时需要提供initial_position")

            self.global_best_position = np.array(initial_position)
            self.particles = []
            for _ in range(self.popsize):
                perturbed_position = initial_position + np.random.uniform(-0.2, 0.2, len(initial_position)) * 0.1 * (
                            Particle.position_max - Particle.position_min)
                particle = Particle(perturbed_position, region=self.region)  # 传入区域信息
                particle.set_params()
                self.particles.append(particle)

    def optimize(self, config):
        """执行优化过程"""
        for t in range(self.T):
            print(f"迭代次数 {t + 1}: ")
            for particle in self.particles:
                particle.update(self.global_best_position, t, self.T)
                # 记录当前状态
                self.history['positions'].append(particle.position.copy())
                self.history['costs'].append(particle.cost)
                
                if particle.best_cost < self.global_best_cost:
                    self.global_best_cost = particle.best_cost
                    self.global_best_position = particle.best_position.copy()
                    self.history['best_position'] = self.global_best_position.copy()
                    self.history['best_cost'] = self.global_best_cost
                    print(f"更新全局最优： {self.global_best_cost} at position {self.global_best_position}\n")

            print(f"全局最优粒子值 {t + 1}: {self.global_best_position}")
            print(f"全局最优适应值: {self.global_best_cost}\n")
            
        return self.global_best_position, self.global_best_cost, self.history

    def get_optimization_history(self):
        """
        返回优化历史记录
        Returns:
            tuple: 包含所有迭代中所有粒子的位置和适应度值
        """
        return self.history['positions'], self.history['costs']

    def get_particles(self):
        """
        返回粒子群中的所有粒子
        Returns:
            list: 包含所有Particle对象的列表
        """
        return self.particles