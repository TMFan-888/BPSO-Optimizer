import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel


class BayesianOptimization:
    def __init__(self, length_scale=1.0, sigma_noise=1e-6):
        """
        初始化贝叶斯优化器，使用高斯过程进行建模。
        :param length_scale:  Matern 核的长度尺度。
        :param sigma_noise: 高斯过程噪声项。
        """

        # 使用 Matern 核替代 RBF 核，并设置合适的边界
        kernel = (ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) *
                  Matern(length_scale=length_scale,
                         length_scale_bounds=(1e-3, 1e3),
                         nu=2.5)) + \
                  WhiteKernel(noise_level=sigma_noise,
                             noise_level_bounds=(1e-10, 1e1))  # 调整噪声范围

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=42  # 添加随机种子以提高稳定性
        )
        self.observed_particles = []  # 已观测的样本
        self.observed_costs = []  # 已观测的适应值

    def gau_pre(self, candidate):
        """使用高斯过程模型预测均值和方差"""
        if len(self.observed_particles) == 0:
            # 如果没有观测数据，返回默认均值和方差
            return 0.0, 1.0
        # 转换数据为 NumPy 数组
        x_train = np.array(self.observed_particles)
        y_train = np.array(self.observed_costs)
        # 用高斯过程模型拟合已有数据
        self.gp.fit(x_train, y_train)
        # 使用高斯过程预测候选点的均值和方差
        mu, sigma = self.gp.predict(np.array([candidate]), return_std=True)
        return mu[0], sigma[0]

    def EI(self, candidates, best_cost, exploration_weight=5.0):
        """计算期望改进 (EI) 并返回最佳候选解"""
        best_candidate = None
        best_ei = -np.inf
        for candidate in candidates:
            mu, sigma = self.gau_pre(candidate)
            if sigma == 0:
                ei = 0
            else:
                z = (best_cost - mu) / sigma
                ei = exploration_weight * sigma + \
                     (best_cost - mu) * norm.cdf(z) + \
                     sigma * norm.pdf(z)
            if ei > best_ei:
                best_ei = ei
                best_candidate = candidate
        return best_candidate

    def update_model(self, candidate, candidate_cost):
        """更新贝叶斯优化模型"""
        self.observed_particles.append(candidate)
        self.observed_costs.append(candidate_cost)

    @staticmethod
    def lhs_samples(param_space, num_samples):
        """生成拉丁超立方采样点"""
        samples = np.random.uniform(0, 1, (num_samples, len(param_space)))
        for i, bounds in enumerate(param_space):
            samples[:, i] = bounds[0] + samples[:, i] * (bounds[-1] - bounds[0])
            if i == 1:  # batch_size的索引
                samples[:, i] = samples[:, i]
        return samples
