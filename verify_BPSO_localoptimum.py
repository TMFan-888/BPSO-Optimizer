import matplotlib
matplotlib.use('TkAgg')  # 设置后端为 TkAgg
import matplotlib.pyplot as plt
import numpy as np
from BPSO_optimizer import BPSOOptimizer



if __name__ == "__main__":
    # 配置参数
    config = {
        "n_dimensions": 2,          # 优化问题的维度
        "xbounds": [-5.12, 5.12],    # 函数优化范围
        "ybounds": [-5.12, 5.12],    # 函数优化范围
        "region_size": [(0.5, 0.5), (0.5, 0.5)],  # 区域划分大小

        "pso_particles": 10,    # PSO粒子个数
        "pso_iterations": 10,  # PSO迭代次数

        "bayesian_iterations": 10,  # 贝叶斯优化迭代次数

        "initial_samples": 10,  # 初始采样点数量
        "ei_samples": 50,  # EI采样点数量
        "ei_select_num": 10,  # 每次选择的EI最优点数量

        "region_visit_threshold": 2,  # 区域访问阈值
        "max_inactive_iterations": 5,  # 最大非活跃迭代次数
    }

    # 创建并运行优化器
    optimizer = BPSOOptimizer(config)
    best_position, best_cost, local_records = optimizer.optimize()
    print("\n优化统计信息:")
    print(f"全局最优成本: {best_cost:.6f}")
    print(f"全局最优位置: x1={best_position[0]:.6f}, x2={best_position[1]:.6f}")





