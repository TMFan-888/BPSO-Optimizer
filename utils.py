# utils.py
import numpy as np

def test_function(x):
    """
    测试函数 - Rastrigin函数
    这是一个经典的优化测试函数，具有多个局部最小值
    f(x) = 10n + Σ(x[i]^2 - 10cos(2πx[i]))
    全局最小值：f(0,0,...,0) = 0
    """
    return 10 * len(x) + sum([(xi**2 - 10 * np.cos(2 * np.pi * xi)) for xi in x])

def evaluate_position(position):
    """
    评估给定位置的适应度值
    Args:
        position: 需要评估的位置坐标
    Returns:
        float: 位置的适应度值（成本）
    """
    return test_function(position)

def initialize_position(config):
    """
    初始化搜索位置
    Args:
        config: 配置参数字典
    Returns:
        numpy.ndarray: 初始化的位置坐标
    """
    n_dimensions = config.get("n_dimensions", 2)
    xbounds = config.get("xbounds", [-5.12, 5.12])
    ybounds = config.get("ybounds", [-5.12, 5.12])
    
    # 在各维度的搜索范围内随机初始化位置
    position = np.zeros(n_dimensions)
    position[0] = np.random.uniform(xbounds[0], xbounds[1])
    position[1] = np.random.uniform(ybounds[0], ybounds[1])
    return position

def initialize_velocity(config):
    """
    初始化粒子速度
    Args:
        config: 配置参数字典
    Returns:
        numpy.ndarray: 初始化的速度向量
    """
    n_dimensions = config.get("n_dimensions", 2)
    xbounds = config.get("xbounds", [-5.12, 5.12])
    ybounds = config.get("ybounds", [-5.12, 5.12])
    
    # 速度范围通常设置为位置范围的一定比例
    velocity = np.zeros(n_dimensions)
    velocity[0] = np.random.uniform(-0.1 * (xbounds[1] - xbounds[0]), 
                                   0.1 * (xbounds[1] - xbounds[0]))
    velocity[1] = np.random.uniform(-0.1 * (ybounds[1] - ybounds[0]), 
                                   0.1 * (ybounds[1] - ybounds[0]))
    return velocity

def get_region_bounds(region_size, bounds):
    """
    计算区域的边界
    Args:
        region_size: 区域大小 [(dx1, dy1), (dx2, dy2)]
        bounds: 搜索空间边界 [[x_min, x_max], [y_min, y_max]]
    Returns:
        list: 区域边界列表
    """
    x_divisions = int((bounds[0][1] - bounds[0][0]) / region_size[0][0])
    y_divisions = int((bounds[1][1] - bounds[1][0]) / region_size[1][0])
    
    region_bounds = []
    for i in range(x_divisions):
        for j in range(y_divisions):
            x_min = bounds[0][0] + i * region_size[0][0]
            x_max = x_min + region_size[0][0]
            y_min = bounds[1][0] + j * region_size[1][0]
            y_max = y_min + region_size[1][0]
            region_bounds.append([[x_min, x_max], [y_min, y_max]])
    
    return region_bounds