import matplotlib
matplotlib.use('TkAgg')  # 设置后端为 TkAgg
import matplotlib.pyplot as plt
import numpy as np
from BPSO_optimizer import BPSOOptimizer
from utils import evaluate_position, test_function

def plot_optimization_process(optimizer, iteration, candidates=None, swarms=None):
    """
    绘制优化过程的3D视图，使用2x2布局
    """
    # 清除当前图形内容，但保持窗口
    plt.gcf().clear()
    
    # 创建2x2子图
    gs = plt.GridSpec(2, 2, figure=plt.gcf())
    ax1 = plt.gcf().add_subplot(gs[0, 0])  # 左上：2D贝叶斯模型预测
    ax2 = plt.gcf().add_subplot(gs[0, 1], projection='3d')  # 右上：当前迭代PSO
    ax3 = plt.gcf().add_subplot(gs[1, 0], projection='3d')  # 左下：观测点拟合曲面
    ax4 = plt.gcf().add_subplot(gs[1, 1], projection='3d')  # 右下：真实函数曲面
    
    # 从optimizer获取搜索范围
    x_bounds = optimizer.bounds[0]  # [x_min, x_max]
    y_bounds = optimizer.bounds[1]  # [y_min, y_max]
    
    # 创建网格数据
    x = np.linspace(x_bounds[0], x_bounds[1], 50)
    y = np.linspace(y_bounds[0], y_bounds[1], 50)
    X, Y = np.meshgrid(x, y)
    Z_pred = np.zeros_like(X)  # 贝叶斯预测值
    Z_fit = np.zeros_like(X)   # 观测点拟合值
    Z_true = np.zeros_like(X)  # 真实值
    
    # 收集所有观测点
    observed_positions = []
    observed_costs = []
    for record in optimizer.local_best_records.values():
        if record['history']:
            for point in record['history']:
                observed_positions.append(point['position'])
                observed_costs.append(point['cost'])
    
    # 计算所有曲面值
    if observed_positions:
        from scipy.interpolate import Rbf
        observed_positions = np.array(observed_positions)
        observed_costs = np.array(observed_costs)
        rbf = Rbf(observed_positions[:,0], observed_positions[:,1], observed_costs, function='multiquadric')
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                pos = np.array([X[i,j], Y[i,j]])
                Z_pred[i,j] = optimizer.bayes_opt.gau_pre(pos)[0]  # 贝叶斯预测
                Z_fit[i,j] = rbf(X[i,j], Y[i,j])                  # RBF拟合
                Z_true[i,j] = test_function(pos)                   # 真实值
    
    # === 第一个图：贝叶斯预测（2D等高线图）===
    ax1.set_title(f'Bayesian Model (Iteration {iteration})')
    contour = ax1.contour(X, Y, Z_pred, levels=20, cmap='viridis')
    plt.colorbar(contour, ax=ax1)
    
    # 绘制区域网格
    for region_id, region in optimizer.regions.items():
        bounds = region.bounds
        # 绘制网格线
        ax1.plot([bounds[0][0], bounds[0][1]], [bounds[1][0], bounds[1][0]], 'k--', alpha=0.2)
        ax1.plot([bounds[0][0], bounds[0][1]], [bounds[1][1], bounds[1][1]], 'k--', alpha=0.2)
        ax1.plot([bounds[0][0], bounds[0][0]], [bounds[1][0], bounds[1][1]], 'k--', alpha=0.2)
        ax1.plot([bounds[0][1], bounds[0][1]], [bounds[1][0], bounds[1][1]], 'k--', alpha=0.2)
        
        # 显示访问次数
        if region.visit_count > 0:
            center_x = (bounds[0][0] + bounds[0][1]) / 2
            center_y = (bounds[1][0] + bounds[1][1]) / 2
            ax1.text(center_x, center_y, str(region.visit_count), 
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.8))
    
    # 绘制当前迭代的候选点
    if candidates is not None:
        candidates = np.array(candidates)
        ax1.scatter(candidates[:,0], candidates[:,1], 
                   c='r', marker='*', s=100, label='Current Candidates')
    
    # 绘制全局最优点
    if optimizer.global_best_position is not None:
        ax1.scatter(optimizer.global_best_position[0], 
                   optimizer.global_best_position[1], 
                   c='g', marker='*', s=200, 
                   label=f'Global Best (Cost: {test_function(optimizer.global_best_position):.4f})')
    
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(x_bounds)
    ax1.set_ylim(y_bounds)
    
    # === 右侧图：当前迭代的PSO过程 ===
    if swarms is not None:
        ax2.set_title(f'PSO Process (Iteration {iteration})')
        ax2.plot_surface(X, Y, Z_pred, cmap='viridis', alpha=0.3)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(swarms)))
        
        for swarm_idx, swarm in enumerate(swarms):
            positions = np.array(swarm.history['positions'])
            if len(positions) > 0:
                z_positions = np.array([test_function(pos) for pos in positions])
                
                # 绘制粒子轨迹
                for i in range(len(swarm.particles)):
                    particle_positions = positions[i::len(swarm.particles)]
                    particle_z = z_positions[i::len(swarm.particles)]
                    if len(particle_positions) > 1:
                        ax2.plot(particle_positions[:,0], particle_positions[:,1], particle_z,
                                '-', color=colors[swarm_idx], alpha=0.3,
                                label=f'Swarm {swarm_idx+1}' if i == 0 else "")
                
                # 绘制最终位置
                final_positions = positions[-len(swarm.particles):]
                final_z = z_positions[-len(swarm.particles):]
                ax2.scatter(final_positions[:,0], final_positions[:,1], final_z,
                           color=colors[swarm_idx], alpha=0.7)
        
        ax2.set_xlabel('x1')
        ax2.set_ylabel('x2')
        ax2.set_zlabel('Cost')
        ax2.legend()
    
    # === 第三个图：观测点拟合曲面 ===
    ax3.set_title('Fitted Surface from Observations')
    if observed_positions.size > 0:
        surf3 = ax3.plot_surface(X, Y, Z_fit, cmap='viridis', alpha=0.6)
        plt.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)
        
        ax3.scatter(observed_positions[:,0], observed_positions[:,1], observed_costs,
                   c='r', marker='.', s=50, alpha=0.5,
                   label=f'Observed Points ({len(observed_positions)})')
        
        if optimizer.global_best_position is not None:
            z_best = test_function(optimizer.global_best_position)
            ax3.scatter(optimizer.global_best_position[0], 
                       optimizer.global_best_position[1], 
                       z_best, c='g', marker='*', s=200,
                       label=f'Global Best (Cost: {z_best:.4f})')
    
    ax3.set_xlabel('x1')
    ax3.set_ylabel('x2')
    ax3.set_zlabel('Cost')
    ax3.legend()
    
    # === 第四个图：真实函数曲面 ===
    ax4.set_title('True Function Surface')
    surf4 = ax4.plot_surface(X, Y, Z_true, cmap='viridis', alpha=0.6)
    plt.colorbar(surf4, ax=ax4, shrink=0.5, aspect=5)
    
    if observed_positions.size > 0:
        ax4.scatter(observed_positions[:,0], observed_positions[:,1], observed_costs,
                   c='r', marker='.', s=50, alpha=0.5,
                   label=f'Observed Points ({len(observed_positions)})')
    
    if optimizer.global_best_position is not None:
        z_best = test_function(optimizer.global_best_position)
        ax4.scatter(optimizer.global_best_position[0], 
                   optimizer.global_best_position[1], 
                   z_best, c='g', marker='*', s=200,
                   label=f'Global Best (Cost: {z_best:.4f})')
    
    ax4.set_xlabel('x1')
    ax4.set_ylabel('x2')
    ax4.set_zlabel('Cost')
    ax4.legend()
    
    # 调整3D图的视角和边距
    for ax in [ax2, ax3, ax4]:
        ax.view_init(elev=20, azim=45)  # 降低仰角
        ax.set_xlim(x_bounds)  # 使用config中的范围
        ax.set_ylim(y_bounds)  # 使用config中的范围
        ax.dist = 12
    
    # 单独设置2D图的范围
    ax1.set_xlim(x_bounds)  # 使用config中的范围
    ax1.set_ylim(y_bounds)  # 使用config中的范围
    
    # 使用紧凑布局
    plt.tight_layout()
    plt.gcf().canvas.draw()
    plt.gcf().canvas.flush_events()
    plt.pause(0.1)

if __name__ == "__main__":
    # 配置参数
    config = {
        "n_dimensions": 2,          # 优化问题的维度
        "xbounds": [-5.12, 5.12],    # x1范围
        "ybounds": [-5.12, 5.12],    # x2范围
        "region_size": [(1.0, 1.0), (1.0, 1.0)],  # 增大区域大小

        "pso_particles": 20,    # 增加粒子数量
        "pso_iterations": 20,   # 增加迭代次数

        "bayesian_iterations": 10,  # 贝叶斯优化迭代次数
        "initial_samples": 10,  # 增加初始采样点
        "ei_samples": 100,      # 增加EI采样点
        "ei_select_num": 10,    # 增加选择的候选点数量

        "region_visit_threshold": 2,  # 区域访问阈值
        "max_inactive_iterations": 5,  # 最大非活跃迭代次数
    }

    # 创建并设置固定大小的窗口
    plt.ion()
    fig = plt.figure(figsize=(10, 10))
    
    # 设置窗口位置和大小
    manager = plt.get_current_fig_manager()
    manager.window.geometry("1000x1000+50+50")  # 宽度x高度+左边距+上边距
    
    # 创建优化器
    optimizer = BPSOOptimizer(config)
    
    best_position, best_cost, local_records = optimizer.optimize(
        callback=plot_optimization_process
    )
    
    plt.ioff()
    plt.show()  # 显示最终结果
    
    print("\n优化统计信息:")
    print(f"全局最优成本: {best_cost:.6f}")
    print(f"全局最优位置: x1={best_position[0]:.6f}, x2={best_position[1]:.6f}")





