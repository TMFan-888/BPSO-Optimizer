import numpy as np
from Bayesian import BayesianOptimization

from Swarm import Swarm
from utils import evaluate_position
import Particle
import random

class BPSOOptimizer:
    """
    结合贝叶斯优化和粒子群优化的混合优化器。

    主要特点：
    - 使用贝叶斯优化进行全局搜索
    - 使用粒子群优化进行局部搜索
    - 包含区域管理机制
    - 包含候选点管理机制
    """
    def __init__(self, config):
        """
        初始化BPSO优化器
        Args:
            config: 配置参数字典
        """
        self.config = config
        # 统一使用bounds表示搜索范围
        self.bounds = np.array([
            config.get('xbounds', [-5.12, 5.12]),  # x1范围
            config.get('ybounds', [-5.12, 5.12])   # x2范围
        ])
        
        # 区域管理参数
        self.region_size = config.get('region_size', [(0.5, 0.5), (0.5, 0.5)])  # 恢复为二维表示
        self.region_visit_threshold = config.get('region_visit_threshold', 2)
        self.explored_regions = {}  # 添加区域记录字典

        # PSO相关参数
        self.pso_popsize = config.get('pso_particles')  # PSO种群大小
        self.pso_iterations = config.get('pso_iterations')  # PSO迭代次数

        # 贝叶斯优化参数
        self.Bayesian_T = config.get('bayesian_iterations')
        self.initial_samples = config.get('initial_samples')
        self.ei_samples = config.get('ei_samples')
        self.ei_select_num = config.get('ei_select_num')

        # 候选点管理参数
        self.max_inactive_iterations = config.get('max_inactive_iterations')

        # 存储优化结果
        self.local_best_records = {}  # 每个候选点的优化历史
        self.active_candidates = set()  # 当前活跃的候选点集合
        self.history = self._init_history()

        self.global_best_cost = float('inf')
        self.global_best_position = None

        # 初始化贝叶斯优化器
        self.bayes_opt = BayesianOptimization()

        # 区域管理
        self.regions = {}  # 存储所有区域
        self._init_regions()

    def _init_regions(self):
        """初始化搜索空间的区域划分"""
        x_divisions = int((self.bounds[0][1] - self.bounds[0][0]) / self.region_size[0][0])
        y_divisions = int((self.bounds[1][1] - self.bounds[1][0]) / self.region_size[1][0])
        
        # 创建二维区域索引
        self.region_indices = np.zeros((x_divisions, y_divisions), dtype=object)
        
        for i in range(x_divisions):
            for j in range(y_divisions):
                x_min = self.bounds[0][0] + i * self.region_size[0][0]
                x_max = x_min + self.region_size[0][0]
                y_min = self.bounds[1][0] + j * self.region_size[1][0]
                y_max = y_min + self.region_size[1][0]
                
                region_id = (i, j)  # 使用元组作为区域ID
                bounds = [[x_min, x_max], [y_min, y_max]]
                self.regions[region_id] = Region(region_id, bounds)
                self.region_indices[i, j] = region_id

    def optimize(self, callback=None):
        """
        执行BPSO优化过程
        Args:
            callback: 可视化回调函数
        """
        # 第一阶段：初始采样
        print("执行初始采样阶段...")
        initial_candidates = self.bayes_opt.lhs_samples(self.bounds, num_samples=self.initial_samples)

        # 处理初始采样点（迭代0）
        current_candidates = []
        current_swarms = []
        
        for idx, candidate in enumerate(initial_candidates):
            candidate_id = f"initial_{idx}"
            self._init_new_candidate(candidate_id)
            record = self.local_best_records[candidate_id]
            
            # 评估候选解
            candidate_cost = self._evaluate_candidate(candidate, candidate_id)
            current_candidates.append(candidate)
            
            # 更新记录
            self._update_region_record(candidate, candidate_cost)
            record['history'].append({
                'position': candidate,
                'cost': candidate_cost
            })
            
            # 更新全局最优
            if candidate_cost < self.global_best_cost:
                self.global_best_cost = candidate_cost
                self.global_best_position = candidate
        
        # 显示初始采样结果
        if callback:
            callback(self, 0, current_candidates, None)
        
        # 使用初始采样数据更新贝叶斯模型
        if hasattr(self.bayes_opt, 'update'):
            positions = [r['best_position'] for r in self.local_best_records.values()
                        if r['best_position'] is not None]
            costs = [r['best_cost'] for r in self.local_best_records.values()
                    if r['best_position'] is not None]
            if positions and costs:
                self.bayes_opt.update(positions, costs)
        
        # 第二阶段：贝叶斯优化迭代
        for i in range(self.Bayesian_T):
            print(f"执行第 {i + 1} 次贝叶斯循环\n")
            current_candidates = []
            current_swarms = []
            
            # 生成并处理候选点
            samples = self.bayes_opt.lhs_samples(self.bounds, num_samples=self.ei_samples)
            candidates = []
            remaining_samples = samples.copy()
            
            # 选择EI值最大的点
            for _ in range(self.ei_select_num):
                if len(remaining_samples) > 0:
                    best_candidate = self.bayes_opt.EI(remaining_samples, self.global_best_cost)
                    candidates.append(best_candidate)
                    remaining_samples = [s for s in remaining_samples
                                      if not np.array_equal(s, best_candidate)]
            
            # 处理候选点
            for candidate_idx, candidate in enumerate(candidates):
                candidate_id = f"candidate_{i}_{candidate_idx}"
                current_candidates.append(candidate)
                
                if candidate_id not in self.local_best_records:
                    self._init_new_candidate(candidate_id)
                
                record = self.local_best_records[candidate_id]
                candidate_cost = self._evaluate_candidate(candidate, candidate_id)
                
                self._update_region_record(candidate, candidate_cost)
                record['history'].append({
                    'position': candidate,
                    'cost': candidate_cost
                })
                
                # 判断是否继续搜索
                if self._is_promising_candidate(candidate, candidate_cost, candidate_id):
                    # 运行PSO并保存粒子群对象
                    swarm = self._run_particle_swarm(candidate, candidate_id)
                    if swarm:
                        current_swarms.append(swarm)
            
            # 显示当前迭代结果
            if callback:
                callback(self, i + 1, current_candidates, current_swarms)
            
            # 更新贝叶斯模型
            if hasattr(self.bayes_opt, 'update'):
                positions = [r['best_position'] for r in self.local_best_records.values()
                           if r['best_position'] is not None]
                costs = [r['best_cost'] for r in self.local_best_records.values()
                        if r['best_position'] is not None]
                if positions and costs:
                    self.bayes_opt.update(positions, costs)
        
        return self.global_best_position, self.global_best_cost, self.local_best_records

    def _init_history(self):
        """初始化优化历史记录"""
        return {
            'iterations': [],
            'best_costs': [],
            'best_positions': [],
            'explored_regions': set()
        }

    def _init_new_candidate(self, candidate_id):
        """初始化新的候选点记录"""
        self.local_best_records[candidate_id] = {
            'best_position': None,
            'best_cost': float('inf'),
            'current_cost': float('inf'),
            'history': [],
            'inactive_iterations': 0,
            'active': True
        }
        self.active_candidates.add(candidate_id)

    def _evaluate_candidate(self, candidate, candidate_id):
        """评估候选解"""
        candidate_cost = evaluate_position(candidate)  # 确保utils.py中的evaluate_position函数接受正确的参数
        print(f"候选解 {candidate_id} 的适应值：{candidate_cost}\n")
        return candidate_cost

    def _run_particle_swarm(self, initial_position, candidate_id):
        """运行粒子群优化"""
        # 设置粒子群的搜索范围
        Particle.Particle.set_search_bounds(
            lower=self.bounds[:, 0],
            upper=self.bounds[:, 1]
        )
        
        # 创建粒子群
        swarm = Swarm(self.pso_popsize, self.pso_iterations, initial_position)
        self.current_swarm = swarm
        
        # 运行PSO优化
        best_position, best_cost, history = swarm.optimize(self.config)
        
        # 更新所有粒子访问过的区域
        for pos in history['positions']:
            region = self._get_region_for_position(pos)
            if region:
                region.update(pos, evaluate_position(pos))
        
        # 更新记录
        record = self.local_best_records[candidate_id]
        if best_cost < record['best_cost']:
            record['best_cost'] = best_cost
            record['best_position'] = best_position.copy()
            record['inactive_iterations'] = 0
        else:
            record['inactive_iterations'] += 1
        
        # 更新全局最优
        if best_cost < self.global_best_cost:
            self.global_best_cost = best_cost
            self.global_best_position = best_position.copy()
        
        return swarm  # 返回粒子群对象以供可视化

    def _update_bayesian_model(self, history):
        """更新贝叶斯模型"""
        if not hasattr(self.bayes_opt, 'update'):
            return
        
        # 收集PSO迭代过程中的所有位置和对应的适应度值
        positions = history['positions']
        costs = history['costs']
        
        # 过滤掉重复的位置
        unique_positions = []
        unique_costs = []
        seen_positions = set()
        
        for pos, cost in zip(positions, costs):
            pos_tuple = tuple(pos)
            if pos_tuple not in seen_positions:
                seen_positions.add(pos_tuple)
                unique_positions.append(pos)
                unique_costs.append(cost)
        
        # 更新贝叶斯模型
        if unique_positions and unique_costs:
            self.bayes_opt.update(unique_positions, unique_costs)

    def _update_history(self, positions, costs, record):
        """更新优化历史记录"""
        for pos, cost in zip(positions, costs):
            self.history['positions'].append(pos)
            self.history['costs'].append(cost)
            self.history['candidate_id'].append(record['candidate_id'])

            if cost < record['best_cost']:
                record['best_cost'] = cost
                record['best_position'] = pos

    def _deactivate_candidate(self, candidate_id, record):
        """停用候选点"""
        record['is_active'] = False
        self.active_candidates.remove(candidate_id)
        print(f"候选点 {candidate_id} 停止搜索，最佳成本：{record['best_cost']}")

    def _get_global_best(self):
        """获取全局最优解"""
        self.best_candidate_id = None

        for candidate_id, record in self.local_best_records.items():
            if record['best_cost'] < self.global_best_cost:
                self.global_best_cost = record['best_cost']
                self.global_best_position = record['best_position']
                self.best_candidate_id = candidate_id

        return self.global_best_position, self.global_best_cost, self.local_best_records


    def _get_region_key(self, position):
        """获取位置所在的区域键值"""
        region_indices = []
        for i, (pos, (size_x, size_y)) in enumerate(zip(position, self.region_size)):
            idx = int((pos - self.bounds[i][0]) / size_x)
            region_indices.append(idx)
        return f"region_{region_indices[0]}_{region_indices[1]}"

    def _find_nearby_best(self, candidate):
        """
        找到候选点附近区域的最优解
        """
        region_key = self._get_region_key(candidate)
        nearby_regions = self._get_nearby_regions(region_key)

        nearby_best_cost = float('inf')
        nearby_best_position = None

        # 检查邻近区域的历史记录
        for r_key in nearby_regions:
            if r_key in self.explored_regions:
                region_record = self.explored_regions[r_key]
                if region_record['best_cost'] < nearby_best_cost:
                    nearby_best_cost = region_record['best_cost']
                    nearby_best_position = region_record['best_position']

        return nearby_best_cost, nearby_best_position

    def _get_nearby_regions(self, region_id):
        """获取邻近区域"""
        x_idx, y_idx = region_id
        x_divisions, y_divisions = self.region_indices.shape
        nearby_regions = []
        
        # 获取3x3邻域内的区域
        for i in range(max(0, x_idx-1), min(x_divisions, x_idx+2)):
            for j in range(max(0, y_idx-1), min(y_divisions, y_idx+2)):
                if (i, j) != region_id:  # 排除当前区域
                    nearby_regions.append(self.regions[(i, j)])
        
        return nearby_regions

    def _update_region_record(self, position, cost):
        """更新区域记录"""
        region_key = self._get_region_key(position)
        
        if region_key not in self.explored_regions:
            self.explored_regions[region_key] = {
                'visit_count': 0,
                'best_cost': float('inf'),
                'best_position': None,
                'history': [],
                'pso_applied': False
            }
        
        region = self.explored_regions[region_key]
        region['visit_count'] += 1
        region['history'].append({
            'position': position.tolist(),
            'cost': cost
        })
        
        if cost < region['best_cost']:
            region['best_cost'] = cost
            region['best_position'] = position.tolist()

    def _is_promising_candidate(self, candidate, candidate_cost, candidate_id):
        """
        评估候选点是否值得进一步探索
        Args:
            candidate: 候选点位置
            candidate_cost: 候选点成本
            candidate_id: 候选点ID
        Returns:
            bool: 是否值得继续探索
        """
        region_key = self._get_region_key(candidate)
        region = self.explored_regions.get(region_key)
        
        if region is None:
            return True
        
        # 多准则判断
        criteria = [
            candidate_cost < self.global_best_cost * 1.1,  # 接近全局最优
            not region['pso_applied'] and region['visit_count'] <= self.region_visit_threshold,  # 区域未充分探索且未应用PSO
            candidate_cost < region['best_cost'] * 1.05,  # 接近区域最优
            len(region['history']) < 3,  # 区域样本数较少
        ]
        
        # 如果没有满足任何准则，将候选点设置为非活跃
        if not any(criteria):
            record = self.local_best_records.get(candidate_id)
            if record:
                record['active'] = False
                if candidate_id in self.active_candidates:
                    self.active_candidates.remove(candidate_id)
        
        return any(criteria)

    def _process_candidate(self, candidate, candidate_id):
        """处理候选点"""
        print(f"\n=== 处理候选点 {candidate_id} ===")
        print(f"候选点位置: {candidate}")
        
        # 获取和打印区域信息
        region = self._get_region_for_position(candidate)
        if region:
            print(f"所在区域: {region.region_id}")
            print(f"区域访问次数: {region.visit_count}")
            print(f"区域最优值: {region.best_cost}")
        
        # 确保候选点记录已初始化
        if candidate_id not in self.local_best_records:
            self._init_new_candidate(candidate_id)
        
        # 评估候选点
        candidate_cost = self._evaluate_candidate(candidate, candidate_id)
        region = self._get_region_for_position(candidate)
        
        if region:
            region.update(candidate, candidate_cost)
        
        record = self.local_best_records[candidate_id]
        record['region_id'] = region.region_id if region else None
        
        # 更新记录
        record['current_cost'] = candidate_cost
        if candidate_cost < record['best_cost']:
            record['best_cost'] = candidate_cost
            record['best_position'] = candidate.copy()
        
        # 更新全局最优
        if candidate_cost < self.global_best_cost:
            self.global_best_cost = candidate_cost
            self.global_best_position = candidate.copy()

        # 更新区域记录
        self._update_region_record(candidate, candidate_cost)

        # 判断是否有希望
        if self._is_promising_candidate(candidate, candidate_cost, candidate_id):
            # 启动粒子群优化
            self._run_particle_swarm(candidate, candidate_id)

        # 打印候选点信息
        print(f"候选点 {candidate_id}:")
        print(f"  位置: x1={candidate[0]:.6f}, x2={candidate[1]:.6f}")
        print(f"  当前成本: {candidate_cost:.6f}")
        print(f"  局部最优: {record['best_cost']:.6f}")
        print(f"  全局最优: {self.global_best_cost:.6f}")
        print(f"  活跃状态: {record['active']}\n")

    def get_current_swarm(self):
        """
        获取当前粒子群对象
        """
        # 在 _run_particle_swarm 中，粒子群是通过 Swarm 类创建的
        # 这里可以存储并返回当前的 Swarm 对象
        return self.current_swarm

    def _select_next_region(self):
        """选择下一个待探索区域"""
        unexplored_regions = []
        for x_idx in range(-10, 11):  # 根据搜索范围调整
            for y_idx in range(-10, 11):
                region_key = f"region_{x_idx}_{y_idx}"
                if region_key not in self.explored_regions:
                    unexplored_regions.append((x_idx, y_idx))
        
        if unexplored_regions:
            x_idx, y_idx = random.choice(unexplored_regions)
            x = self.bounds[0][0] + (x_idx + 0.5) * self.region_size[0][0]
            y = self.bounds[1][0] + (y_idx + 0.5) * self.region_size[1][0]
            return np.array([x, y])
        return None

    def _get_region_for_position(self, position):
        """获取位置所属的区域"""
        x_idx = int((position[0] - self.bounds[0][0]) / self.region_size[0][0])
        y_idx = int((position[1] - self.bounds[1][0]) / self.region_size[1][0])
        
        # 确保索引在有效范围内
        x_divisions, y_divisions = self.region_indices.shape
        if 0 <= x_idx < x_divisions and 0 <= y_idx < y_divisions:
            region_id = (x_idx, y_idx)
            return self.regions.get(region_id)
        return None

    def _select_promising_region(self):
        """选择有价值的区域进行探索"""
        unexplored_regions = [r for r in self.regions.values() if r.visit_count == 0]
        if unexplored_regions:
            return random.choice(unexplored_regions)
        
        # 基于探索价值选择区域
        regions = list(self.regions.values())
        values = [r.exploration_value for r in regions]
        total_value = sum(values)
        if total_value == 0:
            return random.choice(regions)
        
        probs = [v/total_value for v in values]
        return np.random.choice(regions, p=probs)

    def _generate_candidate_in_region(self, region):
        """在指定区域内生成候选点"""
        x = np.random.uniform(region.bounds[0][0], region.bounds[0][1])
        y = np.random.uniform(region.bounds[1][0], region.bounds[1][1])
        return np.array([x, y])

class Region:
    """区域类，用于管理搜索空间的子区域"""
    def __init__(self, region_id, bounds):
        self.region_id = region_id
        self.bounds = bounds  # [[x_min, x_max], [y_min, y_max]]
        self.best_position = None
        self.best_cost = float('inf')
        self.visit_count = 0
        self.particles = []  # 存储在该区域的粒子
        self.exploration_value = 1.0  # 区域的探索价值
        self.last_improvement = 0  # 上次改进的迭代次数

    def update(self, position, cost):
        """更新区域信息"""
        self.visit_count += 1
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_position = position.copy()
            self.last_improvement = 0
        else:
            self.last_improvement += 1
        
        # 更新探索价值
        self.exploration_value = self._calculate_exploration_value()

    def _calculate_exploration_value(self):
        """计算区域的探索价值"""
        if self.visit_count == 0:
            return 1.0
        
        # 基于访问次数和改进情况计算探索价值
        visit_factor = np.exp(-0.1 * self.visit_count)
        improvement_factor = np.exp(-0.05 * self.last_improvement)
        return visit_factor * improvement_factor