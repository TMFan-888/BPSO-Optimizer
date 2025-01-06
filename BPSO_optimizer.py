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
        self.region_size = config.get('region_size', [(0.5, 0.5), (0.5, 0.5)])
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

    def optimize(self):
        """执行BPSO优化过程"""

        # 第一阶段：初始采样
        print("执行初始采样阶段...")
        initial_candidates = self.bayes_opt.lhs_samples(self.bounds, num_samples=self.initial_samples)

        for idx, candidate in enumerate(initial_candidates):
            candidate_id = f"initial_{idx}"
            self._init_new_candidate(candidate_id)
            record = self.local_best_records[candidate_id]

            # 评估候选解
            candidate_cost = self._evaluate_candidate(candidate, candidate_id)

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

            print(f"初始点 {candidate_id}:")
            print(f"  位置: x1={candidate[0]:.6f}, x2={candidate[1]:.6f}")
            print(f"  成本: {candidate_cost:.6f}\n")

        # 使用初始采样数据更新贝叶斯模型
        if hasattr(self.bayes_opt, 'update'):
            positions = [r['best_position'] for r in self.local_best_records.values()
                        if r['best_position'] is not None]
            costs = [r['best_cost'] for r in self.local_best_records.values()
                    if r['best_position'] is not None]
            if positions and costs:
                self.bayes_opt.update(positions, costs)
                print("使用初始采样数据更新贝叶斯模型完成\n")

        # 第二阶段：贝叶斯优化迭代
        for i in range(self.Bayesian_T):
            print(f"执行第 {i + 1} 次贝叶斯循环\n")
            
            # 有概率探索新区域
            if random.random() < 0.2:  # 20%概率探索新区域
                new_candidate = self._select_next_region()
                if new_candidate is not None:
                    candidate_id = f"candidate_{i}_new"
                    self._init_new_candidate(candidate_id)
                    self._process_candidate(new_candidate, candidate_id)
            
            # 生成并处理常规候选点
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
                
                if candidate_id not in self.local_best_records:
                    self._init_new_candidate(candidate_id)
                    if not self.local_best_records[candidate_id]['active']:
                        continue

                record = self.local_best_records[candidate_id]
                candidate_cost = self._evaluate_candidate(candidate, candidate_id)
                
                self._update_region_record(candidate, candidate_cost)
                record['history'].append({
                    'position': candidate,
                    'cost': candidate_cost
                })

                # 更新局部最优和全局最优
                if candidate_cost < record['best_cost']:
                    record['best_cost'] = candidate_cost
                    record['best_position'] = candidate
                    if candidate_cost < self.global_best_cost:
                        self.global_best_cost = candidate_cost
                        self.global_best_position = candidate

                # 判断是否继续搜索
                if self._is_promising_candidate(candidate, candidate_cost):
                    record['inactive_iterations'] = 0
                    self._run_particle_swarm(candidate, candidate_id)
                else:
                    record['inactive_iterations'] += 1
                    if record['inactive_iterations'] >= self.max_inactive_iterations:
                        record['active'] = False
                        self.active_candidates.remove(candidate_id)

                print(f"候选点 {candidate_id}:")
                print(f"  位置: x1={candidate[0]:.6f}, x2={candidate[1]:.6f}")
                print(f"  当前成本: {candidate_cost:.6f}")
                print(f"  局部最优: {record['best_cost']:.6f}")
                print(f"  全局最优: {self.global_best_cost:.6f}")
                print(f"  活跃状态: {record['active']}\n")

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
            lower=self.bounds[:, 0],  # 每个维度的下界
            upper=self.bounds[:, 1]   # 每个维度的上界
        )
        
        # 创建粒子群
        swarm = Swarm(self.pso_popsize, self.pso_iterations, initial_position)
        self.current_swarm = swarm
        
        # 运行PSO优化
        best_position, best_cost, history = swarm.optimize(self.config)
        
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
        
        # 检查是否需要停用该候选点
        if record['inactive_iterations'] >= self.max_inactive_iterations:
            record['active'] = False
            self.active_candidates.remove(candidate_id)

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

    def _get_nearby_regions(self, region_key):
        """
        获取邻近区域的keys
        """
        lr_idx, bs_idx = map(int, region_key.split('_')[1:])
        nearby_regions = []

        # 获取3x3邻域内的区域
        for i in range(-1, 2):
            for j in range(-1, 2):
                nearby_key = f"region_{lr_idx + i}_{bs_idx + j}"
                nearby_regions.append(nearby_key)

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

    def _is_promising_candidate(self, candidate, candidate_cost):
        """评估候选点是否值得进一步探索"""
        region_key = self._get_region_key(candidate)
        region = self.explored_regions.get(region_key)
        
        if region is None:
            return True
        
        # 多准则判断
        criteria = [
            candidate_cost < self.global_best_cost * 1.1,  # 接近全局最优
            not region['pso_applied'] and region['visit_count'] <= self.region_visit_threshold,  # 区域未充分探索
            candidate_cost < region['best_cost'] * 1.05,  # 接近区域最优
            len(region['history']) < 3  # 区域样本数较少
        ]
        
        return any(criteria)

    def _process_candidate(self, candidate, candidate_id):
        """处理候选点"""
        # 评估候选点
        candidate_cost = self._evaluate_candidate(candidate, candidate_id)
        record = self.local_best_records[candidate_id]
        
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
        if self._is_promising_candidate(candidate, candidate_cost):
            # 启动粒子群优化
            self._run_particle_swarm(candidate, candidate_id)

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