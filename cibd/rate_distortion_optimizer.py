import torch
import torch.nn as nn
import numpy as np
from collections import deque


class RateDistortionOptimizer:
    """
    率失真优化器
    动态调整β参数以在率失真曲线上找到最优工作点
    """
    def __init__(self, initial_beta=1.0, target_compression=0.5, 
                 adaptation_rate=0.01, window_size=100):
        self.beta = initial_beta
        self.target_compression = target_compression
        self.adaptation_rate = adaptation_rate
        
        # 历史记录
        self.rate_history = deque(maxlen=window_size)
        self.distortion_history = deque(maxlen=window_size)
        self.beta_history = deque(maxlen=window_size)
        
        # Pareto前沿估计
        self.pareto_points = []
        
    def estimate_pareto_slope(self):
        """估计当前工作点在Pareto前沿上的斜率"""
        if len(self.rate_history) < 10:
            return 1.0
            
        rates = np.array(list(self.rate_history))
        distortions = np.array(list(self.distortion_history))
        
        # 使用最近的点估计局部斜率
        if len(rates) > 1:
            # 线性回归
            slope = np.polyfit(rates[-10:], distortions[-10:], 1)[0]
            return abs(slope)
        return 1.0
    
    def compute_optimal_beta(self, current_rate, current_distortion):
        """
        根据当前率和失真计算最优β
        使用拉格朗日乘数法
        """
        # 记录历史
        self.rate_history.append(current_rate)
        self.distortion_history.append(current_distortion)
        self.beta_history.append(self.beta)
        
        # 计算压缩率误差
        compression_error = current_rate - self.target_compression
        
        # 估计Pareto斜率
        pareto_slope = self.estimate_pareto_slope()
        
        # 自适应调整β
        # 如果压缩不够（率太高），增加β
        # 如果压缩过度（率太低），减小β
        beta_adjustment = self.adaptation_rate * compression_error * pareto_slope
        
        # 指数移动平均更新
        self.beta = self.beta * np.exp(beta_adjustment)
        
        # 限制β的范围
        self.beta = np.clip(self.beta, 0.01, 100.0)
        
        return self.beta
    
    def step(self, task_loss, rd_loss):
        """
        优化步骤
        Args:
            task_loss: 任务损失（如分类损失）
            rd_loss: 率失真损失
        """
        # 分解率失真损失
        current_rate = rd_loss.detach().cpu().item()
        current_distortion = task_loss.detach().cpu().item()
        
        # 计算最优β
        optimal_beta = self.compute_optimal_beta(current_rate, current_distortion)
        
        # 更新Pareto前沿
        self.update_pareto_frontier(current_rate, current_distortion)
        
        return optimal_beta
    
    def update_pareto_frontier(self, rate, distortion):
        """更新Pareto前沿估计"""
        # 添加新点
        new_point = (rate, distortion)
        
        # 过滤被支配的点
        non_dominated = []
        for point in self.pareto_points:
            # 如果新点被支配，不添加
            if point[0] <= rate and point[1] <= distortion:
                return
            # 如果旧点被支配，不保留
            if not (rate <= point[0] and distortion <= point[1]):
                non_dominated.append(point)
                
        non_dominated.append(new_point)
        self.pareto_points = non_dominated
    
    def get_pareto_frontier(self):
        """获取当前Pareto前沿"""
        if not self.pareto_points:
            return [], []
            
        # 按率排序
        sorted_points = sorted(self.pareto_points, key=lambda x: x[0])
        rates = [p[0] for p in sorted_points]
        distortions = [p[1] for p in sorted_points]
        
        return rates, distortions


class AdaptiveRateScheduler:
    """
    自适应压缩率调度器
    在训练过程中逐步增加压缩强度
    """
    def __init__(self, initial_rate=0.1, target_rate=0.5, 
                 warmup_steps=1000, total_steps=10000):
        self.initial_rate = initial_rate
        self.target_rate = target_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0
        
    def get_compression_rate(self):
        """获取当前压缩率"""
        self.current_step += 1
        
        if self.current_step < self.warmup_steps:
            # 预热阶段：线性增加
            progress = self.current_step / self.warmup_steps
            return self.initial_rate + progress * (self.target_rate - self.initial_rate) * 0.5
        else:
            # 主训练阶段：余弦退火
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            
            cosine_factor = 0.5 * (1 + np.cos(np.pi * (1 - progress)))
            return self.target_rate - cosine_factor * (self.target_rate - self.initial_rate) * 0.5