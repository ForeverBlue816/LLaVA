import torch
import copy
from transformers import AutoConfig
from llava.model.language_model.llava_llama import LlavaConfig


def create_compressed_student_config(teacher_config, compression_ratio=0.5):
    """
    创建压缩后的学生模型配置
    真正减少参数量
    """
    student_config = copy.deepcopy(teacher_config)
    
    # 根据压缩率调整模型大小
    if compression_ratio <= 0.3:
        # 激进压缩：大幅减少层数和宽度
        student_config.num_hidden_layers = max(6, int(teacher_config.num_hidden_layers * 0.4))
        student_config.intermediate_size = int(teacher_config.intermediate_size * 0.5)
        student_config.num_attention_heads = max(8, int(teacher_config.num_attention_heads * 0.5))
        
    elif compression_ratio <= 0.5:
        # 中等压缩：适度减少层数和宽度
        student_config.num_hidden_layers = max(12, int(teacher_config.num_hidden_layers * 0.6))
        student_config.intermediate_size = int(teacher_config.intermediate_size * 0.7)
        student_config.num_attention_heads = max(12, int(teacher_config.num_attention_heads * 0.75))
        
    else:
        # 轻度压缩：主要减少层数
        student_config.num_hidden_layers = max(18, int(teacher_config.num_hidden_layers * 0.8))
        student_config.intermediate_size = int(teacher_config.intermediate_size * 0.9)
    
    # 可选：减少视觉编码器大小
    if hasattr(student_config, 'vision_config'):
        # 减少视觉patch大小或层数
        pass  # 根据需求调整
    
    # 添加压缩相关配置
    student_config.compression_ratio = compression_ratio
    student_config.use_information_bottleneck = True
    
    return student_config


def initialize_student_from_teacher(student_model, teacher_model, student_config):
    """
    从教师模型初始化学生模型权重
    处理维度不匹配问题
    """
    teacher_state = teacher_model.state_dict()
    student_state = student_model.state_dict()
    
    # 映射策略：均匀选择层
    teacher_layers = teacher_model.config.num_hidden_layers
    student_layers = student_config.num_hidden_layers
    
    # 计算层映射
    layer_map = {}
    for i in range(student_layers):
        # 均匀映射：学生第i层对应教师第j层
        j = int(i * teacher_layers / student_layers)
        layer_map[i] = j
    
    # 复制权重
    for name, param in student_state.items():
        if 'model.layers' in name:
            # 提取层号
            parts = name.split('.')
            layer_idx = int(parts[2])
            
            if layer_idx in layer_map:
                # 构造对应的教师层参数名
                teacher_layer_idx = layer_map[layer_idx]
                teacher_name = name.replace(f'layers.{layer_idx}', f'layers.{teacher_layer_idx}')
                
                if teacher_name in teacher_state:
                    teacher_param = teacher_state[teacher_name]
                    
                    # 处理维度不匹配
                    if param.shape == teacher_param.shape:
                        param.data.copy_(teacher_param.data)
                    elif len(param.shape) == len(teacher_param.shape):
                        # 尝试截断或插值
                        if param.shape[0] <= teacher_param.shape[0]:
                            param.data.copy_(teacher_param.data[:param.shape[0]])
                        else:
                            # 需要更复杂的初始化
                            nn.init.xavier_uniform_(param.data)
        
        elif name in teacher_state and param.shape == teacher_state[name].shape:
            # 直接复制相同维度的参数
            param.data.copy_(teacher_state[name].data)
    
    return student_model