import torch
import torch.nn as nn
import copy
from transformers import AutoConfig
from llava.model.language_model.llava_llama import LlavaConfig


def create_compressed_student_config(teacher_config, compression_ratio=0.5):
    """
    创建压缩后的学生模型配置
    使用半自动策略：基于压缩率自动计算各维度
    """
    student_config = copy.deepcopy(teacher_config)
    
    # 获取教师模型的原始参数
    teacher_layers = teacher_config.num_hidden_layers
    teacher_hidden = teacher_config.hidden_size
    teacher_intermediate = teacher_config.intermediate_size
    teacher_heads = teacher_config.num_attention_heads
    
    # 半自动计算策略：根据压缩率分配到不同维度
    if compression_ratio <= 0.3:
        # 激进压缩：优先减少层数，其次减少宽度
        layer_ratio = 0.4  # 层数保留40%
        width_ratio = 0.75  # 宽度保留75%
        ffn_ratio = 0.6    # FFN保留60%
    elif compression_ratio <= 0.5:
        # 中等压缩：平衡减少
        layer_ratio = 0.6  # 层数保留60%
        width_ratio = 0.85  # 宽度保留85%
        ffn_ratio = 0.7    # FFN保留70%
    else:
        # 轻度压缩：主要减少层数
        layer_ratio = 0.8  # 层数保留80%
        width_ratio = 0.95  # 宽度保留95%
        ffn_ratio = 0.9    # FFN保留90%
    
    # 计算新的配置（保持维度为合理的倍数）
    student_config.num_hidden_layers = max(4, int(teacher_layers * layer_ratio))
    
    # hidden_size必须能被num_attention_heads整除
    # 同时最好是64的倍数（对GPU友好）
    new_hidden = int(teacher_hidden * width_ratio)
    new_hidden = (new_hidden // 64) * 64  # 向下取整到64的倍数
    new_hidden = max(256, new_hidden)  # 至少256维
    
    # 计算新的注意力头数（必须能整除hidden_size）
    new_heads = int(teacher_heads * width_ratio)
    # 确保hidden_size能被heads整除
    while new_hidden % new_heads != 0 and new_heads > 1:
        new_heads -= 1
    new_heads = max(4, new_heads)  # 至少4个头
    
    # FFN维度（通常是hidden_size的倍数）
    new_intermediate = int(teacher_intermediate * ffn_ratio)
    # 确保是256的倍数（对GPU友好）
    new_intermediate = (new_intermediate // 256) * 256
    new_intermediate = max(512, new_intermediate)
    
    # 应用计算后的配置
    student_config.num_hidden_layers = student_config.num_hidden_layers
    student_config.hidden_size = new_hidden
    student_config.intermediate_size = new_intermediate
    student_config.num_attention_heads = new_heads
    
    # 如果使用了GQA（grouped-query attention），也需要调整
    if hasattr(teacher_config, 'num_key_value_heads'):
        new_kv_heads = max(1, new_heads // 4)  # 通常kv_heads是heads的1/4
        student_config.num_key_value_heads = new_kv_heads
    
    # 添加压缩相关配置
    student_config.compression_ratio = compression_ratio
    student_config.use_information_bottleneck = True
    
    # 打印压缩配置
    print(f"\n{'='*50}")
    print("Student Model Configuration:")
    print(f"{'='*50}")
    print(f"Layers: {teacher_layers} -> {student_config.num_hidden_layers} ({layer_ratio:.0%})")
    print(f"Hidden: {teacher_hidden} -> {new_hidden} ({new_hidden/teacher_hidden:.0%})")
    print(f"FFN: {teacher_intermediate} -> {new_intermediate} ({new_intermediate/teacher_intermediate:.0%})")
    print(f"Heads: {teacher_heads} -> {new_heads} ({new_heads/teacher_heads:.0%})")
    
    # 估算参数量减少
    teacher_params_est = estimate_model_params(teacher_config)
    student_params_est = estimate_model_params(student_config)
    actual_compression = student_params_est / teacher_params_est
    print(f"Estimated compression: {actual_compression:.2%} (target: {compression_ratio:.0%})")
    print(f"{'='*50}\n")
    
    return student_config


def estimate_model_params(config):
    """估算模型参数量"""
    params = 0
    vocab_size = getattr(config, 'vocab_size', 32000)
    hidden = config.hidden_size
    layers = config.num_hidden_layers
    intermediate = config.intermediate_size
    
    # Embeddings
    params += vocab_size * hidden
    
    # Each transformer layer
    per_layer = 0
    # Attention (Q,K,V,O)
    per_layer += 4 * hidden * hidden
    # MLP
    per_layer += 3 * hidden * intermediate
    # LayerNorms
    per_layer += 4 * hidden
    
    params += layers * per_layer
    
    # LM head
    params += vocab_size * hidden
    
    return params


def initialize_student_from_teacher(student_model, teacher_model, student_config):
    """
    从教师模型初始化学生模型权重
    智能处理维度不匹配问题
    """
    teacher_state = teacher_model.state_dict()
    student_state = student_model.state_dict()
    
    # 统计信息
    initialized = 0
    skipped = 0
    truncated = 0
    
    # 层映射策略
    teacher_layers = teacher_model.config.num_hidden_layers
    student_layers = student_config.num_hidden_layers
    
    # 均匀选择要保留的层
    layer_indices = torch.linspace(0, teacher_layers-1, student_layers).long().tolist()
    layer_map = {i: layer_indices[i] for i in range(student_layers)}
    
    print(f"Layer mapping: {layer_map}")
    
    # 遍历学生模型的所有参数
    for name, param in student_state.items():
        # 跳过IB模块和特殊投影层
        if 'visual_compressor' in name or 'feature_projector' in name or 'beta' in name:
            skipped += 1
            continue
            
        # 处理transformer层
        if 'model.layers' in name or 'layers.' in name:
            # 提取层号
            import re
            layer_match = re.search(r'layers\.(\d+)\.', name)
            if layer_match:
                student_layer_idx = int(layer_match.group(1))
                
                if student_layer_idx in layer_map:
                    teacher_layer_idx = layer_map[student_layer_idx]
                    # 构造对应的教师参数名
                    teacher_name = name.replace(
                        f'layers.{student_layer_idx}.', 
                        f'layers.{teacher_layer_idx}.'
                    )
                    
                    if teacher_name in teacher_state:
                        teacher_param = teacher_state[teacher_name]
                        
                        # 智能初始化
                        if param.shape == teacher_param.shape:
                            # 维度完全匹配
                            param.data.copy_(teacher_param.data)
                            initialized += 1
                        else:
                            # 维度不匹配，需要截断或填充
                            if init_param_with_mismatch(param, teacher_param, name):
                                truncated += 1
                            else:
                                skipped += 1
                                
        # 处理embedding和其他层
        elif name in teacher_state:
            teacher_param = teacher_state[name]
            
            if param.shape == teacher_param.shape:
                param.data.copy_(teacher_param.data)
                initialized += 1
            else:
                if init_param_with_mismatch(param, teacher_param, name):
                    truncated += 1
                else:
                    skipped += 1
    
    print(f"\nWeight initialization statistics:")
    print(f"  Initialized (exact match): {initialized}")
    print(f"  Truncated/Padded: {truncated}")
    print(f"  Skipped: {skipped}")
    print(f"  Total: {len(student_state)}")
    
    return student_model


def init_param_with_mismatch(student_param, teacher_param, name):
    """
    处理维度不匹配的参数初始化
    返回是否成功初始化
    """
    s_shape = student_param.shape
    t_shape = teacher_param.shape
    
    # 只处理相同维数的tensor
    if len(s_shape) != len(t_shape):
        return False
    
    try:
        if len(s_shape) == 1:
            # 一维向量（如LayerNorm的weight/bias）
            min_size = min(s_shape[0], t_shape[0])
            student_param.data[:min_size] = teacher_param.data[:min_size]
            
        elif len(s_shape) == 2:
            # 二维矩阵（如Linear层的weight）
            min_rows = min(s_shape[0], t_shape[0])
            min_cols = min(s_shape[1], t_shape[1])
            student_param.data[:min_rows, :min_cols] = teacher_param.data[:min_rows, :min_cols]
            
            # 如果学生参数更大，用xavier初始化剩余部分
            if s_shape[0] > t_shape[0] or s_shape[1] > t_shape[1]:
                nn.init.xavier_uniform_(student_param.data)
                # 保留已复制的部分
                student_param.data[:min_rows, :min_cols] = teacher_param.data[:min_rows, :min_cols]
                
        else:
            # 更高维度的tensor
            # 简单策略：只复制能匹配的部分
            slices = tuple(slice(0, min(s, t)) for s, t in zip(s_shape, t_shape))
            student_param.data[slices] = teacher_param.data[slices]
            
        return True
        
    except Exception as e:
        print(f"Failed to initialize {name}: {e}")
        return False


def analyze_compression(teacher_model, student_model):
    """
    分析压缩效果
    """
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())
    
    # 分层统计
    teacher_layer_params = {}
    student_layer_params = {}
    
    for name, param in teacher_model.named_parameters():
        layer_type = name.split('.')[0]
        if layer_type not in teacher_layer_params:
            teacher_layer_params[layer_type] = 0
        teacher_layer_params[layer_type] += param.numel()
    
    for name, param in student_model.named_parameters():
        layer_type = name.split('.')[0]
        if layer_type not in student_layer_params:
            student_layer_params[layer_type] = 0
        student_layer_params[layer_type] += param.numel()
    
    print(f"\n{'='*50}")
    print("Compression Analysis:")
    print(f"{'='*50}")
    print(f"Total parameters:")
    print(f"  Teacher: {teacher_params/1e6:.2f}M")
    print(f"  Student: {student_params/1e6:.2f}M")
    print(f"  Reduction: {(1 - student_params/teacher_params)*100:.1f}%")
    
    print(f"\nPer-module compression:")
    for module in teacher_layer_params:
        if module in student_layer_params:
            t_params = teacher_layer_params[module]
            s_params = student_layer_params[module]
            reduction = (1 - s_params/t_params) * 100
            print(f"  {module}: {t_params/1e6:.2f}M -> {s_params/1e6:.2f}M ({reduction:.1f}% reduced)")
    
    return {
        'teacher_params': teacher_params,
        'student_params': student_params,
        'compression_ratio': student_params / teacher_params,
        'reduction_percentage': (1 - student_params/teacher_params) * 100
    }
