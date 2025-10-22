import json
import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor
import numpy as np
from typing import Dict, List, Optional


class LLaVADataset(Dataset):
    """
    LLaVA-150K数据集类
    用于CIBD蒸馏训练
    """
    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer,
        image_processor,
        max_length: int = 2048,
        is_training: bool = True,
        use_augmentation: bool = True
    ):
        """
        Args:
            data_path: JSON文件路径 (如 llava_instruct_150k.json)
            image_folder: 图像文件夹路径 (如 coco/train2017)
            tokenizer: LLaVA tokenizer
            image_processor: CLIP image processor
            max_length: 最大序列长度
            is_training: 是否为训练模式
            use_augmentation: 是否使用数据增强
        """
        self.data_path = data_path
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.is_training = is_training
        self.use_augmentation = use_augmentation
        
        # 加载数据
        print(f"Loading dataset from {data_path}...")
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} samples")
        
        # 设置特殊token
        self.image_token_id = tokenizer.convert_tokens_to_ids("<image>")
        if self.image_token_id == tokenizer.unk_token_id:
            # 如果没有image token，添加一个
            tokenizer.add_tokens(["<image>"], special_tokens=True)
            self.image_token_id = tokenizer.convert_tokens_to_ids("<image>")
            
    def __len__(self):
        return len(self.data)
    
    def load_image(self, image_path: str) -> Image.Image:
        """加载和预处理图像"""
        full_path = os.path.join(self.image_folder, image_path)
        image = Image.open(full_path).convert('RGB')
        
        # 数据增强（训练时）
        if self.is_training and self.use_augmentation:
            # 随机水平翻转
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            
            # 随机轻微旋转
            if random.random() > 0.7:
                angle = random.uniform(-10, 10)
                image = image.rotate(angle)
                
        return image
    
    def process_conversation(self, conversations: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        处理对话数据
        Args:
            conversations: 对话列表，每个元素包含 'from' 和 'value'
        Returns:
            处理后的input_ids和labels
        """
        # 构建完整对话
        full_text = ""
        labels_list = []
        
        for i, conv in enumerate(conversations):
            role = conv['from']
            value = conv['value']
            
            if role == 'human':
                # 人类输入，添加特殊标记
                if '<image>' in value:
                    # 替换<image>为实际的token
                    value = value.replace('<image>', f' <image> ')
                prompt = f"USER: {value} ASSISTANT: "
                full_text += prompt
                # 人类输入部分不计算loss
                labels_list.extend([-100] * len(self.tokenizer.encode(prompt, add_special_tokens=False)))
                
            elif role == 'gpt':
                # 模型回复，需要计算loss
                response = f"{value}</s>"
                full_text += response
                response_ids = self.tokenizer.encode(response, add_special_tokens=False)
                labels_list.extend(response_ids)
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encoding.input_ids[0]
        attention_mask = encoding.attention_mask[0]
        
        # 创建labels（-100表示不计算loss的位置）
        labels = torch.full_like(input_ids, -100)
        
        # 对齐labels
        if len(labels_list) <= self.max_length:
            labels[:len(labels_list)] = torch.tensor(labels_list)
        else:
            labels = torch.tensor(labels_list[:self.max_length])
            
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        item = self.data[idx]
        
        # 处理图像
        image_path = item.get('image', None)
        if image_path:
            try:
                image = self.load_image(image_path)
                # 使用CLIP processor处理图像
                image_tensor = self.image_processor(image, return_tensors="pt")['pixel_values'][0]
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                # 使用随机噪声作为fallback
                image_tensor = torch.randn(3, 336, 336)
        else:
            # 某些样本可能没有图像
            image_tensor = torch.zeros(3, 336, 336)
        
        # 处理对话
        conversation_data = self.process_conversation(item['conversations'])
        
        return {
            'input_ids': conversation_data['input_ids'],
            'attention_mask': conversation_data['attention_mask'],
            'labels': conversation_data['labels'],
            'images': image_tensor,
            'id': item.get('id', f'sample_{idx}')
        }


class LLaVACollator:
    """
    自定义数据整理器，用于批处理
    """
    def __init__(self, tokenizer, image_processor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """整理批次数据"""
        # 收集所有字段
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        images = torch.stack([item['images'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'images': images,
            'ids': [item['id'] for item in batch]
        }


class LLaVAPretrainingDataset(Dataset):
    """
    用于第一阶段预训练的数据集（可选）
    只训练projection层
    """
    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer,
        image_processor,
        max_length: int = 512
    ):
        self.data_path = data_path
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        
        # 加载预训练数据（通常是图像-标题对）
        with open(data_path, 'r') as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # 加载图像
        image_path = os.path.join(self.image_folder, item['image'])
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.image_processor(image, return_tensors="pt")['pixel_values'][0]
        
        # 简单的图像描述任务
        caption = item.get('caption', '')
        text = f"USER: <image>\nWhat is in this image? ASSISTANT: {caption}</s>"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 只对caption部分计算loss
        labels = encoding.input_ids[0].clone()
        # 找到ASSISTANT:之后的位置
        assistant_idx = (labels == self.tokenizer.encode("ASSISTANT:", add_special_tokens=False)[0]).nonzero()
        if len(assistant_idx) > 0:
            labels[:assistant_idx[0][0]+2] = -100  # +2 to skip "ASSISTANT: "
            
        return {
            'input_ids': encoding.input_ids[0],
            'attention_mask': encoding.attention_mask[0],
            'labels': labels,
            'images': image_tensor
        }