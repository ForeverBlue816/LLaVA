"""
数据加载和预处理工具
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LLaVADataCollator:
    """
    LLaVA数据整理器
    """
    tokenizer: any
    image_processor: any
    
    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        整理batch数据
        """
        # 提取各个字段
        input_ids = [instance['input_ids'] for instance in instances]
        labels = [instance['labels'] for instance in instances]
        images = [instance['image'] for instance in instances if 'image' in instance]
        
        # Padding
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100  # ignore_index
        )
        
        # Attention mask
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        
        batch = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }
        
        # 处理图像
        if images:
            # 使用image_processor处理图像
            if hasattr(self.image_processor, 'preprocess'):
                images_tensor = self.image_processor.preprocess(
                    images, 
                    return_tensors='pt'
                )['pixel_values']
            else:
                images_tensor = torch.stack([
                    self.image_processor(img) for img in images
                ])
            
            batch['images'] = images_tensor
        
        return batch


class LLaVADataset(Dataset):
    """
    LLaVA数据集
    支持多种格式：LLaVA-150K, COCO, VQA等
    """
    
    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer,
        image_processor,
        max_length: int = 2048,
        is_training: bool = True
    ):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.is_training = is_training
        self.image_folder = image_folder
        
        # 加载数据
        print(f"Loading data from {data_path}...")
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        """
        item = self.data[idx]
        
        # 构建对话
        conversations = item.get('conversations', [])
        
        # 格式化对话
        text = self._format_conversations(conversations)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'][0]
        
        # 创建labels (训练时才需要)
        if self.is_training:
            labels = input_ids.clone()
            # 可以在这里mask掉人类输入部分，只计算模型输出的loss
            # 这需要根据你的对话格式来定制
        else:
            labels = input_ids.clone()
        
        result = {
            'input_ids': input_ids,
            'labels': labels
        }
        
        # 加载图像
        if 'image' in item:
            image_path = os.path.join(self.image_folder, item['image'])
            try:
                image = Image.open(image_path).convert('RGB')
                result['image'] = image
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                # 返回空白图像
                result['image'] = Image.new('RGB', (224, 224), color='white')
        
        return result
    
    def _format_conversations(self, conversations: List[Dict]) -> str:
        """
        格式化对话为文本
        
        LLaVA格式示例:
        [
            {"from": "human", "value": "<image>\nWhat is in this image?"},
            {"from": "gpt", "value": "This is a cat."}
        ]
        """
        formatted_text = ""
        
        for conv in conversations:
            role = conv.get('from', 'human')
            content = conv.get('value', '')
            
            if role == 'human':
                formatted_text += f"USER: {content}\n"
            elif role == 'gpt':
                formatted_text += f"ASSISTANT: {content}\n"
        
        return formatted_text


def make_supervised_data_module(
    tokenizer,
    image_processor,
    data_args: Dict
) -> Dict:
    """
    创建训练和验证数据集
    
    Args:
        tokenizer: 分词器
        image_processor: 图像处理器
        data_args: 数据参数，包含:
            - train_data_path: 训练数据路径
            - eval_data_path: 验证数据路径
            - image_folder: 图像文件夹
            - max_length: 最大长度
    """
    # 训练数据集
    train_dataset = None
    if data_args.get('train_data_path'):
        train_dataset = LLaVADataset(
            data_path=data_args['train_data_path'],
            image_folder=data_args['image_folder'],
            tokenizer=tokenizer,
            image_processor=image_processor,
            max_length=data_args.get('max_length', 2048),
            is_training=True
        )
    
    # 验证数据集
    eval_dataset = None
    if data_args.get('eval_data_path'):
        eval_dataset = LLaVADataset(
            data_path=data_args['eval_data_path'],
            image_folder=data_args['image_folder'],
            tokenizer=tokenizer,
            image_processor=image_processor,
            max_length=data_args.get('max_length', 2048),
            is_training=False
        )
    
    # 数据整理器
    data_collator = LLaVADataCollator(
        tokenizer=tokenizer,
        image_processor=image_processor
    )
    
    return {
        'train_dataset': train_dataset,
        'eval_dataset': eval_dataset,
        'data_collator': data_collator
    }
