#!/usr/bin/env python3
"""
NLP组件统一配置模块
"""

import os
from typing import Optional
from src.infrastructure.utils.config_manager import config_manager

class NLPConfig:
    """NLP组件统一配置类"""
    
    def __init__(self):
        # 从配置管理器获取NLP配置 - 启用mock模式以绕过网络连接问题
        self.use_mock = config_manager.get("nlp.use_mock", True)
        
        # Sentence Transformers配置 - 使用本地模型
        self.sentence_transformer_model = config_manager.get(
            "nlp.sentence_transformer_model", 
            os.path.abspath("./models/sentence_transformers_new/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/e8f8c211226b894fcb81acc59f3b34ba3efd5f42")
        )
        
        # 本地模型缓存目录（绝对路径）
        self.sentence_transformer_cache = config_manager.get(
            "nlp.sentence_transformer_cache", 
            os.path.abspath("./models/sentence_transformers_new")
        )
        
        # 设备配置（cpu或cuda）
        self.device = config_manager.get("nlp.device", "cpu")
        
        # 确保缓存目录存在
        os.makedirs(self.sentence_transformer_cache, exist_ok=True)
    
    @classmethod
    def get_instance(cls) -> "NLPConfig":
        """获取单例实例"""
        if not hasattr(cls, "_instance"):
            cls._instance = cls()
        return cls._instance

# 创建全局配置实例
nlp_config = NLPConfig.get_instance()
