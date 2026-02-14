#!/usr/bin/env python3
"""
事件-价格匹配服务
将新闻事件与价格波动进行智能匹配
"""

from typing import List, Dict, Any, Tuple, Optional
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 导入统一NLP配置
from src.nlp.config import nlp_config

logger = logging.getLogger(__name__)

class EventPriceMatcher:
    """
    事件-价格匹配器
    使用语义相似度将新闻事件与价格波动进行匹配
    """
    
    def __init__(self, is_mock: Optional[bool] = None):
        """初始化事件-价格匹配器"""
        self.logger = logging.getLogger(__name__)
        
        # 如果没有指定mock模式，使用全局配置 - 默认启用mock模式以绕过网络连接问题
        self.use_mock = is_mock if is_mock is not None else True
        
        self.logger.info("EventPriceMatcher initialized with mock mode: %s", self.use_mock)
        
        # 如果是mock模式，直接使用简单的相似度计算方法
        if self.use_mock:
            self.use_sentence_transformers = False
            self.logger.info("Mock mode enabled, using simple similarity measure")
            return
            
        # 即使不是mock模式，也准备好fallback方案
        
        # 尝试导入sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            
            # 使用本地模型路径 - 从项目根目录开始构建绝对路径
            import os
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
            local_model_path = os.path.join(
                project_root,
                "models",
                "sentence_transformers_new",
                "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2",
                "snapshots",
                "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
            )
            self.encoder = SentenceTransformer(
                local_model_path,
                cache_folder=nlp_config.sentence_transformer_cache,
                use_auth_token=False,
                device=nlp_config.device,
                local_files_only=True
            )
            self.use_sentence_transformers = True
            self.logger.info("Successfully loaded sentence-transformers model from local cache")
        except ImportError:
            self.use_sentence_transformers = False
            self.logger.warning("sentence-transformers not installed, using simple similarity measure")
        except Exception as e:
            self.use_sentence_transformers = False
            self.logger.error(f"Failed to load sentence-transformers model: {str(e)}", exc_info=True)
            self.logger.warning("Using simple similarity measure as fallback")
    
    def compute_correlation(self, events: List[Dict], price_changes: List[Dict]) -> List[Dict]:
        """
        使用语义相似度匹配事件与价格变化
        
        Args:
            events: 事件列表，每个事件包含description字段
            price_changes: 价格变化列表，每个变化包含description字段
            
        Returns:
            List[Dict]: 匹配结果列表，包含事件、价格变化和相似度分数
        """
        if not events or not price_changes:
            return []
        
        try:
            # 提取事件描述
            event_descriptions = [event["description"] for event in events]
            # 提取价格变化描述
            price_descriptions = [f"价格{change['description']}" for change in price_changes]
            
            # 计算相似度矩阵
            if self.use_sentence_transformers:
                # 使用sentence-transformers计算嵌入
                event_embeddings = self.encoder.encode(event_descriptions)
                price_embeddings = self.encoder.encode(price_descriptions)
                
                # 计算余弦相似度矩阵
                similarity_matrix = cosine_similarity(event_embeddings, price_embeddings)
            else:
                # 使用简单的词袋相似度
                similarity_matrix = self._simple_similarity(event_descriptions, price_descriptions)
            
            # 找到最佳匹配
            matches = self._find_matches(events, price_changes, similarity_matrix)
            
            return matches
        except Exception as e:
            self.logger.error(f"Correlation computation failed: {str(e)}")
            import traceback
            self.logger.debug(f"Exception details: {traceback.format_exc()}")
            raise
    
    def _simple_similarity(self, texts1: List[str], texts2: List[str]) -> np.ndarray:
        """
        简单的词袋相似度计算
        
        Args:
            texts1: 文本列表1
            texts2: 文本列表2
            
        Returns:
            np.ndarray: 相似度矩阵
        """
        # 构建词汇表
        all_words = set()
        for text in texts1 + texts2:
            words = text.lower().split()
            all_words.update(words)
        
        # 创建词袋向量
        def text_to_vector(text: str) -> np.ndarray:
            words = text.lower().split()
            vector = np.zeros(len(all_words))
            for i, word in enumerate(all_words):
                vector[i] = words.count(word)
            return vector
        
        # 生成向量
        vectors1 = np.array([text_to_vector(text) for text in texts1])
        vectors2 = np.array([text_to_vector(text) for text in texts2])
        
        # 计算余弦相似度
        similarity_matrix = np.zeros((len(texts1), len(texts2)))
        for i in range(len(texts1)):
            for j in range(len(texts2)):
                v1 = vectors1[i]
                v2 = vectors2[j]
                dot_product = np.dot(v1, v2)
                norm_v1 = np.linalg.norm(v1)
                norm_v2 = np.linalg.norm(v2)
                if norm_v1 > 0 and norm_v2 > 0:
                    similarity_matrix[i][j] = dot_product / (norm_v1 * norm_v2)
                else:
                    similarity_matrix[i][j] = 0.0
        
        return similarity_matrix
    
    def _find_matches(self, events: List[Dict], price_changes: List[Dict], 
                     similarity_matrix: np.ndarray) -> List[Dict]:
        """
        找到事件与价格变化的最佳匹配
        
        Args:
            events: 事件列表
            price_changes: 价格变化列表
            similarity_matrix: 相似度矩阵
            
        Returns:
            List[Dict]: 匹配结果列表
        """
        matches = []
        
        # 对每个事件找到最相似的价格变化
        for i, event in enumerate(events):
            # 找到最相似的价格变化
            best_match_idx = np.argmax(similarity_matrix[i])
            best_similarity = similarity_matrix[i][best_match_idx]
            
            # 只返回相似度大于阈值的匹配
            if best_similarity > 0.3:
                match = {
                    "event": event,
                    "price_change": price_changes[best_match_idx],
                    "similarity_score": float(best_similarity),
                    "match_strength": self._get_match_strength(best_similarity)
                }
                matches.append(match)
        
        # 按相似度排序
        matches.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return matches
    
    def _get_match_strength(self, similarity_score: float) -> str:
        """
        根据相似度分数获取匹配强度
        
        Args:
            similarity_score: 相似度分数
            
        Returns:
            str: 匹配强度（高、中、低）
        """
        if similarity_score > 0.7:
            return "高"
        elif similarity_score > 0.5:
            return "中"
        else:
            return "低"
