#!/usr/bin/env python3
"""
主题提取服务
自动识别报告中的核心概念和主题
"""

from typing import List, Dict, Any, Optional
import logging
import re
from collections import Counter

# 导入统一NLP配置
from src.nlp.config import nlp_config

logger = logging.getLogger(__name__)

class TopicExtractor:
    """
    主题提取器
    使用BERTopic或LDA提取核心主题
    """
    
    def __init__(self, is_mock: Optional[bool] = None):
        """初始化主题提取器"""
        self.logger = logging.getLogger(__name__)
        
        # 如果没有指定mock模式，使用全局配置 - 默认启用mock模式以绕过网络连接问题
        self.use_mock = is_mock if is_mock is not None else True
        
        self.logger.info("TopicExtractor initialized with mock mode: %s", self.use_mock)
        
        # 如果是mock模式，直接使用简单的主题提取方法
        if self.use_mock:
            self.use_bertopic = False
            self.logger.info("Mock mode enabled, using simple topic extraction")
            return
        
        # 尝试导入BERTopic
        try:
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer
            from sklearn.feature_extraction.text import CountVectorizer
            
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
            
            # 使用本地模型
            self.embedding_model = SentenceTransformer(
                local_model_path,
                cache_folder=nlp_config.sentence_transformer_cache,
                use_auth_token=False,
                device=nlp_config.device,
                local_files_only=True
            )
            
            # 配置BERTopic
            vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words='english')
            self.topic_model = BERTopic(
                embedding_model=self.embedding_model,
                vectorizer_model=vectorizer_model,
                nr_topics='auto',
                calculate_probabilities=True
            )
            
            self.use_bertopic = True
            self.logger.info("Successfully loaded BERTopic model from local cache")
        except ImportError:
            self.use_bertopic = False
            self.logger.warning("BERTopic not installed, using simple topic extraction")
        except Exception as e:
            self.use_bertopic = False
            self.logger.error(f"Failed to load BERTopic model: {str(e)}", exc_info=True)
            self.logger.warning("Using simple topic extraction as fallback")
    
    def extract_topics(self, report_text: str, num_topics: int = 5) -> List[Dict]:
        """
        使用BERTopic或LDA提取核心主题

        Args:
            report_text: 报告文本
            num_topics: 要提取的主题数量

        Returns:
            List[Dict]: 主题列表，每个主题包含topic、keywords和weight
        """
        if not report_text:
            return []

        try:
            # 确保report_text是字符串
            if isinstance(report_text, list):
                report_text = '\n'.join(report_text)
            
            # 文本预处理
            processed_docs = self._preprocess(report_text)

            if self.use_bertopic:
                # 使用BERTopic提取主题
                topics = self._extract_with_bertopic(processed_docs, num_topics)
            else:
                # 使用简单的主题提取
                topics = self._simple_topic_extraction(report_text, num_topics)
            
            return topics
        except Exception as e:
            self.logger.error(f"Topic extraction failed: {str(e)}")
            import traceback
            self.logger.debug(f"Exception details: {traceback.format_exc()}")
            # 使用简单方法作为备用
            return self._simple_topic_extraction(report_text, num_topics)
    
    def _preprocess(self, text: str) -> List[str]:
        """
        文本预处理
        
        Args:
            text: 输入文本
            
        Returns:
            List[str]: 预处理后的文档列表
        """
        # 移除标点符号和特殊字符，保留中文、英文、数字和空白
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
        
        # 分词
        docs = text.split('\n')
        
        # 过滤空行
        docs = [doc.strip() for doc in docs if doc.strip()]
        
        return docs
    
    def _generate_embeddings(self, docs: List[str]) -> List[List[float]]:
        """
        生成文档嵌入
        
        Args:
            docs: 文档列表
            
        Returns:
            List[List[float]]: 嵌入列表
        """
        if self.use_bertopic:
            return self.embedding_model.encode(docs)
        else:
            # 简单的词袋嵌入
            all_words = set()
            for doc in docs:
                words = doc.split()
                all_words.update(words)
            
            embeddings = []
            for doc in docs:
                words = doc.split()
                vector = []
                for word in all_words:
                    vector.append(words.count(word))
                embeddings.append(vector)
            
            return embeddings
    
    def _extract_with_bertopic(self, docs: List[str], num_topics: int) -> List[Dict]:
        """
        使用BERTopic提取主题
        
        Args:
            docs: 文档列表
            num_topics: 要提取的主题数量
            
        Returns:
            List[Dict]: 主题列表
        """
        # 拟合模型
        topics, probabilities = self.topic_model.fit_transform(docs)
        
        # 获取主题信息
        topic_info = self.topic_model.get_topic_info()
        
        # 过滤掉-1主题（未分类）
        topic_info = topic_info[topic_info['Topic'] != -1]
        
        # 限制主题数量
        topic_info = topic_info.head(num_topics)
        
        # 构建主题结果
        result = []
        for _, row in topic_info.iterrows():
            topic = {
                "topic": row['Name'],
                "keywords": row['Representation'],
                "weight": row['Count'] / len(docs)  # 主题权重
            }
            result.append(topic)
        
        return result
    
    def _simple_topic_extraction(self, text: str, num_topics: int) -> List[Dict]:
        """
        简单的主题提取
        
        Args:
            text: 输入文本
            num_topics: 要提取的主题数量
            
        Returns:
            List[Dict]: 主题列表
        """
        # 移除标点符号和特殊字符，保留中文、英文、数字和空白
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
        
        # 转换为小写
        text = text.lower()
        
        # 分词
        words = text.split()
        
        # 过滤停用词（简单实现）
        stop_words = {
            '的', '了', '和', '是', '在', '有', '为', '与', '他', '她', '它',
            '我们', '你们', '他们', '这', '那', '这些', '那些', '是', '不是',
            '我', '你', '您', '他', '她', '它', '我们', '你们', '他们', '自己',
            '这个', '那个', '这里', '那里', '现在', '过去', '未来', '可以', '可能'
        }
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 1]
        
        # 计算词频
        word_counts = Counter(filtered_words)
        
        # 获取最常见的关键词
        most_common = word_counts.most_common(num_topics * 5)  # 获取更多关键词以形成主题
        
        # 简单的主题聚类（基于共现）
        topics = []
        used_words = set()
        
        for i in range(num_topics):
            if not most_common:
                break
            
            # 选择最常见的词作为主题中心
            center_word, count = most_common.pop(0)
            used_words.add(center_word)
            
            # 找到与中心词共现的词
            keywords = [center_word]
            weight = count
            
            # 简单的共现检测
            for word, word_count in most_common:
                if word not in used_words and self._check_cooccurrence(text, center_word, word):
                    keywords.append(word)
                    used_words.add(word)
                    weight += word_count
                    
                    if len(keywords) >= 3:  # 每个主题至少3个关键词
                        break
            
            # 计算主题权重
            total_words = sum(word_counts.values())
            topic_weight = weight / total_words
            
            topic = {
                "topic": center_word,
                "keywords": keywords,
                "weight": topic_weight
            }
            topics.append(topic)
        
        return topics
    
    def _check_cooccurrence(self, text: str, word1: str, word2: str) -> bool:
        """
        检查两个词是否在文本中共现
        
        Args:
            text: 输入文本
            word1: 第一个词
            word2: 第二个词
            
        Returns:
            bool: 是否共现
        """
        # 简单的共现检测：两个词在同一行出现
        lines = text.split('\n')
        for line in lines:
            if word1 in line and word2 in line:
                return True
        return False
    
    def extract_key_phrases(self, text: str, num_phrases: int = 10) -> List[Dict]:
        """
        提取关键短语
        
        Args:
            text: 输入文本
            num_phrases: 要提取的关键短语数量
            
        Returns:
            List[Dict]: 关键短语列表，包含phrase和weight
        """
        if not text:
            return []
        
        try:
            # 简单的关键短语提取
            # 移除标点符号和特殊字符，保留中文、英文、数字和空白
            text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
            
            # 转换为小写
            text = text.lower()
            
            # 分词
            words = text.split()
            
            # 过滤停用词
            stop_words = {
                '的', '了', '和', '是', '在', '有', '为', '与', '他', '她', '它',
                '我们', '你们', '他们', '这', '那', '这些', '那些', '是', '不是',
                '我', '你', '您', '他', '她', '它', '我们', '你们', '他们', '自己',
                '这个', '那个', '这里', '那里', '现在', '过去', '未来', '可以', '可能'
            }
            
            filtered_words = [word for word in words if word not in stop_words and len(word) > 1]
            
            # 计算n-gram频率
            n_grams = []
            for n in range(1, 4):  # 1-gram, 2-gram, 3-gram
                n_grams.extend(self._get_ngrams(filtered_words, n))
            
            # 计算频率
            n_gram_counts = Counter(n_grams)
            
            # 获取最常见的n-gram
            most_common = n_gram_counts.most_common(num_phrases)
            
            # 构建结果
            result = []
            total_count = sum(n_gram_counts.values())
            
            for phrase, count in most_common:
                result.append({
                    "phrase": " ".join(phrase),
                    "weight": count / total_count
                })
            
            return result
        except Exception as e:
            self.logger.error(f"Key phrase extraction failed: {str(e)}")
            import traceback
            self.logger.debug(f"Exception details: {traceback.format_exc()}")
            return []
    
    def _get_ngrams(self, words: List[str], n: int) -> List[tuple]:
        """
        获取n-gram
        
        Args:
            words: 词列表
            n: n-gram的大小
            
        Returns:
            List[tuple]: n-gram列表
        """
        return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
