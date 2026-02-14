#!/usr/bin/env python3
"""
智能文档分析器：文档深度理解
"""

import re
from typing import Dict, List, Tuple, Set
import logging
from collections import Counter, defaultdict

# 配置日志
logger = logging.getLogger(__name__)

class DocumentAnalyzer:
    """文档深度理解：关键点提取、数据点提取、论点识别、情感分析"""
    
    def __init__(self):
        # 初始化分析规则
        self._init_analysis_rules()
    
    def _init_analysis_rules(self):
        """初始化分析规则和模式"""
        # 数据提取模式
        self.data_patterns = {
            "number": r'\d+\.?\d*',
            "percentage": r'\d+\.?\d*%',
            "currency": r'[¥$€£]\s?\d+\.?\d*|\d+\.?\d*\s?[¥$€£]',
            "date": r'\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}',
            "time": r'\d{1,2}:\d{2}(:\d{2})?',
            "range": r'\d+\s?[-~]\s?\d+'
        }
        
        # 论点识别关键词
        self.argument_keywords = {
            "cause": ["因为", "由于", "原因是", "导致", "造成"],
            "effect": ["所以", "因此", "结果是", "导致", "从而"],
            "conclusion": ["结论", "总结", "因此", "所以", "综上所述", "由此可见"],
            "opinion": ["认为", "觉得", "观点", "看法", "建议", "应该"],
            "evidence": ["数据显示", "研究表明", "调查发现", "根据", "依据"],
            "comparison": ["相比", "比较", "不如", "更", "最"]
        }
        
        # 情感词库
        self.sentiment_words = {
            "positive": ["增长", "提升", "提高", "改善", "优化", "成功", "创新", "优势", "突破", "领先"],
            "negative": ["下降", "降低", "减少", "恶化", "问题", "挑战", "风险", "劣势", "不足", "亏损"],
            "neutral": ["保持", "稳定", "维持", "不变", "一般", "正常", "普通"]
        }
        
        # 关键点提取配置
        self.key_point_config = {
            "sentence_patterns": [
                r'[因此所以综上]，?.*[。！]',  # 结论句
                r'[研究表明数据显示]，?.*[。！]',  # 证据句
                r'.*[关键重要核心主要]\s*[是为].*[。！]',  # 关键句
                r'.*[发现认为建议].*[。！]'  # 观点建议句
            ],
            "max_sentences": 10  # 最多提取10个关键点
        }
    
    def analyze(self, content: str) -> Dict:
        """
        分析文档内容
        
        Args:
            content: 文档内容字符串
            
        Returns:
            Dict: 包含关键点、数据点、论点和情感的分析结果
        """
        try:
            # 1. 提取关键点
            key_points = self._extract_key_points(content)
            
            # 2. 提取数据点
            data_points = self._extract_data(content)
            
            # 3. 识别论点结构
            arguments = self._identify_arguments(content)
            
            # 4. 分析情感倾向
            sentiment = self._analyze_sentiment(content)
            
            return {
                "key_points": key_points,
                "data_points": data_points,
                "arguments": arguments,
                "sentiment": sentiment,
                "word_count": len(content.split()),
                "sentence_count": self._count_sentences(content)
            }
            
        except Exception as e:
            logger.error(f"文档分析失败: {str(e)}")
            return {
                "key_points": [],
                "data_points": [],
                "arguments": [],
                "sentiment": {"overall": "neutral", "score": 0.0},
                "word_count": len(content.split()),
                "sentence_count": self._count_sentences(content)
            }
    
    def _extract_key_points(self, content: str) -> List[str]:
        """
        提取文档关键点
        
        Args:
            content: 文档内容
            
        Returns:
            List[str]: 关键点列表
        """
        key_points = []
        
        # 分割句子
        sentences = self._split_sentences(content)
        
        # 基于模式匹配提取关键句
        pattern_matched = set()
        for pattern in self.key_point_config["sentence_patterns"]:
            for i, sentence in enumerate(sentences):
                if i in pattern_matched:
                    continue
                if re.search(pattern, sentence):
                    key_points.append(sentence.strip())
                    pattern_matched.add(i)
                    if len(key_points) >= self.key_point_config["max_sentences"]:
                        return key_points
        
        # 如果通过模式匹配提取的关键点不足，使用统计方法
        if len(key_points) < 5:
            # 计算句子重要性分数
            sentence_scores = self._calculate_sentence_importance(sentences, content)
            
            # 按分数排序并添加到关键点
            sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
            
            for idx, score in sorted_sentences:
                if idx not in pattern_matched:
                    key_points.append(sentences[idx].strip())
                    if len(key_points) >= self.key_point_config["max_sentences"]:
                        break
        
        return key_points
    
    def _extract_data(self, content: str) -> List[Dict]:
        """
        提取文档中的数据点
        
        Args:
            content: 文档内容
            
        Returns:
            List[Dict]: 数据点列表
        """
        data_points = []
        
        # 提取各种类型的数据
        for data_type, pattern in self.data_patterns.items():
            matches = re.finditer(pattern, content)
            for match in matches:
                value = match.group().strip()
                # 获取上下文
                context_start = max(0, match.start() - 50)
                context_end = min(len(content), match.end() + 50)
                context = content[context_start:context_end].strip()
                
                data_point = {
                    "type": data_type,
                    "value": value,
                    "context": context,
                    "position": match.start()
                }
                data_points.append(data_point)
        
        # 去重（基于值和位置）
        unique_data = []
        seen = set()
        for data in data_points:
            key = (data["value"], data["position"])
            if key not in seen:
                seen.add(key)
                unique_data.append(data)
        
        return unique_data
    
    def _identify_arguments(self, content: str) -> Dict:
        """
        识别文档中的论点结构
        
        Args:
            content: 文档内容
            
        Returns:
            Dict: 论点结构
        """
        arguments = {
            "cause_effect_pairs": [],
            "conclusions": [],
            "opinions": [],
            "evidence": [],
            "comparisons": []
        }
        
        # 分割句子
        sentences = self._split_sentences(content)
        
        # 识别论点元素
        for i, sentence in enumerate(sentences):
            # 识别因果关系
            for cause_word in self.argument_keywords["cause"]:
                if cause_word in sentence:
                    # 查找后续的结果句
                    for j in range(i+1, min(i+3, len(sentences))):
                        for effect_word in self.argument_keywords["effect"]:
                            if effect_word in sentences[j]:
                                arguments["cause_effect_pairs"].append({
                                    "cause": sentence,
                                    "effect": sentences[j],
                                    "confidence": 0.7
                                })
                                break
                    break
            
            # 识别结论
            for conclusion_word in self.argument_keywords["conclusion"]:
                if conclusion_word in sentence:
                    arguments["conclusions"].append({
                        "sentence": sentence,
                        "position": i,
                        "confidence": 0.8
                    })
                    break
            
            # 识别观点
            for opinion_word in self.argument_keywords["opinion"]:
                if opinion_word in sentence:
                    arguments["opinions"].append({
                        "sentence": sentence,
                        "position": i,
                        "confidence": 0.7
                    })
                    break
            
            # 识别证据
            for evidence_word in self.argument_keywords["evidence"]:
                if evidence_word in sentence:
                    arguments["evidence"].append({
                        "sentence": sentence,
                        "position": i,
                        "confidence": 0.8
                    })
                    break
        
        return arguments
    
    def _analyze_sentiment(self, content: str) -> Dict:
        """
        分析文档的情感倾向
        
        Args:
            content: 文档内容
            
        Returns:
            Dict: 情感分析结果
        """
        # 计算情感分数
        sentiment_score = 0
        word_count = 0
        
        for sentiment_type, words in self.sentiment_words.items():
            for word in words:
                count = content.count(word)
                if count > 0:
                    word_count += count
                    if sentiment_type == "positive":
                        sentiment_score += count
                    elif sentiment_type == "negative":
                        sentiment_score -= count
                    # 中性词不影响分数
        
        # 计算总体情感倾向
        if word_count == 0:
            overall_sentiment = "neutral"
        else:
            normalized_score = sentiment_score / word_count
            if normalized_score > 0.1:
                overall_sentiment = "positive"
            elif normalized_score < -0.1:
                overall_sentiment = "negative"
            else:
                overall_sentiment = "neutral"
        
        return {
            "overall": overall_sentiment,
            "score": round(sentiment_score / max(word_count, 1), 2),
            "word_count": word_count
        }
    
    def _split_sentences(self, content: str) -> List[str]:
        """
        分割句子
        
        Args:
            content: 文档内容
            
        Returns:
            List[str]: 句子列表
        """
        # 使用正则表达式分割句子
        sentences = re.split(r'[。！？]\s*', content)
        # 过滤空句子
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _count_sentences(self, content: str) -> int:
        """
        计算句子数量
        
        Args:
            content: 文档内容
            
        Returns:
            int: 句子数量
        """
        return len(self._split_sentences(content))
    
    def _calculate_sentence_importance(self, sentences: List[str], content: str) -> Dict[int, float]:
        """
        计算句子重要性分数
        
        Args:
            sentences: 句子列表
            content: 文档内容
            
        Returns:
            Dict[int, float]: 句子索引到分数的映射
        """
        scores = defaultdict(float)
        
        # 基于词频计算重要性
        word_counter = Counter(re.findall(r'\b\w+\b', content))
        total_words = sum(word_counter.values())
        
        # 计算词的重要性
        word_importance = {word: count / total_words for word, count in word_counter.items()}
        
        # 计算每个句子的重要性
        for i, sentence in enumerate(sentences):
            sentence_words = re.findall(r'\b\w+\b', sentence)
            # 句子长度权重
            length_weight = min(len(sentence_words) / 20, 1.0)
            # 词重要性权重
            word_weight = sum(word_importance.get(word, 0) for word in sentence_words) / max(len(sentence_words), 1)
            # 位置权重（开头和结尾的句子更重要）
            position_weight = 1.0
            if i == 0 or i == len(sentences) - 1:
                position_weight = 1.5
            elif i < 3 or i > len(sentences) - 4:
                position_weight = 1.2
            
            # 综合分数
            scores[i] = (length_weight * 0.3 + word_weight * 0.5 + position_weight * 0.2)
        
        return scores
    
    def analyze_files(self, files: List[str]) -> Dict:
        """
        分析多个文件
        
        Args:
            files: 文件路径列表
            
        Returns:
            Dict: 综合分析结果
        """
        # 这里简化实现，实际应该读取文件内容并分析
        return {
            "total_files": len(files),
            "analysis_results": []
        }

# 测试代码
if __name__ == "__main__":
    # 创建文档分析器实例
    analyzer = DocumentAnalyzer()
    
    # 测试文本
    test_text = """ 
    本周黄金价格表现强劲，周涨幅达到5.2%，创下三个月以来的新高。这主要是由于美元指数下跌2.1%，以及地缘政治风险加剧导致的避险情绪升温。
    数据显示，全球黄金ETF持仓量增加了12.5吨，反映了投资者对黄金的信心提升。相比之下，股票市场表现疲软，主要股指平均下跌1.8%。
    分析认为，黄金价格的上涨趋势可能会持续，建议投资者适当增加黄金资产配置以分散风险。
    然而，需要注意的是，美联储可能的加息政策仍然是黄金价格的潜在风险因素。
    """
    
    print("=== 文档分析测试 ===")
    print(f"测试文本:\n{test_text}\n")
    
    # 进行分析
    result = analyzer.analyze(test_text)
    
    print("1. 关键点:")
    for i, point in enumerate(result["key_points"]):
        print(f"   {i+1}. {point}")
    
    print("\n2. 数据点:")
    for i, data in enumerate(result["data_points"]):
        print(f"   {i+1}. [{data['type']}] {data['value']} - {data['context'][:50]}...")
    
    print("\n3. 论点结构:")
    print(f"   因果关系对数: {len(result['arguments']['cause_effect_pairs'])}")
    print(f"   结论数: {len(result['arguments']['conclusions'])}")
    print(f"   观点数: {len(result['arguments']['opinions'])}")
    print(f"   证据数: {len(result['arguments']['evidence'])}")
    
    print("\n4. 情感分析:")
    print(f"   总体情感: {result['sentiment']['overall']}")
    print(f"   情感分数: {result['sentiment']['score']}")
    print(f"   情感词数: {result['sentiment']['word_count']}")
    
    print("\n5. 文档统计:")
    print(f"   单词数: {result['word_count']}")
    print(f"   句子数: {result['sentence_count']}")
