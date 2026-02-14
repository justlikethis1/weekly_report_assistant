#!/usr/bin/env python3
"""
事件抽取服务
从新闻文本中提取结构化事件信息
"""

from typing import List, Dict, Optional
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class EventExtractor:
    """
    事件抽取器
    从新闻文本中提取事件四要素：时间、地点、实体、动作
    """
    
    def __init__(self):
        """初始化事件抽取器"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("EventExtractor initialized")
        
        # 时间提取的正则表达式
        self.time_patterns = [
            r"(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?)",  # YYYY-MM-DD, YYYY/MM/DD, YYYY年MM月DD日
            r"(\d{1,2}[-/月]\d{1,2}日?)",  # MM-DD, MM/DD, MM月DD日
            r"(\d{4}年\d{1,2}月)",  # YYYY年MM月
            r"(\d{1,2}月\d{1,2}日)",  # MM月DD日
            r"(昨天|今天|明天|前天|后天)",  # 相对时间
            r"(上[个周月季年]|本[个周月季年]|下[个周月季年])"  # 相对时间范围
        ]
        
        # 地点提取的正则表达式
        self.location_pattern = r"([^\s，。！？；：]+[市县区省国家]|北京|上海|广州|深圳|杭州|成都|武汉|西安|重庆|天津)"
        
        # 实体提取的正则表达式（简单实现）
        self.entity_patterns = {
            "organization": r"([^\s，。！？；：]+[公司|集团|银行|机构|部门|协会|组织|大学|学院])",
            "person": r"([^\s，。！？；：]+[先生|女士|教授|博士|经理|主任|董事长|CEO|总裁|行长])",
            "product": r"([^\s，。！？；：]+[产品|系统|软件|服务|设备|项目|计划])"
        }
        
        # 动作提取的正则表达式
        self.action_pattern = r"(上涨|下跌|增加|减少|提升|降低|发布|推出|收购|合并|合作|竞争|影响|导致|促进|阻碍|引发)"
    
    def extract_events(self, news_texts: List[str]) -> List[Dict]:
        """
        从新闻文本中抽取事件四要素
        
        Args:
            news_texts: 新闻文本列表
            
        Returns:
            List[Dict]: 事件列表，每个事件包含时间、地点、实体、动作和情感
        """
        events = []
        for text in news_texts:
            event = {
                "time": self._extract_time(text),
                "location": self._extract_location(text),
                "entities": self._extract_entities(text),
                "action": self._extract_action(text),
                "sentiment": self._analyze_sentiment(text),
                "description": text[:100] + "..." if len(text) > 100 else text
            }
            events.append(event)
        return events
    
    def _extract_time(self, text: str) -> Optional[str]:
        """
        从文本中提取时间
        
        Args:
            text: 输入文本
            
        Returns:
            Optional[str]: 提取的时间字符串
        """
        for pattern in self.time_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None
    
    def _extract_location(self, text: str) -> Optional[str]:
        """
        从文本中提取地点
        
        Args:
            text: 输入文本
            
        Returns:
            Optional[str]: 提取的地点字符串
        """
        match = re.search(self.location_pattern, text)
        if match:
            return match.group(1)
        return None
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        从文本中提取实体
        
        Args:
            text: 输入文本
            
        Returns:
            Dict[str, List[str]]: 实体字典，包含组织、人物和产品
        """
        entities = {
            "organizations": [],
            "persons": [],
            "products": []
        }
        
        # 提取组织
        for match in re.finditer(self.entity_patterns["organization"], text):
            entities["organizations"].append(match.group(1))
        
        # 提取人物
        for match in re.finditer(self.entity_patterns["person"], text):
            entities["persons"].append(match.group(1))
        
        # 提取产品
        for match in re.finditer(self.entity_patterns["product"], text):
            entities["products"].append(match.group(1))
        
        # 去重
        entities["organizations"] = list(set(entities["organizations"]))
        entities["persons"] = list(set(entities["persons"]))
        entities["products"] = list(set(entities["products"]))
        
        return entities
    
    def _extract_action(self, text: str) -> Optional[str]:
        """
        从文本中提取动作
        
        Args:
            text: 输入文本
            
        Returns:
            Optional[str]: 提取的动作字符串
        """
        match = re.search(self.action_pattern, text)
        if match:
            return match.group(1)
        return None
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """
        分析文本情感
        
        Args:
            text: 输入文本
            
        Returns:
            Dict: 情感分析结果，包含得分和标签
        """
        # 简单的情感分析实现
        # 正面词汇
        positive_words = ["上涨", "增加", "提升", "发布", "推出", "合作", "促进", "盈利", "增长", "创新"]
        # 负面词汇
        negative_words = ["下跌", "减少", "降低", "损失", "风险", "挑战", "阻碍", "亏损", "下降", "竞争"]
        
        positive_score = 0
        negative_score = 0
        
        # 计算正面和负面词汇的数量
        for word in positive_words:
            positive_score += text.count(word)
        
        for word in negative_words:
            negative_score += text.count(word)
        
        # 计算最终情感得分
        score = (positive_score - negative_score) / max(positive_score + negative_score + 1, 1)
        
        # 确定情感标签
        if score > 0.3:
            label = "positive"
        elif score < -0.3:
            label = "negative"
        else:
            label = "neutral"
        
        return {
            "score": score,
            "label": label
        }
