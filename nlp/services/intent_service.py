#!/usr/bin/env python3
"""
意图解析服务
"""

from typing import Dict, Any, List
import logging
from ..analyzers.intent_parser import IntentParser

logger = logging.getLogger(__name__)

class IntentService:
    """意图解析服务，封装意图分析功能"""
    
    def __init__(self):
        """初始化意图服务"""
        try:
            self.intent_parser = IntentParser()
            logger.info("IntentService initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize IntentService: {str(e)}")
            raise
    
    def parse_intent(self, query: str) -> Dict[str, Any]:
        """
        解析用户查询的意图
        
        Args:
            query: 用户查询字符串
            
        Returns:
            Dict: 意图解析结果
        """
        try:
            return self.intent_parser.parse(query)
        except Exception as e:
            logger.error(f"Intent parsing failed: {str(e)}")
            raise
    
    def get_domain(self, intent_result: Dict[str, Any]) -> str:
        """
        从意图解析结果中提取主要领域
        
        Args:
            intent_result: 意图解析结果
            
        Returns:
            str: 主要领域
        """
        return intent_result.get("domain", {}).get("primary", "general")
    
    def get_time_range(self, intent_result: Dict[str, Any]) -> Dict[str, str]:
        """
        从意图解析结果中提取时间范围
        
        Args:
            intent_result: 意图解析结果
            
        Returns:
            Dict: 时间范围信息
        """
        return intent_result.get("time", {})
    
    def get_metrics(self, intent_result: Dict[str, Any]) -> List[str]:
        """
        从意图解析结果中提取量化指标
        
        Args:
            intent_result: 意图解析结果
            
        Returns:
            List[str]: 量化指标列表
        """
        return intent_result.get("metrics", {})
