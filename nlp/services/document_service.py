#!/usr/bin/env python3
"""
文档分析服务
"""

from typing import Dict, Any, List
import logging
from ..analyzers.document_analyzer import DocumentAnalyzer

logger = logging.getLogger(__name__)

class DocumentService:
    """文档分析服务，封装文档处理功能"""
    
    def __init__(self):
        """初始化文档服务"""
        try:
            self.document_analyzer = DocumentAnalyzer()
            logger.info("DocumentService initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize DocumentService: {str(e)}")
            raise
    
    def analyze_content(self, content: str) -> Dict[str, Any]:
        """
        分析文档内容
        
        Args:
            content: 文档内容
            
        Returns:
            Dict: 文档分析结果
        """
        try:
            return self.document_analyzer.analyze(content)
        except Exception as e:
            logger.error(f"Document analysis failed: {str(e)}")
            raise
    
    def extract_key_points(self, analysis_result: Dict[str, Any]) -> List[str]:
        """
        从分析结果中提取关键点
        
        Args:
            analysis_result: 文档分析结果
            
        Returns:
            List[str]: 关键点列表
        """
        return analysis_result.get("key_points", [])
    
    def extract_data_points(self, analysis_result: Dict[str, Any]) -> List[str]:
        """
        从分析结果中提取数据点
        
        Args:
            analysis_result: 文档分析结果
            
        Returns:
            List[str]: 数据点列表
        """
        return analysis_result.get("data_points", [])
    
    def analyze_sentiment(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        从分析结果中提取情感分析
        
        Args:
            analysis_result: 文档分析结果
            
        Returns:
            Dict: 情感分析结果
        """
        return analysis_result.get("sentiment", {"score": 0.0, "overall": "neutral"})
