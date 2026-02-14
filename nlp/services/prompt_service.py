#!/usr/bin/env python3
"""
提示词服务
"""

from typing import Dict, Any, List
import logging
from ..analyzers.prompt_enhancer import PromptEnhancer

logger = logging.getLogger(__name__)

class PromptService:
    """提示词服务，封装提示词增强功能"""
    
    def __init__(self):
        """初始化提示词服务"""
        try:
            self.prompt_enhancer = PromptEnhancer()
            logger.info("PromptService initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PromptService: {str(e)}")
            raise
    
    def enhance_prompt(self, intent: Dict[str, Any], document_analysis: Dict[str, Any]) -> str:
        """
        增强提示词
        
        Args:
            intent: 意图分析结果
            document_analysis: 文档分析结果
            
        Returns:
            str: 增强后的提示词
        """
        try:
            return self.prompt_enhancer.enhance(intent, document_analysis)
        except Exception as e:
            logger.error(f"Prompt enhancement failed: {str(e)}")
            raise
    
    def format_prompt(self, prompt_template: str, variables: Dict[str, Any]) -> str:
        """
        格式化提示词模板
        
        Args:
            prompt_template: 提示词模板
            variables: 变量字典
            
        Returns:
            str: 格式化后的提示词
        """
        try:
            formatted_prompt = prompt_template
            for key, value in variables.items():
                formatted_prompt = formatted_prompt.replace(f"{{{{{key}}}}}", str(value))
            return formatted_prompt
        except Exception as e:
            logger.error(f"Prompt formatting failed: {str(e)}")
            raise
