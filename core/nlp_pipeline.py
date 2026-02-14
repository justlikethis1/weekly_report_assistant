#!/usr/bin/env python3
"""
NLP处理流水线
在数据处理层前添加NLP预处理层
"""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class NLPPipeline:
    """
    NLP处理流水线
    整合各种NLP处理组件，形成完整的NLP预处理层
    
    流水线流程：
    文件处理层 → NLP预处理层 → 数据处理层
                 ↓
           文本清洗、分词、实体识别、
           事件抽取、情感分析、摘要生成
    """
    
    def __init__(self):
        """
        初始化NLP处理流水线
        """
        try:
            # 导入NLP服务
            from src.nlp.services.nlp_service import NLPService
            
            # 初始化NLP服务 - 使用mock模式避免网络请求
            self.nlp_service = NLPService(is_mock=True)
            
            logger.info("NLPPipeline initialized successfully with mock NLP service")
            
        except Exception as e:
            logger.error(f"Failed to initialize NLPPipeline: {str(e)}")
            raise
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """
        处理单篇文本
        
        Args:
            text: 输入文本
            
        Returns:
            Dict[str, Any]: 处理结果，包含各种NLP分析结果
        """
        if not text:
            return {}
        
        try:
            # 初始化结果字典
            result = {
                "original_text": text,
                "processed_text": text,  # 初始化为原始文本
                "nlp_results": {}
            }
            
            # 1. 文本清洗（简单实现，更复杂的清洗可以在各个服务中进行）
            result["processed_text"] = self._clean_text(text)
            
            # 2. 意图分析
            intent_analysis = self.nlp_service.analyze_query(text)
            result["nlp_results"]["intent"] = intent_analysis.to_dict()
            
            # 3. 文档处理
            document_analysis = self.nlp_service.process_document(text)
            result["nlp_results"]["document"] = document_analysis.to_dict()
            
            # 4. 实体识别（从意图分析和文档处理中提取）
            entities = []
            if "entities" in result["nlp_results"]["intent"]:
                entities.extend(result["nlp_results"]["intent"]["entities"])
            result["nlp_results"]["entities"] = entities
            
            # 5. 情感分析
            sentiment = {"score": 0.0, "label": "neutral"}
            if "sentiment" in result["nlp_results"]["document"]:
                sentiment = result["nlp_results"]["document"]["sentiment"]
            elif "sentiment" in result["nlp_results"]["intent"]:
                sentiment = result["nlp_results"]["intent"]["sentiment"]
            result["nlp_results"]["sentiment"] = sentiment
            
            # 6. 关键短语提取
            key_phrases = self.nlp_service.extract_key_phrases(text)
            result["nlp_results"]["key_phrases"] = key_phrases
            
            # 7. 主题提取
            topics = self.nlp_service.extract_topics(text)
            result["nlp_results"]["topics"] = topics
            
            return result
        except Exception as e:
            logger.error(f"Text processing failed: {str(e)}")
            import traceback
            logger.debug(f"Exception details: {traceback.format_exc()}")
            return {"original_text": text, "error": str(e)}
    
    def process_news_texts(self, news_texts: List[str]) -> List[Dict[str, Any]]:
        """
        处理多篇新闻文本
        
        Args:
            news_texts: 新闻文本列表
            
        Returns:
            List[Dict[str, Any]]: 处理结果列表
        """
        if not news_texts:
            return []
        
        try:
            results = []
            
            # 处理每篇新闻文本
            for text in news_texts:
                result = self.process_text(text)
                results.append(result)
            
            # 8. 事件抽取
            events = self.nlp_service.extract_events(news_texts)
            
            # 将事件与新闻文本关联
            for i, event in enumerate(events):
                if i < len(results):
                    results[i]["nlp_results"]["event"] = event
            
            return results
        except Exception as e:
            logger.error(f"News texts processing failed: {str(e)}")
            import traceback
            logger.debug(f"Exception details: {traceback.format_exc()}")
            return [{"original_text": text, "error": str(e)} for text in news_texts]
    
    def match_events_with_price_changes(self, events: List[Dict], price_changes: List[Dict]) -> List[Dict]:
        """
        将事件与价格变化进行匹配
        
        Args:
            events: 事件列表
            price_changes: 价格变化列表
            
        Returns:
            List[Dict]: 匹配结果列表
        """
        if not events or not price_changes:
            return []
        
        try:
            # 使用事件-价格匹配服务
            matches = self.nlp_service.match_events_to_prices(events, price_changes)
            return matches
        except Exception as e:
            logger.error(f"Event-price matching failed: {str(e)}")
            import traceback
            logger.debug(f"Exception details: {traceback.format_exc()}")
            return []
    
    def enhance_prompt(self, query: str, context: Optional[Dict] = None) -> str:
        """
        增强提示词
        
        Args:
            query: 用户查询
            context: 查询上下文（可选）
            
        Returns:
            str: 增强后的提示词
        """
        if not query:
            return ""
        
        try:
            # 分析用户查询
            intent_analysis = self.nlp_service.analyze_query(query, context)
            
            # 增强提示词
            enhanced_prompt = self.nlp_service.enhance_prompt(intent_analysis)
            
            return enhanced_prompt
        except Exception as e:
            logger.error(f"Prompt enhancement failed: {str(e)}")
            import traceback
            logger.debug(f"Exception details: {traceback.format_exc()}")
            return query  # 如果增强失败，返回原始查询
    
    def _clean_text(self, text: str) -> str:
        """
        文本清洗
        
        Args:
            text: 输入文本
            
        Returns:
            str: 清洗后的文本
        """
        import re
        
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊字符（保留中文、英文、数字和常见标点）
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：,.!?;:\s]', '', text)
        
        # 移除首尾空白
        text = text.strip()
        
        return text
    
    def get_knowledge_support(self, query: str, domains: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        获取知识支持
        
        Args:
            query: 查询内容
            domains: 限定领域（可选）
            
        Returns:
            Dict[str, Any]: 知识支持结果
        """
        if not query:
            return {}
        
        try:
            # 查询知识库
            knowledge_results = self.nlp_service.query_knowledge(query, domains)
            
            # 提取关键短语
            key_phrases = self.nlp_service.extract_key_phrases(query)
            
            # 提取主题
            topics = self.nlp_service.extract_topics(query)
            
            return {
                "knowledge_results": knowledge_results,
                "key_phrases": key_phrases,
                "topics": topics
            }
        except Exception as e:
            logger.error(f"Knowledge support failed: {str(e)}")
            import traceback
            logger.debug(f"Exception details: {traceback.format_exc()}")
            return {"error": str(e)}
