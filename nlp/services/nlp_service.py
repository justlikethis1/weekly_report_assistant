#!/usr/bin/env python3
"""
统一NLP服务接口
"""

from typing import Dict, Any, List, Optional
import logging

# 导入拆分后的服务
from .intent_service import IntentService
from .document_service import DocumentService
from .prompt_service import PromptService
from .event_extractor import EventExtractor
from .event_price_matcher import EventPriceMatcher
from .topic_extractor import TopicExtractor
from .shieldlm_service import ShieldLMService
from ..config import nlp_config

# 导入对话管理器
from src.memory.dialogue_manager import DialogueManager

# 导入原知识库功能（暂时保留，后续可重构）
from ..analyzers.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

class QueryAnalysis:
    """查询分析结果"""
    def __init__(self, intent: Dict, entities: List, sentiment: Dict, 
                 user_identity: Optional[Dict] = None, context_info: Optional[Dict] = None):
        self.intent = intent
        self.entities = entities
        self.sentiment = sentiment
        self.user_identity = user_identity
        self.context_info = context_info
        
    def to_dict(self) -> Dict:
        return {
            "intent": self.intent,
            "entities": self.entities,
            "sentiment": self.sentiment,
            "user_identity": self.user_identity,
            "context_info": self.context_info
        }

class DocumentAnalysis:
    """文档分析结果"""
    def __init__(self, key_points: List, data_points: List, arguments: Dict, sentiment: Dict):
        self.key_points = key_points
        self.data_points = data_points
        self.arguments = arguments
        self.sentiment = sentiment
        
    def to_dict(self) -> Dict:
        return {
            "key_points": self.key_points,
            "data_points": self.data_points,
            "arguments": self.arguments,
            "sentiment": self.sentiment
        }

class NLPService:
    """统一的NLP服务入口"""
    
    def __init__(self, is_mock: bool = False, allow_auto_fallback: bool = True):
        """
        初始化NLP服务
        
        Args:
            is_mock: 是否使用mock模式
            allow_auto_fallback: 当模型加载失败时是否允许自动降级到mock模式
        """
        try:
            # 初始化拆分后的服务
            self.intent_service = IntentService()
            self.document_service = DocumentService()
            self.prompt_service = PromptService()
            self.event_extractor = EventExtractor()
            self.event_price_matcher = EventPriceMatcher(is_mock=is_mock)
            self.topic_extractor = TopicExtractor(is_mock=is_mock)
            self.knowledge_base = KnowledgeBase()
            
            # 初始化ShieldLM模型服务，传递is_mock参数和设备配置
            self.shieldlm_service = ShieldLMService(
                is_mock=is_mock, 
                allow_auto_fallback=allow_auto_fallback,
                device=nlp_config.device
            )
            
            # 初始化对话管理器
            self.dialogue_manager = DialogueManager()
            
            logger.info("NLPService initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize NLPService: {str(e)}")
            raise
    
    def analyze_query(self, query: str, context: Optional[Dict] = None, 
                     system_response: Optional[str] = None) -> QueryAnalysis:
        """
        分析用户查询
        
        Args:
            query: 用户查询字符串
            context: 查询上下文（可选）
            system_response: 上一轮系统响应（用于多轮对话）
            
        Returns:
            QueryAnalysis: 查询分析结果
        """
        try:
            # 使用意图服务解析查询
            intent_result = self.intent_service.parse_intent(query)
            
            # 提取实体信息
            entities = []
            if intent_result.get("surface") and intent_result["surface"].get("keywords"):
                entities = intent_result["surface"]["keywords"]
            
            # 提取情感信息（简化处理）
            sentiment = {
                "score": 0.0,  # 中性
                "label": "neutral"
            }
            
            # 识别用户身份
            user_identity = self.dialogue_manager.identify_user_identity(query)
            
            # 构建上下文信息
            context_info = {
                "session_id": self.dialogue_manager.get_session_id(),
                "history_length": len(self.dialogue_manager.dialogue_history),
                "last_intent": None
            }
            
            # 更新对话历史
            if system_response:
                self.dialogue_manager.add_turn(query, system_response, intent_result, entities)
            else:
                self.dialogue_manager.add_turn(query, intent=intent_result, entities=entities)
            
            return QueryAnalysis(
                intent=intent_result,
                entities=entities,
                sentiment=sentiment,
                user_identity=user_identity,
                context_info=context_info
            )
            
        except Exception as e:
            logger.error(f"Query analysis failed: {str(e)}")
            raise
    
    def process_document(self, content: str, document_type: str = "text") -> DocumentAnalysis:
        """
        处理文档内容
        
        Args:
            content: 文档内容
            document_type: 文档类型（text, pdf, docx等）
            
        Returns:
            DocumentAnalysis: 文档分析结果
        """
        try:
            # 使用文档服务分析内容
            analysis_result = self.document_service.analyze_content(content)
            
            return DocumentAnalysis(
                key_points=analysis_result.get("key_points", []),
                data_points=analysis_result.get("data_points", []),
                arguments=analysis_result.get("arguments", {}),
                sentiment=analysis_result.get("sentiment", {})
            )
            
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            raise
    
    def enhance_prompt(self, intent_analysis: QueryAnalysis, document_analysis: Optional[DocumentAnalysis] = None) -> str:
        """
        增强提示词
        
        Args:
            intent_analysis: 意图分析结果
            document_analysis: 文档分析结果（可选）
            
        Returns:
            str: 增强后的提示词
        """
        try:
            # 准备文档分析结果（如果有）
            doc_insights = {}
            if document_analysis:
                doc_insights["content_analysis"] = document_analysis.to_dict()
            
            # 使用提示词服务增强
            enhanced_prompt = self.prompt_service.enhance_prompt(
                intent_analysis.intent,
                doc_insights
            )
            
            return enhanced_prompt
            
        except Exception as e:
            logger.error(f"Prompt enhancement failed: {str(e)}")
            raise
    
    def query_knowledge(self, query: str, domains: List[str] = None) -> Dict:
        """
        查询知识库
        
        Args:
            query: 查询内容
            domains: 限定领域（可选）
            
        Returns:
            Dict: 知识库查询结果
        """
        try:
            # 使用原知识库查询
            results = self.knowledge_base.query_knowledge(query, domains=domains)
            return results
        except Exception as e:
            logger.error(f"Knowledge base query failed: {str(e)}")
            raise
    
    def extract_events(self, news_texts: List[str]) -> List[Dict]:
        """
        从新闻文本中抽取结构化事件信息
        
        Args:
            news_texts: 新闻文本列表
            
        Returns:
            List[Dict]: 事件列表，每个事件包含时间、地点、实体、动作和情感
        """
        try:
            # 使用事件抽取服务
            events = self.event_extractor.extract_events(news_texts)
            return events
        except Exception as e:
            logger.error(f"Event extraction failed: {str(e)}")
            raise
    
    def match_events_to_prices(self, events: List[Dict], price_changes: List[Dict]) -> List[Dict]:
        """
        将新闻事件与价格波动进行智能匹配
        
        Args:
            events: 事件列表，每个事件包含description字段
            price_changes: 价格变化列表，每个变化包含description字段
            
        Returns:
            List[Dict]: 匹配结果列表，包含事件、价格变化和相似度分数
        """
        try:
            # 使用事件-价格匹配服务
            matches = self.event_price_matcher.compute_correlation(events, price_changes)
            return matches
        except Exception as e:
            logger.error(f"Event-price matching failed: {str(e)}")
            raise
    
    def extract_topics(self, report_text: str, num_topics: int = 5) -> List[Dict]:
        """
        从报告文本中提取核心主题
        
        Args:
            report_text: 报告文本
            num_topics: 要提取的主题数量
            
        Returns:
            List[Dict]: 主题列表，每个主题包含topic、keywords和weight
        """
        try:
            # 使用主题提取服务
            topics = self.topic_extractor.extract_topics(report_text, num_topics)
            return topics
        except Exception as e:
            logger.error(f"Topic extraction failed: {str(e)}")
            raise
    
    def extract_key_phrases(self, text: str, num_phrases: int = 10) -> List[Dict]:
        """
        从文本中提取关键短语
        
        Args:
            text: 输入文本
            num_phrases: 要提取的关键短语数量
            
        Returns:
            List[Dict]: 关键短语列表，包含phrase和weight
        """
        try:
            # 使用主题提取服务的关键短语提取功能
            key_phrases = self.topic_extractor.extract_key_phrases(text, num_phrases)
            return key_phrases
        except Exception as e:
            logger.error(f"Key phrase extraction failed: {str(e)}")
            raise
    
    def generate_response(self, user_input: str, 
                         system_prompt: Optional[str] = None, 
                         history: Optional[List[Dict[str, str]]] = None, 
                         max_length: Optional[int] = None) -> str:
        """
        使用ShieldLM模型生成响应
        
        Args:
            user_input: 用户输入文本
            system_prompt: 系统提示文本（可选）
            history: 对话历史（可选）
            max_length: 生成文本的最大长度（可选）
            
        Returns:
            str: 模型生成的响应
        """
        try:
            response = self.shieldlm_service.generate_chat_response(
                user_input=user_input,
                system_prompt=system_prompt,
                history=history,
                max_length=max_length
            )
            
            logger.info(f"Generated response for input: {user_input[:50]}...")
            return response
            
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            raise
    
    def get_knowledge_insights(self, query: str, domain: str = None) -> Dict:
        """
        获取知识库洞察
        
        Args:
            query: 查询内容
            domain: 限定领域（可选）
            
        Returns:
            Dict: 知识库洞察结果
        """
        try:
            # 查询知识库
            results = self.knowledge_base.query_knowledge(query, domains=[domain] if domain else None)
            
            # 扩展查询
            expanded_query = self.knowledge_base.expand_query(query, domains=[domain] if domain else None)
            
            # 提取关系知识
            relationships = []
            for term, _ in results.get("terms", []):
                relationships.extend(self.knowledge_base.get_relationships(term, domain=domain))
            
            insights = {
                "query": query,
                "expanded_query": expanded_query,
                "terms": results.get("terms", []),
                "rules": results.get("rules", []),
                "faq": results.get("faq", []),
                "domains": results.get("domains", []),
                "relationships": relationships
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Getting knowledge insights failed: {str(e)}")
            raise
    
    def combine_doc_analysis(self, doc_insights: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并多个文档分析结果
        
        Args:
            doc_insights: 文档分析结果字典
            
        Returns:
            Dict: 合并后的分析结果
        """
        combined = {
            "key_points": [],
            "data_points": [],
            "arguments": {
                "cause_effect_pairs": [],
                "conclusions": [],
                "opinions": [],
                "evidence": [],
                "comparisons": []
            },
            "sentiment": {
                "overall": "neutral",
                "score": 0.0,
                "word_count": 0
            },
            "word_count": 0,
            "sentence_count": 0,
            "files_analyzed": len(doc_insights)
        }
        
        # 合并各个字段
        for file, analysis in doc_insights.items():
            # 处理不同格式的分析结果
            if isinstance(analysis, DocumentAnalysis):
                analysis_dict = analysis.to_dict()
            else:
                analysis_dict = analysis
            
            # 合并关键点
            if "key_points" in analysis_dict:
                combined["key_points"].extend(analysis_dict["key_points"])
            
            # 合并数据点
            if "data_points" in analysis_dict:
                combined["data_points"].extend(analysis_dict["data_points"])
            
            # 合并论点
            if "arguments" in analysis_dict:
                for arg_type, args in analysis_dict["arguments"].items():
                    if arg_type in combined["arguments"]:
                        combined["arguments"][arg_type].extend(args)
            
            # 合并情感分析
            if "sentiment" in analysis_dict:
                sentiment = analysis_dict["sentiment"]
                combined["sentiment"]["word_count"] += sentiment.get("word_count", 0)
                combined["sentiment"]["score"] += sentiment.get("score", 0) * sentiment.get("word_count", 1)
            
            # 合并统计信息
            if "word_count" in analysis_dict:
                combined["word_count"] += analysis_dict["word_count"]
            
            if "sentence_count" in analysis_dict:
                combined["sentence_count"] += analysis_dict["sentence_count"]
        
        # 计算平均情感分数
        if combined["sentiment"]["word_count"] > 0:
            combined["sentiment"]["score"] /= combined["sentiment"]["word_count"]
            
            # 确定总体情感
            if combined["sentiment"]["score"] > 0.1:
                combined["sentiment"]["overall"] = "positive"
            elif combined["sentiment"]["score"] < -0.1:
                combined["sentiment"]["overall"] = "negative"
            else:
                combined["sentiment"]["overall"] = "neutral"
        
        # 限制关键点数量
        combined["key_points"] = combined["key_points"][:20]  # 最多保留20个关键点
        
        return combined
    
    def get_dialogue_context(self, num_turns: Optional[int] = None) -> str:
        """
        获取对话上下文
        
        Args:
            num_turns: 返回的对话轮次数，None表示返回全部
            
        Returns:
            str: 对话上下文字符串
        """
        return self.dialogue_manager.get_context_str(num_turns)
    
    def get_user_identity(self) -> Optional[Dict[str, Any]]:
        """
        获取当前识别的用户身份
        
        Returns:
            Optional[Dict[str, Any]]: 用户身份信息
        """
        return self.dialogue_manager.get_user_identity()
    
    def set_user_identity(self, identity: str, confidence: float = 1.0) -> None:
        """
        手动设置用户身份
        
        Args:
            identity: 用户身份类型
            confidence: 置信度
        """
        self.dialogue_manager.set_user_identity(identity, confidence)
    
    def clear_dialogue_history(self) -> None:
        """
        清除对话历史
        """
        self.dialogue_manager.clear_history()
    
    def get_dialogue_summary(self) -> Dict[str, Any]:
        """
        获取对话摘要
        
        Returns:
            Dict[str, Any]: 对话摘要信息
        """
        return self.dialogue_manager.get_dialogue_summary()

# 测试代码
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    try:
        # 创建NLP服务实例
        nlp_service = NLPService()
        print("✅ NLPService初始化成功")
        
        # 测试查询分析
        query = "分析本周市场价格走势"
        analysis = nlp_service.analyze_query(query)
        print(f"\n查询分析结果:")
        print(f"意图: {analysis.intent['deep']}")
        print(f"领域: {analysis.intent['domain']['primary']}")
        print(f"实体: {analysis.entities}")
        
        # 测试文档处理
        test_document = "本周市场价格表现强劲，周涨幅达到5.2%。主要指数下跌是主要影响因素之一。"
        doc_analysis = nlp_service.process_document(test_document)
        print(f"\n文档分析结果:")
        print(f"关键点: {doc_analysis.key_points}")
        print(f"数据点: {doc_analysis.data_points}")
        
        # 测试提示词增强
        enhanced_prompt = nlp_service.enhance_prompt(analysis, doc_analysis)
        print(f"\n增强提示词:")
        print(enhanced_prompt)
        
        # 测试知识库查询
        knowledge = nlp_service.query_knowledge("指数下跌影响", domains=["finance"])
        print(f"\n知识库查询结果:")
        print(f"规则数量: {len(knowledge.get('rules', []))}")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
