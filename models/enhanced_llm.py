#!/usr/bin/env python3
"""
增强版LLM：集成NLP核心模块，提供智能分析和生成能力
"""

import sys
import os
import logging
import traceback
from typing import List, Dict, Any, Optional
from .report_monitor import monitor

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, project_root)

# 导入基础LLM类
from .llm import LocalLLM

# 导入统一的NLP服务
from src.nlp import NLPService

# 使用配置和日志管理
from src.infrastructure.utils.config_manager import config_manager
from src.infrastructure.utils.log_manager import log_manager

class EnhancedLLM(LocalLLM):
    """
    增强版LLM：集成NLP核心模块，提供智能分析和生成能力
    """
    
    def __init__(self, model_name: str = None, device: str = None, is_mock_model: bool = False, auto_unload: bool = False):
        """
        初始化增强版LLM
        
        Args:
            model_name: 模型名称
            device: 设备名称
            is_mock_model: 是否使用模拟模型
            auto_unload: 是否启用自动卸载
        """
        # 初始化基础LLM
        super().__init__(model_name, device, is_mock_model, auto_unload)
        
        # 初始化NLP核心模块
        self._init_nlp_modules()
        
        self.logger = logging.getLogger("EnhancedLLM")
        self.logger.info("EnhancedLLM initialized with NLP modules")
    
    def _init_nlp_modules(self):
        """
        初始化NLP核心模块
        """
        try:
            # 初始化统一的NLP服务 - 使用配置中的mock设置
            use_mock = config_manager.get("nlp.use_mock", False)
            self.nlp_service = NLPService(is_mock=use_mock)
            
            self.logger.info(f"NLP service initialized successfully, mock mode: {use_mock}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize NLP service: {str(e)}")
            self.logger.debug(f"NLP service initialization exception: {traceback.format_exc()}")
            
            # 初始化失败时，使用空对象
            self.nlp_service = None
    
    def generate_report(self, user_input: str, files: List = None) -> str:
        """
        生成报告：集成NLP模块进行智能分析和生成
        
        Args:
            user_input: 用户输入
            files: 文件列表（可选）
            
        Returns:
            str: 生成的报告
        """
        try:
            self.logger.info(f"Generating report with user input length: {len(user_input)}, files: {files}")
            
            # 1. 解析用户意图
            self.logger.info("Step 1: Parsing user intent")
            intent_analysis = self.nlp_service.analyze_query(user_input)
            self.logger.debug(f"Intent analysis result: {intent_analysis.intent}")
            
            # 2. 分析文档内容（如果有文件）
            self.logger.info("Step 2: Analyzing documents")
            doc_insights = {}
            doc_analyses = []
            
            if files:
                # 这里简化处理，实际应该读取文件内容并分析
                for file in files:
                    self.logger.debug(f"Analyzing file: {file}")
                    # 假设读取文件内容
                    content = f"This is sample content from file {file}"
                    # 使用NLP服务分析文件内容
                    analysis = self.nlp_service.process_document(content)
                    doc_insights[file] = analysis.to_dict()
                    doc_analyses.append(analysis)
            else:
                # 如果没有文件，分析用户输入本身
                self.logger.debug("No files provided, analyzing user input content")
                analysis = self.nlp_service.process_document(user_input)
                doc_insights["user_input"] = analysis.to_dict()
                doc_analyses.append(analysis)
            
            self.logger.debug(f"Document analysis results: {doc_insights}")
            
            # 3. 生成优化提示词
            self.logger.info("Step 3: Generating enhanced prompt")
            
            # 使用NLP服务生成增强提示词
            enhanced_prompt = self.nlp_service.enhance_prompt(
                intent_analysis, 
                doc_analyses[0] if doc_analyses else None
            )
            self.logger.debug(f"Enhanced prompt generated: {enhanced_prompt[:200]}...")
            
            # 4. 调用LLM生成报告
            self.logger.info("Step 4: Calling LLM to generate report")
            try:
                result = super().generate(enhanced_prompt)
                self.logger.info(f"Report generation completed, result length: {len(result)}")
                # 记录成功的LLM调用
                monitor.log_llm_call("report_generation", success=True)
                return result
            except Exception as llm_error:
                self.logger.error(f"LLM generation failed: {str(llm_error)}")
                # 记录失败的LLM调用
                monitor.log_llm_call("report_generation", success=False, error=str(llm_error))
                raise
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            self.logger.debug(f"Report generation exception: {traceback.format_exc()}")
            return f"报告生成失败：{str(e)}"
    
    def generate_chat_completion(self, messages: List[Dict[str, str]], max_length: int = None, language: str = "zh") -> str:
        """
        增强版聊天对话完成：集成NLP模块进行智能分析
        
        Args:
            messages: 聊天消息列表
            max_length: 最大生成长度
            language: 生成语言
            
        Returns:
            str: 生成的对话内容
        """
        try:
            self.logger.info(f"Enhanced chat completion with {len(messages)} messages")
            
            # 如果是最后一条消息是用户输入，进行意图分析
            if messages and messages[-1]["role"] == "user":
                user_input = messages[-1]["content"]
                
                # 使用NLP服务解析意图
                intent_analysis = self.nlp_service.analyze_query(user_input)
                self.logger.debug(f"Chat intent analysis: {intent_analysis.intent}")
                
                # 使用NLP服务分析用户输入内容
                doc_analysis = self.nlp_service.process_document(user_input)
                
                # 使用NLP服务生成增强提示词
                enhanced_prompt = self.nlp_service.enhance_prompt(intent_analysis, doc_analysis)
                
                # 创建新的消息列表，使用增强提示词
                enhanced_messages = messages.copy()
                enhanced_messages[-1]["content"] = enhanced_prompt
                
                # 调用基础LLM生成响应
                try:
                    result = super().generate_chat_completion(enhanced_messages, max_length, language)
                    # 记录成功的LLM调用
                    monitor.log_llm_call("chat_completion", success=True)
                    return result
                except Exception as llm_error:
                    self.logger.error(f"LLM chat completion failed: {str(llm_error)}")
                    # 记录失败的LLM调用
                    monitor.log_llm_call("chat_completion", success=False, error=str(llm_error))
                    raise
            
            # 否则使用基础实现
            try:
                result = super().generate_chat_completion(messages, max_length, language)
                # 记录成功的LLM调用
                monitor.log_llm_call("chat_completion", success=True)
                return result
            except Exception as llm_error:
                self.logger.error(f"LLM chat completion failed: {str(llm_error)}")
                # 记录失败的LLM调用
                monitor.log_llm_call("chat_completion", success=False, error=str(llm_error))
                raise
            
        except Exception as e:
            self.logger.error(f"Enhanced chat completion failed: {str(e)}")
            # 失败时回退到基础实现
            return super().generate_chat_completion(messages, max_length, language)
    
    def analyze_content(self, content: str, user_intent: str = None) -> Dict[str, Any]:
        """
        分析内容：使用NLP模块进行深度分析
        
        Args:
            content: 要分析的内容
            user_intent: 用户意图（可选）
            
        Returns:
            Dict: 分析结果
        """
        try:
            self.logger.info(f"Analyzing content with length: {len(content)}")
            
            # 如果没有提供用户意图，从内容中提取
            if not user_intent:
                user_intent = content
            
            # 1. 使用NLP服务进行意图解析
            intent_analysis = self.nlp_service.analyze_query(user_intent)
            
            # 2. 使用NLP服务进行文档分析
            doc_analysis = self.nlp_service.process_document(content)
            
            # 3. 使用NLP服务进行知识查询
            domain = intent_analysis.intent["domain"]["primary"]
            knowledge_results = self.nlp_service.query_knowledge(user_intent, domains=[domain])
            
            # 4. 使用NLP服务生成增强提示词
            enhanced_prompt = self.nlp_service.enhance_prompt(intent_analysis, doc_analysis)
            
            # 合并分析结果
            analysis_result = {
                "intent": intent_analysis.intent,
                "document_analysis": doc_analysis.to_dict(),
                "knowledge": knowledge_results,
                "enhanced_prompt": enhanced_prompt
            }
            
            self.logger.info("Content analysis completed")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Content analysis failed: {str(e)}")
            return {
                "error": str(e),
                "intent": None,
                "document_analysis": None,
                "knowledge": None,
                "enhanced_prompt": None
            }
    

    
    def get_knowledge_insights(self, query: str, domain: str = None) -> Dict[str, Any]:
        """
        获取知识库洞察
        
        Args:
            query: 查询内容
            domain: 限定领域
            
        Returns:
            Dict: 知识库洞察结果
        """
        try:
            self.logger.info(f"Getting knowledge insights for query: {query}")
            
            # 使用NLP服务获取知识库洞察
            insights = self.nlp_service.get_knowledge_insights(query, domain=domain)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to get knowledge insights: {str(e)}")
            return {
                "error": str(e),
                "query": query,
                "expanded_query": [],
                "terms": [],
                "rules": [],
                "faq": [],
                "domains": [],
                "relationships": []
            }

# 测试代码
if __name__ == "__main__":
    import sys
    
    # 配置日志
    log_manager.configure_logging()
    
    # 创建增强版LLM实例（使用模拟模型）
    enhanced_llm = EnhancedLLM(is_mock_model=True)
    
    # 测试报告生成
    print("=== 测试增强版LLM报告生成 ===")
    
    if len(sys.argv) > 1:
        user_input = sys.argv[1]
    else:
        user_input = "分析本周市场的走势和影响因素"
    
    print(f"用户输入: {user_input}")
    
    # 分析内容
    analysis_result = enhanced_llm.analyze_content(user_input)
    print("\n内容分析结果:")
    print(f"意图分析: {analysis_result['intent']['deep']}")
    print(f"文档分析关键点数量: {len(analysis_result['document_analysis']['key_points'])}")
    print(f"生成的增强提示词:")
    print(analysis_result['enhanced_prompt'])
    
    # 生成报告
    print("\n正在生成报告...")
    report = enhanced_llm.generate_report(user_input)
    print(f"\n生成的报告 (前500字符):")
    print(report[:500] + "...")
    
    # 测试知识库查询
    print("\n=== 测试知识库查询 ===")
    knowledge_insights = enhanced_llm.get_knowledge_insights("美元指数下跌影响", domain="gold")
    print(f"查询: 美元指数下跌影响")
    print(f"扩展查询: {knowledge_insights['expanded_query']}")
    print(f"相关规则:")
    for rule in knowledge_insights['rules']:
        print(f"  - {rule['condition']} -> {rule['conclusion']}")
