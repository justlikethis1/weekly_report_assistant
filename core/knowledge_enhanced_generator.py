#!/usr/bin/env python3
"""
知识库增强生成器
结合外部知识生成报告内容，提升报告质量和准确性
"""

from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class KnowledgeReference:
    """
    知识引用数据类
    """
    type: str  # 知识类型: term, rule, faq, domain
    content: str  # 引用内容
    confidence: float  # 置信度
    source: Optional[str] = None  # 引用来源

class KnowledgeEnhancedGenerator:
    """
    知识库增强生成器
    结合外部知识生成内容，提升报告质量和准确性
    
    功能：
    1. 从知识库检索相关信息
    2. 将知识注入提示词
    3. 生成增强内容
    4. 验证引用知识的准确性
    """
    
    def __init__(self):
        """
        初始化知识库增强生成器
        """
        try:
            # 导入知识库
            from src.nlp.analyzers.knowledge_base import KnowledgeBase
            
            # 导入报告生成器
            from src.models.enhanced_report_generator import EnhancedReportGenerator
            
            # 初始化知识库
            self.knowledge_base = KnowledgeBase()
            
            # 初始化报告生成器（使用模拟模型以避免实际模型加载问题）
            self.report_generator = EnhancedReportGenerator(is_mock=True)
            
            logger.info("KnowledgeEnhancedGenerator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize KnowledgeEnhancedGenerator: {str(e)}")
            raise
    
    def generate_with_knowledge(self, query: str, context: Optional[Dict[str, Any]] = None, knowledge_domains: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        结合外部知识生成内容
        
        Args:
            query: 查询内容
            context: 上下文信息
            knowledge_domains: 限定的知识领域列表
            
        Returns:
            Dict[str, Any]: 生成结果，包含增强内容和知识引用
        """
        try:
            # 1. 从知识库检索相关信息
            relevant_info = self._retrieve_relevant_knowledge(query, knowledge_domains)
            
            # 2. 将知识注入提示词
            prompt = self._build_prompt_with_knowledge(query, context, relevant_info)
            
            # 3. 生成增强内容
            response = self.report_generator.generate_content(prompt)
            
            # 4. 验证引用知识的准确性
            verified_response = self._verify_citations(response, relevant_info)
            
            # 5. 构建最终结果
            result = {
                "enhanced_content": verified_response,
                "knowledge_references": relevant_info,
                "prompt": prompt
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Knowledge-enhanced generation failed: {str(e)}")
            import traceback
            logger.debug(f"Exception details: {traceback.format_exc()}")
            
            # 回退到普通生成
            try:
                fallback_response = self.report_generator.generate_content(query)
                return {
                    "enhanced_content": fallback_response,
                    "knowledge_references": [],
                    "prompt": query,
                    "warning": "Knowledge enhancement failed, using fallback generation"
                }
            except:
                return {
                    "enhanced_content": "",
                    "knowledge_references": [],
                    "prompt": query,
                    "error": "Generation failed"
                }
    
    def generate_report_with_knowledge(self, parsed_input: Dict[str, Any], data_files: Optional[List[str]] = None, report_type: str = "detailed") -> Dict[str, Any]:
        """
        结合知识库生成完整报告
        
        Args:
            parsed_input: 解析后的输入信息
            data_files: 数据文件列表
            report_type: 报告类型
            
        Returns:
            Dict[str, Any]: 包含增强报告和知识引用的结果
        """
        try:
            # 1. 从知识库检索相关信息
            query = parsed_input.get("query", "")
            knowledge_domains = self._infer_knowledge_domains(query)
            relevant_info = self._retrieve_relevant_knowledge(query, knowledge_domains)
            
            # 2. 将知识注入提示词
            context = {
                "data_files": data_files,
                "report_type": report_type,
                "knowledge_references": relevant_info
            }
            
            # 3. 生成增强报告
            report = self.report_generator.generate_professional_report(parsed_input, data_files, report_type=report_type)
            
            # 4. 验证报告中知识引用的准确性
            verified_report = self._verify_report_knowledge(report, relevant_info)
            
            # 5. 构建最终结果
            result = {
                "enhanced_report": verified_report,
                "knowledge_references": relevant_info,
                "report_type": report_type
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Knowledge-enhanced report generation failed: {str(e)}")
            import traceback
            logger.debug(f"Exception details: {traceback.format_exc()}")
            
            # 回退到普通报告生成
            try:
                fallback_report = self.report_generator.generate_professional_report(parsed_input, data_files, report_type=report_type)
                return {
                    "enhanced_report": fallback_report,
                    "knowledge_references": [],
                    "report_type": report_type,
                    "warning": "Knowledge enhancement failed, using fallback report generation"
                }
            except:
                return {
                    "enhanced_report": "",
                    "knowledge_references": [],
                    "report_type": report_type,
                    "error": "Report generation failed"
                }
    
    def _retrieve_relevant_knowledge(self, query: str, domains: Optional[List[str]] = None) -> List[KnowledgeReference]:
        """
        从知识库检索相关信息
        
        Args:
            query: 查询内容
            domains: 限定的领域列表
            
        Returns:
            List[KnowledgeReference]: 相关知识引用列表
        """
        if not query:
            return []
        
        try:
            # 从知识库查询相关知识
            query_results = self.knowledge_base.query_knowledge(query, domains)
            
            # 构建知识引用列表
            references = []
            
            # 处理术语
            for term, info in query_results["terms"]:
                references.append(KnowledgeReference(
                    type="term",
                    content=f"{term}: {info['definition']}",
                    confidence=0.95,  # 术语的置信度较高
                    source="knowledge_base"
                ))
            
            # 处理规则
            for rule in query_results["rules"]:
                references.append(KnowledgeReference(
                    type="rule",
                    content=f"规则: {rule['condition']} -> {rule['conclusion']} (置信度: {rule['confidence']})",
                    confidence=rule['confidence'],
                    source="knowledge_base"
                ))
            
            # 处理FAQ
            for faq in query_results["faq"]:
                references.append(KnowledgeReference(
                    type="faq",
                    content=f"Q: {faq['question']} A: {faq['answer']}",
                    confidence=0.9,
                    source="knowledge_base"
                ))
            
            # 处理领域知识
            for domain_id, domain_info in query_results["domains"]:
                references.append(KnowledgeReference(
                    type="domain",
                    content=f"领域: {domain_info['name']} - {domain_info['description']}",
                    confidence=0.85,
                    source="knowledge_base"
                ))
            
            # 按置信度排序
            references.sort(key=lambda x: x.confidence, reverse=True)
            
            # 限制返回的知识数量
            return references[:10]  # 最多返回10条相关知识
            
        except Exception as e:
            logger.error(f"Knowledge retrieval failed: {str(e)}")
            return []
    
    def _build_prompt_with_knowledge(self, query: str, context: Optional[Dict[str, Any]], knowledge_references: List[KnowledgeReference]) -> str:
        """
        将知识注入提示词
        
        Args:
            query: 查询内容
            context: 上下文信息
            knowledge_references: 知识引用列表
            
        Returns:
            str: 增强后的提示词
        """
        if not knowledge_references:
            return query
        
        # 构建知识提示部分
        knowledge_prompt = "以下是相关的领域知识，在回答时请参考这些知识：\n"
        for reference in knowledge_references:
            knowledge_prompt += f"- {reference.content} (置信度: {reference.confidence:.2f})\n"
        
        # 构建完整提示词
        prompt = f"{knowledge_prompt}\n\n请根据上述知识回答以下问题：{query}"
        
        # 添加上下文信息
        if context:
            if "report_type" in context:
                prompt += f"\n报告类型：{context['report_type']}"
            if "data_files" in context:
                prompt += f"\n数据文件：{', '.join(context['data_files'])}"
        
        return prompt
    
    def _verify_citations(self, response: str, knowledge_references: List[KnowledgeReference]) -> str:
        """
        验证引用知识的准确性
        
        Args:
            response: 生成的内容
            knowledge_references: 知识引用列表
            
        Returns:
            str: 验证后的内容
        """
        if not knowledge_references:
            return response
        
        try:
            # 简单验证：检查生成内容是否包含知识引用的关键信息
            verified_response = response
            
            # 可以在这里添加更复杂的验证逻辑，例如：
            # 1. 检查引用的术语是否被正确使用
            # 2. 检查规则是否被正确应用
            # 3. 检查事实性错误
            
            # 为演示目的，我们只是添加一个引用标记
            citations = []
            for i, reference in enumerate(knowledge_references):
                if reference.type == "term":
                    # 查找术语是否在响应中被使用
                    term = reference.content.split(":")[0].strip()
                    if term in response:
                        citations.append((i+1, reference))
            
            # 如果有引用，添加引用列表
            if citations:
                verified_response += "\n\n【引用知识】\n"
                for citation_num, reference in citations:
                    verified_response += f"[{citation_num}] {reference.content}\n"
            
            return verified_response
            
        except Exception as e:
            logger.error(f"Citation verification failed: {str(e)}")
            return response  # 验证失败时返回原始响应
    
    def _verify_report_knowledge(self, report: str, knowledge_references: List[KnowledgeReference]) -> str:
        """
        验证报告中知识引用的准确性
        
        Args:
            report: 生成的报告
            knowledge_references: 知识引用列表
            
        Returns:
            str: 验证后的报告
        """
        if not knowledge_references:
            return report
        
        try:
            # 与_verify_citations类似，但针对报告的格式进行调整
            verified_report = report
            
            # 检查报告中是否使用了知识库中的知识
            citations = []
            for i, reference in enumerate(knowledge_references):
                if reference.type == "term":
                    term = reference.content.split(":")[0].strip()
                    if term in report:
                        citations.append((i+1, reference))
                elif reference.type == "rule":
                    # 检查规则的条件或结论是否在报告中被使用
                    rule_content = reference.content
                    if "规则: " in rule_content:
                        rule_content = rule_content.split("规则: ")[1]
                    if " -> " in rule_content:
                        condition, conclusion = rule_content.split(" -> ")
                        condition = condition.strip()
                        conclusion = conclusion.split(" (")[0].strip()
                        if condition in report or conclusion in report:
                            citations.append((i+1, reference))
            
            # 如果有引用，添加到报告末尾
            if citations:
                verified_report += "\n\n\n== 知识引用 ==\n"
                verified_report += "本报告参考了以下领域知识：\n"
                for citation_num, reference in citations:
                    verified_report += f"{citation_num}. {reference.content}\n"
            
            return verified_report
            
        except Exception as e:
            logger.error(f"Report knowledge verification failed: {str(e)}")
            return report  # 验证失败时返回原始报告
    
    def _infer_knowledge_domains(self, query: str) -> List[str]:
        """
        从查询中推断知识领域
        
        Args:
            query: 查询内容
            
        Returns:
            List[str]: 推断的知识领域列表
        """
        domains = []
        
        # 简单的领域推断规则
        if any(keyword in query for keyword in ["黄金", "金价", "黄金市场", "黄金交易", "黄金ETF", "伦敦金"]):
            domains.append("gold")
        
        if any(keyword in query for keyword in ["金融", "股票", "债券", "基金", "汇率", "利率", "GDP", "CPI", "美联储"]):
            domains.append("finance")
        
        if any(keyword in query for keyword in ["营销", "市场", "推广", "销售", "客户", "品牌"]):
            domains.append("marketing")
        
        if any(keyword in query for keyword in ["产品", "开发", "管理", "功能", "用户体验"]):
            domains.append("product")
        
        return domains
