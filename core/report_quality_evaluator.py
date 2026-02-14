#!/usr/bin/env python3
"""
报告质量评估器
多维度评估报告质量，生成改进建议
"""

from typing import Dict, Any, List, Optional
import logging
import re
import textstat

logger = logging.getLogger(__name__)

class ReportQualityEvaluator:
    """
    报告质量评估器
    多维度评估报告质量，生成改进建议
    """
    
    def __init__(self, is_mock: bool = True):
        """
        初始化报告质量评估器
        
        Args:
            is_mock: 是否使用mock模式避免网络请求
        """
        try:
            # 导入NLP服务
            from src.nlp.services.nlp_service import NLPService
            
            # 初始化NLP服务 - 使用mock模式避免网络请求
            self.nlp_service = NLPService(is_mock=is_mock)
            self.use_mock = is_mock
            
            logger.info(f"ReportQualityEvaluator initialized successfully with mock mode: {is_mock}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ReportQualityEvaluator: {str(e)}")
            raise
    
    def evaluate_report(self, report: str, golden_standard: Optional[str] = None) -> Dict[str, Any]:
        """
        多维度评估报告质量
        
        Args:
            report: 要评估的报告文本
            golden_standard: 黄金标准报告文本（可选）
            
        Returns:
            Dict[str, Any]: 包含评估指标和改进建议的结果
        """
        if not report:
            return {
                "metrics": {},
                "suggestions": ["报告内容为空，无法进行评估"]
            }
        
        try:
            # 计算各个维度的评估指标
            metrics = {
                "coherence": self._measure_coherence(report),
                "relevance": self._measure_relevance(report, golden_standard),
                "insightfulness": self._count_insights(report),
                "readability": self._calculate_readability_score(report),
                "actionability": self._assess_actionable_recommendations(report)
            }
            
            # 生成改进建议
            suggestions = self._generate_improvement_suggestions(metrics)
            
            # 构建评估结果
            result = {
                "metrics": metrics,
                "suggestions": suggestions,
                "report_length": len(report)
            }
            
            # 如果提供了黄金标准，添加对比分析
            if golden_standard:
                result["comparison"] = self._compare_with_golden_standard(report, golden_standard)
            
            return result
            
        except Exception as e:
            logger.error(f"Report evaluation failed: {str(e)}")
            import traceback
            logger.debug(f"Exception details: {traceback.format_exc()}")
            return {
                "metrics": {},
                "suggestions": [f"评估失败: {str(e)}"],
                "error": str(e)
            }
    
    def _measure_coherence(self, report: str) -> float:
        """
        测量报告的连贯性
        
        Args:
            report: 报告文本
            
        Returns:
            float: 连贯性分数（0-1）
        """
        try:
            # 基于文本结构和逻辑连接词计算连贯性
            sentences = re.split(r'[。！？；:.;?!]', report)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) < 2:
                return 1.0 if sentences else 0.0
            
            # 计算逻辑连接词的使用频率
            logical_conjunctions = [
                "因此", "所以", "然而", "但是", "不过", "此外", "另外", "同时", "此外",
                "而且", "并且", "再者", "还有", "一方面", "另一方面", "首先", "其次", "最后"
            ]
            
            conjunction_count = 0
            for conjunction in logical_conjunctions:
                conjunction_count += report.count(conjunction)
            
            # 计算段落结构得分
            paragraphs = report.split('\n\n')
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            
            # 计算段落平均长度（适中为好）
            avg_paragraph_length = sum(len(p) for p in paragraphs) / len(paragraphs) if paragraphs else 0
            paragraph_score = 0.7 if 100 <= avg_paragraph_length <= 500 else min(avg_paragraph_length / 500, 1.0) if avg_paragraph_length < 500 else max(1.0 - (avg_paragraph_length - 500) / 500, 0.0)
            
            # 计算连贯性得分
            conjunction_density = conjunction_count / len(sentences) if sentences else 0
            coherence_score = (conjunction_density * 2 + paragraph_score) / 3
            
            # 确保得分在0-1之间
            coherence_score = max(0.0, min(1.0, coherence_score))
            
            return coherence_score
            
        except Exception as e:
            logger.error(f"Coherence measurement failed: {str(e)}")
            return 0.5  # 默认得分
    
    def _measure_relevance(self, report: str, golden_standard: Optional[str] = None) -> float:
        """
        测量报告的相关性
        
        Args:
            report: 报告文本
            golden_standard: 黄金标准报告文本（可选）
            
        Returns:
            float: 相关性分数（0-1）
        """
        try:
            if not golden_standard:
                # 如果没有黄金标准，基于报告内容的主题一致性评估相关性
                topics = self.nlp_service.extract_topics(report, num_topics=3)
                if not topics:
                    return 0.7  # 默认得分
                
                # 检查主题分布的均匀性
                main_topic_weight = topics[0]["weight"] if topics else 0.0
                relevance_score = 1.0 - main_topic_weight + 0.3  # 平衡主题分布
                return max(0.0, min(1.0, relevance_score))
            
            # 如果是mock模式，直接返回默认相似度
            if self.use_mock:
                return 0.8  # mock模式下返回较高的默认相似度
            
            # 如果不是mock模式，使用句子嵌入计算相似度
            try:
                from sentence_transformers import SentenceTransformer, util
                from src.nlp.config import nlp_config
                import os
                
                # 使用本地模型路径 - 从项目根目录开始构建绝对路径
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
                local_model_path = os.path.join(
                    project_root,
                    "models",
                    "sentence_transformers_new",
                    "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2",
                    "snapshots",
                    "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
                )
                
                # 使用句子嵌入计算相似度
                model = SentenceTransformer(
                    local_model_path,
                    cache_folder=nlp_config.sentence_transformer_cache,
                    use_auth_token=False,
                    device=nlp_config.device,
                    local_files_only=True
                )
                
                # 计算嵌入
                report_embedding = model.encode(report, convert_to_tensor=True)
                golden_embedding = model.encode(golden_standard, convert_to_tensor=True)
                
                # 计算余弦相似度
                similarity = util.cos_sim(report_embedding, golden_embedding).item()
                
                return max(0.0, min(1.0, similarity))
            except Exception as e:
                logger.error(f"Failed to calculate relevance using SentenceTransformer: {str(e)}")
                # 使用简单方法作为备用
                return 0.7
            
        except Exception as e:
            logger.error(f"Relevance measurement failed: {str(e)}")
            return 0.5  # 默认得分
    
    def _count_insights(self, report: str) -> int:
        """
        计算报告中的洞察数量
        
        Args:
            report: 报告文本
            
        Returns:
            int: 洞察数量
        """
        try:
            # 定义洞察关键词
            insight_keywords = [
                "发现", "洞察", "值得注意的是", "重要的是", "关键在于", "有趣的是",
                "表明", "显示", "揭示", "暗示", "意味着", "指出",
                "因此", "所以", "由此可见", "综上所述", "得出结论",
                "建议", "推荐", "应该", "需要", "可以考虑", "值得考虑"
            ]
            
            # 计算洞察数量
            insight_count = 0
            for keyword in insight_keywords:
                insight_count += report.count(keyword)
            
            # 检查是否包含数字分析
            numerical_analysis_patterns = [
                r"\d+\.\d+%",  # 百分比
                r"\d+\s*倍",  # 倍数
                r"上升\d+", "下降\d+", "增长\d+", "减少\d+",  # 变化
                r"从\d+到\d+", r"\d+至\d+",  # 范围
                r"相关性\s*\d+\.\d+", r"相关系数\s*\d+\.\d+"  # 相关性
            ]
            
            for pattern in numerical_analysis_patterns:
                matches = re.findall(pattern, report)
                insight_count += len(matches)
            
            return insight_count
            
        except Exception as e:
            logger.error(f"Insight counting failed: {str(e)}")
            return 0
    
    def _calculate_readability_score(self, report: str) -> float:
        """
        计算报告的可读性分数
        
        Args:
            report: 报告文本
            
        Returns:
            float: 可读性分数（0-1）
        """
        try:
            # 使用textstat计算可读性指标
            # 注意：textstat主要针对英文，我们需要结合中文特点进行调整
            
            # 计算句子长度
            sentences = re.split(r'[。！？；:.;?!]', report)
            sentences = [s.strip() for s in sentences if s.strip()]
            avg_sentence_length = sum(len(s) for s in sentences) / len(sentences) if sentences else 0
            
            # 计算字符密度（中文文本使用字符而不是单词）
            char_count = len(report.replace(' ', ''))
            word_count = len(report.split())  # 粗略估计单词数量
            
            # 计算可读性得分
            # 基于中文文本的特点：句子长度适中、用词简洁
            readability_score = 0.0
            
            # 句子长度评分（适中为好）
            if 20 <= avg_sentence_length <= 50:
                sentence_score = 1.0
            elif avg_sentence_length < 20:
                sentence_score = 0.8
            else:
                sentence_score = max(0.0, 1.0 - (avg_sentence_length - 50) / 50)
            
            # 用词简洁性评分
            if word_count > 0:
                # 字符/单词比（中文通常是1:1，但这里作为粗略指标）
                char_word_ratio = char_count / word_count
                if 1.0 <= char_word_ratio <= 2.0:
                    word_score = 1.0
                else:
                    word_score = max(0.0, 1.0 - abs(char_word_ratio - 1.5) / 1.5)
            else:
                word_score = 0.5
            
            # 综合评分
            readability_score = (sentence_score * 0.6 + word_score * 0.4)
            
            # 尝试使用textstat的英文可读性指标作为补充
            try:
                # 计算Flesch-Kincaid等级
                fk_grade = textstat.flesch_kincaid_grade(report)
                # 转换为0-1得分
                if 8 <= fk_grade <= 12:
                    fk_score = 1.0
                elif fk_grade < 8:
                    fk_score = 0.8
                else:
                    fk_score = max(0.0, 1.0 - (fk_grade - 12) / 8)
                
                # 综合textstat得分
                readability_score = (readability_score * 0.7 + fk_score * 0.3)
            except Exception:
                # 如果textstat计算失败，忽略
                pass
            
            return max(0.0, min(1.0, readability_score))
            
        except Exception as e:
            logger.error(f"Readability calculation failed: {str(e)}")
            return 0.5  # 默认得分
    
    def _assess_actionable_recommendations(self, report: str) -> float:
        """
        评估报告的可操作性建议
        
        Args:
            report: 报告文本
            
        Returns:
            float: 可操作性分数（0-1）
        """
        try:
            # 定义可操作性关键词
            actionable_keywords = [
                "建议", "推荐", "应该", "需要", "可以考虑", "值得考虑", "建议采取", "建议实施",
                "具体措施", "解决方案", "行动计划", "下一步", "需要关注", "重点关注",
                "可以通过", "建议通过", "应该通过", "需要通过", "可以利用", "建议利用"
            ]
            
            # 计算可操作性关键词数量
            actionable_count = 0
            for keyword in actionable_keywords:
                actionable_count += report.count(keyword)
            
            # 检查建议的具体性
            recommendation_patterns = [
                r"建议.*\d+",  # 包含数字的建议
                r"推荐.*\d+",  # 包含数字的推荐
                r"应该.*具体", r"需要.*具体",  # 具体的建议
                r"措施.*包括", r"方案.*包括",  # 包含具体措施的建议
                r"步骤.*\d+", r"阶段.*\d+"  # 分步骤的建议
            ]
            
            specific_recommendation_count = 0
            for pattern in recommendation_patterns:
                matches = re.findall(pattern, report)
                specific_recommendation_count += len(matches)
            
            # 计算可操作性得分
            if actionable_count == 0:
                return 0.0
            
            actionable_score = (actionable_count + specific_recommendation_count * 2) / (actionable_count + specific_recommendation_count * 2 + 5) * 0.7 + 0.3
            
            return max(0.0, min(1.0, actionable_score))
            
        except Exception as e:
            logger.error(f"Actionability assessment failed: {str(e)}")
            return 0.0
    
    def _generate_improvement_suggestions(self, metrics: Dict[str, Any]) -> List[str]:
        """
        生成改进建议
        
        Args:
            metrics: 评估指标
            
        Returns:
            List[str]: 改进建议列表
        """
        suggestions = []
        
        # 根据各个指标生成建议
        if "coherence" in metrics and metrics["coherence"] < 0.7:
            suggestions.append("提高报告的连贯性：增加逻辑连接词（如因此、然而、同时等），优化段落结构，确保内容流畅衔接。")
        
        if "relevance" in metrics and metrics["relevance"] < 0.7:
            suggestions.append("提高报告的相关性：聚焦核心主题，减少无关内容，确保所有章节都围绕报告目标展开。")
        
        if "insightfulness" in metrics and metrics["insightfulness"] < 5:
            suggestions.append("增加报告的洞察力：添加更多数据驱动的分析，使用具体数字和趋势，提供深入的解释和推断。")
        
        if "readability" in metrics and metrics["readability"] < 0.7:
            suggestions.append("提高报告的可读性：调整句子长度（建议20-50字符），使用简洁明了的语言，避免复杂冗长的表达。")
        
        if "actionability" in metrics and metrics["actionability"] < 0.7:
            suggestions.append("增加报告的可操作性：提供具体、可执行的建议，分步骤说明实施方案，包含明确的目标和时间线。")
        
        # 通用建议
        suggestions.append("建议在报告中增加可视化图表，提升数据呈现效果和可读性。")
        suggestions.append("建议添加结论和总结部分，突出关键发现和建议。")
        
        return suggestions
    
    def _compare_with_golden_standard(self, report: str, golden_standard: str) -> Dict[str, Any]:
        """
        与黄金标准对比分析
        
        Args:
            report: 要评估的报告文本
            golden_standard: 黄金标准报告文本
            
        Returns:
            Dict[str, Any]: 对比分析结果
        """
        try:
            # 提取主题
            report_topics = self.nlp_service.extract_topics(report, num_topics=5)
            golden_topics = self.nlp_service.extract_topics(golden_standard, num_topics=5)
            
            # 提取关键短语
            report_key_phrases = self.nlp_service.extract_key_phrases(report, num_phrases=10)
            golden_key_phrases = self.nlp_service.extract_key_phrases(golden_standard, num_phrases=10)
            
            # 计算主题重叠度
            report_topic_set = set(t["topic"] for t in report_topics)
            golden_topic_set = set(t["topic"] for t in golden_topics)
            topic_overlap = len(report_topic_set.intersection(golden_topic_set)) / len(golden_topic_set) if golden_topic_set else 0.0
            
            # 计算关键短语重叠度
            report_phrase_set = set(p["phrase"] for p in report_key_phrases)
            golden_phrase_set = set(p["phrase"] for p in golden_key_phrases)
            phrase_overlap = len(report_phrase_set.intersection(golden_phrase_set)) / len(golden_phrase_set) if golden_phrase_set else 0.0
            
            return {
                "topic_overlap": topic_overlap,
                "phrase_overlap": phrase_overlap,
                "report_topics": report_topics,
                "golden_topics": golden_topics,
                "report_key_phrases": report_key_phrases,
                "golden_key_phrases": golden_key_phrases
            }
            
        except Exception as e:
            logger.error(f"Golden standard comparison failed: {str(e)}")
            return {}
    
    def iterative_improvement(self, report: str, max_iterations: int = 3) -> Dict[str, Any]:
        """
        迭代优化报告质量
        
        Args:
            report: 原始报告文本
            max_iterations: 最大迭代次数
            
        Returns:
            Dict[str, Any]: 包含优化后的报告和改进历史的结果
        """
        try:
            improvement_history = []
            current_report = report
            
            for i in range(max_iterations):
                # 评估当前报告
                evaluation = self.evaluate_report(current_report)
                
                # 记录改进历史
                improvement_history.append({
                    "iteration": i + 1,
                    "metrics": evaluation["metrics"],
                    "suggestions": evaluation["suggestions"]
                })
                
                # 检查是否达到质量阈值
                avg_score = sum(evaluation["metrics"].values()) / len(evaluation["metrics"]) if evaluation["metrics"] else 0.0
                if avg_score >= 0.85:
                    break
                
                # 生成改进提示
                improvement_prompt = self._generate_improvement_prompt(current_report, evaluation["suggestions"])
                
                # 使用NLP服务改进报告
                from src.models.enhanced_report_generator import EnhancedReportGenerator
                report_generator = EnhancedReportGenerator()
                improved_report = report_generator.generate_content(improvement_prompt)
                
                # 更新当前报告
                current_report = improved_report
            
            # 最终评估
            final_evaluation = self.evaluate_report(current_report)
            improvement_history.append({
                "iteration": len(improvement_history) + 1,
                "metrics": final_evaluation["metrics"],
                "suggestions": final_evaluation["suggestions"],
                "is_final": True
            })
            
            return {
                "optimized_report": current_report,
                "improvement_history": improvement_history,
                "final_evaluation": final_evaluation
            }
            
        except Exception as e:
            logger.error(f"Iterative improvement failed: {str(e)}")
            import traceback
            logger.debug(f"Exception details: {traceback.format_exc()}")
            return {
                "optimized_report": report,  # 返回原始报告
                "improvement_history": [],
                "error": str(e)
            }
    
    def _generate_improvement_prompt(self, report: str, suggestions: List[str]) -> str:
        """
        生成改进提示
        
        Args:
            report: 原始报告文本
            suggestions: 改进建议列表
            
        Returns:
            str: 改进提示文本
        """
        prompt = f"请根据以下改进建议优化报告内容：\n"
        prompt += "\n" + "\n".join([f"- {s}" for s in suggestions])
        prompt += f"\n\n原始报告：\n{report}\n"
        prompt += "\n请保持报告的核心内容和结构，仅根据建议进行优化，确保优化后的报告更加连贯、相关、有洞察力、易读且具有可操作性。"
        
        return prompt
