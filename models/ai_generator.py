#!/usr/bin/env python3
"""
AI生成器模块
负责使用AI生成报告内容
"""

from typing import Dict, Any, List, Optional
import logging
import concurrent.futures
from functools import partial

logger = logging.getLogger(__name__)


class AIGenerator:
    """AI生成器，负责使用AI生成报告内容"""
    
    def __init__(self, mock_mode: bool = True):
        """
        初始化AI生成器
        
        Args:
            mock_mode: 是否使用模拟模式
        """
        self._mock_mode = mock_mode
        
        # 初始化LLM调用策略
        self.llm_strategies = {
            'full_generation': self._full_generation_strategy,
            'template_filling': self._template_filling_strategy,
            'insight_augmentation': self._insight_augmentation_strategy,
            'summary_generation': self._summary_generation_strategy
        }
        
        # 创建EnhancedLLM实例
        from .enhanced_llm import EnhancedLLM
        self.enhanced_llm = EnhancedLLM(is_mock_model=mock_mode)
        
        # 创建AI输出规范化器实例
        from .ai_output_normalizer import AIOutputNormalizer
        self.ai_normalizer = AIOutputNormalizer()
        
        logger.info("AIGenerator初始化完成")
    
    @property
    def mock_mode(self):
        return self._mock_mode
    
    @mock_mode.setter
    def mock_mode(self, value):
        self._mock_mode = value
        # 同时更新EnhancedLLM实例的is_mock_model属性
        if hasattr(self, 'enhanced_llm'):
            self.enhanced_llm.is_mock_model = value
            logger.info(f"EnhancedLLM mock mode updated to: {value}")
    
    def generate_sections(self, sections: List[Dict[str, Any]], data: Dict[str, Any], strategy_type: str = 'template_filling', max_workers: int = 2) -> List[Dict[str, Any]]:
        """
        使用AI增强多个章节的内容
        
        Args:
            sections: 基础章节列表
            data: 参考数据
            strategy_type: 生成策略类型
            max_workers: 并行处理的最大工作线程数
            
        Returns:
            AI增强后的章节列表
        """
        enhanced_sections = []
        
        # 并行处理章节增强
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 创建部分应用函数
            generate_func = partial(self._enhance_section, data=data, strategy_type=strategy_type)
            
            # 提交所有章节增强任务
            future_to_section = {executor.submit(generate_func, section): section for section in sections}
            
            # 获取任务结果
            for future in concurrent.futures.as_completed(future_to_section):
                section = future_to_section[future]
                try:
                    enhanced_section = future.result()
                    if enhanced_section:
                        enhanced_sections.append(enhanced_section)
                except Exception as e:
                    logger.error(f"章节 {section.get('id', '未知')} 增强失败: {e}", exc_info=True)
                    # 失败时保留原始章节
                    enhanced_sections.append(section)
        
        # 保持章节顺序
        enhanced_sections.sort(key=lambda x: next((i for i, s in enumerate(sections) if s['id'] == x['id']), len(sections)))
        
        return enhanced_sections
    
    def _enhance_section(self, section: Dict[str, Any], data: Dict[str, Any], strategy_type: str = 'template_filling') -> Dict[str, Any]:
        """
        使用AI增强单个章节的内容
        
        Args:
            section: 原始章节
            data: 参考数据
            strategy_type: 生成策略类型
            
        Returns:
            AI增强后的章节
        """
        enhanced_section = section.copy()
        
        # 根据章节类型生成不同的提示词
        prompt = self._generate_section_prompt(section, data)
        
        try:
            # 生成增强内容
            ai_content = self.generate_content(prompt, strategy_type, data)
            
            # 更新章节内容
            enhanced_section['content'] = ai_content
            enhanced_section['source'] = 'ai_generated'  # 标记内容来源
            enhanced_section['enhanced'] = True
            
            logger.info(f"章节 {section['id']} AI增强完成")
            return enhanced_section
            
        except Exception as e:
            logger.error(f"章节 {section['id']} AI增强失败: {e}", exc_info=True)
            # 失败时保留原始章节
            enhanced_section['enhanced'] = False
            enhanced_section['error'] = str(e)
            return enhanced_section
    
    def _generate_section_prompt(self, section: Dict[str, Any], data: Dict[str, Any]) -> str:
        """
        为特定章节生成提示词
        
        Args:
            section: 章节信息
            data: 参考数据
            
        Returns:
            生成的提示词
        """
        section_id = section['id']
        section_title = section['title']
        section_content = section['content']
        
        # 根据章节类型生成不同的提示词
        if section_id == 'executive_summary':
            prompt = f"""请基于以下数据和现有执行摘要，生成一个更全面、更有洞察性的执行摘要：

现有摘要：
{section_content}

参考数据：
{data}

要求：
1. 保持专业、简洁的风格
2. 突出核心发现和业务价值
3. 控制在200-300字以内
"""
        elif section_id == 'data_overview':
            prompt = f"""请基于以下数据和现有数据概览，生成一个更生动、更易懂的数据概览：

现有概览：
{section_content}

参考数据：
{data}

要求：
1. 使用清晰的结构和可视化的描述
2. 突出关键数据点和趋势
3. 避免过于技术化的术语
"""
        elif section_id == 'statistical_analysis':
            prompt = f"""请基于以下数据和现有统计分析，生成一个更深入、更专业的统计分析：

现有分析：
{section_content}

参考数据：
{data}

要求：
1. 深入解释统计指标的含义和业务意义
2. 识别数据中的模式和异常
3. 使用专业的统计术语
"""
        elif section_id == 'insights_and_recommendations':
            prompt = f"""请基于以下数据和现有洞察与建议，生成更深入、更有价值的洞察与建议：

现有洞察与建议：
{section_content}

参考数据：
{data}

要求：
1. 提供具体、可操作的建议
2. 基于数据提供有力的支持
3. 考虑潜在的业务影响
"""
        else:
            prompt = f"""请基于以下数据和现有内容，生成一个更全面、更有价值的{section_title}：

现有内容：
{section_content}

参考数据：
{data}

要求：
1. 保持专业的风格
2. 确保内容准确、完整
3. 突出关键信息
"""
        
        return prompt
    
    def generate_content(self, prompt: str, strategy_type: str = 'template_filling', data: Dict[str, Any] = None) -> str:
        """
        使用LLM生成内容
        
        Args:
            prompt: 提示词
            strategy_type: 生成策略类型
            data: 参考数据
            
        Returns:
            生成的内容
        """
        if strategy_type not in self.llm_strategies:
            logger.warning(f"未知的生成策略: {strategy_type}，使用默认策略")
            strategy_type = 'template_filling'
        
        try:
            if self.mock_mode:
                generated_content = self._mock_generate_content(prompt, strategy_type, data)
            else:
                generated_content = self._real_llm_generate_content(prompt, strategy_type, data)
            
            # 记录成功的LLM调用
            from .report_monitor import monitor
            monitor.log_llm_call(strategy_type, success=True)
            
            # 使用AI输出规范化器处理生成的内容
            logger.debug(f"规范化LLM生成的内容，长度: {len(generated_content)}")
            
            # 准备上下文信息用于质量评估
            context = {
                'prompt': prompt,
                'strategy_type': strategy_type,
                'data_keys': list(data.keys()) if data else []
            }
            
            # 规范化内容
            normalized_result = self.ai_normalizer.normalize(generated_content, context)
            
            logger.debug(f"内容规范化完成，质量评分: {normalized_result['quality_score']}, 移除元数据: {normalized_result['metadata_removed']}")
            
            return normalized_result['normalized_content']
        except Exception as e:
            logger.error(f"LLM生成内容失败: {e}")
            # 记录失败的LLM调用
            from .report_monitor import monitor
            monitor.log_llm_call(strategy_type, success=False, error=str(e))
            return f"LLM生成内容失败：{str(e)}"
    
    def _mock_generate_content(self, prompt: str, strategy_type: str, data: Dict[str, Any] = None) -> str:
        """
        模拟生成内容
        
        Args:
            prompt: 提示词
            strategy_type: 生成策略类型
            data: 参考数据
            
        Returns:
            模拟生成的内容
        """
        # 简单的模拟生成
        return f"AI生成的内容（{strategy_type}）：基于提示 '{prompt[:50]}...' 和提供的数据"
    
    def _real_llm_generate_content(self, prompt: str, strategy_type: str, data: Dict[str, Any] = None) -> str:
        """
        使用真实LLM生成内容
        
        Args:
            prompt: 提示词
            strategy_type: 生成策略类型
            data: 参考数据
            
        Returns:
            LLM生成的内容
        """
        try:
            # 使用EnhancedLLM生成内容
            if strategy_type == 'full_generation':
                # 完全生成策略
                result = self.enhanced_llm.generate(prompt)
            elif strategy_type == 'template_filling':
                # 模板填充策略
                # 结合数据生成增强提示词
                enhanced_prompt = f"{prompt}\n\n参考数据：{str(data)}"
                result = self.enhanced_llm.generate(enhanced_prompt)
            elif strategy_type == 'insight_augmentation':
                # 洞察增强策略
                # analyze_content返回的是字典，需要提取相关信息
                analysis_result = self.enhanced_llm.analyze_content(prompt)
                if isinstance(analysis_result, dict) and 'enhanced_prompt' in analysis_result:
                    # 使用增强提示词再次生成内容
                    result = self.enhanced_llm.generate(analysis_result['enhanced_prompt'])
                else:
                    result = self.enhanced_llm.generate(prompt)
            elif strategy_type == 'summary_generation':
                # 摘要生成策略
                result = self.enhanced_llm.generate(prompt)
            else:
                # 默认策略
                result = self.enhanced_llm.generate(prompt)
            
            return result
        except Exception as e:
            logger.error(f"真实LLM生成内容失败: {e}")
            # 失败时回退到模拟生成
            return self._mock_generate_content(prompt, strategy_type, data)
    
    def _full_generation_strategy(self, prompt: str, data: Dict[str, Any] = None) -> str:
        """
        完全生成策略
        
        Args:
            prompt: 提示词
            data: 参考数据
            
        Returns:
            生成的内容
        """
        return self.generate_content(prompt, 'full_generation', data)
    
    def _template_filling_strategy(self, prompt: str, data: Dict[str, Any] = None) -> str:
        """
        模板填充策略
        
        Args:
            prompt: 提示词（包含模板）
            data: 填充数据
            
        Returns:
            生成的内容
        """
        return self.generate_content(prompt, 'template_filling', data)
    
    def _insight_augmentation_strategy(self, prompt: str, data: Dict[str, Any] = None) -> str:
        """
        洞察增强策略
        
        Args:
            prompt: 提示词
            data: 参考数据
            
        Returns:
            生成的内容
        """
        return self.generate_content(prompt, 'insight_augmentation', data)
    
    def _summary_generation_strategy(self, prompt: str, data: Dict[str, Any] = None) -> str:
        """
        摘要生成策略
        
        Args:
            prompt: 提示词
            data: 参考数据
            
        Returns:
            生成的内容
        """
        return self.generate_content(prompt, 'summary_generation', data)
