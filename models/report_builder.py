#!/usr/bin/env python3
"""
报告构建器层
负责组合报告章节、集成LLM和适配输出格式
"""

from typing import Dict, Any, List, Optional
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class SectionComposer:
    """章节组合器，负责组合报告章节"""
    
    def __init__(self, template_engine=None):
        """
        初始化章节组合器
        
        Args:
            template_engine: 模板引擎
        """
        self.template_engine = template_engine
    
    def set_template_engine(self, template_engine):
        """
        设置模板引擎
        
        Args:
            template_engine: 模板引擎
        """
        self.template_engine = template_engine
    
    def compose_sections(self, data: Dict[str, Any], report_config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        根据数据和配置组合报告章节
        
        Args:
            data: 报告数据
            report_config: 报告配置
            
        Returns:
            组合后的章节列表
        """
        sections = []
        
        # 如果有报告配置，使用配置中的章节定义
        if report_config and 'template_sections' in report_config:
            for section_config in report_config['template_sections']:
                section = self._compose_section(data, section_config)
                if section:
                    sections.append(section)
        else:
            # 否则使用默认章节结构
            sections = self._create_default_sections(data)
        
        return sections
    
    def _compose_section(self, data: Dict[str, Any], section_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        组合单个章节
        
        Args:
            data: 报告数据
            section_config: 章节配置
            
        Returns:
            组合后的章节
        """
        if not section_config or 'id' not in section_config or 'title' not in section_config:
            return None
        
        section = {
            'id': section_config['id'],
            'title': section_config['title'],
            'content': '',
            'importance': section_config.get('importance', 3),
            'required': section_config.get('required', False)
        }
        
        # 检查是否需要跳过该章节
        if 'data_binding' in section_config and section_config['data_binding'] not in data:
            return None
        
        # 使用模板渲染内容
        if self.template_engine and 'template' in section_config:
            content = self.template_engine.render(section_config['template'], data)
            if content:
                section['content'] = content
        else:
            # 生成默认内容
            section['content'] = self._generate_default_section_content(section_config['id'], data)
        
        # 验证章节内容
        if self._validate_section(section, section_config):
            return section
        
        return None
    
    def _create_default_sections(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        创建默认章节结构
        
        Args:
            data: 报告数据
            
        Returns:
            默认章节列表
        """
        sections = []
        
        # 添加执行摘要
        sections.append({
            'id': 'executive_summary',
            'title': '执行摘要',
            'content': self._generate_default_section_content('executive_summary', data),
            'importance': 5,
            'required': True
        })
        
        # 添加数据概览
        sections.append({
            'id': 'data_overview',
            'title': '数据概览',
            'content': self._generate_default_section_content('data_overview', data),
            'importance': 5,
            'required': True
        })
        
        # 添加统计分析
        sections.append({
            'id': 'statistical_analysis',
            'title': '统计分析',
            'content': self._generate_default_section_content('statistical_analysis', data),
            'importance': 4,
            'required': True
        })
        
        # 添加洞察与建议
        sections.append({
            'id': 'insights_and_recommendations',
            'title': '洞察与建议',
            'content': self._generate_default_section_content('insights_and_recommendations', data),
            'importance': 5,
            'required': True
        })
        
        return sections
    
    def _generate_default_section_content(self, section_id: str, data: Dict[str, Any]) -> str:
        """
        生成默认章节内容
        
        Args:
            section_id: 章节ID
            data: 报告数据
            
        Returns:
            默认章节内容
        """
        if section_id == 'executive_summary':
            return self._generate_default_executive_summary(data)
        elif section_id == 'data_overview':
            return self._generate_default_data_overview(data)
        elif section_id == 'statistical_analysis':
            return self._generate_default_statistical_analysis(data)
        elif section_id == 'insights_and_recommendations':
            return self._generate_default_insights_and_recommendations(data)
        else:
            return f"这是{section_id}章节的默认内容。"
    
    def _generate_default_executive_summary(self, data: Dict[str, Any]) -> str:
        """
        生成默认执行摘要
        
        Args:
            data: 报告数据
            
        Returns:
            默认执行摘要内容
        """
        summary = "# 执行摘要\n\n"
        
        # 添加数据状态标记
        if 'data_quality' in data and data['data_quality']:
            data_quality = data['data_quality']
            score = data_quality.get('score', 0)
            if score >= 7:
                status_text = "优秀"
            elif score >= 5:
                status_text = "良好"
            else:
                status_text = "一般"
            summary += f"✅ **数据状态**：{status_text}（得分：{score}/10）\n\n"
        
        if 'price_data' in data and data['price_data']:
            price_data = data['price_data']
            if 'start_price' in price_data and 'end_price' in price_data:
                change_percentage = ((price_data['end_price'] - price_data['start_price']) / price_data['start_price']) * 100 if price_data['start_price'] != 0 else 0
                summary += f"📊 **核心发现**：价格从{price_data['start_price']}{price_data.get('unit', '')}变动到{price_data['end_price']}{price_data.get('unit', '')}，变动幅度为{change_percentage:.2f}%\n\n"
        
        summary += "报告包含数据概览、统计分析和洞察建议等章节，为决策提供数据支持。"
        
        return summary
    
    def _generate_default_data_overview(self, data: Dict[str, Any]) -> str:
        """
        生成默认数据概览
        
        Args:
            data: 报告数据
            
        Returns:
            默认数据概览内容
        """
        overview = "# 数据概览\n\n"
        
        if 'price_data' in data and data['price_data']:
            price_data = data['price_data']
            
            overview += "## 价格数据\n\n"
            overview += f"起始价格: {price_data.get('start_price', 0)}{price_data.get('unit', '')}\n"
            overview += f"结束价格: {price_data.get('end_price', 0)}{price_data.get('unit', '')}\n"
            overview += f"最高价格: {price_data.get('high_price', 0)}{price_data.get('unit', '')}\n"
            overview += f"最低价格: {price_data.get('low_price', 0)}{price_data.get('unit', '')}\n\n"
            
            if 'price_changes' in price_data and price_data['price_changes']:
                overview += "## 价格变动\n\n"
                overview += "| 日期 | 价格 | 涨跌幅 |\n"
                overview += "|------|------|--------|\n"
                
                # 只显示最近10条记录
                recent_changes = price_data['price_changes'][-10:]
                for change in recent_changes:
                    change_str = f"+{change['change']}%" if change['change'] > 0 else f"{change['change']}%"
                    overview += f"| {change['date']} | {change['price']}{price_data.get('unit', '')} | {change_str} |\n"
        
        return overview
    
    def _generate_default_statistical_analysis(self, data: Dict[str, Any]) -> str:
        """
        生成默认统计分析
        
        Args:
            data: 报告数据
            
        Returns:
            默认统计分析内容
        """
        analysis = "# 统计分析\n\n"
        
        if 'statistical_analysis' in data and data['statistical_analysis']:
            stats = data['statistical_analysis']
            
            if 'basic_statistics' in stats and stats['basic_statistics']:
                analysis += "## 基本统计指标\n\n"
                for key, value in stats['basic_statistics'].items():
                    analysis += f"- {key}: {value:.2f}\n"
                analysis += "\n"
            
            if 'volatility' in stats and stats['volatility']:
                analysis += "## 波动率分析\n\n"
                for key, value in stats['volatility'].items():
                    if isinstance(value, float):
                        analysis += f"- {key}: {value:.2f}\n"
                    else:
                        analysis += f"- {key}: {value}\n"
                analysis += "\n"
            
            if 'trend' in stats and stats['trend']:
                analysis += "## 趋势分析\n\n"
                trend = stats['trend']
                direction_text = '上涨' if trend['direction'] == 'upward' else ('下跌' if trend['direction'] == 'downward' else '稳定')
                analysis += f"- 趋势方向: {direction_text}\n"
                analysis += f"- 变动幅度: {trend.get('percentage_change', 0):.2f}%\n"
        
        return analysis
    
    def _generate_default_insights_and_recommendations(self, data: Dict[str, Any]) -> str:
        """
        生成默认洞察与建议
        
        Args:
            data: 报告数据
            
        Returns:
            默认洞察与建议内容
        """
        insights = "# 洞察与建议\n\n"
        
        if 'insights' in data and data['insights']:
            insights += "## 关键洞察\n\n"
            for insight in data['insights']:
                insights += f"### {insight['title']}\n"
                insights += f"{insight['description']}\n\n"
        
        insights += "## 建议\n\n"
        insights += "1. 密切关注价格变动趋势\n"
        insights += "2. 结合统计分析结果制定策略\n"
        insights += "3. 定期更新数据并重新评估分析结果"
        
        return insights
    
    def _validate_section(self, section: Dict[str, Any], section_config: Dict[str, Any]) -> bool:
        """
        验证章节内容
        
        Args:
            section: 章节内容
            section_config: 章节配置
            
        Returns:
            如果验证通过返回True，否则返回False
        """
        # 检查内容长度
        if 'min_length' in section_config:
            min_length = section_config['min_length']
            if len(section['content']) < min_length:
                logger.warning(f"章节 {section['id']} 内容长度不足 ({len(section['content'])} < {min_length})")
                return not section.get('required', False)
        
        if 'max_length' in section_config:
            max_length = section_config['max_length']
            if len(section['content']) > max_length:
                logger.warning(f"章节 {section['id']} 内容长度超过限制 ({len(section['content'])} > {max_length})")
        
        return True


class LLMIntegrator:
    """
    LLM集成器：负责与LLM集成，生成报告内容
    """
    
    def __init__(self, mock_mode: bool = True):
        """
        初始化LLM集成器
        
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
        return f"LLM生成的内容（{strategy_type}）：基于提示 '{prompt[:50]}...' 和提供的数据"
    
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
                result = self.enhanced_llm.analyze_content(prompt)
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


class FormatAdapter:
    """格式适配器，负责将报告内容适配为不同格式"""
    
    def __init__(self):
        """初始化格式适配器"""
        self.supported_formats = ['markdown', 'html', 'plain_text']
    
    def can_adapt(self, format_type: str) -> bool:
        """
        检查是否支持该格式
        
        Args:
            format_type: 格式类型
            
        Returns:
            如果支持返回True，否则返回False
        """
        return format_type.lower() in self.supported_formats
    
    def adapt(self, sections: List[Dict[str, Any]], format_type: str = 'markdown', metadata: Dict[str, Any] = None) -> str:
        """
        将报告章节适配为指定格式
        
        Args:
            sections: 报告章节
            format_type: 目标格式
            metadata: 报告元数据
            
        Returns:
            适配后的内容
        """
        if not self.can_adapt(format_type):
            logger.error(f"不支持的格式: {format_type}")
            return ""
        
        format_type = format_type.lower()
        
        if format_type == 'markdown':
            return self._adapt_to_markdown(sections, metadata)
        elif format_type == 'html':
            return self._adapt_to_html(sections, metadata)
        elif format_type == 'plain_text':
            return self._adapt_to_plain_text(sections, metadata)
        else:
            return ""
    
    def _adapt_to_markdown(self, sections: List[Dict[str, Any]], metadata: Dict[str, Any] = None) -> str:
        """
        适配为Markdown格式
        
        Args:
            sections: 报告章节
            metadata: 报告元数据
            
        Returns:
            Markdown格式的报告
        """
        markdown_content = ""
        
        # 添加报告标题
        if metadata and 'title' in metadata:
            markdown_content += f"# {metadata['title']}\n\n"
        
        # 添加报告日期
        if metadata and 'date' in metadata:
            markdown_content += f"*报告日期: {metadata['date']}*\n\n"
        
        # 添加章节内容
        for section in sections:
            markdown_content += f"{section['content']}\n\n"
        
        return markdown_content.strip()
    
    def _adapt_to_html(self, sections: List[Dict[str, Any]], metadata: Dict[str, Any] = None) -> str:
        """
        适配为HTML格式
        
        Args:
            sections: 报告章节
            metadata: 报告元数据
            
        Returns:
            HTML格式的报告
        """
        html_content = "<!DOCTYPE html>\n<html>\n<head>\n"
        html_content += "<meta charset='UTF-8'>\n"
        html_content += "<title>报告</title>\n"
        html_content += "<style>\n"
        html_content += "body { font-family: Arial, sans-serif; margin: 20px; }\n"
        html_content += "h1, h2, h3 { color: #333; }\n"
        html_content += "table { border-collapse: collapse; width: 100%; margin: 20px 0; }\n"
        html_content += "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n"
        html_content += "th { background-color: #f2f2f2; }\n"
        html_content += "</style>\n"
        html_content += "</head>\n<body>\n"
        
        # 添加报告标题
        if metadata and 'title' in metadata:
            html_content += f"<h1>{metadata['title']}</h1>\n"
        
        # 添加报告日期
        if metadata and 'date' in metadata:
            html_content += f"<p><em>报告日期: {metadata['date']}</em></p>\n"
        
        # 添加章节内容
        for section in sections:
            html_content += f"<div class='section'>\n"
            html_content += f"<h2>{section['title']}</h2>\n"
            html_content += f"<div class='content'>{section['content']}</div>\n"
            html_content += "</div>\n"
        
        html_content += "</body>\n</html>"
        
        return html_content
    
    def _adapt_to_plain_text(self, sections: List[Dict[str, Any]], metadata: Dict[str, Any] = None) -> str:
        """
        适配为纯文本格式
        
        Args:
            sections: 报告章节
            metadata: 报告元数据
            
        Returns:
            纯文本格式的报告
        """
        text_content = ""
        
        # 添加报告标题
        if metadata and 'title' in metadata:
            text_content += f"{'='*50}\n"
            text_content += f"{metadata['title']}\n"
            text_content += f"{'='*50}\n\n"
        
        # 添加报告日期
        if metadata and 'date' in metadata:
            text_content += f"报告日期: {metadata['date']}\n\n"
        
        # 添加章节内容
        for section in sections:
            text_content += f"{'-'*30}\n"
            text_content += f"{section['title']}\n"
            text_content += f"{'-'*30}\n"
            text_content += f"{section['content']}\n\n"
        
        return text_content.strip()
