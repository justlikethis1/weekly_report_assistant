#!/usr/bin/env python3
"""
规则引擎模块
负责基于规则和模板生成报告内容
"""

from typing import Dict, Any, List, Optional
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)


class RuleEngine:
    """规则引擎，负责基于规则和模板生成报告内容"""
    
    def __init__(self, template_engine=None):
        """
        初始化规则引擎
        
        Args:
            template_engine: 模板引擎
        """
        self.template_engine = template_engine
        logger.info("RuleEngine初始化完成")
    
    def set_template_engine(self, template_engine):
        """
        设置模板引擎
        
        Args:
            template_engine: 模板引擎
        """
        self.template_engine = template_engine
    
    def generate_sections(self, data: Dict[str, Any], report_config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        根据数据和配置生成报告章节
        
        Args:
            data: 报告数据
            report_config: 报告配置
            
        Returns:
            生成的章节列表
        """
        sections = []
        
        # 如果有报告配置，使用配置中的章节定义
        if report_config and 'template_sections' in report_config:
            for section_config in report_config['template_sections']:
                section = self._generate_section(data, section_config)
                if section:
                    sections.append(section)
        else:
            # 否则使用默认章节结构
            sections = self._create_default_sections(data)
        
        return sections
    
    def _generate_section(self, data: Dict[str, Any], section_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        生成单个章节
        
        Args:
            data: 报告数据
            section_config: 章节配置
            
        Returns:
            生成的章节
        """
        if not section_config or 'id' not in section_config or 'title' not in section_config:
            return None
        
        section = {
            'id': section_config['id'],
            'title': section_config['title'],
            'content': '',
            'importance': section_config.get('importance', 3),
            'required': section_config.get('required', False),
            'source': 'rule_based'  # 标记内容来源
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
            'content': self._generate_default_executive_summary(data),
            'importance': 5,
            'required': True,
            'source': 'rule_based'
        })
        
        # 添加数据概览
        sections.append({
            'id': 'data_overview',
            'title': '数据概览',
            'content': self._generate_default_data_overview(data),
            'importance': 5,
            'required': True,
            'source': 'rule_based'
        })
        
        # 添加统计分析
        sections.append({
            'id': 'statistical_analysis',
            'title': '统计分析',
            'content': self._generate_default_statistical_analysis(data),
            'importance': 4,
            'required': True,
            'source': 'rule_based'
        })
        
        # 添加洞察与建议
        sections.append({
            'id': 'insights_and_recommendations',
            'title': '洞察与建议',
            'content': self._generate_default_insights_and_recommendations(data),
            'importance': 5,
            'required': True,
            'source': 'rule_based'
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
    
    def apply_formatting_rules(self, content: str, formatting_rules: Dict[str, Any] = None) -> str:
        """
        应用格式规则到内容
        
        Args:
            content: 要格式化的内容
            formatting_rules: 格式规则
            
        Returns:
            格式化后的内容
        """
        if not formatting_rules:
            formatting_rules = {
                'max_line_length': 80,
                'heading_style': 'markdown',
                'list_style': 'markdown',
                'spacing': 2
            }
        
        formatted_content = content
        
        # 应用行长度限制
        if 'max_line_length' in formatting_rules:
            max_length = formatting_rules['max_line_length']
            lines = formatted_content.split('\n')
            wrapped_lines = []
            for line in lines:
                if len(line) > max_length:
                    # 简单的行包装
                    words = line.split()
                    current_line = ""
                    for word in words:
                        if len(current_line) + len(word) + 1 <= max_length:
                            current_line += f" {word}" if current_line else word
                        else:
                            wrapped_lines.append(current_line)
                            current_line = word
                    if current_line:
                        wrapped_lines.append(current_line)
                else:
                    wrapped_lines.append(line)
            formatted_content = '\n'.join(wrapped_lines)
        
        return formatted_content
