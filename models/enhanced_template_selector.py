#!/usr/bin/env python3
"""
增强的模板选择器模块
负责智能选择模板，支持自定义规则和缓存机制
"""

import os
import hashlib
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class EnhancedTemplateSelector:
    """增强的模板选择器，负责智能选择模板"""
    
    def __init__(self, template_dir: str = None):
        """
        初始化增强的模板选择器
        
        Args:
            template_dir: 模板目录
        """
        self.template_dir = template_dir
        self.templates = {
            'basic': {
                'name': '基础模板',
                'sections': ['executive_summary', 'data_overview', 'key_findings', 'recommendations'],
                'priority': 10,
                'conditional_sections': {
                    'has_statistics': 'statistical_analysis',
                    'has_trends': 'trend_analysis'
                },
                'requirements': {'min_data_points': 0, 'data_quality_score': 0}
            },
            'standard': {
                'name': '标准模板',
                'sections': ['executive_summary', 'data_overview', 'statistical_analysis', 'key_findings', 'recommendations'],
                'priority': 20,
                'conditional_sections': {
                    'has_trends': 'trend_analysis',
                    'has_comparison': 'market_comparison'
                },
                'requirements': {'min_data_points': 5, 'data_quality_score': 5}
            },
            'detailed': {
                'name': '详细模板',
                'sections': ['executive_summary', 'data_overview', 'statistical_analysis', 'trend_analysis', 'market_comparison', 'risk_analysis', 'recommendations'],
                'priority': 30,
                'conditional_sections': {
                    'has_correlations': 'correlation_analysis',
                    'has_insights': 'advanced_insights'
                },
                'requirements': {'min_data_points': 15, 'data_quality_score': 8}
            },
            'professional': {
                'name': '专业模板',
                'sections': ['executive_summary', 'data_overview', 'statistical_analysis', 'trend_analysis', 'correlation_analysis', 'market_comparison', 'risk_analysis', 'competitor_analysis', 'opportunity_analysis', 'recommendations', 'action_plan'],
                'priority': 40,
                'conditional_sections': {},
                'requirements': {'min_data_points': 20, 'data_quality_score': 9},
                'special_requirements': {'has_all_data_types': True}
            }
        }
        
        # 模板缓存
        self.template_cache = {
            'content_cache': {},  # 模板内容缓存
            'selection_cache': {}  # 模板选择结果缓存
        }
        
        # 自定义选择规则
        self.custom_rules = []
        
        logger.info("EnhancedTemplateSelector初始化完成")
    
    def add_custom_template(self, template_type: str, template_config: Dict[str, Any]):
        """
        添加自定义模板
        
        Args:
            template_type: 模板类型
            template_config: 模板配置
        """
        if template_type in self.templates:
            logger.warning(f"模板类型 {template_type} 已存在，将被覆盖")
        
        # 设置默认值
        template_config.setdefault('priority', 10)
        template_config.setdefault('sections', [])
        template_config.setdefault('conditional_sections', {})
        template_config.setdefault('requirements', {'min_data_points': 0, 'data_quality_score': 0})
        
        self.templates[template_type] = template_config
        logger.info(f"已添加自定义模板: {template_type}")
    
    def add_custom_selection_rule(self, rule: callable):
        """
        添加自定义模板选择规则
        
        Args:
            rule: 自定义规则函数，接收(data_quality, data_volume, data_content)参数，返回可选的模板类型
        """
        self.custom_rules.append(rule)
        logger.info("已添加自定义模板选择规则")
    
    def select_template(self, data_quality: Dict[str, Any], data_volume: int, data_content: Dict[str, Any]) -> str:
        """
        根据数据质量、数据量和数据内容智能选择模板
        
        Args:
            data_quality: 数据质量信息
            data_volume: 数据点数量
            data_content: 数据内容
            
        Returns:
            选择的模板类型
        """
        # 检查缓存
        cache_key = self._generate_cache_key(data_quality, data_volume, data_content)
        if cache_key in self.template_cache['selection_cache']:
            cached_result = self.template_cache['selection_cache'][cache_key]
            logger.debug(f"使用缓存的模板选择结果: {cached_result}")
            return cached_result
        
        # 应用自定义规则
        for rule in self.custom_rules:
            try:
                custom_template = rule(data_quality, data_volume, data_content)
                if custom_template and custom_template in self.templates:
                    logger.info(f"应用自定义规则选择模板: {custom_template}")
                    self.template_cache['selection_cache'][cache_key] = custom_template
                    return custom_template
            except Exception as e:
                logger.error(f"执行自定义规则失败: {e}")
        
        # 获取数据质量得分
        data_quality_score = data_quality.get('score', 0)
        
        # 收集符合条件的模板
        eligible_templates = []
        for template_type, template_info in self.templates.items():
            if self._check_template_requirements(template_info, data_quality_score, data_volume, data_content):
                eligible_templates.append((template_type, template_info))
        
        # 根据优先级排序
        eligible_templates.sort(key=lambda x: x[1]['priority'], reverse=True)
        
        # 选择优先级最高的模板
        if eligible_templates:
            selected_template = eligible_templates[0][0]
            logger.info(f"选择模板: {selected_template} (优先级: {eligible_templates[0][1]['priority']})")
            self.template_cache['selection_cache'][cache_key] = selected_template
            return selected_template
        else:
            logger.warning("没有符合条件的模板，使用默认模板")
            self.template_cache['selection_cache'][cache_key] = 'basic'
            return 'basic'
    
    def _check_template_requirements(self, template_info: Dict[str, Any], data_quality_score: int, data_volume: int, data_content: Dict[str, Any]) -> bool:
        """
        检查模板是否满足要求
        
        Args:
            template_info: 模板信息
            data_quality_score: 数据质量得分
            data_volume: 数据点数量
            data_content: 数据内容
            
        Returns:
            如果满足要求返回True，否则返回False
        """
        # 检查基本要求
        requirements = template_info.get('requirements', {})
        min_data_points = requirements.get('min_data_points', 0)
        min_quality_score = requirements.get('data_quality_score', 0)
        
        if data_quality_score < min_quality_score or data_volume < min_data_points:
            return False
        
        # 检查特殊要求
        special_requirements = template_info.get('special_requirements', {})
        
        # 检查是否包含所有数据类型
        if 'has_all_data_types' in special_requirements and special_requirements['has_all_data_types']:
            required_data_types = ['price_data', 'statistical_analysis', 'insights']
            for data_type in required_data_types:
                if data_type not in data_content or not data_content[data_type]:
                    return False
        
        # 检查数据完整性要求
        if 'required_fields' in special_requirements:
            for data_type, fields in special_requirements['required_fields'].items():
                if data_type in data_content:
                    for field in fields:
                        if field not in data_content[data_type]:
                            return False
        
        # 检查数据量要求
        if 'data_type_volumes' in special_requirements:
            for data_type, min_volume in special_requirements['data_type_volumes'].items():
                if data_type in data_content:
                    if isinstance(data_content[data_type], list) and len(data_content[data_type]) < min_volume:
                        return False
                    elif isinstance(data_content[data_type], dict) and data_type == 'price_data' and 'price_changes' in data_content[data_type]:
                        if len(data_content[data_type]['price_changes']) < min_volume:
                            return False
        
        return True
    
    def get_template_sections(self, template_type: str, data_content: Dict[str, Any]) -> List[str]:
        """
        获取模板的章节列表
        
        Args:
            template_type: 模板类型
            data_content: 数据内容
            
        Returns:
            章节列表
        """
        if template_type not in self.templates:
            logger.warning(f"模板类型 {template_type} 不存在，使用基础模板")
            template_type = 'basic'
        
        template_info = self.templates[template_type]
        sections = template_info['sections'].copy()
        
        # 添加条件章节
        conditional_sections = template_info.get('conditional_sections', {})
        
        for condition, section_name in conditional_sections.items():
            if self._check_section_condition(condition, data_content):
                sections.append(section_name)
                logger.debug(f"满足条件 {condition}，添加章节: {section_name}")
        
        return sections
    
    def _check_section_condition(self, condition: str, data_content: Dict[str, Any]) -> bool:
        """
        检查是否满足章节条件
        
        Args:
            condition: 条件名称
            data_content: 数据内容
            
        Returns:
            如果满足条件返回True，否则返回False
        """
        if condition == 'has_statistics':
            return 'statistical_analysis' in data_content and data_content['statistical_analysis']
        elif condition == 'has_trends':
            return 'statistical_analysis' in data_content and 'trend' in data_content['statistical_analysis']
        elif condition == 'has_correlations':
            return 'correlations' in data_content and data_content['correlations']
        elif condition == 'has_comparison':
            return 'market_sentiment' in data_content and data_content['market_sentiment']
        elif condition == 'has_insights':
            return 'insights' in data_content and data_content['insights']
        elif condition == 'has_events':
            return 'events' in data_content and data_content['events']
        else:
            logger.warning(f"未知的条件: {condition}")
            return False
    
    def calculate_data_quality_score(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算数据质量得分
        
        Args:
            data: 数据内容
            
        Returns:
            数据质量信息，包含score和issues
        """
        score = 10  # 基础分
        issues = []
        
        # 检查数据完整性
        if not data or not isinstance(data, dict):
            score -= 5
            issues.append("数据为空或格式错误")
        else:
            # 检查价格数据
            if 'price_data' not in data or not data['price_data']:
                score -= 2
                issues.append("缺少价格数据")
            else:
                price_data = data['price_data']
                required_price_fields = ['start_price', 'end_price', 'high_price', 'low_price', 'price_changes']
                for field in required_price_fields:
                    if field not in price_data:
                        score -= 1
                        issues.append(f"价格数据缺少字段: {field}")
                
                # 检查价格变动数据量
                if 'price_changes' in price_data and len(price_data['price_changes']) < 2:
                    score -= 1
                    issues.append("价格变动数据量不足")
            
            # 检查统计分析数据
            if 'statistical_analysis' not in data or not data['statistical_analysis']:
                score -= 1
                issues.append("缺少统计分析数据")
            
            # 检查洞察数据
            if 'insights' not in data or not data['insights']:
                score -= 1
                issues.append("缺少洞察数据")
        
        # 确保分数在0-10之间
        score = max(0, min(10, score))
        
        return {
            'score': score,
            'issues': issues,
            'timestamp': datetime.now().isoformat()
        }
    
    def calculate_data_volume(self, data: Dict[str, Any]) -> int:
        """
        计算数据量
        
        Args:
            data: 数据内容
            
        Returns:
            数据点数量
        """
        data_volume = 0
        
        # 计算价格变动数据量
        if 'price_data' in data and 'price_changes' in data['price_data']:
            data_volume += len(data['price_data']['price_changes'])
        
        # 计算事件数据量
        if 'events' in data:
            data_volume += len(data['events'])
        
        # 计算洞察数据量
        if 'insights' in data:
            data_volume += len(data['insights'])
        
        # 计算统计分析数据量
        if 'statistical_analysis' in data:
            for key, value in data['statistical_analysis'].items():
                if isinstance(value, dict):
                    data_volume += len(value)
                elif isinstance(value, list):
                    data_volume += len(value)
                else:
                    data_volume += 1
        
        return data_volume
    
    def _generate_cache_key(self, data_quality: Dict[str, Any], data_volume: int, data_content: Dict[str, Any]) -> str:
        """
        生成缓存键
        
        Args:
            data_quality: 数据质量信息
            data_volume: 数据点数量
            data_content: 数据内容
            
        Returns:
            缓存键
        """
        # 创建数据摘要
        data_digest = {
            'quality_score': data_quality.get('score', 0),
            'volume': data_volume,
            'has_price_data': 'price_data' in data_content and bool(data_content['price_data']),
            'has_statistics': 'statistical_analysis' in data_content and bool(data_content['statistical_analysis']),
            'has_insights': 'insights' in data_content and bool(data_content['insights']),
            'has_events': 'events' in data_content and bool(data_content['events'])
        }
        
        # 转换为字符串并生成MD5哈希
        data_str = str(data_digest)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def clear_cache(self, cache_type: str = None):
        """
        清除缓存
        
        Args:
            cache_type: 缓存类型（'content'或'selection'），如果为None则清除所有缓存
        """
        if cache_type == 'content':
            self.template_cache['content_cache'] = {}
            logger.info("已清除模板内容缓存")
        elif cache_type == 'selection':
            self.template_cache['selection_cache'] = {}
            logger.info("已清除模板选择结果缓存")
        else:
            self.template_cache['content_cache'] = {}
            self.template_cache['selection_cache'] = {}
            logger.info("已清除所有模板缓存")
    
    def get_template_info(self, template_type: str) -> Optional[Dict[str, Any]]:
        """
        获取模板信息
        
        Args:
            template_type: 模板类型
            
        Returns:
            模板信息，如果不存在返回None
        """
        return self.templates.get(template_type)
    
    def list_templates(self) -> List[str]:
        """
        列出所有可用的模板类型
        
        Returns:
            模板类型列表
        """
        return list(self.templates.keys())
    
    def export_template_config(self, template_type: str) -> Optional[Dict[str, Any]]:
        """
        导出模板配置
        
        Args:
            template_type: 模板类型
            
        Returns:
            模板配置，如果不存在返回None
        """
        template_info = self.templates.get(template_type)
        if not template_info:
            return None
        
        # 创建可导出的配置
        export_config = {
            'template_type': template_type,
            'name': template_info.get('name', template_type),
            'sections': template_info.get('sections', []),
            'conditional_sections': template_info.get('conditional_sections', {}),
            'requirements': template_info.get('requirements', {}),
            'special_requirements': template_info.get('special_requirements', {})
        }
        
        return export_config
