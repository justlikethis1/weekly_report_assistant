#!/usr/bin/env python3
"""
模板引擎层
负责加载模板、映射字段和渲染内容
"""

# 导入所有模块
from typing import Dict, Any, List, Optional
import os
import jinja2
import logging

# 导入增强模块
from .enhanced_template_selector import EnhancedTemplateSelector
from .enhanced_content_renderer import EnhancedContentRenderer
from .enhanced_format_converter import EnhancedFormatConverter

logger = logging.getLogger(__name__)

class TemplateLoader:
    """模板加载器，负责加载模板文件"""
    
    def __init__(self, template_dir: str = None):
        """
        初始化模板加载器
        
        Args:
            template_dir: 模板文件目录
        """
        self.template_dir = template_dir
        self.jinja_env = None
        
        if template_dir:
            self._setup_jinja_environment()
    
    def _setup_jinja_environment(self):
        """设置Jinja2环境"""
        try:
            self.jinja_env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(self.template_dir),
                autoescape=True,
                trim_blocks=True,
                lstrip_blocks=True
            )
        except Exception as e:
            logger.error(f"设置Jinja2环境失败: {e}")
            self.jinja_env = None
    
    def set_template_dir(self, template_dir: str):
        """
        设置模板目录
        
        Args:
            template_dir: 模板文件目录
        """
        self.template_dir = template_dir
        self._setup_jinja_environment()
    
    def load_template(self, template_path: str) -> Optional[str]:
        """
        加载模板文件
        
        Args:
            template_path: 模板文件路径
            
        Returns:
            模板内容，如果加载失败返回None
        """
        try:
            if self.jinja_env:
                # 如果使用Jinja2环境，使用get_template
                template = self.jinja_env.get_template(template_path)
                return template.source
            else:
                # 否则直接读取文件
                full_path = os.path.join(self.template_dir, template_path)
                if not os.path.exists(full_path):
                    logger.error(f"模板文件不存在: {full_path}")
                    return None
                
                with open(full_path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"加载模板 {template_path} 失败: {e}")
            return None
    
    def get_template_list(self, report_type: str) -> List[str]:
        """
        获取指定报告类型的所有模板
        
        Args:
            report_type: 报告类型
            
        Returns:
            模板文件路径列表
        """
        templates = []
        if not self.template_dir or not os.path.exists(self.template_dir):
            return templates
        
        report_template_dir = os.path.join(self.template_dir, report_type)
        if not os.path.exists(report_template_dir):
            return templates
        
        # 遍历报告类型目录下的所有模板文件
        for root, _, files in os.walk(report_template_dir):
            for file in files:
                if file.endswith('.md') or file.endswith('.html'):
                    # 计算相对路径
                    rel_path = os.path.relpath(os.path.join(root, file), self.template_dir)
                    templates.append(rel_path)
        
        return templates


class FieldMapper:
    """字段映射器，负责将数据字段映射到模板变量"""
    
    def __init__(self):
        """初始化字段映射器"""
        self.field_mappings = {
            'price_data': {
                'start_price': 'start_price',
                'end_price': 'end_price',
                'high_price': 'high_price',
                'low_price': 'low_price',
                'unit': 'unit',
                'price_changes': 'price_changes'
            },
            'events': {
                'name': 'name',
                'date': 'date',
                'description': 'description',
                'impact': 'impact'
            }
        }
    
    def map_fields(self, data: Dict[str, Any], template_type: str = None) -> Dict[str, Any]:
        """
        将数据字段映射到模板变量
        
        Args:
            data: 原始数据
            template_type: 模板类型
            
        Returns:
            映射后的数据
        """
        mapped_data = {}
        
        # 映射价格数据
        if 'price_data' in data:
            mapped_data['price_data'] = self._map_price_data(data['price_data'])
        
        # 映射事件数据
        if 'events' in data:
            mapped_data['events'] = self._map_events(data['events'])
        
        # 映射市场情绪数据
        if 'market_sentiment' in data:
            mapped_data['market_sentiment'] = data['market_sentiment']
        
        # 映射统计分析数据
        if 'statistical_analysis' in data:
            mapped_data['statistical_analysis'] = data['statistical_analysis']
        
        # 添加通用数据
        mapped_data['report_date'] = self._get_current_date()
        
        return mapped_data
    
    def _map_price_data(self, price_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        映射价格数据
        
        Args:
            price_data: 原始价格数据
            
        Returns:
            映射后的价格数据
        """
        mapped = {}
        for source_field, target_field in self.field_mappings['price_data'].items():
            if source_field in price_data:
                mapped[target_field] = price_data[source_field]
        
        # 添加计算字段
        if 'start_price' in mapped and 'end_price' in mapped:
            mapped['price_change'] = mapped['end_price'] - mapped['start_price']
            # 避免除零错误
            if mapped['start_price'] != 0:
                mapped['price_change_percentage'] = ((mapped['end_price'] - mapped['start_price']) / mapped['start_price']) * 100
            else:
                mapped['price_change_percentage'] = 0.0
        
        return mapped
    
    def _map_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        映射事件数据
        
        Args:
            events: 原始事件数据
            
        Returns:
            映射后的事件数据
        """
        mapped_events = []
        for event in events:
            mapped_event = {}
            for source_field, target_field in self.field_mappings['events'].items():
                if source_field in event:
                    mapped_event[target_field] = event[source_field]
            
            if mapped_event:
                mapped_events.append(mapped_event)
        
        return mapped_events
    
    def _get_current_date(self) -> str:
        """
        获取当前日期
        
        Returns:
            当前日期字符串
        """
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d')
    
    def add_custom_mapping(self, data_type: str, source_field: str, target_field: str):
        """
        添加自定义字段映射
        
        Args:
            data_type: 数据类型
            source_field: 源字段名
            target_field: 目标字段名
        """
        if data_type not in self.field_mappings:
            self.field_mappings[data_type] = {}
        self.field_mappings[data_type][source_field] = target_field


class ContentRenderer:
    """内容渲染器，负责渲染模板内容"""
    
    def __init__(self, template_loader: TemplateLoader = None):
        """
        初始化内容渲染器
        
        Args:
            template_loader: 模板加载器
        """
        self.template_loader = template_loader
        self.jinja_env = None
        
        if template_loader and template_loader.jinja_env:
            self.jinja_env = template_loader.jinja_env
    
    def set_template_loader(self, template_loader: TemplateLoader):
        """
        设置模板加载器
        
        Args:
            template_loader: 模板加载器
        """
        self.template_loader = template_loader
        if template_loader and template_loader.jinja_env:
            self.jinja_env = template_loader.jinja_env
    
    def render(self, template_path: str, data: Dict[str, Any]) -> Optional[str]:
        """
        使用Jinja2渲染模板
        
        Args:
            template_path: 模板文件路径
            data: 渲染数据
            
        Returns:
            渲染后的内容，如果渲染失败返回None
        """
        try:
            if self.jinja_env:
                # 使用Jinja2环境渲染
                template = self.jinja_env.get_template(template_path)
                return template.render(**data)
            elif self.template_loader:
                # 如果没有Jinja2环境，直接使用字符串替换（简单模式）
                template_content = self.template_loader.load_template(template_path)
                if not template_content:
                    return None
                
                # 简单的字符串替换
                rendered_content = template_content
                for key, value in data.items():
                    placeholder = f"{{{{ {key} }}}}"
                    rendered_content = rendered_content.replace(placeholder, str(value))
                
                return rendered_content
        except Exception as e:
            logger.error(f"渲染模板 {template_path} 失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def render_string(self, template_string: str, data: Dict[str, Any]) -> Optional[str]:
        """
        使用Jinja2渲染字符串模板
        
        Args:
            template_string: 模板字符串
            data: 渲染数据
            
        Returns:
            渲染后的内容，如果渲染失败返回None
        """
        try:
            if self.jinja_env:
                template = self.jinja_env.from_string(template_string)
                return template.render(**data)
            else:
                # 简单的字符串替换
                rendered_content = template_string
                for key, value in data.items():
                    placeholder = f"{{{{ {key} }}}}"
                    rendered_content = rendered_content.replace(placeholder, str(value))
                
                return rendered_content
        except Exception as e:
            logger.error(f"渲染模板字符串失败: {e}")
            return None


class ReportTemplate:
    """报告模板系统，根据数据质量选择合适的模板"""
    
    def __init__(self):
        """初始化报告模板系统"""
        self.templates = {
            'basic': {
                'sections': ['执行摘要', '数据概览', '关键发现', '建议'],
                'conditional_sections': {
                    'has_statistics': '统计分析',
                    'has_trends': '趋势分析'
                }
            },
            'detailed': {
                'sections': ['执行摘要', '数据概览', '深度分析', '市场对比', '风险分析', '建议'],
                'requirements': {'min_data_points': 10, 'data_quality_score': 7},
                'conditional_sections': {
                    'has_statistics': '统计分析',
                    'has_trends': '趋势分析'
                }
            }
        }
    
    def select_template(self, data_quality: Dict[str, Any], data_volume: int) -> str:
        """
        根据数据质量选择合适的模板
        
        Args:
            data_quality: 数据质量信息，包含score和issues
            data_volume: 数据点数量
            
        Returns:
            选择的模板类型
        """
        # 获取数据质量得分
        data_quality_score = data_quality.get('score', 0)
        
        # 检查是否满足详细报告的要求
        detailed_requirements = self.templates['detailed']['requirements']
        if (data_quality_score >= detailed_requirements['data_quality_score'] and 
            data_volume >= detailed_requirements['min_data_points']):
            return 'detailed'
        else:
            return 'basic'
    
    def get_template_sections(self, template_type: str, data: Dict[str, Any]) -> List[str]:
        """
        获取模板的章节列表
        
        Args:
            template_type: 模板类型
            data: 报告数据
            
        Returns:
            章节列表
        """
        if template_type not in self.templates:
            template_type = 'basic'  # 默认使用基础模板
        
        sections = self.templates[template_type]['sections'].copy()
        
        # 添加条件章节
        conditional_sections = self.templates[template_type].get('conditional_sections', {})
        
        if 'has_statistics' in conditional_sections and 'statistical_analysis' in data:
            sections.append(conditional_sections['has_statistics'])
        
        if 'has_trends' in conditional_sections and 'insights' in data and data['insights']:
            sections.append(conditional_sections['has_trends'])
        
        return sections
    
    def calculate_data_volume(self, data: Dict[str, Any]) -> int:
        """
        计算数据量
        
        Args:
            data: 报告数据
            
        Returns:
            数据点数量
        """
        data_volume = 0
        
        if 'price_data' in data and 'price_changes' in data['price_data']:
            data_volume = len(data['price_data']['price_changes'])
        
        return data_volume
