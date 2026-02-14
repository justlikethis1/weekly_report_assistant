#!/usr/bin/env python3
"""
配置管理模块
负责加载和管理报告生成所需的各种配置文件
"""

import yaml
import os
from typing import Dict, Any, Optional


class ConfigManager:
    """配置管理器，负责加载和管理所有配置文件"""
    
    def __init__(self, config_dir: str = None):
        """
        初始化配置管理器
        
        Args:
            config_dir: 配置文件目录，默认为项目根目录下的configs
        """
        if config_dir is None:
            self.config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs')
        else:
            self.config_dir = config_dir
        
        self.report_configs: Dict[str, Dict[str, Any]] = {}
        self.analysis_pipelines: Dict[str, Any] = {}
        self.template_configs: Dict[str, Any] = {}
        
        # 加载所有配置
        self._load_all_configs()
    
    def _load_all_configs(self):
        """加载所有配置文件"""
        # 加载报告配置
        self._load_report_configs()
        
        # 加载分析流水线配置
        self._load_analysis_pipelines()
        
        # 加载模板配置
        self._load_template_configs()
    
    def _load_report_configs(self):
        """加载报告配置文件"""
        report_configs_dir = os.path.join(self.config_dir, 'report_configs')
        if not os.path.exists(report_configs_dir):
            print(f"警告: 报告配置目录 {report_configs_dir} 不存在")
            return
        
        for filename in os.listdir(report_configs_dir):
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                config_path = os.path.join(report_configs_dir, filename)
                report_type = os.path.splitext(filename)[0]
                
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        self.report_configs[report_type] = yaml.safe_load(f)
                except Exception as e:
                    print(f"警告: 无法加载报告配置文件 {filename}: {e}")
    
    def _load_analysis_pipelines(self):
        """加载分析流水线配置"""
        pipelines_config_path = os.path.join(self.config_dir, 'analysis_pipelines.yaml')
        
        if os.path.exists(pipelines_config_path):
            try:
                with open(pipelines_config_path, 'r', encoding='utf-8') as f:
                    self.analysis_pipelines = yaml.safe_load(f)
            except Exception as e:
                print(f"警告: 无法加载分析流水线配置文件: {e}")
        else:
            # 使用默认分析流水线
            self.analysis_pipelines = {
                "financial": [
                    {"name": "price_trend_analysis", "weight": 0.3},
                    {"name": "correlation_analysis", "weight": 0.25},
                    {"name": "risk_assessment", "weight": 0.2},
                    {"name": "market_sentiment", "weight": 0.15},
                    {"name": "comparative_analysis", "weight": 0.1}
                ],
                "operational": [
                    {"name": "kpi_tracking", "weight": 0.4},
                    {"name": "issue_diagnosis", "weight": 0.3},
                    {"name": "improvement_suggestions", "weight": 0.3}
                ]
            }
    
    def _load_template_configs(self):
        """加载模板配置"""
        templates_config_path = os.path.join(self.config_dir, 'templates.yaml')
        
        if os.path.exists(templates_config_path):
            try:
                with open(templates_config_path, 'r', encoding='utf-8') as f:
                    self.template_configs = yaml.safe_load(f)
            except Exception as e:
                print(f"警告: 无法加载模板配置文件: {e}")
        else:
            # 使用默认模板配置
            self.template_configs = {
                "default_template_engine": "jinja2",
                "template_dir": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'templates'),
                "default_report_type": "general"
            }
    
    def get_report_config(self, report_type: str) -> Optional[Dict[str, Any]]:
        """
        获取指定报告类型的配置
        
        Args:
            report_type: 报告类型
            
        Returns:
            报告配置字典，如果不存在则返回None
        """
        return self.report_configs.get(report_type)
    
    def get_analysis_pipeline(self, pipeline_type: str) -> Optional[Dict[str, Any]]:
        """
        获取指定类型的分析流水线
        
        Args:
            pipeline_type: 流水线类型
            
        Returns:
            分析流水线配置，如果不存在则返回None
        """
        return self.analysis_pipelines.get(pipeline_type)
    
    def get_template_config(self, key: str = None) -> Any:
        """
        获取模板配置
        
        Args:
            key: 配置键，如果为None则返回所有模板配置
            
        Returns:
            模板配置值或所有配置
        """
        if key is None:
            return self.template_configs
        return self.template_configs.get(key)
    
    def get_default_report_type(self) -> str:
        """
        获取默认报告类型
        
        Returns:
            默认报告类型
        """
        return self.template_configs.get('default_report_type', 'general')
    
    def get_template_dir(self) -> str:
        """
        获取模板目录
        
        Returns:
            模板目录路径
        """
        return self.template_configs.get('template_dir')
