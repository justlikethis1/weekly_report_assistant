#!/usr/bin/env python3
"""
周报生成助手模型层
包含所有报告生成相关的模块
"""

# 导出主要类以便外部使用
from .config_manager import ConfigManager
from .data_extractor import DataExtractor, FileParser, DataValidator
from .template_engine import TemplateLoader, FieldMapper, ContentRenderer
from .analysis_pipeline import StatisticalAnalyzer, CorrelationCalculator, InsightGenerator
from .report_builder import SectionComposer, LLMIntegrator, FormatAdapter
from .quality_assurance import QualityAssurance, DataConsistencyChecker, StructuralValidator, StyleEnforcer
from .report_orchestrator import ReportOrchestrator
from .enhanced_analyzer import EnhancedAnalyzer

__all__ = [
    # 配置管理
    'ConfigManager',
    
    # 数据提取
    'DataExtractor',
    'FileParser',
    'DataValidator',
    
    # 模板引擎
    'TemplateLoader',
    'FieldMapper',
    'ContentRenderer',
    
    # 分析流水线
    'StatisticalAnalyzer',
    'CorrelationCalculator',
    'InsightGenerator',
    
    # 报告构建
    'SectionComposer',
    'LLMIntegrator',
    'FormatAdapter',
    
    # 质量保证
    'QualityAssurance',
    'DataConsistencyChecker',
    'StructuralValidator',
    'StyleEnforcer',
    
    # 增强分析
    'EnhancedAnalyzer',
    
    # 协调器
    'ReportOrchestrator'
]

