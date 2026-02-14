#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
报告模板管理器，用于智能选择和管理报告模板
"""

from typing import Dict, List, Any, Optional


class ReportTemplate:
    """
    报告模板类，支持多语言、多风格、多详细程度的模板
    """
    
    def __init__(self, name: str, chinese: bool = True):
        self.name = name
        self.chinese = chinese
        self.sections = self._generate_sections()
    
    def _generate_sections(self) -> List[Dict[str, Any]]:
        """
        根据语言和模板类型生成章节结构
        
        Returns:
            List[Dict]: 章节结构列表
        """
        if self.chinese:
            return [
                {
                    "type": "section",
                    "title": "执行摘要",
                    "subsections": [],
                    "required": True,
                    "min_length": 300,
                    "purpose": "提供报告的核心观点和结论概述"
                },
                {
                    "type": "section",
                    "title": "背景与目标",
                    "subsections": [],
                    "required": True,
                    "min_length": 200,
                    "purpose": "说明报告的背景、目的和预期目标"
                },
                {
                    "type": "section",
                    "title": "方法论",
                    "subsections": [],
                    "required": True,
                    "min_length": 200,
                    "purpose": "描述数据收集和分析的方法"
                },
                {
                    "type": "section",
                    "title": "关键指标分析",
                    "subsections": [],
                    "required": True,
                    "min_length": 300,
                    "purpose": "分析核心业务指标的表现和趋势"
                },
                {
                    "type": "section",
                    "title": "深度业务洞察",
                    "subsections": [],
                    "required": True,
                    "min_length": 400,
                    "purpose": "基于数据提供深入的业务见解"
                },
                {
                    "type": "section",
                    "title": "竞争与市场分析",
                    "subsections": [],
                    "required": False,
                    "min_length": 300,
                    "purpose": "分析市场竞争格局和行业趋势"
                },
                {
                    "type": "section",
                    "title": "风险评估",
                    "subsections": [],
                    "required": False,
                    "min_length": 200,
                    "purpose": "识别潜在风险和挑战"
                },
                {
                    "type": "section",
                    "title": "未来展望",
                    "subsections": [],
                    "required": True,
                    "min_length": 200,
                    "purpose": "展望未来发展趋势"
                },
                {
                    "type": "section",
                    "title": "战略建议",
                    "subsections": [],
                    "required": True,
                    "min_length": 300,
                    "purpose": "提出针对性的战略建议"
                },
                {
                    "type": "section",
                    "title": "行动计划",
                    "subsections": [],
                    "required": True,
                    "min_length": 300,
                    "purpose": "制定具体的实施计划和时间线"
                },
                {
                    "type": "appendix",
                    "title": "附录",
                    "subsections": [],
                    "required": False,
                    "min_length": 0,
                    "purpose": "提供补充数据和资料"
                }
            ]
        else:
            return [
                {
                    "type": "section",
                    "title": "Executive Summary",
                    "subsections": [],
                    "required": True,
                    "min_length": 300,
                    "purpose": "Provide an overview of key findings and conclusions"
                },
                {
                    "type": "section",
                    "title": "Background & Objectives",
                    "subsections": [],
                    "required": True,
                    "min_length": 200,
                    "purpose": "Explain the background, purpose, and objectives of the report"
                },
                {
                    "type": "section",
                    "title": "Methodology",
                    "subsections": [],
                    "required": True,
                    "min_length": 200,
                    "purpose": "Describe data collection and analysis methods"
                },
                {
                    "type": "section",
                    "title": "Key Metrics Analysis",
                    "subsections": [],
                    "required": True,
                    "min_length": 300,
                    "purpose": "Analyze core business metrics performance and trends"
                },
                {
                    "type": "section",
                    "title": "In-depth Business Insights",
                    "subsections": [],
                    "required": True,
                    "min_length": 400,
                    "purpose": "Provide deep business insights based on data"
                },
                {
                    "type": "section",
                    "title": "Competitive & Market Analysis",
                    "subsections": [],
                    "required": False,
                    "min_length": 300,
                    "purpose": "Analyze market competition and industry trends"
                },
                {
                    "type": "section",
                    "title": "Risk Assessment",
                    "subsections": [],
                    "required": False,
                    "min_length": 200,
                    "purpose": "Identify potential risks and challenges"
                },
                {
                    "type": "section",
                    "title": "Future Outlook",
                    "subsections": [],
                    "required": True,
                    "min_length": 200,
                    "purpose": "Provide future development trends"
                },
                {
                    "type": "section",
                    "title": "Strategic Recommendations",
                    "subsections": [],
                    "required": True,
                    "min_length": 300,
                    "purpose": "Propose targeted strategic recommendations"
                },
                {
                    "type": "section",
                    "title": "Action Plan",
                    "subsections": [],
                    "required": True,
                    "min_length": 300,
                    "purpose": "Develop specific implementation plans and timelines"
                },
                {
                    "type": "appendix",
                    "title": "Appendices",
                    "subsections": [],
                    "required": False,
                    "min_length": 0,
                    "purpose": "Provide supplementary data and information"
                }
            ]
    
    def customize_for_audience(self, audience: str) -> None:
        """
        根据受众类型定制模板
        
        Args:
            audience: 受众类型 (executive, technical, general, external)
        """
        # 根据受众调整章节的详细程度和重点
        for section in self.sections:
            if audience == "executive":
                # 管理层受众：简化方法论，强调结论和建议
                if section["title"] in ["方法论", "Methodology"]:
                    section["required"] = False
                    section["min_length"] = 100
                elif section["title"] in ["执行摘要", "Executive Summary", "战略建议", "Strategic Recommendations"]:
                    section["min_length"] = 400
            elif audience == "technical":
                # 技术受众：详细方法论，强调数据分析
                if section["title"] in ["方法论", "Methodology", "关键指标分析", "Key Metrics Analysis"]:
                    section["min_length"] = 500
            elif audience == "external":
                # 外部受众：重点竞争分析和未来展望
                if section["title"] in ["竞争与市场分析", "Competitive & Market Analysis", "未来展望", "Future Outlook"]:
                    section["required"] = True
                    section["min_length"] = 400
    
    def adjust_detail_level(self, detail_level: str) -> None:
        """
        根据详细程度调整模板
        
        Args:
            detail_level: 详细程度 (high, standard, low)
        """
        for section in self.sections:
            if detail_level == "high":
                section["min_length"] = int(section["min_length"] * 1.5)
                # 添加子章节
                if section["title"] in ["关键指标分析", "Key Metrics Analysis"]:
                    section["subsections"] = [
                        {
                            "title": "用户增长指标",
                            "required": True,
                            "min_length": 200
                        },
                        {
                            "title": "功能使用指标",
                            "required": True,
                            "min_length": 200
                        },
                        {
                            "title": "收入与成本指标",
                            "required": True,
                            "min_length": 200
                        }
                    ]
            elif detail_level == "low":
                section["min_length"] = int(section["min_length"] * 0.5)
                # 简化章节结构
                section["subsections"] = []
                # 标记非必需章节为可选
                if not section["required"]:
                    section["required"] = False
    
    def get(self, key: str) -> Any:
        """获取模板属性"""
        return getattr(self, key, None)
    
    def get_required_sections(self) -> List[Dict[str, Any]]:
        """
        获取必需的章节
        
        Returns:
            List[Dict]: 必需章节列表
        """
        return [section for section in self.sections if section["required"]]


class ReportTemplateManager:
    """
    报告模板管理器，用于智能选择和应用报告模板
    """
    
    def __init__(self):
        self.templates = {
            "default": {},
            "weekly": {},
            "monthly": {},
            "quarterly": {},
            "annual": {},
            "project": {},
            "sales": {}
        }
        self._initialize_templates()
    
    def _initialize_templates(self):
        """
        初始化所有模板
        """
        for template_type in self.templates.keys():
            self.templates[template_type]["chinese"] = ReportTemplate(template_type, chinese=True)
            self.templates[template_type]["english"] = ReportTemplate(template_type, chinese=False)
    
    def select_template(self, report_type: str = "default", 
                       audience: str = "general",
                       detail_level: str = "standard",
                       style: str = "formal",
                       chinese: bool = True) -> ReportTemplate:
        """
        智能选择最合适的模板
        
        Args:
            report_type: 报告类型 (default, weekly, monthly, quarterly, annual, project, sales)
            audience: 受众类型 (executive, technical, general, external)
            detail_level: 详细程度 (high, standard, low)
            style: 风格偏好 (formal, casual, technical, business)
            chinese: 是否为中文模板
            
        Returns:
            ReportTemplate: 定制化的模板实例
        """
        # 获取基础模板
        report_type = report_type if report_type in self.templates else "default"
        base_template = self.templates[report_type]["chinese" if chinese else "english"]
        
        # 创建模板副本以避免修改原始模板
        import copy
        custom_template = copy.deepcopy(base_template)
        
        # 根据受众和详细程度定制模板
        custom_template.customize_for_audience(audience)
        custom_template.adjust_detail_level(detail_level)
        
        return custom_template
    
    def apply_formatting(self, document, template: ReportTemplate, chinese: bool = True) -> None:
        """
        应用格式化设置到文档
        
        Args:
            document: Word文档对象
            template: 报告模板
            chinese: 是否为中文文档
        """
        # 设置页面边距
        for section in document.sections:
            section.left_margin = 2.54  # 2.54厘米 = 1英寸
            section.right_margin = 2.54
            section.top_margin = 2.54
            section.bottom_margin = 2.54
        
        # 设置字体
        from docx.shared import Pt, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        
        for paragraph in document.paragraphs:
            for run in paragraph.runs:
                if chinese:
                    run.font.name = '微软雅黑'
                    run.element.rPr.rFonts.set(run.element.rPr.rFonts.qn('w:eastAsia'), '微软雅黑')
                else:
                    run.font.name = 'Calibri'
                    run.element.rPr.rFonts.set(run.element.rPr.rFonts.qn('w:eastAsia'), 'Calibri')
                
                # 设置字体大小
                if paragraph.style.name.startswith('Heading 1'):
                    run.font.size = Pt(16)
                    run.font.bold = True
                    run.font.color.rgb = RGBColor(0, 51, 102)
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                elif paragraph.style.name.startswith('Heading 2'):
                    run.font.size = Pt(14)
                    run.font.bold = True
                    run.font.color.rgb = RGBColor(0, 51, 102)
                elif paragraph.style.name.startswith('Heading 3'):
                    run.font.size = Pt(12)
                    run.font.bold = True
                else:
                    run.font.size = Pt(11)
        
        # 设置段落间距
        for paragraph in document.paragraphs:
            if paragraph.style.name.startswith('Heading'):
                paragraph.space_after = Pt(12)
            else:
                paragraph.space_before = Pt(3)
                paragraph.space_after = Pt(3)