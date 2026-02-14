from typing import List, Dict, Any, Optional
from .advanced_report_generator import AdvancedReportGenerator
import re
import random
from datetime import datetime
import os
import pandas as pd

class EnhancedReportGenerator(AdvancedReportGenerator):
    """
    增强型报告生成器类，在AdvancedReportGenerator基础上扩展功能：
    1. 支持更多报告类型和主题
    2. 增强的分析框架和方法
    3. 更丰富的报告内容生成能力
    4. 支持多语言报告生成
    5. 增强的数据可视化和表格处理
    """
    
    def __init__(self):
        super().__init__()
        
        # 扩展报告模板
        self.report_templates.update({
            "weekly": "周报",
            "monthly": "月报",
            "quarterly": "季报",
            "annual": "年报",
            "project": "项目报告",
            "sales": "销售报告"
        })
        
        # 扩展主题属性库
        self.topic_attributes.update({
            "数字化转型": {
                "type": "企业战略,技术应用",
                "key_attributes": "技术架构,流程优化,组织变革,数据驱动,客户体验",
                "industry_chain": "战略规划,技术选型,实施落地,运营优化,持续创新",
                "key_metrics": "数字化成熟度,流程效率提升,成本降低,营收增长,客户满意度"
            },
            "供应链管理": {
                "type": "运营管理,物流领域",
                "key_attributes": "供应链韧性,成本控制,交付效率,风险管控,数字化水平",
                "industry_chain": "供应商管理,采购,生产,仓储,物流配送,客户服务",
                "key_metrics": "供应商绩效,库存周转率,准时交付率,供应链成本,风险事件数"
            },
            "客户体验": {
                "type": "客户关系,服务领域",
                "key_attributes": "用户旅程,满意度,忠诚度,服务质量,反馈管理",
                "industry_chain": "客户获取,客户转化,客户服务,客户留存,客户推荐",
                "key_metrics": "NPS得分,CSAT得分,客户流失率,服务响应时间,解决率"
            }
        })
        
        # 扩展分析框架库
        self.analysis_frameworks.update({
            "growth": ["增长模型分析", "客户生命周期价值", "市场渗透策略", "收入结构分析"],
            "cost": ["成本结构分析", "成本效益评估", "浪费识别", "成本优化策略"],
            "risk": ["风险矩阵分析", "风险影响评估", "风险缓解策略", "应急预案"],
            "innovation": ["创新漏斗分析", "技术路线图", "创新投资回报率", "创新文化评估"]
        })
        
        # 扩展具体表述库
        self.specific_expressions.update({
            "显著提升": ["同比提升{growth_rate}%", "环比增长{growth_rate}%", "提升幅度达{growth_rate}%"],
            "大幅降低": ["同比降低{reduction_rate}%", "环比减少{reduction_rate}%", "降低幅度达{reduction_rate}%"],
            "持续改善": ["连续{period}个月改善", "稳步提升{improvement_rate}%", "逐步优化至{target_level}"],
            "快速响应": ["响应时间缩短至{response_time}分钟", "处理效率提升{efficiency_rate}%", "服务水平提升至{service_level}%"]
        })
    
    def generate_report(self, parsed_input: Dict[str, Any], data_files: List[str] = None, is_mock: bool = False) -> str:
        """
        生成增强型报告
        
        Args:
            parsed_input: 解析后的用户输入
            data_files: 数据文件路径列表
            is_mock: 是否使用模拟数据
            
        Returns:
            生成的报告文本
        """
        # 调用父类方法生成基础报告
        basic_report = super().generate_report(parsed_input, data_files, is_mock)
        
        # 增强报告内容
        enhanced_report = self._enhance_report_content(basic_report, parsed_input, is_mock)
        
        return enhanced_report
    
    def _enhance_report_content(self, basic_report: str, parsed_input: Dict[str, Any], is_mock: bool) -> str:
        """
        增强报告内容
        
        Args:
            basic_report: 基础报告文本
            parsed_input: 解析后的用户输入
            is_mock: 是否使用模拟数据
            
        Returns:
            增强后的报告文本
        """
        # 根据报告类型添加特定内容
        report_type = parsed_input.get("report_type", "")
        
        if "周" in report_type:
            return self._add_weekly_specific_content(basic_report, parsed_input, is_mock)
        elif "月" in report_type:
            return self._add_monthly_specific_content(basic_report, parsed_input, is_mock)
        elif "季" in report_type:
            return self._add_quarterly_specific_content(basic_report, parsed_input, is_mock)
        elif "年" in report_type:
            return self._add_annual_specific_content(basic_report, parsed_input, is_mock)
        elif "项目" in report_type:
            return self._add_project_specific_content(basic_report, parsed_input, is_mock)
        elif "销售" in report_type:
            return self._add_sales_specific_content(basic_report, parsed_input, is_mock)
        else:
            return basic_report
    
    def _add_weekly_specific_content(self, basic_report: str, parsed_input: Dict[str, Any], is_mock: bool) -> str:
        """
        添加周报特定内容
        """
        # 添加本周重点工作回顾
        weekly_content = "# 0. 本周重点工作回顾\n\n"
        weekly_content += "## 0.1 主要完成工作\n\n"
        weekly_content += "- 完成了本周计划的核心任务\n"
        weekly_content += "- 推进了关键项目的实施进度\n"
        weekly_content += "- 解决了工作中的主要问题\n\n"
        
        # 添加下周工作计划
        weekly_content += "## 0.2 下周工作计划\n\n"
        weekly_content += "- 继续推进重点项目的实施\n"
        weekly_content += "- 完成预定的工作任务\n"
        weekly_content += "- 关注和解决可能出现的问题\n\n"
        
        return weekly_content + basic_report
    
    def _add_monthly_specific_content(self, basic_report: str, parsed_input: Dict[str, Any], is_mock: bool) -> str:
        """
        添加月报特定内容
        """
        monthly_content = "# 0. 月度工作概述\n\n"
        monthly_content += "## 0.1 月度目标完成情况\n\n"
        monthly_content += "- 总体目标完成率：XX%\n"
        monthly_content += "- 核心指标达成情况：XX\n"
        monthly_content += "- 重点任务完成情况：XX\n\n"
        
        monthly_content += "## 0.2 月度工作亮点\n\n"
        monthly_content += "- 工作中的创新点和突破\n"
        monthly_content += "- 取得的重要成果\n"
        monthly_content += "- 值得推广的经验\n\n"
        
        return monthly_content + basic_report
    
    def _add_quarterly_specific_content(self, basic_report: str, parsed_input: Dict[str, Any], is_mock: bool) -> str:
        """
        添加季报特定内容
        """
        quarterly_content = "# 0. 季度工作回顾\n\n"
        quarterly_content += "## 0.1 季度目标完成情况\n\n"
        quarterly_content += "- 季度KPI完成率：XX%\n"
        quarterly_content += "- 关键业绩指标达成情况：XX\n"
        quarterly_content += "- 重点项目进度：XX\n\n"
        
        quarterly_content += "## 0.2 季度工作总结\n\n"
        quarterly_content += "- 主要工作成果\n"
        quarterly_content += "- 存在的问题和挑战\n"
        quarterly_content += "- 改进措施和建议\n\n"
        
        return quarterly_content + basic_report
    
    def _add_annual_specific_content(self, basic_report: str, parsed_input: Dict[str, Any], is_mock: bool) -> str:
        """
        添加年报特定内容
        """
        annual_content = "# 0. 年度工作总结与展望\n\n"
        annual_content += "## 0.1 年度目标完成情况\n\n"
        annual_content += "- 年度总目标完成率：XX%\n"
        annual_content += "- 核心业务指标达成情况：XX\n"
        annual_content += "- 战略规划执行情况：XX\n\n"
        
        annual_content += "## 0.2 年度工作亮点\n\n"
        annual_content += "- 年度重大成果\n"
        annual_content += "- 战略转型进展\n"
        annual_content += "- 创新突破和亮点\n\n"
        
        annual_content += "## 0.3 下一年度展望\n\n"
        annual_content += "- 下一年度战略目标\n"
        annual_content += "- 重点工作方向\n"
        annual_content += "- 预期成果和挑战\n\n"
        
        return annual_content + basic_report
    
    def _add_project_specific_content(self, basic_report: str, parsed_input: Dict[str, Any], is_mock: bool) -> str:
        """
        添加项目报告特定内容
        """
        project_content = "# 0. 项目概述\n\n"
        project_content += "## 0.1 项目基本信息\n\n"
        project_content += "- 项目名称：XX\n"
        project_content += "- 项目周期：XX\n"
        project_content += "- 项目团队：XX\n\n"
        
        project_content += "## 0.2 项目目标和范围\n\n"
        project_content += "- 项目目标：XX\n"
        project_content += "- 项目范围：XX\n"
        project_content += "- 项目交付物：XX\n\n"
        
        return project_content + basic_report
    
    def _add_sales_specific_content(self, basic_report: str, parsed_input: Dict[str, Any], is_mock: bool) -> str:
        """
        添加销售报告特定内容
        """
        sales_content = "# 0. 销售情况概述\n\n"
        sales_content += "## 0.1 销售目标完成情况\n\n"
        sales_content += "- 销售目标：XX\n"
        sales_content += "- 实际完成：XX\n"
        sales_content += "- 完成率：XX%\n\n"
        
        sales_content += "## 0.2 销售数据分析\n\n"
        sales_content += "- 产品线销售情况\n"
        sales_content += "- 区域销售分布\n"
        sales_content += "- 客户类型分析\n\n"
        
        return sales_content + basic_report
    
    def _get_specific_expression(self, general_term: str, topic: str) -> str:
        """
        扩展的具体表述生成方法
        """
        # 先尝试使用扩展的表述
        if general_term in self.specific_expressions:
            expressions = self.specific_expressions[general_term]
            expression = random.choice(expressions)
            
            # 构建格式化参数字典
            format_args = {}
            
            # 根据主题填充具体内容
            if "{growth_rate}" in expression:
                format_args["growth_rate"] = random.randint(5, 30)
            if "{reduction_rate}" in expression:
                format_args["reduction_rate"] = random.randint(5, 40)
            if "{improvement_rate}" in expression:
                format_args["improvement_rate"] = random.randint(5, 25)
            if "{response_time}" in expression:
                format_args["response_time"] = random.randint(1, 60)
            if "{efficiency_rate}" in expression:
                format_args["efficiency_rate"] = random.randint(10, 50)
            if "{service_level}" in expression:
                format_args["service_level"] = random.randint(80, 100)
            if "{period}" in expression:
                format_args["period"] = random.randint(2, 12)
            if "{target_level}" in expression:
                format_args["target_level"] = random.randint(70, 95)
            
            # 应用格式化，确保所有占位符都有默认值
            try:
                return expression.format(**format_args)
            except KeyError as e:
                # 如果缺少某些参数，尝试提供默认值
                missing_key = str(e).strip("'")
                format_args[missing_key] = "相关"
                return expression.format(**format_args)
        
        # 否则调用父类方法
        return super()._get_specific_expression(general_term, topic)
