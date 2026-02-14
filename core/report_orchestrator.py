#!/usr/bin/env python3
"""
报告生成编排器 - 统一报告生成入口
"""

from typing import Dict, Any, List, Optional
import logging

# 导入核心服务
from src.nlp import NLPService
from src.models.enhanced_llm import EnhancedLLM

# 导入新的模块
from src.models.config_manager import ConfigManager
from src.models.data_extractor import DataExtractor
from src.models.template_engine import TemplateLoader, FieldMapper, ContentRenderer
from src.models.analysis_pipeline import StatisticalAnalyzer, CorrelationCalculator, InsightGenerator
from src.models.report_builder import SectionComposer, LLMIntegrator, FormatAdapter
from src.models.quality_assurance import QualityAssurance
from src.models.report_orchestrator import ReportOrchestrator as ModelReportOrchestrator

# 导入报告验证器
from src.utils.report_validator import ReportValidator

logger = logging.getLogger(__name__)

class ReportRequest:
    """报告生成请求"""
    
    def __init__(self, user_query: str, files: List[str] = None, 
                 report_type: str = "general", analysis_depth: str = "standard"):
        """
        初始化报告生成请求
        
        Args:
            user_query: 用户查询内容
            files: 上传的文件列表
            report_type: 报告类型（general, detailed, summary等）
            analysis_depth: 分析深度（standard, deep等）
        """
        self.user_query = user_query
        self.files = files or []
        self.report_type = report_type
        self.analysis_depth = analysis_depth

class Report:
    """报告对象"""
    
    def __init__(self, content: str, metadata: Dict[str, Any] = None):
        """
        初始化报告
        
        Args:
            content: 报告内容
            metadata: 报告元数据
        """
        self.content = content
        self.metadata = metadata or {}
        self.validation_result = None
        
    def set_validation_result(self, result: Dict[str, Any]):
        """设置报告验证结果"""
        self.validation_result = result

class ReportOrchestrator:
    """报告生成编排器（单一入口）"""
    
    def __init__(self, is_mock_model: bool = False):
        """
        初始化报告生成编排器
        
        Args:
            is_mock_model: 是否使用模拟模型
        """
        try:
            # 初始化核心服务
            self.nlp_service = NLPService(is_mock=is_mock_model)
            self.llm = EnhancedLLM(is_mock_model=is_mock_model)
            self.report_validator = ReportValidator()
            
            # 使用新的报告协调器
            self.model_orchestrator = ModelReportOrchestrator()
            self.model_orchestrator.set_llm_mode(is_mock_model)
            
            logger.info("ReportOrchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ReportOrchestrator: {str(e)}")
            raise
    
    def generate_report(self, request: ReportRequest) -> Report:
        """
        统一报告生成入口
        
        Args:
            request: 报告生成请求
            
        Returns:
            Report: 生成的报告对象
        """
        try:
            logger.info(f"Generating report with query: {request.user_query[:50]}..., files: {len(request.files)}")
            
            # 1. 解析请求
            logger.info("Step 1: Parsing request")
            query_analysis = self.nlp_service.analyze_query(request.user_query)
            
            # 2. 处理文档（如果有）
            logger.info("Step 2: Processing documents")
            document_analyses = []
            if request.files:
                for file in request.files:
                    # 这里简化处理，实际应该读取文件内容
                    content = f"Sample content from file: {file}"
                    analysis = self.nlp_service.process_document(content)
                    document_analyses.append(analysis)
            
            # 3. 使用新的ModelReportOrchestrator生成报告
            logger.info("Step 3: Generating report using ModelReportOrchestrator")
            report_result = self.model_orchestrator.generate_report(
                files=request.files,
                report_type=request.report_type,
                output_format='markdown'
            )
            
            # 4. 构建符合原有接口的Report对象
            metadata = {
                "query_analysis": query_analysis.intent,
                "report_type": request.report_type,
                "analysis_depth": request.analysis_depth,
                "file_count": len(request.files),
                "document_analyses": [analysis.to_dict() for analysis in document_analyses],
                "report_metadata": report_result.get('report_metadata', {}),
                "analysis_results": report_result.get('statistical_analysis', {})
            }
            
            report = Report(
                content=report_result.get('report_content', ''),
                metadata=metadata
            )
            
            # 5. 使用新的验证结果
            report.set_validation_result(report_result.get('quality_validation', {}))
            
            logger.info("Report generation completed successfully")
            
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            import traceback
            logger.debug(f"Report generation exception: {traceback.format_exc()}")
            
            # 返回包含错误信息的报告
            error_report = Report(
                content=f"报告生成失败: {str(e)}",
                metadata={"error": str(e)}
            )
            return error_report
    
    def get_report_analysis(self, report: Report) -> Dict[str, Any]:
        """
        获取报告分析信息
        
        Args:
            report: 报告对象
            
        Returns:
            Dict: 报告分析信息
        """
        try:
            # 分析报告内容
            content_analysis = self.nlp_service.process_document(report.content)
            
            analysis = {
                "report_length": len(report.content),
                "content_analysis": content_analysis.to_dict(),
                "validation_result": report.validation_result,
                "metadata": report.metadata
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Report analysis failed: {str(e)}")
            return {"error": str(e)}

# 测试代码
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    try:
        # 创建报告编排器
        orchestrator = ReportOrchestrator(is_mock_model=True)
        
        # 创建报告请求
        request = ReportRequest(
            user_query="分析本周黄金价格走势和影响因素",
            report_type="financial",
            analysis_depth="deep"
        )
        
        # 生成报告
        print("\n=== 生成报告 ===")
        report = orchestrator.generate_report(request)
        
        # 输出报告信息
        print(f"\n报告内容 (前300字符):")
        print(report.content[:300] + "...")
        
        print(f"\n报告元数据:")
        print(f"  报告类型: {report.metadata['report_type']}")
        print(f"  分析深度: {report.metadata['analysis_depth']}")
        print(f"  主要领域: {report.metadata['query_analysis']['domain']['primary']}")
        
        print(f"\n报告验证结果:")
        print(f"  验证状态: {'通过' if report.validation_result['is_valid'] else '失败'}")
        
        # 获取报告分析
        print(f"\n=== 报告分析 ===")
        analysis = orchestrator.get_report_analysis(report)
        print(f"  报告长度: {analysis['report_length']} 字符")
        print(f"  关键点数量: {len(analysis['content_analysis']['key_points'])}")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
