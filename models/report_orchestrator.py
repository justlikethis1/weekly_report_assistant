#!/usr/bin/env python3
"""
报告协调器
整合所有模块，实现完整的报告生成流程
"""

from typing import Dict, Any, List, Optional
import logging
import os
import time
from datetime import datetime

# 导入所有模块
from .config_manager import ConfigManager
from .data_extractor import DataExtractor
from .template_engine import TemplateLoader, FieldMapper, ContentRenderer, ReportTemplate
from .analysis_pipeline import StatisticalAnalyzer, CorrelationCalculator, InsightGenerator
from .report_builder import FormatAdapter
from .quality_assurance import QualityAssurance
from .report_monitor import monitor

logger = logging.getLogger(__name__)


class ReportOrchestrator:
    """报告协调器，整合所有模块实现完整的报告生成流程"""
    
    def __init__(self, config_dir: str = None):
        """
        初始化报告协调器
        
        Args:
            config_dir: 配置文件目录
        """
        # 初始化各模块
        self.config_manager = ConfigManager(config_dir)
        self.data_extractor = DataExtractor()
        
        # 模板引擎相关
        self.template_loader = TemplateLoader()
        self.field_mapper = FieldMapper()
        self.content_renderer = ContentRenderer(self.template_loader)
        
        # 分析流水线相关
        self.statistical_analyzer = StatisticalAnalyzer()
        self.correlation_calculator = CorrelationCalculator()
        self.insight_generator = InsightGenerator()
        
        # 报告构建相关
        self.format_adapter = FormatAdapter()
        
        # 质量保证
        self.quality_assurance = QualityAssurance()
        
        # 报告模板系统
        self.report_template = ReportTemplate()
        
        logger.info("报告协调器初始化完成")
    
    def set_llm_mode(self, mock_mode: bool):
        """
        设置LLM模式
        
        Args:
            mock_mode: 是否使用模拟模式
        """
        # 这个方法现在主要用于日志记录，实际设置在ContentGenerationCoordinator中进行
        logger.info(f"LLM模式设置为 {'模拟' if mock_mode else '真实'}")
    
    def generate_report(self, files: List[str] = None, report_type: str = 'default', output_format: str = 'markdown', report_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        生成报告 - 改进版，实现完整的数据验证和异常处理
        
        Args:
            files: 数据文件列表
            report_type: 报告类型
            output_format: 输出格式
            report_config: 报告配置
            
        Returns:
            生成的报告信息
        """
        logger.info(f"开始生成报告，类型: {report_type}, 格式: {output_format}")
        
        # 记录报告生成开始时间
        start_time = time.time()
        
        result = {
            'success': False,
            'report_content': '',
            'report_sections': [],
            'report_metadata': {},
            'quality_validation': {},
            'data_quality': {'score': 0, 'issues': []},
            'errors': [],
            'generation_time': 0
        }
        
        try:
            # 如果没有提供配置，从配置管理器获取
            if not report_config and report_type:
                report_config = self.config_manager.get_report_config(report_type)
            
            # 步骤1: 数据提取
            logger.info("执行数据提取...")
            data = self.data_extractor.extract_data(files, report_type)
            
            # 步骤2: 数据验证
            logger.info("执行数据验证...")
            from .data_extractor import DataValidator
            price_validation = DataValidator.validate_price_data(data['price_data'])
            
            if not price_validation['is_valid']:
                logger.warning(f"价格数据验证失败: {price_validation}")
                result['errors'].extend(price_validation['issues'])
                result['data_quality']['issues'] = price_validation['issues']
                result['data_quality']['missing_fields'] = price_validation['missing_fields']
                
                # 实现智能降级策略
                logger.info("执行智能降级，生成基础报告...")
                return self._generate_fallback_report(result, price_validation, report_type)
            
            # 步骤3: 数据预处理（字段映射）
            logger.info("执行数据预处理...")
            mapped_data = self.field_mapper.map_fields(data, report_config)
            
            # 计算数据质量得分并添加到映射数据中
            result['data_quality']['score'] = self._calculate_data_quality_score(data)
            mapped_data['data_quality'] = result['data_quality']
            
            # 步骤4: 执行分析流水线
            logger.info("执行分析流水线...")
            analysis_results = self._run_analysis_pipeline(mapped_data)
            
            # 合并分析结果到数据中
            mapped_data['statistical_analysis'] = analysis_results['statistical_analysis']
            mapped_data['correlations'] = analysis_results['correlations']
            mapped_data['insights'] = analysis_results['insights']
            
            # 步骤5-7: 使用内容生成协调器生成并增强报告章节
            logger.info("执行内容生成协调器生成并增强报告章节...")
            from .content_generation_coordinator import ContentGenerationCoordinator
            content_coordinator = ContentGenerationCoordinator()
            
            # 生成报告章节（规则生成 + AI增强 + 内容融合）
            enhanced_sections = content_coordinator.generate_report_sections(
                mapped_data, 
                report_config, 
                ai_enhancement=True, 
                max_workers=2
            )
            
            # 步骤7: 格式转换
            logger.info("执行格式转换...")
            report_metadata = self._create_report_metadata(report_type, files)
            report_content = self.format_adapter.adapt(enhanced_sections, output_format, report_metadata)
            
            # 步骤8: 质量验证
            logger.info("执行质量验证...")
            quality_validation = self.quality_assurance.validate_report(mapped_data, enhanced_sections, report_config)
            
            # 更新结果
            result['success'] = True
            result['report_content'] = report_content
            result['report_sections'] = enhanced_sections
            result['report_metadata'] = report_metadata
            result['quality_validation'] = quality_validation
                
            # 将数据质量信息添加到报告元数据中，以便在模板中使用
            report_metadata['data_quality'] = result['data_quality']
                
            logger.info(f"报告生成完成，质量得分: {quality_validation['overall_result']['score']}/100, 数据质量得分: {result['data_quality']['score']}/10")
            
        except Exception as e:
            logger.error(f"报告生成失败: {e}", exc_info=True)
            result['errors'].append(str(e))
            # 生成错误报告
            result = self._generate_error_report(result, str(e), report_type)
            
            # 记录错误
            for error in result['errors']:
                monitor.log_error(error, error_type="report_generation_error")
        
        # 计算并记录生成时间
        result['generation_time'] = time.time() - start_time
        
        # 使用监控器记录报告生成结果
        monitor.log_generation(result)
        
        logger.info(f"报告生成完成，总耗时: {result['generation_time']:.2f}秒")
        
        return result
    
    def _run_analysis_pipeline(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行分析流水线
        
        Args:
            data: 要分析的数据
            
        Returns:
            分析结果
        """
        results = {
            'statistical_analysis': {},
            'correlations': {},
            'insights': []
        }
        
        # 1. 统计分析
        if 'price_data' in data:
            results['statistical_analysis'] = self.statistical_analyzer.analyze(data['price_data'])
        
        # 2. 相关性计算
        results['correlations'] = self.correlation_calculator.calculate_correlations(data)
        
        # 3. 洞察生成
        results['insights'] = self.insight_generator.generate_insights(
            results['statistical_analysis'],
            results['correlations']
        )
        
        return results
    

    
    def _create_report_metadata(self, report_type: str, files: List[str] = None) -> Dict[str, Any]:
        """
        创建报告元数据
        
        Args:
            report_type: 报告类型
            files: 使用的数据文件
            
        Returns:
            报告元数据
        """
        metadata = {
            'title': f"{report_type}报告",
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'version': "1.0",
            'generated_by': "Weekly Report Assistant",
            'report_type': report_type
        }
        
        if files:
            metadata['source_files'] = [os.path.basename(file) for file in files]
            metadata['source_count'] = len(files)
        
        return metadata
        
    def _calculate_data_quality_score(self, data: Dict[str, Any]) -> int:
        """
        计算数据质量得分
        
        Args:
            data: 要评估的数据
            
        Returns:
            数据质量得分（0-10）
        """
        score = 10
        
        # 检查价格数据
        if 'price_data' in data:
            price_data = data['price_data']
            
            # 检查是否有价格数据
            if not price_data or 'price_changes' not in price_data:
                score -= 5
            
            # 检查价格变动记录数量
            elif len(price_data['price_changes']) < 2:
                score -= 3
            
            # 检查是否有异常值
            prices = [pc['price'] for pc in price_data['price_changes'] if 'price' in pc]
            if prices:
                max_price = max(prices)
                min_price = min(prices)
                
                # 检查价格范围是否合理
                if max_price == 0 and min_price == 0:
                    score -= 10
                elif max_price > 10000 or min_price < 0:
                    score -= 5
        
        # 确保得分在0-10范围内
        return max(0, min(10, score))
        
    def _generate_fallback_report(self, result: Dict[str, Any], validation_result: Dict[str, Any], report_type: str) -> Dict[str, Any]:
        """
        生成降级报告 - 当数据质量较差时使用
        
        Args:
            result: 报告结果对象
            validation_result: 数据验证结果
            report_type: 报告类型
            
        Returns:
            更新后的报告结果
        """
        logger.info("生成降级报告")
        
        # 创建降级报告内容
        report_metadata = self._create_report_metadata(report_type)
        
        report_content = f"# {report_type}报告 - 数据质量警告\n\n"
        report_content += "## 数据质量问题\n\n"
        report_content += "我们在生成报告时遇到了以下数据质量问题：\n\n"
        
        for issue in validation_result['issues']:
            report_content += f"- {issue}\n"
        
        if validation_result['missing_fields']:
            report_content += f"\n缺少以下必要字段：\n\n"
            for field in validation_result['missing_fields']:
                report_content += f"- {field}\n"
        
        report_content += f"\n## 建议解决方案\n\n"
        report_content += validation_result['suggestion']
        report_content += "\n\n请检查数据源连接或API配置，确保数据质量后重新生成报告。\n"
        
        # 构建降级报告章节
        sections = [
            {
                'id': 'data_quality_warning',
                'title': '数据质量警告',
                'content': "我们在生成报告时遇到了数据质量问题，无法生成完整报告。",
                'importance': 5
            },
            {
                'id': 'data_quality_issues',
                'title': '发现的问题',
                'content': "\n".join(validation_result['issues']),
                'importance': 5
            },
            {
                'id': 'suggestions',
                'title': '建议解决方案',
                'content': validation_result['suggestion'],
                'importance': 5
            }
        ]
        
        # 更新结果
        result['report_content'] = report_content
        result['report_sections'] = sections
        result['report_metadata'] = report_metadata
        result['data_quality']['score'] = 0
        result['success'] = False  # 虽然生成了报告，但数据质量有问题
        
        logger.info("降级报告生成完成")
        return result
        
    def _generate_error_report(self, result: Dict[str, Any], error_message: str, report_type: str) -> Dict[str, Any]:
        """
        生成错误报告 - 当发生异常时使用
        
        Args:
            result: 报告结果对象
            error_message: 错误信息
            report_type: 报告类型
            
        Returns:
            更新后的报告结果
        """
        logger.info("生成错误报告")
        
        # 创建错误报告内容
        report_metadata = self._create_report_metadata(report_type)
        
        report_content = f"# {report_type}报告 - 生成失败\n\n"
        report_content += "## 生成错误\n\n"
        report_content += f"报告生成过程中遇到错误：\n\n{error_message}\n\n"
        
        report_content += "## 可能的原因\n\n"
        report_content += "- 数据源连接问题\n"
        report_content += "- 文件格式错误或损坏\n"
        report_content += "- 配置文件缺失或错误\n"
        report_content += "- 系统资源不足\n\n"
        
        report_content += "## 建议解决方案\n\n"
        report_content += "1. 检查数据源连接或API配置\n"
        report_content += "2. 验证输入文件格式和内容\n"
        report_content += "3. 检查配置文件是否正确\n"
        report_content += "4. 确保系统有足够的内存和磁盘空间\n\n"
        
        report_content += "如果问题持续存在，请联系技术支持。\n"
        
        # 构建错误报告章节
        sections = [
            {
                'id': 'generation_error',
                'title': '报告生成错误',
                'content': f"报告生成过程中遇到错误：\n{error_message}",
                'importance': 5
            },
            {
                'id': 'possible_causes',
                'title': '可能的原因',
                'content': "- 数据源连接问题\n- 文件格式错误或损坏\n- 配置文件缺失或错误\n- 系统资源不足",
                'importance': 5
            },
            {
                'id': 'solutions',
                'title': '建议解决方案',
                'content': "1. 检查数据源连接或API配置\n2. 验证输入文件格式和内容\n3. 检查配置文件是否正确\n4. 确保系统有足够的内存和磁盘空间",
                'importance': 5
            }
        ]
        
        # 更新结果
        result['report_content'] = report_content
        result['report_sections'] = sections
        result['report_metadata'] = report_metadata
        result['data_quality']['score'] = 0
        result['success'] = False
        
        logger.info("错误报告生成完成")
        return result
    
    def save_report(self, report_content: str, output_path: str, output_format: str = 'markdown') -> bool:
        """
        保存报告到文件
        
        Args:
            report_content: 报告内容
            output_path: 输出文件路径
            output_format: 输出格式
            
        Returns:
            保存是否成功
        """
        try:
            # 确保目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # 保存文件
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"报告已保存到: {output_path}")
            return True
        except Exception as e:
            logger.error(f"保存报告失败: {e}")
            return False
    
    def generate_quality_assurance_report(self, validation_results: Dict[str, Any], output_path: str = None) -> str:
        """
        生成质量保证报告
        
        Args:
            validation_results: 质量验证结果
            output_path: 输出文件路径
            
        Returns:
            质量报告内容
        """
        quality_report = self.quality_assurance.generate_quality_report(validation_results)
        
        if output_path:
            self.save_report(quality_report, output_path, 'markdown')
        
        return quality_report
