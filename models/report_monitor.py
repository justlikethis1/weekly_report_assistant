#!/usr/bin/env python3
"""
报告监控器
负责跟踪报告生成过程中的各种指标和事件
"""

from typing import Dict, Any, List
import logging
import json
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class ReportMonitor:
    """报告监控器，用于跟踪报告生成过程中的各种指标"""
    
    def __init__(self):
        """初始化报告监控器"""
        self.metrics = {
            'data_quality': [],
            'generation_time': [],
            'llm_calls': 0,
            'errors': [],
            'warnings': [],
            'report_counts': {
                'total': 0,
                'success': 0,
                'failure': 0,
                'fallback': 0
            }
        }
        
        # 会话开始时间
        self.session_start_time = datetime.now()
        
        logger.info("报告监控器初始化完成")
    
    def log_generation(self, report_result: Dict[str, Any], data: Dict[str, Any] = None) -> None:
        """
        记录报告生成过程
        
        Args:
            report_result: 报告生成结果
            data: 报告数据
        """
        # 更新报告计数
        self.metrics['report_counts']['total'] += 1
        
        if report_result.get('success', False):
            self.metrics['report_counts']['success'] += 1
        else:
            self.metrics['report_counts']['failure'] += 1
            
        # 检查是否是降级报告
        if 'errors' in report_result and report_result['errors']:
            self.metrics['report_counts']['fallback'] += 1
        
        # 记录数据质量
        if 'data_quality' in report_result:
            self.metrics['data_quality'].append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'score': report_result['data_quality'].get('score', 0),
                'issues': report_result['data_quality'].get('issues', []),
                'report_success': report_result.get('success', False)
            })
        
        # 记录生成时间
        if 'generation_time' in report_result:
            self.metrics['generation_time'].append(report_result['generation_time'])
    
    def log_llm_call(self, prompt_type: str, success: bool = True, error: str = None) -> None:
        """
        记录LLM调用
        
        Args:
            prompt_type: 提示类型
            success: 是否成功
            error: 错误信息（如果有）
        """
        self.metrics['llm_calls'] += 1
        
        if not success:
            self.log_error(f"LLM调用失败: {error}", error_type="llm_error")
    
    def log_error(self, error_message: str, error_type: str = "general_error") -> None:
        """
        记录错误
        
        Args:
            error_message: 错误信息
            error_type: 错误类型
        """
        self.metrics['errors'].append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'message': error_message,
            'type': error_type
        })
        
        logger.error(error_message)
    
    def log_warning(self, warning_message: str, warning_type: str = "general_warning") -> None:
        """
        记录警告
        
        Args:
            warning_message: 警告信息
            warning_type: 警告类型
        """
        self.metrics['warnings'].append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'message': warning_message,
            'type': warning_type
        })
        
        logger.warning(warning_message)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        获取所有指标
        
        Returns:
            所有指标
        """
        return self.metrics
    
    def generate_report(self) -> str:
        """
        生成监控报告
        
        Returns:
            监控报告内容
        """
        report = f"# 报告监控器 - 运行报告\n\n"
        report += f"会话开始时间: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # 报告计数
        report += "## 报告计数\n\n"
        report += f"总报告数: {self.metrics['report_counts']['total']}\n"
        report += f"成功报告数: {self.metrics['report_counts']['success']}\n"
        report += f"失败报告数: {self.metrics['report_counts']['failure']}\n"
        report += f"降级报告数: {self.metrics['report_counts']['fallback']}\n\n"
        
        # 数据质量统计
        if self.metrics['data_quality']:
            report += "## 数据质量统计\n\n"
            avg_score = sum(item['score'] for item in self.metrics['data_quality']) / len(self.metrics['data_quality'])
            report += f"平均数据质量得分: {avg_score:.2f}/10\n"
            
            # 统计各质量级别的报告数量
            quality_levels = {
                'excellent': 0,  # 8-10分
                'good': 0,        # 6-7分
                'fair': 0,        # 4-5分
                'poor': 0         # 0-3分
            }
            
            for item in self.metrics['data_quality']:
                score = item['score']
                if score >= 8:
                    quality_levels['excellent'] += 1
                elif score >= 6:
                    quality_levels['good'] += 1
                elif score >= 4:
                    quality_levels['fair'] += 1
                else:
                    quality_levels['poor'] += 1
            
            report += f"优秀数据质量报告数 (8-10分): {quality_levels['excellent']}\n"
            report += f"良好数据质量报告数 (6-7分): {quality_levels['good']}\n"
            report += f"一般数据质量报告数 (4-5分): {quality_levels['fair']}\n"
            report += f"较差数据质量报告数 (0-3分): {quality_levels['poor']}\n\n"
        
        # LLM调用统计
        report += "## LLM调用统计\n\n"
        report += f"总调用次数: {self.metrics['llm_calls']}\n\n"
        
        # 错误和警告统计
        report += "## 错误和警告统计\n\n"
        report += f"错误总数: {len(self.metrics['errors'])}\n"
        report += f"警告总数: {len(self.metrics['warnings'])}\n\n"
        
        # 最近的错误（最多5个）
        if self.metrics['errors']:
            report += "### 最近的错误\n\n"
            for i, error in enumerate(self.metrics['errors'][-5:], 1):
                report += f"{i}. [{error['timestamp']}] {error['type']}: {error['message']}\n"
            
            if len(self.metrics['errors']) > 5:
                report += f"... 还有 {len(self.metrics['errors']) - 5} 个错误未显示\n\n"
        
        # 最近的警告（最多5个）
        if self.metrics['warnings']:
            report += "### 最近的警告\n\n"
            for i, warning in enumerate(self.metrics['warnings'][-5:], 1):
                report += f"{i}. [{warning['timestamp']}] {warning['type']}: {warning['message']}\n"
            
            if len(self.metrics['warnings']) > 5:
                report += f"... 还有 {len(self.metrics['warnings']) - 5} 个警告未显示\n\n"
        
        return report
    
    def save_metrics(self, file_path: str = "monitor_metrics.json") -> bool:
        """
        将监控指标保存到文件中
        
        Args:
            file_path: 保存文件路径
            
        Returns:
            bool: 保存是否成功
        """
        try:
            import json
            import os
            
            # 创建目录（如果不存在）
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # 保存指标到文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.metrics, f, ensure_ascii=False, indent=2)
            
            logger.info(f"监控指标已保存到 {file_path}")
            return True
        except Exception as e:
            logger.error(f"保存监控指标失败: {e}", exc_info=True)
            return False
    
    @staticmethod
    def load_metrics(file_path: str = "monitor_metrics.json") -> Dict[str, Any]:
        """
        从文件中加载监控指标
        
        Args:
            file_path: 加载文件路径
            
        Returns:
            Dict[str, Any]: 监控指标
        """
        try:
            import json
            import os
            
            if not os.path.exists(file_path):
                logger.warning(f"监控指标文件不存在: {file_path}")
                return {
                    'data_quality': [],
                    'generation_time': [],
                    'llm_calls': 0,
                    'errors': [],
                    'warnings': [],
                    'report_counts': {
                        'total': 0,
                        'success': 0,
                        'failure': 0,
                        'fallback': 0
                    }
                }
            
            # 从文件中加载指标
            with open(file_path, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            
            logger.info(f"监控指标已从 {file_path} 加载")
            return metrics
        except Exception as e:
            logger.error(f"加载监控指标失败: {e}", exc_info=True)
            return {
                'data_quality': [],
                'generation_time': [],
                'llm_calls': 0,
                'errors': [],
                'warnings': [],
                'report_counts': {
                    'total': 0,
                    'success': 0,
                    'failure': 0,
                    'fallback': 0
                }
            }
    
    def reset(self) -> None:
        """
        重置所有指标
        """
        self.metrics = {
            'data_quality': [],
            'generation_time': [],
            'llm_calls': 0,
            'errors': [],
            'warnings': [],
            'report_counts': {
                'total': 0,
                'success': 0,
                'failure': 0,
                'fallback': 0
            }
        }
        
        logger.info("监控器指标已重置")


# 创建一个全局监控器实例
monitor = ReportMonitor()