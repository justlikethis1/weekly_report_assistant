#!/usr/bin/env python3
"""
质量保证层
负责检查数据一致性、结构完整性和风格合规性
"""

from typing import Dict, Any, List, Optional
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)


class DataConsistencyChecker:
    """数据一致性检查器，负责检查数据一致性"""
    
    def __init__(self):
        """初始化数据一致性检查器"""
        pass
    
    def check_consistency(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查数据一致性
        
        Args:
            data: 要检查的数据
            
        Returns:
            一致性检查结果
        """
        results = {
            'passed': True,
            'issues': [],
            'score': 100.0
        }
        
        # 检查价格数据一致性
        if 'price_data' in data:
            price_issues = self._check_price_data_consistency(data['price_data'])
            if price_issues:
                results['passed'] = False
                results['issues'].extend(price_issues)
        
        # 检查分析结果一致性
        if 'statistical_analysis' in data:
            analysis_issues = self._check_analysis_consistency(data['statistical_analysis'])
            if analysis_issues:
                results['passed'] = False
                results['issues'].extend(analysis_issues)
        
        # 计算一致性得分
        if results['issues']:
            results['score'] = max(0.0, 100.0 - len(results['issues']) * 10.0)
        
        return results
    
    def _check_price_data_consistency(self, price_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        检查价格数据一致性
        
        Args:
            price_data: 价格数据
            
        Returns:
            发现的问题列表
        """
        issues = []
        
        # 检查起始价格和结束价格
        if 'start_price' in price_data and 'end_price' in price_data:
            if price_data['start_price'] < 0 or price_data['end_price'] < 0:
                issues.append({
                    'type': 'negative_price',
                    'severity': 'error',
                    'message': '价格不能为负值'
                })
        
        # 检查最高价和最低价
        if 'high_price' in price_data and 'low_price' in price_data:
            if price_data['high_price'] < price_data['low_price']:
                issues.append({
                    'type': 'invalid_price_range',
                    'severity': 'error',
                    'message': '最高价不能低于最低价'
                })
        
        # 检查价格变动数据
        if 'price_changes' in price_data and price_data['price_changes']:
            price_changes = price_data['price_changes']
            
            # 检查日期顺序
            dates = []
            prices = []
            
            for item in price_changes:
                if 'date' in item and 'price' in item:
                    dates.append(item['date'])
                    prices.append(item['price'])
            
            # 检查日期是否重复
            if len(dates) != len(set(dates)):
                issues.append({
                    'type': 'duplicate_dates',
                    'severity': 'warning',
                    'message': '发现重复的日期记录'
                })
            
            # 检查价格是否在合理范围内
            if prices:
                min_price = min(prices)
                max_price = max(prices)
                
                if min_price < 0:
                    issues.append({
                        'type': 'negative_price_in_changes',
                        'severity': 'error',
                        'message': '价格变动数据中包含负值'
                    })
                
                # 检查价格变动是否异常大
                for i in range(1, len(prices)):
                    change_percent = abs((prices[i] - prices[i-1]) / prices[i-1] * 100)
                    if change_percent > 20:  # 超过20%的变动视为异常
                        issues.append({
                            'type': 'abnormal_price_change',
                            'severity': 'warning',
                            'message': f'第{i+1}条记录价格变动异常大：{change_percent:.2f}%'
                        })
        
        return issues
    
    def _check_analysis_consistency(self, statistical_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        检查分析结果一致性
        
        Args:
            statistical_analysis: 统计分析结果
            
        Returns:
            发现的问题列表
        """
        issues = []
        
        # 检查波动率分析结果
        if 'volatility' in statistical_analysis:
            volatility = statistical_analysis['volatility']
            
            if 'annualized_volatility' in volatility and volatility['annualized_volatility'] < 0:
                issues.append({
                    'type': 'negative_volatility',
                    'severity': 'error',
                    'message': '波动率不能为负值'
                })
        
        return issues


class StructuralValidator:
    """结构验证器，负责验证报告结构完整性"""
    
    def __init__(self):
        """初始化结构验证器"""
        pass
    
    def validate_structure(self, sections: List[Dict[str, Any]], report_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        验证报告结构完整性
        
        Args:
            sections: 报告章节
            report_config: 报告配置
            
        Returns:
            结构验证结果
        """
        results = {
            'passed': True,
            'issues': [],
            'score': 100.0
        }
        
        # 检查是否有章节
        if not sections:
            results['passed'] = False
            results['issues'].append({
                'type': 'no_sections',
                'severity': 'error',
                'message': '报告没有任何章节'
            })
            results['score'] = 0.0
            return results
        
        # 根据配置检查必填章节
        if report_config and 'template_sections' in report_config:
            config_issues = self._validate_against_config(sections, report_config['template_sections'])
            if config_issues:
                results['passed'] = False
                results['issues'].extend(config_issues)
        else:
            # 使用默认验证规则
            default_issues = self._validate_against_default_rules(sections)
            if default_issues:
                results['passed'] = False
                results['issues'].extend(default_issues)
        
        # 检查章节顺序
        order_issues = self._check_section_order(sections)
        if order_issues:
            results['issues'].extend(order_issues)
        
        # 计算结构得分
        if results['issues']:
            results['score'] = max(0.0, 100.0 - len(results['issues']) * 15.0)
        
        return results
    
    def _validate_against_config(self, sections: List[Dict[str, Any]], section_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        根据配置验证章节
        
        Args:
            sections: 报告章节
            section_configs: 章节配置
            
        Returns:
            发现的问题列表
        """
        issues = []
        
        # 检查必填章节是否存在
        section_ids = [section['id'] for section in sections]
        
        for section_config in section_configs:
            if section_config.get('required', False) and section_config['id'] not in section_ids:
                issues.append({
                    'type': 'missing_required_section',
                    'severity': 'error',
                    'message': f'缺少必填章节：{section_config.get("title", section_config["id"])}'
                })
        
        return issues
    
    def _validate_against_default_rules(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        使用默认规则验证章节
        
        Args:
            sections: 报告章节
            
        Returns:
            发现的问题列表
        """
        issues = []
        
        # 默认规则：报告应该包含执行摘要和建议章节
        section_ids = [section['id'] for section in sections]
        
        if 'executive_summary' not in section_ids:
            issues.append({
                'type': 'missing_executive_summary',
                'severity': 'warning',
                'message': '报告缺少执行摘要章节'
            })
        
        if 'insights_and_recommendations' not in section_ids:
            issues.append({
                'type': 'missing_recommendations',
                'severity': 'warning',
                'message': '报告缺少洞察与建议章节'
            })
        
        return issues
    
    def _check_section_order(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        检查章节顺序
        
        Args:
            sections: 报告章节
            
        Returns:
            发现的问题列表
        """
        issues = []
        
        # 默认章节顺序规则
        expected_order = ['executive_summary', 'data_overview', 'statistical_analysis', 'insights_and_recommendations']
        
        # 提取实际章节顺序
        actual_order = [section['id'] for section in sections]
        
        # 检查是否按照期望顺序排列
        for i, section_id in enumerate(expected_order):
            if section_id in actual_order:
                actual_index = actual_order.index(section_id)
                
                # 检查后续章节是否在正确位置
                for j in range(i+1, len(expected_order)):
                    next_section_id = expected_order[j]
                    if next_section_id in actual_order:
                        next_index = actual_order.index(next_section_id)
                        if next_index < actual_index:
                            issues.append({
                                'type': 'incorrect_section_order',
                                'severity': 'info',
                                'message': f'章节顺序不正确：{section_id} 应该在 {next_section_id} 之前'
                            })
                            break
        
        return issues


class StyleEnforcer:
    """风格执行器，负责确保报告遵循一致的格式和风格"""
    
    def __init__(self):
        """初始化风格执行器"""
        self.style_rules = {
            'max_line_length': 80,
            'heading_style': 'markdown',  # markdown, html, plain
            'list_style': 'markdown',     # markdown, plain
            'spacing': 2,                 # 段落间距
            'font_size': 12,              # 字体大小
            'font_family': 'Arial',       # 字体
            'color_scheme': 'default'     # 配色方案
        }
    
    def set_style_rules(self, style_rules: Dict[str, Any]):
        """
        设置风格规则
        
        Args:
            style_rules: 风格规则
        """
        self.style_rules.update(style_rules)
    
    def enforce_style(self, content: str, format_type: str = 'markdown') -> Dict[str, Any]:
        """
        执行风格检查
        
        Args:
            content: 要检查的内容
            format_type: 内容格式
            
        Returns:
            风格检查结果
        """
        results = {
            'passed': True,
            'issues': [],
            'score': 100.0,
            'formatted_content': content  # 格式化后的内容
        }
        
        # 根据格式类型检查风格
        format_type = format_type.lower()
        
        if format_type == 'markdown':
            style_issues = self._check_markdown_style(content)
        elif format_type == 'html':
            style_issues = self._check_html_style(content)
        elif format_type == 'plain_text':
            style_issues = self._check_plain_text_style(content)
        else:
            style_issues = []
        
        if style_issues:
            results['passed'] = False
            results['issues'].extend(style_issues)
            results['score'] = max(0.0, 100.0 - len(style_issues) * 5.0)
        
        return results
    
    def _check_markdown_style(self, content: str) -> List[Dict[str, Any]]:
        """
        检查Markdown风格
        
        Args:
            content: Markdown内容
            
        Returns:
            发现的问题列表
        """
        issues = []
        
        # 检查行长度
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if len(line) > self.style_rules['max_line_length']:
                issues.append({
                    'type': 'line_too_long',
                    'severity': 'info',
                    'message': f'第{i+1}行长度超过限制（{len(line)} > {self.style_rules["max_line_length"]}）'
                })
        
        # 检查标题格式
        heading_pattern = re.compile(r'^(#{1,6})\s+(.*)$')
        for i, line in enumerate(lines):
            match = heading_pattern.match(line)
            if match:
                hashes, title = match.groups()
                
                # 检查标题后是否有空格
                if line.startswith(hashes) and not line.startswith(f'{hashes} '):
                    issues.append({
                        'type': 'heading_format',
                        'severity': 'warning',
                        'message': f'第{i+1}行标题格式不正确：# 后应该有空格'
                    })
        
        # 检查列表格式
        list_pattern = re.compile(r'^([*+-]|\d+\.)\s+(.*)$')
        for i, line in enumerate(lines):
            match = list_pattern.match(line)
            if match:
                marker, text = match.groups()
                
                # 检查列表项后是否有空格
                if line.startswith(marker) and not line.startswith(f'{marker} '):
                    issues.append({
                        'type': 'list_format',
                        'severity': 'warning',
                        'message': f'第{i+1}行列项目格式不正确：列表标记后应该有空格'
                    })
        
        # 检查代码块格式
        code_block_pattern = re.compile(r'^```.*$')
        in_code_block = False
        code_block_start = 0
        
        for i, line in enumerate(lines):
            if code_block_pattern.match(line):
                if in_code_block:
                    # 检查代码块是否有内容
                    if i == code_block_start + 1:
                        issues.append({
                            'type': 'empty_code_block',
                            'severity': 'info',
                            'message': f'第{i}行附近有空代码块'
                        })
                    in_code_block = False
                else:
                    in_code_block = True
                    code_block_start = i
        
        # 检查未闭合的代码块
        if in_code_block:
            issues.append({
                'type': 'unclosed_code_block',
                'severity': 'warning',
                'message': f'第{code_block_start+1}行开始的代码块未闭合'
            })
        
        return issues
    
    def _check_html_style(self, content: str) -> List[Dict[str, Any]]:
        """
        检查HTML风格
        
        Args:
            content: HTML内容
            
        Returns:
            发现的问题列表
        """
        issues = []
        
        # 检查HTML标签是否闭合
        tag_pattern = re.compile(r'<(/?)(\w+)([^>]*)>')
        tags = []
        
        for match in tag_pattern.finditer(content):
            is_closing = match.group(1) == '/'
            tag_name = match.group(2).lower()
            
            if is_closing:
                if not tags or tags[-1] != tag_name:
                    issues.append({
                        'type': 'unclosed_tag',
                        'severity': 'error',
                        'message': f'发现未闭合的HTML标签：{tag_name}'
                    })
                else:
                    tags.pop()
            else:
                # 检查自闭合标签
                if not match.group(3).endswith('/'):
                    # 非自闭合标签添加到栈中
                    tags.append(tag_name)
        
        # 检查未闭合的标签
        if tags:
            for tag in tags:
                issues.append({
                    'type': 'unclosed_tag',
                    'severity': 'error',
                    'message': f'发现未闭合的HTML标签：{tag}'
                })
        
        return issues
    
    def _check_plain_text_style(self, content: str) -> List[Dict[str, Any]]:
        """
        检查纯文本风格
        
        Args:
            content: 纯文本内容
            
        Returns:
            发现的问题列表
        """
        issues = []
        
        # 检查行长度
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if len(line) > self.style_rules['max_line_length']:
                issues.append({
                    'type': 'line_too_long',
                    'severity': 'info',
                    'message': f'第{i+1}行长度超过限制（{len(line)} > {self.style_rules["max_line_length"]}）'
                })
        
        # 检查多余空行
        for i in range(1, len(lines)):
            if lines[i] == '' and lines[i-1] == '':
                issues.append({
                    'type': 'excessive_empty_lines',
                    'severity': 'info',
                    'message': f'第{i+1}行附近有空行过多'
                })
        
        return issues


class ContentQualityValidator:
    """内容质量验证器，负责检查元数据泄露、内容完整性、可读性等"""
    
    def __init__(self):
        """初始化内容质量验证器"""
        # 元数据模式
        self.metadata_patterns = [
            r'\[INST\].*?\[/INST\]',  # 指令标记
            r'\[SYSTEM\].*?\[/SYSTEM\]',  # 系统提示标记
            r'\[USER\].*?\[/USER\]',  # 用户提示标记
            r'\{\{.*?\}\}',  # 模板变量
            r'\[DEBUG\].*?\[/DEBUG\]',  # 调试信息
            r'\[LOG\].*?\[/LOG\]'  # 日志信息
        ]
        
        # 敏感信息模式
        self.sensitive_patterns = [
            r'\b(?:API|api)\s*[_-]?\s*key\s*[:：]?\s*\w+',  # API密钥
            r'\bpassword\s*[:：]?\s*\w+',  # 密码
            r'\bsecret\s*[:：]?\s*\w+',  # 密钥
            r'\b(?:token|Token)\s*[:：]?\s*\w+'  # 令牌
        ]
        
        logger.info("ContentQualityValidator初始化完成")
    
    def check_metadata_leakage(self, content: str) -> Dict[str, Any]:
        """
        检查元数据泄露
        
        Args:
            content: 要检查的内容
            
        Returns:
            元数据泄露检查结果
        """
        results = {
            'passed': True,
            'issues': [],
            'score': 100.0
        }
        
        for pattern in self.metadata_patterns:
            matches = re.finditer(pattern, content, re.DOTALL)
            for match in matches:
                results['passed'] = False
                results['issues'].append({
                    'type': 'metadata_leakage',
                    'severity': 'error',
                    'message': f'发现元数据泄露：{match.group(0)[:50]}...'
                })
        
        # 计算得分
        if results['issues']:
            results['score'] = max(0.0, 100.0 - len(results['issues']) * 20.0)
        
        return results
    
    def check_content_integrity(self, sections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        检查内容完整性
        
        Args:
            sections: 报告章节
            
        Returns:
            内容完整性检查结果
        """
        results = {
            'passed': True,
            'issues': [],
            'score': 100.0
        }
        
        for section in sections:
            section_content = section.get('content', '')
            
            # 检查章节是否为空
            if not section_content.strip():
                results['passed'] = False
                results['issues'].append({
                    'type': 'empty_section',
                    'severity': 'error',
                    'message': f'章节 {section.get("title", section.get("id", "未知"))} 内容为空'
                })
                continue
            
            # 检查内容长度
            word_count = len(section_content.split())
            if word_count < 50:
                results['passed'] = False
                results['issues'].append({
                    'type': 'content_too_short',
                    'severity': 'warning',
                    'message': f'章节 {section.get("title", section.get("id", "未知"))} 内容过短（{word_count} 词）'
                })
        
        # 计算得分
        if results['issues']:
            results['score'] = max(0.0, 100.0 - len(results['issues']) * 10.0)
        
        return results
    
    def evaluate_readability(self, content: str) -> Dict[str, Any]:
        """
        评估内容可读性
        
        Args:
            content: 要评估的内容
            
        Returns:
            可读性评估结果
        """
        results = {
            'passed': True,
            'issues': [],
            'score': 100.0,
            'metrics': {
                'word_count': 0,
                'sentence_count': 0,
                'avg_sentence_length': 0.0,
                'avg_word_length': 0.0
            }
        }
        
        # 计算单词数
        words = content.split()
        results['metrics']['word_count'] = len(words)
        
        # 计算句子数（简单实现）
        sentences = re.split(r'[。！？\n]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        results['metrics']['sentence_count'] = len(sentences)
        
        # 计算平均句子长度
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            results['metrics']['avg_sentence_length'] = round(avg_sentence_length, 2)
            
            # 平均句子长度超过20个词视为可读性差
            if avg_sentence_length > 20:
                results['passed'] = False
                results['issues'].append({
                    'type': 'long_sentences',
                    'severity': 'warning',
                    'message': f'平均句子长度过长（{avg_sentence_length:.2f} 词），影响可读性'
                })
        
        # 计算平均单词长度
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
            results['metrics']['avg_word_length'] = round(avg_word_length, 2)
        
        # 计算得分
        if results['issues']:
            results['score'] = max(0.0, 100.0 - len(results['issues']) * 15.0)
        
        return results
    
    def screen_sensitive_info(self, content: str) -> Dict[str, Any]:
        """
        筛查敏感信息
        
        Args:
            content: 要筛查的内容
            
        Returns:
            敏感信息筛查结果
        """
        results = {
            'passed': True,
            'issues': [],
            'score': 100.0
        }
        
        for pattern in self.sensitive_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                results['passed'] = False
                results['issues'].append({
                    'type': 'sensitive_info',
                    'severity': 'error',
                    'message': f'发现敏感信息：{match.group(0)}'
                })
        
        # 计算得分
        if results['issues']:
            results['score'] = 0.0  # 发现敏感信息直接得0分
        
        return results


class QualityAssurance:
    """质量保证主类，整合所有质量检查功能"""
    
    def __init__(self):
        """初始化质量保证系统"""
        self.data_checker = DataConsistencyChecker()
        self.structural_validator = StructuralValidator()
        self.style_enforcer = StyleEnforcer()
        self.content_validator = ContentQualityValidator()  # 新添加的内容质量验证器
    
    def validate_report(self, report_data: Dict[str, Any], sections: List[Dict[str, Any]], report_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        验证整个报告
        
        Args:
            report_data: 报告数据
            sections: 报告章节
            report_config: 报告配置
            
        Returns:
            完整的质量验证结果
        """
        results = {
            'overall_result': {
                'passed': True,
                'score': 100.0
            },
            'data_consistency': {},
            'structural_validation': {},
            'style_enforcement': {},
            'content_quality': {
                'metadata_leakage': {},
                'content_integrity': {},
                'readability': {},
                'sensitive_info': {}
            }
        }
        
        # 数据一致性检查
        logger.info("开始数据一致性检查...")
        results['data_consistency'] = self.data_checker.check_consistency(report_data)
        
        # 结构验证
        logger.info("开始结构验证...")
        results['structural_validation'] = self.structural_validator.validate_structure(sections, report_config)
        
        # 风格检查
        logger.info("开始风格检查...")
        
        # 组合所有章节内容进行风格检查
        combined_content = ""
        for section in sections:
            combined_content += f"{section['content']}\n\n"
        
        results['style_enforcement'] = self.style_enforcer.enforce_style(combined_content, 'markdown')
        
        # 内容质量验证
        logger.info("开始内容质量验证...")
        
        # 元数据泄露检查
        logger.info("开始元数据泄露检查...")
        results['content_quality']['metadata_leakage'] = self.content_validator.check_metadata_leakage(combined_content)
        
        # 内容完整性检查
        logger.info("开始内容完整性检查...")
        results['content_quality']['content_integrity'] = self.content_validator.check_content_integrity(sections)
        
        # 可读性评估
        logger.info("开始可读性评估...")
        results['content_quality']['readability'] = self.content_validator.evaluate_readability(combined_content)
        
        # 敏感信息筛查
        logger.info("开始敏感信息筛查...")
        results['content_quality']['sensitive_info'] = self.content_validator.screen_sensitive_info(combined_content)
        
        # 计算整体得分（数据20%，结构25%，风格15%，内容质量40%）
        content_quality_score = (
            results['content_quality']['metadata_leakage']['score'] * 0.4 +
            results['content_quality']['content_integrity']['score'] * 0.2 +
            results['content_quality']['readability']['score'] * 0.2 +
            results['content_quality']['sensitive_info']['score'] * 0.2
        )
        
        overall_score = (
            results['data_consistency']['score'] * 0.2 +
            results['structural_validation']['score'] * 0.25 +
            results['style_enforcement']['score'] * 0.15 +
            content_quality_score * 0.4
        )
        
        results['overall_result']['score'] = round(overall_score, 2)
        
        # 整体结果
        if not (results['data_consistency']['passed'] and 
                results['structural_validation']['passed'] and 
                results['style_enforcement']['passed'] and
                results['content_quality']['metadata_leakage']['passed'] and
                results['content_quality']['sensitive_info']['passed']):
            results['overall_result']['passed'] = False
        
        return results
    
    def generate_quality_report(self, validation_results: Dict[str, Any]) -> str:
        """
        生成质量报告
        
        Args:
            validation_results: 质量验证结果
            
        Returns:
            质量报告内容
        """
        report = "# 质量保证报告\n\n"
        
        # 整体结果
        report += "## 整体结果\n"
        report += f"- 状态: {'通过' if validation_results['overall_result']['passed'] else '未通过'}\n"
        report += f"- 总分: {validation_results['overall_result']['score']}/100\n\n"
        
        # 数据一致性
        report += "## 数据一致性检查\n"
        report += f"- 得分: {validation_results['data_consistency']['score']}/100\n"
        report += f"- 问题数量: {len(validation_results['data_consistency']['issues'])}\n\n"
        
        if validation_results['data_consistency']['issues']:
            report += "### 发现的问题\n"
            for issue in validation_results['data_consistency']['issues']:
                report += f"- [{issue['severity']}] {issue['message']}\n"
            report += "\n"
        
        # 结构验证
        report += "## 结构验证\n"
        report += f"- 得分: {validation_results['structural_validation']['score']}/100\n"
        report += f"- 问题数量: {len(validation_results['structural_validation']['issues'])}\n\n"
        
        if validation_results['structural_validation']['issues']:
            report += "### 发现的问题\n"
            for issue in validation_results['structural_validation']['issues']:
                report += f"- [{issue['severity']}] {issue['message']}\n"
            report += "\n"
        
        # 风格检查
        report += "## 风格检查\n"
        report += f"- 得分: {validation_results['style_enforcement']['score']}/100\n"
        report += f"- 问题数量: {len(validation_results['style_enforcement']['issues'])}\n\n"
        
        if validation_results['style_enforcement']['issues']:
            report += "### 发现的问题\n"
            for issue in validation_results['style_enforcement']['issues']:
                report += f"- [{issue['severity']}] {issue['message']}\n"
            report += "\n"
        
        # 内容质量验证
        report += "## 内容质量验证\n"
        
        # 元数据泄露检查
        report += "### 元数据泄露检查\n"
        report += f"- 得分: {validation_results['content_quality']['metadata_leakage']['score']}/100\n"
        report += f"- 问题数量: {len(validation_results['content_quality']['metadata_leakage']['issues'])}\n\n"
        
        if validation_results['content_quality']['metadata_leakage']['issues']:
            report += "#### 发现的问题\n"
            for issue in validation_results['content_quality']['metadata_leakage']['issues']:
                report += f"- [{issue['severity']}] {issue['message']}\n"
            report += "\n"
        
        # 内容完整性检查
        report += "### 内容完整性检查\n"
        report += f"- 得分: {validation_results['content_quality']['content_integrity']['score']}/100\n"
        report += f"- 问题数量: {len(validation_results['content_quality']['content_integrity']['issues'])}\n\n"
        
        if validation_results['content_quality']['content_integrity']['issues']:
            report += "#### 发现的问题\n"
            for issue in validation_results['content_quality']['content_integrity']['issues']:
                report += f"- [{issue['severity']}] {issue['message']}\n"
            report += "\n"
        
        # 可读性评估
        report += "### 可读性评估\n"
        report += f"- 得分: {validation_results['content_quality']['readability']['score']}/100\n"
        report += f"- 问题数量: {len(validation_results['content_quality']['readability']['issues'])}\n"
        report += f"- 单词数: {validation_results['content_quality']['readability']['metrics']['word_count']}\n"
        report += f"- 句子数: {validation_results['content_quality']['readability']['metrics']['sentence_count']}\n"
        report += f"- 平均句子长度: {validation_results['content_quality']['readability']['metrics']['avg_sentence_length']} 词\n"
        report += f"- 平均单词长度: {validation_results['content_quality']['readability']['metrics']['avg_word_length']} 字符\n\n"
        
        if validation_results['content_quality']['readability']['issues']:
            report += "#### 发现的问题\n"
            for issue in validation_results['content_quality']['readability']['issues']:
                report += f"- [{issue['severity']}] {issue['message']}\n"
            report += "\n"
        
        # 敏感信息筛查
        report += "### 敏感信息筛查\n"
        report += f"- 得分: {validation_results['content_quality']['sensitive_info']['score']}/100\n"
        report += f"- 问题数量: {len(validation_results['content_quality']['sensitive_info']['issues'])}\n\n"
        
        if validation_results['content_quality']['sensitive_info']['issues']:
            report += "#### 发现的问题\n"
            for issue in validation_results['content_quality']['sensitive_info']['issues']:
                report += f"- [{issue['severity']}] {issue['message']}\n"
        
        return report
