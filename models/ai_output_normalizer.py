#!/usr/bin/env python3
"""
AI输出规范化模块：处理LLM生成的内容，确保格式正确，无元数据泄露
"""

import re
import json
import logging
from typing import Dict, Any, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class ContentQuality(Enum):
    """内容质量级别"""
    EXCELLENT = 5
    GOOD = 4
    FAIR = 3
    POOR = 2
    INVALID = 1


class AIOutputNormalizer:
    """
    AI输出规范化器：负责处理LLM生成的内容
    1. 元数据过滤：移除提示词、中间分析等内部数据
    2. 格式标准化：将JSON/字典转换为自然语言段落
    3. 内容提取：从AI响应中提取有效报告内容
    4. 质量评分：评估生成内容的相关性和可用性
    """
    
    def __init__(self):
        """初始化AI输出规范化器"""
        # 元数据模式匹配
        self.metadata_patterns = [
            # 匹配提示词标记
            re.compile(r'## 任务：.*?##', re.DOTALL),
            re.compile(r'## 数据上下文：.*?##', re.DOTALL),
            re.compile(r'## 分析要点：.*?##', re.DOTALL),
            re.compile(r'## 格式要求：.*?##', re.DOTALL),
            re.compile(r'## 可用数据（完整）：.*?##', re.DOTALL),
            
            # 匹配中间分析过程
            re.compile(r'\[分析中\].*?\[分析完成\]', re.DOTALL),
            re.compile(r'\{"analysis".*?\}', re.DOTALL),
            
            # 匹配系统提示词
            re.compile(r'\[SYSTEM\].*?\[END SYSTEM\]', re.DOTALL),
            re.compile(r'\[PROMPT\].*?\[END PROMPT\]', re.DOTALL),
        ]
        
        # 格式转换规则
        self.format_rules = {
            'json_to_text': self._json_to_natural_text,
            'dict_to_text': self._dict_to_natural_text,
            'markdown_cleanup': self._clean_markdown_formatting
        }
        
        logger.info("AIOutputNormalizer初始化完成")
    
    def normalize(self, content: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        规范化AI输出内容
        
        Args:
            content: AI生成的原始内容
            context: 上下文信息（用于质量评估）
            
        Returns:
            规范化后的结果，包含：
            - normalized_content: 规范化后的内容
            - quality_score: 质量评分
            - quality_level: 质量级别
            - original_content: 原始内容（可选）
            - metadata_removed: 是否移除了元数据
        """
        if not content:
            return {
                'normalized_content': '',
                'quality_score': 1,
                'quality_level': ContentQuality.INVALID,
                'metadata_removed': False
            }
        
        result = {
            'original_content': content,
            'normalized_content': content,
            'metadata_removed': False,
            'format_converted': False
        }
        
        # 步骤1: 元数据过滤
        filtered_content = self._filter_metadata(content)
        if filtered_content != content:
            result['normalized_content'] = filtered_content
            result['metadata_removed'] = True
            logger.debug("成功移除元数据")
        
        # 步骤2: 格式标准化
        normalized_content = self._standardize_format(result['normalized_content'])
        if normalized_content != result['normalized_content']:
            result['normalized_content'] = normalized_content
            result['format_converted'] = True
            logger.debug("成功标准化格式")
        
        # 步骤3: 内容提取（如果需要）
        extracted_content = self._extract_valid_content(result['normalized_content'])
        if extracted_content and extracted_content != result['normalized_content']:
            result['normalized_content'] = extracted_content
            logger.debug("成功提取有效内容")
        
        # 步骤4: 质量评分
        quality_score = self._evaluate_quality(result['normalized_content'], context)
        result['quality_score'] = quality_score
        result['quality_level'] = self._get_quality_level(quality_score)
        
        logger.info(f"内容规范化完成，质量评分: {quality_score}, 质量级别: {result['quality_level'].name}")
        
        return result
    
    def _filter_metadata(self, content: str) -> str:
        """
        过滤内容中的元数据
        
        Args:
            content: AI生成的内容
            
        Returns:
            过滤后的内容
        """
        filtered_content = content
        
        # 应用所有元数据过滤模式
        for pattern in self.metadata_patterns:
            filtered_content = pattern.sub('', filtered_content)
        
        # 清理多余的空白字符
        filtered_content = re.sub(r'\n{3,}', '\n\n', filtered_content)
        filtered_content = filtered_content.strip()
        
        return filtered_content
    
    def _standardize_format(self, content: str) -> str:
        """
        标准化内容格式
        
        Args:
            content: AI生成的内容
            
        Returns:
            标准化后的内容
        """
        standardized_content = content
        
        # 检测内容类型并应用相应的转换规则
        if self._is_json(content):
            standardized_content = self.format_rules['json_to_text'](content)
        elif self._is_dict_like(content):
            standardized_content = self.format_rules['dict_to_text'](content)
        
        # 清理Markdown格式
        standardized_content = self.format_rules['markdown_cleanup'](standardized_content)
        
        return standardized_content
    
    def _extract_valid_content(self, content: str) -> Optional[str]:
        """
        从AI响应中提取有效报告内容
        
        Args:
            content: AI生成的内容
            
        Returns:
            提取的有效内容，如果没有提取到则返回None
        """
        # 查找主要内容部分
        main_content_patterns = [
            re.compile(r'###.*?\n(.*?)(?=###|$)', re.DOTALL),  # 提取Markdown章节
            re.compile(r'\d+\.\s+(.*?)(?=\d+\.\s+|$)', re.DOTALL),  # 提取列表项
            re.compile(r'\*\s+(.*?)(?=\*\s+|$)', re.DOTALL),  # 提取项目符号
        ]
        
        for pattern in main_content_patterns:
            matches = pattern.findall(content)
            if matches:
                # 合并提取的内容
                extracted = '\n'.join([match.strip() for match in matches])
                return extracted if extracted else None
        
        # 如果没有找到特定模式，返回清理后的内容
        return content
    
    def _evaluate_quality(self, content: str, context: Dict[str, Any] = None) -> int:
        """
        评估内容质量
        
        Args:
            content: 要评估的内容
            context: 上下文信息
            
        Returns:
            质量评分（1-5）
        """
        if not content:
            return 1
        
        score = 5  # 初始评分
        
        # 1. 长度评估
        if len(content) < 100:
            score -= 2
        elif len(content) < 300:
            score -= 1
        
        # 2. 内容相关性评估
        if context and 'expected_topics' in context:
            missing_topics = 0
            for topic in context['expected_topics']:
                if topic not in content:
                    missing_topics += 1
            
            if missing_topics > len(context['expected_topics']) / 2:
                score -= 2
            elif missing_topics > 0:
                score -= 1
        
        # 3. 格式完整性评估
        if not any(marker in content for marker in ['#', '##', '###', '- ', '* ']):
            score -= 1
        
        # 4. 语言流畅性评估
        if re.search(r'\.{3,}|\?{3,}|!{3,}', content):  # 过多标点符号
            score -= 1
        
        if re.search(r'\b(?:\w+\s+){5,}\b', content):  # 过长的连续单词
            score -= 1
        
        # 确保评分在1-5范围内
        return max(1, min(5, score))
    
    def _get_quality_level(self, score: int) -> ContentQuality:
        """
        根据评分获取质量级别
        
        Args:
            score: 质量评分
            
        Returns:
            质量级别
        """
        if score == 5:
            return ContentQuality.EXCELLENT
        elif score == 4:
            return ContentQuality.GOOD
        elif score == 3:
            return ContentQuality.FAIR
        elif score == 2:
            return ContentQuality.POOR
        else:
            return ContentQuality.INVALID
    
    def _json_to_natural_text(self, content: str) -> str:
        """
        将JSON格式转换为自然语言文本
        
        Args:
            content: JSON格式的内容
            
        Returns:
            自然语言文本
        """
        try:
            data = json.loads(content)
            return self._dict_to_natural_text(str(data))
        except json.JSONDecodeError:
            return content
    
    def _dict_to_natural_text(self, content: str) -> str:
        """
        将字典格式转换为自然语言文本
        
        Args:
            content: 字典格式的内容
            
        Returns:
            自然语言文本
        """
        # 移除字典的大括号
        text = content.strip('{}')
        
        # 转换键值对为自然语言
        lines = []
        for item in text.split(','):
            if ':' in item:
                key, value = item.split(':', 1)
                key = key.strip().strip("'\"").replace('_', ' ').capitalize()
                value = value.strip().strip("'\"")
                lines.append(f"{key}: {value}")
        
        return '\n'.join(lines)
    
    def _clean_markdown_formatting(self, content: str) -> str:
        """
        清理Markdown格式，确保格式一致
        
        Args:
            content: Markdown格式的内容
            
        Returns:
            清理后的内容
        """
        # 修复标题格式
        content = re.sub(r'\n\s*#', '\n#', content)
        
        # 修复列表格式
        content = re.sub(r'\n\s*(-|\*)', '\n-', content)
        
        # 清理多余的空行
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # 清理格式标记
        content = re.sub(r'\[.*?\]', '', content)
        
        return content.strip()
    
    def _is_json(self, content: str) -> bool:
        """
        检查内容是否为JSON格式
        
        Args:
            content: 要检查的内容
            
        Returns:
            如果是JSON格式返回True，否则返回False
        """
        content = content.strip()
        return content.startswith('{') and content.endswith('}') or content.startswith('[') and content.endswith(']')
    
    def _is_dict_like(self, content: str) -> bool:
        """
        检查内容是否为字典类似格式
        
        Args:
            content: 要检查的内容
            
        Returns:
            如果是字典类似格式返回True，否则返回False
        """
        content = content.strip()
        return content.startswith('{') and content.endswith('}') and ':' in content
    
    def get_supported_formats(self) -> List[str]:
        """
        获取支持的格式列表
        
        Returns:
            支持的格式列表
        """
        return list(self.format_rules.keys())


# 测试代码
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.DEBUG)
    
    # 创建规范化器实例
    normalizer = AIOutputNormalizer()
    
    # 测试数据
    test_content = """
## 任务：生成报告执行摘要
## 数据上下文：
价格数据：从1900美元到1950美元
统计分析：包含基本统计指标、波动率分析和趋势分析

## 分析要点：
1. 总结主要发现
2. 强调关键洞察
3. 提供简明的结论

## 格式要求：
使用简洁明了的语言，不超过500字

## 可用数据（完整）：
{"price_data": {"start_price": 1900, "end_price": 1950, "unit": "美元"}, "statistical_analysis": {...}}

## 请基于以上信息生成报告内容

### 执行摘要
黄金价格本周呈现上涨趋势，从1900美元上涨至1950美元，涨幅约2.63%。市场分析显示，美联储加息预期减弱是推动价格上涨的主要因素。投资者风险偏好下降，黄金作为避险资产的需求增加。建议密切关注下周的CPI数据发布，这可能会对金价产生重大影响。
"""
    
    # 测试规范化
    print("=== 测试AI输出规范化 ===")
    print("原始内容：")
    print(test_content)
    
    result = normalizer.normalize(test_content, context={"expected_topics": ["黄金价格", "上涨趋势", "美联储", "CPI数据"]})
    
    print("\n规范化后内容：")
    print(result['normalized_content'])
    
    print("\n质量评估：")
    print(f"质量评分：{result['quality_score']}/5")
    print(f"质量级别：{result['quality_level'].name}")
    print(f"元数据移除：{result['metadata_removed']}")
    print(f"格式转换：{result['format_converted']}")
