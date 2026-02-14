#!/usr/bin/env python3
"""
内容融合策略模块：结合规则生成的内容和AI生成的内容
"""

import re
import logging
from typing import Dict, Any, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class ContentPriority(Enum):
    """内容优先级"""
    RULE_BASED = 1  # 规则生成的内容优先级高
    AI_GENERATED = 2  # AI生成的内容优先级高
    MERGED = 3  # 合并内容，保持两者的优势


class ContentFusionStrategy:
    """
    内容融合策略：负责合并不同来源的内容
    1. 优先级规则：定义不同内容来源的优先级顺序
    2. 去重机制：识别并移除重复内容
    3. 衔接处理：确保各章节间逻辑连贯
    4. 冲突解决：处理规则与AI生成内容间的矛盾
    """
    
    def __init__(self):
        """初始化内容融合策略"""
        # 内容来源优先级规则
        self.priority_rules = {
            'executive_summary': ContentPriority.MERGED,
            'data_overview': ContentPriority.RULE_BASED,
            'statistical_analysis': ContentPriority.RULE_BASED,
            'insights_and_recommendations': ContentPriority.AI_GENERATED
        }
        
        # 章节衔接模板
        self.transition_templates = {
            'executive_summary_to_data_overview': "\n基于以上摘要，让我们详细分析数据：\n",
            'data_overview_to_statistical_analysis': "\n接下来进行深入的统计分析：\n",
            'statistical_analysis_to_insights': "\n根据统计分析，我们得出以下洞察：\n"
        }
        
        logger.info("ContentFusionStrategy初始化完成")
    
    def fuse_content(self, rule_based_content: Dict[str, Any], ai_generated_content: Dict[str, Any], section_id: str) -> Dict[str, Any]:
        """
        融合规则生成的内容和AI生成的内容
        
        Args:
            rule_based_content: 规则生成的内容
            ai_generated_content: AI生成的内容
            section_id: 章节ID
            
        Returns:
            融合后的内容
        """
        if not rule_based_content and not ai_generated_content:
            return {
                'content': '',
                'fused': False,
                'source': 'none'
            }
        
        # 获取该章节的优先级规则
        priority = self.priority_rules.get(section_id, ContentPriority.MERGED)
        
        # 根据优先级规则融合内容
        if priority == ContentPriority.RULE_BASED or not ai_generated_content:
            # 规则生成的内容优先级高
            result = self._rule_based_priority_fusion(rule_based_content, ai_generated_content)
        elif priority == ContentPriority.AI_GENERATED or not rule_based_content:
            # AI生成的内容优先级高
            result = self._ai_generated_priority_fusion(rule_based_content, ai_generated_content)
        else:
            # 合并内容
            result = self._merged_fusion(rule_based_content, ai_generated_content)
        
        result['section_id'] = section_id
        result['priority'] = priority.name
        
        return result
    
    def _rule_based_priority_fusion(self, rule_based_content: Dict[str, Any], ai_generated_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        规则生成内容优先的融合策略
        
        Args:
            rule_based_content: 规则生成的内容
            ai_generated_content: AI生成的内容
            
        Returns:
            融合后的内容
        """
        result = {
            'content': rule_based_content.get('content', ''),
            'fused': False,
            'source': 'rule_based'
        }
        
        if ai_generated_content:
            ai_content = ai_generated_content.get('content', '')
            if ai_content:
                # 尝试在规则内容后添加AI生成的额外洞察
                additional_insights = self._extract_additional_insights(ai_content, rule_based_content.get('content', ''))
                if additional_insights:
                    result['content'] += f"\n\n## 额外洞察\n{additional_insights}"
                    result['fused'] = True
                    result['source'] = 'rule_based_with_ai_insights'
                    logger.debug("成功在规则内容中添加AI生成的额外洞察")
        
        return result
    
    def _ai_generated_priority_fusion(self, rule_based_content: Dict[str, Any], ai_generated_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        AI生成内容优先的融合策略
        
        Args:
            rule_based_content: 规则生成的内容
            ai_generated_content: AI生成的内容
            
        Returns:
            融合后的内容
        """
        result = {
            'content': ai_generated_content.get('content', ''),
            'fused': False,
            'source': 'ai_generated'
        }
        
        if rule_based_content:
            rule_content = rule_based_content.get('content', '')
            if rule_content:
                # 验证AI内容的准确性，补充规则内容中的关键数据
                validated_content = self._validate_ai_content_with_rule_data(result['content'], rule_content)
                if validated_content != result['content']:
                    result['content'] = validated_content
                    result['fused'] = True
                    result['source'] = 'ai_generated_with_rule_validation'
                    logger.debug("成功使用规则内容验证并补充AI生成的内容")
        
        return result
    
    def _merged_fusion(self, rule_based_content: Dict[str, Any], ai_generated_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并内容的融合策略
        
        Args:
            rule_based_content: 规则生成的内容
            ai_generated_content: AI生成的内容
            
        Returns:
            融合后的内容
        """
        rule_content = rule_based_content.get('content', '')
        ai_content = ai_generated_content.get('content', '')
        
        if not rule_content:
            return {
                'content': ai_content,
                'fused': False,
                'source': 'ai_generated'
            }
        
        if not ai_content:
            return {
                'content': rule_content,
                'fused': False,
                'source': 'rule_based'
            }
        
        # 合并内容：规则内容提供结构和数据，AI内容提供洞察和解释
        merged_content = self._merge_content_structure(rule_content, ai_content)
        
        return {
            'content': merged_content,
            'fused': True,
            'source': 'merged'
        }
    
    def _extract_additional_insights(self, ai_content: str, rule_content: str) -> str:
        """
        从AI生成的内容中提取规则内容中没有的额外洞察
        
        Args:
            ai_content: AI生成的内容
            rule_content: 规则生成的内容
            
        Returns:
            提取的额外洞察
        """
        # 简单实现：查找AI内容中与规则内容不同的部分
        # 实际应用中可以使用更复杂的自然语言处理技术
        ai_sentences = [sentence.strip() for sentence in re.split(r'[。！？\n]', ai_content) if sentence.strip()]
        rule_sentences = [sentence.strip() for sentence in re.split(r'[。！？\n]', rule_content) if sentence.strip()]
        
        additional_insights = []
        for ai_sentence in ai_sentences:
            is_additional = True
            for rule_sentence in rule_sentences:
                if ai_sentence in rule_sentence or rule_sentence in ai_sentence:
                    is_additional = False
                    break
            if is_additional:
                additional_insights.append(ai_sentence)
        
        return '\n'.join(additional_insights)
    
    def _validate_ai_content_with_rule_data(self, ai_content: str, rule_content: str) -> str:
        """
        使用规则内容中的数据验证AI内容的准确性
        
        Args:
            ai_content: AI生成的内容
            rule_content: 规则生成的内容
            
        Returns:
            验证后的内容
        """
        # 提取规则内容中的关键数据
        data_patterns = [
            r'\d+(\.\d+)?\s*\w+',  # 数字+单位
            r'\b(?:平均值|最大值|最小值|中位数|标准差)\s*[:：]\s*\d+(\.\d+)?',  # 统计指标
            r'\b\d{4}-\d{2}-\d{2}\b'  # 日期
        ]
        
        key_data = []
        for pattern in data_patterns:
            matches = re.findall(pattern, rule_content)
            key_data.extend(matches)
        
        # 验证AI内容中是否包含这些关键数据
        validated_content = ai_content
        missing_data = []
        
        for data in key_data:
            if data not in validated_content:
                missing_data.append(data)
        
        # 如果有缺失的数据，添加到AI内容中
        if missing_data:
            validated_content += f"\n\n## 补充数据\n"
            for data in missing_data:
                validated_content += f"- {data}\n"
            
        return validated_content
    
    def _merge_content_structure(self, rule_content: str, ai_content: str) -> str:
        """
        合并内容结构
        
        Args:
            rule_content: 规则生成的内容
            ai_content: AI生成的内容
            
        Returns:
            合并后的内容
        """
        # 提取规则内容的结构（标题）
        rule_headings = re.findall(r'^(#+)\s*(.+)', rule_content, re.MULTILINE)
        
        if not rule_headings:
            # 如果规则内容没有结构，直接合并
            return f"{rule_content}\n\n{ai_content}"
        
        # 合并内容：保留规则内容的结构，填充AI内容的洞察
        merged_content = ""
        last_heading_level = 0
        
        for heading_level, heading_text in rule_headings:
            # 添加标题
            merged_content += f"\n{heading_level} {heading_text}\n"
            
            # 提取标题下的内容
            content_pattern = re.compile(r'^' + re.escape(heading_level) + r'\s*' + re.escape(heading_text) + r'\n(.*?)(?=^#+\s|$)', re.DOTALL | re.MULTILINE)
            content_match = content_pattern.search(rule_content)
            
            if content_match:
                # 添加规则内容
                merged_content += f"{content_match.group(1)}"
                
                # 根据标题查找AI内容中的对应部分
                ai_section_pattern = re.compile(r'.*' + re.escape(heading_text) + r'.*?\n(.*?)(?=\n###|$)', re.DOTALL | re.IGNORECASE)
                ai_section_match = ai_section_pattern.search(ai_content)
                
                if ai_section_match:
                    # 添加AI内容中的对应部分
                    merged_content += f"\n\n## AI洞察\n{ai_section_match.group(1)}"
        
        return merged_content.strip()
    
    def add_section_transitions(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        为章节添加衔接内容
        
        Args:
            sections: 章节列表
            
        Returns:
            添加衔接内容后的章节列表
        """
        if len(sections) <= 1:
            return sections
        
        enhanced_sections = []
        
        for i in range(len(sections)):
            section = sections[i].copy()
            
            # 为非第一个章节添加衔接内容
            if i > 0:
                previous_section = sections[i-1]
                transition_key = f"{previous_section['id']}_to_{section['id']}"
                
                if transition_key in self.transition_templates:
                    section['content'] = f"{self.transition_templates[transition_key]}{section['content']}"
                    section['has_transition'] = True
                    logger.debug(f"为章节 {previous_section['id']} -> {section['id']} 添加衔接内容")
            
            enhanced_sections.append(section)
        
        return enhanced_sections
    
    def remove_duplicate_content(self, content: str) -> str:
        """
        移除重复的内容
        
        Args:
            content: 要处理的内容
            
        Returns:
            移除重复内容后的文本
        """
        if not content:
            return ""
        
        # 按句子分割内容
        sentences = [sentence.strip() for sentence in re.split(r'[。！？\n]', content) if sentence.strip()]
        
        # 移除重复句子
        seen = set()
        unique_sentences = []
        
        for sentence in sentences:
            if sentence not in seen:
                seen.add(sentence)
                unique_sentences.append(sentence)
        
        # 重新组合内容
        return '\n'.join(unique_sentences)
    
    def set_priority_rule(self, section_id: str, priority: ContentPriority):
        """
        设置章节的优先级规则
        
        Args:
            section_id: 章节ID
            priority: 优先级
        """
        self.priority_rules[section_id] = priority
        logger.info(f"为章节 {section_id} 设置优先级规则: {priority.name}")
    
    def get_priority_rules(self) -> Dict[str, ContentPriority]:
        """
        获取所有章节的优先级规则
        
        Returns:
            章节优先级规则
        """
        return self.priority_rules


# 测试代码
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.DEBUG)
    
    # 创建内容融合策略实例
    fusion_strategy = ContentFusionStrategy()
    
    # 测试数据
    rule_based_content = {
        'content': '# 统计分析\n\n## 基本统计指标\n- 平均值: 1925.5美元\n- 最大值: 1950美元\n- 最小值: 1900美元\n- 标准差: 20.3美元\n',
        'source': 'rule_based'
    }
    
    ai_generated_content = {
        'content': '# 统计分析\n\n## 基本统计指标\n黄金价格的平均值为1925.5美元，最大值达到1950美元，最小值为1900美元。\n\n## 洞察\n从统计数据可以看出，黄金价格本周呈现稳定上涨趋势，波动率较低，表明市场情绪相对稳定。\n',
        'source': 'ai_generated'
    }
    
    # 测试融合功能
    print("=== 测试内容融合策略 ===")
    
    # 测试规则优先融合
    print("\n1. 规则优先融合：")
    result = fusion_strategy._rule_based_priority_fusion(rule_based_content, ai_generated_content)
    print(result['content'])
    
    # 测试AI优先融合
    print("\n2. AI优先融合：")
    result = fusion_strategy._ai_generated_priority_fusion(rule_based_content, ai_generated_content)
    print(result['content'])
    
    # 测试合并融合
    print("\n3. 合并融合：")
    result = fusion_strategy._merged_fusion(rule_based_content, ai_generated_content)
    print(result['content'])
    
    # 测试去重功能
    print("\n4. 去重功能：")
    duplicate_content = "黄金价格上涨。黄金价格上涨。黄金价格本周呈现稳定趋势。"
    print(f"原始内容：{duplicate_content}")
    print(f"去重后：{fusion_strategy.remove_duplicate_content(duplicate_content)}")
