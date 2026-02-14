#!/usr/bin/env python3
"""
内容生成协调器模块
负责协调规则引擎和AI生成器的并行工作流程
"""

from typing import Dict, Any, List, Optional
import logging
import concurrent.futures

logger = logging.getLogger(__name__)


class ContentGenerationCoordinator:
    """内容生成协调器，负责协调规则引擎和AI生成器的并行工作流程"""
    
    def __init__(self, rule_engine=None, ai_generator=None, fusion_strategy=None):
        """
        初始化内容生成协调器
        
        Args:
            rule_engine: 规则引擎实例
            ai_generator: AI生成器实例
            fusion_strategy: 内容融合策略实例
        """
        # 初始化规则引擎
        if rule_engine is None:
            from .rule_engine import RuleEngine
            self.rule_engine = RuleEngine()
        else:
            self.rule_engine = rule_engine
        
        # 初始化AI生成器
        if ai_generator is None:
            from .ai_generator import AIGenerator
            from src.infrastructure.utils.config_manager import config_manager
            use_mock = config_manager.get("nlp.use_mock", False)
            self.ai_generator = AIGenerator(mock_mode=use_mock)
        else:
            self.ai_generator = ai_generator
        
        # 初始化内容融合策略
        if fusion_strategy is None:
            from .content_fusion_strategy import ContentFusionStrategy
            self.fusion_strategy = ContentFusionStrategy()
        else:
            self.fusion_strategy = fusion_strategy
        
        logger.info("ContentGenerationCoordinator初始化完成")
    
    def generate_report_sections(self, data: Dict[str, Any], report_config: Dict[str, Any] = None, ai_enhancement: bool = True, max_workers: int = 2) -> List[Dict[str, Any]]:
        """
        生成报告章节
        
        Args:
            data: 报告数据
            report_config: 报告配置
            ai_enhancement: 是否使用AI增强
            max_workers: 并行处理的最大工作线程数
            
        Returns:
            生成的章节列表
        """
        # 步骤1: 使用规则引擎生成基础章节
        logger.info("步骤1: 使用规则引擎生成基础章节...")
        rule_based_sections = self.rule_engine.generate_sections(data, report_config)
        
        if not ai_enhancement or not rule_based_sections:
            logger.info("未启用AI增强或规则生成失败，返回规则生成的章节")
            return rule_based_sections
        
        # 步骤2: 使用AI生成器并行增强章节内容
        logger.info("步骤2: 使用AI生成器并行增强章节内容...")
        ai_enhanced_sections = self.ai_generator.generate_sections(
            rule_based_sections, 
            data, 
            strategy_type='insight_augmentation', 
            max_workers=max_workers
        )
        
        # 步骤3: 融合规则生成和AI生成的内容
        logger.info("步骤3: 融合规则生成和AI生成的内容...")
        fused_sections = self._fuse_generated_content(rule_based_sections, ai_enhanced_sections)
        
        # 步骤4: 添加章节衔接，确保逻辑连贯
        logger.info("步骤4: 添加章节衔接，确保逻辑连贯...")
        enhanced_sections = self.fusion_strategy.add_section_transitions(fused_sections)
        
        # 步骤5: 去重处理，确保内容不重复
        logger.info("步骤5: 去重处理，确保内容不重复...")
        for section in enhanced_sections:
            if 'content' in section:
                section['content'] = self.fusion_strategy.remove_duplicate_content(section['content'])
        
        logger.info("报告章节生成完成")
        return enhanced_sections
    
    def _fuse_generated_content(self, rule_based_sections: List[Dict[str, Any]], ai_enhanced_sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        融合规则生成和AI生成的内容
        
        Args:
            rule_based_sections: 规则生成的章节
            ai_enhanced_sections: AI增强的章节
            
        Returns:
            融合后的章节列表
        """
        fused_sections = []
        
        # 创建章节ID到章节的映射
        rule_based_map = {section['id']: section for section in rule_based_sections}
        ai_enhanced_map = {section['id']: section for section in ai_enhanced_sections}
        
        # 遍历所有章节，进行融合
        for section_id in rule_based_map.keys():
            rule_section = rule_based_map.get(section_id)
            ai_section = ai_enhanced_map.get(section_id)
            
            if not rule_section:
                continue
            
            if not ai_section or not ai_section.get('enhanced', False):
                # 如果没有AI增强或增强失败，使用规则生成的内容
                fused_sections.append(rule_section)
                continue
            
            # 使用内容融合策略融合内容
            fused_section = self._fuse_single_section(rule_section, ai_section)
            fused_sections.append(fused_section)
        
        return fused_sections
    
    def _fuse_single_section(self, rule_section: Dict[str, Any], ai_section: Dict[str, Any]) -> Dict[str, Any]:
        """
        融合单个章节的规则生成内容和AI生成内容
        
        Args:
            rule_section: 规则生成的章节
            ai_section: AI生成的章节
            
        Returns:
            融合后的章节
        """
        # 创建融合后的章节
        fused_section = {
            'id': rule_section['id'],
            'title': rule_section['title'],
            'importance': rule_section['importance'],
            'required': rule_section['required'],
            'source': 'fused',  # 标记内容来源为融合
            'enhanced': True
        }
        
        # 准备规则内容和AI内容
        rule_content = {
            'content': rule_section['content'],
            'source': 'rule_based'
        }
        
        ai_content = {
            'content': ai_section['content'],
            'source': 'ai_generated'
        }
        
        # 使用内容融合策略融合内容
        fusion_result = self.fusion_strategy.fuse_content(rule_content, ai_content, rule_section['id'])
        
        # 设置融合后的内容
        fused_section['content'] = fusion_result['content']
        fused_section['fusion_priority'] = fusion_result['priority']
        fused_section['fusion_source'] = fusion_result['source']
        
        logger.debug(f"章节 {rule_section['id']} 内容融合完成，使用融合策略: {fusion_result['priority']}")
        
        return fused_section
    
    def generate_incremental_sections(self, existing_sections: List[Dict[str, Any]], new_data: Dict[str, Any], section_ids: List[str] = None, ai_enhancement: bool = True) -> List[Dict[str, Any]]:
        """
        增量生成报告章节
        
        Args:
            existing_sections: 现有章节
            new_data: 新数据
            section_ids: 要重新生成的章节ID列表
            ai_enhancement: 是否使用AI增强
            
        Returns:
            更新后的章节列表
        """
        if not section_ids:
            logger.info("未指定章节ID，重新生成所有章节")
            return self.generate_report_sections(new_data, ai_enhancement=ai_enhancement)
        
        # 创建现有章节的映射
        existing_map = {section['id']: section for section in existing_sections}
        
        # 重新生成指定的章节
        new_sections = []
        for section_id in section_ids:
            if section_id in existing_map:
                # 如果章节存在，更新配置
                section_config = {
                    'id': section_id,
                    'title': existing_map[section_id]['title'],
                    'importance': existing_map[section_id]['importance'],
                    'required': existing_map[section_id]['required']
                }
                
                # 生成新章节
                new_section = self.rule_engine._generate_section(new_data, section_config)
                if new_section:
                    # 如果启用AI增强，增强新章节
                    if ai_enhancement:
                        enhanced_section = self.ai_generator._enhance_section(new_section, new_data)
                        new_sections.append(enhanced_section)
                    else:
                        new_sections.append(new_section)
        
        # 替换现有章节中的更新章节
        updated_sections = []
        for section in existing_sections:
            if section['id'] in section_ids:
                # 找到更新的章节
                updated_section = next((s for s in new_sections if s['id'] == section['id']), None)
                if updated_section:
                    updated_sections.append(updated_section)
                else:
                    updated_sections.append(section)  # 如果更新失败，保留原章节
            else:
                updated_sections.append(section)  # 保持原章节不变
        
        logger.info(f"成功增量更新 {len(new_sections)} 个章节")
        return updated_sections
    
    def set_ai_mock_mode(self, mock_mode: bool):
        """
        设置AI生成器的模拟模式
        
        Args:
            mock_mode: 是否使用模拟模式
        """
        self.ai_generator.mock_mode = mock_mode
        logger.info(f"AI生成器模拟模式设置为: {mock_mode}")
    
    def set_template_engine(self, template_engine):
        """
        设置模板引擎
        
        Args:
            template_engine: 模板引擎
        """
        self.rule_engine.set_template_engine(template_engine)
        logger.info("模板引擎设置完成")
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """
        获取内容生成统计信息
        
        Returns:
            统计信息
        """
        # 从AI生成器获取统计信息
        from .report_monitor import monitor
        stats = monitor.get_llm_call_stats()
        
        return stats
