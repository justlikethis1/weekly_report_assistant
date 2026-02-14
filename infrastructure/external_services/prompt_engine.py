import os
import json
import logging
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from jinja2 import Template, Environment, FileSystemLoader
from src.infrastructure.utils.config_manager import config_manager

# 配置日志记录
logger = logging.getLogger(__name__)

# 提示词类型枚举
class PromptType:
    SYSTEM = "system"
    REPORT_GENERATION = "report_generation"
    FILE_PROCESSING = "file_processing"
    CONTEXT_MANAGEMENT = "context_management"
    UI_INTERACTION = "ui_interaction"
    QUALITY_ASSESSMENT = "quality_assessment"
    FORMAT_PROCESSING = "format_processing"

# 提示词模板元数据
@dataclass
class PromptMetadata:
    name: str
    type: str
    description: str
    version: str
    language: str
    template_path: str

# 提示词优化建议
@dataclass
class PromptOptimizationSuggestion:
    issue: str
    suggestion: str
    priority: str  # low, medium, high
    impact: str    # low, medium, high

# 抽象提示词生成器
class AbstractPromptGenerator(ABC):
    @abstractmethod
    def generate_prompt(self, prompt_type: str, context: Dict[str, Any] = None) -> str:
        pass

# 基础提示词生成器
class BasePromptGenerator(AbstractPromptGenerator):
    def __init__(self, template_dir: str = None):
        self.template_dir = template_dir or os.path.join(os.path.dirname(__file__), "../templates/prompts")
        self.env = Environment(loader=FileSystemLoader(self.template_dir))
        self.prompt_metadata = self._load_prompt_metadata()
    
    def _load_prompt_metadata(self) -> Dict[str, PromptMetadata]:
        """加载提示词模板元数据"""
        metadata_file = os.path.join(self.template_dir, "metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata_data = json.load(f)
                return {k: PromptMetadata(**v) for k, v in metadata_data.items()}
        return {}
    
    def generate_prompt(self, prompt_type: str, context: Dict[str, Any] = None) -> str:
        """生成基础提示词"""
        if context is None:
            context = {}
            
        template_path = self._get_template_path(prompt_type)
        if not template_path:
            logger.error(f"提示词模板不存在: {prompt_type}")
            return ""
            
        try:
            template = self.env.get_template(template_path)
            return template.render(**context)
        except Exception as e:
            logger.error(f"生成提示词失败: {e}")
            return ""
    
    def _get_template_path(self, prompt_type: str) -> Optional[str]:
        """获取模板文件路径"""
        # 默认模板映射
        template_mapping = {
            PromptType.SYSTEM: "system_prompt.j2",
            PromptType.REPORT_GENERATION: "report_generation_prompt.j2",
            PromptType.FILE_PROCESSING: "file_processing_prompt.j2",
            PromptType.CONTEXT_MANAGEMENT: "context_management_prompt.j2",
            PromptType.UI_INTERACTION: "ui_interaction_prompt.j2",
            PromptType.QUALITY_ASSESSMENT: "quality_assessment_prompt.j2",
            PromptType.FORMAT_PROCESSING: "format_processing_prompt.j2",
        }
        return template_mapping.get(prompt_type)

# 智能提示词生成器
class SmartPromptGenerator(BasePromptGenerator):
    def __init__(self, template_dir: str = None):
        super().__init__(template_dir)
        self.context_enhancer = ContextEnhancer()
        self.prompt_optimizer = PromptOptimizer()
    
    def generate_prompt(self, prompt_type: str, context: Dict[str, Any] = None) -> str:
        """生成智能提示词"""
        if context is None:
            context = {}
            
        # 增强上下文
        enhanced_context = self.context_enhancer.enhance_context(context, prompt_type)
        
        # 生成基础提示词
        base_prompt = super().generate_prompt(prompt_type, enhanced_context)
        
        # 优化提示词
        optimized_prompt = self.prompt_optimizer.optimize_prompt(base_prompt, prompt_type, context)
        
        return optimized_prompt

# 上下文增强器
class ContextEnhancer:
    def enhance_context(self, context: Dict[str, Any], prompt_type: str) -> Dict[str, Any]:
        """增强提示词上下文"""
        enhanced = context.copy()
        
        # 添加配置信息
        enhanced["config"] = {
            "model_name": config_manager.get("model.name"),
            "report_type": config_manager.get("ui.title"),
            "max_response_length": config_manager.get("memory.max_ai_response_length"),
        }
        
        # 根据提示词类型添加特定信息
        if prompt_type == PromptType.REPORT_GENERATION:
            enhanced = self._enhance_report_generation_context(enhanced)
        elif prompt_type == PromptType.FILE_PROCESSING:
            enhanced = self._enhance_file_processing_context(enhanced)
        elif prompt_type == PromptType.UI_INTERACTION:
            enhanced = self._enhance_ui_interaction_context(enhanced)
        
        return enhanced
    
    def _enhance_report_generation_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """增强报告生成上下文"""
        # 添加报告类型默认值
        if "report_type" not in context:
            context["report_type"] = "自定义报告"
        
        # 添加受众默认值
        if "audience" not in context:
            context["audience"] = "高管层"
        
        # 添加分析深度默认值
        if "depth" not in context:
            context["depth"] = "中等"
        
        return context
    
    def _enhance_file_processing_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """增强文件处理上下文"""
        # 添加文件类型默认值
        if "file_type" not in context:
            context["file_type"] = "文本文件"
        
        return context
    
    def _enhance_ui_interaction_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """增强UI交互上下文"""
        # 添加交互阶段默认值
        if "interaction_stage" not in context:
            context["interaction_stage"] = "welcome"
        
        return context

# 提示词优化器
class PromptOptimizer:
    def optimize_prompt(self, prompt: str, prompt_type: str, context: Dict[str, Any]) -> str:
        """优化提示词"""
        if not prompt:
            return prompt
        
        # 根据提示词类型应用不同的优化策略
        if prompt_type == PromptType.REPORT_GENERATION:
            return self._optimize_report_generation_prompt(prompt, context)
        elif prompt_type == PromptType.FILE_PROCESSING:
            return self._optimize_file_processing_prompt(prompt, context)
        elif prompt_type == PromptType.SYSTEM:
            return self._optimize_system_prompt(prompt, context)
        
        return prompt
    
    def _optimize_report_generation_prompt(self, prompt: str, context: Dict[str, Any]) -> str:
        """优化报告生成提示词"""
        # 调整提示词长度
        max_length = config_manager.get("memory.max_ai_response_length") * 0.3  # 提示词长度不应超过响应长度的30%
        if len(prompt) > max_length:
            # 保留核心内容，移除次要信息
            core_sections = ["执行摘要", "数据概览", "核心分析", "关键发现", "结论建议"]
            optimized_prompt = self._extract_core_sections(prompt, core_sections)
            if optimized_prompt:
                return optimized_prompt
        
        return prompt
    
    def _optimize_file_processing_prompt(self, prompt: str, context: Dict[str, Any]) -> str:
        """优化文件处理提示词"""
        # 根据文件类型调整提示词
        file_type = context.get("file_type", "").lower()
        if file_type == "excel" or file_type == "csv":
            # 增强表格处理能力
            prompt += "\n\n特别注意：请详细提取表格中的所有数据，包括表头、数据行和汇总信息。"
        elif file_type == "pdf":
            # 增强PDF处理能力
            prompt += "\n\n特别注意：请提取PDF中的所有文本内容和表格数据，保持页面结构和内容顺序。"
        
        return prompt
    
    def _optimize_system_prompt(self, prompt: str, context: Dict[str, Any]) -> str:
        """优化系统提示词"""
        # 移除重复内容
        lines = prompt.split("\n")
        seen = set()
        unique_lines = []
        for line in lines:
            stripped_line = line.strip()
            if stripped_line and stripped_line not in seen:
                seen.add(stripped_line)
                unique_lines.append(line)
        
        return "\n".join(unique_lines)
    
    def _extract_core_sections(self, prompt: str, core_sections: List[str]) -> str:
        """提取核心章节"""
        optimized_prompt = []
        current_section = None
        
        for line in prompt.split("\n"):
            line_stripped = line.strip()
            
            # 检查是否是核心章节标题
            for section in core_sections:
                if section in line_stripped:
                    current_section = section
                    optimized_prompt.append(line)
                    break
            
            # 如果是当前核心章节的内容，添加到优化后的提示词中
            if current_section is not None:
                optimized_prompt.append(line)
        
        return "\n".join(optimized_prompt) if optimized_prompt else prompt

# 提示词评估器
class PromptEvaluator:
    def evaluate_prompt(self, prompt: str, prompt_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """评估提示词质量"""
        evaluation = {
            "prompt_type": prompt_type,
            "length": len(prompt),
            "complexity": self._calculate_complexity(prompt),
            "optimization_suggestions": self._generate_optimization_suggestions(prompt, prompt_type, context),
            "quality_score": self._calculate_quality_score(prompt, prompt_type, context),
        }
        return evaluation
    
    def _calculate_complexity(self, prompt: str) -> str:
        """计算提示词复杂度"""
        word_count = len(prompt.split())
        if word_count < 500:
            return "low"
        elif word_count < 2000:
            return "medium"
        else:
            return "high"
    
    def _generate_optimization_suggestions(self, prompt: str, prompt_type: str, context: Dict[str, Any]) -> List[PromptOptimizationSuggestion]:
        """生成优化建议"""
        suggestions = []
        
        # 检查提示词长度
        if len(prompt) > 10000:
            suggestions.append(PromptOptimizationSuggestion(
                issue="提示词过长",
                suggestion="移除重复内容和非核心信息，保持提示词简洁",
                priority="high",
                impact="high"
            ))
        
        # 检查是否包含必要的上下文变量
        required_variables = self._get_required_variables(prompt_type)
        for var in required_variables:
            if f"{{{{{var}}}}}" in prompt and var not in context:
                suggestions.append(PromptOptimizationSuggestion(
                    issue=f"缺少必要的上下文变量: {var}",
                    suggestion=f"请提供{var}变量的值",
                    priority="medium",
                    impact="medium"
                ))
        
        return suggestions
    
    def _get_required_variables(self, prompt_type: str) -> List[str]:
        """获取提示词所需的必要变量"""
        variable_mapping = {
            PromptType.REPORT_GENERATION: ["file_contents", "user_input"],
            PromptType.FILE_PROCESSING: ["file_type"],
            PromptType.CONTEXT_MANAGEMENT: ["conversation_history"],
        }
        return variable_mapping.get(prompt_type, [])
    
    def _calculate_quality_score(self, prompt: str, prompt_type: str, context: Dict[str, Any]) -> float:
        """计算提示词质量分数（0-100）"""
        score = 100
        
        # 长度检查
        if len(prompt) > 10000:
            score -= 20
        elif len(prompt) < 500:
            score -= 10
        
        # 必要变量检查
        required_variables = self._get_required_variables(prompt_type)
        for var in required_variables:
            if f"{{{{{var}}}}}" in prompt and var not in context:
                score -= 15
        
        # 复杂度检查
        complexity = self._calculate_complexity(prompt)
        if complexity == "high" and prompt_type != PromptType.SYSTEM:
            score -= 10
        
        return max(0, min(100, score))

# 提示词工程系统
class PromptEngineeringSystem:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PromptEngineeringSystem, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        self.generator = SmartPromptGenerator()
        self.evaluator = PromptEvaluator()
        self._ensure_template_directory_exists()
    
    def _ensure_template_directory_exists(self):
        """确保模板目录存在"""
        template_dir = os.path.join(os.path.dirname(__file__), "../templates/prompts")
        os.makedirs(template_dir, exist_ok=True)
        
        # 创建示例模板文件
        example_templates = {
            "system_prompt.j2": "您是一位专业的商业数据分析师和战略顾问，专门负责从原始数据中提取洞察并生成深度商业报告。",
            "report_generation_prompt.j2": "请根据提供的用户上传文件内容和文字要求，生成一份结构清晰、内容专业的中文商业报告。\n\n用户上传的文件内容：{{ file_contents }}\n\n用户的文字要求：{{ user_input }}",
            "file_processing_prompt.j2": "请根据用户上传的文件类型，提取其中的关键信息。\n\n用户上传的文件类型：{{ file_type }}",
        }
        
        for template_name, content in example_templates.items():
            template_path = os.path.join(template_dir, template_name)
            if not os.path.exists(template_path):
                with open(template_path, "w", encoding="utf-8") as f:
                    f.write(content)
    
    def generate_prompt(self, prompt_type: str, context: Dict[str, Any] = None) -> str:
        """生成提示词"""
        return self.generator.generate_prompt(prompt_type, context)
    
    def evaluate_prompt(self, prompt: str, prompt_type: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """评估提示词"""
        return self.evaluator.evaluate_prompt(prompt, prompt_type, context)
    
    def optimize_prompt(self, prompt: str, prompt_type: str, context: Dict[str, Any] = None) -> str:
        """优化提示词"""
        evaluator = PromptEvaluator()
        optimizer = PromptOptimizer()
        return optimizer.optimize_prompt(prompt, prompt_type, context)
    
    def generate_and_evaluate(self, prompt_type: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """生成并评估提示词"""
        prompt = self.generate_prompt(prompt_type, context)
        evaluation = self.evaluate_prompt(prompt, prompt_type, context)
        return {
            "prompt": prompt,
            "evaluation": evaluation
        }

# 创建全局提示词工程系统实例
prompt_engineering_system = PromptEngineeringSystem()
