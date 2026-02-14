#!/usr/bin/env python3
"""
动态提示词生成器：基于意图和文档生成优化提示
"""

from typing import Dict, Any, List
import logging
import json

# 配置日志
logger = logging.getLogger(__name__)

class PromptEnhancer:
    """动态提示词生成器：基于意图和文档生成优化提示"""
    
    def __init__(self):
        # 初始化提示词模板
        self._init_prompt_templates()
    
    def _init_prompt_templates(self):
        """
        初始化提示词模板
        """
        self.templates = {
            "base": """# 角色设定
{role}

# 任务描述
{task_description}

# 输入数据
{input_data}

# 分析要求
{analysis_requirements}

# 输出格式
{output_format}
""",
            
            "analyze": {
                "role": "您是一位专业的{domain}分析专家，具有丰富的行业经验和数据分析能力。",
                "task": "对提供的内容进行深入分析，识别关键信息、模式和趋势。",
                "requirements": [
                    "提供详细的分析过程和逻辑",
                    "识别潜在的机会和风险",
                    "给出基于数据的结论"
                ]
            },
            
            "summarize": {
                "role": "您是一位专业的内容总结专家，擅长提取核心信息并清晰呈现。",
                "task": "对提供的内容进行精准总结，提取关键要点和核心信息。",
                "requirements": [
                    "保持信息的完整性和准确性",
                    "突出最重要的发现和结论",
                    "使用清晰简洁的语言"
                ]
            },
            
            "compare": {
                "role": "您是一位专业的比较分析专家，擅长识别事物之间的异同点。",
                "task": "对提供的内容进行比较分析，识别相似点和差异点。",
                "requirements": [
                    "系统性地比较各个方面",
                    "提供具体的数据支持",
                    "分析比较结果的意义"
                ]
            },
            
            "trend": {
                "role": "您是一位专业的趋势分析专家，擅长识别和预测发展趋势。",
                "task": "分析提供内容中的趋势和变化，预测未来可能的发展方向。",
                "requirements": [
                    "识别明显的趋势和变化模式",
                    "分析趋势背后的驱动因素",
                    "提供合理的预测和建议"
                ]
            },
            
            "suggest": {
                "role": "您是一位专业的战略顾问，擅长提供有价值的建议和解决方案。",
                "task": "基于提供的内容，提供具体、可行的建议和解决方案。",
                "requirements": [
                    "提供具体的行动建议",
                    "分析建议的可行性和潜在影响",
                    "考虑不同的情况和可能的结果"
                ]
            }
        }
        
        # 输出格式模板
        self.output_formats = {
            "default": "使用清晰、结构化的格式输出结果，包含标题、摘要和详细内容。",
            "bullet": "使用 bullet 点列表格式，清晰列出每个要点。",
            "table": "使用表格格式呈现比较数据或结构化信息。",
            "markdown": "使用 Markdown 格式，包含标题、列表和强调。",
            "json": "使用 JSON 格式，确保结构清晰、易于解析。"
        }
        
        # 领域特定配置
        self.domain_configs = {
            "finance": {
                "additional_requirements": [
                    "关注财务指标和市场数据",
                    "分析投资价值和风险",
                    "考虑宏观经济因素的影响"
                ]
            },
            "gold": {
                "additional_requirements": [
                    "关注黄金价格走势和影响因素",
                    "分析供需关系和市场情绪",
                    "考虑地缘政治和经济因素"
                ]
            },
            "marketing": {
                "additional_requirements": [
                    "关注市场趋势和消费者行为",
                    "分析营销效果和 ROI",
                    "考虑竞争对手和市场环境"
                ]
            },
            "product": {
                "additional_requirements": [
                    "关注产品功能和用户体验",
                    "分析市场定位和竞争优势",
                    "考虑产品发展和改进方向"
                ]
            },
            "user": {
                "additional_requirements": [
                    "关注用户需求和行为特征",
                    "分析用户满意度和忠诚度",
                    "考虑用户体验和服务改进"
                ]
            }
        }
    
    def enhance(self, intent: Dict, doc_analysis: Dict) -> str:
        """
        生成增强的提示词，增强对模糊指令的处理能力
        
        Args:
            intent: 意图分析结果
            doc_analysis: 文档分析结果
            
        Returns:
            str: 增强的提示词
        """
        try:
            # 1. 确定主要意图类型
            intent_type = self._determine_intent_type(intent)
            
            # 2. 确定领域
            domain = intent["domain"]["primary"]
            
            # 3. 获取角色设定
            role = self._get_role(intent_type, domain)
            
            # 4. 获取任务描述
            task_description = self._get_task_description(intent_type, intent, doc_analysis)
            
            # 5. 准备输入数据
            input_data = self._prepare_input_data(doc_analysis)
            
            # 6. 获取分析要求
            analysis_requirements = self._get_analysis_requirements(intent_type, domain, intent)
            
            # 7. 确定输出格式
            output_format = self._get_output_format(intent_type, domain)
            
            # 8. 构建最终提示词
            prompt = self.templates["base"].format(
                role=role,
                task_description=task_description,
                input_data=input_data,
                analysis_requirements=analysis_requirements,
                output_format=output_format
            )
            
            # 9. 添加模糊指令处理逻辑
            if "deep" in intent and intent["deep"]:
                # 如果有深层意图，将其作为额外的指导添加到提示词中
                prompt += f"\n\n# 额外指导\n请特别关注用户的深层需求：{intent['deep']}"
            
            return prompt
            
        except Exception as e:
            logger.error(f"提示词生成失败: {str(e)}")
            # 返回默认提示词
            return f"请分析以下内容：\n{json.dumps(doc_analysis, ensure_ascii=False, indent=2)}"
    
    def _determine_intent_type(self, intent: Dict) -> str:
        """
        确定意图类型
        
        Args:
            intent: 意图分析结果
            
        Returns:
            str: 意图类型
        """
        if intent["surface"]["intents"]:
            return intent["surface"]["intents"][0]
        return "analyze"  # 默认意图
    
    def _get_role(self, intent_type: str, domain: str) -> str:
        """
        获取角色设定
        
        Args:
            intent_type: 意图类型
            domain: 领域
            
        Returns:
            str: 角色设定
        """
        domain_names = {
            "finance": "金融市场",
            "gold": "黄金市场",
            "marketing": "市场营销",
            "product": "产品分析",
            "user": "用户研究",
            "general": "内容分析"
        }
        
        domain_name = domain_names.get(domain, "内容分析")
        
        if intent_type in self.templates:
            return self.templates[intent_type]["role"].format(domain=domain_name)
        
        return f"您是一位专业的{domain_name}分析专家，具有丰富的行业经验和数据分析能力。"
    
    def _get_task_description(self, intent_type: str, intent: Dict, doc_analysis: Dict) -> str:
        """
        获取任务描述
        
        Args:
            intent_type: 意图类型
            intent: 意图分析结果
            doc_analysis: 文档分析结果
            
        Returns:
            str: 任务描述
        """
        if intent_type in self.templates:
            base_task = self.templates[intent_type]["task"]
        else:
            base_task = "对提供的内容进行深入分析。"
        
        # 添加时间范围信息
        time_desc = intent["time"]["range"]
        
        return f"{time_desc}，{base_task}"
    
    def _prepare_input_data(self, doc_analysis: Dict) -> str:
        """
        准备输入数据
        
        Args:
            doc_analysis: 文档分析结果
            
        Returns:
            str: 输入数据描述
        """
        data_parts = []
        
        # 添加关键点
        if doc_analysis.get("key_points"):
            key_points_text = "\n".join([f"- {point}" for point in doc_analysis["key_points"]])
            data_parts.append(f"### 文档关键点\n{key_points_text}")
        
        # 添加数据点
        if doc_analysis.get("data_points"):
            data_values = [data["value"] for data in doc_analysis["data_points"]]
            data_text = "\n".join([f"- {value}" for value in data_values[:10]])  # 最多显示10个数据点
            if len(data_values) > 10:
                data_text += f"\n- ... 等共 {len(data_values)} 个数据点"
            data_parts.append(f"### 关键数据\n{data_text}")
        
        # 添加情感分析
        if doc_analysis.get("sentiment"):
            sentiment = doc_analysis["sentiment"]["overall"]
            sentiment_score = doc_analysis["sentiment"]["score"]
            data_parts.append(f"### 情感分析\n整体情感倾向：{sentiment} (分数：{sentiment_score})")
        
        # 添加文档统计信息
        if doc_analysis.get("word_count"):
            word_count = doc_analysis["word_count"]
            sentence_count = doc_analysis["sentence_count"]
            data_parts.append(f"### 文档统计\n- 字数：{word_count}\n- 句子数：{sentence_count}")
        
        if not data_parts:
            return "### 输入数据\n请分析提供的内容。"
        
        return "\n".join(data_parts)
    
    def _get_analysis_requirements(self, intent_type: str, domain: str, intent: Dict) -> str:
        """
        获取分析要求
        
        Args:
            intent_type: 意图类型
            domain: 领域
            intent: 意图分析结果
            
        Returns:
            str: 分析要求
        """
        requirements = []
        
        # 添加基础要求
        if intent_type in self.templates:
            requirements.extend(self.templates[intent_type]["requirements"])
        else:
            requirements.append("提供详细的分析和结论")
        
        # 添加领域特定要求
        if domain in self.domain_configs:
            requirements.extend(self.domain_configs[domain]["additional_requirements"])
        
        # 添加量化指标要求
        if intent["metrics"]:
            metrics = [metric["name"] for metric in intent["metrics"]]
            if metrics:
                requirements.append(f"重点分析以下指标：{', '.join(metrics)}")
        
        # 格式化要求
        requirements_text = "\n".join([f"- {req}" for req in requirements])
        return f"### 分析要求\n{requirements_text}"
    
    def _get_output_format(self, intent_type: str, domain: str) -> str:
        """
        获取输出格式
        
        Args:
            intent_type: 意图类型
            domain: 领域
            
        Returns:
            str: 输出格式要求
        """
        # 根据意图类型和领域选择合适的输出格式
        format_requirements = []
        
        # 添加基础格式要求
        format_requirements.append(self.output_formats["default"])
        
        # 根据意图类型添加特定格式
        if intent_type in ["compare", "analyze"]:
            format_requirements.append("使用清晰的分点结构，便于阅读和理解。")
        elif intent_type == "summarize":
            format_requirements.append("使用简洁明了的语言，避免冗余信息。")
        elif intent_type == "trend":
            format_requirements.append("使用时间线或结构化方式呈现趋势。")
        
        # 格式化输出格式要求
        format_text = "\n".join([f"- {req}" for req in format_requirements])
        return f"### 输出格式\n{format_text}"

# 测试代码
if __name__ == "__main__":
    # 创建动态提示词生成器实例
    enhancer = PromptEnhancer()
    
    # 示例1：通用金融领域意图分析结果
    example_intent = {
        "surface": {
            "intents": ["analyze"],
            "domains": ["finance"],
            "time": ["week"],
            "metrics": ["price", "volume"],
            "keywords": ["分析", "本周", "价格", "走势"]
        },
        "deep": "分析本周市场价格的波动趋势和影响因素",
        "domain": {
            "primary": "finance",
            "secondary": [],
            "confidence": 0.9
        },
        "metrics": [
            {"type": "price", "name": "价格", "confidence": 0.8},
            {"type": "volume", "name": "交易量", "confidence": 0.8}
        ],
        "time": {
            "type": "week",
            "range": "本周"
        }
    }
    
    # 示例1：通用金融领域文档分析结果
    example_doc_analysis = {
        "key_points": [
            "本周市场价格表现强劲，周涨幅达到5.2%",
            "主要影响因素是宏观经济数据和市场情绪",
            "交易量显著增加，市场活跃度提升",
            "机构投资者持仓量有所增加"
        ],
        "data_points": [
            {"type": "percentage", "value": "5.2%", "context": "本周市场价格表现强劲，周涨幅达到5.2%"},
            {"type": "percentage", "value": "2.1%", "context": "主要影响因素是相关指数变化"},
            {"type": "number", "value": "12.5", "context": "机构投资者持仓量增加了12.5吨/万股"}
        ],
        "arguments": {
            "cause_effect_pairs": [
                {"cause": "宏观经济数据变化", "effect": "市场价格上涨", "confidence": 0.7}
            ],
            "conclusions": ["市场价格的上涨趋势可能会持续"],
            "opinions": ["建议投资者关注市场动态并做出相应决策"],
            "evidence": ["数据显示，机构投资者持仓量有所增加"]
        },
        "sentiment": {
            "overall": "positive",
            "score": 0.75,
            "word_count": 5
        },
        "word_count": 150,
        "sentence_count": 8
    }
    
    print("=== 动态提示词生成测试 ===")
    print("\n1. 输入意图分析结果:")
    print(json.dumps(example_intent, ensure_ascii=False, indent=2))
    
    print("\n2. 输入文档分析结果:")
    print(json.dumps(example_doc_analysis, ensure_ascii=False, indent=2))
    
    print("\n3. 生成的提示词:")
    prompt = enhancer.enhance(example_intent, example_doc_analysis)
    print(prompt)
