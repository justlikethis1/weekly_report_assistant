class PromptEngineer:
    """
    提示词工程师，用于构建高质量的LLM提示词，增强对模糊指令的处理能力
    """
    
    def __init__(self):
        """
        初始化提示词工程师
        """
        self.chinese = True  # 默认使用中文
        
        # 中文提示词模板
        self.chinese_templates = {
            "weekly": {
                "basic": "你是一位专业的周报生成助手，请根据以下信息生成一份清晰、结构化的周报：\n\n{context}\n\n用户要求：\n{user_input}\n\n请确保周报包含：本周工作内容、完成情况、存在的问题及解决方案、下周计划等部分。",
                "detailed": "你是一位专业的周报生成专家，请根据以下信息生成一份详细、专业的周报：\n\n{context}\n\n用户要求：\n{user_input}\n\n请确保周报包含：本周工作内容及进度、关键成果与亮点、遇到的挑战与解决方案、下周工作计划与优先级、资源需求等详细信息。使用结构化的格式，突出重点和数据支撑。",
                "strategic": "你是一位高级管理人员，请根据以下信息生成一份具有战略视角的周报：\n\n{context}\n\n用户要求：\n{user_input}\n\n请从战略高度分析本周工作，包括：战略目标推进情况、关键成果对整体目标的贡献、市场环境变化对工作的影响、团队能力建设情况、下周工作的战略重点等。使用专业的管理术语，提供深入的分析和建议。"
            },
            "monthly": {
                "basic": "你是一位专业的月报生成助手，请根据以下信息生成一份清晰、结构化的月报：\n\n{context}\n\n用户要求：\n{user_input}\n\n请确保月报包含：月度工作回顾、目标完成情况、重点项目进展、存在的问题、下月计划等部分。",
                "detailed": "你是一位专业的月报生成专家，请根据以下信息生成一份详细、专业的月报：\n\n{context}\n\n用户要求：\n{user_input}\n\n请确保月报包含：月度工作总体回顾、目标完成情况分析、重点项目详细进展、关键成果与亮点、存在的问题及改进措施、下月工作计划与优先级、资源需求等详细信息。使用结构化的格式，突出重点和数据支撑。",
                "strategic": "你是一位高级管理人员，请根据以下信息生成一份具有战略视角的月报：\n\n{context}\n\n用户要求：\n{user_input}\n\n请从战略高度分析月度工作，包括：战略目标推进情况、关键成果对整体目标的贡献、市场环境变化对工作的影响、团队能力建设情况、下月工作的战略重点等。使用专业的管理术语，提供深入的分析和建议。"
            },
            "general": {
                "basic": "你是一位专业的内容分析助手，请根据以下信息生成一份清晰、结构化的分析报告：\n\n{context}\n\n用户要求：\n{user_input}\n\n请确保报告包含核心信息、关键发现和相关结论。",
                "detailed": "你是一位专业的内容分析专家，请根据以下信息生成一份详细、专业的分析报告：\n\n{context}\n\n用户要求：\n{user_input}\n\n请确保报告包含：详细的分析过程、关键信息提取、模式识别、趋势分析、潜在机会和风险等。使用结构化的格式，突出重点和数据支撑。",
                "strategic": "你是一位高级分析师，请根据以下信息生成一份具有战略视角的分析报告：\n\n{context}\n\n用户要求：\n{user_input}\n\n请从战略高度分析提供的信息，包括：战略意义、对整体目标的贡献、市场环境影响、能力建设需求、未来发展方向等。使用专业的分析术语，提供深入的洞察和建议。"
            },
            # 可以根据需要添加更多报告类型的模板
        }
        
        # 英文提示词模板
        self.english_templates = {
            "weekly": {
                "basic": "You are a professional weekly report generator. Please generate a clear and structured weekly report based on the following information:\n\n{context}\n\nUser requirements:\n{user_input}\n\nPlease ensure the report includes: this week's work content, completion status, problems and solutions, next week's plan, etc.",
                "detailed": "You are a professional weekly report expert. Please generate a detailed and professional weekly report based on the following information:\n\n{context}\n\nUser requirements:\n{user_input}\n\nPlease ensure the report includes: this week's work content and progress, key achievements and highlights, challenges and solutions, next week's work plan and priorities, resource requirements, etc. Use a structured format, highlight key points and data support.",
                "strategic": "You are a senior manager. Please generate a strategically oriented weekly report based on the following information:\n\n{context}\n\nUser requirements:\n{user_input}\n\nPlease analyze this week's work from a strategic perspective, including: strategic goal progress, contribution of key achievements to overall goals, impact of market environment changes on work, team capacity building, strategic focus for next week, etc. Use professional management terminology, provide in-depth analysis and recommendations."
            },
            "monthly": {
                "basic": "You are a professional monthly report generator. Please generate a clear and structured monthly report based on the following information:\n\n{context}\n\nUser requirements:\n{user_input}\n\nPlease ensure the report includes: monthly work review, goal completion status, key project progress, problems, next month's plan, etc.",
                "detailed": "You are a professional monthly report expert. Please generate a detailed and professional monthly report based on the following information:\n\n{context}\n\nUser requirements:\n{user_input}\n\nPlease ensure the report includes: overall monthly work review, goal completion analysis, detailed progress of key projects, key achievements and highlights, problems and improvement measures, next month's work plan and priorities, resource requirements, etc. Use a structured format, highlight key points and data support.",
                "strategic": "You are a senior manager. Please generate a strategically oriented monthly report based on the following information:\n\n{context}\n\nUser requirements:\n{user_input}\n\nPlease analyze monthly work from a strategic perspective, including: strategic goal progress, contribution of key achievements to overall goals, impact of market environment changes on work, team capacity building, strategic focus for next month, etc. Use professional management terminology, provide in-depth analysis and recommendations."
            },
            "general": {
                "basic": "You are a professional content analysis assistant. Please generate a clear and structured analysis report based on the following information:\n\n{context}\n\nUser requirements:\n{user_input}\n\nPlease ensure the report includes core information, key findings, and relevant conclusions.",
                "detailed": "You are a professional content analysis expert. Please generate a detailed and professional analysis report based on the following information:\n\n{context}\n\nUser requirements:\n{user_input}\n\nPlease ensure the report includes: detailed analysis process, key information extraction, pattern recognition, trend analysis, potential opportunities and risks, etc. Use a structured format, highlight key points and data support.",
                "strategic": "You are a senior analyst. Please generate a strategically oriented analysis report based on the following information:\n\n{context}\n\nUser requirements:\n{user_input}\n\nPlease analyze the provided information from a strategic perspective, including: strategic significance, contribution to overall goals, market environment impact, capacity building needs, future development directions, etc. Use professional analytical terminology, provide in-depth insights and recommendations."
            },
            # 可以根据需要添加更多报告类型的模板
        }
    
    def get_prompt(self, report_type: str, analysis_depth: str, include_context: bool = False) -> str:
        """
        获取构建好的提示词
        
        Args:
            report_type: 报告类型
            analysis_depth: 分析深度（basic, detailed, strategic）
            include_context: 是否包含上下文
            
        Returns:
            str: 构建好的提示词
        """
        # 获取合适的模板
        if self.chinese:
            templates = self.chinese_templates
        else:
            templates = self.english_templates
        
        # 确保报告类型存在
        if report_type not in templates:
            report_type = "general"  # 默认使用通用报告模板
        
        # 确保分析深度存在
        if analysis_depth not in templates[report_type]:
            analysis_depth = "basic"  # 默认使用基础深度
        
        # 获取模板
        template = templates[report_type][analysis_depth]
        
        # 处理上下文
        context = "" if not include_context else "{context}"
        
        # 返回构建好的提示词（不包含用户输入部分，用户输入将在调用处添加）
        return template.replace("{context}", context)
    
    def infer_report_type(self, user_input: str) -> str:
        """
        从用户输入中推断报告类型
        
        Args:
            user_input: 用户输入字符串
            
        Returns:
            str: 推断出的报告类型
        """
        user_input_lower = user_input.lower()
        
        # 从用户输入中推断报告类型
        if any(word in user_input_lower for word in ["周报", "周报告", "每周报告", "这周工作", "本周工作"]):
            return "weekly"
        elif any(word in user_input_lower for word in ["月报", "月报告", "每月报告", "这个月工作", "本月工作"]):
            return "monthly"
        else:
            return "general"  # 默认通用报告类型
    
    def infer_analysis_depth(self, user_input: str) -> str:
        """
        从用户输入中推断分析深度
        
        Args:
            user_input: 用户输入字符串
            
        Returns:
            str: 推断出的分析深度
        """
        user_input_lower = user_input.lower()
        
        # 从用户输入中推断分析深度
        if any(word in user_input_lower for word in ["详细", "深入", "全面", "完整", "具体", "详细分析"]):
            return "detailed"
        elif any(word in user_input_lower for word in ["战略", "高度", "宏观", "长远", "战略分析"]):
            return "strategic"
        else:
            return "basic"  # 默认基础深度
    
    def generate_prompt(self, user_input: str, context: str = "", report_type: str = None, analysis_depth: str = None) -> str:
        """
        生成优化的提示词，自动推断报告类型和分析深度
        
        Args:
            user_input: 用户输入字符串
            context: 上下文信息
            report_type: 报告类型（可选，如果不提供将自动推断）
            analysis_depth: 分析深度（可选，如果不提供将自动推断）
            
        Returns:
            str: 生成的提示词
        """
        # 自动推断报告类型
        if not report_type:
            report_type = self.infer_report_type(user_input)
        
        # 自动推断分析深度
        if not analysis_depth:
            analysis_depth = self.infer_analysis_depth(user_input)
        
        # 获取基础提示词
        prompt = self.get_prompt(report_type, analysis_depth, include_context=bool(context))
        
        # 替换上下文和用户输入
        if context:
            prompt = prompt.replace("{context}", context)
        
        prompt = prompt.replace("{user_input}", user_input)
        
        return prompt
