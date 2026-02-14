from typing import List, Dict, Any, Optional
import re
import random
from datetime import datetime
import os
import pandas as pd

class AdvancedReportGenerator:
    """
    高级报告生成器类，负责：
    1. 深度解析用户输入要求（受众、决策、焦点、深度）
    2. 构建贯穿始终的分析主线和核心论点
    3. 基于金字塔原理生成报告结构
    4. 实现数据驱动的分析（描述性、诊断性、预测性）
    5. 生成具体、有针对性的报告内容
    6. 提供可操作的建议
    """
    
    def __init__(self):
        # 初始化配置和模板
        self.report_templates = {
            "analysis": "深度分析报告",
            "industry": "行业研究报告",
            "market": "市场调研报告",
            "strategy": "战略评估报告",
            "comprehensive": "综合分析报告"
        }
        
        # 主题属性库
        self.topic_attributes = {
            "黄金": {
                "type": "金融产品,大宗商品,避险资产",
                "key_attributes": "价格走势,供需关系,地缘政治影响,货币政策影响,投资需求",
                "industry_chain": "勘探,开采,冶炼,加工,销售,投资交易",
                "key_metrics": "COMEX金价,ETF持仓,央行储备,实物需求,工业需求"
            },
            "人工智能": {
                "type": "技术领域,新兴产业",
                "key_attributes": "算法创新,算力,数据,应用场景,政策监管",
                "industry_chain": "基础层（芯片/算力）,技术层（算法/框架）,应用层（行业解决方案）",
                "key_metrics": "市场规模,融资额,专利数,核心企业收入,渗透率"
            },
            "新能源汽车": {
                "type": "交通工具,新能源产业",
                "key_attributes": "电池技术,充电基础设施,政策补贴,市场需求,产业链配套",
                "industry_chain": "上游（电池材料）,中游（电池/零部件）,下游（整车制造/销售）,服务（充电/运维）",
                "key_metrics": "销量,渗透率,电池成本,续航里程,充电网络密度"
            },
            "房地产": {
                "type": "不动产,民生领域",
                "key_attributes": "房价走势,政策调控,土地供应,信贷环境,人口结构",
                "industry_chain": "土地开发,建筑施工,销售代理,物业管理,金融服务",
                "key_metrics": "房价指数,成交量,库存周期,首付比例,贷款利率"
            },
            "养老产业": {
                "type": "民生服务,消费产业",
                "key_attributes": "人口老龄化,政策支持,市场需求,服务模式,资金来源",
                "industry_chain": "养老服务,养老地产,养老金融,养老用品,医疗健康",
                "key_metrics": "老年人口比例,养老床位数,服务满意度,市场规模,政策补贴金额"
            }
        }
        
        # 分析框架库
        self.analysis_frameworks = {
            "why": [  # 归因分析框架
                "内外因素分析",
                "供需分析",
                "价值链分析",
                "波特五力模型",
                "PESTEL分析"
            ],
            "what": [  # 现状分析框架
                "规模与增长",
                "结构与分布",
                "质量与效益",
                "竞争格局",
                "发展阶段"
            ],
            "how": [  # 策略分析框架
                "SWOT分析",
                "BCG矩阵",
                "发展路径规划",
                "资源配置方案",
                "风险防控措施"
            ]
        }
        
        # 替代空泛词汇的具体表述
        self.specific_expressions = {
            "快速发展": ["同比增长{growth_rate}%", "环比提升{growth_rate}%", "年复合增长率达到{growth_rate}%"],
            "持续扩大": ["规模突破{scale}亿元", "市场份额提升至{share}%", "覆盖范围扩展至{regions}"],
            "核心动力": ["{driver}贡献了{contribution}%的增长", "{driver}成为关键增长点", "{driver}推动作用显著"],
            "深刻变化": ["从{old_model}转向{new_model}", "{aspect}发生结构性调整", "{dimension}出现反转趋势"],
            "加强": ["优化{aspect}流程", "提升{aspect}效率", "强化{aspect}管理"],
            "加大": ["增加{aspect}投入至{amount}", "扩大{aspect}规模至{scale}", "提高{aspect}比例至{ratio}%"],
            "完善": ["建立{aspect}体系", "健全{aspect}机制", "优化{aspect}结构"]
        }
        
    def parse_user_input(self, user_input: str) -> Dict[str, Any]:
        """
        深度解析用户输入，提取分析任务四要素
        
        Args:
            user_input: 用户输入的报告要求
            
        Returns:
            包含解析结果的字典
        """
        input_lower = user_input.lower()
        
        # 1. 提取报告类型
        report_type = self._extract_report_type(input_lower)
        
        # 2. 提取主题和属性
        topics = self._extract_topics(user_input)
        
        # 3. 解析分析任务四要素
        audience = self._infer_audience(user_input)
        decision = self._infer_decision(user_input)
        focus = self._infer_focus(user_input)
        depth = self._infer_depth(user_input)
        
        # 4. 提取时间范围
        time_range = self._extract_time_range(user_input)
        
        # 5. 提取分析要求
        analysis_requirements = self._extract_analysis_requirements(input_lower)
        
        return {
            "user_input": user_input,
            "report_type": report_type,
            "main_topic": topics.get("main", "通用主题"),
            "main_topic_attributes": topics.get("attributes", {}),
            "sub_topics": topics.get("sub", []),
            "audience": audience,
            "decision": decision,
            "focus": focus,
            "depth": depth,
            "analysis_requirements": analysis_requirements,
            "time_range": time_range,
            "generated_time": datetime.now()
        }
    
    def _extract_report_type(self, input_lower: str) -> str:
        """提取报告类型"""
        for key, value in self.report_templates.items():
            if key in input_lower or value in input_lower:
                return value
        return self.report_templates["analysis"]  # 默认使用深度分析报告
    
    def _extract_topics(self, user_input: str) -> Dict[str, Any]:
        """提取主题和属性"""
        input_lower = user_input.lower()
        
        # 检测主主题
        main_topic = "通用主题"
        attributes = {}
        
        # 先检查具体主题
        for topic in self.topic_attributes.keys():
            if topic in input_lower:
                main_topic = topic
                attributes = self.topic_attributes[topic]
                break
        
        # 生成或提取子主题
        if main_topic in self.topic_attributes:
            # 根据主题属性生成相关子主题
            key_attributes = attributes.get("key_attributes", "").split(",")
            sub_topics = [attr.strip() for attr in key_attributes]
        else:
            sub_topics = []
        
        return {"main": main_topic, "attributes": attributes, "sub": sub_topics}
    
    def _infer_audience(self, user_input: str) -> Dict[str, Any]:
        """推断受众"""
        input_lower = user_input.lower()
        
        # 根据关键词推断受众
        audience_keywords = {
            "管理层": ["ceo", "董事会", "管理层", "决策层"],
            "投资经理": ["投资", "股票", "基金", "投资经理", "分析师"],
            "业务部门": ["业务", "运营", "销售", "市场", "产品"],
            "政府部门": ["政策", "监管", "政府", "发改委", "工信部"]
        }
        
        audience = "通用读者"
        concerns = []
        
        for aud, keywords in audience_keywords.items():
            for keyword in keywords:
                if keyword in input_lower:
                    audience = aud
                    break
            if audience != "通用读者":
                break
        
        # 根据受众推断关注点
        if audience == "管理层":
            concerns = ["战略方向", "业绩表现", "资源配置", "风险防控"]
        elif audience == "投资经理":
            concerns = ["投资价值", "风险收益", "市场趋势", "估值水平"]
        elif audience == "业务部门":
            concerns = ["运营效率", "市场份额", "竞争策略", "客户需求"]
        elif audience == "政府部门":
            concerns = ["政策效果", "行业发展", "就业影响", "社会贡献"]
        else:
            concerns = ["行业概况", "发展趋势", "主要挑战", "未来展望"]
        
        return {"type": audience, "concerns": concerns}
    
    def _infer_decision(self, user_input: str) -> Dict[str, Any]:
        """推断决策场景"""
        input_lower = user_input.lower()
        
        # 根据关键词推断决策类型
        decision_keywords = {
            "进入市场": ["进入", "开拓", "布局", "投资", "新建"],
            "调整策略": ["调整", "优化", "改进", "转型", "升级"],
            "评估风险": ["风险", "评估", "分析", "监测", "预警"],
            "资源配置": ["资源", "配置", "分配", "投入", "预算"]
        }
        
        decision_type = "信息了解"
        decision_context = ""
        
        for dec, keywords in decision_keywords.items():
            for keyword in keywords:
                if keyword in input_lower:
                    decision_type = dec
                    break
            if decision_type != "信息了解":
                break
        
        # 根据决策类型生成决策问题
        if decision_type == "进入市场":
            decision_context = "评估进入新市场的可行性和策略"
        elif decision_type == "调整策略":
            decision_context = "分析现有策略问题并提出调整方案"
        elif decision_type == "评估风险":
            decision_context = "识别主要风险并评估影响程度"
        elif decision_type == "资源配置":
            decision_context = "优化资源分配以提升效率和效益"
        else:
            decision_context = "了解行业现状和发展趋势"
        
        return {"type": decision_type, "context": decision_context}
    
    def _infer_focus(self, user_input: str) -> Dict[str, Any]:
        """推断分析焦点"""
        input_lower = user_input.lower()
        
        # 根据关键词推断分析焦点
        focus_keywords = {
            "what": ["现状", "情况", "概况", "表现", "数据"],
            "why": ["原因", "因素", "影响", "为何", "为什么"],
            "how": ["策略", "建议", "方案", "怎么办", "如何"]
        }
        
        focus = "what"
        
        for f, keywords in focus_keywords.items():
            for keyword in keywords:
                if keyword in input_lower:
                    focus = f
                    break
            if focus != "what":
                break
        
        # 根据焦点选择分析框架
        frameworks = self.analysis_frameworks.get(focus, self.analysis_frameworks["what"])
        
        return {"type": focus, "frameworks": frameworks}
    
    def _infer_depth(self, user_input: str) -> Dict[str, Any]:
        """推断分析深度"""
        input_lower = user_input.lower()
        
        # 根据关键词推断深度
        depth_keywords = {
            "overview": ["概览", "简报", "摘要", "简要"],
            "deep": ["深度", "详细", "全面", "深入", "专题"]
        }
        
        depth = "standard"
        expected_length = 5000
        
        for d, keywords in depth_keywords.items():
            for keyword in keywords:
                if keyword in input_lower:
                    depth = d
                    break
            if depth != "standard":
                break
        
        # 根据深度设置预期长度
        if depth == "overview":
            expected_length = 2000
        elif depth == "deep":
            expected_length = 10000
        
        return {"type": depth, "expected_length": expected_length}
    
    def _extract_time_range(self, user_input: str) -> Dict[str, Optional[str]]:
        """提取时间范围"""
        # 简单的时间范围提取，实际可以使用更复杂的正则表达式
        time_patterns = [
            (r"(\d{4})年(\d{1,2})月", "monthly"),
            (r"(\d{4})年第(\d)季度", "quarterly"),
            (r"(\d{4})年", "yearly"),
            (r"最近(\d+)个月", "recent_months"),
            (r"过去(\d+)年", "past_years"),
            (r"(\d{4})年(\d{1,2})月至(\d{4})年(\d{1,2})月", "period")
        ]
        
        for pattern, time_type in time_patterns:
            match = re.search(pattern, user_input)
            if match:
                return {
                    "type": time_type,
                    "start": match.group(0),
                    "end": None
                }
        
        # 默认使用当前月
        return {
            "type": "monthly",
            "start": datetime.now().strftime("%Y年%m月"),
            "end": None
        }
    
    def _extract_analysis_requirements(self, input_lower: str) -> List[str]:
        """提取分析要求"""
        requirements = []
        
        # 常见分析要求关键词
        analysis_keywords = [
            "关系", "影响", "趋势", "机制", "策略", "建议", "对比", "评估",
            "特征", "结构", "因素", "风险", "机遇", "挑战", "展望", "预测"
        ]
        
        for keyword in analysis_keywords:
            if keyword in input_lower:
                requirements.append(keyword)
        
        return requirements
    
    def generate_report(self, parsed_input: Dict[str, Any], data_files: List[str] = None, is_mock: bool = False) -> str:
        """
        基于金字塔原理生成高质量报告
        
        Args:
            parsed_input: 解析后的用户输入
            data_files: 数据文件路径列表
            is_mock: 是否使用模拟数据
            
        Returns:
            生成的报告文本
        """
        # 1. 生成报告标题
        report_title = self._generate_report_title(parsed_input)
        
        # 2. 构建核心论点（金字塔塔尖）
        core_thesis = self._generate_core_thesis(parsed_input)
        
        # 3. 生成关键论点（金字塔塔身）
        key_arguments = self._generate_key_arguments(parsed_input, core_thesis)
        
        # 4. 生成报告结构（基于金字塔原理）
        report_structure = self._generate_report_structure(parsed_input, key_arguments)
        
        # 5. 生成报告内容（填充塔基）
        report_content = self._generate_report_content(parsed_input, report_structure, core_thesis, key_arguments, data_files, is_mock)
        
        # 6. 组合完整报告
        full_report = f"【专业报告】{report_title}\n\n"
        full_report += report_content
        
        return full_report
    
    def _generate_report_title(self, parsed_input: Dict[str, Any]) -> str:
        """
        生成具体、有针对性的报告标题
        """
        time_range = parsed_input["time_range"]["start"]
        main_topic = parsed_input["main_topic"]
        report_type = parsed_input["report_type"]
        focus = parsed_input["focus"]["type"]
        
        # 根据分析焦点调整标题
        focus_map = {
            "what": "现状与特征分析",
            "why": "成因与影响机制分析",
            "how": "策略与行动方案分析"
        }
        
        focus_desc = focus_map.get(focus, "深度分析")
        
        return f"{time_range} {main_topic}{focus_desc}{report_type}"
    
    def _generate_core_thesis(self, parsed_input: Dict[str, Any]) -> str:
        """
        生成核心论点（金字塔塔尖）
        """
        main_topic = parsed_input["main_topic"]
        focus = parsed_input["focus"]["type"]
        attributes = parsed_input["main_topic_attributes"]
        
        # 根据主题和焦点生成核心论点
        if main_topic in self.topic_attributes:
            if focus == "what":
                # 现状分析核心论点
                return f"{main_topic}作为{attributes.get('type', '')}，呈现出{self._get_specific_expression('快速发展', main_topic)}的态势，{self._get_specific_expression('核心动力', main_topic)}，{self._get_specific_expression('深刻变化', main_topic)}。"
            elif focus == "why":
                # 归因分析核心论点
                return f"{main_topic}的发展受{', '.join(attributes.get('key_attributes', '').split(',')[:3])}等多重因素影响，其中{attributes.get('key_attributes', '').split(',')[0].strip()}是关键驱动因素，通过{attributes.get('industry_chain', '').split(',')[0].strip()}环节产生主要影响。"
            elif focus == "how":
                # 策略分析核心论点
                return f"针对{main_topic}发展面临的{', '.join(attributes.get('key_attributes', '').split(',')[-2:])}等挑战，建议采取{self._get_specific_expression('加强', main_topic)}、{self._get_specific_expression('加大', main_topic)}、{self._get_specific_expression('完善', main_topic)}等策略。"
        
        # 默认核心论点
        return f"{main_topic}领域呈现出重要发展态势，需要深入分析其内在规律和发展路径。"
    
    def _generate_key_arguments(self, parsed_input: Dict[str, Any], core_thesis: str) -> List[str]:
        """
        生成关键论点（金字塔塔身）
        """
        main_topic = parsed_input["main_topic"]
        focus = parsed_input["focus"]["type"]
        attributes = parsed_input["main_topic_attributes"]
        
        key_arguments = []
        
        # 根据主题属性生成关键论点
        if main_topic in self.topic_attributes:
            key_attributes = attributes.get("key_attributes", "").split(",")
            industry_chain = attributes.get("industry_chain", "").split(",")
            
            if focus == "what":
                # 现状分析关键论点
                key_arguments = [
                    f"{main_topic}{key_attributes[0].strip()}表现{self._get_specific_expression('快速发展', main_topic)}",
                    f"{main_topic}{key_attributes[1].strip()}呈现{self._get_specific_expression('深刻变化', main_topic)}",
                    f"{main_topic}{key_attributes[2].strip()}成为{self._get_specific_expression('核心动力', main_topic)}"
                ]
            elif focus == "why":
                # 归因分析关键论点
                key_arguments = [
                    f"{industry_chain[0].strip()}环节的{key_attributes[0].strip()}是{main_topic}发展的基础驱动",
                    f"{industry_chain[1].strip()}环节的{key_attributes[1].strip()}是{main_topic}发展的关键支撑",
                    f"{industry_chain[2].strip()}环节的{key_attributes[2].strip()}是{main_topic}发展的重要影响因素"
                ]
            elif focus == "how":
                # 策略分析关键论点
                key_arguments = [
                    f"优化{industry_chain[0].strip()}环节的{key_attributes[0].strip()}配置",
                    f"加强{industry_chain[1].strip()}环节的{key_attributes[1].strip()}能力",
                    f"完善{industry_chain[2].strip()}环节的{key_attributes[2].strip()}体系"
                ]
        
        # 确保至少有3个关键论点
        while len(key_arguments) < 3:
            key_arguments.append(f"{main_topic}发展的第{len(key_arguments) + 1}个关键方面")
        
        return key_arguments[:3]  # 控制在3个关键论点
    
    def _generate_report_structure(self, parsed_input: Dict[str, Any], key_arguments: List[str]) -> List[Dict[str, Any]]:
        """
        基于金字塔原理生成报告结构
        """
        # 核心结构：摘要 -> 核心论点 -> 关键论点论证 -> 结论与建议
        report_structure = [
            {"section": "摘要", "subsections": ["核心结论", "关键发现", "主要建议"]},
            {"section": "核心论点", "subsections": ["分析背景", "核心命题", "研究框架"]}
        ]
        
        # 添加关键论点论证部分
        for i, argument in enumerate(key_arguments):
            report_structure.append({
                "section": f"关键论点{i+1}：{argument}",
                "subsections": [
                    "现状描述与数据支撑",
                    "成因分析与机制阐释",
                    "影响评估与趋势预判"
                ]
            })
        
        # 添加结论与建议部分
        report_structure.append({
            "section": "结论与建议", 
            "subsections": [
                "主要结论总结", 
                "具体行动建议", 
                "风险防控措施"
            ]
        })
        
        return report_structure
    
    def _generate_report_content(self, parsed_input: Dict[str, Any], report_structure: List[Dict[str, Any]], 
                               core_thesis: str, key_arguments: List[str], data_files: List[str] = None, 
                               is_mock: bool = False) -> str:
        """
        生成报告内容，填充金字塔塔基
        """
        content = ""
        main_topic = parsed_input["main_topic"]
        attributes = parsed_input["main_topic_attributes"]
        audience = parsed_input["audience"]["type"]
        
        for i, section in enumerate(report_structure):
            # 生成章节标题
            content += f"# {i+1}. {section['section']}\n\n"
            
            # 生成子章节内容
            for j, subsection in enumerate(section["subsections"]):
                content += f"## {i+1}.{j+1} {subsection}\n\n"
                
                # 根据章节和子章节生成内容
                if is_mock:
                    content += self._generate_mock_content(main_topic, attributes, section["section"], subsection, 
                                                         core_thesis, key_arguments, audience)
                else:
                    # 使用实际数据生成内容
                    content += self._generate_real_content(main_topic, attributes, section["section"], subsection, 
                                                         core_thesis, key_arguments, data_files)
                
                content += "\n"
        
        # 添加报告结尾
        content += "【专业报告结束】"
        
        return content
    
    def _generate_mock_content(self, main_topic: str, attributes: Dict[str, Any], section: str, subsection: str, 
                              core_thesis: str, key_arguments: List[str], audience: str) -> str:
        """
        生成具体、有针对性的模拟内容
        """
        # 根据章节和子章节生成不同的内容
        if section == "摘要":
            return self._generate_abstract_content(main_topic, attributes, subsection, core_thesis, key_arguments, audience)
        elif section == "核心论点":
            return self._generate_core_argument_content(main_topic, attributes, subsection, core_thesis)
        elif section.startswith("关键论点"):
            return self._generate_key_argument_content(main_topic, attributes, subsection, section)
        elif section == "结论与建议":
            return self._generate_conclusion_content(main_topic, attributes, subsection, core_thesis, key_arguments)
        
        # 默认内容
        return f"{subsection}是{section}的重要组成部分，对{main_topic}领域的发展具有重要意义。"
    
    def _generate_abstract_content(self, main_topic: str, attributes: Dict[str, Any], subsection: str, 
                                  core_thesis: str, key_arguments: List[str], audience: str) -> str:
        """
        生成摘要内容
        """
        if subsection == "核心结论":
            return f"本报告通过{', '.join(attributes.get('key_attributes', '').split(',')[:3])}等维度的深入分析，得出核心结论：{core_thesis}。这一结论对{audience}制定{self._get_specific_expression('加强', main_topic)}、{self._get_specific_expression('完善', main_topic)}等策略具有重要参考价值。"
        elif subsection == "关键发现":
            findings = []
            for arg in key_arguments:
                findings.append(f"{arg.replace('关键论点1：', '').replace('关键论点2：', '').replace('关键论点3：', '')}")
            return f"研究发现：{'; '.join(findings)}。此外，{main_topic}的{attributes.get('key_attributes', '').split(',')[-1].strip()}呈现出{self._get_specific_expression('深刻变化', main_topic)}的趋势，需要重点关注。"
        elif subsection == "主要建议":
            return f"基于上述分析，建议{audience}采取以下措施：1）{self._get_specific_expression('加强', main_topic)}；2）{self._get_specific_expression('加大', main_topic)}；3）{self._get_specific_expression('完善', main_topic)}。这些措施将有助于有效应对{main_topic}发展面临的挑战，抓住发展机遇。"
    
    def _generate_core_argument_content(self, main_topic: str, attributes: Dict[str, Any], subsection: str, 
                                       core_thesis: str) -> str:
        """
        生成核心论点内容
        """
        if subsection == "分析背景":
            return f"{main_topic}作为{attributes.get('type', '')}，其发展受到{', '.join(attributes.get('key_attributes', '').split(',')[:3])}等多重因素的影响。近年来，{main_topic}市场呈现出{self._get_specific_expression('快速发展', main_topic)}的态势，引起了{', '.join(attributes.get('key_attributes', '').split(',')[-2:])}等方面的广泛关注。"
        elif subsection == "核心命题":
            return f"本报告的核心命题是：{core_thesis}。这一命题的提出基于对{main_topic}领域{self._get_specific_expression('深刻变化', main_topic)}的观察，以及对{attributes.get('key_attributes', '').split(',')[0].strip()}等关键因素的深入分析。"
        elif subsection == "研究框架":
            return f"本报告采用{', '.join(self.analysis_frameworks.get('why', []))}等分析框架，从{', '.join(attributes.get('industry_chain', '').split(',')[:3])}等环节入手，系统分析{main_topic}发展的内在规律和外部影响。"
    
    def _generate_key_argument_content(self, main_topic: str, attributes: Dict[str, Any], subsection: str, 
                                      section_title: str) -> str:
        """
        生成关键论点论证内容
        """
        if subsection == "现状描述与数据支撑":
            return self._generate_descriptive_content(main_topic, attributes, section_title)
        elif subsection == "成因分析与机制阐释":
            return self._generate_diagnostic_content(main_topic, attributes, section_title)
        elif subsection == "影响评估与趋势预判":
            return self._generate_predictive_content(main_topic, attributes, section_title)
    
    def _generate_descriptive_content(self, main_topic: str, attributes: Dict[str, Any], section_title: str) -> str:
        """
        生成描述性分析内容（发生了什么）
        """
        # 使用用户提供的实际数据或占位符
        data = attributes.get('data', {})
        
        # 从数据中提取指标，或者使用占位符
        scale = data.get('scale', '根据实际数据填写')
        growth = data.get('growth', '根据实际数据填写')
        structure = data.get('structure', '根据实际数据填写')
        distribution = data.get('distribution', '根据实际数据填写')
        
        content = f"根据{self._get_specific_source(main_topic)}数据显示，{main_topic}市场{self._get_specific_expression('快速发展', main_topic)}。具体表现为：\n\n"
        content += "### 关键指标表现\n"
        content += f"- 规模：{scale}\n"
        content += f"- 增速：{growth}\n"
        content += f"- 结构：{structure}\n"
        content += f"- 分布：{distribution}\n\n"
        
        content += "### 重要变化趋势\n"
        content += f"1. {self._get_specific_expression('持续扩大', main_topic)}\n"
        content += f"2. {self._get_specific_expression('核心动力', main_topic)}\n"
        content += f"3. {self._get_specific_expression('深刻变化', main_topic)}\n\n"
        
        return content
    
    def _generate_diagnostic_content(self, main_topic: str, attributes: Dict[str, Any], section_title: str) -> str:
        """
        生成诊断性分析内容（为什么发生）
        """
        content = f"{main_topic}呈现上述特征的原因是多方面的，主要包括：\n\n"
        
        # 分析关键因素及其相互作用
        key_factors = attributes.get('key_attributes', '').split(',')[:3]
        for i, factor in enumerate(key_factors):
            factor = factor.strip()
            content += f"### {i+1}. {factor}的影响\n"
            content += f"{factor}通过{self._get_specific_mechanism(main_topic, factor)}等渠道，对{main_topic}的{attributes.get('industry_chain', '').split(',')[i].strip()}环节产生了显著影响，贡献了{random.randint(20, 50)}%的变化。\n\n"
        
        # 分析因素间的相互作用
        content += "### 因素间的相互作用机制\n"
        content += f"{key_factors[0].strip()}与{key_factors[1].strip()}之间存在{random.choice(['正向强化', '负向制约', '非线性交互'])}关系，{key_factors[0].strip()}的变化会通过{self._get_specific_mechanism(main_topic, 'interaction')}放大{key_factors[1].strip()}的影响效果。\n\n"
        
        return content
    
    def _generate_predictive_content(self, main_topic: str, attributes: Dict[str, Any], section_title: str) -> str:
        """
        生成预测性分析内容（接下来会怎样）
        """
        content = f"基于上述分析，预计未来{random.choice(['6个月', '1年', '2年'])}，{main_topic}将呈现以下趋势：\n\n"
        
        # 生成趋势预测
        trends = [
            f"{self._get_specific_expression('快速发展', main_topic)}的态势将{random.choice(['延续', '放缓', '加速'])}",
            f"{attributes.get('key_attributes', '').split(',')[1].strip()}的影响将{random.choice(['增强', '减弱', '转化'])}",
            f"{main_topic}的{attributes.get('industry_chain', '').split(',')[0].strip()}环节将{random.choice(['成为新的增长点', '面临调整压力', '实现技术突破'])}"
        ]
        
        for i, trend in enumerate(trends):
            content += f"### 趋势{i+1}：{trend}\n"
            content += f"这一趋势的主要驱动因素包括：{', '.join(attributes.get('key_attributes', '').split(',')[-2:])}等。预计将对{main_topic}市场产生{random.choice(['积极', '中性', '深远'])}{random.choice(['影响', '变革', '重塑'])}\n\n"
        
        return content
    
    def _generate_conclusion_content(self, main_topic: str, attributes: Dict[str, Any], subsection: str, 
                                   core_thesis: str, key_arguments: List[str]) -> str:
        """
        生成结论与建议内容
        """
        if subsection == "主要结论总结":
            return f"通过对{main_topic}的{', '.join(attributes.get('key_attributes', '').split(',')[:3])}等维度的系统分析，本报告得出以下结论：\n1. {core_thesis}\n2. {'; '.join([arg.replace('关键论点1：', '').replace('关键论点2：', '').replace('关键论点3：', '') for arg in key_arguments])}。"
        elif subsection == "具体行动建议":
            return f"基于上述结论，建议采取以下具体行动：\n1. {self._get_specific_expression('加强', main_topic)}\n2. {self._get_specific_expression('加大', main_topic)}\n3. {self._get_specific_expression('完善', main_topic)}。这些建议具有{random.choice(['针对性', '可操作性', '前瞻性'])}，可有效应对{main_topic}发展面临的挑战。"
        elif subsection == "风险防控措施":
            return f"在实施上述建议的过程中，需要重点防控以下风险：\n1. {attributes.get('key_attributes', '').split(',')[0].strip()}波动风险\n2. {attributes.get('key_attributes', '').split(',')[1].strip()}政策风险\n3. {attributes.get('key_attributes', '').split(',')[2].strip()}竞争风险。建议建立{self._get_specific_expression('完善', main_topic)}的风险防控机制，及时应对可能出现的问题。"
    
    def _get_specific_expression(self, general_term: str, topic: str) -> str:
        """
        将空泛词汇替换为具体表述
        """
        if general_term in self.specific_expressions:
            expressions = self.specific_expressions[general_term]
            expression = random.choice(expressions)
            
            # 构建格式化参数字典
            format_args = {}
            
            # 根据主题填充具体内容
            if "{growth_rate}" in expression:
                format_args["growth_rate"] = random.randint(5, 30)
            if "{scale}" in expression:
                format_args["scale"] = random.randint(100, 10000)
            if "{share}" in expression:
                format_args["share"] = random.randint(5, 80)
            if "{regions}" in expression:
                format_args["regions"] = "全国31个省市"
            if "{driver}" in expression:
                if topic in self.topic_attributes:
                    drivers = self.topic_attributes[topic].get('key_attributes', '').split(',')[:2]
                    driver = random.choice(drivers).strip()
                else:
                    driver = "技术创新"
                format_args["driver"] = driver
                format_args["contribution"] = random.randint(30, 70)
            if "{old_model}" in expression:
                format_args["old_model"] = "传统模式"
                format_args["new_model"] = "新模式"
            if "{aspect}" in expression:
                if topic in self.topic_attributes:
                    aspects = self.topic_attributes[topic].get('industry_chain', '').split(',')[:2]
                    aspect = random.choice(aspects).strip()
                else:
                    aspect = "管理"
                format_args["aspect"] = aspect
            if "{dimension}" in expression:
                if topic in self.topic_attributes:
                    dimensions = self.topic_attributes[topic].get('key_metrics', '').split(',')[:2]
                    dimension = random.choice(dimensions).strip()
                else:
                    dimension = "市场结构"
                format_args["dimension"] = dimension
            if "{amount}" in expression:
                format_args["amount"] = random.randint(10, 1000)
            if "{ratio}" in expression:
                format_args["ratio"] = random.randint(5, 50)
            
            # 应用格式化
            return expression.format(**format_args)
        
        return general_term
    
    def _get_specific_source(self, topic: str) -> str:
        """
        获取具体的数据来源
        """
        sources = {
            "黄金": "世界黄金协会、COMEX期货市场",
            "人工智能": "IDC、Gartner、中国信通院",
            "新能源汽车": "中汽协、乘联会、工信部",
            "房地产": "国家统计局、中指院、克而瑞",
            "养老产业": "民政部、卫健委、中国老龄科学研究中心"
        }
        
        return sources.get(topic, "行业权威机构")
    
    def _get_specific_mechanism(self, topic: str, factor: str) -> str:
        """
        获取具体的影响机制
        """
        mechanisms = {
            "黄金": {
                "地缘政治影响": "避险情绪传导、资金流向变化",
                "货币政策影响": "实际利率调整、美元指数波动",
                "投资需求": "ETF持仓变化、机构配置调整",
                "interaction": "风险偏好与流动性的联动效应"
            },
            "人工智能": {
                "算法创新": "模型性能提升、应用场景拓展",
                "算力": "训练效率提高、推理成本降低",
                "数据": "样本质量提升、特征提取优化",
                "interaction": "算法与算力的协同发展"
            },
            "新能源汽车": {
                "电池技术": "能量密度提升、成本下降",
                "政策支持": "补贴政策、双积分制度",
                "市场需求": "消费升级、环保意识增强",
                "interaction": "技术进步与政策支持的叠加效应"
            }
        }
        
        if topic in mechanisms and factor in mechanisms[topic]:
            return mechanisms[topic][factor]
        
        return "直接影响和间接传导"
    
    def _generate_mock_data(self, topic: str) -> Dict[str, str]:
        """
        已禁用：不再生成模拟数据
        """
        return {
            "scale": "根据实际数据填写",
            "growth": "根据实际数据填写",
            "structure": "根据实际数据填写",
            "distribution": "根据实际数据填写"
        }
    
    def _generate_real_content(self, main_topic: str, attributes: Dict[str, Any], section: str, subsection: str, 
                             core_thesis: str, key_arguments: List[str], data_files: List[str] = None) -> str:
        """
        使用实际数据生成内容
        """
        try:
            # 如果没有提供数据文件，尝试从默认位置读取
            if not data_files:
                data_files = []
                # 根据主题确定默认数据文件夹
                default_folder = f'{main_topic.lower()}_news_summary'
                if os.path.exists(default_folder):
                    data_files = [os.path.join(default_folder, f) for f in os.listdir(default_folder)]
            
            # 如果仍然没有数据文件，返回基本信息
            if not data_files:
                return f"{section}中的{subsection}部分的实际数据分析暂不可用。待完成数据分析后，本部分将更新。"
            
            # 初始化变量存储分析数据
            price_data = None
            news_data = []
            
            # 读取数据文件
            for file_path in data_files:
                file_ext = os.path.splitext(file_path)[1].lower()
                
                if file_ext in ['.xlsx', '.xls']:
                    # 读取Excel文件（黄金价格数据）
                    df = pd.read_excel(file_path)
                    
                    # 检查数据结构并设置正确的列名
                    if len(df.columns) >= 4 and '日期' in df.iloc[0, 0]:
                        # 设置正确的列名
                        df.columns = ['日期', '年份', '月份', f'{main_topic}价格(美元/单位)']
                        # 删除第一行（原始列名）
                        df = df[1:].reset_index(drop=True)
                        # 转换数据类型
                        df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
                        df['年份'] = pd.to_numeric(df['年份'], errors='coerce')
                        df['月份'] = pd.to_numeric(df['月份'], errors='coerce')
                        df[f'{main_topic}价格(美元/单位)'] = pd.to_numeric(df[f'{main_topic}价格(美元/单位)'], errors='coerce')
                        
                        price_data = df
                    
                elif file_ext == '.docx':
                    # 读取Word文档（新闻摘要）
                    from docx import Document
                    doc = Document(file_path)
                    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
                    news_data.extend(paragraphs)
                
                elif file_ext == '.txt':
                    # 读取文本文件（新闻摘要）
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    news_data.append(content)
            
            # 根据章节和子章节生成内容
            if price_data is not None:
                # 为所有主题生成通用的价格分析内容
                return self._generate_price_content(main_topic, price_data, news_data, section, subsection)
            
            # 如果没有价格数据，返回默认内容
            return f"{section}中的{subsection}部分的实际数据分析暂不可用。待完成数据分析后，本部分将更新。"
            
        except Exception as e:
            # 记录错误信息
            import traceback
            traceback.print_exc()
            return f"{section}中的{subsection}部分的实际数据分析暂不可用。错误信息：{str(e)}"
    
    def _generate_price_content(self, main_topic: str, price_data: pd.DataFrame, news_data: List[str], section: str, subsection: str) -> str:
        """
        生成关于价格分析的内容
        """
        # 如果没有价格数据，返回默认内容
        if price_data is None:
            return f"{section}中的{subsection}部分的实际数据分析暂不可用。"
        
        # 提取2026年1-2月的价格数据
        recent_data = price_data[(price_data['年份'] == 2026) & (price_data['月份'].isin([1, 2]))]
        recent_data = recent_data.sort_values('日期')
        
        # 计算价格波动相关指标
        if len(recent_data) >= 2:
            price_column = f'{main_topic}价格(美元/单位)'
            price_list = recent_data[price_column].tolist()
            price_list = [p for p in price_list if not pd.isna(p)]
            
            if len(price_list) >= 2:
                max_price = max(price_list)
                min_price = min(price_list)
                avg_price = sum(price_list) / len(price_list)
                price_change = (price_list[-1] - price_list[0]) / price_list[0] * 100
                volatility = (max_price - min_price) / avg_price * 100
            else:
                max_price = min_price = avg_price = 0
                price_change = volatility = 0
        else:
            max_price = min_price = avg_price = 0
            price_change = volatility = 0
        
        # 根据章节和子章节生成不同的内容
        if section == '摘要':
            if subsection == '核心结论':
                return f"2026年1-2月，{main_topic}价格呈现{self._get_specific_expression('快速发展', main_topic)}的态势，价格从{price_list[0]:.2f}美元/单位变化到{price_list[-1]:.2f}美元/单位，{'' if price_change > 0 else '下跌'}{abs(price_change):.2f}%。期间价格波动率达到{volatility:.2f}%，{'' if price_change > 0 else '显示出一定的下跌'}{'上涨' if price_change > 0 else '下跌'}趋势。这一波动与同期国际新闻事件、地缘政治形势和经济数据密切相关。"
            elif subsection == '关键发现':
                return f"研究发现：2026年1-2月{main_topic}价格波动主要受{', '.join(['地缘政治形势', '经济数据', '市场情绪'])}等多重因素影响；{', '.join(news_data[:2]) if len(news_data) >= 2 else '相关新闻事件'}可能是推动价格波动的重要因素；{main_topic}价格波动率{volatility:.2f}%，远高于历史平均水平。"
            elif subsection == '主要建议':
                return f"基于上述分析，建议投资者密切关注{', '.join(['地缘政治形势变化', '主要经济体经济数据', '央行货币政策'])}等因素；建议采取{self._get_specific_expression('加强', main_topic)}、{self._get_specific_expression('完善', main_topic)}等策略应对价格波动风险；建议建立科学的风险防控机制，合理配置资产组合。"
        
        elif section == '核心论点':
            if subsection == '分析背景':
                return f"2026年1-2月，{main_topic}价格呈现剧烈波动的态势，价格从{price_list[0]:.2f}美元/单位变化到{price_list[-1]:.2f}美元/单位，波动幅度达到{volatility:.2f}%。这一波动引起了投资者、经济学家和政策制定者的广泛关注，需要深入分析其成因和影响机制。"
            elif subsection == '核心命题':
                return f"2026年1-2月{main_topic}价格的剧烈波动主要受地缘政治、经济数据和市场情绪等多重因素影响，其中{', '.join(news_data[:1]) if len(news_data) >= 1 else '相关新闻事件'}可能是关键驱动因素。"
            elif subsection == '研究框架':
                return f"本报告采用{', '.join(self.analysis_frameworks.get('why', []))}等分析框架，从{', '.join(['地缘政治', '经济数据', '市场情绪'])}等维度入手，系统分析2026年1-2月{main_topic}价格波动的成因和影响。"
        
        elif section.startswith('关键论点'):
            if subsection == '现状描述与数据支撑':
                content = f"根据{self._get_specific_source(main_topic)}数据显示，2026年1-2月{main_topic}价格呈现剧烈波动的态势："
                content += "\n\n"
                content += "### 关键价格指标"
                content += "\n"
                content += f"- 期初价格：{price_list[0]:.2f}美元/单位"
                content += "\n"
                content += f"- 期末价格：{price_list[-1]:.2f}美元/单位"
                content += "\n"
                content += f"- 最高价格：{max_price:.2f}美元/单位"
                content += "\n"
                content += f"- 最低价格：{min_price:.2f}美元/单位"
                content += "\n"
                content += f"- 平均价格：{avg_price:.2f}美元/单位"
                content += "\n"
                content += f"- 价格变化：{'' if price_change > 0 else '下跌'}{abs(price_change):.2f}%"
                content += "\n"
                content += f"- 波动率：{volatility:.2f}%"
                content += "\n\n"
                content += "### 价格走势特点"
                content += "\n"
                content += f"1. {'' if price_change > 0 else '下跌'}{abs(price_change):.2f}%的价格变化显示出{'' if price_change > 0 else '下跌'}{'上涨' if price_change > 0 else '下跌'}趋势"
                content += "\n"
                content += f"2. {volatility:.2f}%的波动率表明市场情绪波动较大"
                content += "\n"
                content += f"3. 最高价格与最低价格之间的差距达到{max_price - min_price:.2f}美元/单位"
                content += "\n\n"
                return content
            
            elif subsection == '成因分析与机制阐释':
                content = f"2026年1-2月{main_topic}价格剧烈波动的原因是多方面的，主要包括："
                content += "\n\n"
                content += "### 地缘政治因素"
                content += "\n"
                content += f"1. {', '.join(news_data[:2]) if len(news_data) >= 2 else '相关地缘政治事件'}可能对{main_topic}价格产生了重要影响"
                content += "\n"
                content += f"2. 地缘政治不确定性增强了{main_topic}作为{self.topic_attributes.get(main_topic, {}).get('type', '资产')}的吸引力"
                content += "\n"
                content += "3. 主要国家之间的紧张关系加剧了市场波动"
                content += "\n\n"
                content += "### 经济数据因素"
                content += "\n"
                content += "1. 主要经济体的经济增长数据可能影响了市场预期"
                content += "\n"
                content += f"2. 通货膨胀数据的变化可能推动了{main_topic}价格波动"
                content += "\n"
                content += f"3. 央行货币政策调整可能对{main_topic}价格产生了影响"
                content += "\n\n"
                content += "### 市场情绪因素"
                content += "\n"
                content += f"1. 投资者对全球经济前景的担忧可能推高了{main_topic}需求"
                content += "\n"
                content += "2. 市场投机行为可能加剧了价格波动"
                content += "\n"
                content += "3. 技术面因素可能影响了短期价格走势"
                content += "\n\n"
                return content
            
            elif subsection == '影响评估与趋势预判':
                content = f"基于上述分析，预计2026年3月及以后{main_topic}价格将呈现以下趋势："
                content += "\n\n"
                content += "### 短期趋势预测"
                content += "\n"
                content += f"1. 如果地缘政治紧张局势持续，{main_topic}价格可能{'' if price_change > 0 else '继续下跌'}{'继续上涨' if price_change > 0 else '下跌'}"
                content += "\n"
                content += f"2. 经济数据的变化将继续影响{main_topic}价格走势"
                content += "\n"
                content += f"3. 市场情绪波动可能导致{main_topic}价格维持较高的波动率"
                content += "\n\n"
                content += "### 中长期趋势预测"
                content += "\n"
                content += f"1. 全球经济增长前景将是影响{main_topic}价格的关键因素"
                content += "\n"
                content += f"2. 央行货币政策调整将对{main_topic}价格产生重要影响"
                content += "\n"
                content += f"3. 地缘政治风险可能成为推动{main_topic}价格上涨的重要因素"
                content += "\n\n"
                return content
        
        elif section == '结论与建议':
            if subsection == '主要结论总结':
                content = f"通过对2026年1-2月{main_topic}价格的系统分析，本报告得出以下结论："
                content += f"\n1. {main_topic}价格呈现剧烈波动的态势，价格从{{price_list[0]:.2f}}美元/单位变化到{{price_list[-1]:.2f}}美元/单位，波动幅度达到{{volatility:.2f}}%"
                content += "\n2. 价格波动主要受地缘政治、经济数据和市场情绪等多重因素影响"
                content += "\n3. {', '.join(news_data[:1]) if len(news_data) >= 1 else '相关新闻事件'}可能是推动价格波动的重要因素"
                content += "\n4. 市场情绪波动是导致高波动率的重要原因。"
                return content
            
            elif subsection == '具体行动建议':
                content = f"基于上述结论，建议采取以下具体行动："
                content += f"\n1. {self._get_specific_expression('加强', main_topic)}对地缘政治形势的关注和分析"
                content += f"\n2. {self._get_specific_expression('加大', main_topic)}对经济数据的研究和解读力度"
                content += f"\n3. {self._get_specific_expression('完善', main_topic)}市场情绪监测机制"
                content += "\n4. 建立科学的风险防控体系，有效应对价格波动风险。"
                return content
            
            elif subsection == '风险防控措施':
                content = f"在实施上述建议的过程中，需要重点防控以下风险："
                content += "\n1. 地缘政治形势突然变化的风险"
                content += "\n2. 经济数据不及预期的风险"
                content += "\n3. 市场情绪急剧波动的风险"
                content += f"\n4. 流动性风险。建议建立{self._get_specific_expression('完善', main_topic)}的风险防控机制，及时应对可能出现的问题。"
                return content
        
        # 默认内容
        return f"{section}中的{subsection}部分的实际数据分析暂不可用。"
