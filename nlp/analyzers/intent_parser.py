#!/usr/bin/env python3
"""
三层意图解析器：表层→深层→领域
"""

import re
from typing import Dict, List, Set, Any
import logging

# 配置日志
logger = logging.getLogger(__name__)

class IntentParser:
    """三层意图解析器：表层→深层→领域"""
    
    def __init__(self):
        # 初始化关键词库
        self._init_keyword_libraries()
        # 意图优先级配置（数值越小，优先级越高）
        self.intent_priority = {"predict": 1, "suggest": 2, "evaluate": 3, "analyze": 4, "compare": 5, "trend": 6, "summarize": 7}
        # 初始化缓存机制
        self.cache = {}
        self.cache_max_size = 1000  # 缓存最大条数
        self.cache_hits = 0         # 缓存命中次数
        self.cache_misses = 0       # 缓存未命中次数
        
    def _init_keyword_libraries(self):
        """初始化各种关键词库"""
        # 意图关键词
        self.intent_keywords = {
            "analyze": ["分析", "研究", "评估", "解析", "探讨", "考察", "审视"],
            "summarize": ["总结", "概括", "摘要", "概述", "总览", "归纳"],
            "compare": ["比较", "对比", "对比分析", "差异", "不同"],
            "trend": ["趋势", "走势", "变化", "发展", "动态", "趋势分析"],
            "predict": ["预测", "预估", "展望", "未来", "预期", "可能"],
            "evaluate": ["评估", "评价", "评测", "考量", "判断"],
            "suggest": ["建议", "推荐", "提出", "方案", "策略", "措施"]
        }
        
        # 问题类型关键词
        self.question_type_keywords = {
            "explanation": ["是什么", "什么是", "定义", "含义", "概念", "指的是"],
            "method": ["如何", "怎样", "方法", "步骤", "做法", "操作"],
            "analysis": ["为什么", "原因", "理由", "为何", "因素", "影响"]
        }
        
        # 核心概念关键词
        self.core_concept_keywords = {
            "volatility_monitoring": ["波动率监控", "波动监控", "波动率管理", "波动管理"],
            "risk_control": ["风险管控", "风险管理", "风险控制", "风险防范"],
            "gold": ["黄金", "金价", "黄金市场"]
        }
        
        # 领域关键词
        self.domain_keywords = {
            "finance": ["金融", "财务", "资金", "投资", "股票", "债券", "基金", "汇率"],
            "gold": ["黄金", "金价", "金矿", "黄金市场", "黄金价格"],
            "product": ["产品", "商品", "服务", "业务", "项目", "产品线"],
            "user": ["用户", "客户", "消费者", "用户群", "客户群"],
            "marketing": ["营销", "推广", "销售", "市场", "渠道", "市场份额"]
        }
        
        # 时间关键词
        self.time_keywords = {
            "week": ["本周", "上周", "每周", "周", "weekly"],
            "month": ["本月", "上月", "每月", "月", "monthly"],
            "quarter": ["本季度", "上季度", "季度", "季度报告", "quarterly"],
            "year": ["本年", "去年", "每年", "年", "年度", "yearly"]
        }
        
        # 量化指标关键词
        self.metric_keywords = {
            "volume": ["交易量", "成交量", "交易金额", "销售额", "销量"],
            "growth": ["增长率", "增长", "增长幅度", "涨幅", "增速"],
            "price": ["价格", "单价", "成本", "售价"],
            "profit": ["利润", "利润率", "盈利", "收益"],
            "market_share": ["市场份额", "占有率", "占比"]
        }
    
    def _extract_question_type(self, query: str) -> str:
        """
        提取问题类型
        
        Args:
            query: 用户查询字符串
            
        Returns:
            str: 问题类型（explanation/method/analysis）
        """
        for question_type, keywords in self.question_type_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    return question_type
        return "explanation"  # 默认解释型问题
    
    def _extract_core_concepts(self, query: str) -> List[str]:
        """
        提取核心概念
        
        Args:
            query: 用户查询字符串
            
        Returns:
            List[str]: 核心概念列表
        """
        core_concepts = []
        for concept_type, keywords in self.core_concept_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    core_concepts.append(concept_type)
        return list(set(core_concepts)) if core_concepts else []
    
    def _resolve_intent(self, intents: List[str]) -> str:
        """
        基于优先级选择主要意图
        
        Args:
            intents: 检测到的意图列表
            
        Returns:
            str: 主要意图（优先级最高的）
        """
        if not intents:
            return "analyze"  # 默认意图
        # 根据优先级选择最高的意图
        return min(intents, key=lambda x: self.intent_priority.get(x, 99))
    
    def parse(self, query: str) -> Dict:
        """
        解析用户查询的意图
        
        Args:
            query: 用户查询字符串
            
        Returns:
            Dict: 包含表层、深层、领域和指标的意图解析结果
        """
        try:
            # 检查缓存
            if query in self.cache:
                self.cache_hits += 1
                return self.cache[query]
            
            self.cache_misses += 1
            # 1. 表层解析：关键词提取
            surface_analysis = self._extract_keywords(query)
            
            # 2. 确定主要意图（基于优先级）
            primary_intent = self._resolve_intent(surface_analysis.get("intents", []))
            
            # 3. 深层解析：推断真实需求
            deep_analysis = self._infer_real_needs(query, surface_analysis)
            
            # 4. 领域映射
            domain_analysis = self._map_to_domain(query, surface_analysis)
            
            # 5. 量化指标提取
            metrics_analysis = self._extract_metrics(query, surface_analysis)
            
            # 6. 时间范围提取
            time_analysis = self._extract_time_range(query)
            
            # 7. 问题类型提取
            question_type = self._extract_question_type(query)
            
            # 8. 核心概念提取
            core_concepts = self._extract_core_concepts(query)
            
            # 9. 构建回答框架
            answer_framework = self._build_answer_framework(question_type, core_concepts)
            
            # 10. 计算置信度
            confidence = self._calc_confidence(query, surface_analysis)
            
            # 构建解析结果
            result = {
                "surface": surface_analysis,
                "deep": deep_analysis,
                "domain": domain_analysis,
                "metrics": metrics_analysis,
                "time": time_analysis,
                "question_type": question_type,
                "core_concepts": core_concepts,
                "answer_framework": answer_framework,
                "confidence": confidence
            }
            
            # 兼容旧版API
            if "intents" in surface_analysis:
                result["intent"] = primary_intent
            
            # 更新缓存
            if len(self.cache) >= self.cache_max_size:
                # 移除最旧的缓存项（FIFO策略）
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[query] = result
            
            return result
            
        except Exception as e:
            logger.error(f"意图解析失败: {str(e)}")
            # 优雅回退机制
            return self._fallback(query)
    
    def _fallback(self, query):
        """
        解析失败时的优雅回退
        
        Args:
            query: 用户查询字符串
            
        Returns:
            Dict: 基础解析结果
        """
        result = {
            "surface": {},
            "deep": "分析用户提供的内容",
            "domain": {"primary": "general", "secondary": [], "confidence": 0.3},
            "metrics": [],
            "time": {"type": "recent", "range": "最近"},
            "question_type": "explanation",
            "core_concepts": [],
            "answer_framework": [],
            "confidence": {"overall": 0.3},
            "is_fallback": True,
            "intent": "analyze"
        }
        
        # 更新缓存
        if len(self.cache) >= self.cache_max_size:
            # 移除最旧的缓存项（FIFO策略）
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[query] = result
        
        return result
    
    def get_cache_stats(self):
        """
        获取缓存统计信息
        
        Returns:
            Dict: 缓存统计信息
        """
        total = self.cache_hits + self.cache_misses
        hit_rate = round(self.cache_hits / max(total, 1) * 100, 2)
        
        return {
            "total_queries": total,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "max_cache_size": self.cache_max_size
        }
    
    def _build_answer_framework(self, question_type: str, core_concepts: List[str]) -> List[str]:
        """
        构建回答框架
        
        Args:
            question_type: 问题类型
            core_concepts: 核心概念列表
            
        Returns:
            List[str]: 回答框架结构
        """
        # 基于问题类型和核心概念构建回答框架
        if question_type == "explanation":
            return [
                "定义解释",
                "核心要素",
                "应用场景",
                "与其他概念的区别"
            ]
        elif question_type == "method":
            return [
                "方法步骤",
                "关键技术",
                "实施工具",
                "注意事项"
            ]
        elif question_type == "analysis":
            return [
                "影响因素",
                "作用机制",
                "实证分析",
                "结论建议"
            ]
        else:
            return [
                "问题分析",
                "解决方案",
                "实施建议",
                "效果评估"
            ]
    
    def _extract_keywords(self, query: str) -> Dict:
        """
        提取表层关键词
        
        Args:
            query: 用户查询字符串
            
        Returns:
            Dict: 包含各种关键词类别的字典
        """
        result = {
            "intents": [],
            "domains": [],
            "time": [],
            "metrics": [],
            "keywords": []
        }
        
        # 提取意图关键词
        for intent, keywords in self.intent_keywords.items():
            for keyword in keywords:
                if self._fuzzy_match(query, keyword):
                    result["intents"].append(intent)
                    result["keywords"].append(keyword)
        
        # 提取领域关键词
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if self._fuzzy_match(query, keyword):
                    result["domains"].append(domain)
                    result["keywords"].append(keyword)
        
        # 提取时间关键词
        for time_type, keywords in self.time_keywords.items():
            for keyword in keywords:
                if self._fuzzy_match(query, keyword):
                    result["time"].append(time_type)
                    result["keywords"].append(keyword)
        
        # 提取指标关键词
        for metric, keywords in self.metric_keywords.items():
            for keyword in keywords:
                if self._fuzzy_match(query, keyword):
                    result["metrics"].append(metric)
                    result["keywords"].append(keyword)
        
        # 去重
        for key in result:
            if isinstance(result[key], list):
                result[key] = list(set(result[key]))
        
        return result
    
    def _infer_real_needs(self, query: str, surface: Dict) -> str:
        """
        推断深层需求，增强对模糊指令的处理能力
        
        Args:
            query: 用户查询字符串
            surface: 表层分析结果
            
        Returns:
            str: 深层需求描述
        """
        # 基于意图关键词和时间关键词推断深层需求
        if surface["intents"]:
            intent = surface["intents"][0]  # 取主要意图
        else:
            # 从查询中推断意图
            query_lower = query.lower()
            if any(word in query_lower for word in ["总结", "概括", "摘要", "概述", "总览", "归纳"]):
                intent = "summarize"
            elif any(word in query_lower for word in ["比较", "对比", "差异", "不同"]):
                intent = "compare"
            elif any(word in query_lower for word in ["趋势", "走势", "变化", "发展", "动态"]):
                intent = "trend"
            elif any(word in query_lower for word in ["预测", "预估", "展望", "未来", "预期"]):
                intent = "predict"
            elif any(word in query_lower for word in ["建议", "推荐", "方案", "策略", "措施"]):
                intent = "suggest"
            else:
                intent = "analyze"  # 默认意图
        
        # 基于领域关键词推断领域
        if surface["domains"]:
            domain = surface["domains"][0]
        else:
            # 从查询中推断领域
            query_lower = query.lower()
            if any(word in query_lower for word in ["黄金", "金价", "黄金市场"]):
                domain = "gold"
            elif any(word in query_lower for word in ["金融", "财务", "资金", "投资", "股票", "债券", "基金", "汇率"]):
                domain = "finance"
            elif any(word in query_lower for word in ["产品", "商品", "服务", "业务", "项目"]):
                domain = "product"
            elif any(word in query_lower for word in ["用户", "客户", "消费者", "用户群"]):
                domain = "user"
            elif any(word in query_lower for word in ["营销", "推广", "销售", "市场", "渠道"]):
                domain = "marketing"
            else:
                domain = "general"
        
        # 基于时间关键词推断时间范围
        if surface["time"]:
            time = surface["time"][0]
        else:
            # 从查询中推断时间范围
            query_lower = query.lower()
            if any(word in query_lower for word in ["本周", "上周", "每周", "周"]):
                time = "week"
            elif any(word in query_lower for word in ["本月", "上月", "每月", "月"]):
                time = "month"
            elif any(word in query_lower for word in ["本季度", "上季度", "季度"]):
                time = "quarter"
            elif any(word in query_lower for word in ["本年", "去年", "每年", "年", "年度"]):
                time = "year"
            else:
                time = "recent"
        
        # 构建深层需求
        需求模板 = {
            "analyze": {
                "finance": f"分析{self._get_time_desc(time)}金融市场的运行情况和关键指标",
                "gold": f"分析{self._get_time_desc(time)}黄金价格的波动趋势和影响因素",
                "product": f"分析{self._get_time_desc(time)}产品的表现和市场反馈",
                "user": f"分析{self._get_time_desc(time)}用户行为特征和偏好变化",
                "marketing": f"分析{self._get_time_desc(time)}营销活动的效果和 ROI",
                "general": f"分析{self._get_time_desc(time)}提供的内容和数据"
            },
            "summarize": {
                "finance": f"总结{self._get_time_desc(time)}金融市场的重要事件和趋势",
                "gold": f"总结{self._get_time_desc(time)}黄金价格的主要变化和影响因素",
                "product": f"总结{self._get_time_desc(time)}产品的核心亮点和关键数据",
                "user": f"总结{self._get_time_desc(time)}用户的主要行为模式和需求",
                "marketing": f"总结{self._get_time_desc(time)}营销活动的核心成果和经验",
                "general": f"总结{self._get_time_desc(time)}提供的内容要点"
            },
            "compare": {
                "finance": f"比较{self._get_time_desc(time)}不同金融产品或市场的表现差异",
                "gold": f"比较{self._get_time_desc(time)}黄金与其他资产的表现差异",
                "product": f"比较{self._get_time_desc(time)}不同产品或版本的表现差异",
                "user": f"比较{self._get_time_desc(time)}不同用户群体的行为差异",
                "marketing": f"比较{self._get_time_desc(time)}不同营销渠道或策略的效果差异",
                "general": f"比较提供的不同内容或数据之间的差异"
            },
            "trend": {
                "finance": f"分析{self._get_time_desc(time)}金融市场的发展趋势和未来走向",
                "gold": f"分析{self._get_time_desc(time)}黄金价格的变化趋势和未来展望",
                "product": f"分析{self._get_time_desc(time)}产品的市场趋势和发展方向",
                "user": f"分析{self._get_time_desc(time)}用户需求的变化趋势和发展方向",
                "marketing": f"分析{self._get_time_desc(time)}营销行业的发展趋势和创新方向",
                "general": f"分析提供内容中反映的趋势和变化"
            },
            "suggest": {
                "finance": f"基于{self._get_time_desc(time)}的金融数据提供投资建议",
                "gold": f"基于{self._get_time_desc(time)}的黄金价格走势提供策略建议",
                "product": f"基于{self._get_time_desc(time)}的产品表现提供改进建议",
                "user": f"基于{self._get_time_desc(time)}的用户数据提供优化建议",
                "marketing": f"基于{self._get_time_desc(time)}的营销效果提供策略建议",
                "general": f"基于提供的内容提供相关建议"
            },
            "predict": {
                "finance": f"预测{self._get_time_desc(time)}金融市场的未来发展和投资机会",
                "gold": f"预测{self._get_time_desc(time)}黄金价格的未来走势和投资价值",
                "product": f"预测{self._get_time_desc(time)}产品的市场前景和发展潜力",
                "user": f"预测{self._get_time_desc(time)}用户需求的变化趋势和发展方向",
                "marketing": f"预测{self._get_time_desc(time)}营销行业的发展趋势和创新方向",
                "general": f"基于提供的内容预测未来趋势和发展方向"
            }
        }
        
        # 获取对应的需求描述
        return 需求模板.get(intent, {}).get(domain, "分析提供的内容")
    
    def _map_to_domain(self, query: str, surface: Dict) -> Dict:
        """
        领域映射
        
        Args:
            query: 用户查询字符串
            surface: 表层分析结果
            
        Returns:
            Dict: 领域映射结果
        """
        # 确定主要领域
        if surface["domains"]:
            primary_domain = surface["domains"][0]
        else:
            primary_domain = "general"
        
        # 确定辅助领域
        secondary_domains = surface["domains"][1:] if len(surface["domains"]) > 1 else []
        
        return {
            "primary": primary_domain,
            "secondary": secondary_domains,
            "confidence": self._calculate_domain_confidence(query, primary_domain)
        }
    
    def _extract_metrics(self, query: str, surface: Dict) -> List[Dict]:
        """
        提取量化指标
        
        Args:
            query: 用户查询字符串
            surface: 表层分析结果
            
        Returns:
            List[Dict]: 量化指标列表
        """
        metrics = []
        
        # 基于关键词提取指标
        for metric in surface["metrics"]:
            metric_info = {
                "type": metric,
                "name": self._get_metric_name(metric),
                "confidence": 0.8  # 默认置信度
            }
            metrics.append(metric_info)
        
        # 提取具体数值
        numeric_values = self._extract_numeric_values(query)
        if numeric_values:
            metrics.append({
                "type": "numeric",
                "name": "具体数值",
                "values": numeric_values,
                "confidence": 0.9
            })
        
        # 提取字数要求
        word_count_requirement = self._extract_word_count_requirement(query)
        if word_count_requirement:
            metrics.append({
                "type": "word_count",
                "name": "字数要求",
                "requirement": word_count_requirement,
                "confidence": 0.95
            })
        
        return metrics
    
    def _extract_time_range(self, query: str) -> Dict:
        """
        提取时间范围
        
        Args:
            query: 用户查询字符串
            
        Returns:
            Dict: 时间范围信息
        """
        import re
        
        # 1. 正则匹配时间范围（如：从1月到3月，自2024年1月到2024年6月）
        range_match = re.search(r'(从|自)(\d+月|\d{4}年\d+月)(到|至)(\d+月|\d{4}年\d+月)', query)
        if range_match:
            start = range_match.group(2)
            end = range_match.group(4)
            return {
                "type": "range",
                "range": f"{start}到{end}",
                "start": start,
                "end": end
            }
        
        # 2. 正则匹配具体时间范围（如：过去30天、最近2周、过去一年半）
        # 先匹配复合时间单位（支持中文和阿拉伯数字）
        chinese_to_arabic = {
            "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
            "六": 6, "七": 7, "八": 8, "九": 9, "十": 10
        }
        
        # 匹配中文数字
        compound_match_zh = re.search(r'(过去|最近|前|上)([一二三四五六七八九十])年半', query)
        if compound_match_zh:
            zh_num = compound_match_zh.group(2)
            value = chinese_to_arabic.get(zh_num, 1)
            return {
                "type": "specific",
                "range": f"{zh_num}年半",
                "value": value,
                "unit": "年",
                "is_compound": True
            }
        
        # 匹配阿拉伯数字
        compound_match = re.search(r'(过去|最近|前|上)(\d+)年半', query)
        if compound_match:
            value = int(compound_match.group(2))
            return {
                "type": "specific",
                "range": f"{value}年半",
                "value": value,
                "unit": "年",
                "is_compound": True
            }
        
        # 再匹配普通时间单位
        match = re.search(r'(过去|最近|前|上)(\d+)(天|周|月|年)', query)
        if match:
            unit = match.group(3)
            value = int(match.group(2))
            return {
                "type": "specific",
                "range": f"{value}{unit}",
                "value": value,
                "unit": unit
            }
        
        # 3. 精确匹配季度（包含具体季度数字）
        quarter_pattern = r'(第[一二三四]季度|Q[1234])'
        quarter_match = re.search(quarter_pattern, query)
        if quarter_match:
            quarter_text = quarter_match.group(0)
            return {
                "type": "quarter",
                "range": quarter_text,
                "quarter": quarter_text
            }
        
        # 4. 检查是否包含数字+季度的模式（如：1季度）
        number_quarter_match = re.search(r'(\d+季度)', query)
        if number_quarter_match:
            return {
                "type": "quarter",
                "range": number_quarter_match.group(0),
                "quarter": number_quarter_match.group(0)
            }
        
        # 4. 正则匹配具体年月
        date_match = re.search(r'(\d{4})年(\d{1,2})月', query)
        if date_match:
            year = int(date_match.group(1))
            month = int(date_match.group(2))
            return {
                "type": "specific_date",
                "range": f"{year}年{month}月",
                "year": year,
                "month": month
            }
        
        # 5. 基础关键词匹配（排除已处理的情况）
        for t, keywords in self.time_keywords.items():
            # 跳过季度，因为已经单独处理
            if t == "quarter":
                continue
            for keyword in keywords:
                if keyword in query:
                    return {
                        "type": t,
                        "range": self._get_time_desc(t)
                    }
        
        # 6. 专门处理季度关键词
        quarter_keywords = self.time_keywords.get("quarter", [])
        for keyword in quarter_keywords:
            if keyword in query and not re.search(quarter_pattern, query):
                return {
                    "type": "quarter",
                    "range": self._get_time_desc("quarter"),
                    "quarter": self._get_time_desc("quarter")
                }
        
        # 7. 默认返回
        return {
            "type": "recent",
            "range": "最近"
        }
    
    def _calculate_domain_confidence(self, query: str, domain: str) -> float:
        """
        计算领域置信度
        
        Args:
            query: 用户查询字符串
            domain: 领域名称
            
        Returns:
            float: 置信度（0-1）
        """
        if domain not in self.domain_keywords:
            return 0.5
        
        # 计算匹配的关键词数量
        matching_keywords = 0
        total_keywords = len(self.domain_keywords[domain])
        
        for keyword in self.domain_keywords[domain]:
            if keyword in query:
                matching_keywords += 1
        
        # 计算置信度
        confidence = matching_keywords / max(total_keywords, 1)
        return min(confidence * 0.8 + 0.2, 1.0)  # 最低置信度0.2
    
    def _calc_confidence(self, query: str, surface_analysis: Dict) -> Dict:
        """
        计算整体解析置信度
        
        Args:
            query: 用户查询字符串
            surface_analysis: 表层分析结果
            
        Returns:
            Dict: 各维度置信度和整体置信度
        """
        import re
        
        # 1. 关键词密度：匹配关键词数量与查询长度的比例
        total_matched = len(surface_analysis.get("keywords", []))
        query_length = len(query)
        keyword_density = min(total_matched / max(query_length, 1), 1.0)
        
        # 2. 意图置信度：基于检测到的意图数量
        intent_count = len(surface_analysis.get("intents", []))
        intent_confidence = min(intent_count * 0.3, 0.9)
        
        # 3. 领域置信度：基于匹配的领域关键词
        domain_confidence = 0.6
        if surface_analysis.get("domains"):
            primary_domain = surface_analysis["domains"][0]
            domain_confidence = self._calculate_domain_confidence(query, primary_domain)
        
        # 4. 时间置信度：基于是否检测到时间信息
        time_confidence = min(len(surface_analysis.get("time", [])) * 0.4, 1.0)
        
        # 5. 指标置信度：基于检测到的指标数量
        metrics_confidence = min(len(surface_analysis.get("metrics", [])) * 0.25, 0.8)
        
        # 6. 位置权重：句首关键词赋予更高权重
        position_weight = 0.0
        for keyword in surface_analysis.get("keywords", []):
            if query.startswith(keyword) or query[:10].find(keyword) != -1:
                position_weight += 0.1
        position_weight = min(position_weight, 0.3)
        
        # 7. 组合模式：特定关键词组合增强置信度
        combination_bonus = 0.0
        if "分析" in query and any(domain_keyword in query for domain in ["gold", "finance"] for domain_keyword in self.domain_keywords.get(domain, [])):
            combination_bonus += 0.1
        if "预测" in query and "趋势" in query:
            combination_bonus += 0.1
        combination_bonus = min(combination_bonus, 0.2)
        
        # 8. 否定检测：识别否定词降低相关意图的置信度
        negative_words = ["不", "没", "无", "非", "不是"]
        has_negative = any(word in query for word in negative_words)
        negative_penalty = 0.2 if has_negative else 0.0
        
        # 计算整体置信度：加权平均
        overall = (intent_confidence * 0.4 + 
                  domain_confidence * 0.3 + 
                  time_confidence * 0.2 + 
                  metrics_confidence * 0.1 + 
                  keyword_density * 0.2 + 
                  position_weight + 
                  combination_bonus - 
                  negative_penalty)
        
        # 限制置信度在0-1范围内
        overall = max(0.0, min(overall, 1.0))
        
        return {
            "overall": round(overall, 2),
            "intent": round(intent_confidence, 2),
            "domain": round(domain_confidence, 2),
            "time": round(time_confidence, 2),
            "metrics": round(metrics_confidence, 2),
            "keyword_density": round(keyword_density, 2),
            "position_weight": round(position_weight, 2),
            "combination_bonus": round(combination_bonus, 2),
            "negative_penalty": round(negative_penalty, 2)
        }
    
    def _get_time_desc(self, time_type: str) -> str:
        """
        获取时间描述
        
        Args:
            time_type: 时间类型
            
        Returns:
            str: 时间描述
        """
        time_descriptions = {
            "week": "本周",
            "month": "本月",
            "quarter": "本季度",
            "year": "本年",
            "recent": "最近"
        }
        return time_descriptions.get(time_type, "最近")
    
    def _get_metric_name(self, metric_type: str) -> str:
        """
        获取指标名称
        
        Args:
            metric_type: 指标类型
            
        Returns:
            str: 指标名称
        """
        metric_names = {
            "volume": "交易量",
            "growth": "增长率",
            "price": "价格",
            "profit": "利润",
            "market_share": "市场份额"
        }
        return metric_names.get(metric_type, metric_type)
    
    def _extract_numeric_values(self, query: str) -> List[float]:
        """
        提取文本中的数值
        
        Args:
            query: 用户查询字符串
            
        Returns:
            List[float]: 提取的数值列表
        """
        # 使用正则表达式提取数值
        numeric_pattern = r'\d+\.?\d*'
        matches = re.findall(numeric_pattern, query)
        
        # 转换为浮点数
        numeric_values = []
        for match in matches:
            try:
                numeric_values.append(float(match))
            except ValueError:
                continue
        
        return numeric_values
    
    def _extract_word_count_requirement(self, query: str) -> Dict[str, Any]:
        """
        提取用户查询中的字数要求
        
        Args:
            query: 用户查询字符串
            
        Returns:
            Dict: 字数要求信息，如果没有则返回None
        """
        import re
        
        # 匹配字数要求的正则表达式
        word_count_patterns = [
            r'(\d+)\s*(字|字符|个字|字符数)(以上|以内|以内|以下|之间|左右)',  # 如：4000字以上
            r'(\d+)\s*(字|字符|个字|字符数)',  # 如：4000字
            r'(\d+)\s*word\s*(以上|以内|以下|之间|左右)',  # 如：4000 word以上
            r'(\d+)\s*words?',  # 如：4000 words
        ]
        
        for pattern in word_count_patterns:
            match = re.search(pattern, query)
            if match:
                groups = match.groups()
                
                # 提取字数
                word_count = int(groups[0])
                
                # 提取条件（如果有）
                condition = groups[2] if len(groups) > 2 and groups[2] else "以上"
                
                return {
                    "word_count": word_count,
                    "condition": condition,
                    "original": match.group(0)
                }
        
        return None
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        计算编辑距离
        
        Args:
            s1: 字符串1
            s2: 字符串2
            
        Returns:
            int: 编辑距离
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        # 处理空字符串的情况
        if len(s2) == 0:
            return len(s1)
        
        # 初始化距离矩阵
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _fuzzy_match(self, query: str, keyword: str, max_distance: int = None) -> bool:
        """
        模糊匹配关键词
        
        Args:
            query: 用户查询字符串
            keyword: 关键词
            max_distance: 最大编辑距离，默认为关键词长度的1/3
            
        Returns:
            bool: 是否匹配
        """
        # 直接匹配
        if keyword in query:
            return True
        
        # 编辑距离匹配
        query_length = len(query)
        keyword_length = len(keyword)
        
        # 如果关键词太长，不适合模糊匹配
        if keyword_length > query_length + 2:
            return False
        
        # 根据关键词长度动态调整最大编辑距离
        if max_distance is None:
            max_distance = max(1, int(keyword_length / 3))
            # 短关键词使用更小的最大编辑距离
            if keyword_length <= 2:
                max_distance = 1
        
        # 在查询中滑动窗口检查编辑距离
        for i in range(0, query_length - keyword_length + 1):
            window = query[i:i + keyword_length]
            distance = self._levenshtein_distance(window, keyword)
            if distance <= max_distance:
                return True
        
        return False

# 测试代码
if __name__ == "__main__":
    # 创建意图解析器实例
    parser = IntentParser()
    
    # 测试查询
    test_queries = [
        "分析本周市场价格的走势和影响因素",
        "总结本月金融市场的主要变化",
        "比较不同产品的用户满意度",
        "预测未来市场价格的可能走势",
        "评估营销活动的效果和ROI"
    ]
    
    # 测试解析
    for query in test_queries:
        print(f"\n查询: {query}")
        result = parser.parse(query)
        print(f"解析结果:")
        for key, value in result.items():
            print(f"  {key}: {value}")
