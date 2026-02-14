#!/usr/bin/env python3
"""
领域知识库：存储和管理领域专业知识
"""

from typing import Dict, List, Tuple, Any, Set
import logging
import json
from collections import defaultdict

# 配置日志
logger = logging.getLogger(__name__)

class KnowledgeBase:
    """领域知识库：存储和管理领域专业知识"""
    
    def __init__(self, knowledge_file: str = None):
        # 初始化知识库
        self._init_knowledge_structure()
        
        # 如果提供了知识文件，加载知识
        if knowledge_file:
            self.load_knowledge(knowledge_file)
        else:
            # 加载默认知识
            self._load_default_knowledge()
    
    def _init_knowledge_structure(self):
        """
        初始化知识库结构
        """
        self.knowledge = {
            "domains": {},  # 领域知识
            "terms": {},  # 术语库
            "rules": [],  # 规则库
            "faq": [],  # 常见问题
            "relationships": []  # 关系知识
        }
        
        # 建立索引以提高查询效率
        self.term_index = defaultdict(set)  # 术语到领域的映射
        self.domain_term_index = defaultdict(set)  # 领域到术语的映射
    
    def _load_default_knowledge(self):
        """
        加载默认知识
        """
        # 默认领域知识
        default_knowledge = {
            "domains": {
                "gold": {
                    "name": "黄金市场",
                    "description": "关于黄金价格、交易和市场分析的知识",
                    "keywords": ["黄金", "金价", "黄金市场", "黄金交易", "黄金投资"]
                },
                "finance": {
                    "name": "金融市场",
                    "description": "关于金融产品、市场和分析的知识",
                    "keywords": ["金融", "股票", "债券", "基金", "汇率", "利率"]
                },
                "marketing": {
                    "name": "市场营销",
                    "description": "关于市场推广、营销策略和分析的知识",
                    "keywords": ["营销", "市场", "推广", "销售", "客户", "品牌"]
                },
                "product": {
                    "name": "产品管理",
                    "description": "关于产品开发、管理和分析的知识",
                    "keywords": ["产品", "开发", "管理", "功能", "用户体验"]
                }
            },
            
            "terms": {
                # 黄金领域术语
                "黄金ETF": {
                    "domain": "gold",
                    "definition": "黄金交易所交易基金，是一种以黄金为基础资产的交易工具",
                    "synonyms": ["黄金指数基金", "黄金交易所基金"],
                    "context": "用于跟踪黄金价格走势，方便投资者参与黄金投资"
                },
                "伦敦金": {
                    "domain": "gold",
                    "definition": "伦敦黄金市场交易的黄金，通常指现货黄金交易",
                    "synonyms": ["国际黄金", "现货黄金"],
                    "context": "全球最大的黄金交易市场，价格具有权威性"
                },
                "黄金储备": {
                    "domain": "gold",
                    "definition": "各国中央银行持有的黄金资产",
                    "synonyms": ["央行黄金储备"],
                    "context": "作为国际储备资产，用于维护货币稳定"
                },
                
                # 金融领域术语
                "GDP": {
                    "domain": "finance",
                    "definition": "国内生产总值，衡量一个国家经济活动总量的指标",
                    "synonyms": ["国内生产总值"],
                    "context": "反映经济增长速度和整体经济规模"
                },
                "CPI": {
                    "domain": "finance",
                    "definition": "消费者价格指数，衡量物价水平变化的指标",
                    "synonyms": ["消费者物价指数", "居民消费价格指数"],
                    "context": "反映通货膨胀或通货紧缩的程度"
                },
                "美联储": {
                    "domain": "finance",
                    "definition": "美国联邦储备系统，美国的中央银行",
                    "synonyms": ["美国央行", "Fed"],
                    "context": "负责制定美国货币政策，对全球金融市场有重要影响"
                }
            },
            
            "rules": [
                {
                    "id": "gold_rule_1",
                    "domain": "gold",
                    "condition": "美元指数下跌",
                    "conclusion": "黄金价格通常会上涨",
                    "confidence": 0.85,
                    "explanation": "黄金以美元计价，美元贬值使得黄金对其他货币持有者更便宜"
                },
                {
                    "id": "gold_rule_2",
                    "domain": "gold",
                    "condition": "地缘政治风险加剧",
                    "conclusion": "黄金价格通常会上涨",
                    "confidence": 0.8,
                    "explanation": "黄金作为避险资产，在风险增加时需求上升"
                },
                {
                    "id": "finance_rule_1",
                    "domain": "finance",
                    "condition": "通货膨胀率上升",
                    "conclusion": "央行可能会加息",
                    "confidence": 0.75,
                    "explanation": "加息是控制通货膨胀的常用货币政策工具"
                }
            ],
            
            "faq": [
                {
                    "domain": "gold",
                    "question": "影响黄金价格的主要因素有哪些？",
                    "answer": "影响黄金价格的主要因素包括：美元汇率、地缘政治风险、通货膨胀率、利率水平、黄金供需关系、央行政策和市场情绪等。"
                },
                {
                    "domain": "finance",
                    "question": "什么是货币政策？",
                    "answer": "货币政策是中央银行通过控制货币供应量和利率，以影响经济活动和物价稳定的政策工具。"
                }
            ],
            
            "relationships": [
                {
                    "type": "causal",
                    "source": "美元指数下跌",
                    "target": "黄金价格上涨",
                    "domain": "gold",
                    "confidence": 0.85
                },
                {
                    "type": "causal",
                    "source": "利率上升",
                    "target": "债券价格下跌",
                    "domain": "finance",
                    "confidence": 0.9
                },
                {
                    "type": "similar",
                    "source": "黄金ETF",
                    "target": "黄金期货",
                    "domain": "gold",
                    "confidence": 0.7
                }
            ]
        }
        
        # 加载默认知识
        self.knowledge = default_knowledge
        
        # 建立索引
        self._build_indexes()
    
    def _build_indexes(self):
        """
        建立索引
        """
        # 建立术语索引
        for term, info in self.knowledge["terms"].items():
            domain = info["domain"]
            self.term_index[term].add(domain)
            self.domain_term_index[domain].add(term)
            
            # 为同义词也建立索引
            if "synonyms" in info:
                for synonym in info["synonyms"]:
                    self.term_index[synonym].add(domain)
    
    def load_knowledge(self, file_path: str):
        """
        从文件加载知识
        
        Args:
            file_path: 知识文件路径
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.knowledge = json.load(f)
            
            # 建立索引
            self._build_indexes()
            
            logger.info(f"成功从文件 {file_path} 加载知识")
        except Exception as e:
            logger.error(f"加载知识文件失败: {str(e)}")
            # 加载默认知识
            self._load_default_knowledge()
    
    def save_knowledge(self, file_path: str):
        """
        保存知识到文件
        
        Args:
            file_path: 保存路径
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge, f, ensure_ascii=False, indent=2)
            
            logger.info(f"成功将知识保存到文件 {file_path}")
        except Exception as e:
            logger.error(f"保存知识文件失败: {str(e)}")
    
    def add_domain(self, domain_id: str, name: str, description: str, keywords: List[str] = None):
        """
        添加领域
        
        Args:
            domain_id: 领域ID
            name: 领域名称
            description: 领域描述
            keywords: 领域关键词
        """
        if domain_id in self.knowledge["domains"]:
            logger.warning(f"领域 {domain_id} 已存在")
            return False
        
        self.knowledge["domains"][domain_id] = {
            "name": name,
            "description": description,
            "keywords": keywords or []
        }
        
        logger.info(f"成功添加领域: {domain_id}")
        return True
    
    def add_term(self, term: str, domain: str, definition: str, synonyms: List[str] = None, context: str = None):
        """
        添加术语
        
        Args:
            term: 术语
            domain: 领域
            definition: 术语定义
            synonyms: 同义词列表
            context: 上下文信息
        """
        if term in self.knowledge["terms"]:
            logger.warning(f"术语 {term} 已存在")
            return False
        
        # 检查领域是否存在
        if domain not in self.knowledge["domains"]:
            logger.warning(f"领域 {domain} 不存在")
            return False
        
        self.knowledge["terms"][term] = {
            "domain": domain,
            "definition": definition,
            "synonyms": synonyms or [],
            "context": context
        }
        
        # 更新索引
        self.term_index[term].add(domain)
        self.domain_term_index[domain].add(term)
        
        # 为同义词建立索引
        if synonyms:
            for synonym in synonyms:
                self.term_index[synonym].add(domain)
        
        logger.info(f"成功添加术语: {term} (领域: {domain})")
        return True
    
    def add_rule(self, condition: str, conclusion: str, domain: str, confidence: float = 1.0, explanation: str = None):
        """
        添加规则
        
        Args:
            condition: 条件
            conclusion: 结论
            domain: 领域
            confidence: 置信度
            explanation: 解释
        """
        rule_id = f"{domain}_rule_{len(self.knowledge['rules']) + 1}"
        
        rule = {
            "id": rule_id,
            "condition": condition,
            "conclusion": conclusion,
            "domain": domain,
            "confidence": confidence,
            "explanation": explanation
        }
        
        self.knowledge["rules"].append(rule)
        
        logger.info(f"成功添加规则: {rule_id}")
        return True
    
    def get_term_info(self, term: str) -> Dict[str, Any]:
        """
        获取术语信息
        
        Args:
            term: 术语
            
        Returns:
            Dict: 术语信息
        """
        # 直接查找
        if term in self.knowledge["terms"]:
            return self.knowledge["terms"][term]
        
        # 查找同义词
        for t, info in self.knowledge["terms"].items():
            if "synonyms" in info and term in info["synonyms"]:
                # 返回原术语的信息，但标记为同义词
                result = info.copy()
                result["synonym_of"] = t
                return result
        
        return None
    
    def get_domain_terms(self, domain: str) -> List[str]:
        """
        获取领域的所有术语
        
        Args:
            domain: 领域
            
        Returns:
            List[str]: 术语列表
        """
        if domain not in self.domain_term_index:
            return []
        
        return list(self.domain_term_index[domain])
    
    def get_domain_rules(self, domain: str) -> List[Dict]:
        """
        获取领域的所有规则
        
        Args:
            domain: 领域
            
        Returns:
            List[Dict]: 规则列表
        """
        return [rule for rule in self.knowledge["rules"] if rule["domain"] == domain]
    
    def query_knowledge(self, query: str, domains: List[str] = None) -> Dict[str, Any]:
        """
        查询知识
        
        Args:
            query: 查询内容
            domains: 限定的领域列表
            
        Returns:
            Dict: 查询结果
        """
        results = {
            "terms": [],  # 相关术语
            "rules": [],  # 相关规则
            "faq": [],  # 相关FAQ
            "domains": []  # 相关领域
        }
        
        # 查询术语
        for term in self.term_index:
            if term in query:
                domains_for_term = self.term_index[term]
                if domains is None or any(d in domains_for_term for d in domains):
                    term_info = self.get_term_info(term)
                    results["terms"].append((term, term_info))
        
        # 查询规则
        for rule in self.knowledge["rules"]:
            if domains is None or rule["domain"] in domains:
                if rule["condition"] in query or rule["conclusion"] in query:
                    results["rules"].append(rule)
        
        # 查询FAQ
        for faq_item in self.knowledge["faq"]:
            if domains is None or faq_item["domain"] in domains:
                if faq_item["question"] in query:
                    results["faq"].append(faq_item)
        
        # 查询领域
        if domains is None:
            for domain_id, domain_info in self.knowledge["domains"].items():
                if any(keyword in query for keyword in domain_info["keywords"]):
                    results["domains"].append((domain_id, domain_info))
        else:
            for domain_id in domains:
                if domain_id in self.knowledge["domains"]:
                    results["domains"].append((domain_id, self.knowledge["domains"][domain_id]))
        
        return results
    
    def expand_query(self, query: str, domains: List[str] = None) -> List[str]:
        """
        扩展查询
        
        Args:
            query: 原始查询
            domains: 领域列表
            
        Returns:
            List[str]: 扩展后的查询词列表
        """
        expanded = set([query])
        
        # 基于术语和同义词扩展
        for term in self.term_index:
            if term in query:
                domains_for_term = self.term_index[term]
                if domains is None or any(d in domains_for_term for d in domains):
                    term_info = self.get_term_info(term)
                    expanded.add(term)
                    
                    # 添加同义词
                    if "synonyms" in term_info:
                        for synonym in term_info["synonyms"]:
                            expanded.add(synonym)
        
        return list(expanded)
    
    def get_relationships(self, entity: str, domain: str = None) -> List[Dict]:
        """
        获取实体的关系
        
        Args:
            entity: 实体
            domain: 领域
            
        Returns:
            List[Dict]: 关系列表
        """
        relationships = []
        
        for rel in self.knowledge["relationships"]:
            if (rel["source"] == entity or rel["target"] == entity) and (domain is None or rel["domain"] == domain):
                relationships.append(rel)
        
        return relationships
    
    def update_term(self, term: str, updates: Dict[str, Any]):
        """
        更新术语
        
        Args:
            term: 术语
            updates: 更新内容
        """
        if term not in self.knowledge["terms"]:
            logger.warning(f"术语 {term} 不存在")
            return False
        
        # 记录原始信息以便更新索引
        original_info = self.knowledge["terms"][term]
        original_domain = original_info["domain"]
        original_synonyms = original_info.get("synonyms", [])
        
        # 更新术语信息
        self.knowledge["terms"][term].update(updates)
        
        # 更新索引
        new_domain = self.knowledge["terms"][term]["domain"]
        new_synonyms = self.knowledge["terms"][term].get("synonyms", [])
        
        # 处理领域变化
        if new_domain != original_domain:
            # 从旧领域中移除
            self.domain_term_index[original_domain].discard(term)
            if not self.domain_term_index[original_domain]:
                del self.domain_term_index[original_domain]
            
            # 添加到新领域
            self.domain_term_index[new_domain].add(term)
            
            # 更新术语索引
            self.term_index[term].remove(original_domain)
            self.term_index[term].add(new_domain)
        
        # 处理同义词变化
        removed_synonyms = set(original_synonyms) - set(new_synonyms)
        added_synonyms = set(new_synonyms) - set(original_synonyms)
        
        # 移除不再是同义词的术语索引
        for synonym in removed_synonyms:
            if synonym in self.term_index and original_domain in self.term_index[synonym]:
                self.term_index[synonym].remove(original_domain)
                if not self.term_index[synonym]:
                    del self.term_index[synonym]
        
        # 添加新的同义词索引
        for synonym in added_synonyms:
            self.term_index[synonym].add(new_domain)
        
        logger.info(f"成功更新术语: {term}")
        return True

# 测试代码
if __name__ == "__main__":
    # 创建知识库实例
    kb = KnowledgeBase()
    
    print("=== 领域知识库测试 ===")
    
    # 测试1: 获取领域术语
    print("\n1. 获取黄金领域的术语:")
    gold_terms = kb.get_domain_terms("gold")
    for term in gold_terms:
        print(f"   - {term}: {kb.get_term_info(term)['definition']}")
    
    # 测试2: 查询知识
    print("\n2. 查询'美元指数下跌影响':")
    query_results = kb.query_knowledge("美元指数下跌影响", domains=["gold"])
    
    print("   相关术语:")
    for term, info in query_results["terms"]:
        print(f"     - {term}: {info['definition']}")
    
    print("   相关规则:")
    for rule in query_results["rules"]:
        print(f"     - {rule['condition']} -> {rule['conclusion']} (置信度: {rule['confidence']})")
    
    # 测试3: 扩展查询
    print("\n3. 扩展查询'黄金ETF':")
    expanded = kb.expand_query("黄金ETF", domains=["gold"])
    print(f"   扩展结果: {expanded}")
    
    # 测试4: 添加新术语
    print("\n4. 添加新术语'白银ETF':")
    result = kb.add_term(
        term="白银ETF",
        domain="gold",
        definition="白银交易所交易基金，是一种以白银为基础资产的交易工具",
        synonyms=["白银指数基金"],
        context="用于跟踪白银价格走势"
    )
    print(f"   添加结果: {'成功' if result else '失败'}")
    
    # 测试5: 查询新添加的术语
    print("\n5. 查询新添加的术语'白银ETF':")
    term_info = kb.get_term_info("白银ETF")
    if term_info:
        print(f"   - 术语: 白银ETF")
        print(f"   - 领域: {term_info['domain']}")
        print(f"   - 定义: {term_info['definition']}")
        print(f"   - 同义词: {term_info['synonyms']}")
        print(f"   - 上下文: {term_info['context']}")
