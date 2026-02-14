#!/usr/bin/env python3
"""
统一NLP服务模块
"""

# 重新启用NLPService的导入，支持PyTorch依赖的优雅降级
from .services.nlp_service import NLPService, QueryAnalysis, DocumentAnalysis
__all__ = [
    "NLPService",
    "QueryAnalysis",
    "DocumentAnalysis"
]
