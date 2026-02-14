import os
import re
import logging
import statistics
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, validator
import pandas as pd
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

# 配置日志记录
logger = logging.getLogger(__name__)

# 质量评估模型
@dataclass
class QualityScore:
    