#!/usr/bin/env python3
"""
对话管理器
负责处理多轮对话上下文、用户身份识别和历史记录管理
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DialogueTurn:
    """对话轮次信息"""
    
    def __init__(self, user_input: str, system_response: Optional[str] = None, 
                 intent: Optional[Dict] = None, entities: Optional[List] = None):
        self.timestamp = datetime.now()
        self.user_input = user_input
        self.system_response = system_response
        self.intent = intent
        self.entities = entities
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "user_input": self.user_input,
            "system_response": self.system_response,
            "intent": self.intent,
            "entities": self.entities
        }

class DialogueManager:
    """对话管理器，处理多轮对话上下文"""
    
    def __init__(self, max_history: int = 10):
        """
        初始化对话管理器
        
        Args:
            max_history: 最大保存的对话历史轮次
        """
        self.max_history = max_history
        self.dialogue_history: List[DialogueTurn] = []
        self.user_identity: Optional[Dict[str, Any]] = None
        self.session_id: str = datetime.now().strftime("%Y%m%d%H%M%S%f")
        
        # 用户身份关键词
        self.user_identity_keywords = {
            "analyst": ["分析师", "研究员", "分析员", "研究人员"],
            "manager": ["经理", "主管", "总监", "负责人", "领导"],
            "investor": ["投资者", "投资经理", "基金经理", "股民", "股东"],
            "trader": ["交易员", "操盘手", "交易者"],
            "student": ["学生", "学习者", "研究生", "本科生"],
            "media": ["媒体", "记者", "编辑"]
        }
    
    def add_turn(self, user_input: str, system_response: Optional[str] = None, 
                 intent: Optional[Dict] = None, entities: Optional[List] = None) -> None:
        """
        添加对话轮次
        
        Args:
            user_input: 用户输入
            system_response: 系统响应
            intent: 用户意图
            entities: 实体列表
        """
        turn = DialogueTurn(user_input, system_response, intent, entities)
        self.dialogue_history.append(turn)
        
        # 保持历史记录不超过最大限制
        if len(self.dialogue_history) > self.max_history:
            self.dialogue_history = self.dialogue_history[-self.max_history:]
    
    def get_context(self, num_turns: Optional[int] = None) -> List[DialogueTurn]:
        """
        获取对话上下文
        
        Args:
            num_turns: 返回的对话轮次数，None表示返回全部
            
        Returns:
            List[DialogueTurn]: 对话轮次列表
        """
        if num_turns is None:
            return self.dialogue_history
        return self.dialogue_history[-num_turns:]
    
    def get_context_str(self, num_turns: Optional[int] = None) -> str:
        """
        获取对话上下文的字符串表示
        
        Args:
            num_turns: 返回的对话轮次数，None表示返回全部
            
        Returns:
            str: 对话上下文字符串
        """
        context = self.get_context(num_turns)
        context_str = ""
        for i, turn in enumerate(context):
            context_str += f"用户 {i+1}: {turn.user_input}\n"
            if turn.system_response:
                context_str += f"系统 {i+1}: {turn.system_response}\n"
        return context_str.strip()
    
    def identify_user_identity(self, user_input: str) -> Optional[Dict[str, Any]]:
        """
        识别用户身份
        
        Args:
            user_input: 用户输入
            
        Returns:
            Dict[str, Any]: 用户身份信息
        """
        if self.user_identity:
            return self.user_identity
        
        user_input_lower = user_input.lower()
        for identity, keywords in self.user_identity_keywords.items():
            for keyword in keywords:
                if keyword in user_input_lower:
                    self.user_identity = {
                        "type": identity,
                        "confidence": 0.9,
                        "detected_from": keyword
                    }
                    logger.info(f"识别到用户身份: {identity}")
                    return self.user_identity
        
        # 默认返回None
        return None
    
    def set_user_identity(self, identity: str, confidence: float = 1.0) -> None:
        """
        手动设置用户身份
        
        Args:
            identity: 用户身份类型
            confidence: 置信度
        """
        self.user_identity = {
            "type": identity,
            "confidence": confidence,
            "detected_from": "manual"
        }
    
    def get_user_identity(self) -> Optional[Dict[str, Any]]:
        """
        获取用户身份
        
        Returns:
            Dict[str, Any]: 用户身份信息
        """
        return self.user_identity
    
    def clear_history(self) -> None:
        """
        清除对话历史
        """
        self.dialogue_history = []
    
    def get_session_id(self) -> str:
        """
        获取会话ID
        
        Returns:
            str: 会话ID
        """
        return self.session_id
    
    def get_dialogue_summary(self) -> Dict[str, Any]:
        """
        获取对话摘要
        
        Returns:
            Dict[str, Any]: 对话摘要信息
        """
        if not self.dialogue_history:
            return {"empty": True}
        
        # 提取核心意图和实体
        intents = []
        entities = []
        for turn in self.dialogue_history:
            if turn.intent:
                intents.append(turn.intent)
            if turn.entities:
                entities.extend(turn.entities)
        
        return {
            "session_id": self.session_id,
            "turn_count": len(self.dialogue_history),
            "start_time": self.dialogue_history[0].timestamp.isoformat(),
            "latest_time": self.dialogue_history[-1].timestamp.isoformat(),
            "user_identity": self.user_identity,
            "intents": intents,
            "entities": list(set(entities)) if entities else []
        }
