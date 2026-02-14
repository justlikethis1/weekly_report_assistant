import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

# 导入配置类
from src.utils.config import Config

class MemoryManager:
    """对话记忆管理器"""
    
    def __init__(self):
        self.max_memory_size = Config.MAX_MEMORY_SIZE
        self.max_user_input_length = Config.MAX_USER_INPUT_LENGTH
        self.max_ai_response_length = Config.MAX_AI_RESPONSE_LENGTH
        self.memory: List[Dict[str, Any]] = []
        
    def add_conversation(self, user_input: str, ai_response: str, 
                        user_files: Optional[List[str]] = None, 
                        generated_report: Optional[str] = None) -> bool:
        """
        添加对话记录
        
        Args:
            user_input: 用户输入
            ai_response: AI回复
            user_files: 用户上传的文件列表
            generated_report: 生成的报告路径
            
        Returns:
            bool: 添加是否成功
        """
        try:
            # 检查输入长度限制
            if len(user_input) > self.max_user_input_length:
                return False
            
            if len(ai_response) > self.max_ai_response_length:
                return False
            
            # 创建对话记录
            conversation = {
                "id": self._generate_conversation_id(),
                "timestamp": datetime.now().isoformat(),
                "user_input": user_input,
                "ai_response": ai_response,
                "user_files": user_files or [],
                "generated_report": generated_report,
                "length": {
                    "user_input": len(user_input),
                    "ai_response": len(ai_response)
                }
            }
            
            # 添加到记忆中
            self.memory.append(conversation)
            
            # 保持记忆大小不超过限制
            if len(self.memory) > self.max_memory_size:
                self.memory = self.memory[-self.max_memory_size:]
            
            return True
            
        except Exception as e:
            print(f"添加对话记录失败: {str(e)}")
            return False
    
    def get_recent_conversations(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        获取最近的对话记录
        
        Args:
            limit: 返回的对话记录数量限制
            
        Returns:
            List[Dict[str, Any]]: 对话记录列表
        """
        if limit is None:
            return self.memory
        return self.memory[-limit:]
    
    def get_conversation_by_id(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        根据ID获取对话记录
        
        Args:
            conversation_id: 对话ID
            
        Returns:
            Optional[Dict[str, Any]]: 对话记录，如果不存在则返回None
        """
        for conversation in self.memory:
            if conversation["id"] == conversation_id:
                return conversation
        return None
    
    def clear_memory(self) -> bool:
        """
        清空对话记忆
        
        Returns:
            bool: 清空是否成功
        """
        try:
            self.memory = []
            return True
        except Exception as e:
            print(f"清空对话记忆失败: {str(e)}")
            return False
    
    def save_memory(self, file_path: str = "memory.json") -> bool:
        """
        保存对话记忆到文件
        
        Args:
            file_path: 保存文件路径
            
        Returns:
            bool: 保存是否成功
        """
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.memory, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存对话记忆失败: {str(e)}")
            return False
    
    def load_memory(self, file_path: str = "memory.json") -> bool:
        """
        从文件加载对话记忆
        
        Args:
            file_path: 加载文件路径
            
        Returns:
            bool: 加载是否成功
        """
        try:
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    self.memory = json.load(f)
                # 确保记忆大小不超过限制
                if len(self.memory) > self.max_memory_size:
                    self.memory = self.memory[-self.max_memory_size:]
            return True
        except Exception as e:
            print(f"加载对话记忆失败: {str(e)}")
            return False
    
    def get_context(self, recent_count: int = 3) -> str:
        """
        获取对话上下文，用于生成连贯的回复
        
        Args:
            recent_count: 使用最近的对话数量
            
        Returns:
            str: 格式化的对话上下文
        """
        if not self.memory:
            return ""
        
        # 获取最近的对话记录
        recent_conversations = self.memory[-recent_count:]
        
        # 格式化上下文
        context = ""
        for conv in recent_conversations:
            context += f"用户: {conv['user_input']}\n"
            context += f"助手: {conv['ai_response']}\n\n"
        
        return context.strip()
    
    def _generate_conversation_id(self) -> str:
        """
        生成唯一的对话ID
        
        Returns:
            str: 对话ID
        """
        timestamp = datetime.now().timestamp()
        conversation_count = len(self.memory) + 1
        return f"conv_{timestamp:.0f}_{conversation_count}"
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取记忆统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        if not self.memory:
            return {
                "total_conversations": 0,
                "average_user_input_length": 0,
                "average_ai_response_length": 0,
                "memory_usage": 0
            }
        
        total_user_input_length = sum(conv["length"]["user_input"] for conv in self.memory)
        total_ai_response_length = sum(conv["length"]["ai_response"] for conv in self.memory)
        
        return {
            "total_conversations": len(self.memory),
            "average_user_input_length": total_user_input_length / len(self.memory),
            "average_ai_response_length": total_ai_response_length / len(self.memory),
            "memory_usage": len(self.memory) / self.max_memory_size * 100
        }
