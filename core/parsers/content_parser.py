from abc import ABC, abstractmethod
from typing import Dict, Any

class FileProcessor(ABC):
    """文件处理器基类"""
    
    @abstractmethod
    def process(self, file_path: str) -> Dict[str, Any]:
        """
        处理文件并返回提取的信息
        
        Args:
            file_path: 文件路径
            
        Returns:
            Dict[str, Any]: 提取的信息，包含text、metadata等字段
        """
        pass
    
    @staticmethod
    def get_file_extension(file_path: str) -> str:
        """获取文件扩展名"""
        return file_path.split('.')[-1].lower()
