from typing import Dict, Any
from .base import FileProcessor
import os

class TextProcessor(FileProcessor):
    """文本文件处理器"""
    
    def process(self, file_path: str) -> Dict[str, Any]:
        """
        处理TXT文件
        
        Args:
            file_path: TXT文件路径
            
        Returns:
            Dict[str, Any]: 包含文本内容和元数据的字典
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            metadata = {
                'file_type': 'txt',
                'file_name': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path),
                'content_length': len(content)
            }
            
            return {
                'text': content,
                'metadata': metadata,
                'success': True
            }
        except Exception as e:
            return {
                'text': '',
                'metadata': {},
                'success': False,
                'error': str(e)
            }
