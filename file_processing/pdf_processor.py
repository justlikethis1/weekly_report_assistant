from typing import Dict, Any
from .base import FileProcessor
import os
from PyPDF2 import PdfReader

class PDFProcessor(FileProcessor):
    """PDF文件处理器"""
    
    def process(self, file_path: str) -> Dict[str, Any]:
        """
        处理PDF文件（优化版本）
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            Dict[str, Any]: 包含文本内容和元数据的字典
        """
        try:
            reader = PdfReader(file_path)
            text_parts = []  # 使用列表收集文本，避免频繁字符串拼接
            
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            
            # 一次性拼接所有文本
            text = "".join(text_parts)
            
            metadata = {
                'file_type': 'pdf',
                'file_name': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path),
                'page_count': len(reader.pages),
                'content_length': len(text)
            }
            
            return {
                'text': text,
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
