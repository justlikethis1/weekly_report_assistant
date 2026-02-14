from typing import Dict, Any
from .base import FileProcessor
from .text_processor import TextProcessor
from .pdf_processor import PDFProcessor
from .docx_processor import DOCXProcessor
from .spreadsheet_processor import SpreadsheetProcessor
from .image_processor import ImageProcessor
import os
import hashlib

class FileProcessorFactory:
    """文件处理器工厂类"""
    
    # 注册的处理器映射
    _processors = {
        'txt': TextProcessor(),
        'pdf': PDFProcessor(),
        'docx': DOCXProcessor(),
        'xlsx': SpreadsheetProcessor(),
        'csv': SpreadsheetProcessor(),
        'png': ImageProcessor(),
        'jpg': ImageProcessor(),
        'jpeg': ImageProcessor()
    }
    
    # 文件内容缓存，使用文件哈希作为键
    _content_cache = {}
    
    @staticmethod
    def _get_file_hash(file_path: str) -> str:
        """计算文件的SHA256哈希值，用于缓存键"""
        hasher = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                # 分块读取文件，避免大文件占用过多内存
                for chunk in iter(lambda: f.read(4096), b''):
                    hasher.update(chunk)
            # 包含文件路径和修改时间，确保文件变更时重新处理
            file_stat = os.stat(file_path)
            return hasher.hexdigest() + str(file_stat.st_mtime)
        except Exception:
            return str(os.path.abspath(file_path)) + str(os.stat(file_path).st_mtime)
    
    @classmethod
    def get_processor(cls, file_extension: str) -> FileProcessor:
        """
        根据文件扩展名获取相应的处理器
        
        Args:
            file_extension: 文件扩展名
            
        Returns:
            FileProcessor: 对应的文件处理器实例
            
        Raises:
            ValueError: 如果文件扩展名不被支持
        """
        # 移除扩展名前面的点（如果有的话）
        file_extension = file_extension.lower().lstrip('.')
        if file_extension not in cls._processors:
            raise ValueError(f"不支持的文件格式: {file_extension}")
        return cls._processors[file_extension]
    
    @classmethod
    def process_file(cls, file_path: str) -> Dict[str, Any]:
        """
        处理文件的统一接口，带缓存功能
        
        Args:
            file_path: 文件路径
            
        Returns:
            Dict[str, Any]: 提取的信息
        """
        try:
            # 计算文件哈希值，用于缓存键
            file_hash = cls._get_file_hash(file_path)
            
            # 检查缓存中是否已有结果
            if file_hash in cls._content_cache:
                # 返回缓存的结果，并标记为已缓存
                cached_result = cls._content_cache[file_hash].copy()
                cached_result['from_cache'] = True
                return cached_result
            
            # 缓存中没有，正常处理文件
            file_extension = FileProcessor.get_file_extension(file_path)
            processor = cls.get_processor(file_extension)
            result = processor.process(file_path)
            
            # 将结果存入缓存，标记为非缓存
            result['from_cache'] = False
            cls._content_cache[file_hash] = result
            
            # 限制缓存大小，避免内存占用过大
            if len(cls._content_cache) > 100:
                # 移除最早的缓存项（使用字典顺序）
                oldest_key = next(iter(cls._content_cache))
                del cls._content_cache[oldest_key]
            
            return result
        except Exception as e:
            return {
                'text': '',
                'metadata': {},
                'success': False,
                'error': str(e)
            }
    
    @classmethod
    def get_supported_formats(cls) -> list:
        """获取支持的文件格式列表"""
        return list(cls._processors.keys())
