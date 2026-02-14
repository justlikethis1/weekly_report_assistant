from typing import Dict, Any
from .base import FileProcessor
import os
from PIL import Image
import pytesseract

class ImageProcessor(FileProcessor):
    """图像文件处理器，支持PNG和JPG格式"""
    
    def process(self, file_path: str) -> Dict[str, Any]:
        """
        处理图像文件，使用OCR提取文本
        
        Args:
            file_path: 图像文件路径
            
        Returns:
            Dict[str, Any]: 包含提取的文本和元数据的字典
        """
        try:
            # 打开图像文件
            image = Image.open(file_path)
            
            # 使用OCR提取文本
            text = pytesseract.image_to_string(image, lang='chi_sim+eng')
            
            # 获取图像信息
            width, height = image.size
            
            metadata = {
                'file_type': self.get_file_extension(file_path),
                'file_name': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path),
                'image_width': width,
                'image_height': height,
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
