import os
from dotenv import load_dotenv
from typing import Dict, Any

# 加载环境变量
load_dotenv()

class Config:
    """配置管理类"""
    
    # 模型配置
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen2.5-7B-Instruct-INT4")
    MODEL_PATH = os.getenv("MODEL_PATH", "./models/qwen2.5-7B-Instruct-INT4")
    MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./models/cache")
    
    # 内存配置
    MAX_MEMORY_SIZE = int(os.getenv("MAX_MEMORY_SIZE", "10"))
    MAX_USER_INPUT_LENGTH = int(os.getenv("MAX_USER_INPUT_LENGTH", "500"))
    MAX_AI_RESPONSE_LENGTH = int(os.getenv("MAX_AI_RESPONSE_LENGTH", "15000"))
    
    # 文件处理配置
    SUPPORTED_FILE_TYPES = os.getenv("SUPPORTED_FILE_TYPES", "pdf,docx,xlsx,csv,txt,png,jpg").split(",")
    
    # UI配置
    UI_TITLE = os.getenv("UI_TITLE", "周报生成助手")
    UI_THEME = os.getenv("UI_THEME", "soft")
    
    # Word文档配置
    WORD_TEMPLATE_PATH = os.getenv("WORD_TEMPLATE_PATH", "./templates/weekly_report_template.docx")
    
    # 日志配置
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "app.log")
    
    @staticmethod
    def get_config() -> Dict[str, Any]:
        """获取所有配置"""
        return {
            "model": {
                "name": Config.MODEL_NAME,
                "path": Config.MODEL_PATH
            },
            "memory": {
                "max_size": Config.MAX_MEMORY_SIZE,
                "max_user_input_length": Config.MAX_USER_INPUT_LENGTH,
                "max_ai_response_length": Config.MAX_AI_RESPONSE_LENGTH
            },
            "file_processing": {
                "supported_types": Config.SUPPORTED_FILE_TYPES
            },
            "ui": {
                "title": Config.UI_TITLE,
                "theme": Config.UI_THEME
            },
            "word": {
                "template_path": Config.WORD_TEMPLATE_PATH
            },
            "log": {
                "level": Config.LOG_LEVEL,
                "file": Config.LOG_FILE
            }
        }
    
    @staticmethod
    def update_config(key: str, value: Any) -> bool:
        """
        更新配置
        
        Args:
            key: 配置键，使用点分隔路径，如 "model.name"
            value: 配置值
            
        Returns:
            bool: 更新是否成功
        """
        try:
            parts = key.split(".")
            if len(parts) != 2:
                return False
            
            section, config_key = parts
            
            if section == "model":
                if config_key == "name":
                    Config.MODEL_NAME = value
                elif config_key == "path":
                    Config.MODEL_PATH = value
            elif section == "memory":
                if config_key == "max_size":
                    Config.MAX_MEMORY_SIZE = int(value)
                elif config_key == "max_user_input_length":
                    Config.MAX_USER_INPUT_LENGTH = int(value)
                elif config_key == "max_ai_response_length":
                    Config.MAX_AI_RESPONSE_LENGTH = int(value)
            elif section == "file_processing":
                if config_key == "supported_types":
                    Config.SUPPORTED_FILE_TYPES = value.split(",")
            elif section == "ui":
                if config_key == "title":
                    Config.UI_TITLE = value
                elif config_key == "theme":
                    Config.UI_THEME = value
            elif section == "word":
                if config_key == "template_path":
                    Config.WORD_TEMPLATE_PATH = value
            elif section == "log":
                if config_key == "level":
                    Config.LOG_LEVEL = value
                elif config_key == "file":
                    Config.LOG_FILE = value
            else:
                return False
            
            return True
        except Exception as e:
            print(f"更新配置失败: {str(e)}")
            return False
