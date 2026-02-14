import os
import logging
import logging.handlers
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pydantic import BaseModel, Field, validator
import traceback

# 日志配置模型
class LogConfig(BaseModel):
    level: str = Field(default="INFO", description="日志级别")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="日志格式")
    file_path: str = Field(default="app.log", description="日志文件路径")
    max_bytes: int = Field(default=10 * 1024 * 1024, description="日志文件最大字节数")
    backup_count: int = Field(default=5, description="日志文件备份数量")

# 日志记录器
class LogManager:
    def __init__(self, config: Optional[LogConfig] = None):
        if config is None:
            config = LogConfig()
        
        self.config = config
        self.logger = None
        self._setup_logger()
    
    def _setup_logger(self):
        # 创建日志记录器
        self.logger = logging.getLogger("LogManager")
        self.logger.setLevel(getattr(logging, self.config.level))
        
        # 清空现有的处理器
        self.logger.handlers = []
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.config.level))
        console_handler.setFormatter(logging.Formatter(self.config.format))
        self.logger.addHandler(console_handler)
        
        # 文件处理器
        file_handler = logging.handlers.RotatingFileHandler(
            self.config.file_path,
            maxBytes=self.config.max_bytes,
            backupCount=self.config.backup_count
        )
        file_handler.setLevel(getattr(logging, self.config.level))
        file_handler.setFormatter(logging.Formatter(self.config.format))
        self.logger.addHandler(file_handler)
        
    def get_logger(self, name: str) -> logging.Logger:
        """获取指定名称的日志记录器"""
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, self.config.level))
        
        # 检查是否已经添加了处理器
        if not logger.handlers:
            # 控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, self.config.level))
            console_handler.setFormatter(logging.Formatter(self.config.format))
            logger.addHandler(console_handler)
            
            # 文件处理器
            file_handler = logging.handlers.RotatingFileHandler(
                self.config.file_path,
                maxBytes=self.config.max_bytes,
                backupCount=self.config.backup_count
            )
            file_handler.setLevel(getattr(logging, self.config.level))
            file_handler.setFormatter(logging.Formatter(self.config.format))
            logger.addHandler(file_handler)
        
        return logger
    
    def update_config(self, config: LogConfig):
        """更新日志配置"""
        self.config = config
        self._setup_logger()

# 性能监控装饰器
def performance_monitor(func=None, logger_name=None, operation_name=None):
    """性能监控装饰器"""
    def decorator(_func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = _func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                duration = (end_time - start_time) * 1000  # 转换为毫秒
                
                # 获取日志记录器名称
                log_name = logger_name or (args[0].__class__.__name__ if args else _func.__name__)
                logger = logging.getLogger(log_name)
                
                # 构造日志消息
                op_name = operation_name or _func.__name__
                logger.info(f"{op_name} executed in {duration:.2f}ms")
        
        return wrapper
    
    # 支持两种调用方式：@performance_monitor 或 @performance_monitor(logger_name="name")
    if func is None:
        return decorator
    else:
        return decorator(func)

# 全局日志管理器实例
log_manager = LogManager()