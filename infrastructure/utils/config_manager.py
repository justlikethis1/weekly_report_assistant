import os
import yaml
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置验证模型
class ModelConfig(BaseModel):
    name: str = Field(default="Qwen2.5-7B-Instruct-INT4")
    path: str = Field(default="Qwen/Qwen2.5-7B-Instruct-INT4")
    cache_dir: str = Field(default="./models/cache")

class MemoryConfig(BaseModel):
    max_size: int = Field(default=10, gt=0)
    max_user_input_length: int = Field(default=500, gt=0)
    max_ai_response_length: int = Field(default=15000, gt=0)

class FileProcessingConfig(BaseModel):
    supported_types: List[str] = Field(default_factory=lambda: ["pdf", "docx", "xlsx", "csv", "txt", "png", "jpg"])

class UIConfig(BaseModel):
    title: str = Field(default="周报生成助手")
    theme: str = Field(default="soft")

class WordConfig(BaseModel):
    template_path: str = Field(default="./templates/weekly_report_template.docx")

class NLPConfig(BaseModel):
    use_mock: bool = Field(default=True)  # 是否使用mock模式，避免下载外部模型
    sentence_transformer_model: str = Field(default="paraphrase-multilingual-MiniLM-L12-v2")  # 使用的sentence_transformer模型
    sentence_transformer_cache: str = Field(default="./models/sentence_transformers_new")  # 模型缓存目录
    device: str = Field(default="cpu")  # 使用的设备（cpu或cuda）

class LogConfig(BaseModel):
    level: str = Field(default="INFO")
    file: str = Field(default="app.log")

class AppConfig(BaseModel):
    model: ModelConfig
    memory: MemoryConfig
    file_processing: FileProcessingConfig
    ui: UIConfig
    word: WordConfig
    log: LogConfig
    nlp: NLPConfig

class ConfigManager:
    """配置管理中心，支持分层配置和配置验证"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """初始化配置"""
        self._config = None
        self._logger = logging.getLogger(__name__)
        self._load_config()
    
    def _load_config(self):
        """加载配置，优先级：环境变量 > YAML配置文件 > 默认值"""
        # 从YAML文件加载配置（如果存在）
        yaml_config = self._load_yaml_config()
        
        # 从环境变量加载配置
        env_config = self._load_env_config()
        
        # 合并配置
        merged_config = self._merge_configs(yaml_config, env_config)
        
        # 确保包含所有必要的配置节
        required_sections = ["model", "memory", "file_processing", "ui", "word", "log", "nlp"]
        for section in required_sections:
            if section not in merged_config:
                merged_config[section] = {}
        
        # 验证配置
        self._config = AppConfig(**merged_config)
    
    def _load_yaml_config(self) -> Dict[str, Any]:
        """
        从YAML配置文件加载配置
        """
        config_path = os.getenv("CONFIG_PATH", "./config/config.yaml")
        # 转换为绝对路径
        config_path = os.path.abspath(config_path)
        self._logger.debug(f"Loading YAML config from: {config_path}")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        self._logger.warning(f"YAML config file not found: {config_path}")
        return {}
    
    def _load_env_config(self) -> Dict[str, Any]:
        """从环境变量加载配置"""
        env_config = {}
        
        # 只有当环境变量存在时才添加到配置中
        if os.getenv("MODEL_NAME"):
            env_config.setdefault("model", {})["name"] = os.getenv("MODEL_NAME")
        if os.getenv("MODEL_PATH"):
            env_config.setdefault("model", {})["path"] = os.getenv("MODEL_PATH")
        if os.getenv("MODEL_CACHE_DIR"):
            env_config.setdefault("model", {})["cache_dir"] = os.getenv("MODEL_CACHE_DIR")
        
        if os.getenv("MAX_MEMORY_SIZE"):
            env_config.setdefault("memory", {})["max_size"] = os.getenv("MAX_MEMORY_SIZE")
        if os.getenv("MAX_USER_INPUT_LENGTH"):
            env_config.setdefault("memory", {})["max_user_input_length"] = os.getenv("MAX_USER_INPUT_LENGTH")
        if os.getenv("MAX_AI_RESPONSE_LENGTH"):
            env_config.setdefault("memory", {})["max_ai_response_length"] = os.getenv("MAX_AI_RESPONSE_LENGTH")
        
        if os.getenv("SUPPORTED_FILE_TYPES"):
            env_config.setdefault("file_processing", {})["supported_types"] = os.getenv("SUPPORTED_FILE_TYPES").split(",")
        
        if os.getenv("UI_TITLE"):
            env_config.setdefault("ui", {})["title"] = os.getenv("UI_TITLE")
        if os.getenv("UI_THEME"):
            env_config.setdefault("ui", {})["theme"] = os.getenv("UI_THEME")
        
        if os.getenv("WORD_TEMPLATE_PATH"):
            env_config.setdefault("word", {})["template_path"] = os.getenv("WORD_TEMPLATE_PATH")
        
        if os.getenv("LOG_LEVEL"):
            env_config.setdefault("log", {})["level"] = os.getenv("LOG_LEVEL")
        if os.getenv("LOG_FILE"):
            env_config.setdefault("log", {})["file"] = os.getenv("LOG_FILE")
        
        # NLP配置
        if os.getenv("NLP_USE_MOCK"):
            env_config.setdefault("nlp", {})["use_mock"] = os.getenv("NLP_USE_MOCK").lower() in ["true", "1", "yes"]
        if os.getenv("NLP_SENTENCE_TRANSFORMER_MODEL"):
            env_config.setdefault("nlp", {})["sentence_transformer_model"] = os.getenv("NLP_SENTENCE_TRANSFORMER_MODEL")
        if os.getenv("NLP_SENTENCE_TRANSFORMER_CACHE"):
            env_config.setdefault("nlp", {})["sentence_transformer_cache"] = os.getenv("NLP_SENTENCE_TRANSFORMER_CACHE")
        if os.getenv("NLP_DEVICE"):
            env_config.setdefault("nlp", {})["device"] = os.getenv("NLP_DEVICE")
        
        return env_config
    
    def _merge_configs(self, yaml_config: Dict[str, Any], env_config: Dict[str, Any]) -> Dict[str, Any]:
        """合并配置"""
        merged = {}
        
        # 合并所有可能的配置项
        all_keys = set(yaml_config.keys()).union(env_config.keys())
        
        for key in all_keys:
            if key in yaml_config and key in env_config:
                if isinstance(yaml_config[key], dict) and isinstance(env_config[key], dict):
                    merged[key] = self._merge_configs(yaml_config[key], env_config[key])
                else:
                    merged[key] = env_config[key]  # 环境变量优先级更高
            elif key in env_config:
                merged[key] = env_config[key]
            elif key in yaml_config:
                merged[key] = yaml_config[key]
        
        return merged
    
    def _remove_none_values(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """移除字典中的None值"""
        result = {}
        for key, value in data.items():
            if value is not None:
                if isinstance(value, dict):
                    value = self._remove_none_values(value)
                result[key] = value
        return result
    
    @property
    def config(self) -> AppConfig:
        """获取配置"""
        return self._config
    
    def get(self, path: str, default: Optional[Any] = None) -> Any:
        """根据路径获取配置值
        
        Args:
            path: 配置路径，如 "model.name"
            default: 默认值
            
        Returns:
            配置值
        """
        try:
            parts = path.split(".")
            value = self._config
            for part in parts:
                value = getattr(value, part)
            return value
        except (AttributeError, ValueError):
            return default
    
    def reload(self) -> None:
        """重新加载配置"""
        self._load_config()

# 创建全局配置实例
config_manager = ConfigManager()

# 配置日志记录
logging.basicConfig(
    level=getattr(logging, config_manager.get("log.level")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(config_manager.get("log.file")),
        logging.StreamHandler()
    ]
)
