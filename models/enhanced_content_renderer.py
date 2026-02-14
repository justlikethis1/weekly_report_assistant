#!/usr/bin/env python3
"""
增强的内容渲染器模块
负责渲染模板内容，支持高级功能和更好的性能
"""

import os
import hashlib
from typing import Dict, Any, List, Optional, Callable
import jinja2
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class EnhancedContentRenderer:
    """增强的内容渲染器，负责渲染模板内容"""
    
    def __init__(self, template_loader=None, cache_enabled: bool = True, cache_timeout: int = 3600):
        """
        初始化增强的内容渲染器
        
        Args:
            template_loader: 模板加载器
            cache_enabled: 是否启用缓存
            cache_timeout: 缓存超时时间（秒）
        """
        self.template_loader = template_loader
        self.jinja_env = None
        self.cache_enabled = cache_enabled
        self.cache_timeout = cache_timeout
        
        # 模板内容缓存
        self.template_cache = {
            'content': {},  # 模板内容缓存
            'rendered': {}  # 渲染结果缓存
        }
        
        # 自定义过滤器和扩展
        self.custom_filters = {
            'format_currency': self._format_currency,
            'format_percentage': self._format_percentage,
            'format_date': self._format_date,
            'truncate': self._truncate_text,
            'word_count': self._count_words,
            'sentence_count': self._count_sentences,
            'highlight_keywords': self._highlight_keywords
        }
        
        # 自定义全局函数
        self.custom_globals = {
            'now': datetime.now,
            'today': datetime.today,
            'getenv': os.getenv,
            'len': len,
            'sum': sum,
            'min': min,
            'max': max,
            'range': range,
            'sorted': sorted
        }
        
        # 初始化Jinja2环境
        if template_loader:
            self._setup_jinja_environment()
        
        logger.info("EnhancedContentRenderer初始化完成")
    
    def set_template_loader(self, template_loader):
        """
        设置模板加载器
        
        Args:
            template_loader: 模板加载器
        """
        self.template_loader = template_loader
        self._setup_jinja_environment()
    
    def _setup_jinja_environment(self):
        """
        设置Jinja2环境
        """
        try:
            if hasattr(self.template_loader, 'template_dir') and self.template_loader.template_dir:
                self.jinja_env = jinja2.Environment(
                    loader=jinja2.FileSystemLoader(self.template_loader.template_dir),
                    autoescape=True,
                    trim_blocks=True,
                    lstrip_blocks=True,
                    extensions=[
                        'jinja2.ext.do',
                        'jinja2.ext.loopcontrols',
                        'jinja2.ext.with_'  # 启用with语句扩展
                    ]
                )
                
                # 添加自定义过滤器
                for name, filter_func in self.custom_filters.items():
                    self.jinja_env.filters[name] = filter_func
                
                # 添加自定义全局函数
                for name, func in self.custom_globals.items():
                    self.jinja_env.globals[name] = func
                
                logger.info("Jinja2环境设置完成")
            else:
                self.jinja_env = None
                logger.warning("模板加载器未配置模板目录，无法设置Jinja2环境")
        
        except Exception as e:
            logger.error(f"设置Jinja2环境失败: {e}")
            self.jinja_env = None
    
    def render(self, template_path: str, data: Dict[str, Any]) -> Optional[str]:
        """
        渲染模板
        
        Args:
            template_path: 模板文件路径
            data: 渲染数据
            
        Returns:
            渲染后的内容，如果渲染失败返回None
        """
        try:
            # 检查缓存
            cache_key = self._generate_render_cache_key(template_path, data)
            if self.cache_enabled and cache_key in self.template_cache['rendered']:
                cached_result = self.template_cache['rendered'][cache_key]
                # 检查缓存是否过期
                if datetime.now() - cached_result['timestamp'] < timedelta(seconds=self.cache_timeout):
                    logger.debug(f"使用缓存的渲染结果: {template_path}")
                    return cached_result['content']
                else:
                    # 缓存过期，删除
                    del self.template_cache['rendered'][cache_key]
                    logger.debug(f"渲染结果缓存过期: {template_path}")
            
            if self.jinja_env:
                # 使用Jinja2环境渲染
                template = self.jinja_env.get_template(template_path)
                rendered_content = template.render(**data)
                
                # 缓存渲染结果
                if self.cache_enabled:
                    self.template_cache['rendered'][cache_key] = {
                        'content': rendered_content,
                        'timestamp': datetime.now()
                    }
                    logger.debug(f"缓存渲染结果: {template_path}")
                
                return rendered_content
            elif self.template_loader:
                # 如果没有Jinja2环境，直接使用字符串替换（简单模式）
                template_content = self._load_template_with_cache(template_path)
                if not template_content:
                    return None
                
                # 简单的字符串替换
                rendered_content = self._simple_template_render(template_content, data)
                
                # 缓存渲染结果
                if self.cache_enabled:
                    self.template_cache['rendered'][cache_key] = {
                        'content': rendered_content,
                        'timestamp': datetime.now()
                    }
                
                return rendered_content
            
        except Exception as e:
            logger.error(f"渲染模板 {template_path} 失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def render_string(self, template_string: str, data: Dict[str, Any]) -> Optional[str]:
        """
        渲染字符串模板
        
        Args:
            template_string: 模板字符串
            data: 渲染数据
            
        Returns:
            渲染后的内容，如果渲染失败返回None
        """
        try:
            # 检查缓存
            cache_key = self._generate_render_cache_key('__string_template__', data, template_string)
            if self.cache_enabled and cache_key in self.template_cache['rendered']:
                cached_result = self.template_cache['rendered'][cache_key]
                if datetime.now() - cached_result['timestamp'] < timedelta(seconds=self.cache_timeout):
                    logger.debug("使用缓存的字符串模板渲染结果")
                    return cached_result['content']
                else:
                    del self.template_cache['rendered'][cache_key]
            
            if self.jinja_env:
                # 使用Jinja2环境渲染
                template = self.jinja_env.from_string(template_string)
                rendered_content = template.render(**data)
                
                # 缓存渲染结果
                if self.cache_enabled:
                    self.template_cache['rendered'][cache_key] = {
                        'content': rendered_content,
                        'timestamp': datetime.now()
                    }
                
                return rendered_content
            else:
                # 简单的字符串替换
                rendered_content = self._simple_template_render(template_string, data)
                
                # 缓存渲染结果
                if self.cache_enabled:
                    self.template_cache['rendered'][cache_key] = {
                        'content': rendered_content,
                        'timestamp': datetime.now()
                    }
                
                return rendered_content
            
        except Exception as e:
            logger.error(f"渲染字符串模板失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _load_template_with_cache(self, template_path: str) -> Optional[str]:
        """
        加载模板内容（带缓存）
        
        Args:
            template_path: 模板文件路径
            
        Returns:
            模板内容，如果加载失败返回None
        """
        # 检查缓存
        if self.cache_enabled and template_path in self.template_cache['content']:
            cached_template = self.template_cache['content'][template_path]
            if datetime.now() - cached_template['timestamp'] < timedelta(seconds=self.cache_timeout):
                logger.debug(f"使用缓存的模板内容: {template_path}")
                return cached_template['content']
            else:
                # 缓存过期，删除
                del self.template_cache['content'][template_path]
        
        # 加载模板内容
        if self.template_loader:
            template_content = self.template_loader.load_template(template_path)
            if template_content:
                # 缓存模板内容
                if self.cache_enabled:
                    self.template_cache['content'][template_path] = {
                        'content': template_content,
                        'timestamp': datetime.now()
                    }
                    logger.debug(f"缓存模板内容: {template_path}")
                return template_content
        
        return None
    
    def _simple_template_render(self, template_content: str, data: Dict[str, Any]) -> str:
        """
        简单的字符串替换渲染
        
        Args:
            template_content: 模板内容
            data: 渲染数据
            
        Returns:
            渲染后的内容
        """
        rendered_content = template_content
        
        # 简单的字符串替换
        for key, value in data.items():
            # 支持 {{ key }} 格式
            placeholder = f"{{{{ {key} }}}}"
            rendered_content = rendered_content.replace(placeholder, str(value))
            
            # 支持 {{key}} 格式（无空格）
            placeholder_no_space = f"{{{{{key}}}}}"
            rendered_content = rendered_content.replace(placeholder_no_space, str(value))
        
        return rendered_content
    
    def _generate_render_cache_key(self, template_path: str, data: Dict[str, Any], template_string: str = None) -> str:
        """
        生成渲染缓存键
        
        Args:
            template_path: 模板路径
            data: 渲染数据
            template_string: 模板字符串（用于字符串模板）
            
        Returns:
            缓存键
        """
        # 创建数据摘要
        data_digest = {
            'template_path': template_path,
            'template_string': template_string[:100] if template_string else None,
            'data_keys': sorted(data.keys()),
            'data_hash': hashlib.md5(str(sorted(data.items())).encode()).hexdigest()
        }
        
        # 转换为字符串并生成MD5哈希
        data_str = str(data_digest)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _format_currency(self, value: Any, currency_symbol: str = '$', decimal_places: int = 2) -> str:
        """
        格式化货币
        
        Args:
            value: 数值
            currency_symbol: 货币符号
            decimal_places: 小数位数
            
        Returns:
            格式化后的货币字符串
        """
        try:
            return f"{currency_symbol}{float(value):,.{decimal_places}f}"
        except (TypeError, ValueError):
            return str(value)
    
    def _format_percentage(self, value: Any, decimal_places: int = 2) -> str:
        """
        格式化百分比
        
        Args:
            value: 数值
            decimal_places: 小数位数
            
        Returns:
            格式化后的百分比字符串
        """
        try:
            return f"{float(value) * 100:.{decimal_places}f}%"
        except (TypeError, ValueError):
            return str(value)
    
    def _format_date(self, value: Any, format_str: str = '%Y-%m-%d') -> str:
        """
        格式化日期
        
        Args:
            value: 日期值
            format_str: 日期格式
            
        Returns:
            格式化后的日期字符串
        """
        try:
            if isinstance(value, str):
                from dateutil import parser
                value = parser.parse(value)
            return value.strftime(format_str)
        except (TypeError, ValueError):
            return str(value)
    
    def _truncate_text(self, value: str, length: int = 100, suffix: str = '...') -> str:
        """
        截断文本
        
        Args:
            value: 文本
            length: 最大长度
            suffix: 后缀
            
        Returns:
            截断后的文本
        """
        try:
            if len(value) <= length:
                return value
            return value[:length - len(suffix)] + suffix
        except (TypeError, ValueError):
            return str(value)
    
    def _count_words(self, value: str) -> int:
        """
        计算单词数
        
        Args:
            value: 文本
            
        Returns:
            单词数
        """
        try:
            return len(value.split())
        except (TypeError, ValueError):
            return 0
    
    def _count_sentences(self, value: str) -> int:
        """
        计算句子数
        
        Args:
            value: 文本
            
        Returns:
            句子数
        """
        try:
            import re
            return len(re.findall(r'[.!?]+', value)) or 1
        except (TypeError, ValueError):
            return 0
    
    def _highlight_keywords(self, value: str, keywords: List[str] or str, css_class: str = 'highlight') -> str:
        """
        高亮关键词
        
        Args:
            value: 文本
            keywords: 关键词列表或字符串
            css_class: CSS类名
            
        Returns:
            高亮后的HTML文本
        """
        try:
            import re
            
            if isinstance(keywords, str):
                keywords = [keywords]
            
            highlighted_text = value
            for keyword in keywords:
                if keyword:
                    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                    highlighted_text = pattern.sub(f'<span class="{css_class}">{keyword}</span>', highlighted_text)
            
            return highlighted_text
        except (TypeError, ValueError):
            return str(value)
    
    def add_custom_filter(self, name: str, filter_func: Callable):
        """
        添加自定义过滤器
        
        Args:
            name: 过滤器名称
            filter_func: 过滤器函数
        """
        self.custom_filters[name] = filter_func
        
        # 更新Jinja2环境
        if self.jinja_env:
            self.jinja_env.filters[name] = filter_func
        
        logger.info(f"已添加自定义过滤器: {name}")
    
    def add_custom_global(self, name: str, global_func: Callable):
        """
        添加自定义全局函数
        
        Args:
            name: 函数名称
            global_func: 全局函数
        """
        self.custom_globals[name] = global_func
        
        # 更新Jinja2环境
        if self.jinja_env:
            self.jinja_env.globals[name] = global_func
        
        logger.info(f"已添加自定义全局函数: {name}")
    
    def clear_cache(self, cache_type: str = None):
        """
        清除缓存
        
        Args:
            cache_type: 缓存类型（'content'或'rendered'），如果为None则清除所有缓存
        """
        if cache_type == 'content':
            self.template_cache['content'] = {}
            logger.info("已清除模板内容缓存")
        elif cache_type == 'rendered':
            self.template_cache['rendered'] = {}
            logger.info("已清除渲染结果缓存")
        else:
            self.template_cache['content'] = {}
            self.template_cache['rendered'] = {}
            logger.info("已清除所有模板缓存")
    
    def set_cache_enabled(self, enabled: bool):
        """
        设置缓存是否启用
        
        Args:
            enabled: 是否启用缓存
        """
        self.cache_enabled = enabled
        logger.info(f"模板缓存已{'启用' if enabled else '禁用'}")
    
    def set_cache_timeout(self, timeout: int):
        """
        设置缓存超时时间
        
        Args:
            timeout: 超时时间（秒）
        """
        self.cache_timeout = timeout
        logger.info(f"模板缓存超时时间已设置为: {timeout}秒")
    
    def render_multiple(self, template_path: str, data_list: List[Dict[str, Any]]) -> List[str]:
        """
        批量渲染模板
        
        Args:
            template_path: 模板路径
            data_list: 数据列表
            
        Returns:
            渲染结果列表
        """
        results = []
        for data in data_list:
            result = self.render(template_path, data)
            results.append(result)
        return results
    
    def render_sections(self, sections: List[Dict[str, Any]], data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        渲染多个章节
        
        Args:
            sections: 章节列表
            data: 渲染数据
            
        Returns:
            渲染后的章节列表
        """
        rendered_sections = []
        
        for section in sections:
            if 'template_path' in section:
                rendered_content = self.render(section['template_path'], data)
                rendered_section = section.copy()
                rendered_section['content'] = rendered_content
                rendered_sections.append(rendered_section)
            elif 'content' in section:
                # 如果已经有内容，直接使用
                rendered_sections.append(section.copy())
        
        return rendered_sections
