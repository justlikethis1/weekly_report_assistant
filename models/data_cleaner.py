#!/usr/bin/env python3
"""
数据清洗器模块
负责清洗和转换数据
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class DataCleaner:
    """数据清洗器，负责清洗和转换数据"""
    
    def __init__(self):
        """初始化数据清洗器"""
        logger.info("DataCleaner初始化完成")
    
    def clean_dataframe(self, df: pd.DataFrame, cleaning_rules: Dict[str, Any]) -> pd.DataFrame:
        """
        清洗数据框
        
        Args:
            df: 要清洗的数据框
            cleaning_rules: 清洗规则
            
        Returns:
            清洗后的数据框
        """
        cleaned_df = df.copy()
        
        # 处理缺失值
        if 'missing_values' in cleaning_rules:
            cleaned_df = self._handle_missing_values(cleaned_df, cleaning_rules['missing_values'])
        
        # 转换列类型
        if 'column_types' in cleaning_rules:
            cleaned_df = self._convert_column_types(cleaned_df, cleaning_rules['column_types'])
        
        # 去重处理
        if 'duplicates' in cleaning_rules and cleaning_rules['duplicates']:
            cleaned_df = self._remove_duplicates(cleaned_df, cleaning_rules['duplicates'])
        
        # 处理异常值
        if 'outliers' in cleaning_rules:
            cleaned_df = self._handle_outliers(cleaned_df, cleaning_rules['outliers'])
        
        # 文本清洗
        if 'text_cleaning' in cleaning_rules:
            cleaned_df = self._clean_text_columns(cleaned_df, cleaning_rules['text_cleaning'])
        
        # 时间序列处理
        if 'time_series' in cleaning_rules:
            cleaned_df = self._process_time_series(cleaned_df, cleaning_rules['time_series'])
        
        # 重命名列
        if 'rename_columns' in cleaning_rules:
            cleaned_df = self._rename_columns(cleaned_df, cleaning_rules['rename_columns'])
        
        return cleaned_df
    
    def _handle_missing_values(self, df: pd.DataFrame, rules: Dict[str, str]) -> pd.DataFrame:
        """
        处理缺失值
        
        Args:
            df: 数据框
            rules: 缺失值处理规则
            
        Returns:
            处理后的数据框
        """
        cleaned_df = df.copy()
        
        for col, rule in rules.items():
            if col not in cleaned_df.columns:
                continue
            
            try:
                if rule == 'drop':
                    cleaned_df = cleaned_df.dropna(subset=[col])
                    logger.debug(f"删除列 {col} 中的缺失值，剩余 {len(cleaned_df)} 行")
                
                elif rule == 'drop_all':
                    cleaned_df = cleaned_df.dropna()
                    logger.debug(f"删除所有包含缺失值的行，剩余 {len(cleaned_df)} 行")
                
                elif rule == 'fill_empty':
                    # 根据列类型填充空值
                    if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                        cleaned_df[col] = cleaned_df[col].fillna(0)
                    elif pd.api.types.is_string_dtype(cleaned_df[col]):
                        cleaned_df[col] = cleaned_df[col].fillna('')
                    logger.debug(f"用默认值填充列 {col} 中的缺失值")
                
                elif rule == 'mean' and pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    mean_val = cleaned_df[col].mean()
                    cleaned_df[col] = cleaned_df[col].fillna(mean_val)
                    logger.debug(f"用均值 {mean_val} 填充列 {col} 中的缺失值")
                
                elif rule == 'median' and pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    median_val = cleaned_df[col].median()
                    cleaned_df[col] = cleaned_df[col].fillna(median_val)
                    logger.debug(f"用中位数 {median_val} 填充列 {col} 中的缺失值")
                
                elif rule == 'mode':
                    mode_val = cleaned_df[col].mode()[0]
                    cleaned_df[col] = cleaned_df[col].fillna(mode_val)
                    logger.debug(f"用众数 {mode_val} 填充列 {col} 中的缺失值")
                
                elif rule == 'forward_fill':
                    cleaned_df[col] = cleaned_df[col].fillna(method='ffill')
                    logger.debug(f"用前向填充法填充列 {col} 中的缺失值")
                
                elif rule == 'backward_fill':
                    cleaned_df[col] = cleaned_df[col].fillna(method='bfill')
                    logger.debug(f"用后向填充法填充列 {col} 中的缺失值")
                
                elif rule.startswith('constant:'):
                    # 用指定常量填充
                    constant_val = rule.split(':', 1)[1]
                    cleaned_df[col] = cleaned_df[col].fillna(constant_val)
                    logger.debug(f"用常量 '{constant_val}' 填充列 {col} 中的缺失值")
                
            except Exception as e:
                logger.error(f"处理列 {col} 的缺失值失败: {e}")
        
        return cleaned_df
    
    def _convert_column_types(self, df: pd.DataFrame, rules: Dict[str, str]) -> pd.DataFrame:
        """
        转换列类型
        
        Args:
            df: 数据框
            rules: 列类型转换规则
            
        Returns:
            转换后的数据框
        """
        cleaned_df = df.copy()
        
        for col, dtype in rules.items():
            if col not in cleaned_df.columns:
                continue
            
            try:
                if dtype == 'datetime' or dtype == 'date':
                    cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
                    logger.debug(f"将列 {col} 转换为日期类型")
                
                elif dtype == 'int' or dtype == 'integer':
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce').astype('Int64')
                    logger.debug(f"将列 {col} 转换为整数类型")
                
                elif dtype == 'float':
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce').astype(float)
                    logger.debug(f"将列 {col} 转换为浮点数类型")
                
                elif dtype == 'bool' or dtype == 'boolean':
                    cleaned_df[col] = cleaned_df[col].astype(bool)
                    logger.debug(f"将列 {col} 转换为布尔类型")
                
                elif dtype == 'string':
                    cleaned_df[col] = cleaned_df[col].astype(str)
                    logger.debug(f"将列 {col} 转换为字符串类型")
                
            except Exception as e:
                logger.error(f"转换列 {col} 的类型为 {dtype} 失败: {e}")
        
        return cleaned_df
    
    def _remove_duplicates(self, df: pd.DataFrame, rules: Dict[str, Any]) -> pd.DataFrame:
        """
        去重处理
        
        Args:
            df: 数据框
            rules: 去重规则
            
        Returns:
            去重后的数据框
        """
        cleaned_df = df.copy()
        
        if isinstance(rules, bool):
            original_len = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates()
            logger.debug(f"删除重复行，从 {original_len} 行减少到 {len(cleaned_df)} 行")
        elif isinstance(rules, dict):
            subset = rules.get('subset', None)
            keep = rules.get('keep', 'first')
            original_len = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates(subset=subset, keep=keep)
            logger.debug(f"删除指定列的重复行，从 {original_len} 行减少到 {len(cleaned_df)} 行")
        
        return cleaned_df
    
    def _handle_outliers(self, df: pd.DataFrame, rules: Dict[str, Any]) -> pd.DataFrame:
        """
        处理异常值
        
        Args:
            df: 数据框
            rules: 异常值处理规则
            
        Returns:
            处理后的数据框
        """
        cleaned_df = df.copy()
        
        for col, rule in rules.items():
            if col not in cleaned_df.columns or not pd.api.types.is_numeric_dtype(cleaned_df[col]):
                continue
            
            try:
                if rule == 'drop':
                    # 使用IQR方法检测异常值并删除
                    q1 = cleaned_df[col].quantile(0.25)
                    q3 = cleaned_df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    original_len = len(cleaned_df)
                    cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
                    logger.debug(f"删除列 {col} 中的异常值，从 {original_len} 行减少到 {len(cleaned_df)} 行")
                
                elif rule == 'capping':
                    # 使用IQR方法检测异常值并进行截断
                    q1 = cleaned_df[col].quantile(0.25)
                    q3 = cleaned_df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    cleaned_df[col] = cleaned_df[col].clip(lower=lower_bound, upper=upper_bound)
                    logger.debug(f"截断列 {col} 中的异常值")
                
                elif rule.startswith('z_score:'):
                    # 使用Z-score方法检测异常值
                    threshold = float(rule.split(':', 1)[1])
                    z_scores = np.abs((cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std())
                    
                    original_len = len(cleaned_df)
                    cleaned_df = cleaned_df[z_scores <= threshold]
                    logger.debug(f"使用Z-score方法删除列 {col} 中的异常值，从 {original_len} 行减少到 {len(cleaned_df)} 行")
                
            except Exception as e:
                logger.error(f"处理列 {col} 的异常值失败: {e}")
        
        return cleaned_df
    
    def _clean_text_columns(self, df: pd.DataFrame, rules: Dict[str, List[str]]) -> pd.DataFrame:
        """
        文本清洗
        
        Args:
            df: 数据框
            rules: 文本清洗规则
            
        Returns:
            清洗后的数据框
        """
        cleaned_df = df.copy()
        
        for col, cleaning_ops in rules.items():
            if col not in cleaned_df.columns or not pd.api.types.is_string_dtype(cleaned_df[col]):
                continue
            
            try:
                for op in cleaning_ops:
                    if op == 'trim':
                        cleaned_df[col] = cleaned_df[col].str.strip()
                        logger.debug(f"去除列 {col} 中的首尾空格")
                    
                    elif op == 'lower':
                        cleaned_df[col] = cleaned_df[col].str.lower()
                        logger.debug(f"将列 {col} 转换为小写")
                    
                    elif op == 'upper':
                        cleaned_df[col] = cleaned_df[col].str.upper()
                        logger.debug(f"将列 {col} 转换为大写")
                    
                    elif op == 'title':
                        cleaned_df[col] = cleaned_df[col].str.title()
                        logger.debug(f"将列 {col} 转换为标题格式")
                    
                    elif op == 'remove_whitespace':
                        cleaned_df[col] = cleaned_df[col].str.replace(r'\s+', ' ', regex=True)
                        logger.debug(f"去除列 {col} 中的多余空格")
                    
                    elif op.startswith('remove:'):
                        # 移除指定字符或字符串
                        pattern = op.split(':', 1)[1]
                        cleaned_df[col] = cleaned_df[col].str.replace(pattern, '', regex=False)
                        logger.debug(f"从列 {col} 中移除 '{pattern}'")
                    
                    elif op.startswith('replace:'):
                        # 替换指定字符或字符串
                        parts = op.split(':', 2)
                        if len(parts) >= 3:
                            old = parts[1]
                            new = parts[2]
                            cleaned_df[col] = cleaned_df[col].str.replace(old, new, regex=False)
                            logger.debug(f"将列 {col} 中的 '{old}' 替换为 '{new}'")
                    
            except Exception as e:
                logger.error(f"清洗列 {col} 的文本失败: {e}")
        
        return cleaned_df
    
    def _process_time_series(self, df: pd.DataFrame, rules: Dict[str, Any]) -> pd.DataFrame:
        """
        时间序列处理
        
        Args:
            df: 数据框
            rules: 时间序列处理规则
            
        Returns:
            处理后的数据框
        """
        cleaned_df = df.copy()
        
        # 检查是否指定了日期列
        if 'date_column' not in rules:
            logger.warning("时间序列处理规则中未指定日期列")
            return cleaned_df
        
        date_column = rules['date_column']
        if date_column not in cleaned_df.columns:
            logger.warning(f"数据框中不存在日期列 '{date_column}'")
            return cleaned_df
        
        # 确保日期列是日期类型
        if not pd.api.types.is_datetime64_any_dtype(cleaned_df[date_column]):
            cleaned_df[date_column] = pd.to_datetime(cleaned_df[date_column], errors='coerce')
            logger.debug(f"将列 {date_column} 转换为日期类型")
        
        # 排序
        if 'sort' in rules and rules['sort']:
            cleaned_df = cleaned_df.sort_values(by=date_column)
            logger.debug(f"按日期列 {date_column} 排序")
        
        # 去重日期
        if 'remove_duplicate_dates' in rules and rules['remove_duplicate_dates']:
            cleaned_df = cleaned_df.drop_duplicates(subset=[date_column])
            logger.debug(f"去除重复的日期")
        
        # 设置日期列作为索引
        if 'set_index' in rules and rules['set_index']:
            cleaned_df = cleaned_df.set_index(date_column)
            logger.debug(f"将日期列 {date_column} 设置为索引")
        
        # 填充时间序列缺失值
        if 'fill_missing_dates' in rules and rules['fill_missing_dates']:
            freq = rules.get('frequency', 'D')  # 默认频率为日
            cleaned_df = cleaned_df.resample(freq).ffill()
            logger.debug(f"填充缺失的日期，频率为 {freq}")
        
        return cleaned_df
    
    def _rename_columns(self, df: pd.DataFrame, rules: Dict[str, str]) -> pd.DataFrame:
        """
        重命名列
        
        Args:
            df: 数据框
            rules: 列重命名规则
            
        Returns:
            重命名后的数据框
        """
        cleaned_df = df.copy()
        
        try:
            cleaned_df = cleaned_df.rename(columns=rules)
            logger.debug(f"重命名列: {rules}")
        except Exception as e:
            logger.error(f"重命名列失败: {e}")
        
        return cleaned_df
    
    def normalize_numeric_columns(self, df: pd.DataFrame, columns: List[str], method: str = 'min-max') -> pd.DataFrame:
        """
        归一化数值列
        
        Args:
            df: 数据框
            columns: 要归一化的列
            method: 归一化方法，可选值：min-max, z-score
            
        Returns:
            归一化后的数据框
        """
        cleaned_df = df.copy()
        
        for col in columns:
            if col not in cleaned_df.columns or not pd.api.types.is_numeric_dtype(cleaned_df[col]):
                continue
            
            try:
                if method == 'min-max':
                    # Min-Max归一化
                    min_val = cleaned_df[col].min()
                    max_val = cleaned_df[col].max()
                    if max_val != min_val:
                        cleaned_df[col] = (cleaned_df[col] - min_val) / (max_val - min_val)
                    logger.debug(f"使用Min-Max方法归一化列 {col}")
                
                elif method == 'z-score':
                    # Z-score归一化
                    mean_val = cleaned_df[col].mean()
                    std_val = cleaned_df[col].std()
                    if std_val != 0:
                        cleaned_df[col] = (cleaned_df[col] - mean_val) / std_val
                    logger.debug(f"使用Z-score方法归一化列 {col}")
                
            except Exception as e:
                logger.error(f"归一化列 {col} 失败: {e}")
        
        return cleaned_df
    
    def clean_currency_data(self, df: pd.DataFrame, columns: List[str], currency_symbol: str = '$') -> pd.DataFrame:
        """
        清洗货币数据
        
        Args:
            df: 数据框
            columns: 要清洗的货币列
            currency_symbol: 货币符号
            
        Returns:
            清洗后的数据框
        """
        cleaned_df = df.copy()
        
        for col in columns:
            if col not in cleaned_df.columns:
                continue
            
            try:
                # 去除货币符号和千位分隔符
                cleaned_df[col] = cleaned_df[col].str.replace(currency_symbol, '', regex=False)
                cleaned_df[col] = cleaned_df[col].str.replace(',', '', regex=False)
                
                # 转换为数值类型
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                logger.debug(f"清洗列 {col} 的货币数据")
            
            except Exception as e:
                logger.error(f"清洗列 {col} 的货币数据失败: {e}")
        
        return cleaned_df
