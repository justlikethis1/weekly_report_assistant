#!/usr/bin/env python3
"""
数据增强器模块
负责增强和丰富数据
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DataEnhancer:
    """数据增强器，负责增强和丰富数据"""
    
    def __init__(self):
        """初始化数据增强器"""
        logger.info("DataEnhancer初始化完成")
    
    def enhance_dataframe(self, df: pd.DataFrame, enhancement_rules: Dict[str, Any]) -> pd.DataFrame:
        """
        增强数据框
        
        Args:
            df: 要增强的数据框
            enhancement_rules: 增强规则
            
        Returns:
            增强后的数据框
        """
        enhanced_df = df.copy()
        
        # 计算派生列
        if 'derived_columns' in enhancement_rules:
            enhanced_df = self._calculate_derived_columns(enhanced_df, enhancement_rules['derived_columns'])
        
        # 特征工程
        if 'feature_engineering' in enhancement_rules:
            enhanced_df = self._perform_feature_engineering(enhanced_df, enhancement_rules['feature_engineering'])
        
        # 时间特征提取
        if 'time_features' in enhancement_rules:
            enhanced_df = self._extract_time_features(enhanced_df, enhancement_rules['time_features'])
        
        # 统计特征计算
        if 'statistical_features' in enhancement_rules:
            enhanced_df = self._calculate_statistical_features(enhanced_df, enhancement_rules['statistical_features'])
        
        # 分组聚合特征
        if 'group_features' in enhancement_rules:
            enhanced_df = self._calculate_group_features(enhanced_df, enhancement_rules['group_features'])
        
        # 文本特征提取
        if 'text_features' in enhancement_rules:
            enhanced_df = self._extract_text_features(enhanced_df, enhancement_rules['text_features'])
        
        # 分类特征编码
        if 'categorical_encoding' in enhancement_rules:
            enhanced_df = self._encode_categorical_features(enhanced_df, enhancement_rules['categorical_encoding'])
        
        # 标准化/归一化
        if 'scaling' in enhancement_rules:
            enhanced_df = self._scale_features(enhanced_df, enhancement_rules['scaling'])
        
        return enhanced_df
    
    def _calculate_derived_columns(self, df: pd.DataFrame, rules: Dict[str, str]) -> pd.DataFrame:
        """
        计算派生列
        
        Args:
            df: 数据框
            rules: 派生列计算规则
            
        Returns:
            增强后的数据框
        """
        enhanced_df = df.copy()
        
        for new_col, expression in rules.items():
            try:
                # 使用eval函数计算表达式
                # 注意：eval可能存在安全风险，这里假设规则是可信的
                enhanced_df[new_col] = enhanced_df.eval(expression)
                logger.debug(f"计算派生列 {new_col}: {expression}")
            except Exception as e:
                logger.error(f"计算派生列 {new_col} 失败: {e}")
        
        return enhanced_df
    
    def _perform_feature_engineering(self, df: pd.DataFrame, rules: Dict[str, Any]) -> pd.DataFrame:
        """
        特征工程
        
        Args:
            df: 数据框
            rules: 特征工程规则
            
        Returns:
            增强后的数据框
        """
        enhanced_df = df.copy()
        
        # 移动窗口特征
        if 'rolling_features' in rules:
            enhanced_df = self._calculate_rolling_features(enhanced_df, rules['rolling_features'])
        
        # 指数加权移动平均
        if 'ewma_features' in rules:
            enhanced_df = self._calculate_ewma_features(enhanced_df, rules['ewma_features'])
        
        # 差分特征
        if 'diff_features' in rules:
            enhanced_df = self._calculate_diff_features(enhanced_df, rules['diff_features'])
        
        # 滞后特征
        if 'lag_features' in rules:
            enhanced_df = self._calculate_lag_features(enhanced_df, rules['lag_features'])
        
        return enhanced_df
    
    def _calculate_rolling_features(self, df: pd.DataFrame, rules: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        计算移动窗口特征
        
        Args:
            df: 数据框
            rules: 移动窗口特征计算规则
            
        Returns:
            增强后的数据框
        """
        enhanced_df = df.copy()
        
        for col, config in rules.items():
            if col not in enhanced_df.columns:
                continue
            
            window = config.get('window', 7)
            functions = config.get('functions', ['mean', 'std'])
            
            try:
                for func in functions:
                    if func == 'mean':
                        enhanced_df[f'{col}_rolling_{window}_mean'] = enhanced_df[col].rolling(window=window).mean()
                    elif func == 'std':
                        enhanced_df[f'{col}_rolling_{window}_std'] = enhanced_df[col].rolling(window=window).std()
                    elif func == 'max':
                        enhanced_df[f'{col}_rolling_{window}_max'] = enhanced_df[col].rolling(window=window).max()
                    elif func == 'min':
                        enhanced_df[f'{col}_rolling_{window}_min'] = enhanced_df[col].rolling(window=window).min()
                    elif func == 'sum':
                        enhanced_df[f'{col}_rolling_{window}_sum'] = enhanced_df[col].rolling(window=window).sum()
                    elif func == 'median':
                        enhanced_df[f'{col}_rolling_{window}_median'] = enhanced_df[col].rolling(window=window).median()
                    
                    logger.debug(f"计算列 {col} 的移动窗口 {window} {func} 特征")
            
            except Exception as e:
                logger.error(f"计算列 {col} 的移动窗口特征失败: {e}")
        
        return enhanced_df
    
    def _calculate_ewma_features(self, df: pd.DataFrame, rules: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        计算指数加权移动平均特征
        
        Args:
            df: 数据框
            rules: 指数加权移动平均特征计算规则
            
        Returns:
            增强后的数据框
        """
        enhanced_df = df.copy()
        
        for col, config in rules.items():
            if col not in enhanced_df.columns:
                continue
            
            span = config.get('span', 10)
            
            try:
                enhanced_df[f'{col}_ewma_{span}'] = enhanced_df[col].ewm(span=span).mean()
                logger.debug(f"计算列 {col} 的指数加权移动平均特征，span={span}")
            
            except Exception as e:
                logger.error(f"计算列 {col} 的指数加权移动平均特征失败: {e}")
        
        return enhanced_df
    
    def _calculate_diff_features(self, df: pd.DataFrame, rules: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        计算差分特征
        
        Args:
            df: 数据框
            rules: 差分特征计算规则
            
        Returns:
            增强后的数据框
        """
        enhanced_df = df.copy()
        
        for col, config in rules.items():
            if col not in enhanced_df.columns:
                continue
            
            periods = config.get('periods', 1)
            
            try:
                enhanced_df[f'{col}_diff_{periods}'] = enhanced_df[col].diff(periods=periods)
                logger.debug(f"计算列 {col} 的差分特征，periods={periods}")
            
            except Exception as e:
                logger.error(f"计算列 {col} 的差分特征失败: {e}")
        
        return enhanced_df
    
    def _calculate_lag_features(self, df: pd.DataFrame, rules: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        计算滞后特征
        
        Args:
            df: 数据框
            rules: 滞后特征计算规则
            
        Returns:
            增强后的数据框
        """
        enhanced_df = df.copy()
        
        for col, config in rules.items():
            if col not in enhanced_df.columns:
                continue
            
            lags = config.get('lags', [1, 3, 7])
            
            try:
                for lag in lags:
                    enhanced_df[f'{col}_lag_{lag}'] = enhanced_df[col].shift(lag)
                    logger.debug(f"计算列 {col} 的滞后特征，lag={lag}")
            
            except Exception as e:
                logger.error(f"计算列 {col} 的滞后特征失败: {e}")
        
        return enhanced_df
    
    def _extract_time_features(self, df: pd.DataFrame, rules: Dict[str, Any]) -> pd.DataFrame:
        """
        提取时间特征
        
        Args:
            df: 数据框
            rules: 时间特征提取规则
            
        Returns:
            增强后的数据框
        """
        enhanced_df = df.copy()
        
        if 'date_column' not in rules:
            logger.warning("时间特征提取规则中未指定日期列")
            return enhanced_df
        
        date_column = rules['date_column']
        if date_column not in enhanced_df.columns:
            logger.warning(f"数据框中不存在日期列 '{date_column}'")
            return enhanced_df
        
        # 确保日期列是日期类型
        if not pd.api.types.is_datetime64_any_dtype(enhanced_df[date_column]):
            enhanced_df[date_column] = pd.to_datetime(enhanced_df[date_column], errors='coerce')
            logger.debug(f"将列 {date_column} 转换为日期类型")
        
        features = rules.get('features', ['year', 'month', 'day', 'weekday', 'quarter'])
        
        try:
            for feature in features:
                if feature == 'year':
                    enhanced_df[f'{date_column}_year'] = enhanced_df[date_column].dt.year
                elif feature == 'month':
                    enhanced_df[f'{date_column}_month'] = enhanced_df[date_column].dt.month
                elif feature == 'day':
                    enhanced_df[f'{date_column}_day'] = enhanced_df[date_column].dt.day
                elif feature == 'weekday':
                    enhanced_df[f'{date_column}_weekday'] = enhanced_df[date_column].dt.weekday
                elif feature == 'quarter':
                    enhanced_df[f'{date_column}_quarter'] = enhanced_df[date_column].dt.quarter
                elif feature == 'weekofyear':
                    enhanced_df[f'{date_column}_weekofyear'] = enhanced_df[date_column].dt.isocalendar().week
                elif feature == 'dayofyear':
                    enhanced_df[f'{date_column}_dayofyear'] = enhanced_df[date_column].dt.dayofyear
                elif feature == 'is_weekend':
                    enhanced_df[f'{date_column}_is_weekend'] = enhanced_df[date_column].dt.weekday.isin([5, 6]).astype(int)
                elif feature == 'is_month_start':
                    enhanced_df[f'{date_column}_is_month_start'] = enhanced_df[date_column].dt.is_month_start.astype(int)
                elif feature == 'is_month_end':
                    enhanced_df[f'{date_column}_is_month_end'] = enhanced_df[date_column].dt.is_month_end.astype(int)
                
                logger.debug(f"从日期列 {date_column} 提取 {feature} 特征")
        
        except Exception as e:
            logger.error(f"从日期列 {date_column} 提取时间特征失败: {e}")
        
        return enhanced_df
    
    def _calculate_statistical_features(self, df: pd.DataFrame, rules: Dict[str, Any]) -> pd.DataFrame:
        """
        计算统计特征
        
        Args:
            df: 数据框
            rules: 统计特征计算规则
            
        Returns:
            增强后的数据框
        """
        enhanced_df = df.copy()
        
        for config in rules:
            columns = config.get('columns', [])
            functions = config.get('functions', ['mean', 'std', 'min', 'max'])
            
            for col in columns:
                if col not in enhanced_df.columns or not pd.api.types.is_numeric_dtype(enhanced_df[col]):
                    continue
                
                try:
                    for func in functions:
                        if func == 'mean':
                            enhanced_df[f'{col}_mean'] = enhanced_df[col].mean()
                        elif func == 'std':
                            enhanced_df[f'{col}_std'] = enhanced_df[col].std()
                        elif func == 'min':
                            enhanced_df[f'{col}_min'] = enhanced_df[col].min()
                        elif func == 'max':
                            enhanced_df[f'{col}_max'] = enhanced_df[col].max()
                        elif func == 'median':
                            enhanced_df[f'{col}_median'] = enhanced_df[col].median()
                        elif func == 'sum':
                            enhanced_df[f'{col}_sum'] = enhanced_df[col].sum()
                        elif func == 'count':
                            enhanced_df[f'{col}_count'] = enhanced_df[col].count()
                        elif func == 'q25':
                            enhanced_df[f'{col}_q25'] = enhanced_df[col].quantile(0.25)
                        elif func == 'q75':
                            enhanced_df[f'{col}_q75'] = enhanced_df[col].quantile(0.75)
                        
                        logger.debug(f"计算列 {col} 的 {func} 统计特征")
                
                except Exception as e:
                    logger.error(f"计算列 {col} 的统计特征失败: {e}")
        
        return enhanced_df
    
    def _calculate_group_features(self, df: pd.DataFrame, rules: Dict[str, Any]) -> pd.DataFrame:
        """
        计算分组聚合特征
        
        Args:
            df: 数据框
            rules: 分组聚合特征计算规则
            
        Returns:
            增强后的数据框
        """
        enhanced_df = df.copy()
        
        for config in rules:
            groupby_cols = config.get('groupby', [])
            agg_cols = config.get('aggregations', {})
            
            if not groupby_cols or not agg_cols:
                continue
            
            try:
                # 计算分组聚合
                grouped = enhanced_df.groupby(groupby_cols).agg(agg_cols).reset_index()
                
                # 重命名列
                new_columns = []
                for col in grouped.columns:
                    if isinstance(col, tuple):
                        # 如果是多层列名，组合成新列名
                        new_columns.append(f'{col[0]}_{col[1]}')
                    else:
                        new_columns.append(col)
                grouped.columns = new_columns
                
                # 合并回原始数据框
                enhanced_df = enhanced_df.merge(grouped, on=groupby_cols, how='left')
                
                logger.debug(f"计算分组聚合特征，分组列: {groupby_cols}，聚合列: {list(agg_cols.keys())}")
            
            except Exception as e:
                logger.error(f"计算分组聚合特征失败: {e}")
        
        return enhanced_df
    
    def _extract_text_features(self, df: pd.DataFrame, rules: Dict[str, Any]) -> pd.DataFrame:
        """
        提取文本特征
        
        Args:
            df: 数据框
            rules: 文本特征提取规则
            
        Returns:
            增强后的数据框
        """
        enhanced_df = df.copy()
        
        for col, config in rules.items():
            if col not in enhanced_df.columns or not pd.api.types.is_string_dtype(enhanced_df[col]):
                continue
            
            features = config.get('features', ['length', 'word_count'])
            
            try:
                for feature in features:
                    if feature == 'length':
                        enhanced_df[f'{col}_length'] = enhanced_df[col].str.len()
                    elif feature == 'word_count':
                        enhanced_df[f'{col}_word_count'] = enhanced_df[col].str.split().str.len()
                    elif feature == 'sentence_count':
                        enhanced_df[f'{col}_sentence_count'] = enhanced_df[col].str.count(r'[.!?]') + 1
                    elif feature == 'has_number':
                        enhanced_df[f'{col}_has_number'] = enhanced_df[col].str.contains(r'\d').astype(int)
                    elif feature == 'has_upper':
                        enhanced_df[f'{col}_has_upper'] = enhanced_df[col].str.contains(r'[A-Z]').astype(int)
                    
                    logger.debug(f"从文本列 {col} 提取 {feature} 特征")
            
            except Exception as e:
                logger.error(f"从文本列 {col} 提取特征失败: {e}")
        
        return enhanced_df
    
    def _encode_categorical_features(self, df: pd.DataFrame, rules: Dict[str, Any]) -> pd.DataFrame:
        """
        编码分类特征
        
        Args:
            df: 数据框
            rules: 分类特征编码规则
            
        Returns:
            增强后的数据框
        """
        enhanced_df = df.copy()
        
        for col, config in rules.items():
            if col not in enhanced_df.columns:
                continue
            
            encoding_type = config.get('type', 'onehot')
            
            try:
                if encoding_type == 'onehot':
                    # 独热编码
                    dummies = pd.get_dummies(enhanced_df[col], prefix=col, drop_first=True)
                    enhanced_df = pd.concat([enhanced_df, dummies], axis=1)
                    logger.debug(f"对分类列 {col} 进行独热编码")
                
                elif encoding_type == 'label':
                    # 标签编码
                    enhanced_df[f'{col}_label'] = enhanced_df[col].astype('category').cat.codes
                    logger.debug(f"对分类列 {col} 进行标签编码")
                
                elif encoding_type == 'frequency':
                    # 频率编码
                    freq_encoding = enhanced_df[col].value_counts(normalize=True)
                    enhanced_df[f'{col}_frequency'] = enhanced_df[col].map(freq_encoding)
                    logger.debug(f"对分类列 {col} 进行频率编码")
            
            except Exception as e:
                logger.error(f"编码分类列 {col} 失败: {e}")
        
        return enhanced_df
    
    def _scale_features(self, df: pd.DataFrame, rules: Dict[str, Any]) -> pd.DataFrame:
        """
        标准化/归一化特征
        
        Args:
            df: 数据框
            rules: 特征缩放规则
            
        Returns:
            增强后的数据框
        """
        enhanced_df = df.copy()
        
        for col, config in rules.items():
            if col not in enhanced_df.columns or not pd.api.types.is_numeric_dtype(enhanced_df[col]):
                continue
            
            scaling_type = config.get('type', 'minmax')
            
            try:
                if scaling_type == 'minmax':
                    # Min-Max归一化
                    min_val = enhanced_df[col].min()
                    max_val = enhanced_df[col].max()
                    if max_val != min_val:
                        enhanced_df[f'{col}_scaled'] = (enhanced_df[col] - min_val) / (max_val - min_val)
                    logger.debug(f"对列 {col} 进行Min-Max归一化")
                
                elif scaling_type == 'standard':
                    # 标准归一化 (Z-score)
                    mean_val = enhanced_df[col].mean()
                    std_val = enhanced_df[col].std()
                    if std_val != 0:
                        enhanced_df[f'{col}_scaled'] = (enhanced_df[col] - mean_val) / std_val
                    logger.debug(f"对列 {col} 进行标准归一化")
            
            except Exception as e:
                logger.error(f"缩放列 {col} 失败: {e}")
        
        return enhanced_df
