#!/usr/bin/env python3
"""
增强型数据处理器
负责数据文件解读与结构化信息提取，支持复杂的数据分析和洞察生成
"""

from typing import Dict, Any, List, Optional
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class EnhancedDataProcessor:
    """
    增强型数据处理器，提供高级数据文件解读与结构化信息提取功能
    """
    
    def __init__(self):
        self.supported_formats = ['xlsx', 'csv', 'txt']
        self.keyword_patterns = {
            'time_series': ['date', 'datetime', 'time', 'timestamp', 'period', 'day', 'month', 'year'],
            'event': ['event', 'news', 'incident', 'announcement', 'release', 'speech', 'meeting'],
            'price': ['price', 'value', 'rate', 'cost', 'amount', 'volume'],
            'change': ['change', 'delta', 'difference', 'variation', 'percent_change', 'return']
        }
    
    def load_and_process_data(self, file_path: str) -> Dict[str, Any]:
        """
        加载并处理数据文件
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            Dict: 包含数据、元数据和洞察的处理结果
        """
        try:
            file_ext = self._get_file_extension(file_path)
            
            if file_ext not in self.supported_formats:
                raise ValueError(f"不支持的文件格式: {file_ext}")
            
            # 加载数据
            df = self._load_data(file_path, file_ext)
            
            # 数据清洗
            df = self._clean_data(df)
            
            # 增强数据（添加时间特征、识别事件等）
            df, enhancements = self._enhance_data(df)
            
            # 生成洞察
            insights = self._generate_insights(df, enhancements)
            
            # 生成可视化建议
            visualization_suggestions = self._generate_visualization_suggestions(df, insights)
            
            # 构建结果
            result = {
                'data': df.to_dict('records'),
                'metadata': self._generate_metadata(df, file_path),
                'enhancements': enhancements,
                'insights': insights,
                'visualization_suggestions': visualization_suggestions,
                'success': True
            }
            
            return result
            
        except Exception as e:
            logger.error(f"数据处理失败: {str(e)}")
            return {
                'error': str(e),
                'success': False
            }
    
    def _get_file_extension(self, file_path: str) -> str:
        """
        获取文件扩展名
        """
        return os.path.splitext(file_path)[1].lower().lstrip('.')
    
    def _load_data(self, file_path: str, file_ext: str) -> pd.DataFrame:
        """
        根据文件格式加载数据
        """
        if file_ext == 'csv':
            return self._load_csv(file_path)
        elif file_ext == 'xlsx':
            return self._load_excel(file_path)
        elif file_ext == 'txt':
            return self._load_text(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}")
    
    def _load_csv(self, file_path: str) -> pd.DataFrame:
        """
        加载CSV文件
        """
        try:
            return pd.read_csv(file_path, index_col=False, parse_dates=True)
        except UnicodeDecodeError:
            return pd.read_csv(file_path, index_col=False, parse_dates=True, encoding='gbk')
    
    def _load_excel(self, file_path: str) -> pd.DataFrame:
        """
        加载Excel文件
        """
        return pd.read_excel(file_path, index_col=False, parse_dates=True)
    
    def _load_text(self, file_path: str) -> pd.DataFrame:
        """
        加载文本文件
        """
        return pd.read_csv(file_path, index_col=False, sep='\s+', parse_dates=True)
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗
        """
        cleaned_df = df.copy()
        
        # 标准化列名
        cleaned_df.columns = [col.strip().lower().replace(' ', '_') for col in cleaned_df.columns]
        
        # 处理缺失值
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                cleaned_df[col] = cleaned_df[col].fillna('')
            else:
                cleaned_df[col] = cleaned_df[col].fillna(0)
        
        # 标准化日期格式
        for col in cleaned_df.columns:
            if any(keyword in col for keyword in self.keyword_patterns['time_series']):
                try:
                    cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
                except Exception:
                    pass
        
        return cleaned_df
    
    def _enhance_data(self, df: pd.DataFrame) -> (pd.DataFrame, Dict[str, Any]):
        """
        增强数据，添加时间特征、识别事件等
        """
        enhanced_df = df.copy()
        enhancements = {
            'time_features': {},
            'event_detection': {},
            'trend_features': {}
        }
        
        # 识别时间序列数据
        time_columns = [col for col in enhanced_df.columns 
                       if enhanced_df[col].dtype == 'datetime64[ns]' 
                       or any(keyword in col for keyword in self.keyword_patterns['time_series'])]
        
        if time_columns:
            time_col = time_columns[0]
            enhanced_df = enhanced_df.sort_values(by=time_col)
            
            # 添加时间特征
            enhancements['time_features'] = self._add_time_features(enhanced_df, time_col)
            
            # 检测事件标签
            enhancements['event_detection'] = self._detect_events(enhanced_df, time_col)
            
            # 添加趋势特征
            numeric_cols = enhanced_df.select_dtypes(include=[np.number]).columns.tolist()
            for col in numeric_cols:
                enhancements['trend_features'][col] = self._add_trend_features(enhanced_df[col])
        
        return enhanced_df, enhancements
    
    def _add_time_features(self, df: pd.DataFrame, time_col: str) -> Dict[str, Any]:
        """
        添加时间特征
        """
        features = {
            'time_column': time_col,
            'time_range': {
                'start': df[time_col].min().isoformat() if not df[time_col].empty else '',
                'end': df[time_col].max().isoformat() if not df[time_col].empty else ''
            },
            'periodicity': self._detect_periodicity(df, time_col)
        }
        
        return features
    
    def _detect_periodicity(self, df: pd.DataFrame, time_col: str) -> str:
        """
        检测时间序列的周期性
        """
        if len(df) < 3:
            return 'unknown'
        
        # 计算时间差
        df_sorted = df.sort_values(by=time_col)
        time_diffs = df_sorted[time_col].diff().dropna()
        
        # 检测周期性
        if len(time_diffs) > 0:
            mode_diff = time_diffs.mode().iloc[0]
            
            if mode_diff < timedelta(days=2):
                return 'daily'
            elif timedelta(days=2) <= mode_diff < timedelta(days=10):
                return 'weekly'
            elif timedelta(days=10) <= mode_diff < timedelta(days=40):
                return 'monthly'
            elif timedelta(days=40) <= mode_diff < timedelta(days=100):
                return 'quarterly'
            else:
                return 'yearly'
        
        return 'unknown'
    
    def _detect_events(self, df: pd.DataFrame, time_col: str) -> Dict[str, Any]:
        """
        检测数据中的事件
        """
        events = {
            'significant_changes': [],
            'outliers': [],
            'trend_changes': []
        }
        
        # 检查是否有事件列
        event_columns = [col for col in df.columns 
                        if any(keyword in col for keyword in self.keyword_patterns['event'])]
        
        if event_columns:
            events['event_columns'] = event_columns
            
            # 提取事件信息
            for col in event_columns:
                events['events_from_columns'] = df[[time_col, col]].dropna().to_dict('records')
        
        # 检测数值变化事件
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            # 检测显著变化（超过标准差的变化）
            significant_changes = self._detect_significant_changes(df, time_col, col)
            if significant_changes:
                events['significant_changes'].extend(significant_changes)
            
            # 检测异常值
            outliers = self._detect_outliers(df, time_col, col)
            if outliers:
                events['outliers'].extend(outliers)
        
        return events
    
    def _detect_significant_changes(self, df: pd.DataFrame, time_col: str, value_col: str) -> List[Dict]:
        """
        检测数值的显著变化
        """
        changes = []
        df_sorted = df.sort_values(by=time_col)
        
        # 计算变化率
        df_sorted['change_rate'] = df_sorted[value_col].pct_change()
        df_sorted['absolute_change'] = df_sorted[value_col].diff()
        
        # 设置阈值
        threshold = df_sorted['change_rate'].std() * 2
        
        # 检测显著变化
        significant_changes = df_sorted[abs(df_sorted['change_rate']) > threshold]
        
        for _, row in significant_changes.iterrows():
            changes.append({
                'date': row[time_col].isoformat(),
                'column': value_col,
                'previous_value': row[value_col] - row['absolute_change'],
                'current_value': row[value_col],
                'change_rate': row['change_rate'],
                'absolute_change': row['absolute_change']
            })
        
        return changes
    
    def _detect_outliers(self, df: pd.DataFrame, time_col: str, value_col: str) -> List[Dict]:
        """
        检测异常值
        """
        outliers = []
        
        # 使用IQR方法检测异常值
        Q1 = df[value_col].quantile(0.25)
        Q3 = df[value_col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (df[value_col] < lower_bound) | (df[value_col] > upper_bound)
        outlier_rows = df[outlier_mask]
        
        for _, row in outlier_rows.iterrows():
            outliers.append({
                'date': row[time_col].isoformat() if time_col in row and pd.notna(row[time_col]) else '',
                'column': value_col,
                'value': row[value_col],
                'is_outlier': True
            })
        
        return outliers
    
    def _add_trend_features(self, series: pd.Series) -> Dict[str, Any]:
        """
        添加趋势特征
        """
        features = {}
        
        # 计算移动平均
        features['sma_3'] = series.rolling(window=3).mean().tolist()
        features['sma_7'] = series.rolling(window=7).mean().tolist()
        
        # 计算趋势
        if len(series) > 1:
            features['trend'] = {
                'slope': float(np.polyfit(range(len(series)), series, 1)[0]),
                'direction': '上升' if np.polyfit(range(len(series)), series, 1)[0] > 0 else '下降' if np.polyfit(range(len(series)), series, 1)[0] < 0 else '平稳'
            }
        
        return features
    
    def _generate_insights(self, df: pd.DataFrame, enhancements: Dict[str, Any]) -> List[Dict]:
        """
        生成数据洞察
        """
        insights = []
        
        # 趋势洞察
        trend_insights = self._generate_trend_insights(df, enhancements)
        insights.extend(trend_insights)
        
        # 相关性洞察
        correlation_insights = self._generate_correlation_insights(df)
        insights.extend(correlation_insights)
        
        # 事件影响洞察
        event_insights = self._generate_event_impact_insights(df, enhancements)
        insights.extend(event_insights)
        
        return insights
    
    def _generate_trend_insights(self, df: pd.DataFrame, enhancements: Dict[str, Any]) -> List[Dict]:
        """
        生成趋势洞察
        """
        insights = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        time_features = enhancements.get('time_features', {})
        
        if time_features and numeric_cols:
            time_col = time_features['time_column']
            
            for col in numeric_cols:
                # 计算整体趋势
                if len(df) > 1:
                    slope = float(np.polyfit(range(len(df)), df[col], 1)[0])
                    direction = '上升' if slope > 0 else '下降' if slope < 0 else '平稳'
                    
                    insights.append({
                        'type': 'trend',
                        'category': 'overall_trend',
                        'description': f'{col} 在 {time_features["time_range"]["start"]} 到 {time_features["time_range"]["end"]} 期间呈现 {direction} 趋势',
                        'data_points': {
                            'column': col,
                            'time_column': time_col,
                            'slope': slope,
                            'trend_direction': direction
                        },
                        'relevance': 0.8
                    })
        
        return insights
    
    def _generate_correlation_insights(self, df: pd.DataFrame) -> List[Dict]:
        """
        生成相关性洞察
        """
        insights = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            # 计算相关系数矩阵
            corr_matrix = df[numeric_cols].corr()
            
            # 找出强相关关系
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    col1 = numeric_cols[i]
                    col2 = numeric_cols[j]
                    corr = corr_matrix.iloc[i, j]
                    
                    if abs(corr) > 0.7:
                        correlation_type = '正相关' if corr > 0 else '负相关'
                        strength = '强' if abs(corr) > 0.9 else '中强' if abs(corr) > 0.8 else '较强'
                        
                        insights.append({
                            'type': 'correlation',
                            'category': 'strong_correlation',
                            'description': f'{col1} 与 {col2} 之间存在 {strength} {correlation_type} 关系（相关系数: {corr:.2f}）',
                            'data_points': {
                                'columns': [col1, col2],
                                'correlation_coefficient': corr,
                                'correlation_type': correlation_type,
                                'strength': strength
                            },
                            'relevance': 0.85
                        })
        
        return insights
    
    def _generate_event_impact_insights(self, df: pd.DataFrame, enhancements: Dict[str, Any]) -> List[Dict]:
        """
        生成事件影响洞察
        """
        insights = []
        
        event_detection = enhancements.get('event_detection', {})
        significant_changes = event_detection.get('significant_changes', [])
        
        if significant_changes:
            # 按列分组
            changes_by_col = {}
            for change in significant_changes:
                col = change['column']
                if col not in changes_by_col:
                    changes_by_col[col] = []
                changes_by_col[col].append(change)
            
            # 生成洞察
            for col, changes in changes_by_col.items():
                insights.append({
                    'type': 'event',
                    'category': 'significant_changes',
                    'description': f'{col} 在观察期内发生了 {len(changes)} 次显著变化',
                    'data_points': {
                        'column': col,
                        'change_count': len(changes),
                        'changes': changes
                    },
                    'relevance': 0.9
                })
        
        return insights
    
    def _generate_visualization_suggestions(self, df: pd.DataFrame, insights: List[Dict]) -> List[Dict]:
        """
        生成可视化建议
        """
        suggestions = []
        
        # 根据洞察类型生成建议
        for insight in insights:
            if insight['type'] == 'trend':
                # 趋势图建议
                suggestions.append({
                    'chart_type': 'line_chart',
                    'title': f'{insight["data_points"]["column"]} 趋势图',
                    'description': '展示时间序列数据的趋势变化',
                    'data_columns': [insight["data_points"]["time_column"], insight["data_points"]["column"]],
                    'insight_reference': insight['description']
                })
            
            elif insight['type'] == 'correlation':
                # 相关性图表建议
                suggestions.append({
                    'chart_type': 'scatter_plot',
                    'title': f'{insight["data_points"]["columns"][0]} 与 {insight["data_points"]["columns"][1]} 相关性分析',
                    'description': '展示两个变量之间的相关关系',
                    'data_columns': insight["data_points"]["columns"],
                    'insight_reference': insight['description']
                })
        
        # 生成综合建议
        time_columns = [col for col in df.columns 
                       if df[col].dtype == 'datetime64[ns]' 
                       or any(keyword in col for keyword in self.keyword_patterns['time_series'])]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if time_columns and numeric_cols:
            # 时间序列对比图
            suggestions.append({
                'chart_type': 'multiple_line_chart',
                'title': '多指标时间序列对比',
                'description': '展示多个数值指标随时间的变化趋势',
                'data_columns': time_columns + numeric_cols,
                'insight_reference': '综合展示各指标的时间变化趋势'
            })
        
        if len(numeric_cols) >= 3:
            # 相关系数热力图
            suggestions.append({
                'chart_type': 'correlation_heatmap',
                'title': '指标相关性热力图',
                'description': '展示所有数值指标之间的相关关系',
                'data_columns': numeric_cols,
                'insight_reference': '综合分析各指标之间的相关性'
            })
        
        return suggestions
    
    def _generate_metadata(self, df: pd.DataFrame, file_path: str) -> Dict[str, Any]:
        """
        生成数据元数据
        """
        return {
            'file_name': os.path.basename(file_path),
            'file_path': file_path,
            'file_size': os.path.getsize(file_path),
            'rows': len(df),
            'columns': list(df.columns),
            'column_types': df.dtypes.astype(str).to_dict(),
            'generated_at': datetime.now().isoformat()
        }
    
    def batch_process_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        批量处理多个文件
        
        Args:
            file_paths: 文件路径列表
            
        Returns:
            Dict: 包含所有文件处理结果的汇总
        """
        results = {
            'processed_files': [],
            'summary': {
                'total_files': len(file_paths),
                'successful': 0,
                'failed': 0
            }
        }
        
        for file_path in file_paths:
            result = self.load_and_process_data(file_path)
            result['file_path'] = file_path
            results['processed_files'].append(result)
            
            if result['success']:
                results['summary']['successful'] += 1
            else:
                results['summary']['failed'] += 1
        
        return results
