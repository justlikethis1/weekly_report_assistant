from typing import Dict, Any, List
from .base import FileProcessor
import os
import pandas as pd
import numpy as np
from datetime import datetime

class SpreadsheetProcessor(FileProcessor):
    """电子表格文件处理器，支持XLSX和CSV格式，包含数据清洗和标准化功能"""
    
    def __init__(self):
        """
        初始化电子表格处理器
        """
        super().__init__()
        self.supported_extensions = ['xlsx', 'csv']
    
    def process(self, file_path: str) -> Dict[str, Any]:
        """
        处理电子表格文件，包括读取、清洗和标准化
        
        Args:
            file_path: 电子表格文件路径
            
        Returns:
            Dict[str, Any]: 包含文本内容、处理后的数据和元数据的字典
        """
        try:
            file_ext = self.get_file_extension(file_path)
            
            # 根据文件扩展名选择不同的读取方式
            if file_ext not in self.supported_extensions:
                return {
                    'text': '',
                    'metadata': {},
                    'success': False,
                    'error': f"不支持的文件格式: {file_ext}"
                }
            
            # 读取文件
            df = self._read_file(file_path, file_ext)
            
            # 清洗和标准化数据
            cleaned_df = self._clean_and_standardize(df)
            
            # 转换为文本表示
            text = cleaned_df.to_string()
            
            # 提取基本统计信息
            stats = self._extract_statistics(cleaned_df)
            
            # 提取结构化数据
            structured_data = self._extract_structured_data(cleaned_df)
            
            metadata = {
                'file_type': file_ext,
                'file_name': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path),
                'spreadsheet_stats': stats,
                'content_length': len(text),
                'data_cleaning_info': {
                    'original_rows': len(df),
                    'cleaned_rows': len(cleaned_df),
                    'columns_processed': len(cleaned_df.columns),
                    'missing_values': cleaned_df.isnull().sum().to_dict()
                }
            }
            
            return {
                'text': text,
                'dataframe': cleaned_df.to_dict('records'),
                'structured_data': structured_data,
                'metadata': metadata,
                'success': True
            }
        except Exception as e:
            import traceback
            return {
                'text': '',
                'dataframe': [],
                'structured_data': {},
                'metadata': {},
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _read_file(self, file_path: str, file_ext: str) -> pd.DataFrame:
        """
        根据文件扩展名读取文件
        
        Args:
            file_path: 文件路径
            file_ext: 文件扩展名
            
        Returns:
            pd.DataFrame: 读取的数据框
        """
        if file_ext == 'xlsx':
            return pd.read_excel(file_path)
        elif file_ext == 'csv':
            # 自动检测编码和分隔符
            try:
                return pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    return pd.read_csv(file_path, encoding='gbk')
                except UnicodeDecodeError:
                    return pd.read_csv(file_path, encoding='latin1')
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}")
    
    def _clean_and_standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗和标准化数据
        
        Args:
            df: 原始数据框
            
        Returns:
            pd.DataFrame: 清洗后的数据框
        """
        cleaned_df = df.copy()
        
        # 1. 标准化列名
        cleaned_df.columns = [col.strip().lower().replace(' ', '_') for col in cleaned_df.columns]
        
        # 2. 处理缺失值
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                # 文本列用空字符串填充
                cleaned_df[col] = cleaned_df[col].fillna('')
            else:
                # 数值列用0填充
                cleaned_df[col] = cleaned_df[col].fillna(0)
        
        # 3. 标准化日期格式
        for col in cleaned_df.columns:
            if any(keyword in col for keyword in ['date', 'time', 'datetime']):
                try:
                    cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='ignore')
                except Exception as e:
                    print(f"日期列 {col} 处理失败: {e}")
        
        # 4. 去除重复行
        cleaned_df = cleaned_df.drop_duplicates()
        
        # 5. 去除文本列的前后空格
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                cleaned_df[col] = cleaned_df[col].str.strip()
        
        return cleaned_df
    
    def _extract_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        提取数据统计信息
        
        Args:
            df: 数据框
            
        Returns:
            Dict: 统计信息
        """
        stats = {
            'rows': len(df),
            'columns': len(df.columns),
            'columns_list': df.columns.tolist()
        }
        
        # 提取数值列的统计信息
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_stats = df[numeric_cols].describe().to_dict()
            stats['numeric_statistics'] = numeric_stats
        
        # 提取文本列的统计信息
        text_cols = df.select_dtypes(include=['object']).columns
        if len(text_cols) > 0:
            text_stats = {}
            for col in text_cols:
                text_stats[col] = {
                    'unique_values': df[col].nunique(),
                    'top_value': df[col].mode().iloc[0] if not df[col].empty else ''
                }
            stats['text_statistics'] = text_stats
        
        return stats
    
    def _extract_structured_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        提取结构化数据
        
        Args:
            df: 数据框
            
        Returns:
            Dict: 结构化数据
        """
        structured_data = {
            'time_series_data': {},
            'categorical_data': {},
            'numerical_data': {}
        }
        
        # 识别时间序列数据
        time_cols = [col for col in df.columns if df[col].dtype == 'datetime64[ns]' or 'date' in col.lower()]
        if time_cols:
            # 假设第一个时间列为时间索引
            time_col = time_cols[0]
            structured_data['time_series_data'] = {
                'time_column': time_col,
                'time_range': {
                    'start': df[time_col].min().isoformat() if not df[time_col].empty else '',
                    'end': df[time_col].max().isoformat() if not df[time_col].empty else ''
                }
            }
        
        # 识别分类数据和数值数据
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].nunique() < len(df) * 0.5:
                structured_data['categorical_data'][col] = {
                    'categories': df[col].unique().tolist(),
                    'counts': df[col].value_counts().to_dict()
                }
            elif df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                structured_data['numerical_data'][col] = {
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'median': float(df[col].median()),
                    'std': float(df[col].std())
                }
        
        return structured_data
