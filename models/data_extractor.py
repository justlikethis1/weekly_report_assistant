#!/usr/bin/env python3
"""
数据提取层
负责从各种文件中提取和验证数据
"""

import pandas as pd
import os
import logging
from typing import Dict, Any, List
from docx import Document  # 添加对Word文档的支持

logger = logging.getLogger(__name__)

# 导入自定义数据处理类
from .data_cleaner import DataCleaner
from .data_enhancer import DataEnhancer


class FileParser:
    """文件解析器，负责解析不同格式的文件"""
    
    def __init__(self):
        """初始化文件解析器"""
        self.supported_formats = ['.xlsx', '.xls', '.csv', '.docx']  # 添加对.docx格式的支持    
    def can_parse(self, file_path: str) -> bool:
        """
        检查文件是否可以被解析
        
        Args:
            file_path: 文件路径
            
        Returns:
            如果文件可以被解析返回True，否则返回False
        """
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.supported_formats
    
    def parse(self, file_path: str) -> pd.DataFrame:
        """
        解析文件并返回DataFrame
        
        Args:
            file_path: 文件路径
            
        Returns:
            解析后的数据
            
        Raises:
            ValueError: 如果文件格式不支持或解析失败
        """
        if not self.can_parse(file_path):
            raise ValueError(f"不支持的文件格式: {file_path}")
        
        try:
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext in ['.xlsx', '.xls']:
                return pd.read_excel(file_path)
            elif ext == '.csv':
                return pd.read_csv(file_path, encoding='utf-8')
            elif ext == '.docx':
                # 解析Word文档
                doc = Document(file_path)
                paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
                
                # 将文档内容转换为DataFrame
                df = pd.DataFrame({
                    'paragraphs': paragraphs,
                    'document_type': ['word_document'] * len(paragraphs),
                    'file_name': [os.path.basename(file_path)] * len(paragraphs)
                })
                return df
        except Exception as e:
            logger.error(f"解析文件 {file_path} 失败: {e}")
            raise ValueError(f"解析文件 {file_path} 失败: {e}")
    
    def detect_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        自动检测数据框中的列类型
        
        Args:
            df: 数据框
            
        Returns:
            列类型映射
        """
        column_types = {}
        
        for column in df.columns:
            # 检测日期列
            if pd.api.types.is_datetime64_any_dtype(df[column]):
                column_types[column] = 'date'
            # 检测数值列
            elif pd.api.types.is_numeric_dtype(df[column]):
                column_types[column] = 'numeric'
            # 检测文本列
            else:
                column_types[column] = 'text'
        
        return column_types


class DataValidator:
    """数据验证器，负责验证数据的质量和一致性"""
    
    def __init__(self):
        """初始化数据验证器"""
        logger.info("DataValidator初始化完成")
    
    @staticmethod
    def validate_price_data(price_data):
        """验证价格数据有效性"""
        issues = []
        
        # 检查是否为0
        if price_data.get('start_price', 0) == 0 and price_data.get('end_price', 0) == 0:
            issues.append("所有价格数据均为0，请检查数据源")
        
        # 检查价格合理性（根据业务逻辑）
        if price_data.get('low_price', 0) < 0:
            issues.append("价格不能为负数")
        
        # 检查数据完整性
        required_fields = ['start_price', 'end_price', 'high_price', 'low_price']
        missing_fields = [field for field in required_fields if field not in price_data]
        
        if issues or missing_fields:
            return {
                'is_valid': False,
                'issues': issues,
                'missing_fields': missing_fields,
                'suggestion': '请检查数据源连接或API配置'
            }
        
        return {'is_valid': True}
    
    def validate_dataframe(self, df: pd.DataFrame, validation_rules: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证数据框是否符合验证规则
        
        Args:
            df: 要验证的数据框
            validation_rules: 验证规则
            
        Returns:
            验证结果，包含is_valid、errors、missing_fields和suggestion字段
        """
        result = {
            'is_valid': True,
            'errors': [],
            'missing_fields': [],
            'warning': [],
            'suggestion': '数据验证通过'
        }
        
        # 检查数据框是否为空
        if df.empty:
            result['is_valid'] = False
            result['errors'].append("数据框为空，没有数据可验证")
            result['suggestion'] = '请检查数据源，确保数据存在'
            return result
        
        # 检查必填列
        if 'required_columns' in validation_rules:
            required_columns = validation_rules['required_columns']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                result['is_valid'] = False
                result['missing_fields'] = missing_columns
                result['errors'].append(f"缺少必填列: {missing_columns}")
        
        # 检查数据类型
        if 'column_types' in validation_rules:
            for col, expected_type in validation_rules['column_types'].items():
                if col in df.columns:
                    actual_type = str(df[col].dtype)
                    if not self._check_column_type(actual_type, expected_type):
                        result['warning'].append(f"列 {col} 的数据类型不符合预期: {actual_type} (预期: {expected_type})")
        
        # 检查数值范围
        if 'numeric_columns' in validation_rules:
            for col, rules in validation_rules['numeric_columns'].items():
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    # 检查是否全为0
                    if (df[col] == 0).all():
                        result['is_valid'] = False
                        result['errors'].append(f"列 {col} 的所有值均为0，请检查数据源")
                    
                    if 'min_value' in rules:
                        min_val = rules['min_value']
                        if df[col].min() < min_val:
                            result['is_valid'] = False
                            result['errors'].append(f"列 {col} 的值小于最小值 {min_val}")
                    
                    if 'max_value' in rules:
                        max_val = rules['max_value']
                        if df[col].max() > max_val:
                            result['is_valid'] = False
                            result['errors'].append(f"列 {col} 的值大于最大值 {max_val}")
                    
                    if 'not_null' in rules and rules['not_null']:
                        if df[col].isna().any():
                            result['is_valid'] = False
                            result['errors'].append(f"列 {col} 包含空值")
        
        # 检查数据完整性
        if 'completeness' in validation_rules:
            min_completeness = validation_rules['completeness'].get('min_completeness', 0.8)
            for column in df.columns:
                completeness = df[column].notna().sum() / len(df)
                if completeness < min_completeness:
                    result['is_valid'] = False
                    result['errors'].append(f"列 {column} 的数据完整性不足 ({completeness:.2%} < {min_completeness:.2%})")
        
        # 检查异常值
        if 'outliers' in validation_rules and validation_rules['outliers']:
            numeric_columns = df.select_dtypes(include=['number']).columns
            for col in numeric_columns:
                # 使用IQR方法检查异常值
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                if not outliers.empty:
                    result['is_valid'] = False
                    result['errors'].append(f"列 {col} 包含 {len(outliers)} 个异常值，请检查数据质量")
        
        # 检查时间序列数据
        if 'time_series' in validation_rules:
            date_column = validation_rules['time_series'].get('date_column')
            if date_column and date_column in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
                    result['is_valid'] = False
                    result['errors'].append(f"时间序列的日期列 {date_column} 不是日期类型")
                else:
                    # 检查日期连续性
                    if validation_rules['time_series'].get('check_continuity', False):
                        df_sorted = df.sort_values(by=date_column)
                        date_diff = df_sorted[date_column].diff().dt.days.dropna()
                        if date_diff.max() > 1:
                            result['warning'].append(f"日期列 {date_column} 存在不连续的日期")
        
        # 检查业务规则
        if 'business_rules' in validation_rules:
            for rule in validation_rules['business_rules']:
                if 'expression' in rule:
                    try:
                        violation_count = len(df[~df.eval(rule['expression'])])
                        if violation_count > 0:
                            result['is_valid'] = False
                            result['errors'].append(f"业务规则违反: {rule['description']} (违反次数: {violation_count})")
                    except Exception as e:
                        logger.error(f"执行业务规则失败: {rule['expression']}, 错误: {e}")
        
        # 检查重复值
        if 'duplicates' in validation_rules and validation_rules['duplicates']:
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                result['warning'].append(f"数据框包含 {duplicate_count} 个重复行")
        
        # 如果验证失败，添加建议
        if not result['is_valid']:
            result['suggestion'] = '请检查数据源连接或API配置，确保数据质量'
        
        return result
    
    def _check_column_type(self, actual_type: str, expected_type: str) -> bool:
        """
        检查列的数据类型是否符合预期
        
        Args:
            actual_type: 实际数据类型
            expected_type: 预期数据类型
            
        Returns:
            如果类型符合预期返回True，否则返回False
        """
        type_mapping = {
            'int': ['int', 'int64', 'Int64'],
            'float': ['float', 'float64'],
            'string': ['object', 'string'],
            'date': ['datetime64', 'datetime64[ns]'],
            'datetime': ['datetime64', 'datetime64[ns]'],
            'bool': ['bool', 'boolean']
        }
        
        return actual_type in type_mapping.get(expected_type.lower(), [expected_type])
    



class DataExtractor:
    """数据提取器，负责从文件中提取和验证数据"""
    
    def __init__(self, config_manager=None):
        """
        初始化数据提取器
        
        Args:
            config_manager: 配置管理器
        """
        self.parser = FileParser()
        self.validator = DataValidator()
        self.cleaner = DataCleaner()
        self.enhancer = DataEnhancer()
        self.config_manager = config_manager
    
    def extract_data(self, files: List[str] = None, report_type: str = None) -> Dict[str, Any]:
        """
        从文件中提取数据
        
        Args:
            files: 文件路径列表
            report_type: 报告类型，用于加载相应的配置
            
        Returns:
            提取后的数据结构
        """
        extracted_data = {
            'price_data': None,
            'events': [],
            'market_sentiment': {},
            'statistical_analysis': {}
        }
        
        # 加载报告配置
        report_config = None
        if self.config_manager and report_type:
            report_config = self.config_manager.get_report_config(report_type)
        
        # 如果没有提供文件或文件列表为空，使用默认数据
        if not files:
            extracted_data['price_data'] = self._get_default_price_data()
            return extracted_data
        
        # 解析所有文件
        for file_path in files:
            if not self.parser.can_parse(file_path):
                logger.warning(f"跳过不支持的文件: {file_path}")
                continue
            
            try:
                df = self.parser.parse(file_path)
                
                # 检测列类型
                column_types = self.parser.detect_columns(df)
                
                # 验证数据
                if report_config and 'data_fields' in report_config and 'price_data' in report_config['data_fields']:
                    validation_rules = report_config['data_fields']['price_data'].get('validation', {})
                    validation_result = self.validator.validate_dataframe(df, validation_rules)
                    
                    if not validation_result['is_valid']:
                        logger.warning(f"数据验证失败: {validation_result['errors']}")
                    else:
                        # 清洗数据
                        cleaning_rules = report_config['data_fields']['price_data'].get('cleaning', {})
                        df = self.cleaner.clean_dataframe(df, cleaning_rules)
                        
                        # 增强数据
                        enhancement_rules = report_config['data_fields']['price_data'].get('enhancement', {})
                        df = self.enhancer.enhance_dataframe(df, enhancement_rules)
                        
                        # 映射字段
                        extracted_data['price_data'] = self._map_fields(df, column_types, report_config)
            except Exception as e:
                logger.error(f"处理文件 {file_path} 失败: {e}")
        
        # 如果没有提取到数据，使用默认数据
        if extracted_data['price_data'] is None:
            extracted_data['price_data'] = self._get_default_price_data()
        else:
            # 验证价格数据
            validation_result = DataValidator.validate_price_data(extracted_data['price_data'])
            if not validation_result['is_valid']:
                logger.warning(f"价格数据验证失败: {validation_result}")
                # 使用默认数据替代
                extracted_data['price_data'] = self._get_default_price_data()
                logger.info("使用默认数据替代验证失败的数据")
        
        return extracted_data
    
    def _map_fields(self, df: pd.DataFrame, column_types: Dict[str, str], report_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        映射字段到数据结构
        
        Args:
            df: 数据框
            column_types: 列类型映射
            report_config: 报告配置
            
        Returns:
            映射后的数据结构
        """
        price_data = {
            'price_changes': []
        }
        
        # 获取字段映射
        mapping = {}
        if report_config and 'data_fields' in report_config and 'price_data' in report_config['data_fields']:
            mapping = report_config['data_fields']['price_data'].get('mapping', {})
        
        # 检测日期列和价格列
        date_column = mapping.get('date_column')
        price_column = mapping.get('price_column')
        
        # 如果没有指定映射，自动检测
        if not date_column:
            date_columns = [col for col, dtype in column_types.items() if dtype == 'date']
            if date_columns:
                date_column = date_columns[0]
        
        if not price_column:
            price_columns = [col for col, dtype in column_types.items() if dtype == 'numeric']
            if price_columns:
                price_column = price_columns[0]
        
        if not date_column or not price_column:
            logger.warning("无法自动检测日期列和价格列")
            return price_data
        
        # 构建价格变化数据
        for _, row in df.iterrows():
            if pd.notna(row[date_column]) and pd.notna(row[price_column]):
                price_change = {
                    'date': row[date_column].strftime('%Y-%m-%d') if hasattr(row[date_column], 'strftime') else str(row[date_column]),
                    'price': float(row[price_column]),
                    'change': 0.0,  # 默认值，后续会计算
                    'event': ''
                }
                price_data['price_changes'].append(price_change)
        
        # 计算涨跌幅
        if len(price_data['price_changes']) > 1:
            for i in range(1, len(price_data['price_changes'])):
                prev_price = price_data['price_changes'][i-1]['price']
                curr_price = price_data['price_changes'][i]['price']
                change = ((curr_price - prev_price) / prev_price) * 100
                price_data['price_changes'][i]['change'] = round(change, 2)
        
        # 设置其他价格信息
        if price_data['price_changes']:
            price_data['start_price'] = price_data['price_changes'][0]['price']
            price_data['end_price'] = price_data['price_changes'][-1]['price']
            price_data['high_price'] = max(pc['price'] for pc in price_data['price_changes'])
            price_data['low_price'] = min(pc['price'] for pc in price_data['price_changes'])
            price_data['unit'] = mapping.get('unit', '')
        
        return price_data
    
    def _get_default_price_data(self) -> Dict[str, Any]:
        """
        获取默认价格数据
        
        Returns:
            默认价格数据
        """
        return {
            'start_price': 1000,
            'end_price': 1050,
            'high_price': 1080,
            'low_price': 980,
            'unit': '美元/盎司',
            'price_changes': [
                {'date': '2026-01-01', 'price': 1000, 'change': 0.0, 'event': '开始日期'},
                {'date': '2026-01-02', 'price': 1020, 'change': 2.0, 'event': '上涨'},
                {'date': '2026-01-03', 'price': 1080, 'change': 6.0, 'event': '创新高'},
                {'date': '2026-01-04', 'price': 1030, 'change': -4.6, 'event': '回调'},
                {'date': '2026-01-05', 'price': 1050, 'change': 2.0, 'event': '结束日期'}
            ]
        }
