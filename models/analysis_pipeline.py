#!/usr/bin/env python3
"""
分析流水线
负责对数据进行统计分析、相关性计算和洞察生成
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """统计分析器，负责对数据进行统计分析"""
    
    def __init__(self):
        """初始化统计分析器"""
        pass
    
    def analyze(self, price_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        对价格数据进行统计分析
        
        Args:
            price_data: 价格数据
            
        Returns:
            统计分析结果
        """
        analysis_results = {
            'basic_statistics': {},
            'volatility': {},
            'trend': {},
            'seasonality': {}
        }
        
        if 'price_changes' not in price_data or not price_data['price_changes']:
            return analysis_results
        
        # 提取价格和日期数据
        df = self._convert_to_dataframe(price_data['price_changes'])
        if df is None or df.empty:
            return analysis_results
        
        # 基本统计分析
        analysis_results['basic_statistics'] = self._calculate_basic_statistics(df)
        
        # 波动率分析
        analysis_results['volatility'] = self._calculate_volatility(df)
        
        # 趋势分析
        analysis_results['trend'] = self._analyze_trend(df)
        
        return analysis_results
    
    def _convert_to_dataframe(self, price_changes: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
        """
        将价格变动数据转换为DataFrame
        
        Args:
            price_changes: 价格变动数据
            
        Returns:
            转换后的DataFrame
        """
        try:
            # 提取日期和价格
            dates = []
            prices = []
            changes = []
            
            for item in price_changes:
                if 'date' in item and 'price' in item:
                    dates.append(item['date'])
                    prices.append(item['price'])
                    changes.append(item.get('change', 0.0))
            
            if not dates or not prices:
                return None
            
            df = pd.DataFrame({
                'date': pd.to_datetime(dates),
                'price': prices,
                'change': changes
            })
            
            df = df.sort_values('date')
            df = df.set_index('date')
            
            return df
        except Exception as e:
            logger.error(f"转换数据为DataFrame失败: {e}")
            return None
    
    def _calculate_basic_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        计算基本统计指标
        
        Args:
            df: 价格数据DataFrame
            
        Returns:
            基本统计指标
        """
        stats = {
            'mean': float(df['price'].mean()),
            'median': float(df['price'].median()),
            'std': float(df['price'].std()),
            'min': float(df['price'].min()),
            'max': float(df['price'].max()),
            'range': float(df['price'].max() - df['price'].min()),
            'count': len(df)
        }
        
        return stats
    
    def _calculate_volatility(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        计算波动率指标
        
        Args:
            df: 价格数据DataFrame
            
        Returns:
            波动率指标
        """
        volatility = {
            'daily_volatility': float(df['change'].std() if 'change' in df.columns else 0.0),
            'annualized_volatility': float(df['change'].std() * np.sqrt(252) if 'change' in df.columns else 0.0),
            'monthly_volatility': {}
        }
        
        # 按月计算波动率
        try:
            monthly_data = df['price'].resample('ME').last()
            monthly_changes = monthly_data.pct_change() * 100
            volatility['monthly_volatility'] = float(monthly_changes.std() if len(monthly_changes) > 1 else 0.0)
        except Exception as e:
            logger.warning(f"计算月度波动率失败: {e}")
        
        return volatility
    
    def _analyze_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        分析价格趋势
        
        Args:
            df: 价格数据DataFrame
            
        Returns:
            趋势分析结果
        """
        trend = {
            'direction': 'stable',
            'slope': 0.0,
            'duration': len(df),
            'percentage_change': 0.0
        }
        
        if len(df) < 2:
            return trend
        
        # 计算价格变化百分比
        percentage_change = (df['price'].iloc[-1] - df['price'].iloc[0]) / df['price'].iloc[0] * 100
        trend['percentage_change'] = float(percentage_change)
        
        # 确定趋势方向
        if percentage_change > 5:
            trend['direction'] = 'upward'
        elif percentage_change < -5:
            trend['direction'] = 'downward'
        
        # 计算趋势斜率
        try:
            x = np.arange(len(df))
            y = df['price'].values
            slope, _ = np.polyfit(x, y, 1)
            trend['slope'] = float(slope)
        except Exception as e:
            logger.warning(f"计算趋势斜率失败: {e}")
        
        return trend


class CorrelationCalculator:
    """相关性计算器，负责计算不同变量之间的相关性"""
    
    def __init__(self):
        """初始化相关性计算器"""
        pass
    
    def calculate_correlations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算数据之间的相关性
        
        Args:
            data: 包含价格数据和其他相关数据
            
        Returns:
            相关性计算结果
        """
        correlations = {
            'internal_correlations': {},
            'external_correlations': {}
        }
        
        # 计算内部相关性
        if 'price_data' in data and 'price_changes' in data['price_data']:
            df = self._convert_price_data_to_dataframe(data['price_data']['price_changes'])
            if df is not None and len(df) > 1:
                correlations['internal_correlations'] = self._calculate_internal_correlations(df)
        
        return correlations
    
    def _convert_price_data_to_dataframe(self, price_changes: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
        """
        将价格数据转换为DataFrame
        
        Args:
            price_changes: 价格变动数据
            
        Returns:
            转换后的DataFrame
        """
        try:
            dates = []
            prices = []
            changes = []
            
            for item in price_changes:
                if 'date' in item and 'price' in item:
                    dates.append(item['date'])
                    prices.append(item['price'])
                    changes.append(item.get('change', 0.0))
            
            if not dates or not prices:
                return None
            
            df = pd.DataFrame({
                'date': pd.to_datetime(dates),
                'price': prices,
                'change': changes
            })
            
            df = df.sort_values('date')
            return df
        except Exception as e:
            logger.error(f"转换价格数据为DataFrame失败: {e}")
            return None
    
    def _calculate_internal_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        计算内部相关性
        
        Args:
            df: 价格数据DataFrame
            
        Returns:
            内部相关性计算结果
        """
        internal_correlations = {}
        
        try:
            # 计算价格与涨跌幅的相关性
            if 'change' in df.columns:
                price_change_corr = df['price'].corr(df['change'])
                internal_correlations['price_change_correlation'] = float(price_change_corr)
            
            # 计算价格自相关性
            price_autocorr = df['price'].autocorr(lag=1)
            internal_correlations['price_autocorrelation'] = float(price_autocorr)
            
            # 计算涨跌幅自相关性
            if 'change' in df.columns:
                change_autocorr = df['change'].autocorr(lag=1)
                internal_correlations['change_autocorrelation'] = float(change_autocorr)
        except Exception as e:
            logger.warning(f"计算内部相关性失败: {e}")
        
        return internal_correlations


class InsightGenerator:
    """洞察生成器，负责基于统计分析和相关性生成洞察"""
    
    def __init__(self):
        """初始化洞察生成器"""
        pass
    
    def generate_insights(self, statistical_analysis: Dict[str, Any], correlations: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        基于统计分析和相关性生成洞察
        
        Args:
            statistical_analysis: 统计分析结果
            correlations: 相关性计算结果
            
        Returns:
            生成的洞察列表
        """
        insights = []
        
        # 基于基本统计数据生成洞察
        if 'basic_statistics' in statistical_analysis:
            insights.extend(self._generate_statistics_insights(statistical_analysis['basic_statistics']))
        
        # 基于波动率生成洞察
        if 'volatility' in statistical_analysis:
            insights.extend(self._generate_volatility_insights(statistical_analysis['volatility']))
        
        # 基于趋势生成洞察
        if 'trend' in statistical_analysis:
            insights.extend(self._generate_trend_insights(statistical_analysis['trend']))
        
        # 基于相关性生成洞察
        if 'internal_correlations' in correlations and correlations['internal_correlations']:
            insights.extend(self._generate_correlation_insights(correlations['internal_correlations']))
        
        return insights
    
    def _generate_statistics_insights(self, basic_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        基于基本统计数据生成洞察
        
        Args:
            basic_stats: 基本统计数据
            
        Returns:
            生成的洞察列表
        """
        insights = []
        
        # 检查价格范围
        if 'range' in basic_stats and 'mean' in basic_stats:
            price_range = basic_stats['range']
            mean_price = basic_stats['mean']
            
            if price_range > mean_price * 0.5:
                insights.append({
                    'type': 'price_range',
                    'title': '价格波动范围较大',
                    'description': f"数据显示价格波动范围（{price_range:.2f}）超过平均价格（{mean_price:.2f}）的50%，表明市场波动性较高。",
                    'importance': 4
                })
        
        return insights
    
    def _generate_volatility_insights(self, volatility: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        基于波动率生成洞察
        
        Args:
            volatility: 波动率数据
            
        Returns:
            生成的洞察列表
        """
        insights = []
        
        # 检查年化波动率
        if 'annualized_volatility' in volatility:
            annualized_vol = volatility['annualized_volatility']
            
            if annualized_vol > 20:
                insights.append({
                    'type': 'high_volatility',
                    'title': '年化波动率较高',
                    'description': f"年化波动率（{annualized_vol:.2f}%）超过20%，表明资产价格波动较大，风险较高。",
                    'importance': 5
                })
            elif annualized_vol < 10:
                insights.append({
                    'type': 'low_volatility',
                    'title': '年化波动率较低',
                    'description': f"年化波动率（{annualized_vol:.2f}%）低于10%，表明资产价格相对稳定，风险较低。",
                    'importance': 3
                })
        
        return insights
    
    def _generate_trend_insights(self, trend: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        基于趋势生成洞察
        
        Args:
            trend: 趋势数据
            
        Returns:
            生成的洞察列表
        """
        insights = []
        
        # 分析趋势方向
        if 'direction' in trend and trend['direction'] != 'stable':
            direction_text = '上涨' if trend['direction'] == 'upward' else '下跌'
            percentage_change = trend.get('percentage_change', 0.0)
            
            insights.append({
                'type': 'trend',
                'title': f'价格呈现明显{direction_text}趋势',
                'description': f"数据显示价格呈现明显的{direction_text}趋势，变动幅度为{percentage_change:.2f}%，表明市场有明确的方向性。",
                'importance': 5
            })
        
        return insights
    
    def _generate_correlation_insights(self, correlations: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        基于相关性生成洞察
        
        Args:
            correlations: 相关性数据
            
        Returns:
            生成的洞察列表
        """
        insights = []
        
        # 分析价格与涨跌幅的相关性
        if 'price_change_correlation' in correlations:
            corr_value = correlations['price_change_correlation']
            
            if abs(corr_value) > 0.5:
                direction = '正相关' if corr_value > 0 else '负相关'
                insights.append({
                    'type': 'price_change_correlation',
                    'title': f'价格与涨跌幅呈现强{direction}关系',
                    'description': f"价格与涨跌幅的相关性为{corr_value:.2f}，呈现强{direction}关系，表明价格变动幅度与当前价格水平有显著关联。",
                    'importance': 4
                })
        
        return insights
