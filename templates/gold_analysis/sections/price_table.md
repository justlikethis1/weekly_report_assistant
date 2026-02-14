# 价格概览

## 价格变动表格
| 日期 | 价格 ({{ price_data.unit | default('美元/盎司') }}) | 涨跌幅 (%) |
|------|-----------------------------------|----------|
{% for change in price_data.price_changes | default([]) %}
| {{ change.date }} | {{ change.price }} | {% if change.change > 0 %}+{{ change.change }}{% else %}{{ change.change }}{% endif %} |
{% endfor %}

## 价格统计
- **起始价格**: {{ price_data.start_price | default(0) }}{{ price_data.unit | default('美元/盎司') }}
- **结束价格**: {{ price_data.end_price | default(0) }}{{ price_data.unit | default('美元/盎司') }}
- **最高价**: {{ price_data.high_price | default(0) }}{{ price_data.unit | default('美元/盎司') }}
- **最低价**: {{ price_data.low_price | default(0) }}{{ price_data.unit | default('美元/盎司') }}
- **平均价格**: {{ statistical_analysis.basic_statistics.mean | default(0) | round(2) }}{{ price_data.unit | default('美元/盎司') }}
- **价格范围**: {{ statistical_analysis.basic_statistics.range | default(0) | round(2) }}{{ price_data.unit | default('美元/盎司') }}

## 价格趋势图
```
价格趋势图
(此处可插入实际的价格趋势图表)
```

## 波动率分析
- **日波动率**: {{ statistical_analysis.volatility.daily_volatility | default(0) | round(2) }}%
- **年化波动率**: {{ statistical_analysis.volatility.annualized_volatility | default(0) | round(2) }}%
- **月波动率**: {{ statistical_analysis.volatility.monthly_volatility | default(0) | round(2) }}%
