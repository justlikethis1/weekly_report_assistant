# 执行摘要

✅ **数据状态**：{% if data_quality.score >= 7 %}优秀{% elif data_quality.score >= 5 %}良好{% else %}一般{% endif %}（得分：{{ data_quality.score | default(0) }}/10）

📊 **核心发现**：
- 黄金价格从{{ price_data.start_price | default(0) }}{{ price_data.unit | default('美元/盎司') }}变动到{{ price_data.end_price | default(0) }}{{ price_data.unit | default('美元/盎司') }}，变动幅度为{{ statistical_analysis.trend.percentage_change | default(0) | round(2) }}%
- 期间最高价为{{ price_data.high_price | default(0) }}{{ price_data.unit | default('美元/盎司') }}，最低价为{{ price_data.low_price | default(0) }}{{ price_data.unit | default('美元/盎司') }}
- 平均价格为{{ statistical_analysis.basic_statistics.mean | default(0) | round(2) }}{{ price_data.unit | default('美元/盎司') }}

🔄 **主要趋势**：{% if statistical_analysis.trend.direction == 'upward' %}上涨{% elif statistical_analysis.trend.direction == 'downward' %}下跌{% else %}稳定{% endif %}趋势

⚠️ **风险提示**：{% if statistical_analysis.volatility.annualized_volatility | default(0) > 20 %}市场波动率较高{% else %}市场波动率相对稳定{% endif %}（年化波动率：{{ statistical_analysis.volatility.annualized_volatility | default(0) | round(2) }}%）