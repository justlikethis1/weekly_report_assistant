# 相关性分析

## 内部相关性

### 价格与涨跌幅相关性
{% if correlations.internal_correlations.price_change_correlation is defined %}
- 相关系数: {{ correlations.internal_correlations.price_change_correlation | round(2) }}
  {% if correlations.internal_correlations.price_change_correlation | abs > 0.7 %}
  → **强相关**
  {% elif correlations.internal_correlations.price_change_correlation | abs > 0.3 %}
  → **中等相关**
  {% else %}
  → **弱相关**
  {% endif %}
{% endif %}

### 价格自相关性
{% if correlations.internal_correlations.price_autocorrelation is defined %}
- 相关系数: {{ correlations.internal_correlations.price_autocorrelation | round(2) }}
  {% if correlations.internal_correlations.price_autocorrelation | abs > 0.7 %}
  → **强自相关**
  {% elif correlations.internal_correlations.price_autocorrelation | abs > 0.3 %}
  → **中等自相关**
  {% else %}
  → **弱自相关**
  {% endif %}
{% endif %}

### 涨跌幅自相关性
{% if correlations.internal_correlations.change_autocorrelation is defined %}
- 相关系数: {{ correlations.internal_correlations.change_autocorrelation | round(2) }}
  {% if correlations.internal_correlations.change_autocorrelation | abs > 0.7 %}
  → **强自相关**
  {% elif correlations.internal_correlations.change_autocorrelation | abs > 0.3 %}
  → **中等自相关**
  {% else %}
  → **弱自相关**
  {% endif %}
{% endif %}

## 外部相关性

### 美元指数相关性
(待实现：与美元指数的相关性分析)

### 原油价格相关性
(待实现：与原油价格的相关性分析)

### 股票市场相关性
(待实现：与股票市场的相关性分析)

## 相关性矩阵

```
相关性矩阵
(此处可插入实际的相关性矩阵图表)
```

## 分析结论
{% if correlations.internal_correlations.price_change_correlation | default(0) | abs > 0.5 %}
- 价格与涨跌幅呈现强相关关系，表明价格变动幅度与当前价格水平有显著关联
{% endif %}

{% if correlations.internal_correlations.price_autocorrelation | default(0) | abs > 0.5 %}
- 价格呈现较强的自相关性，表明历史价格对当前价格有较大影响
{% endif %}
