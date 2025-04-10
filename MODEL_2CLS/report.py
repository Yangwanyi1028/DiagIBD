import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

REPORT_PATH = '/Users/wanyiddl/Downloads/AImodel/model_performance_report_20250401.csv'
DATA_DIR = "/Users/wanyiddl/Downloads/AImodel/"

# 读取原始数据（不进行分组平均）
report_df = pd.read_csv(REPORT_PATH)
report_df['Feature_Count'] = report_df['Dataset'].str.extract('(\d+)feat').astype(int)

# 设置全局字体为英文
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.titleweight'] = 'bold'

# ========== 三指标并列图 ==========
# 原三图并列代码基础上增加y轴限制
plt.figure(figsize=(18, 5))

# Accuracy subplot
plt.subplot(1, 3, 1)
ax1 = sns.barplot(x='Feature_Count', y='ACC', hue='Model', 
                data=report_df, ci=None, palette='viridis')
ax1.set(ylim=(0.6, 0.9))  # 新增y轴限制
plt.title('Feature Count vs Accuracy', y=1.02, fontsize=12)
plt.xlabel('Feature Count', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)

# Sensitivity subplot
plt.subplot(1, 3, 2)
ax2 = sns.barplot(x='Feature_Count', y='SEN', hue='Model',
                data=report_df, ci=None, palette='rocket')
ax2.set(ylim=(0.6, 0.9))  # 新增y轴限制
plt.title('Feature Count vs Sensitivity', y=1.02, fontsize=12)
plt.xlabel('Feature Count', fontsize=10)
plt.ylabel('Sensitivity', fontsize=10)

# Specificity subplot
plt.subplot(1, 3, 3)
ax3 = sns.barplot(x='Feature_Count', y='SPE', hue='Model',
                data=report_df, ci=None, palette='mako')
ax3.set(ylim=(0.6, 0.9))  # 新增y轴限制
plt.title('Feature Count vs Specificity', y=1.02, fontsize=12)
plt.xlabel('Feature Count', fontsize=10)
plt.ylabel('Specificity', fontsize=10)

plt.tight_layout()
plt.savefig(f'{DATA_DIR}feature_performance_triple_en_scaled.png', dpi=300, bbox_inches='tight')

# ========== 散点矩阵图 ==========
plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(x='SEN', y='SPE', 
                         size='ACC', 
                         hue='Feature_Count',
                         sizes=(50, 500),
                         alpha=0.7,
                         palette='coolwarm',
                         data=report_df)
plt.title('Sensitivity-Specificity Relationship', fontsize=12)
plt.xlabel('Sensitivity', fontsize=10)
plt.ylabel('Specificity', fontsize=10)
plt.legend(bbox_to_anchor=(1.05, 1), title='Feature Count')
plt.savefig(f'{DATA_DIR}scatter_matrix_en.png', dpi=300, bbox_inches='tight')

# ========== 箱线图 ==========
plt.figure(figsize=(12, 6))
sns.boxplot(x='Feature_Count', y='ACC', 
           hue='Model', 
           data=report_df,
           palette='Set2',
           linewidth=1)
plt.title('Accuracy Distribution by Feature Count', fontsize=12)
plt.xlabel('Feature Count', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.savefig(f'{DATA_DIR}boxplot_en.png', dpi=300, bbox_inches='tight')

# ========== 交互式图表 ==========
import plotly.express as px
fig = px.scatter(report_df,
                x="Feature_Count",
                y="ACC",
                color="Model",
                size="SEN",
                hover_data=['SPE'],
                title="Interactive Performance Analysis",
                labels={
                    "Feature_Count": "Feature Count",
                    "ACC": "Accuracy",
                    "SPE": "Specificity",
                    "SEN": "Sensitivity"
                })
fig.write_html(f"{DATA_DIR}interactive_chart_en.html")

# ========== 折线趋势图 ==========
plt.figure(figsize=(12, 6))
sns.lineplot(x='Feature_Count', y='ACC', 
            hue='Model',
            style='Model',
            markers=True,
            dashes=False,
            data=report_df,
            ci='sd',
            palette='Dark2')
plt.title('Accuracy Trend by Feature Count', fontsize=12)
plt.xlabel('Feature Count', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig(f'{DATA_DIR}trend_line_en.png', dpi=300, bbox_inches='tight')
