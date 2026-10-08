import matplotlib.pyplot as plt
import numpy as np
# 数据集准备
topic = [3, 4, 5, 6, 7, 8]
map2019 = [0.7281, 0.7332, 0.7452, 0.7403, 0.7253, 0.7196]
map2020 = [0.7068, 0.7177, 0.7259, 0.7183, 0.7151, 0.6934]
# map2020top10 = [0, 0, 0.11, 0.81, 1, 1]
# label = "CDF of per-query NDCG@10 gains"

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12  # 可选：统一字体大小

# 调整图形尺寸（按需修改宽高，这里示例为更紧凑的尺寸）
plt.figure(figsize=(8, 6))

# 绘制折线
plt.plot(topic, map2019, 'y-^', label='TREC 2019')
plt.plot(topic, map2020, 'r-o', label='TREC 2020')
# plt.plot(topic, map2020top10, 'c-*', label='Weight+Adaptive-K(ours)')
# 设置标题、坐标轴标签
plt.title(
    ' ',
    fontweight='bold',
    loc='center',
    fontfamily='Times New Roman',
    fontsize=12,
    y=1.03  # 调整y值，大于1时标题上移，可根据需要微调（如1.03、1.06等）
)
plt.xlabel('Top-P', fontfamily='Times New Roman')
plt.ylabel('NDCG@10', fontfamily='Times New Roman')

# 自定义x轴刻度（匹配topic数据）
plt.xticks(topic)
plt.ylim(0.69, 0.75)
plt.xlim(2.5, 8.5)  # x轴范围匹配topic的最大值

# 添加图例
plt.legend(loc='upper right', frameon=True, fontsize=12)
# 添加图例（放在右下角）
# plt.legend(loc='lower right', frameon=True, fontsize=12)

# 替换 plt.show() 为：
plt.savefig('P-NDCG.png')  # 保存图片
plt.close()  # 关闭图形

# 或者如果您想在 PyCharm 中显示：
plt.show(block=True)