import matplotlib.pyplot as plt
import numpy as np
# 数据集准备
topic = [4, 6, 8, 10, 12, 14]
map2019 = [0.5130, 0.5217, 0.5378, 0.5306, 0.5182, 0.5099]
map2020 = [0.5057, 0.5089, 0.5258, 0.5247, 0.5095, 0.5053]
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
plt.xlabel('k', fontfamily='Times New Roman')
plt.ylabel('MAP', fontfamily='Times New Roman')

# 自定义x轴刻度（匹配topic数据）
plt.xticks(topic)
plt.ylim(0.49, 0.54)
plt.xlim(3.3, 15)  # x轴范围匹配topic的最大值

# 添加图例
# plt.legend(loc='upper right', frameon=True, fontsize=10)
# 添加图例（放在右下角）
plt.legend(loc='lower right', frameon=True, fontsize=10)

# 替换 plt.show() 为：
plt.savefig('k-map.png')  # 保存图片
plt.close()  # 关闭图形

# 或者如果您想在 PyCharm 中显示：
plt.show(block=True)