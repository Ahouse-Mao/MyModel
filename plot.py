import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

# 创建画布和坐标轴
fig, ax = plt.subplots(figsize=(10, 6))

# 设置网格参数
rows, cols = 6, 10
cell_size = 1

# 定义基础红色
base_color = (1.0, 0.0, 0.0)  # 正红色

# 生成10种更明亮的红色变体
def generate_lighter_reds():
    lighter_colors = []
    sats = np.linspace(0.2, 0.4, 5)   # 降低饱和度（原0.3-0.6 → 0.2-0.4）
    vals = np.linspace(0.7, 1.0, 2)   # 提高明度（原0.5-1.0 → 0.7-1.0）
    for sat in sats:
        for val in vals:
            hsv = rgb_to_hsv(np.array(base_color).reshape(1, 1, 3)).squeeze()
            hsv[1] = sat  # 设置新饱和度
            hsv[2] = val  # 设置新明度
            lighter_colors.append(tuple(hsv_to_rgb(hsv)))
    return lighter_colors

# 使用新调色板
color_palette = generate_lighter_reds()  # 生成10种更亮的红色

# 绘制单元格
for i in range(rows):
    for j in range(cols):
        rect = patches.Rectangle(
            (j*cell_size, i*cell_size),
            cell_size, cell_size,
            facecolor=random.choice(color_palette),  # 从新调色板中选择
            edgecolor='none'
        )
        ax.add_patch(rect)

# 添加灰色外边框
border = patches.Rectangle(
    (0, 0), cols, rows,
    linewidth=2,
    edgecolor='gray',
    facecolor='none'
)
ax.add_patch(border)

# 设置坐标轴范围和比例
ax.set_xlim(0, cols)
ax.set_ylim(0, rows)
ax.set_aspect('equal')
ax.axis('off')  # 隐藏坐标轴

plt.tight_layout()
plt.show()
plt.savefig('grid_plot.png')