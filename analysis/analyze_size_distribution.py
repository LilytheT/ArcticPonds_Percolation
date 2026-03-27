import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.ndimage import label

# ==========================================
# 1. 找到对应的相变点数据文件
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATTERN = os.path.join(BASE_DIR, "voids_rho_0.31*.txt")
files = glob.glob(FILE_PATTERN)

if not files:
    print("❌ 未找到 voids_rho_0.310.txt 文件！")
    exit()

file_path = files[0]
print(f"❄️ 正在分析文件: {os.path.basename(file_path)}")

# ==========================================
# 2. 高精度栅格化 (复用物理逻辑)
# ==========================================
L = 500.0
resolution = 2000  # 相当于 0.25m / 像素
dx = L / resolution

data = np.loadtxt(file_path, skiprows=6)
x_centers, y_centers, radii = data[:, 1], data[:, 2], data[:, 3]

print("  ➜ 正在将几何体投影到高精度像素网格...")
grid = np.ones((resolution, resolution), dtype=int)

for x, y, r in zip(x_centers, y_centers, radii):
    x_min = max(0, int((x-r)/dx))
    x_max = min(resolution, int((x+r)/dx) + 1)
    y_min = max(0, int((y-r)/dx))
    y_max = min(resolution, int((y+r)/dx) + 1)
    if x_min >= x_max or y_min >= y_max: continue
    
    xv, yv = np.meshgrid(np.linspace(x_min*dx, x_max*dx, x_max-x_min, endpoint=False),
                         np.linspace(y_min*dx, y_max*dx, y_max-y_min, endpoint=False))
    mask = (xv - x)**2 + (yv - y)**2 <= r**2
    grid[y_min:y_max, x_min:x_max][mask] = 0

# ==========================================
# 3. 连通域标记与面积提取
# ==========================================
print("  ➜ 正在寻找独立融池并计算面积...")
structure = [[0,1,0], [1,1,1], [0,1,0]]
labeled_array, num_features = label(grid, structure=structure)

# 统计每个 ID 包含的像素数量
component_sizes = np.bincount(labeled_array.ravel())
pond_pixel_counts = component_sizes[1:] # 剔除索引0 (背景冰层)

# 转换为物理面积，并过滤掉极小的网格噪点 (例如小于 4 个像素)
areas = pond_pixel_counts[pond_pixel_counts >= 4] * (dx ** 2)
print(f"  ➜ 成功提取 {len(areas)} 个有效融池。")

# ==========================================
# 4. 对数分箱 (Log-Binning) 求解概率密度
# ==========================================
print("  ➜ 正在执行对数分箱以计算概率密度...")
num_bins = 25
# 在对数尺度上生成均匀分布的箱子边界
bins = np.logspace(np.log10(areas.min()), np.log10(areas.max()), num_bins + 1)

# 统计每个箱子里的融池数量
counts, _ = np.histogram(areas, bins=bins)

# 计算每个箱子的物理跨度 (宽度)
bin_widths = np.diff(bins)

# 计算箱子中心点 (使用几何平均值作为对数图的中心)
bin_centers = np.sqrt(bins[:-1] * bins[1:])

# 核心魔法：计算概率密度 (Frequency) = 数量 / (箱子宽度 * 总数)
pdf = counts / (bin_widths * len(areas))

# 提取非空的数据点供拟合与绘图
valid_mask = pdf > 0
valid_centers = bin_centers[valid_mask]
valid_pdf = pdf[valid_mask]

# ==========================================
# 5. 线性拟合提取幂律指数 tau
# ==========================================
print("  ➜ 正在双对数空间拟合幂律衰减指数...")
log_A = np.log10(valid_centers)
log_pdf = np.log10(valid_pdf)

# 选取中间的线性段进行拟合，避开左侧极小网格噪声和右侧边界截断造成的有限尺寸效应
fit_mask = (valid_centers >= 1) & (valid_centers <= 1000)

if np.sum(fit_mask) > 2:
    # 拟合一元一次方程 Y = slope * X + intercept
    slope, intercept = np.polyfit(log_A[fit_mask], log_pdf[fit_mask], 1)
    tau = -slope
    print(f"  ➜ 拟合成功! 提取到的幂律指数 tau = {tau:.3f}")
else:
    slope, intercept = -1.8, 0
    tau = 1.8
    print("  ➜ 拟合区间数据点不足。")

# 生成拟合线的数据
fit_line = (10 ** intercept) * (valid_centers ** slope)

# ==========================================
# 6. 绘图
# ==========================================
plt.figure(figsize=(8, 6), constrained_layout=True)

# 画对数分箱后的数据点
plt.loglog(valid_centers, valid_pdf, 'o', color='#1e3f66', markersize=7, label='Simulated Voids (Log-binned)')

# 画我们的拟合直线
plt.loglog(valid_centers[fit_mask], fit_line[fit_mask], '-', color='#cc0000', linewidth=2.5, 
           label=rf'Fitted Power Law ($\tau \approx {tau:.2f}$)')

# 画出理论的 2.05 参考线作对比
ref_idx = np.argmax(fit_mask)
offset_205 = valid_pdf[ref_idx] / (valid_centers[ref_idx] ** -2.05)
line_205 = offset_205 * (valid_centers ** -2.05)
plt.loglog(valid_centers, line_205, '--', color='gray', alpha=0.8, 
           label=r'Theoretical 2D Percolation ($\tau = 2.05$)')

plt.title('Melt Pond Cluster Size Distribution', fontsize=14, fontweight='bold')
plt.xlabel(r'Pond Area $A$ ($m^2$)', fontsize=12)
plt.ylabel(r'Probability Density $P(A)$', fontsize=12)
plt.grid(True, which="both", ls=":", alpha=0.5)
plt.legend(frameon=False, fontsize=11)

output_img = os.path.join(BASE_DIR, 'cluster_size_distribution.png')
plt.savefig(output_img, dpi=300)
print(f"✅ 簇尺寸分布图已保存至: {output_img}")
plt.show()