import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import label
from skimage.measure import regionprops
import glob

# 1. 找到对应的文件 (以 rho=0.31 附近的文件为例，相变点特征最明显)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATTERN = os.path.join(BASE_DIR, "voids_rho_0.31*.txt")
files = glob.glob(FILE_PATTERN)

if not files:
    print("❌ 未找到 voids_rho_0.310.txt 文件！")
    exit()

file_path = files[0]
print(f"❄️ 正在分析文件: {os.path.basename(file_path)}")

# 2. 重新进行高精度栅格化 (复用我们之前的物理逻辑)
L = 500.0
resolution = 2000  # 高分辨率，相当于 0.25m / 像素
dx = L / resolution # 每个像素的实际物理边长

data = np.loadtxt(file_path, skiprows=6)
x_centers, y_centers, radii = data[:, 1], data[:, 2], data[:, 3]

print("  ➜ 正在将几何体投影到高精度像素网格...")
grid = np.ones((resolution, resolution), dtype=int) # 1代表融池

for x, y, r in zip(x_centers, y_centers, radii):
    x_min, x_max = max(0, int((x-r)/dx)), min(resolution, int((x+r)/dx) + 1)
    y_min, y_max = max(0, int((y-r)/dx)), min(resolution, int((y+r)/dx) + 1)
    if x_min >= x_max or y_min >= y_max: continue
    
    xv, yv = np.meshgrid(np.linspace(x_min*dx, x_max*dx, x_max-x_min, endpoint=False),
                         np.linspace(y_min*dx, y_max*dx, y_max-y_min, endpoint=False))
    mask = (xv - x)**2 + (yv - y)**2 <= r**2
    grid[y_min:y_max, x_min:x_max][mask] = 0

# 3. 连通域标记
print("  ➜ 正在寻找独立融池 (Cluster Labeling)...")
structure = [[0,1,0], [1,1,1], [0,1,0]]
labeled_array, num_features = label(grid, structure=structure)
print(f"  ➜ 共发现 {num_features} 个独立融池。")

# 4. 提取面积 (A) 和 周长 (P)
print("  ➜ 正在提取每个融池的物理面积与修正周长...")
# regionprops 会自动计算连通区域的各种几何特征，并且其 perimeter 方法自带对角线修正
props = regionprops(labeled_array)

areas = []
perimeters = []

for prop in props:
    # 忽略面积太小（小于3个像素）的噪点
    if prop.area < 3: 
        continue
    
    # 将像素数量转换为真实的物理单位 (平方米 和 米)
    physical_area = prop.area * (dx ** 2)
    physical_perimeter = prop.perimeter * dx
    
    areas.append(physical_area)
    perimeters.append(physical_perimeter)

# 转换为 numpy 数组方便计算
A = np.array(areas)
P = np.array(perimeters)

# ==========================================
# 5. 对数分箱 (Log-Binning) 与 分形维数计算
# ==========================================
print("  ➜ 正在执行对数分箱与平滑求导...")
# 将数据转入对数空间
log_A = np.log10(A)
log_P = np.log10(P)

# 设置对数分箱的箱子边界 (例如分成 30 个箱子)
num_bins = 30
bins = np.linspace(log_A.min(), log_A.max(), num_bins + 1)
bin_centers = (bins[:-1] + bins[1:]) / 2

# 计算每个箱子内的平均 log(P)
mean_log_P = np.zeros(num_bins)
for i in range(num_bins):
    mask = (log_A >= bins[i]) & (log_A < bins[i+1])
    if np.any(mask):
        mean_log_P[i] = np.mean(log_P[mask])
    else:
        mean_log_P[i] = np.nan

# 过滤掉空的箱子
valid = ~np.isnan(mean_log_P)
valid_log_A = bin_centers[valid]
valid_mean_log_P = mean_log_P[valid]

# 计算局部分形维数 D = 2 * d(logP) / d(logA)
# 使用 np.gradient 计算数值导数
D_values = 2 * np.gradient(valid_mean_log_P, valid_log_A)

# ==========================================
# 6. 绘制结果对比图
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), constrained_layout=True)

# --- 图 1：散点与对数分箱均值 ---
ax1.plot(log_A, log_P, 'o', color='#8ca3ba', markersize=2, alpha=0.15, label='Individual ponds')
ax1.plot(valid_log_A, valid_mean_log_P, 's-', color='#cc0000', linewidth=2, markersize=6, label=r'Log-binned Mean $\langle P \\rangle$')

# 红色虚线参考线 D=1 (平移以对齐均值曲线起点)
offset = valid_mean_log_P[0] - 0.5 * valid_log_A[0]
ax1.plot(valid_log_A, 0.5 * valid_log_A + offset, '--', color='black', alpha=0.8, label='Theoretical $D=1$ (Slope=0.5)')

ax1.set_title('Log-Binned Perimeter vs. Area', fontsize=14, fontweight='bold')
ax1.set_xlabel(r'$\log_{10}(\text{Area } A)$', fontsize=12)
ax1.set_ylabel(r'$\log_{10}(\text{Perimeter } P)$', fontsize=12)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(frameon=False)

# --- 图 2：分形维数 D(A) 的转变曲线 ---
ax2.plot(valid_log_A, D_values, 'o-', color='#1e3f66', linewidth=2, markersize=6)
ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.6, label='Lower Limit (D=1)')
ax2.axhline(y=2.0, color='#cc0000', linestyle='--', alpha=0.6, label='Upper Limit (D=2, Spanning)')

# 根据论文，D应该在1到2之间。由于网格噪声局部求导可能略有溢出，可限制显示范围
ax2.set_ylim(0.5, 2.5)
ax2.set_title('Fractal Dimension Transition $D(A)$', fontsize=14, fontweight='bold')
ax2.set_xlabel(r'$\log_{10}(\text{Area } A)$', fontsize=12)
ax2.set_ylabel(r'Fractal Dimension $D$', fontsize=12)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(frameon=False)

output_img = os.path.join(BASE_DIR, 'fractal_dimension_binned.png')
plt.savefig(output_img, dpi=300)
print(f"✅ 分形维数分析图已保存至: {output_img}")
plt.show()