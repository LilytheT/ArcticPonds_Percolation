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

# 5. 绘制 P-A 双对数散点图 (Log-Log Scatter Plot)
print("  ➜ 正在绘制分形维数散点图...")
plt.figure(figsize=(8, 6), constrained_layout=True)

# 绘制散点
plt.loglog(A, P, 'o', color='#1e3f66', markersize=2, alpha=0.3, label='Individual pond data')

# 添加理论参考线：规则形状 (D=1, P ∝ A^0.5)
A_ref = np.logspace(np.log10(min(A)), np.log10(max(A)), 100)
P_ref_1d = 4 * np.sqrt(A_ref) # 类似正方形的比例关系
plt.loglog(A_ref, P_ref_1d, '--', color='#cc0000', linewidth=2, label=r'Regular shape ($P \propto A^{0.5}$, $D=1$)')

plt.title('Melt Pond Perimeter vs. Area (Log-Log)', fontsize=14, fontweight='bold')
plt.xlabel(r'Pond Area $A$ ($m^2$)', fontsize=12)
plt.ylabel(r'Pond Perimeter $P$ ($m$)', fontsize=12)
plt.grid(True, which="both", ls=":", alpha=0.5)
plt.legend(frameon=False, fontsize=11)

output_img = os.path.join(BASE_DIR, 'fractal_dimension_scatter.png')
plt.savefig(output_img, dpi=300)
print(f"✅ P-A 散点图已保存至: {output_img}")
plt.show()