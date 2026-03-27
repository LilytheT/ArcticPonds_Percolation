import numpy as np
import glob
import os
from scipy.ndimage import label
import re

# 1. 定位数据文件
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATTERN = os.path.join(BASE_DIR, "voids_rho_*.txt")
files = sorted(glob.glob(FILE_PATTERN))

if not files:
    print("❌ 未找到 voids_rho_*.txt 数据文件！")
    exit()

rho_values = []
phi_values = []

# 设定统一分析网格分辨率 (2000x2000 对 500m 系统足够精确)
L = 500.0
resolution = 2000 

print("❄️ 开始分析北极融池渗流相变特征...")

for file_path in files:
    # A. 提取 rho 目标值
    rho_target = 0.0
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if 'Void Area Ratio (rho):' in line:
                numbers = re.findall(r'[\d.]+', line)
                if numbers: rho_target = float(numbers[0])
                break
    
    # B. 读取圆的坐标和半径
    data = np.loadtxt(file_path, skiprows=6)
    if data.ndim == 1: data = data.reshape(1, -1)
    if data.shape[1] < 4: continue
    
    x_centers, y_centers, radii = data[:, 1], data[:, 2], data[:, 3]
    
    # C. 将圆（冰层）栅格化投影到网格上
    grid = np.ones((resolution, resolution), dtype=int) # 1代表融池 (Voids)
    
    for x, y, r in zip(x_centers, y_centers, radii):
        # 优化计算：只在圆的包围盒内进行判断
        x_min = max(0, int((x-r)*resolution/L))
        x_max = min(resolution, int((x+r)*resolution/L) + 1)
        y_min = max(0, int((y-r)*resolution/L))
        y_max = min(resolution, int((y+r)*resolution/L) + 1)
        
        if x_min >= x_max or y_min >= y_max: continue
        
        xv, yv = np.meshgrid(
            np.linspace(x_min*L/resolution, x_max*L/resolution, x_max-x_min, endpoint=False),
            np.linspace(y_min*L/resolution, y_max*L/resolution, y_max-y_min, endpoint=False)
        )
        
        # 将圆内部标记为 0 (冰层)
        mask = (xv - x)**2 + (yv - y)**2 <= r**2
        grid[y_min:y_max, x_min:x_max][mask] = 0
        
    # D. 连通域计算 (Percolation Cluster Analysis)
    # 使用 4 连通 (十字结构) 寻找相连的融池
    structure = [[0,1,0],
                 [1,1,1],
                 [0,1,0]]
    labeled_array, num_features = label(grid, structure=structure)
    
    # E. 计算序参量 Phi = S_max / S_total
    if num_features > 0:
        component_sizes = np.bincount(labeled_array.ravel())
        component_sizes[0] = 0 # 索引 0 是冰层（背景），忽略它
        
        s_max = np.max(component_sizes)
        s_total = np.sum(component_sizes)
        
        phi = s_max / s_total if s_total > 0 else 0.0
    else:
        phi = 0.0
        
    rho_values.append(rho_target)
    phi_values.append(phi)
    print(f"  ➜ 处理完成: ρ = {rho_target:.3f} | 最大连通比(Φ) = {phi:.4f}")

# F. 保存结果供绘图使用
output_file = os.path.join(BASE_DIR, 'simulation_results.npz')
np.savez(output_file, rho=rho_values, phi=phi_values)
print(f"\n✅ 渗流序参量分析完成，数据已写入: {output_file}")