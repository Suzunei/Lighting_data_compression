"""
3D光照测试信号集合
模拟各种真实环境光照场景，用于压缩算法测试
每个函数返回: signal [D, H, W, C]
"""

import torch
import numpy as np

# ============================================================
# 1. 室内柔和光照 - Indoor Soft Lighting
# ============================================================
def create_indoor_soft_lighting(grid_size=32, num_channels=3):
    """
    模拟室内柔和漫射光照。
    特点: 低对比度、均匀分布、温暖色调
    """
    x = torch.linspace(-1, 1, grid_size)
    y = torch.linspace(-1, 1, grid_size)
    z = torch.linspace(-1, 1, grid_size)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

    signal = torch.zeros(grid_size, grid_size, grid_size, num_channels)

    # === Red Channel: 温暖基底 + 轻微衰减 ===
    signal[..., 0] = 0.55 + 0.1 * torch.cos(0.5 * np.pi * Z)
    signal[..., 0] += 0.05 * torch.sin(0.8 * np.pi * X) * torch.cos(0.6 * np.pi * Y)

    # === Green Channel: 均匀漫射 ===
    signal[..., 1] = 0.50 + 0.08 * torch.cos(0.6 * np.pi * (X + Y))
    signal[..., 1] += 0.04 * torch.sin(1.0 * np.pi * Z)

    # === Blue Channel: 略低的蓝色(暖光) ===
    signal[..., 2] = 0.42 + 0.06 * torch.cos(0.7 * np.pi * Z)
    signal[..., 2] += 0.03 * torch.sin(0.9 * np.pi * X) * torch.cos(0.8 * np.pi * Y)

    for c in range(num_channels):
        signal[..., c] = torch.clamp(signal[..., c], 0.1, 0.9)

    return signal


# ============================================================
# 2. 户外阳光直射 - Outdoor Direct Sunlight
# ============================================================
def create_outdoor_sunlight(grid_size=32, num_channels=3):
    """
    模拟户外强烈阳光直射。
    特点: 高对比度、明显的方向性、强烈高光
    """
    x = torch.linspace(-1, 1, grid_size)
    y = torch.linspace(-1, 1, grid_size)
    z = torch.linspace(-1, 1, grid_size)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

    # 阳光方向向量 (从左上方照射)
    sun_dir = torch.tensor([0.5, 0.7, 0.5])
    sun_dir = sun_dir / torch.norm(sun_dir)
    dot_product = X * sun_dir[0] + Y * sun_dir[1] + Z * sun_dir[2]

    signal = torch.zeros(grid_size, grid_size, grid_size, num_channels)

    # === Red Channel: 强烈暖光 ===
    signal[..., 0] = 0.45 + 0.35 * torch.clamp(dot_product, 0, 1)
    signal[..., 0] += 0.08 * torch.sin(2.0 * np.pi * X) * torch.cos(1.5 * np.pi * Z)

    # === Green Channel: 中等强度 ===
    signal[..., 1] = 0.42 + 0.32 * torch.clamp(dot_product, 0, 1)
    signal[..., 1] += 0.06 * torch.sin(1.8 * np.pi * Y) * torch.cos(2.0 * np.pi * X)

    # === Blue Channel: 天空散射光 ===
    signal[..., 2] = 0.35 + 0.25 * torch.clamp(dot_product, 0, 1)
    signal[..., 2] += 0.12 * (1 + Z) * 0.5  # 天空蓝色渐变

    for c in range(num_channels):
        signal[..., c] = torch.clamp(signal[..., c], 0.1, 0.9)

    return signal


# ============================================================
# 3. 日落黄昏光照 - Sunset/Golden Hour
# ============================================================
def create_sunset_lighting(grid_size=32, num_channels=3):
    """
    模拟日落黄昏时分的金色光照。
    特点: 橙红色调、水平方向光、长阴影
    """
    x = torch.linspace(-1, 1, grid_size)
    y = torch.linspace(-1, 1, grid_size)
    z = torch.linspace(-1, 1, grid_size)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

    # 低角度阳光
    horizon_factor = torch.exp(-2.0 * (Z + 0.8)**2)

    signal = torch.zeros(grid_size, grid_size, grid_size, num_channels)

    # === Red Channel: 强烈橙红 ===
    signal[..., 0] = 0.40 + 0.40 * horizon_factor
    signal[..., 0] += 0.10 * torch.sin(1.2 * np.pi * X) * (1 - Z * 0.3)

    # === Green Channel: 金色中调 ===
    signal[..., 1] = 0.35 + 0.28 * horizon_factor
    signal[..., 1] += 0.08 * torch.cos(1.0 * np.pi * Y) * (1 - Z * 0.2)

    # === Blue Channel: 暗淡蓝紫 ===
    signal[..., 2] = 0.25 + 0.15 * (1 - horizon_factor)
    signal[..., 2] += 0.10 * torch.clamp(Z + 0.5, 0, 1)  # 天空残余蓝

    for c in range(num_channels):
        signal[..., c] = torch.clamp(signal[..., c], 0.1, 0.9)

    return signal


# ============================================================
# 4. 阴天漫射光 - Overcast Diffuse Light
# ============================================================
def create_overcast_lighting(grid_size=32, num_channels=3):
    """
    模拟阴天多云的漫射光照。
    特点: 极低对比度、灰白色调、几乎无阴影
    """
    x = torch.linspace(-1, 1, grid_size)
    y = torch.linspace(-1, 1, grid_size)
    z = torch.linspace(-1, 1, grid_size)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

    signal = torch.zeros(grid_size, grid_size, grid_size, num_channels)

    # 所有通道接近相等(灰白)，轻微变化
    base = 0.52 + 0.05 * Z
    noise_x = 0.03 * torch.sin(1.5 * np.pi * X)
    noise_y = 0.02 * torch.cos(1.3 * np.pi * Y)

    # === Red Channel ===
    signal[..., 0] = base + noise_x + 0.02

    # === Green Channel ===
    signal[..., 1] = base + noise_y + 0.01

    # === Blue Channel: 略偏蓝 ===
    signal[..., 2] = base + 0.03 * torch.cos(1.0 * np.pi * Z)

    for c in range(num_channels):
        signal[..., c] = torch.clamp(signal[..., c], 0.1, 0.9)

    return signal


# ============================================================
# 5. 点光源场景 - Point Light Source
# ============================================================
def create_point_light(grid_size=32, num_channels=3):
    """
    模拟单个点光源(如灯泡)的光照。
    特点: 径向衰减、中心高亮、平方反比衰减
    """
    x = torch.linspace(-1, 1, grid_size)
    y = torch.linspace(-1, 1, grid_size)
    z = torch.linspace(-1, 1, grid_size)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

    # 光源位置
    light_pos = torch.tensor([0.0, 0.0, 0.3])
    dist = torch.sqrt((X - light_pos[0])**2 + (Y - light_pos[1])**2 + (Z - light_pos[2])**2 + 0.01)

    # 平方反比衰减
    attenuation = 1.0 / (1.0 + 3.0 * dist**2)

    signal = torch.zeros(grid_size, grid_size, grid_size, num_channels)

    # === Red Channel: 暖光 ===
    signal[..., 0] = 0.15 + 0.65 * attenuation

    # === Green Channel ===
    signal[..., 1] = 0.12 + 0.55 * attenuation

    # === Blue Channel: 较冷 ===
    signal[..., 2] = 0.10 + 0.40 * attenuation

    for c in range(num_channels):
        signal[..., c] = torch.clamp(signal[..., c], 0.1, 0.9)

    return signal


# ============================================================
# 6. 城市夜景霓虹 - Urban Neon Night
# ============================================================
def create_neon_lighting(grid_size=32, num_channels=3):
    """
    模拟城市夜景霓虹灯效果。
    特点: 多色光源、高频变化、强烈色彩对比
    """
    x = torch.linspace(-1, 1, grid_size)
    y = torch.linspace(-1, 1, grid_size)
    z = torch.linspace(-1, 1, grid_size)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

    signal = torch.zeros(grid_size, grid_size, grid_size, num_channels)

    # === Red Channel: 红色霓虹条纹 ===
    signal[..., 0] = 0.20 + 0.35 * torch.abs(torch.sin(3.0 * np.pi * X))
    signal[..., 0] += 0.15 * torch.abs(torch.cos(2.5 * np.pi * Z))

    # === Green Channel: 绿色点阵 ===
    signal[..., 1] = 0.15 + 0.25 * torch.abs(torch.sin(2.8 * np.pi * Y) * torch.cos(2.2 * np.pi * X))
    signal[..., 1] += 0.10 * torch.abs(torch.sin(3.5 * np.pi * Z))

    # === Blue Channel: 蓝紫霓虹 ===
    signal[..., 2] = 0.25 + 0.40 * torch.abs(torch.cos(2.5 * np.pi * (X + Y)))
    signal[..., 2] += 0.12 * torch.abs(torch.sin(3.0 * np.pi * Z) * torch.cos(2.0 * np.pi * Y))

    for c in range(num_channels):
        signal[..., c] = torch.clamp(signal[..., c], 0.1, 0.9)

    return signal


# ============================================================
# 7. 森林斑驳光照 - Dappled Forest Light
# ============================================================
def create_forest_dappled_light(grid_size=32, num_channels=3):
    """
    模拟阳光穿过树叶的斑驳光照。
    特点: 随机高频光斑、绿色环境光、高对比度
    """
    x = torch.linspace(-1, 1, grid_size)
    y = torch.linspace(-1, 1, grid_size)
    z = torch.linspace(-1, 1, grid_size)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

    # 模拟树叶间隙的光斑
    dapple = torch.abs(torch.sin(4.0 * np.pi * X) * torch.cos(3.5 * np.pi * Y) * torch.sin(3.0 * np.pi * Z))
    dapple += 0.3 * torch.abs(torch.sin(5.0 * np.pi * (X + Z)) * torch.cos(4.5 * np.pi * Y))

    signal = torch.zeros(grid_size, grid_size, grid_size, num_channels)

    # === Red Channel: 阳光温暖 ===
    signal[..., 0] = 0.25 + 0.30 * dapple
    signal[..., 0] += 0.08 * torch.sin(1.5 * np.pi * Z)

    # === Green Channel: 树叶反射(较强) ===
    signal[..., 1] = 0.30 + 0.35 * dapple
    signal[..., 1] += 0.12 * (0.5 - Z * 0.3)  # 地面绿色环境

    # === Blue Channel: 天空散射 ===
    signal[..., 2] = 0.20 + 0.20 * dapple
    signal[..., 2] += 0.10 * torch.clamp(Z + 0.5, 0, 1)

    for c in range(num_channels):
        signal[..., c] = torch.clamp(signal[..., c], 0.1, 0.9)

    return signal


# ============================================================
# 8. 水下焦散效果 - Underwater Caustics
# ============================================================
def create_underwater_caustics(grid_size=32, num_channels=3):
    """
    模拟水下光线焦散效果。
    特点: 波浪状光带、蓝绿色调、深度衰减
    """
    x = torch.linspace(-1, 1, grid_size)
    y = torch.linspace(-1, 1, grid_size)
    z = torch.linspace(-1, 1, grid_size)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

    # 焦散波纹
    caustic1 = torch.sin(3.5 * np.pi * X + 1.5 * np.pi * Y) * torch.cos(2.0 * np.pi * Z)
    caustic2 = torch.cos(2.8 * np.pi * X - 2.0 * np.pi * Y) * torch.sin(2.5 * np.pi * Z)
    caustic = 0.5 * (caustic1**2 + caustic2**2)

    # 深度衰减
    depth_atten = torch.exp(-1.5 * (1 - Z))

    signal = torch.zeros(grid_size, grid_size, grid_size, num_channels)

    # === Red Channel: 快速衰减 ===
    signal[..., 0] = 0.15 + 0.20 * caustic * depth_atten

    # === Green Channel: 中等穿透 ===
    signal[..., 1] = 0.25 + 0.35 * caustic * depth_atten
    signal[..., 1] += 0.10 * depth_atten

    # === Blue Channel: 最强穿透 ===
    signal[..., 2] = 0.35 + 0.40 * caustic * depth_atten
    signal[..., 2] += 0.15 * depth_atten

    for c in range(num_channels):
        signal[..., c] = torch.clamp(signal[..., c], 0.1, 0.9)

    return signal


# ============================================================
# 9. 舞台聚光灯 - Stage Spotlight
# ============================================================
def create_stage_spotlight(grid_size=32, num_channels=3):
    """
    模拟舞台聚光灯效果。
    特点: 锥形光束、边缘衰减、多个光源
    """
    x = torch.linspace(-1, 1, grid_size)
    y = torch.linspace(-1, 1, grid_size)
    z = torch.linspace(-1, 1, grid_size)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

    # 主聚光灯(中心)
    spot1_dist = torch.sqrt(X**2 + Y**2)
    spot1 = torch.exp(-4.0 * spot1_dist**2) * torch.clamp(1 - Z, 0, 1)

    # 侧聚光灯
    spot2_dist = torch.sqrt((X - 0.5)**2 + (Y - 0.3)**2)
    spot2 = 0.6 * torch.exp(-5.0 * spot2_dist**2) * torch.clamp(1 - Z, 0, 1)

    signal = torch.zeros(grid_size, grid_size, grid_size, num_channels)

    # === Red Channel: 暖白主光 ===
    signal[..., 0] = 0.12 + 0.60 * spot1 + 0.25 * spot2

    # === Green Channel ===
    signal[..., 1] = 0.10 + 0.55 * spot1 + 0.20 * spot2

    # === Blue Channel: 略带蓝调 ===
    signal[..., 2] = 0.15 + 0.50 * spot1 + 0.30 * spot2

    for c in range(num_channels):
        signal[..., c] = torch.clamp(signal[..., c], 0.1, 0.9)

    return signal


# ============================================================
# 10. 火焰/壁炉光照 - Fire/Fireplace Light
# ============================================================
def create_fire_lighting(grid_size=32, num_channels=3):
    """
    模拟火焰/壁炉的动态光照(静态快照)。
    特点: 橙红色、闪烁效果、底部光源
    """
    x = torch.linspace(-1, 1, grid_size)
    y = torch.linspace(-1, 1, grid_size)
    z = torch.linspace(-1, 1, grid_size)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

    # 火焰在底部
    fire_dist = torch.sqrt(X**2 + Y**2 + (Z + 0.8)**2)
    fire_glow = torch.exp(-2.0 * fire_dist)

    # 模拟闪烁(高频扰动)
    flicker = 0.15 * torch.sin(5.0 * np.pi * X) * torch.cos(4.5 * np.pi * Y) * torch.sin(4.0 * np.pi * Z)

    signal = torch.zeros(grid_size, grid_size, grid_size, num_channels)

    # === Red Channel: 强烈橙红 ===
    signal[..., 0] = 0.20 + 0.65 * fire_glow + torch.abs(flicker) * 0.5

    # === Green Channel: 橙黄 ===
    signal[..., 1] = 0.12 + 0.40 * fire_glow + torch.abs(flicker) * 0.3

    # === Blue Channel: 微弱 ===
    signal[..., 2] = 0.10 + 0.10 * fire_glow

    for c in range(num_channels):
        signal[..., c] = torch.clamp(signal[..., c], 0.1, 0.9)

    return signal


# ============================================================
# 11. 窗户光照 - Window Light
# ============================================================
def create_window_light(grid_size=32, num_channels=3):
    """
    模拟从窗户照入的自然光。
    特点: 矩形光斑、边缘柔化、方向性强
    """
    x = torch.linspace(-1, 1, grid_size)
    y = torch.linspace(-1, 1, grid_size)
    z = torch.linspace(-1, 1, grid_size)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

    # 窗户形状(使用sigmoid实现软边缘)
    window_x = torch.sigmoid(10 * (X + 0.5)) * torch.sigmoid(10 * (0.5 - X))
    window_z = torch.sigmoid(10 * (Z + 0.3)) * torch.sigmoid(10 * (0.7 - Z))
    window = window_x * window_z

    # 光线穿透深度衰减
    depth_factor = torch.exp(-1.5 * (Y + 1))

    signal = torch.zeros(grid_size, grid_size, grid_size, num_channels)

    # === Red Channel ===
    signal[..., 0] = 0.18 + 0.50 * window * depth_factor
    signal[..., 0] += 0.08 * torch.cos(1.0 * np.pi * Y)

    # === Green Channel ===
    signal[..., 1] = 0.16 + 0.48 * window * depth_factor
    signal[..., 1] += 0.06 * torch.cos(0.8 * np.pi * Y)

    # === Blue Channel: 天光偏蓝 ===
    signal[..., 2] = 0.20 + 0.52 * window * depth_factor
    signal[..., 2] += 0.10 * torch.cos(1.2 * np.pi * Y)

    for c in range(num_channels):
        signal[..., c] = torch.clamp(signal[..., c], 0.1, 0.9)

    return signal


# ============================================================
# 12. 全局照明/环境遮蔽 - Global Illumination with AO
# ============================================================
def create_gi_with_ao(grid_size=32, num_channels=3):
    """
    模拟全局照明配合环境遮蔽效果。
    特点: 角落暗化、软阴影、间接光照
    """
    x = torch.linspace(-1, 1, grid_size)
    y = torch.linspace(-1, 1, grid_size)
    z = torch.linspace(-1, 1, grid_size)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

    # 环境遮蔽 - 角落更暗
    corner_dist = torch.sqrt(X**2 + Y**2 + Z**2)
    ao = 1.0 - 0.3 * torch.clamp(corner_dist - 0.5, 0, 1)

    # 全局照明(来自上方)
    gi = 0.5 + 0.3 * Z
    gi += 0.1 * torch.cos(1.5 * np.pi * X) * torch.sin(1.2 * np.pi * Y)

    signal = torch.zeros(grid_size, grid_size, grid_size, num_channels)

    # === 三通道应用GI和AO ===
    signal[..., 0] = (0.50 + 0.08 * torch.sin(0.8 * np.pi * X)) * gi * ao
    signal[..., 1] = (0.48 + 0.06 * torch.cos(0.9 * np.pi * Y)) * gi * ao
    signal[..., 2] = (0.52 + 0.07 * torch.sin(1.0 * np.pi * Z)) * gi * ao

    for c in range(num_channels):
        signal[..., c] = torch.clamp(signal[..., c], 0.1, 0.9)

    return signal


# ============================================================
# 13. HDR天空盒 - HDR Skybox
# ============================================================
def create_hdr_skybox(grid_size=32, num_channels=3):
    """
    模拟HDR天空环境贴图的光照。
    特点: 太阳高光、天空渐变、地平线效果
    """
    x = torch.linspace(-1, 1, grid_size)
    y = torch.linspace(-1, 1, grid_size)
    z = torch.linspace(-1, 1, grid_size)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

    # 天空渐变(从地平线到天顶)
    sky_gradient = 0.5 + 0.4 * Z

    # 太阳光晕
    sun_pos = torch.tensor([0.6, 0.3, 0.7])
    sun_dist = torch.sqrt((X - sun_pos[0])**2 + (Y - sun_pos[1])**2 + (Z - sun_pos[2])**2)
    sun_glow = torch.exp(-3.0 * sun_dist)

    signal = torch.zeros(grid_size, grid_size, grid_size, num_channels)

    # === Red Channel: 暖色太阳 ===
    signal[..., 0] = 0.30 * sky_gradient + 0.50 * sun_glow
    signal[..., 0] += 0.05 * torch.sin(2.0 * np.pi * X)

    # === Green Channel ===
    signal[..., 1] = 0.35 * sky_gradient + 0.40 * sun_glow
    signal[..., 1] += 0.04 * torch.cos(1.8 * np.pi * Y)

    # === Blue Channel: 天空蓝 ===
    signal[..., 2] = 0.50 * sky_gradient + 0.25 * sun_glow
    signal[..., 2] += 0.08 * torch.sin(1.5 * np.pi * Z)

    for c in range(num_channels):
        signal[..., c] = torch.clamp(signal[..., c], 0.1, 0.9)

    return signal


# ============================================================
# 14. 体积光/丁达尔效应 - Volumetric Light (God Rays)
# ============================================================
def create_volumetric_light(grid_size=32, num_channels=3):
    """
    模拟体积光/丁达尔效应(上帝光线)。
    特点: 光束穿透、尘埃散射、径向光线
    """
    x = torch.linspace(-1, 1, grid_size)
    y = torch.linspace(-1, 1, grid_size)
    z = torch.linspace(-1, 1, grid_size)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

    # 从顶部射入的光束
    beam_angle = torch.atan2(X, Z + 1 + 1e-6)
    beam_pattern = torch.exp(-8.0 * (beam_angle - 0.3)**2)
    beam_pattern += 0.5 * torch.exp(-10.0 * (beam_angle + 0.4)**2)

    # 深度散射
    scatter = torch.exp(-0.8 * (1 - Z))

    signal = torch.zeros(grid_size, grid_size, grid_size, num_channels)

    # === Red Channel ===
    signal[..., 0] = 0.15 + 0.45 * beam_pattern * scatter
    signal[..., 0] += 0.05 * torch.sin(3.0 * np.pi * Y)

    # === Green Channel ===
    signal[..., 1] = 0.12 + 0.40 * beam_pattern * scatter
    signal[..., 1] += 0.04 * torch.cos(2.5 * np.pi * X)

    # === Blue Channel ===
    signal[..., 2] = 0.18 + 0.35 * beam_pattern * scatter
    signal[..., 2] += 0.06 * torch.sin(2.8 * np.pi * Z)

    for c in range(num_channels):
        signal[..., c] = torch.clamp(signal[..., c], 0.1, 0.9)

    return signal


# ============================================================
# 15. 多光源混合 - Multiple Light Sources
# ============================================================
def create_multiple_lights(grid_size=32, num_channels=3):
    """
    模拟多个不同颜色光源的混合。
    特点: 红绿蓝三原色光源、颜色混合
    """
    x = torch.linspace(-1, 1, grid_size)
    y = torch.linspace(-1, 1, grid_size)
    z = torch.linspace(-1, 1, grid_size)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

    # 红光源位置
    red_pos = torch.tensor([-0.5, -0.5, 0.0])
    red_dist = torch.sqrt((X - red_pos[0])**2 + (Y - red_pos[1])**2 + (Z - red_pos[2])**2 + 0.01)
    red_light = 0.6 / (1.0 + 4.0 * red_dist**2)

    # 绿光源位置
    green_pos = torch.tensor([0.5, -0.5, 0.0])
    green_dist = torch.sqrt((X - green_pos[0])**2 + (Y - green_pos[1])**2 + (Z - green_pos[2])**2 + 0.01)
    green_light = 0.6 / (1.0 + 4.0 * green_dist**2)

    # 蓝光源位置
    blue_pos = torch.tensor([0.0, 0.5, 0.3])
    blue_dist = torch.sqrt((X - blue_pos[0])**2 + (Y - blue_pos[1])**2 + (Z - blue_pos[2])**2 + 0.01)
    blue_light = 0.6 / (1.0 + 4.0 * blue_dist**2)

    signal = torch.zeros(grid_size, grid_size, grid_size, num_channels)

    # === 光源混合 ===
    signal[..., 0] = 0.10 + red_light + 0.2 * green_light + 0.2 * blue_light
    signal[..., 1] = 0.10 + 0.2 * red_light + green_light + 0.2 * blue_light
    signal[..., 2] = 0.10 + 0.2 * red_light + 0.2 * green_light + blue_light

    for c in range(num_channels):
        signal[..., c] = torch.clamp(signal[..., c], 0.1, 0.9)

    return signal


# ============================================================
# 测试信号索引字典
# ============================================================
TEST_SIGNALS = {
    "indoor_soft": create_indoor_soft_lighting,
    "outdoor_sunlight": create_outdoor_sunlight,
    "sunset": create_sunset_lighting,
    "overcast": create_overcast_lighting,
    "point_light": create_point_light,
    "neon": create_neon_lighting,
    "forest_dappled": create_forest_dappled_light,
    "underwater_caustics": create_underwater_caustics,
    "stage_spotlight": create_stage_spotlight,
    "fire": create_fire_lighting,
    "window_light": create_window_light,
    "gi_ao": create_gi_with_ao,
    "hdr_skybox": create_hdr_skybox,
    "volumetric": create_volumetric_light,
    "multiple_lights": create_multiple_lights,
}


def get_all_test_signals(grid_size=32, num_channels=3):
    """
    获取所有测试信号的字典。
    返回: {name: signal_tensor} 字典
    """
    return {name: func(grid_size, num_channels) for name, func in TEST_SIGNALS.items()}


def get_test_signal_by_name(name, grid_size=32, num_channels=3):
    """
    按名称获取特定测试信号。
    """
    if name not in TEST_SIGNALS:
        raise ValueError(f"Unknown test signal: {name}. Available: {list(TEST_SIGNALS.keys())}")
    return TEST_SIGNALS[name](grid_size, num_channels)
