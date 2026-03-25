import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from test_signal_3d import get_test_signal_by_name, get_all_test_signals

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# ==================== 辅助函数：四元数与旋转矩阵 ====================
def quaternion_to_rotation_matrix(q):
    """
    将四元数转换为旋转矩阵 (3DGS标准实现)。
    q: [..., 4] 四元数 (w, x, y, z)
    返回: [..., 3, 3] 旋转矩阵
    """
    q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    R = torch.stack([
        torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y], dim=-1),
        torch.stack([2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x], dim=-1),
        torch.stack([2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y], dim=-1)
    ], dim=-2)
    return R

# ==================== 第1步：构建测试数据 (3D) ====================
print("Step 1: Generating 3D test data (simulating SH coefficients at probe positions)...")

def create_test_signal_3d(grid_size=32, num_channels=3):
    """
    创建3D模拟光探针的测试信号（RGB 3通道）。
    中等频率设计，目标PSNR 45-50dB。
    与MBD_Control.py和MBD.py使用相同的测试信号以便对照。
    返回: 信号 [D, H, W, 3]
    """
    x = torch.linspace(-1, 1, grid_size)
    y = torch.linspace(-1, 1, grid_size)
    z = torch.linspace(-1, 1, grid_size)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    R = torch.sqrt(X**2 + Y**2 + Z**2 + 1e-8)

    signal = torch.zeros(grid_size, grid_size, grid_size, num_channels)

    # === Red Channel: 环境光 + 软阴影 ===
    signal[..., 0] = 0.5 + 0.2 * torch.cos(np.pi * R * 0.8)
    signal[..., 0] += 0.1 * torch.sin(1.5 * np.pi * X) * torch.cos(1.2 * np.pi * Y)
    signal[..., 0] += 0.06 * torch.sin(2.0 * np.pi * Z) * torch.cos(1.8 * np.pi * X)

    # === Green Channel: 方向性光照 ===
    signal[..., 1] = 0.5 + 0.18 * X * torch.cos(1.0 * np.pi * Y)
    signal[..., 1] += 0.08 * torch.sin(1.8 * np.pi * X) * torch.cos(1.5 * np.pi * Z)

    # === Blue Channel: 天空渐变 ===
    signal[..., 2] = 0.5 + 0.15 * Z * torch.sin(1.0 * np.pi * (X + Y))
    signal[..., 2] += 0.06 * torch.sin(2.0 * np.pi * Z) * torch.cos(1.8 * np.pi * Y)

    # 将信号值限制在合理范围
    for c in range(num_channels):
        signal[..., c] = torch.clamp(signal[..., c], 0.1, 0.9)

    return signal

# 生成3D测试信号（与MBD_Control和MBD使用相同的RGB测试信号）
grid_size = 32  # 3D网格使用较小的尺寸 (32^3 = 32768个探针)
ground_truth_SH = create_test_signal_3d(grid_size, num_channels=3)  # RGB数据 [D,H,W,C]
D, H, W, C = ground_truth_SH.shape
print(f"Generated 3D simulated SH coefficient volume size: {D}x{H}x{W}x{C}")

# 准备训练数据：展平3D网格，每个点代表一个探针的位置和其SH系数
x_coords = torch.linspace(0, 1, D)
y_coords = torch.linspace(0, 1, H)
z_coords = torch.linspace(0, 1, W)
X_grid, Y_grid, Z_grid = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
probe_positions = torch.stack([X_grid.flatten(), Y_grid.flatten(), Z_grid.flatten()], dim=-1)  # [N, 3]
probe_SH_data = ground_truth_SH.view(-1, C)  # [N, C]

# ==================== 第2步：实现纯高斯压缩模型（无MLP） ====================
print("\nStep 2: Building Gaussian-Only Compression Model (NO MLP)...")

class GaussianOnlyCompressor3D(nn.Module):
    """
    纯高斯压缩器 (Gaussian-Only Compressor)
    没有MLP解码器，高斯函数直接输出SH系数。
    核心：使用完整协方差矩阵的3D高斯函数直接加权插值SH系数。
    """
    def __init__(self, num_gaussians=64, sh_dim=9):
        """
        参数:
            num_gaussians (K): 高斯函数的数量
            sh_dim (C): 球谐系数的维度（每个高斯直接携带SH系数）
        """
        super().__init__()
        self.K = num_gaussians
        self.C = sh_dim

        # ========== 3D高斯函数参数（可学习） ==========
        # 位置 mu: [K, 3] (3D位置)
        self.mu = nn.Parameter(torch.rand(num_gaussians, 3))

        # 对数尺度 log_s: [K, 3] (使用对数确保为正)
        self.log_s = nn.Parameter(torch.randn(num_gaussians, 3) * 0.3 - 1.5)

        # 四元数旋转 q: [K, 4] (w, x, y, z)
        self.q = nn.Parameter(torch.zeros(num_gaussians, 4))
        with torch.no_grad():
            self.q[:, 0] = 1.0

        # SH系数 F: [K, C] - 每个高斯直接携带SH系数
        self.F = nn.Parameter(torch.randn(num_gaussians, sh_dim) * 0.1 + 0.5)

        # 初始化高斯位置
        with torch.no_grad():
            indices = torch.randperm(probe_positions.shape[0])[:num_gaussians]
            self.mu.copy_(probe_positions[indices])

        print(f"Gaussian-Only model initialized: K={num_gaussians} gaussians, SH dim={sh_dim}")
        print(f"  Transforms: Position(3D) + Scale(3D) + Rotation(Quaternion)")
        print(f"  NO MLP decoder - direct SH coefficient weighted interpolation")

    def gaussian_function_3d(self, p, mu_j, s_j, q_j):
        """
        计算位置p处的完整协方差3D高斯函数值 G_j(p)。
        使用完整的协方差矩阵 Σ = R @ S @ S^T @ R^T
        """
        R = quaternion_to_rotation_matrix(q_j)  # [K, 3, 3]
        s_inv_sq = 1.0 / (s_j ** 2 + 1e-8)  # [K, 3]
        S_inv_sq = torch.diag_embed(s_inv_sq)  # [K, 3, 3]
        precision = R @ S_inv_sq @ R.transpose(-1, -2)  # [K, 3, 3]

        diff = p.unsqueeze(1) - mu_j.unsqueeze(0)  # [N, K, 3]
        diff_expanded = diff.unsqueeze(-1)  # [N, K, 3, 1]
        precision_expanded = precision.unsqueeze(0)  # [1, K, 3, 3]

        mahalanobis_sq = (diff_expanded.transpose(-1, -2) @ precision_expanded @ diff_expanded).squeeze(-1).squeeze(-1)

        return torch.exp(-0.5 * mahalanobis_sq)

    def compute_influence_radius(self, s_j):
        """计算高斯函数的影响半径"""
        return 3 * s_j.max(dim=1)[0]

    def forward(self, probe_pos):
        """
        前向传播（无MLP，直接加权插值）。
        输入: probe_pos [N, 3] 3D探针位置
        返回:
            reconstructed_SH [N, C]: 重建的SH系数
            weights [N, K]: 高斯加权权重
        """
        s = torch.exp(self.log_s)

        # 1. 计算每个3D高斯函数在探针位置的值
        gaussian_vals = self.gaussian_function_3d(probe_pos, self.mu, s, self.q)  # [N, K]

        # 2. 计算影响半径掩码
        radii = self.compute_influence_radius(s)  # [K]
        dist = torch.norm(probe_pos.unsqueeze(1) - self.mu.unsqueeze(0), dim=2)  # [N, K]
        mask = dist < radii.unsqueeze(0)  # [N, K]
        gaussian_vals = gaussian_vals * mask.float()

        # 3. 归一化加权并直接插值SH系数
        weights = gaussian_vals / (gaussian_vals.sum(dim=1, keepdim=True) + 1e-8)
        reconstructed_SH = torch.matmul(weights, self.F)  # [N, C]

        return reconstructed_SH, weights

    def get_compression_ratio(self, original_size_bytes):
        """
        计算压缩比。
        压缩后: 高斯参数(K*(3+3+4+C)*4)
        """
        gaussian_params_size = self.K * (3 + 3 + 4 + self.C) * 4
        compressed_size_bytes = gaussian_params_size
        ratio = original_size_bytes / compressed_size_bytes
        return ratio, compressed_size_bytes

    def get_gaussian_params(self):
        """获取高斯参数用于可视化"""
        with torch.no_grad():
            mu = self.mu.cpu().numpy()
            s = torch.exp(self.log_s).cpu().numpy()
            q = self.q.cpu().numpy()
            q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
        return mu, s, q


class GaussianOnlySolver3D:
    """纯高斯压缩求解器"""
    def __init__(self, model, lambda_reg=1e-4):
        self.model = model
        self.lambda_reg = lambda_reg

        gaussian_params = [self.model.mu, self.model.log_s, self.model.q, self.model.F]
        self.optimizer = optim.Adam(gaussian_params, lr=0.005)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=50, factor=0.5)

    def compute_loss(self, pred_SH, target_SH):
        """计算损失函数"""
        mse_loss = torch.mean((pred_SH - target_SH) ** 2)
        reg_loss = self.lambda_reg * (torch.sum(self.model.F ** 2) + torch.sum(torch.exp(self.model.log_s) ** 2))
        total_loss = mse_loss + reg_loss
        return total_loss, mse_loss, reg_loss

    def train_step(self, pos_batch, sh_batch):
        """单次训练步骤"""
        self.optimizer.zero_grad()

        pred_SH, _ = self.model(pos_batch)
        total_loss, mse_loss, reg_loss = self.compute_loss(pred_SH, sh_batch)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'mse_loss': mse_loss.item(),
            'reg_loss': reg_loss.item()
        }

    def train(self, probe_pos, probe_sh, epochs=500, batch_size=2048):
        """训练循环"""
        num_samples = probe_pos.shape[0]
        losses = []

        print(f"Starting Gaussian-Only training ({epochs} epochs)...")
        for epoch in range(epochs):
            indices = torch.randperm(num_samples)[:batch_size]
            pos_batch = probe_pos[indices]
            sh_batch = probe_sh[indices]

            loss_dict = self.train_step(pos_batch, sh_batch)
            losses.append(loss_dict)

            if epoch % 100 == 0 or epoch == epochs - 1:
                print(f"  Epoch {epoch:4d} | Loss: {loss_dict['total_loss']:.6f} | MSE: {loss_dict['mse_loss']:.6f}")

        return losses


# ==================== 创建并训练纯高斯模型 ====================
print("\n" + "="*70)
print("GAUSSIAN-ONLY COMPRESSION MODEL (NO MLP)")
print("="*70)

# ==================== 对照组参数配置 ====================
# 与 MBD_Control.py (纯MBD) 和 MBD.py (MBD+3DGS) 统一的对照组参数
num_gaussians = 128  # 高斯数量（与MBD的M=64+N=64相近）
original_size_bytes = probe_positions.shape[0] * C * 4  # N * C * sizeof(float32)

# 创建纯高斯模型
model = GaussianOnlyCompressor3D(
    num_gaussians=num_gaussians,
    sh_dim=3  # RGB 3通道（与MBD对照组统一）
)
comp_ratio, comp_size = model.get_compression_ratio(original_size_bytes)
print(f"\nOriginal data size: {original_size_bytes/1024:.1f} KB")
print(f"Compressed model size: ~{comp_size/1024:.1f} KB")
print(f"Compression ratio: {comp_ratio:.1f}:1")

# 训练模型（统一训练参数：epochs=1500, batch_size=4096）
solver = GaussianOnlySolver3D(model, lambda_reg=1e-5)  # 与MBD.py统一正则化强度
losses = solver.train(probe_positions, probe_SH_data, epochs=1500, batch_size=4096)

# ==================== 第3步：评估与可视化 ====================
print("\nStep 3: Evaluating Gaussian-Only Compression and Reconstruction...")

model.eval()
with torch.no_grad():
    reconstructed_SH, weights = model(probe_positions)
    reconstructed_vol = reconstructed_SH.view(D, H, W, C).cpu().numpy()

# 计算重建质量指标
def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def compute_ssim(img1, img2):
    """简化的SSIM计算"""
    C1, C2 = (0.01 * 1.0)**2, (0.03 * 1.0)**2
    mu1, mu2 = img1.mean(), img2.mean()
    sigma1_sq, sigma2_sq = img1.var(), img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    ssim = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim

# 计算每通道指标
psnr_values = []
ssim_values = []
for c in range(C):
    psnr_c = compute_psnr(ground_truth_SH[..., c].numpy(), reconstructed_vol[..., c])
    ssim_c = compute_ssim(ground_truth_SH[..., c].numpy().flatten(), reconstructed_vol[..., c].flatten())
    psnr_values.append(psnr_c)
    ssim_values.append(ssim_c)

avg_psnr = np.mean(psnr_values)
avg_ssim = np.mean(ssim_values)

print(f"\n" + "="*50)
print("GAUSSIAN-ONLY RECONSTRUCTION QUALITY")
print("="*50)
print(f"  Avg PSNR: {avg_psnr:.2f} dB")
print(f"  Avg SSIM: {avg_ssim:.4f}")
print(f"  Final Loss: {losses[-1]['total_loss']:.6f}")
print(f"  Model Size: {comp_size/1024:.1f} KB (Ratio: {comp_ratio:.1f}:1)")
print("="*50)

# ==================== 可视化结果 ====================
print("\nGenerating visualization results...")

# 取中间切片进行可视化
z_slice = D // 2
gt_slice = ground_truth_SH[z_slice, :, :, :].numpy()  # [H, W, C]
rec_slice = reconstructed_vol[z_slice, :, :, :]  # [H, W, C]

fig = plt.figure(figsize=(20, 10))

# 1. 原始SH系数图
ax1 = plt.subplot(2, 4, 1)
im1 = ax1.imshow(gt_slice[..., :3], vmin=0, vmax=1)
ax1.set_title(f'Original SH (Z={z_slice})\n{C} channels, {D}x{H}x{W}')
ax1.set_ylabel('Y index')
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

# 2. 高斯重建图
ax2 = plt.subplot(2, 4, 2)
im2 = ax2.imshow(rec_slice[..., :3], vmin=0, vmax=1)
ax2.set_title(f'Gaussian-Only Reconstruction\nPSNR: {avg_psnr:.1f} dB')
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

# 3. 误差图
ax3 = plt.subplot(2, 4, 3)
error = np.abs(gt_slice - rec_slice).mean(axis=-1)
im3 = ax3.imshow(error, cmap='hot', vmin=0, vmax=0.15)
ax3.set_title(f'Reconstruction Error\nSSIM: {avg_ssim:.4f}')
plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

# 4. 3D高斯分布位置
ax4 = fig.add_subplot(2, 4, 4, projection='3d')
mu, s, q = model.get_gaussian_params()

ax4.scatter(mu[:, 0], mu[:, 1], mu[:, 2],
            c='black', s=20, alpha=0.8, label='Gaussian Centers')

def draw_ellipsoid(ax, center, scale, quaternion, n_points=20, alpha=0.15, color='blue'):
    """在3D坐标系中绘制椭球体"""
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points // 2)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    x = x * scale[0]
    y = y * scale[1]
    z = z * scale[2]

    q_norm = quaternion / (np.linalg.norm(quaternion) + 1e-8)
    w, x_q, y_q, z_q = q_norm

    R = np.array([
        [1 - 2*y_q*y_q - 2*z_q*z_q, 2*x_q*y_q - 2*w*z_q, 2*x_q*z_q + 2*w*y_q],
        [2*x_q*y_q + 2*w*z_q, 1 - 2*x_q*x_q - 2*z_q*z_q, 2*y_q*z_q - 2*w*x_q],
        [2*x_q*z_q - 2*w*y_q, 2*y_q*z_q + 2*w*x_q, 1 - 2*x_q*x_q - 2*y_q*y_q]
    ])

    points = np.array([x.flatten(), y.flatten(), z.flatten()])
    rotated_points = R @ points

    x_rot = rotated_points[0, :].reshape(x.shape) + center[0]
    y_rot = rotated_points[1, :].reshape(y.shape) + center[1]
    z_rot = rotated_points[2, :].reshape(z.shape) + center[2]

    ax.plot_surface(x_rot, y_rot, z_rot, alpha=alpha, color=color,
                    linewidth=0, antialiased=True)

# 绘制部分椭球
num_ellipsoids = min(model.K, 64)
indices = np.linspace(0, model.K - 1, num_ellipsoids, dtype=int)
colors = plt.cm.viridis(np.linspace(0, 1, num_ellipsoids))
for idx, color in zip(indices, colors):
    draw_ellipsoid(ax4, mu[idx], s[idx], q[idx], color=color, alpha=0.12)

ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.set_zlim(0, 1)
ax4.set_title(f'3D Gaussian Ellipsoids\nK={model.K}')
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_zlabel('Z')


# 5. 训练损失曲线
ax5 = plt.subplot(2, 4, 5)
total_losses = [l['total_loss'] for l in losses]
mse_losses = [l['mse_loss'] for l in losses]
ax5.semilogy(total_losses, 'b-', linewidth=2, label='Total Loss')
ax5.semilogy(mse_losses, 'g--', linewidth=1.5, alpha=0.7, label='MSE Loss')
ax5.set_title('Training Loss (Log Scale)')
ax5.set_xlabel('Iterations')
ax5.set_ylabel('Loss')
ax5.legend()
ax5.grid(True, alpha=0.3)


# 6. 拟合对比曲线
ax6 = plt.subplot(2, 4, 6)
y_line = H // 2
channel_colors = ['#E74C3C', '#27AE60', '#3498DB']
channel_names = ['R', 'G', 'B']

for c, (col, name) in enumerate(zip(channel_colors[:3], channel_names)):
    ax6.plot(gt_slice[y_line, :, c], color=col, linestyle='-', linewidth=1.5, alpha=0.7, label=f'Original {name}')
    ax6.plot(rec_slice[y_line, :, c], color=col, linestyle='--', linewidth=1.5, alpha=0.9, label=f'Recon {name}')

ax6.set_title(f'Fitting Comparison (Z={z_slice}, Y={y_line})')
ax6.set_xlabel('X Coordinate')
ax6.set_ylabel('SH Coefficient Value')
ax6.legend(loc='upper right', fontsize='x-small', ncol=2)
ax6.grid(True, alpha=0.3)


# 7. 旋转角度分布
ax7 = plt.subplot(2, 4, 7)
q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
rotation_angles = 2 * np.arccos(np.clip(q_norm[:, 0], -1, 1)) * 180 / np.pi

ax7.hist(rotation_angles, bins=30, alpha=0.7, color='purple', edgecolor='black')
ax7.axvline(x=rotation_angles.mean(), color='red', linestyle='--',
            label=f'Mean: {rotation_angles.mean():.1f}°')
ax7.set_title('Rotation Angle Distribution')
ax7.set_xlabel('Rotation Angle (degrees)')
ax7.set_ylabel('Count')
ax7.legend()
ax7.grid(True, alpha=0.3)

# 8. 综合信息
ax8 = plt.subplot(2, 4, 8)
ax8.axis('off')
num_probes = D * H * W
scale_ratios = s.max(axis=1) / (s.min(axis=1) + 1e-8)

info_text = f"""
GAUSSIAN-ONLY COMPRESSION SUMMARY
==================================
Original Data:
  Volume: {D}x{H}x{W} = {num_probes} probes
  SH Dim: {C} (L=0,1,2 bands)
  Size: {original_size_bytes/1024:.1f} KB

Gaussian-Only Model (NO MLP):
  Gaussians (K): {model.K}
  Est. Size: {comp_size/1024:.1f} KB
  Ratio: {comp_ratio:.1f}:1

Transform Statistics:
  Scale X: {s[:, 0].mean():.4f} ± {s[:, 0].std():.4f}
  Scale Y: {s[:, 1].mean():.4f} ± {s[:, 1].std():.4f}
  Scale Z: {s[:, 2].mean():.4f} ± {s[:, 2].std():.4f}
  Anisotropy: {scale_ratios.mean():.2f} ± {scale_ratios.std():.2f}
  Rotation: {rotation_angles.mean():.1f}° ± {rotation_angles.std():.1f}°

Reconstruction Quality:
  Avg PSNR: {avg_psnr:.1f} dB
  Avg SSIM: {avg_ssim:.4f}
  Final Loss: {losses[-1]['total_loss']:.6f}
"""
ax8.text(0.05, 0.5, info_text, fontsize=9,
        family='monospace', verticalalignment='center')

plt.suptitle('3D Gaussian-Only Compression - Control Group 1 (NO MBD)', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('output/3Dgaussian_only.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("对照组1: 纯高斯压缩演示完成！")
print("="*70)
print(f"实验配置:")
print(f"  - 模型类型: Gaussian-Only (无MBD结构)")
print(f"  - 高斯数量: K={model.K}")
print(f"  - 数据通道: {C} (RGB)")
print(f"  - 训练轮数: 1500 epochs")
print(f"关键结论:")
print(f"  1. 使用完整的3D高斯表示: 位置(3D) + 尺度(3D) + 旋转(四元数)")
print(f"  2. 无MBD解码器 - 直接使用高斯加权插值数据")
print(f"  3. 压缩比: {comp_ratio:.1f}:1")
print(f"  4. 重建质量: PSNR={avg_psnr:.1f}dB, SSIM={avg_ssim:.4f}")
print(f"  5. 协方差矩阵: Σ = R @ S @ S^T @ R^T（完整旋转支持）")
print("="*70)
