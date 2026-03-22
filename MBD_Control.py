import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D

#运行指令：$env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + [Environment]::GetEnvironmentVariable("Path", "Machine")
#python MBD_Control.py 2>&1

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==================== Step 1: Construct 3D test data ====================
print("Step 1: Generating 3D test data...")

def create_test_signal_3d(grid_size=32, num_channels=3):
    """
    创建3D模拟光探针的测试信号。
    中等频率设计，目标PSNR 45-50dB。
    返回: 信号 [D, H, W, C]
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

# Generate 3D test signal
grid_size = 32  # Smaller grid for 3D due to memory
ground_truth = create_test_signal_3d(grid_size)
D, H, W, C = ground_truth.shape
print(f"Generated 3D test signal size: {D}x{H}x{W}x{C}")

# Prepare training data: flatten 3D grid
x_coords = torch.linspace(0, 1, D)
y_coords = torch.linspace(0, 1, H)
z_coords = torch.linspace(0, 1, W)
X_grid, Y_grid, Z_grid = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
coords = torch.stack([X_grid.flatten(), Y_grid.flatten(), Z_grid.flatten()], dim=-1)  # [N, 3]
target_data = ground_truth.view(-1, C)  # [N, 3]

# ==================== Step 2: Implement MBD model and solver ====================
print("\nStep 2: Building MBD model and solver...")

class MBDCompressor3D(nn.Module):
    """
    Moving Basis Decomposition (MBD) 3D Compressor
    Implements core formulas from the paper for 3D data:
        c_l(x) = Σ_m φ_m(x) * c_{m,l}
        b_l(x) = Σ_n ψ_n(x) * B_{n,l}
        f̂(x) = Σ_l c_l(x) * b_l(x)
    
    This is a pure MBD baseline without Gaussian components (control group).
    """
    def __init__(self, num_bases=6, coeff_res=12, basis_res=8, data_dim=3,
                 coeff_kernel_type='gaussian', coeff_kernel_scale=0.09,
                 basis_kernel_type='gaussian', basis_kernel_scale=0.18,
                 spatial_dim=3):  # New parameter for spatial dimension
        super().__init__()
        self.L = num_bases
        self.data_dim = data_dim  # Renamed to avoid confusion with spatial_dim
        self.spatial_dim = spatial_dim  # 3D spatial coordinates
        # Coefficient kernel parameters (for C)
        self.coeff_kernel_type = coeff_kernel_type
        self.coeff_kernel_scale = coeff_kernel_scale
        # Basis kernel parameters (for B)
        self.basis_kernel_type = basis_kernel_type
        self.basis_kernel_scale = basis_kernel_scale

        # Initialize coefficient control points and basis control points (uniformly distributed in [0,1])
        # Now using 3D coordinates
        self.coeff_points = nn.Parameter(
            torch.rand(coeff_res, spatial_dim), requires_grad=False
        )  # [M, 3]
        self.basis_points = nn.Parameter(
            torch.rand(basis_res, spatial_dim), requires_grad=False
        )  # [N, 3]

        # Learnable parameters: coefficient tensor C and basis tensor B
        # C: [M, L] - scalar coefficients at coefficient control points
        # B: [N, L, D] - basis vectors at basis control points
        self.C = nn.Parameter(torch.randn(coeff_res, self.L) * 0.1)
        self.B = nn.Parameter(torch.randn(basis_res, self.L, self.data_dim) * 0.1)

        # Initialize statistics
        self.M = coeff_res
        self.N = basis_res
        print(f"MBD 3D model initialized: {self.M} coefficient points, {self.N} basis points, {self.L} bases, {spatial_dim}D spatial")

    def compute_kernel_weights(self, query_pts, control_pts, kernel_type, kernel_scale):
        """Compute kernel weights (φ_m(x) or ψ_n(x))"""
        # query_pts: [Q, 2], control_pts: [C, 2]
        if kernel_type == 'gaussian':
            # Gaussian kernel
            dist_sq = torch.cdist(query_pts, control_pts, p=2).pow(2)
            weights = torch.exp(-dist_sq / (2 * kernel_scale**2))
        elif kernel_type == 'inverse':
            # Inverse distance weights
            dist = torch.cdist(query_pts, control_pts, p=2) + 1e-8
            weights = 1.0 / (dist + 0.1)
        else:
            # Linear kernel
            dist = torch.cdist(query_pts, control_pts, p=2)
            weights = torch.relu(1.0 - dist / kernel_scale)

        # Normalize weights
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        return weights

    def forward(self, coords):
        """Forward pass: reconstruct signal from coordinates"""
        # 1. Compute kernel weights with separate kernels for C and B
        phi_weights = self.compute_kernel_weights(coords, self.coeff_points,
                                                   self.coeff_kernel_type,
                                                   self.coeff_kernel_scale)  # [Q, M]
        psi_weights = self.compute_kernel_weights(coords, self.basis_points,
                                                   self.basis_kernel_type,
                                                   self.basis_kernel_scale)   # [Q, N]

        #B
        # 2. Compute moving coefficients c_l(x) = Σ_m φ_m(x) * C_{m,l}
        moving_coeff = torch.matmul(phi_weights, self.C)  # [Q, L] matmul will dot the vector and then sum the result

        #C interpolate the basis to reconstruct the sparse control points
        # 3. Compute moving bases b_l(x) = Σ_n ψ_n(x) * B_{n,l}
        B_flat = self.B.view(-1, self.L * self.data_dim)  # [N, L*D]
        basis_interp_flat = torch.matmul(psi_weights, B_flat)  # [Q, L*D]
        #interpolate the basis vectors to get the moving basis
        moving_basis = basis_interp_flat.view(-1, self.L, self.data_dim)  # [Q, L, D]

        # 4. Reconstruct signal f̂(x) = Σ_l c_l(x) * b_l(x)
        reconstruction = torch.sum(moving_coeff.unsqueeze(-1) * moving_basis, dim=1)  # [Q, D]

        return reconstruction, moving_coeff, moving_basis

    def get_compression_ratio(self, original_size):
        """Compute compression ratio"""
        # Original data size: D*H*W*C*4 bytes (float32)
        # Compressed: M*L*4 + N*L*data_dim*4 + M*spatial_dim*4 + N*spatial_dim*4 bytes
        compressed_size = (self.M * self.L + self.N * self.L * self.data_dim + 
                          self.M * self.spatial_dim + self.N * self.spatial_dim) * 4
        ratio = original_size / compressed_size
        return ratio, compressed_size

class MBDSolver:
    """MBD Solver (simplified stochastic quasi-Newton optimization)"""
    def __init__(self, model, lambda_reg=0.01):
        self.model = model
        self.lambda_reg = lambda_reg

        # 使用Adam优化器
        self.optimizer = optim.Adam(model.parameters(), lr=0.02)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=50
        )

    def compute_loss(self, pred, target, coeff_params):
        """Compute loss function (including regularization term)"""
        # Reconstruction error
        recon_loss = torch.mean((pred - target) ** 2)

        # Frobenius norm regularization (prevents scale ambiguity)
        reg_loss = self.lambda_reg * torch.sum(coeff_params ** 2)

        # Total loss
        total_loss = recon_loss + reg_loss

        return total_loss, recon_loss, reg_loss

    def train_step(self, coords_batch, target_batch):
        """Single training step"""
        self.optimizer.zero_grad()

        # Forward pass
        pred, moving_coeff, _ = self.model(coords_batch)

        # Compute loss
        total_loss, recon_loss, reg_loss = self.compute_loss(
            pred, target_batch, self.model.C
        )

        # Backward pass
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Optimization
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'reg_loss': reg_loss.item()
        }

    def train(self, coords, target, epochs=1000, batch_size=2048):
        """Training loop"""
        print(f"Starting training, {epochs} epochs, batch_size={batch_size}")

        losses = []
        num_samples = coords.shape[0]

        for epoch in range(epochs):
            # Random batch sampling
            indices = torch.randperm(num_samples)[:batch_size]
            coords_batch = coords[indices]
            target_batch = target[indices]

            # Training step
            loss_dict = self.train_step(coords_batch, target_batch)
            losses.append(loss_dict)

            # Learning rate scheduling
            if epoch % 100 == 0:
                self.scheduler.step(loss_dict['total_loss'])

            # Print progress
            if epoch % 200 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:4d}/{epochs} | "
                      f"Total Loss: {loss_dict['total_loss']:.6f} | "
                      f"Recon: {loss_dict['recon_loss']:.6f} | "
                      f"Reg: {loss_dict['reg_loss']:.6f}")

        return losses

# ==================== 对照组2: 纯MBD模型（无3D高斯） ====================
# 创建3D模型和求解器
model = MBDCompressor3D(
    num_bases=8,      # 基的数量L（与MBD.py统一）
    coeff_res=64,     # 系数控制点数量M（与MBD.py统一）
    basis_res=64,     # 基控制点数量N（与MBD.py统一）
    data_dim=3,       # 数据维度D (RGB)
    coeff_kernel_type='gaussian',
    coeff_kernel_scale=0.15,    # 与MBD.py统一
    basis_kernel_type='gaussian',
    basis_kernel_scale=0.2,     # 与MBD.py统一
    spatial_dim=3     # 3D spatial coordinates
)

solver = MBDSolver(model, lambda_reg=1e-5)  # 与MBD.py统一正则化强度

# 计算压缩比
original_size = D * H * W * 3 * 4  # float32
comp_ratio, comp_size = model.get_compression_ratio(original_size)
print(f"Original size: {original_size/1024:.1f} KB")
print(f"Compressed: {comp_size/1024:.1f} KB")
print(f"Compression ratio: {comp_ratio:.1f}:1")

# 训练模型
print("\nStarting compression (training)...")
losses = solver.train(coords, target_data, epochs=1500, batch_size=4096)

# ==================== Step 3: Evaluation and visualization ====================
print("\nStep 3: Evaluating compression and reconstruction quality...")

# Reconstruct entire 3D volume using trained model
model.eval()
with torch.no_grad():
    reconstructed, _, _ = model(coords)
    reconstructed_vol = reconstructed.view(D, H, W, 3).cpu().numpy()
    # Clip to valid range [0, 1] to avoid imshow warnings
    reconstructed_vol = np.clip(reconstructed_vol, 0, 1)

# Compute PSNR and SSIM used for evaluation the reconstruction quality
def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def compute_ssim(img1, img2, window_size=11):
    """Compute SSIM for multi-channel images"""
    from scipy.signal import fftconvolve
    from numpy import asarray, prod

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Process multi-channel images: compute SSIM for each channel separately, then average
    if img1.ndim == 3:
        ssim_channels = []
        for c in range(img1.shape[2]):
            ssim_c = compute_ssim(img1[:, :, c], img2[:, :, c], window_size)
            ssim_channels.append(ssim_c)
        return np.mean(ssim_channels)

    # Generate Gaussian window
    gaussian = np.outer(
        np.exp(-(np.arange(window_size) - window_size//2)**2 / 1.5),
        np.exp(-(np.arange(window_size) - window_size//2)**2 / 1.5)
    )
    gaussian /= gaussian.sum()

    # Compute local statistics
    def filter_window(x):
        return fftconvolve(x, gaussian, mode='valid')

    mu1 = filter_window(img1)
    mu2 = filter_window(img2)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = filter_window(img1*img1) - mu1_sq
    sigma2_sq = filter_window(img2*img2) - mu2_sq
    sigma12 = filter_window(img1*img2) - mu1_mu2

    # SSIM formula
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(ssim_map)

# 计算指标 (3D volume metrics)
def compute_psnr_3d(vol1, vol2):
    """Compute PSNR for 3D volumes"""
    mse = np.mean((vol1 - vol2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def compute_ssim_3d(vol1, vol2, window_size=7):
    """Compute SSIM for 3D volumes (average over slices)"""
    ssim_slices = []
    for d in range(vol1.shape[0]):
        ssim_d = compute_ssim(vol1[d], vol2[d], window_size)
        ssim_slices.append(ssim_d)
    return np.mean(ssim_slices)

psnr_value = compute_psnr_3d(ground_truth.numpy(), reconstructed_vol)
ssim_value = compute_ssim_3d(ground_truth.numpy(), reconstructed_vol)

print(f"Reconstruction quality metrics (3D):")
print(f"  PSNR: {psnr_value:.2f} dB")
print(f"  SSIM: {ssim_value:.4f}")
print(f"  Final loss: {losses[-1]['total_loss']:.6f}")

# ==================== Visualization results ====================
print("\nGenerating visualization results...")

fig = plt.figure(figsize=(20, 12))

# Select middle slices for visualization
mid_slice = D // 2

# 1. Original 3D slices (XY plane at mid Z)
ax1 = plt.subplot(2, 4, 1)
ax1.imshow(ground_truth[mid_slice].numpy(), vmin=0, vmax=1)
ax1.set_title(f'Original Signal (Z={mid_slice} slice)')
ax1.set_xlabel('Y')
ax1.set_ylabel('X')
ax1.grid(False)

# 2. MBD reconstructed slice
ax2 = plt.subplot(2, 4, 2)
ax2.imshow(reconstructed_vol[mid_slice], vmin=0, vmax=1)
ax2.set_title(f'MBD 3D Reconstruction\nPSNR: {psnr_value:.1f}dB, SSIM: {ssim_value:.4f}')
ax2.set_xlabel('Y')
ax2.grid(False)

# 3. Error map for the slice
ax3 = plt.subplot(2, 4, 3)
error = np.abs(ground_truth[mid_slice].numpy() - reconstructed_vol[mid_slice])
error_img = ax3.imshow(error.mean(axis=-1), cmap='hot', vmin=0, vmax=0.2)
ax3.set_title(f'Absolute Error (Z={mid_slice} slice)')
ax3.set_xlabel('Y')
plt.colorbar(error_img, ax=ax3, fraction=0.046, pad=0.04)
ax3.grid(False)

# 4. 3D Control point distribution
ax4 = fig.add_subplot(2, 4, 4, projection='3d')
coeff_pts = model.coeff_points.detach().cpu().numpy()
basis_pts = model.basis_points.detach().cpu().numpy()
ax4.scatter(coeff_pts[:, 0], coeff_pts[:, 1], coeff_pts[:, 2],
           c='red', s=20, alpha=0.5, label=f'Coeff Points (M={model.M})')
ax4.scatter(basis_pts[:, 0], basis_pts[:, 1], basis_pts[:, 2],
           c='blue', s=40, marker='s', alpha=0.7, label=f'Basis Points (N={model.N})')
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.set_zlim(0, 1)
ax4.set_title('3D Control Point Distribution')
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_zlabel('Z')
ax4.legend(fontsize='small')

# 5. Loss curves
ax5 = plt.subplot(2, 4, 5)
total_losses = [l['total_loss'] for l in losses]
recon_losses = [l['recon_loss'] for l in losses]
reg_losses = [l['reg_loss'] for l in losses]

ax5.semilogy(total_losses, 'b-', linewidth=2, label='Total Loss')
ax5.semilogy(recon_losses, 'g--', linewidth=1.5, alpha=0.7, label='Reconstruction Loss')
ax5.semilogy(reg_losses, 'r:', linewidth=1, alpha=0.5, label='Regularization Loss')
ax5.set_title('Training Loss Curves (Log Scale)')
ax5.set_xlabel('Iterations')
ax5.set_ylabel('Loss Value')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Channel comparison (1D line through 3D volume)
ax6 = plt.subplot(2, 4, 6)
x_slice = D // 2
y_slice = H // 2
for c in range(3):
    ax6.plot(ground_truth[x_slice, y_slice, :, c].numpy(),
            color=['r', 'g', 'b'][c],
            linestyle='-', alpha=0.7, label=f'Original C{c}')
    ax6.plot(reconstructed_vol[x_slice, y_slice, :, c],
            color=['r', 'g', 'b'][c],
            linestyle='--', alpha=0.9, label=f'Reconstructed C{c}')
ax6.set_title(f'1D Line at X={x_slice}, Y={y_slice}')
ax6.set_xlabel('Z Coordinate')
ax6.set_ylabel('Intensity')
ax6.legend(loc='upper right', fontsize='small')
ax6.grid(True, alpha=0.3)

# 7. Basis vector visualization
ax7 = plt.subplot(2, 4, 7)
with torch.no_grad():
    # Get basis vectors at a 3D point
    test_point = torch.tensor([[0.5, 0.5, 0.5]])
    _, _, moving_basis = model(test_point)
    basis_vectors = moving_basis[0].cpu().numpy()  # [L, D]

    # Plot basis vectors
    colors = plt.cm.viridis(np.linspace(0, 1, model.L))
    for l in range(model.L):
        ax7.bar(range(3), basis_vectors[l], alpha=0.7,
               color=colors[l], label=f'Basis {l+1}')

ax7.set_title('Moving Basis Vectors at 3D Test Point')
ax7.set_xlabel('Channel (RGB)')
ax7.set_ylabel('Basis Vector Value')
ax7.set_xticks([0, 1, 2])
ax7.set_xticklabels(['R', 'G', 'B'])
ax7.grid(True, alpha=0.3, axis='y')

# 8. Compression info
ax8 = plt.subplot(2, 4, 8)
ax8.axis('off')
info_text = f"""
3D MBD Compression Summary
(Control Group - No Gaussian)
=============================
Original Size: {D}×{H}×{W}×{C}
Original: {original_size/1024:.1f} KB
Compressed: {comp_size/1024:.1f} KB
Compression Ratio: {comp_ratio:.1f}:1
Num Bases (L): {model.L}
Coeff Points (M): {model.M}
Basis Points (N): {model.N}
Spatial Dim: {model.spatial_dim}D
Final Loss: {losses[-1]['total_loss']:.6f}
PSNR: {psnr_value:.1f} dB
SSIM: {ssim_value:.4f}
"""
ax8.text(0.1, 0.5, info_text, fontsize=10,
        family='monospace', verticalalignment='center')

plt.suptitle('Moving Basis Decomposition (MBD) 3D - Control Group 2 (No Gaussian)', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('output/MBD_Control_3D.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nDemo completed!")
print("="*70)
print("对照组2: 纯MBD 3D压缩演示完成！")
print("="*70)
print(f"实验配置:")
print(f"  - 模型类型: MBD-Only (无3D高斯，使用固定核函数)")
print(f"  - 系数控制点: M={model.M}")
print(f"  - 基控制点: N={model.N}")
print(f"  - 基数量: L={model.L}")
print(f"  - 数据通道: {C} (RGB)")
print(f"  - 训练轮数: 1500 epochs")
print(f"关键结论:")
print(f"  1. 使用稀疏控制点 (M={model.M} + N={model.N}) 重建3D体积")
print(f"  2. 压缩比: {comp_ratio:.1f}:1")
print(f"  3. 重建质量: PSNR={psnr_value:.1f}dB, SSIM={ssim_value:.4f}")
print(f"  4. 使用固定高斯核，无可训练的位置/尺度/旋转参数")
print(f"  5. Frobenius正则化解决尺度模糊问题")
print("="*70)