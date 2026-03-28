import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from test_signal_3d import get_test_signal_by_name, get_all_test_signals

#运行指令：$env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + [Environment]::GetEnvironmentVariable("Path", "Machine")
#python MBD.py 2>&1

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==================== 辅助函数：四元数与旋转矩阵 ====================
def quaternion_to_rotation_matrix(q):
    """
    将四元数转换为旋转矩阵 (3DGS标准实现)。
    q: [..., 4] 四元数 (w, x, y, z)
    返回: [..., 3, 3] 旋转矩阵
    """
    # 归一化四元数
    q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # 构建旋转矩阵
    R = torch.stack([
        torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y], dim=-1),
        torch.stack([2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x], dim=-1),
        torch.stack([2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y], dim=-1)
    ], dim=-2)
    return R

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
grid_size = 32  # 3D网格使用较小的尺寸 (32^3 = 32768个点)
ground_truth = get_test_signal_by_name("forest_dappled", grid_size=32, num_channels=3)
D, H, W, C = ground_truth.shape
print(f"Generated 3D test signal size: {D}x{H}x{W}x{C}")

# Prepare training data: flatten 3D grid
x_coords = torch.linspace(0, 1, D)
y_coords = torch.linspace(0, 1, H)
z_coords = torch.linspace(0, 1, W)
X_grid, Y_grid, Z_grid = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
coords = torch.stack([X_grid.flatten(), Y_grid.flatten(), Z_grid.flatten()], dim=-1)  # [N, 3]
target_data = ground_truth.view(-1, C)  # [N, C]

# ==================== Step 2: Implement MBD model and solver ====================
print("\nStep 2: Building MBD model and solver...")

class MBDCompressor3D(nn.Module):
    """
    Moving Basis Decomposition (MBD) Compressor with 3D Trainable Gaussians + MLP Decoder
    Implements core formulas from the paper:
        c_l(x) = Σ_m φ_m(x) * c_{m,l}
        b_l(x) = Σ_n ψ_n(x) * B_{n,l}
        f̂(x) = MLP(Σ_l c_l(x) * b_l(x), x)

    基于3DGS论文实现完整的3D高斯表示，包含位置、尺度、旋转（四元数）。
    增加MLP解码器，将MBD重建结果与位置信息融合后输出最终SH系数。
    """
    def __init__(self, num_bases=6, coeff_res=12, basis_res=8, data_dim=3,
                 coeff_kernel_scale=0.15, basis_kernel_scale=0.2, mlp_hidden=64):
        super().__init__()
        self.L = num_bases
        self.data_dim = data_dim
        self.mlp_hidden = mlp_hidden

        # ========== Coefficient 3D Gaussian Parameters ==========
        # 位置 mu: [M, 3] - 可训练3D位置
        self.coeff_mu = nn.Parameter(torch.rand(coeff_res, 3))
        # 对数尺度 log_s: [M, 3] - 可训练各向异性尺度
        init_coeff_log_scale = np.log(coeff_kernel_scale)
        self.coeff_log_s = nn.Parameter(
            torch.ones(coeff_res, 3) * init_coeff_log_scale + torch.randn(coeff_res, 3) * 0.1
        )
        # 四元数旋转 q: [M, 4] (w, x, y, z) - 可训练旋转
        self.coeff_q = nn.Parameter(torch.zeros(coeff_res, 4))
        with torch.no_grad():
            self.coeff_q[:, 0] = 1.0  # 初始化为单位四元数 [1, 0, 0, 0]

        # ========== Basis 3D Gaussian Parameters ==========
        # 位置 mu: [N, 3]
        self.basis_mu = nn.Parameter(torch.rand(basis_res, 3))
        # 对数尺度 log_s: [N, 3]
        init_basis_log_scale = np.log(basis_kernel_scale)
        self.basis_log_s = nn.Parameter(
            torch.ones(basis_res, 3) * init_basis_log_scale + torch.randn(basis_res, 3) * 0.1
        )
        # 四元数旋转 q: [N, 4]
        self.basis_q = nn.Parameter(torch.zeros(basis_res, 4))
        with torch.no_grad():
            self.basis_q[:, 0] = 1.0

        # ========== MBD Coefficient/Basis Tensors ==========
        # C: [M, L] - scalar coefficients at coefficient control points
        # B: [N, L, D] - basis vectors at basis control points
        self.C = nn.Parameter(torch.randn(coeff_res, self.L) * 0.1)
        self.B = nn.Parameter(torch.randn(basis_res, self.L, self.data_dim) * 0.1)

        # ========== MLP Decoder ==========
        # 输入: MBD重建结果(D) + 3D位置(3) -> 隐藏层 -> 输出SH系数(D)
        self.decoder = nn.Sequential(
            nn.Linear(data_dim + 3, mlp_hidden),  # 输入: MBD重建(D) + 位置(3)
            nn.ReLU(),
            nn.Linear(mlp_hidden, data_dim)  # 输出: 最终SH系数(D)
        )

        # Initialize statistics
        self.M = coeff_res
        self.N = basis_res
        print(f"MBD+MLP model initialized with 3D Trainable Gaussians:")
        print(f"  Coefficient 3D Gaussians: M={self.M} (position + scale + rotation)")
        print(f"  Basis 3D Gaussians: N={self.N} (position + scale + rotation)")
        print(f"  Num bases: L={self.L}, Data dim: D={self.data_dim}")
        print(f"  MLP Decoder: ({self.data_dim}+3) -> {mlp_hidden} -> {mlp_hidden} -> {self.data_dim}")
        print(f"  Transform: Position(3D) + Scale(3D) + Rotation(Quaternion)")

    def gaussian_function_3d(self, p, mu, s, q):
        """
        计算位置p处的完整协方差3D高斯函数值。
        使用完整的协方差矩阵 Σ = R @ S @ S^T @ R^T

        p: [N, 3] 查询位置
        mu: [K, 3] 高斯中心
        s: [K, 3] 尺度
        q: [K, 4] 四元数旋转
        返回: [N, K] 高斯函数值
        """
        K = mu.shape[0]

        # 构建协方差矩阵的逆 (precision matrix)
        # Σ^{-1} = R @ S^{-2} @ R^T
        R = quaternion_to_rotation_matrix(q)  # [K, 3, 3]
        s_inv_sq = 1.0 / (s ** 2 + 1e-8)  # [K, 3]
        S_inv_sq = torch.diag_embed(s_inv_sq)  # [K, 3, 3]
        precision = R @ S_inv_sq @ R.transpose(-1, -2)  # [K, 3, 3] 精度矩阵

        # 计算马氏距离平方: (p - mu)^T @ Σ^{-1} @ (p - mu)
        diff = p.unsqueeze(1) - mu.unsqueeze(0)  # [N, K, 3]

        # 批量计算: diff @ precision @ diff^T
        diff_expanded = diff.unsqueeze(-1)  # [N, K, 3, 1]
        precision_expanded = precision.unsqueeze(0)  # [1, K, 3, 3]

        # (p-mu)^T @ Σ^{-1} @ (p-mu)
        mahalanobis_sq = (diff_expanded.transpose(-1, -2) @ precision_expanded @ diff_expanded).squeeze(-1).squeeze(-1)  # [N, K]

        return torch.exp(-0.5 * mahalanobis_sq)

    def compute_gaussian_weights_3d(self, query_pts, mu, log_s, q):
        """
        计算3D高斯权重（归一化）。

        query_pts: [Q, 3] - 查询位置
        mu: [K, 3] - 高斯中心位置
        log_s: [K, 3] - 对数尺度
        q: [K, 4] - 四元数旋转
        返回: [Q, K] - 归一化高斯权重
        """
        s = torch.exp(log_s)  # [K, 3]
        gaussian_vals = self.gaussian_function_3d(query_pts, mu, s, q)  # [Q, K]

        # 归一化权重
        weights = gaussian_vals / (gaussian_vals.sum(dim=1, keepdim=True) + 1e-8)
        return weights

    def forward(self, coords):
        """Forward pass: reconstruct signal from 3D coordinates with MLP decoder"""
        # 1. Compute 3D Gaussian weights for C and B
        phi_weights = self.compute_gaussian_weights_3d(
            coords, self.coeff_mu, self.coeff_log_s, self.coeff_q
        )  # [Q, M]
        psi_weights = self.compute_gaussian_weights_3d(
            coords, self.basis_mu, self.basis_log_s, self.basis_q
        )  # [Q, N]

        # 2. Compute moving coefficients c_l(x) = Σ_m φ_m(x) * C_{m,l}
        moving_coeff = torch.matmul(phi_weights, self.C)  # [Q, L]

        # 3. Compute moving bases b_l(x) = Σ_n ψ_n(x) * B_{n,l}
        B_flat = self.B.view(-1, self.L * self.data_dim)  # [N, L*D]
        basis_interp_flat = torch.matmul(psi_weights, B_flat)  # [Q, L*D]
        moving_basis = basis_interp_flat.view(-1, self.L, self.data_dim)  # [Q, L, D]

        # 4. Compute MBD reconstruction f̂_mbd(x) = Σ_l c_l(x) * b_l(x)
        mbd_reconstruction = torch.sum(moving_coeff.unsqueeze(-1) * moving_basis, dim=1)  # [Q, D]

        # 5. Apply MLP decoder: concat MBD reconstruction with position, then decode
        mlp_input = torch.cat([mbd_reconstruction, coords], dim=1)  # [Q, D+3]
        reconstruction = self.decoder(mlp_input)  # [Q, D]

        return reconstruction, moving_coeff, moving_basis, mbd_reconstruction

    def get_compression_ratio(self, original_size):
        """Compute compression ratio (including MLP parameters)"""
        # 3D高斯参数: mu(3) + log_s(3) + q(4) = 10 per gaussian
        # coeff: M*(10 + L), basis: N*(10 + L*D)
        coeff_params = self.M * (3 + 3 + 4 + self.L)
        basis_params = self.N * (3 + 3 + 4 + self.L * self.data_dim)

        # MLP参数: 2层MLP
        # Layer1: (D+3) * H + H (weights + bias)
        # Layer2: H * D + D
        H = self.mlp_hidden
        D = self.data_dim
        mlp_params = (D + 3) * H + H + H * D + D

        total_params = coeff_params + basis_params + mlp_params
        compressed_size = total_params * 4  # float32
        ratio = original_size / compressed_size
        return ratio, compressed_size

    def get_gaussian_params(self):
        """获取高斯参数用于可视化"""
        with torch.no_grad():
            coeff_mu = self.coeff_mu.cpu().numpy()
            coeff_s = torch.exp(self.coeff_log_s).cpu().numpy()
            coeff_q = self.coeff_q.cpu().numpy()
            coeff_q = coeff_q / (np.linalg.norm(coeff_q, axis=1, keepdims=True) + 1e-8)

            basis_mu = self.basis_mu.cpu().numpy()
            basis_s = torch.exp(self.basis_log_s).cpu().numpy()
            basis_q = self.basis_q.cpu().numpy()
            basis_q = basis_q / (np.linalg.norm(basis_q, axis=1, keepdims=True) + 1e-8)
        return {
            'coeff_mu': coeff_mu, 'coeff_s': coeff_s, 'coeff_q': coeff_q,
            'basis_mu': basis_mu, 'basis_s': basis_s, 'basis_q': basis_q
        }

class MBDSolver3D:
    """MBD 3D Solver with separate optimizers for Gaussian, MBD and MLP params
    支持两阶段训练：主训练阶段(所有参数) + 量化微调阶段(仅MLP)
    """
    def __init__(self, model, lambda_reg=0.01):
        self.model = model
        self.lambda_reg = lambda_reg

        # 为高斯参数、MBD参数和MLP参数使用不同的优化器
        gaussian_params = [
            self.model.coeff_mu, self.model.coeff_log_s, self.model.coeff_q,
            self.model.basis_mu, self.model.basis_log_s, self.model.basis_q
        ]
        mbd_params = [self.model.C, self.model.B]
        mlp_params = self.model.decoder.parameters()

        self.optimizer_gaussian = optim.Adam(gaussian_params, lr=0.005)
        self.optimizer_mbd = optim.Adam(mbd_params, lr=0.01)
        self.optimizer_mlp = optim.Adam(mlp_params, lr=0.005)

        self.scheduler_gaussian = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_gaussian, patience=50, factor=0.5
        )
        self.scheduler_mbd = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_mbd, patience=50, factor=0.5
        )
        self.scheduler_mlp = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_mlp, patience=50, factor=0.5
        )

    def compute_loss(self, pred, target):
        """Compute loss function"""
        # Reconstruction error (MSE)
        recon_loss = torch.mean((pred - target) ** 2)

        # Regularization: 防止尺度过大
        coeff_s = torch.exp(self.model.coeff_log_s)
        basis_s = torch.exp(self.model.basis_log_s)
        reg_loss = self.lambda_reg * (
            torch.sum(self.model.C ** 2) +
            torch.sum(coeff_s ** 2) + torch.sum(basis_s ** 2)
        )

        total_loss = recon_loss + reg_loss
        return total_loss, recon_loss, reg_loss

    def train_step(self, coords_batch, target_batch, stage='main'):
        """Single training step
        stage='main': 训练所有参数（高斯+MBD+MLP）
        stage='quant': 冻结高斯和MBD参数，只训练MLP（量化微调阶段）
        """
        if stage == 'quant':
            # 量化阶段：只训练MLP，冻结高斯和MBD参数
            self.optimizer_mlp.zero_grad()
        else:
            # 主训练阶段：训练所有参数
            self.optimizer_gaussian.zero_grad()
            self.optimizer_mbd.zero_grad()
            self.optimizer_mlp.zero_grad()

        # Forward pass
        pred, _, _, _ = self.model(coords_batch)

        # Compute loss
        total_loss, recon_loss, reg_loss = self.compute_loss(pred, target_batch)

        # Backward pass
        total_loss.backward()

        # Gradient clipping
        if stage == 'quant':
            torch.nn.utils.clip_grad_norm_(self.model.decoder.parameters(), max_norm=1.0)
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Optimization
        if stage == 'quant':
            self.optimizer_mlp.step()
        else:
            self.optimizer_gaussian.step()
            self.optimizer_mbd.step()
            self.optimizer_mlp.step()

        return {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'reg_loss': reg_loss.item()
        }

    def quantize_parameters(self, bits=10):
        """
        模拟参数量化 (论文第3.3节)。
        在实际应用中，这里会实现公式13,14的量化与反量化，并微调。
        此处为演示，仅模拟概念。
        """
        print(f"  [Sim] Quantizing parameters to {bits} bits...")
        # 在实际实现中，这里会对高斯和MBD参数进行量化并替换为整数存储
        # 随后会进行量化感知微调
        pass

    def train(self, coords, target, epochs_main=1000, epochs_quant_finetune=500, batch_size=2048):
        """Training loop with two-stage training
        Stage 1: 主训练阶段 - 训练所有参数（高斯+MBD+MLP）
        Stage 2: 量化微调阶段 - 冻结高斯和MBD参数，只训练MLP
        """
        print(f"Starting MAIN training stage ({epochs_main} epochs), batch_size={batch_size}")

        losses = []
        num_samples = coords.shape[0]

        # Stage 1: 主训练阶段
        for epoch in range(epochs_main):
            # Random batch sampling
            indices = torch.randperm(num_samples)[:batch_size]
            coords_batch = coords[indices]
            target_batch = target[indices]

            # Training step
            loss_dict = self.train_step(coords_batch, target_batch, 'main')
            losses.append(loss_dict)

            # Learning rate scheduling
            if epoch % 100 == 0:
                self.scheduler_gaussian.step(loss_dict['total_loss'])
                self.scheduler_mbd.step(loss_dict['total_loss'])
                self.scheduler_mlp.step(loss_dict['total_loss'])

            # Print progress
            if epoch % 200 == 0 or epoch == epochs_main - 1:
                print(f"  Epoch {epoch:4d}/{epochs_main} | "
                      f"Total Loss: {loss_dict['total_loss']:.6f} | "
                      f"Recon: {loss_dict['recon_loss']:.6f} | "
                      f"Reg: {loss_dict['reg_loss']:.6f}")

        # Stage 2: 量化微调阶段
        print(f"\nStarting QUANTIZATION-AWARE finetuning ({epochs_quant_finetune} epochs)...")
        print(f"  [Freeze] Gaussian and MBD parameters frozen, only training MLP decoder")
        self.quantize_parameters(bits=10)

        for epoch in range(epochs_quant_finetune):
            indices = torch.randperm(num_samples)[:batch_size]
            coords_batch = coords[indices]
            target_batch = target[indices]

            loss_dict = self.train_step(coords_batch, target_batch, 'quant')
            losses.append(loss_dict)

            if epoch % 100 == 0 or epoch == epochs_quant_finetune - 1:
                print(f"  [Quant] Epoch {epoch:4d}/{epochs_quant_finetune} | "
                      f"Loss: {loss_dict['total_loss']:.6f}")

        return losses

# ==================== 对照组: MBD + 3D可训练高斯 + MLP解码器 ====================
# 创建3D模型和求解器（使用可训练3D高斯 + MLP解码器）
model = MBDCompressor3D(
    num_bases=2,      # 基的数量L（与MBD_Control.py统一）
    coeff_res=64,     # 系数3D高斯数量M（与MBD_Control.py统一）
    basis_res=64,     # 基3D高斯数量N（与MBD_Control.py统一）
    data_dim=C,       # 数据维度D (RGB)
    coeff_kernel_scale=0.15,   # 初始尺度（与MBD_Control.py统一）
    basis_kernel_scale=0.20,    # 初始尺度（与MBD_Control.py统一）
    mlp_hidden=32     # MLP隐藏层大小（与gaussian.py统一）
)

solver = MBDSolver3D(model, lambda_reg=1e-5)  # 与其他对照组统一正则化强度

# 计算压缩比
original_size = D * H * W * C * 4  # float32
comp_ratio, comp_size = model.get_compression_ratio(original_size)
print(f"Original size: {original_size/1024:.1f} KB")
print(f"Compressed: {comp_size/1024:.1f} KB")
print(f"Compression ratio: {comp_ratio:.1f}:1")

# 训练模型（两阶段训练：1200 main + 300 quant finetune，与gaussian.py统一）
print("\nStarting 3D compression (two-stage training)...")
losses = solver.train(coords, target_data, epochs_main=2500, epochs_quant_finetune=0, batch_size=4096)

# ==================== Step 3: Evaluation and visualization ====================
print("\nStep 3: Evaluating compression and reconstruction quality...")

# Reconstruct entire 3D volume using trained model
model.eval()
with torch.no_grad():
    reconstructed, _, _, mbd_recon = model(coords)
    reconstructed_vol = reconstructed.view(D, H, W, C).cpu().numpy()
    mbd_recon_vol = mbd_recon.view(D, H, W, C).cpu().numpy()  # MBD重建（无MLP）
    # Clip to valid range [0, 1] to avoid imshow warnings
    reconstructed_vol = np.clip(reconstructed_vol, 0, 1)
    mbd_recon_vol = np.clip(mbd_recon_vol, 0, 1)

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

# 计算指标（对每个通道计算平均PSNR）
psnr_values = []
ssim_values = []
z_slice = D // 2  # 取Z中间切片进行评估
for c in range(C):
    gt_slice = ground_truth[z_slice, :, :, c].numpy()
    rec_slice = reconstructed_vol[z_slice, :, :, c]
    psnr_c = compute_psnr(gt_slice, rec_slice)
    ssim_c = compute_ssim(gt_slice, rec_slice)
    psnr_values.append(psnr_c)
    ssim_values.append(ssim_c)

psnr_value = np.mean(psnr_values)
ssim_value = np.mean(ssim_values)

print(f"Reconstruction quality metrics (Z={z_slice} slice):")
print(f"  Avg PSNR: {psnr_value:.2f} dB")
print(f"  Avg SSIM: {ssim_value:.4f}")
print(f"  Final loss: {losses[-1]['total_loss']:.6f}")

# ==================== Visualization results (3D) ====================
print("\nGenerating 3D visualization results...")

# 取中间切片进行可视化
gt_slice = ground_truth[z_slice, :, :, :].numpy()  # [H, W, C]
rec_slice = reconstructed_vol[z_slice, :, :, :]  # [H, W, C]

fig = plt.figure(figsize=(20, 14))

# 1. 原始信号图 (Z中间切片)
ax1 = plt.subplot(2, 4, 1)
im1 = ax1.imshow(gt_slice, vmin=0, vmax=1)
ax1.set_title(f'Original Signal (Z={z_slice} slice)\n3D Volume: {D}x{H}x{W}x{C}')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.grid(False)

# 2. MBD 3D Gaussian重建图
ax2 = plt.subplot(2, 4, 2)
im2 = ax2.imshow(rec_slice, vmin=0, vmax=1)
ax2.set_title(f'MBD+MLP+3D Gaussian Reconstruction\nPSNR: {psnr_value:.1f}dB, Ratio: {comp_ratio:.1f}:1')
ax2.set_xlabel('X')
ax2.grid(False)

# 3. 误差图
ax3 = plt.subplot(2, 4, 3)
error = np.abs(gt_slice - rec_slice)
error_img = ax3.imshow(error.mean(axis=-1), cmap='hot', vmin=0, vmax=0.15)
ax3.set_title(f'Reconstruction Error\n Avg SSIM: {ssim_value:.4f}')
ax3.set_xlabel('X')
plt.colorbar(error_img, ax=ax3, fraction=0.046, pad=0.04)
ax3.grid(False)

# 4. 3D高斯分布位置（用椅球体表示）
ax4 = fig.add_subplot(2, 4, 4, projection='3d')

gaussian_params = model.get_gaussian_params()
coeff_mu = gaussian_params['coeff_mu']
coeff_s = gaussian_params['coeff_s']
coeff_q = gaussian_params['coeff_q']
basis_mu = gaussian_params['basis_mu']
basis_s = gaussian_params['basis_s']
basis_q = gaussian_params['basis_q']

# 绘制高斯中心点
ax4.scatter(coeff_mu[:, 0], coeff_mu[:, 1], coeff_mu[:, 2],
            c='red', s=25, alpha=0.7, label='Coeff Centers')
ax4.scatter(basis_mu[:, 0], basis_mu[:, 1], basis_mu[:, 2],
            c='blue', s=35, marker='s', alpha=0.7, label='Basis Centers')

# 绘制椅球体表示每个高斯的协方差形状
def draw_ellipsoid(ax, center, scale, quaternion, n_points=12, alpha=0.1, color='blue'):
    """在3D坐标系中绘制椅球体"""
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points // 2)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    # 应用尺度变换
    x = x * scale[0]
    y = y * scale[1]
    z = z * scale[2]

    # 应用旋转变换 (四元数转旋转矩阵)
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

    ax.plot_surface(x_rot, y_rot, z_rot, alpha=alpha, color=color, linewidth=0)

# 选择部分高斯绘制椅球（避免过多重叠）
num_show = min(32, model.M)
indices_coeff = np.linspace(0, model.M - 1, num_show, dtype=int)
for idx in indices_coeff:
    draw_ellipsoid(ax4, coeff_mu[idx], coeff_s[idx], coeff_q[idx], color='red', alpha=0.08)

indices_basis = np.linspace(0, model.N - 1, num_show, dtype=int)
for idx in indices_basis:
    draw_ellipsoid(ax4, basis_mu[idx], basis_s[idx], basis_q[idx], color='blue', alpha=0.08)

ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.set_zlim(0, 1)
ax4.set_title(f'3D Gaussian Ellipsoids\nM={model.M} coeff, N={model.N} basis')
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_zlabel('Z')
ax4.legend(fontsize='small', loc='upper left')

# 5. 训练损失曲线
ax5 = plt.subplot(2, 4, 5)
total_losses = [l['total_loss'] for l in losses]
recon_losses = [l['recon_loss'] for l in losses]
reg_losses = [l['reg_loss'] for l in losses]

ax5.semilogy(total_losses, 'b-', linewidth=2, label='Total Loss')
ax5.semilogy(recon_losses, 'g--', linewidth=1.5, alpha=0.7, label='Reconstruction Loss')
ax5.semilogy(reg_losses, 'r:', linewidth=1, alpha=0.5, label='Regularization Loss')
ax5.axvline(x=1500, color='orange', linestyle=':', alpha=0.7, label='Quant Finetune Start')
ax5.set_title('Training Loss Curves (Log Scale)')
ax5.set_xlabel('Iterations')
ax5.set_ylabel('Loss Value')
ax5.legend(fontsize='x-small')
ax5.grid(True, alpha=0.3)

# 6. 训练数据拟合度对比曲线 (Z中间切片的一条线)
ax6 = plt.subplot(2, 4, 6)
y_line = H // 2
channel_colors = ['#E74C3C', '#27AE60', '#3498DB']
channel_names = ['R', 'G', 'B']

for c in range(C):
    ax6.plot(gt_slice[y_line, :, c],
            color=channel_colors[c], linestyle='-', alpha=0.7, linewidth=1.5,
            label=f'Original {channel_names[c]}')
    ax6.plot(rec_slice[y_line, :, c],
            color=channel_colors[c], linestyle='--', alpha=0.9, linewidth=1.5,
            label=f'Recon {channel_names[c]}')

ax6.set_title(f'Training Data Fitting (Z={z_slice}, Y={y_line} slice)')
ax6.set_xlabel('X Coordinate')
ax6.set_ylabel('Value')
ax6.legend(loc='upper right', fontsize='x-small', ncol=2)
ax6.grid(True, alpha=0.3)

# 7. 旋转角度分布 (四元数转换为角度)
ax7 = plt.subplot(2, 4, 7)

coeff_q_norm = coeff_q / (np.linalg.norm(coeff_q, axis=1, keepdims=True) + 1e-8)
basis_q_norm = basis_q / (np.linalg.norm(basis_q, axis=1, keepdims=True) + 1e-8)
coeff_angles = 2 * np.arccos(np.clip(coeff_q_norm[:, 0], -1, 1)) * 180 / np.pi
basis_angles = 2 * np.arccos(np.clip(basis_q_norm[:, 0], -1, 1)) * 180 / np.pi

ax7.hist(coeff_angles, bins=25, alpha=0.6, color='red', edgecolor='darkred', label='Coeff')
ax7.hist(basis_angles, bins=25, alpha=0.6, color='blue', edgecolor='darkblue', label='Basis')
ax7.axvline(x=coeff_angles.mean(), color='red', linestyle='--', linewidth=1.5,
            label=f'Coeff Mean: {coeff_angles.mean():.1f}°')
ax7.axvline(x=basis_angles.mean(), color='blue', linestyle='--', linewidth=1.5,
            label=f'Basis Mean: {basis_angles.mean():.1f}°')
ax7.set_title('Rotation Angle Distribution')
ax7.set_xlabel('Rotation Angle (degrees)')
ax7.set_ylabel('Count')
ax7.legend(fontsize='x-small')
ax7.grid(True, alpha=0.3)

# 8. 综合信息：Transform统计
ax8 = plt.subplot(2, 4, 8)
ax8.axis('off')
num_probes = D * H * W

# Scale各向异性分析
coeff_scale_ratios = coeff_s.max(axis=1) / (coeff_s.min(axis=1) + 1e-8)
basis_scale_ratios = basis_s.max(axis=1) / (basis_s.min(axis=1) + 1e-8)

info_text = f"""
MBD + 3D Gaussian + MLP Compression Summary
============================================
Original Data:
  Volume: {D}x{H}x{W}x{C} = {num_probes} points
  Size: {original_size/1024:.1f} KB

Compressed Model:
  Coeff 3D Gaussians (M): {model.M}
  Basis 3D Gaussians (N): {model.N}
  Num Bases (L): {model.L}
  MLP Hidden: {model.mlp_hidden}
  Est. Size: {comp_size/1024:.1f} KB
  Ratio: {comp_ratio:.1f}:1

Coeff Transform Stats:
  Scale: {coeff_s.mean():.4f} ± {coeff_s.std():.4f}
  Anisotropy: {coeff_scale_ratios.mean():.2f} ± {coeff_scale_ratios.std():.2f}
  Rotation: {coeff_angles.mean():.1f}° ± {coeff_angles.std():.1f}°

Basis Transform Stats:
  Scale: {basis_s.mean():.4f} ± {basis_s.std():.4f}
  Anisotropy: {basis_scale_ratios.mean():.2f} ± {basis_scale_ratios.std():.2f}
  Rotation: {basis_angles.mean():.1f}° ± {basis_angles.std():.1f}°

Reconstruction Quality:
  Avg PSNR: {psnr_value:.1f} dB
  Avg SSIM: {ssim_value:.4f}
  Final Loss: {losses[-1]['total_loss']:.6f}
"""
ax8.text(0.02, 0.5, info_text, fontsize=8,
        family='monospace', verticalalignment='center')

plt.suptitle('MBD + 3D Gaussians + MLP Decoder - Control Group (Full Model)', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

print("\nDemo completed!")
print("="*70)
print("对照组: MBD + 3D可训练高斯 + MLP解码器演示完成！")
print("="*70)
print(f"实验配置:")
print(f"  - 模型类型: MBD + 3D Trainable Gaussians + MLP Decoder")
print(f"  - 系数3D高斯: M={model.M}")
print(f"  - 基3D高斯: N={model.N}")
print(f"  - 基数量: L={model.L}")
print(f"  - MLP隐藏层: {model.mlp_hidden}")
print(f"  - 数据通道: {C} (RGB)")
print(f"  - 训练轮数: 1500 main + 300 quant finetune")
print(f"关键结论:")
print(f"  1. 3D高斯参数: 位置(3D) + 尺度(3D) + 旋转(四元数)")
print(f"  2. MLP解码器: MBD重建({C}) + pos(3) -> H({model.mlp_hidden}) -> RGB({C})")
print(f"  3. 两阶段训练: 主训练(所有参数) + 量化微调(仅MLP)")
print(f"  4. 压缩比: {comp_ratio:.1f}:1")
print(f"  5. 重建质量: PSNR={psnr_value:.1f}dB, SSIM={ssim_value:.4f}")
print(f"  6. Coeff旋转: {coeff_angles.mean():.1f}° ± {coeff_angles.std():.1f}°")
print(f"  7. Basis旋转: {basis_angles.mean():.1f}° ± {basis_angles.std():.1f}°")
print("="*70)
