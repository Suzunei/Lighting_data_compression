import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def build_covariance_matrix(s, q):
    """
    从尺度和旋转构建协方差矩阵 Σ = R @ S @ S^T @ R^T
    s: [K, 3] 尺度
    q: [K, 4] 四元数
    返回: [K, 3, 3] 协方差矩阵
    """
    R = quaternion_to_rotation_matrix(q)  # [K, 3, 3]
    S = torch.diag_embed(s)  # [K, 3, 3] 对角尺度矩阵
    # Σ = R @ S @ S^T @ R^T = R @ S^2 @ R^T
    cov = R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)
    return cov

# ==================== 第1步：构建测试数据 (3D) ====================
print("Step 1: Generating 3D test data (simulating SH coefficients at probe positions)...")

def create_test_signal_3d(grid_size=32, num_channels=9):
    """
    创建3D模拟光探针球谐（SH）系数的测试信号。
    中等频率设计，目标PSNR 45-50dB。
    用9通道模拟2阶球谐系数(L=0,1,2共计9个系数)。
    返回: 信号 [D, H, W, 9]，模拟SH系数
    """
    x = torch.linspace(-1, 1, grid_size)
    y = torch.linspace(-1, 1, grid_size)
    z = torch.linspace(-1, 1, grid_size)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    R = torch.sqrt(X**2 + Y**2 + Z**2 + 1e-8)
    
    signal = torch.zeros(grid_size, grid_size, grid_size, num_channels)
    
    # === L=0 (1个系数): 环境光 + 软阴影 ===
    signal[..., 0] = 0.5 + 0.2 * torch.cos(np.pi * R * 0.8)
    signal[..., 0] += 0.1 * torch.sin(1.5 * np.pi * X) * torch.cos(1.2 * np.pi * Y)
    signal[..., 0] += 0.06 * torch.sin(2.0 * np.pi * Z) * torch.cos(1.8 * np.pi * X)
    
    # === L=1 (3个系数): 方向性光照 ===
    signal[..., 1] = 0.5 + 0.18 * X * torch.cos(1.0 * np.pi * Y)
    signal[..., 1] += 0.08 * torch.sin(1.8 * np.pi * X) * torch.cos(1.5 * np.pi * Z)
    
    signal[..., 2] = 0.5 + 0.18 * Y * torch.cos(1.2 * np.pi * Z)
    signal[..., 2] += 0.08 * torch.sin(1.6 * np.pi * Y) * torch.sin(1.4 * np.pi * X)
    
    signal[..., 3] = 0.5 + 0.15 * Z * torch.sin(1.0 * np.pi * (X + Y))
    signal[..., 3] += 0.06 * torch.sin(2.0 * np.pi * Z) * torch.cos(1.8 * np.pi * Y)
    
    # === L=2 (5个系数): 二阶球谐，中频细节 ===
    signal[..., 4] = 0.5 + 0.12 * (X**2 - Y**2) * torch.cos(1.5 * np.pi * R)
    signal[..., 4] += 0.06 * torch.sin(2.2 * np.pi * X) * torch.cos(1.8 * np.pi * Y)
    
    signal[..., 5] = 0.5 + 0.12 * X * Y * torch.sin(1.2 * np.pi * R)
    signal[..., 5] += 0.06 * torch.sin(2.0 * np.pi * Y) * torch.cos(1.6 * np.pi * Z)
    
    signal[..., 6] = 0.5 + 0.1 * (2*Z**2 - X**2 - Y**2) * torch.cos(1.0 * np.pi * X)
    signal[..., 6] += 0.06 * torch.sin(1.8 * np.pi * Z) * torch.sin(1.5 * np.pi * (X + Y))
    
    signal[..., 7] = 0.5 + 0.1 * X * Z * torch.sin(1.5 * np.pi * Y)
    signal[..., 7] += 0.05 * torch.sin(2.0 * np.pi * X) * torch.cos(1.8 * np.pi * Z)
    
    signal[..., 8] = 0.5 + 0.1 * Y * Z * torch.sin(1.2 * np.pi * X)
    signal[..., 8] += 0.05 * torch.sin(1.8 * np.pi * Y) * torch.cos(1.5 * np.pi * Z)
    
    # 将信号值限制在合理范围
    for c in range(num_channels):
        signal[..., c] = torch.clamp(signal[..., c], 0.1, 0.9)
    
    return signal

# 生成3D测试信号（模拟在3D网格上均匀分布的光探针SH系数）
grid_size = 32  # 3D网格使用较小的尺寸 (32^3 = 32768个探针)
ground_truth_SH = create_test_signal_3d(grid_size)  # 模拟的SH系数 [D,H,W,C]
D, H, W, C = ground_truth_SH.shape
print(f"Generated 3D simulated SH coefficient volume size: {D}x{H}x{W}x{C}")

# 准备训练数据：展平3D网格，每个点代表一个探针的位置和其SH系数
x_coords = torch.linspace(0, 1, D)
y_coords = torch.linspace(0, 1, H)
z_coords = torch.linspace(0, 1, W)
X_grid, Y_grid, Z_grid = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
probe_positions = torch.stack([X_grid.flatten(), Y_grid.flatten(), Z_grid.flatten()], dim=-1)  # [N, 3] 探针位置
probe_SH_data = ground_truth_SH.view(-1, C)  # [N, C] 探针SH系数（模拟值）

# ==================== 第2步：实现论文中的高斯压缩模型 ====================
print("\nStep 2: Building Gaussian Compression Model (Gaussian-MLP Hybrid)...")

class GaussianProbeCompressor3D(nn.Module):
    """
    3D高斯探针压缩器 (3D Gaussian Probe Compressor)
    基于3DGS论文实现完整的3D高斯表示，包含位置、尺度、旋转。
    核心：使用完整协方差矩阵的3D高斯函数拟合探针的潜在特征。
    """
    def __init__(self, num_gaussians=64, latent_dim=9, sh_dim=9, mlp_hidden=64):
        """
        参数:
            num_gaussians (K): 高斯函数的数量
            latent_dim (D): 每个高斯函数携带的潜在特征维度
            sh_dim (C): 球谐系数的维度
            mlp_hidden: 解码器MLP隐藏层大小
        """
        super().__init__()
        self.K = num_gaussians
        self.D = latent_dim
        self.C = sh_dim
        
        # ========== 3D高斯函数参数（可学习） ==========
        # 位置 mu: [K, 3] (3D位置)
        self.mu = nn.Parameter(torch.rand(num_gaussians, 3))
        
        # 对数尺度 log_s: [K, 3] (使用对数确保为正)
        # 初始化为较大值，让高斯覆盖范围更广
        self.log_s = nn.Parameter(torch.randn(num_gaussians, 3) * 0.3 - 1.5)
        
        # 四元数旋转 q: [K, 4] (w, x, y, z)
        # 初始化为单位四元数 [1, 0, 0, 0]
        self.q = nn.Parameter(torch.zeros(num_gaussians, 4))
        with torch.no_grad():
            self.q[:, 0] = 1.0  # w=1, x=y=z=0 表示无旋转
        
        # 潜在特征 F: [K, D]
        self.F = nn.Parameter(torch.randn(num_gaussians, latent_dim) * 0.1)
        
        # 解码器MLP: 输入为潜在特征+3D位置，输出为SH系数
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 3, mlp_hidden),  # 输入: F(p) 和 3D位置 p
            nn.ReLU(),
            nn.Linear(mlp_hidden, sh_dim)  # 输出: 重建的SH系数
        )
        
        # 初始化高斯位置：随机采样探针位置
        with torch.no_grad():
            indices = torch.randperm(probe_positions.shape[0])[:num_gaussians]
            self.mu.copy_(probe_positions[indices])
        
        print(f"3D-GPC model initialized: K={num_gaussians} gaussians, D={latent_dim} latent dim, SH dim={sh_dim}")
        print(f"  Transforms: Position(3D) + Scale(3D) + Rotation(Quaternion)")

    def gaussian_function_3d(self, p, mu_j, s_j, q_j):
        """
        计算位置p处的完整协方差3D高斯函数值 G_j(p)。
        使用完整的协方差矩阵 Σ = R @ S @ S^T @ R^T
        
        p: [N, 3] 探针位置
        mu_j: [K, 3] 高斯中心
        s_j: [K, 3] 尺度
        q_j: [K, 4] 四元数旋转
        返回: [N, K] 高斯函数值
        """
        N = p.shape[0]
        K = mu_j.shape[0]
        
        # 构建协方差矩阵的逆 (precision matrix)
        # Σ^{-1} = R @ S^{-2} @ R^T
        R = quaternion_to_rotation_matrix(q_j)  # [K, 3, 3]
        s_inv_sq = 1.0 / (s_j ** 2 + 1e-8)  # [K, 3]
        S_inv_sq = torch.diag_embed(s_inv_sq)  # [K, 3, 3]
        precision = R @ S_inv_sq @ R.transpose(-1, -2)  # [K, 3, 3] 精度矩阵
        
        # 计算马氏距离平方: (p - mu)^T @ Σ^{-1} @ (p - mu)
        diff = p.unsqueeze(1) - mu_j.unsqueeze(0)  # [N, K, 3]
        
        # 批量计算: diff @ precision @ diff^T
        # diff: [N, K, 3] -> [N, K, 3, 1]
        # precision: [K, 3, 3] -> [1, K, 3, 3]
        diff_expanded = diff.unsqueeze(-1)  # [N, K, 3, 1]
        precision_expanded = precision.unsqueeze(0)  # [1, K, 3, 3]
        
        # (p-mu)^T @ Σ^{-1} @ (p-mu)
        mahalanobis_sq = (diff_expanded.transpose(-1, -2) @ precision_expanded @ diff_expanded).squeeze(-1).squeeze(-1)  # [N, K]
        
        return torch.exp(-0.5 * mahalanobis_sq)

    def compute_influence_radius(self, s_j):
        """
        计算高斯函数的影响半径 R(G_j)。
        使用最大尺度 * 3 来近似99%置信区域。
        s_j: [K, 3]
        返回: [K]
        """
        return 3 * s_j.max(dim=1)[0]  # 3 * max(s_x, s_y, s_z)

    def forward(self, probe_pos):
        """
        前向传播。
        输入: probe_pos [N, 3] 3D探针位置
        返回: 
            reconstructed_SH [N, C]: 重建的SH系数
            latent_features [N, D]: 探针的潜在特征F(p)
            weights [N, K]: 高斯加权权重
        """
        N = probe_pos.shape[0]
        s = torch.exp(self.log_s)  # 获取实际尺度 [K, 3]
        
        # 1. 计算每个3D高斯函数在探针位置的值（使用完整协方差矩阵）
        gaussian_vals = self.gaussian_function_3d(probe_pos, self.mu, s, self.q)  # [N, K]
        
        # 2. 计算每个高斯函数的影响半径，并掩码掉影响范围外的
        radii = self.compute_influence_radius(s)  # [K]
        dist = torch.norm(probe_pos.unsqueeze(1) - self.mu.unsqueeze(0), dim=2)  # [N, K]
        mask = dist < radii.unsqueeze(0)  # [N, K] 布尔掩码
        gaussian_vals = gaussian_vals * mask.float()
        
        # 3. 计算每个探针的潜在特征 F(p) = Σ_j F_j * G_j(p)
        weights = gaussian_vals / (gaussian_vals.sum(dim=1, keepdim=True) + 1e-8)
        latent_features = torch.matmul(weights, self.F)  # [N, D]
        
        # 4. 将潜在特征 F(p) 与 3D位置 p 拼接，输入MLP解码器
        mlp_input = torch.cat([latent_features, probe_pos], dim=1)  # [N, D+3]
        reconstructed_SH = self.decoder(mlp_input)  # [N, C]
        
        return reconstructed_SH, latent_features, weights

    def get_compression_ratio(self, original_size_bytes):
        """
        计算压缩比。
        原始数据: N * C * 4 字节 (float32 SH系数)
        压缩后: 高斯参数(K*(3+3+4+D)*4) + MLP参数 * 4
        """
        # 3D高斯参数: mu(3), log_s(3), q(4), F(D) -> 每个高斯 (10+D)个float32
        gaussian_params_size = self.K * (3 + 3 + 4 + self.D) * 4
        
        # MLP参数: 2层MLP (D+3)*H + H + H*C + C
        H = self.decoder[0].out_features
        mlp_params = ((self.D + 3) * H + H + H * self.C + self.C)
        mlp_size = mlp_params * 4
        
        compressed_size_bytes = gaussian_params_size + mlp_size
        ratio = original_size_bytes / compressed_size_bytes
        return ratio, compressed_size_bytes
    
    def get_gaussian_params(self):
        """获取高斯参数用于可视化"""
        with torch.no_grad():
            mu = self.mu.cpu().numpy()
            s = torch.exp(self.log_s).cpu().numpy()
            q = self.q.cpu().numpy()
            # 归一化四元数
            q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
        return mu, s, q

class GaussianCompressionSolver3D:
    """3D高斯压缩求解器，实现两阶段训练（支持四元数旋转）"""
    def __init__(self, model, lambda_reg=1e-4):
        self.model = model
        self.lambda_reg = lambda_reg
        
        # 为高斯参数和MLP使用不同的优化器
        # 注意：现在包含四元数旋转参数 q
        gaussian_params = [self.model.mu, self.model.log_s, self.model.q, self.model.F]
        mlp_params = self.model.decoder.parameters()
        
        self.optimizer_gaussian = optim.Adam(gaussian_params, lr=0.005)
        self.optimizer_mlp = optim.Adam(mlp_params, lr=0.005)
        
        self.scheduler_gaussian = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_gaussian, patience=50, factor=0.5)
        self.scheduler_mlp = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_mlp, patience=50, factor=0.5)

    def compute_loss(self, pred_SH, target_SH):
        """计算损失函数 (论文公式8，均方误差MSE)"""
        mse_loss = torch.mean((pred_SH - target_SH) ** 2)
        
        # 可选：添加高斯参数的正则项（防止尺度/特征值过大）
        reg_loss = self.lambda_reg * (torch.sum(self.model.F ** 2) + torch.sum(torch.exp(self.model.log_s) ** 2))
        
        total_loss = mse_loss + reg_loss
        return total_loss, mse_loss, reg_loss

    def train_step(self, pos_batch, sh_batch, stage='main'):
        """单次训练步骤"""
        self.optimizer_gaussian.zero_grad()
        self.optimizer_mlp.zero_grad()
        
        pred_SH, _, _ = self.model(pos_batch)
        total_loss, mse_loss, reg_loss = self.compute_loss(pred_SH, sh_batch)
        
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer_gaussian.step()
        self.optimizer_mlp.step()
        
        return {
            'total_loss': total_loss.item(),
            'mse_loss': mse_loss.item(),
            'reg_loss': reg_loss.item()
        }

    def quantize_parameters(self, bits=10):
        """
        模拟参数量化 (论文第3.3节)。
        在实际应用中，这里会实现公式13,14的量化与反量化，并微调。
        此处为演示，仅模拟概念。
        """
        print(f"  [Sim] Quantizing parameters to {bits} bits...")
        # 在实际实现中，这里会对 mu, s, F 进行量化并替换为整数存储
        # 随后会进行量化感知微调
        pass

    def train(self, probe_pos, probe_sh, epochs_main=500, epochs_quant_finetune=100, batch_size=2048):
        """训练循环，模拟两阶段训练"""
        num_samples = probe_pos.shape[0]
        losses = []
        
        print(f"Starting MAIN training stage ({epochs_main} epochs)...")
        for epoch in range(epochs_main):
            indices = torch.randperm(num_samples)[:batch_size]
            pos_batch = probe_pos[indices]
            sh_batch = probe_sh[indices]
            
            loss_dict = self.train_step(pos_batch, sh_batch, 'main')
            losses.append(loss_dict)
            
            if epoch % 100 == 0 or epoch == epochs_main - 1:
                print(f"  Epoch {epoch:4d} | Loss: {loss_dict['total_loss']:.6f} | MSE: {loss_dict['mse_loss']:.6f}")
        
        # 模拟量化阶段
        print(f"\nStarting QUANTIZATION-AWARE finetuning ({epochs_quant_finetune} epochs)...")
        self.quantize_parameters(bits=10)
        for epoch in range(epochs_quant_finetune):
            indices = torch.randperm(num_samples)[:batch_size]
            pos_batch = probe_pos[indices]
            sh_batch = probe_sh[indices]
            # 量化训练阶段，可以冻结高斯参数，只训练MLP（论文策略之一）
            loss_dict = self.train_step(pos_batch, sh_batch, 'quant')
            losses.append(loss_dict)
        
        return losses

# 创建3D高斯压缩模型
num_gaussians = 256  # 高斯数量
model = GaussianProbeCompressor3D(
    num_gaussians=num_gaussians,
    latent_dim=12,    # 潜在特征维度
    sh_dim=9,         # 2阶球谐系数维度
    mlp_hidden=32     # MLP隐藏层
)
# 3D高斯参数: 位置(3) + 尺度(3) + 四元数(4) + 潜在特征(D)

# 计算压缩比 (9通道SH系数)
original_size_bytes = probe_positions.shape[0] * C * 4  # N * C * sizeof(float32)
comp_ratio, comp_size = model.get_compression_ratio(original_size_bytes)
print(f"Original data size: {original_size_bytes/1024:.1f} KB")
print(f"Compressed model size: ~{comp_size/1024:.1f} KB")
print(f"Estimated compression ratio: {comp_ratio:.1f}:1")
print(f"Number of Gaussians: {model.K}")

# 训练模型
solver = GaussianCompressionSolver3D(model, lambda_reg=1e-7)
losses = solver.train(probe_positions, probe_SH_data, epochs_main=2500, epochs_quant_finetune=500, batch_size=4096)

# ==================== 第3步：评估与可视化 ====================
print("\nStep 3: Evaluating Gaussian Compression and Reconstruction...")

# 使用训练好的模型重建整个3D体积
model.eval()
with torch.no_grad():
    reconstructed_SH, latent_features, _ = model(probe_positions)
    reconstructed_vol = reconstructed_SH.view(D, H, W, C).cpu().numpy()
    latent_vol = latent_features[:, 0].view(D, H, W).cpu().numpy()  # 可视化第一个潜在通道

# 计算重建质量指标 (PSNR, SSIM)
def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def compute_ssim(img1, img2):
    """简化的SSIM计算（单通道）"""
    C1, C2 = (0.01 * 1.0)**2, (0.03 * 1.0)**2
    mu1, mu2 = img1.mean(), img2.mean()
    sigma1_sq, sigma2_sq = img1.var(), img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    ssim = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim

# 对每个通道计算PSNR，然后平均
psnr_values = []
ssim_values = []
for c in range(C):
    psnr_c = compute_psnr(ground_truth_SH[..., c].numpy(), reconstructed_vol[..., c])
    ssim_c = compute_ssim(ground_truth_SH[..., c].numpy().flatten(), reconstructed_vol[..., c].flatten())
    psnr_values.append(psnr_c)
    ssim_values.append(ssim_c)

avg_psnr = np.mean(psnr_values)
avg_ssim = np.mean(ssim_values)

print(f"\nReconstruction quality metrics (per-channel avg):")
print(f"  PSNR: {avg_psnr:.2f} dB")
print(f"  SSIM: {avg_ssim:.4f}")
print(f"  Final training loss: {losses[-1]['total_loss']:.6f}")

# ==================== 可视化结果 (3D版本) ====================
print("\nGenerating 3D visualization results...")

# 取中间切片进行可视化
z_slice = D // 2
gt_slice = ground_truth_SH[z_slice, :, :, :].numpy()  # [H, W, C]
rec_slice = reconstructed_vol[z_slice, :, :, :]  # [H, W, C]
latent_slice = latent_vol[z_slice, :, :]  # [H, W]

fig = plt.figure(figsize=(20, 14))

# 1. 原始SH系数图 (取前3通道用于RGB显示) - Z中间切片
ax1 = plt.subplot(2, 4, 1)
im1 = ax1.imshow(gt_slice[..., :3], vmin=0, vmax=1)
ax1.set_title(f'Original SH (Z={z_slice} slice)\n{C} channels, 3D volume {D}x{H}x{W}')
#ax1.set_xlabel('X index')
ax1.set_ylabel('Y index')
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

# 2. 高斯压缩重建图 (取前3通道)
ax2 = plt.subplot(2, 4, 2)
im2 = ax2.imshow(rec_slice[..., :3], vmin=0, vmax=1)
ax2.set_title(f'3D Gaussian Reconstruction\nAvg PSNR: {avg_psnr:.1f} dB')
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

# 3. 误差图 (所有通道平均)
ax3 = plt.subplot(2, 4, 3)
error = np.abs(gt_slice - rec_slice).mean(axis=-1)
im3 = ax3.imshow(error, cmap='hot', vmin=0, vmax=0.1)
ax3.set_title(f'Reconstruction Error\nAvg SSIM: {avg_ssim:.4f}')
plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

# 4. 3D高斯分布位置（以尺度表示）
ax4 = fig.add_subplot(2, 4, 4, projection='3d')
mu, s, q = model.get_gaussian_params()

# 绘制高斯中心点
ax4.scatter(mu[:, 0], mu[:, 1], mu[:, 2], 
            c='black', s=20, alpha=0.8, label='Gaussian Centers')

# 绘制椭球体表示每个高斯的协方差形状
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_ellipsoid(ax, center, scale, quaternion, n_points=20, alpha=0.15, color='blue'):
    """
    在3D坐标系中绘制椭球体，展示scale和rotation
    center: [3] 中心位置
    scale: [3] 三个方向的尺度 (s_x, s_y, s_z)
    quaternion: [4] 四元数 (w, x, y, z)
    """
    # 生成单位球面的参数坐标
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points // 2)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    # 应用尺度变换 -> 椭球
    x = x * scale[0]
    y = y * scale[1]
    z = z * scale[2]
    
    # 应用旋转变换 (四元数转旋转矩阵)
    q_norm = quaternion / (np.linalg.norm(quaternion) + 1e-8)
    w, x_q, y_q, z_q = q_norm
    
    # 构建旋转矩阵
    R = np.array([
        [1 - 2*y_q*y_q - 2*z_q*z_q, 2*x_q*y_q - 2*w*z_q, 2*x_q*z_q + 2*w*y_q],
        [2*x_q*y_q + 2*w*z_q, 1 - 2*x_q*x_q - 2*z_q*z_q, 2*y_q*z_q - 2*w*x_q],
        [2*x_q*z_q - 2*w*y_q, 2*y_q*z_q + 2*w*x_q, 1 - 2*x_q*x_q - 2*y_q*y_q]
    ])
    
    # 旋转椭球
    points = np.array([x.flatten(), y.flatten(), z.flatten()])
    rotated_points = R @ points
    
    # 平移到中心位置
    x_rot = rotated_points[0, :].reshape(x.shape) + center[0]
    y_rot = rotated_points[1, :].reshape(y.shape) + center[1]
    z_rot = rotated_points[2, :].reshape(z.shape) + center[2]
    
    # 绘制椭球表面
    ax.plot_surface(x_rot, y_rot, z_rot, alpha=alpha, color=color, 
                    linewidth=0, antialiased=True)

# 选择部分高斯绘制椭球（避免过多重叠）
num_ellipsoids = model.K # 最多显示32个椭球
indices = np.linspace(0, model.K - 1, num_ellipsoids, dtype=int)

colors = plt.cm.viridis(np.linspace(0, 1, num_ellipsoids))
for idx, color in zip(indices, colors):
    draw_ellipsoid(ax4, mu[idx], s[idx], q[idx], color=color, alpha=0.12)

ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.set_zlim(0, 1)
ax4.set_title(f'3D Gaussian Ellipsoids\nK={model.K}, showing {num_ellipsoids} covariances')
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_zlabel('Z')

# 5. 训练数据拟合度对比曲线 (选择Z中间切片的一条线)
ax5 = plt.subplot(2, 4, 5)
y_line = H // 2  # 选择Y方向中间位置的一条线
channel_colors = ['#E74C3C', '#27AE60', '#3498DB', '#9B59B6', '#F39C12', '#1ABC9C', '#E67E22', '#8E44AD', '#2ECC71']
channel_names = ['L0', 'L1-X', 'L1-Y', 'L1-Z', 'L2-0', 'L2-1', 'L2-2', 'L2-3', 'L2-4']

# 选取前3个通道展示 (L0和L1的两个分量)
show_channels = [0, 1, 2]
for c in show_channels:
    ax5.plot(gt_slice[y_line, :, c], 
            color=channel_colors[c], linestyle='-', alpha=0.7, linewidth=1.5,
            label=f'Original {channel_names[c]}')
    ax5.plot(rec_slice[y_line, :, c], 
            color=channel_colors[c], linestyle='--', alpha=0.9, linewidth=1.5,
            label=f'Recon {channel_names[c]}')

ax5.set_title(f'Training Data Fitting (Z={z_slice}, Y={y_line} slice)')
ax5.set_xlabel('X Coordinate')
ax5.set_ylabel('SH Coefficient Value')
ax5.legend(loc='upper right', fontsize='x-small', ncol=2)
ax5.grid(True, alpha=0.3)

# 6. Rotation 可视化 (四元数分布)
ax6 = plt.subplot(2, 4, 6)

# 计算旋转角度 (从四元数转换为旋转角度)
q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
rotation_angles = 2 * np.arccos(np.clip(q_norm[:, 0], -1, 1)) * 180 / np.pi  # 转换为角度

# 绘制旋转角度的分布
ax6.hist(rotation_angles, bins=30, alpha=0.7, color='purple', edgecolor='black')
ax6.axvline(x=rotation_angles.mean(), color='red', linestyle='--', 
            label=f'Mean: {rotation_angles.mean():.1f}°')
ax6.set_title('Rotation Angle Distribution')
ax6.set_xlabel('Rotation Angle (degrees)')
ax6.set_ylabel('Count')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 添加统计信息
rot_stats = f"Rotation Stats:\nMean: {rotation_angles.mean():.1f}°\nStd: {rotation_angles.std():.1f}°\nRange: [{rotation_angles.min():.1f}°, {rotation_angles.max():.1f}°]"
ax6.text(0.98, 0.98, rot_stats, transform=ax6.transAxes, fontsize=8,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='plum', alpha=0.5))

# 7. 训练损失曲线
ax7 = plt.subplot(2, 4, 7)
total_losses = [l['total_loss'] for l in losses]
mse_losses = [l['mse_loss'] for l in losses]
ax7.semilogy(total_losses, 'b-', linewidth=2, label='Total Loss')
ax7.semilogy(mse_losses, 'g--', linewidth=1.5, alpha=0.7, label='MSE Loss')
ax7.axvline(x=1500, color='r', linestyle=':', alpha=0.5, label='Quant Finetune Start')
ax7.set_title('Training Loss Curves (Log Scale)')
ax7.set_xlabel('Iterations')
ax7.set_ylabel('Loss')
ax7.legend()
ax7.grid(True, alpha=0.3)

# 8. 综合信息：通道对比 + Transform统计
ax8 = plt.subplot(2, 4, 8)
ax8.axis('off')
num_probes = D * H * W

# Scale各向异性分析 (最大/最小尺度的比值)
scale_ratios = s.max(axis=1) / (s.min(axis=1) + 1e-8)

info_text = f"""
3D Gaussian Compression Summary
================================
Original Data:
  Volume: {D}x{H}x{W} = {num_probes} probes
  SH Dim: {C} (L=0,1,2 bands)
  Size: {original_size_bytes/1024:.1f} KB

Compressed Model:
  Gaussians (K): {model.K}
  Latent Dim (D): {model.D}
  Est. Size: {comp_size/1024:.1f} KB
  Ratio: {comp_ratio:.1f}:1

Transform Statistics:
  Scale X: {s[:, 0].mean():.4f} ± {s[:, 0].std():.4f}
  Scale Y: {s[:, 1].mean():.4f} ± {s[:, 1].std():.4f}
  Scale Z: {s[:, 2].mean():.4f} ± {s[:, 2].std():.4f}
  Anisotropy (max/min): {scale_ratios.mean():.2f} ± {scale_ratios.std():.2f}
  Rotation Angle: {rotation_angles.mean():.1f}° ± {rotation_angles.std():.1f}°

Reconstruction Quality:
  Avg PSNR: {avg_psnr:.1f} dB
  Avg SSIM: {avg_ssim:.4f}
  Final Loss: {losses[-1]['total_loss']:.6f}
"""
ax8.text(0.05, 0.5, info_text, fontsize=8.5,
        family='monospace', verticalalignment='center')

plt.suptitle('3D Gaussian Compression for Precomputed Indirect Illumination', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("3D高斯压缩演示完成！关键结论：")
print(f"1. 使用完整的3D高斯表示: 位置(3D) + 尺度(3D) + 旋转(四元数)")
print(f"2. 使用 {model.K} 个高斯函数拟合了 {num_probes} 个探针的空间分布，压缩比约 {comp_ratio:.1f}:1")
print(f"3. 重建质量：平均PSNR= {avg_psnr:.1f} dB, 平均SSIM={avg_ssim:.4f}")
print(f"4. 协方差矩阵计算: Σ = R @ S @ S^T @ R^T（完整旋转支持）")
print(f"5. 可视化展示了3D高斯分布位置、潜在特征图及重建误差")
print("="*70)