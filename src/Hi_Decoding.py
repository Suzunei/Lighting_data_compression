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

# ==================== Positional Encoding for Fine MLP ====================
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for better high-frequency learning"""
    def __init__(self, num_freqs=6):
        super().__init__()
        self.num_freqs = num_freqs
        # Precompute frequency bands: 2^0, 2^1, ..., 2^(L-1)
        self.freq_bands = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
        
    def forward(self, x):
        """
        x: [N, 3] input coordinates
        Returns: [N, 3 + 3*2*num_freqs] encoded coordinates
        """
        device = x.device
        freq_bands = self.freq_bands.to(device)
        
        encoded = [x]  # Original coordinates
        for freq in freq_bands:
            encoded.append(torch.sin(freq * np.pi * x))
            encoded.append(torch.cos(freq * np.pi * x))
        return torch.cat(encoded, dim=-1)
    
    def get_output_dim(self, input_dim=3):
        return input_dim * (1 + 2 * self.num_freqs)


class MBDCompressor3D(nn.Module):
    """
    Moving Basis Decomposition (MBD) with Hierarchical Decoding (Coarse-to-Fine)
    
    Architecture:
        Coarse Branch (MBD): Low-frequency global illumination
            c_l(x) = Σ_m φ_m(x) * c_{m,l}
            b_l(x) = Σ_n ψ_n(x) * B_{n,l}
            f_coarse(x) = Σ_l c_l(x) * b_l(x)
        
        Fine Branch (MLP): High-frequency details
            f_fine(x) = MLP(PE(x))
        
        Adaptive Gate: Position-dependent fusion
            gate(x) = σ(MLP_gate(x))
            f_final(x) = (1 - gate) * f_coarse + gate * f_fine
    
    Key Innovation: Hierarchical decoding with adaptive gating allows:
        - MBD to focus on smooth, low-frequency components (better compression)
        - MLP to capture high-frequency details (better quality)
        - Gate to adaptively blend based on local complexity
    """
    def __init__(self, num_bases=6, coeff_res=12, basis_res=8, data_dim=3,
                 coeff_kernel_scale=0.15, basis_kernel_scale=0.2, mlp_hidden=64,
                 pe_num_freqs=4, fine_mlp_depth=2):
        super().__init__()
        self.L = num_bases
        self.data_dim = data_dim
        self.mlp_hidden = mlp_hidden
        self.pe_num_freqs = pe_num_freqs

        # ========== Coefficient 3D Gaussian Parameters ==========
        # Position mu: [M, 3] - Trainable 3D position
        self.coeff_mu = nn.Parameter(torch.rand(coeff_res, 3))
        # Log-scale log_s: [M, 3] - Trainable anisotropic scale
        init_coeff_log_scale = np.log(coeff_kernel_scale)
        self.coeff_log_s = nn.Parameter(
            torch.ones(coeff_res, 3) * init_coeff_log_scale + torch.randn(coeff_res, 3) * 0.1
        )
        # Quaternion rotation q: [M, 4] (w, x, y, z) - Trainable rotation
        self.coeff_q = nn.Parameter(torch.zeros(coeff_res, 4))
        with torch.no_grad():
            self.coeff_q[:, 0] = 1.0  # Initialize to identity quaternion [1, 0, 0, 0]

        # ========== Basis 3D Gaussian Parameters ==========
        # Position mu: [N, 3]
        self.basis_mu = nn.Parameter(torch.rand(basis_res, 3))
        # Log-scale log_s: [N, 3]
        init_basis_log_scale = np.log(basis_kernel_scale)
        self.basis_log_s = nn.Parameter(
            torch.ones(basis_res, 3) * init_basis_log_scale + torch.randn(basis_res, 3) * 0.1
        )
        # Quaternion rotation q: [N, 4]
        self.basis_q = nn.Parameter(torch.zeros(basis_res, 4))
        with torch.no_grad():
            self.basis_q[:, 0] = 1.0

        # ========== MBD Coefficient/Basis Tensors ==========
        # C: [M, L] - scalar coefficients at coefficient control points
        # B: [N, L, D] - basis vectors at basis control points
        self.C = nn.Parameter(torch.randn(coeff_res, self.L) * 0.1)
        self.B = nn.Parameter(torch.randn(basis_res, self.L, self.data_dim) * 0.1)

        # ========== Positional Encoding ==========
        self.pos_encoder = PositionalEncoding(num_freqs=pe_num_freqs)
        pe_dim = self.pos_encoder.get_output_dim(3)  # 3 + 3*2*num_freqs

        # ========== Fine Branch: MLP with Positional Encoding ==========
        # Captures high-frequency details that MBD cannot represent
        fine_layers = []
        fine_layers.append(nn.Linear(pe_dim, mlp_hidden))
        fine_layers.append(nn.ReLU())
        for _ in range(fine_mlp_depth - 1):
            fine_layers.append(nn.Linear(mlp_hidden, mlp_hidden))
            fine_layers.append(nn.ReLU())
        fine_layers.append(nn.Linear(mlp_hidden, data_dim))
        self.fine_branch = nn.Sequential(*fine_layers)

        # ========== Adaptive Gate Network ==========
        # Learns position-dependent blending weight between coarse and fine
        # Higher gate value = more reliance on fine branch (high-frequency regions)
        self.gate_network = nn.Sequential(
            nn.Linear(3, mlp_hidden // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden // 2, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

        # ========== Residual Refinement (Optional Enhancement) ==========
        # Small MLP to refine the final output
        self.residual_refiner = nn.Sequential(
            nn.Linear(data_dim + 3, mlp_hidden // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden // 2, data_dim)
        )

        # Initialize statistics
        self.M = coeff_res
        self.N = basis_res
        print(f"Hierarchical MBD+MLP model initialized (Coarse-to-Fine):")
        print(f"  [Coarse Branch] MBD with 3D Gaussians:")
        print(f"    - Coefficient Gaussians: M={self.M}")
        print(f"    - Basis Gaussians: N={self.N}")
        print(f"    - Num bases: L={self.L}")
        print(f"  [Fine Branch] MLP with Positional Encoding:")
        print(f"    - PE frequencies: {pe_num_freqs} -> dim {pe_dim}")
        print(f"    - MLP depth: {fine_mlp_depth} layers, hidden={mlp_hidden}")
        print(f"  [Gate Network] Adaptive blending:")
        print(f"    - Input: 3D position -> Output: gate weight")
        print(f"  [Residual Refiner] Final enhancement")

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

    def forward(self, coords, return_components=False):
        """
        Forward pass with hierarchical coarse-to-fine decoding.
        
        Args:
            coords: [Q, 3] query coordinates
            return_components: if True, return all intermediate results for analysis
            
        Returns:
            reconstruction: [Q, D] final output
            moving_coeff: [Q, L] MBD coefficients
            moving_basis: [Q, L, D] MBD bases
            coarse_output: [Q, D] MBD branch output (for intermediate supervision)
        """
        # ============ Coarse Branch: MBD Reconstruction ============
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

        # 4. MBD output (Coarse): f_coarse(x) = Σ_l c_l(x) * b_l(x)
        coarse_output = torch.sum(moving_coeff.unsqueeze(-1) * moving_basis, dim=1)  # [Q, D]

        # ============ Fine Branch: MLP with Positional Encoding ============
        # 5. Apply positional encoding to coordinates
        coords_encoded = self.pos_encoder(coords)  # [Q, pe_dim]
        
        # 6. Fine MLP output: high-frequency details
        fine_output = self.fine_branch(coords_encoded)  # [Q, D]

        # ============ Adaptive Gating ============
        # 7. Compute position-dependent gate weights
        gate = self.gate_network(coords)  # [Q, 1]
        
        # 8. Blend coarse and fine branches
        # gate=0 -> pure coarse (MBD), gate=1 -> pure fine (MLP)
        blended = (1 - gate) * coarse_output + gate * fine_output  # [Q, D]

        # ============ Residual Refinement ============
        # 9. Small residual correction
        refine_input = torch.cat([blended, coords], dim=1)  # [Q, D+3]
        residual = self.residual_refiner(refine_input)  # [Q, D]
        
        # 10. Final output with residual connection
        reconstruction = blended + 0.1 * residual  # [Q, D]

        if return_components:
            return {
                'reconstruction': reconstruction,
                'coarse_output': coarse_output,
                'fine_output': fine_output,
                'gate': gate,
                'blended': blended,
                'moving_coeff': moving_coeff,
                'moving_basis': moving_basis
            }
        
        return reconstruction, moving_coeff, moving_basis, coarse_output

    def get_compression_ratio(self, original_size, bytes_per_param=4):
        """Compute compression ratio (including all network parameters)
        
        Args:
            original_size: original data size in bytes
            bytes_per_param: bytes per parameter (4 for float32, 2 for float16)
        """
        # 3D Gaussian params: mu(3) + log_s(3) + q(4) = 10 per gaussian
        coeff_params = self.M * (3 + 3 + 4 + self.L)
        basis_params = self.N * (3 + 3 + 4 + self.L * self.data_dim)

        # Count all network parameters
        fine_branch_params = sum(p.numel() for p in self.fine_branch.parameters())
        gate_params = sum(p.numel() for p in self.gate_network.parameters())
        refiner_params = sum(p.numel() for p in self.residual_refiner.parameters())

        total_params = coeff_params + basis_params + fine_branch_params + gate_params + refiner_params
        compressed_size = total_params * bytes_per_param  # 支持不同精度
        ratio = original_size / compressed_size
        return ratio, compressed_size
    
    def get_branch_info(self):
        """Get detailed parameter count for each branch"""
        coeff_params = self.M * (3 + 3 + 4 + self.L)
        basis_params = self.N * (3 + 3 + 4 + self.L * self.data_dim)
        fine_params = sum(p.numel() for p in self.fine_branch.parameters())
        gate_params = sum(p.numel() for p in self.gate_network.parameters())
        refiner_params = sum(p.numel() for p in self.residual_refiner.parameters())
        
        return {
            'coarse_mbd': coeff_params + basis_params,
            'fine_mlp': fine_params,
            'gate_network': gate_params,
            'residual_refiner': refiner_params,
            'total': coeff_params + basis_params + fine_params + gate_params + refiner_params
        }

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

    def quantize_to_fp16(self):
        """
        将所有模型参数量化为 float16 再转回 float32，模拟 float16 存储精度损失。
        这是一种 Post-Training Quantization (PTQ) 策略。
        返回量化前后的参数差异统计。
        """
        stats = {}
        with torch.no_grad():
            for name, param in self.named_parameters():
                original = param.data.clone()
                # float32 -> float16 -> float32 (模拟精度损失)
                param.data = param.data.half().float()
                # 记录量化误差
                diff = (original - param.data).abs()
                stats[name] = {
                    'max_error': diff.max().item(),
                    'mean_error': diff.mean().item(),
                    'relative_error': (diff / (original.abs() + 1e-8)).mean().item()
                }
        return stats

class MBDSolver3D:
    """
    Hierarchical MBD Solver with Coarse-to-Fine Training Strategy.
    
    Training Strategy:
        Stage 1 (Coarse Focus): Train MBD branch with higher weight on coarse loss
        Stage 2 (Fine Focus): Gradually shift focus to fine branch
        Stage 3 (Joint Refinement): Fine-tune all branches together
    """
    def __init__(self, model, lambda_reg=0.01, lambda_coarse=0.5):
        self.model = model
        self.lambda_reg = lambda_reg
        self.lambda_coarse = lambda_coarse  # Weight for coarse supervision
        self.initial_lambda_coarse = lambda_coarse

        # Separate parameter groups for different learning dynamics
        gaussian_params = [
            self.model.coeff_mu, self.model.coeff_log_s, self.model.coeff_q,
            self.model.basis_mu, self.model.basis_log_s, self.model.basis_q
        ]
        mbd_params = [self.model.C, self.model.B]
        fine_params = list(self.model.fine_branch.parameters())
        gate_params = list(self.model.gate_network.parameters())
        refiner_params = list(self.model.residual_refiner.parameters())

        # Different learning rates for different components
        self.optimizer_gaussian = optim.Adam(gaussian_params, lr=0.005)
        self.optimizer_mbd = optim.Adam(mbd_params, lr=0.01)
        self.optimizer_fine = optim.Adam(fine_params, lr=0.003)
        self.optimizer_gate = optim.Adam(gate_params, lr=0.002)
        self.optimizer_refiner = optim.Adam(refiner_params, lr=0.003)

        self.scheduler_gaussian = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_gaussian, patience=50, factor=0.5
        )
        self.scheduler_mbd = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_mbd, patience=50, factor=0.5
        )
        self.scheduler_fine = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_fine, patience=50, factor=0.5
        )

    def compute_loss(self, pred, target, coarse_output=None):
        """
        Compute hierarchical loss with intermediate supervision.
        
        Loss = λ_final * L_final + λ_coarse * L_coarse + λ_reg * L_reg
        
        Intermediate supervision on coarse branch encourages MBD to learn
        meaningful low-frequency representation independently.
        """
        # 1. Final reconstruction loss (MSE)
        final_loss = torch.mean((pred - target) ** 2)
        
        # 2. Coarse branch intermediate supervision (if provided)
        coarse_loss = torch.tensor(0.0, device=pred.device)
        if coarse_output is not None:
            coarse_loss = torch.mean((coarse_output - target) ** 2)
        
        # 3. Regularization: prevent scale explosion
        coeff_s = torch.exp(self.model.coeff_log_s)
        basis_s = torch.exp(self.model.basis_log_s)
        reg_loss = self.lambda_reg * (
            torch.sum(self.model.C ** 2) +
            torch.sum(coeff_s ** 2) + torch.sum(basis_s ** 2)
        )
        
        # 4. Gate regularization: encourage diversity (not all 0 or all 1)
        # This is computed in train_step where we have access to gate values
        
        # Weighted combination
        # λ_coarse gradually decreases during training to let fine branch take over
        lambda_coarse = self.lambda_coarse
        total_loss = final_loss + lambda_coarse * coarse_loss + reg_loss
        
        return total_loss, final_loss, coarse_loss, reg_loss

    def train_step(self, coords_batch, target_batch, stage='main'):
        """
        Single training step with hierarchical loss.
        
        Stages:
            'coarse': Focus on MBD branch (higher λ_coarse)
            'main': Train all branches jointly
            'fine': Fine-tune with focus on fine branch (lower λ_coarse)
        """
        # Zero gradients for all optimizers
        self.optimizer_gaussian.zero_grad()
        self.optimizer_mbd.zero_grad()
        self.optimizer_fine.zero_grad()
        self.optimizer_gate.zero_grad()
        self.optimizer_refiner.zero_grad()

        # Forward pass with component outputs
        pred, moving_coeff, moving_basis, coarse_output = self.model(coords_batch)

        # Compute hierarchical loss
        total_loss, final_loss, coarse_loss, reg_loss = self.compute_loss(
            pred, target_batch, coarse_output
        )
        
        # Gate diversity regularization
        # Encourage gate to have variance (not stuck at 0 or 1)
        with torch.no_grad():
            gate_values = self.model.gate_network(coords_batch)
            gate_mean = gate_values.mean().item()
            gate_std = gate_values.std().item()

        # Backward pass
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Optimization based on stage
        if stage == 'coarse':
            # Focus on coarse branch
            self.optimizer_gaussian.step()
            self.optimizer_mbd.step()
        elif stage == 'fine':
            # Focus on fine branch
            self.optimizer_fine.step()
            self.optimizer_gate.step()
            self.optimizer_refiner.step()
        else:  # 'main'
            # Train all branches
            self.optimizer_gaussian.step()
            self.optimizer_mbd.step()
            self.optimizer_fine.step()
            self.optimizer_gate.step()
            self.optimizer_refiner.step()

        return {
            'total_loss': total_loss.item(),
            'final_loss': final_loss.item(),
            'coarse_loss': coarse_loss.item(),
            'reg_loss': reg_loss.item(),
            'gate_mean': gate_mean,
            'gate_std': gate_std
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

    def train(self, coords, target, epochs_coarse=500, epochs_main=1500, epochs_fine=500, batch_size=2048):
        """
        Three-stage hierarchical training:
        
        Stage 1 (Coarse Focus): Train MBD to capture low-frequency structure
            - High λ_coarse weight
            - Only update Gaussian and MBD parameters
            
        Stage 2 (Joint Training): Train all branches together
            - Gradually decrease λ_coarse
            - Update all parameters
            
        Stage 3 (Fine Focus): Refine high-frequency details
            - Low λ_coarse weight
            - Focus on fine branch and gate
        """
        losses = []
        num_samples = coords.shape[0]
        total_epochs = epochs_coarse + epochs_main + epochs_fine
        
        print(f"="*60)
        print(f"Hierarchical Training Strategy (Coarse-to-Fine)")
        print(f"="*60)
        print(f"  Stage 1 (Coarse): {epochs_coarse} epochs - MBD focus")
        print(f"  Stage 2 (Joint):  {epochs_main} epochs - All branches")
        print(f"  Stage 3 (Fine):   {epochs_fine} epochs - Detail refinement")
        print(f"  Total: {total_epochs} epochs, batch_size={batch_size}")
        print(f"="*60)
        
        # ============ Stage 1: Coarse Focus ============
        print(f"\n[Stage 1] Coarse Focus Training ({epochs_coarse} epochs)...")
        print(f"  λ_coarse = {self.lambda_coarse:.2f} (high weight on MBD)")
        
        for epoch in range(epochs_coarse):
            indices = torch.randperm(num_samples)[:batch_size]
            coords_batch = coords[indices]
            target_batch = target[indices]

            loss_dict = self.train_step(coords_batch, target_batch, 'coarse')
            losses.append(loss_dict)

            if epoch % 100 == 0:
                self.scheduler_gaussian.step(loss_dict['total_loss'])
                self.scheduler_mbd.step(loss_dict['total_loss'])

            if epoch % 200 == 0 or epoch == epochs_coarse - 1:
                print(f"  Epoch {epoch:4d}/{epochs_coarse} | "
                      f"Total: {loss_dict['total_loss']:.6f} | "
                      f"Coarse: {loss_dict['coarse_loss']:.6f} | "
                      f"Gate: {loss_dict['gate_mean']:.3f}±{loss_dict['gate_std']:.3f}")

        # ============ Stage 2: Joint Training ============
        print(f"\n[Stage 2] Joint Training ({epochs_main} epochs)...")
        
        for epoch in range(epochs_main):
            # Gradually decrease λ_coarse (curriculum learning)
            progress = epoch / epochs_main
            self.lambda_coarse = self.initial_lambda_coarse * (1 - 0.8 * progress)  # Decay from 1.0 to 0.2
            
            indices = torch.randperm(num_samples)[:batch_size]
            coords_batch = coords[indices]
            target_batch = target[indices]

            loss_dict = self.train_step(coords_batch, target_batch, 'main')
            losses.append(loss_dict)

            if epoch % 100 == 0:
                self.scheduler_gaussian.step(loss_dict['total_loss'])
                self.scheduler_mbd.step(loss_dict['total_loss'])
                self.scheduler_fine.step(loss_dict['total_loss'])

            if epoch % 300 == 0 or epoch == epochs_main - 1:
                print(f"  Epoch {epoch:4d}/{epochs_main} | "
                      f"Total: {loss_dict['total_loss']:.6f} | "
                      f"Final: {loss_dict['final_loss']:.6f} | "
                      f"Coarse: {loss_dict['coarse_loss']:.6f} | "
                      f"λ_c: {self.lambda_coarse:.2f} | "
                      f"Gate: {loss_dict['gate_mean']:.3f}")

        # ============ Stage 3: Fine Focus ============
        print(f"\n[Stage 3] Fine Focus Training ({epochs_fine} epochs)...")
        self.lambda_coarse = 0.1  # Low weight on coarse
        
        for epoch in range(epochs_fine):
            indices = torch.randperm(num_samples)[:batch_size]
            coords_batch = coords[indices]
            target_batch = target[indices]

            loss_dict = self.train_step(coords_batch, target_batch, 'fine')
            losses.append(loss_dict)

            if epoch % 200 == 0 or epoch == epochs_fine - 1:
                print(f"  Epoch {epoch:4d}/{epochs_fine} | "
                      f"Total: {loss_dict['total_loss']:.6f} | "
                      f"Final: {loss_dict['final_loss']:.6f} | "
                      f"Gate: {loss_dict['gate_mean']:.3f}±{loss_dict['gate_std']:.3f}")

        print(f"\n[Training Complete] Total epochs: {len(losses)}")
        return losses

# ==================== Hierarchical MBD + MLP with Coarse-to-Fine Decoding ====================
# Create model with hierarchical architecture
model = MBDCompressor3D(
    num_bases=8,              # Number of bases L
    coeff_res=32,             # Coefficient 3D Gaussians M
    basis_res=32,             # Basis 3D Gaussians N
    data_dim=C,               # Data dimension D (RGB)
    coeff_kernel_scale=0.18,  # Initial scale
    basis_kernel_scale=0.22,  # Initial scale
    mlp_hidden=48,            # MLP hidden size
    pe_num_freqs=4,           # Positional encoding frequencies
    fine_mlp_depth=2          # Fine branch MLP depth
)

# Print detailed architecture info
branch_info = model.get_branch_info()
print(f"\nModel Parameter Distribution:")
print(f"  Coarse (MBD):     {branch_info['coarse_mbd']:6d} params")
print(f"  Fine (MLP+PE):    {branch_info['fine_mlp']:6d} params")
print(f"  Gate Network:     {branch_info['gate_network']:6d} params")
print(f"  Residual Refiner: {branch_info['residual_refiner']:6d} params")
print(f"  Total:            {branch_info['total']:6d} params")

# Create solver with hierarchical training strategy
solver = MBDSolver3D(model, lambda_reg=1e-5, lambda_coarse=0.5)

# 计算原始数据大小
original_size = D * H * W * C * 4  # float32
print(f"Original size: {original_size/1024:.1f} KB")

# Train model with hierarchical strategy (Coarse -> Joint -> Fine)
print("\nStarting Hierarchical Training (Coarse-to-Fine)...")
losses = solver.train(
    coords, target_data, 
    epochs_coarse=400,    # Stage 1: Focus on MBD
    epochs_main=1600,     # Stage 2: Joint training
    epochs_fine=500,      # Stage 3: Fine-tune details
    batch_size=4096
)

# ==================== Float16 Quantization ====================
print("\n" + "="*60)
print("Applying Float16 Post-Training Quantization...")
print("="*60)

# 量化前的压缩比
comp_ratio_fp32, comp_size_fp32 = model.get_compression_ratio(original_size, bytes_per_param=4)
print(f"\n[Before Quantization] float32:")
print(f"  Compressed: {comp_size_fp32/1024:.2f} KB")
print(f"  Compression ratio: {comp_ratio_fp32:.1f}:1")

# 执行 float16 量化
quant_stats = model.quantize_to_fp16()

# 量化后的压缩比
comp_ratio_fp16, comp_size_fp16 = model.get_compression_ratio(original_size, bytes_per_param=2)
print(f"\n[After Quantization] float16:")
print(f"  Compressed: {comp_size_fp16/1024:.2f} KB")
print(f"  Compression ratio: {comp_ratio_fp16:.1f}:1")
print(f"  Ratio improvement: {comp_ratio_fp16/comp_ratio_fp32:.1f}x")

# 显示关键参数的量化误差
print(f"\nQuantization Error Statistics:")
for name, stat in quant_stats.items():
    if stat['max_error'] > 1e-6:  # 只显示有意义的误差
        print(f"  {name:30s} | max_err: {stat['max_error']:.6f} | mean_err: {stat['mean_error']:.6f} | rel_err: {stat['relative_error']:.4f}")

# 使用 float16 压缩比作为最终展示值
comp_ratio = comp_ratio_fp16
comp_size = comp_size_fp16

# ==================== Step 3: Evaluation and visualization ====================
print("\nStep 3: Evaluating compression and reconstruction quality...")

# Reconstruct entire 3D volume using trained model
model.eval()
with torch.no_grad():
    # Get all components for analysis
    results = model(coords, return_components=True)
    reconstructed = results['reconstruction']
    coarse_recon = results['coarse_output']
    fine_recon = results['fine_output']
    gate_values = results['gate']
    
    reconstructed_vol = reconstructed.view(D, H, W, C).cpu().numpy()
    coarse_vol = coarse_recon.view(D, H, W, C).cpu().numpy()
    fine_vol = fine_recon.view(D, H, W, C).cpu().numpy()
    gate_vol = gate_values.view(D, H, W).cpu().numpy()
    
    # Clip to valid range [0, 1]
    reconstructed_vol = np.clip(reconstructed_vol, 0, 1)
    coarse_vol = np.clip(coarse_vol, 0, 1)
    fine_vol = np.clip(fine_vol, 0, 1)

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

# ==================== Visualization results (Hierarchical Decoding) ====================
print("\nGenerating Hierarchical Visualization results...")

# Get middle slice for visualization
gt_slice = ground_truth[z_slice, :, :, :].numpy()  # [H, W, C]
rec_slice = reconstructed_vol[z_slice, :, :, :]    # [H, W, C]
coarse_slice = coarse_vol[z_slice, :, :, :]        # [H, W, C]
fine_slice = fine_vol[z_slice, :, :, :]            # [H, W, C]
gate_slice = gate_vol[z_slice, :, :]               # [H, W]

fig = plt.figure(figsize=(24, 16))

# Row 1: Original, Final, Coarse, Fine
# 1. Original signal
ax1 = plt.subplot(3, 4, 1)
im1 = ax1.imshow(gt_slice, vmin=0, vmax=1)
ax1.set_title(f'Ground Truth (Z={z_slice})\n{D}x{H}x{W}x{C}')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.grid(False)

# 2. Final reconstruction (Coarse + Fine blended)
ax2 = plt.subplot(3, 4, 2)
im2 = ax2.imshow(rec_slice, vmin=0, vmax=1)
ax2.set_title(f'Final Output (Blended)\nPSNR: {psnr_value:.1f}dB, Ratio: {comp_ratio:.1f}:1')
ax2.set_xlabel('X')
ax2.grid(False)

# 3. Coarse branch output (MBD only)
coarse_psnr = compute_psnr(gt_slice.mean(axis=-1), coarse_slice.mean(axis=-1))
ax3 = plt.subplot(3, 4, 3)
im3 = ax3.imshow(coarse_slice, vmin=0, vmax=1)
ax3.set_title(f'Coarse Branch (MBD)\nPSNR: {coarse_psnr:.1f}dB')
ax3.set_xlabel('X')
ax3.grid(False)

# 4. Fine branch output (MLP only)
fine_psnr = compute_psnr(gt_slice.mean(axis=-1), fine_slice.mean(axis=-1))
ax4 = plt.subplot(3, 4, 4)
im4 = ax4.imshow(fine_slice, vmin=0, vmax=1)
ax4.set_title(f'Fine Branch (MLP+PE)\nPSNR: {fine_psnr:.1f}dB')
ax4.set_xlabel('X')
ax4.grid(False)

# Row 2: Error maps and Gate visualization
# 5. Final error map
ax5 = plt.subplot(3, 4, 5)
error = np.abs(gt_slice - rec_slice)
error_img = ax5.imshow(error.mean(axis=-1), cmap='hot', vmin=0, vmax=0.15)
ax5.set_title(f'Final Error\nSSIM: {ssim_value:.4f}')
ax5.set_xlabel('X')
plt.colorbar(error_img, ax=ax5, fraction=0.046, pad=0.04)
ax5.grid(False)

# 6. Coarse error map
ax6 = plt.subplot(3, 4, 6)
coarse_error = np.abs(gt_slice - coarse_slice)
coarse_error_img = ax6.imshow(coarse_error.mean(axis=-1), cmap='hot', vmin=0, vmax=0.15)
ax6.set_title('Coarse Branch Error')
ax6.set_xlabel('X')
plt.colorbar(coarse_error_img, ax=ax6, fraction=0.046, pad=0.04)
ax6.grid(False)

# 7. Gate weight visualization (shows where fine branch is used more)
ax7 = plt.subplot(3, 4, 7)
gate_img = ax7.imshow(gate_slice, cmap='RdYlBu_r', vmin=0, vmax=1)
ax7.set_title(f'Gate Weights\n0=Coarse, 1=Fine\nMean: {gate_slice.mean():.3f}')
ax7.set_xlabel('X')
plt.colorbar(gate_img, ax=ax7, fraction=0.046, pad=0.04)
ax7.grid(False)

# 8. Gate histogram
ax8 = plt.subplot(3, 4, 8)
ax8.hist(gate_vol.flatten(), bins=50, alpha=0.7, color='steelblue', edgecolor='black')
ax8.axvline(x=gate_vol.mean(), color='red', linestyle='--', linewidth=2,
            label=f'Mean: {gate_vol.mean():.3f}')
ax8.set_title('Gate Distribution (All Voxels)')
ax8.set_xlabel('Gate Value')
ax8.set_ylabel('Count')
ax8.legend()
ax8.grid(True, alpha=0.3)

# Row 3: Training curves and statistics
# 9. Training loss curves
ax9 = plt.subplot(3, 4, 9)
total_losses = [l['total_loss'] for l in losses]
final_losses = [l['final_loss'] for l in losses]
coarse_losses = [l['coarse_loss'] for l in losses]

ax9.semilogy(total_losses, 'b-', linewidth=2, label='Total Loss')
ax9.semilogy(final_losses, 'g--', linewidth=1.5, alpha=0.7, label='Final Loss')
ax9.semilogy(coarse_losses, 'r:', linewidth=1.5, alpha=0.7, label='Coarse Loss')

# Mark training stages
epochs_coarse, epochs_main, epochs_fine = 400, 1600, 500
ax9.axvline(x=epochs_coarse, color='orange', linestyle=':', alpha=0.7, label='Stage 1→2')
ax9.axvline(x=epochs_coarse + epochs_main, color='purple', linestyle=':', alpha=0.7, label='Stage 2→3')
ax9.set_title('Training Loss (3 Stages)')
ax9.set_xlabel('Iterations')
ax9.set_ylabel('Loss Value')
ax9.legend(fontsize='x-small', loc='upper right')
ax9.grid(True, alpha=0.3)

# 10. Gate evolution during training
ax10 = plt.subplot(3, 4, 10)
gate_means = [l['gate_mean'] for l in losses]
gate_stds = [l['gate_std'] for l in losses]
epochs_arr = np.arange(len(gate_means))

ax10.plot(gate_means, 'b-', linewidth=2, label='Gate Mean')
ax10.fill_between(epochs_arr, 
                   np.array(gate_means) - np.array(gate_stds),
                   np.array(gate_means) + np.array(gate_stds),
                   alpha=0.3, color='blue', label='±1 Std')
ax10.axvline(x=epochs_coarse, color='orange', linestyle=':', alpha=0.7)
ax10.axvline(x=epochs_coarse + epochs_main, color='purple', linestyle=':', alpha=0.7)
ax10.set_title('Gate Weight Evolution')
ax10.set_xlabel('Iterations')
ax10.set_ylabel('Gate Value')
ax10.set_ylim(0, 1)
ax10.legend(fontsize='small')
ax10.grid(True, alpha=0.3)

# 11. Channel comparison (1D line)
ax11 = plt.subplot(3, 4, 11)
y_line = H // 2
channel_colors = ['#E74C3C', '#27AE60', '#3498DB']
channel_names = ['R', 'G', 'B']

for c in range(C):
    ax11.plot(gt_slice[y_line, :, c],
              color=channel_colors[c], linestyle='-', alpha=0.6, linewidth=1.5,
              label=f'GT {channel_names[c]}')
    ax11.plot(rec_slice[y_line, :, c],
              color=channel_colors[c], linestyle='--', alpha=0.9, linewidth=1.5,
              label=f'Pred {channel_names[c]}')

ax11.set_title(f'1D Fitting (Z={z_slice}, Y={y_line})')
ax11.set_xlabel('X Coordinate')
ax11.set_ylabel('Value')
ax11.legend(loc='upper right', fontsize='x-small', ncol=2)
ax11.grid(True, alpha=0.3)

# 12. Summary information
ax12 = plt.subplot(3, 4, 12)
ax12.axis('off')
num_probes = D * H * W

# Get Gaussian stats
gaussian_params = model.get_gaussian_params()
coeff_s = gaussian_params['coeff_s']
basis_s = gaussian_params['basis_s']
coeff_q = gaussian_params['coeff_q']
basis_q = gaussian_params['basis_q']

coeff_scale_ratios = coeff_s.max(axis=1) / (coeff_s.min(axis=1) + 1e-8)
basis_scale_ratios = basis_s.max(axis=1) / (basis_s.min(axis=1) + 1e-8)

coeff_q_norm = coeff_q / (np.linalg.norm(coeff_q, axis=1, keepdims=True) + 1e-8)
basis_q_norm = basis_q / (np.linalg.norm(basis_q, axis=1, keepdims=True) + 1e-8)
coeff_angles = 2 * np.arccos(np.clip(coeff_q_norm[:, 0], -1, 1)) * 180 / np.pi
basis_angles = 2 * np.arccos(np.clip(basis_q_norm[:, 0], -1, 1)) * 180 / np.pi

info_text = f"""
Hierarchical MBD (Coarse-to-Fine) Summary
==========================================
Original Data:
  Volume: {D}x{H}x{W}x{C} = {num_probes} points
  Size: {original_size/1024:.1f} KB

Model Architecture:
  Coarse (MBD): M={model.M}, N={model.N}, L={model.L}
  Fine (MLP):   PE({model.pe_num_freqs}) + {model.mlp_hidden}h
  Total params: {branch_info['total']}

Compression (float16 quantized):
  FP32 Size: {comp_size_fp32/1024:.1f} KB ({comp_ratio_fp32:.1f}:1)
  FP16 Size: {comp_size_fp16/1024:.1f} KB ({comp_ratio_fp16:.1f}:1)

Training (3 Stages):
  Stage 1: {epochs_coarse} epochs (Coarse focus)
  Stage 2: {epochs_main} epochs (Joint)
  Stage 3: {epochs_fine} epochs (Fine focus)

Gate Statistics:
  Mean: {gate_vol.mean():.3f} ± {gate_vol.std():.3f}
  Min: {gate_vol.min():.3f}, Max: {gate_vol.max():.3f}

Reconstruction Quality:
  Final PSNR: {psnr_value:.1f} dB
  Final SSIM: {ssim_value:.4f}
  Coarse PSNR: {coarse_psnr:.1f} dB
"""
ax12.text(0.02, 0.5, info_text, fontsize=8,
          family='monospace', verticalalignment='center')

plt.suptitle('Hierarchical MBD with Coarse-to-Fine Decoding', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()

print("\nDemo completed!")
print("="*70)
print("Hierarchical MBD with Coarse-to-Fine Decoding")
print("="*70)
print(f"Architecture:")
print(f"  [Coarse Branch] MBD with 3D Gaussians")
print(f"    - Coefficient Gaussians: M={model.M}")
print(f"    - Basis Gaussians: N={model.N}")
print(f"    - Number of Bases: L={model.L}")
print(f"  [Fine Branch] MLP with Positional Encoding")
print(f"    - PE frequencies: {model.pe_num_freqs}")
print(f"    - Hidden size: {model.mlp_hidden}")
print(f"  [Gate Network] Adaptive blending")
print(f"    - Learned position-dependent weights")
print(f"\nTraining Strategy (3 Stages):")
print(f"  Stage 1: {epochs_coarse} epochs - Focus on Coarse (MBD)")
print(f"  Stage 2: {epochs_main} epochs - Joint training (all branches)")
print(f"  Stage 3: {epochs_fine} epochs - Focus on Fine (details)")
print(f"\nResults:")
print(f"  Compression (FP32): {comp_ratio_fp32:.1f}:1 ({comp_size_fp32/1024:.2f} KB)")
print(f"  Compression (FP16): {comp_ratio_fp16:.1f}:1 ({comp_size_fp16/1024:.2f} KB)")
print(f"  Final PSNR: {psnr_value:.1f} dB")
print(f"  Final SSIM: {ssim_value:.4f}")
print(f"  Coarse-only PSNR: {coarse_psnr:.1f} dB")
print(f"  Gate Mean: {gate_vol.mean():.3f} (0=Coarse, 1=Fine)")
print(f"\nKey Innovations:")
print(f"  1. Hierarchical decoding: MBD for low-freq + MLP for high-freq")
print(f"  2. Positional Encoding: Better high-frequency learning")
print(f"  3. Adaptive Gate: Position-dependent branch selection")
print(f"  4. Intermediate Supervision: Coarse branch also supervised")
print(f"  5. Curriculum Learning: λ_coarse decay during training")
print("="*70)
