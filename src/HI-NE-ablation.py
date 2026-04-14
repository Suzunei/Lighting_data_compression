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
ground_truth = get_test_signal_by_name("neon", grid_size=32, num_channels=3)
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


# ==================== Ablation Configuration ====================
# NOTE: Refiner is DISABLED for all ablation configs (use_refiner=False)
# because it acts as a secondary function approximator that compensates for
# any removed component, making ablation comparisons meaningless.
# Only 'full_with_refiner' keeps it on to show its standalone contribution.
ABLATION_CONFIGS = {
    'full': {  # Full HI-NE-GBD baseline (WITHOUT refiner, for fair ablation comparison)
        'use_fine_branch': True,
        'use_coarse_branch': True,
        'use_alpha': True,
        'use_learnable_scale': True,
        'use_pe': True,
        'use_gaussian_guidance': True,
        'use_refiner': False,
        'use_staged_training': True,
        'description': 'Full HI-NE-GBD (no refiner, ablation baseline)'
    },
    'full_with_refiner': {  # Full model WITH refiner (to measure refiner contribution)
        'use_fine_branch': True,
        'use_coarse_branch': True,
        'use_alpha': True,
        'use_learnable_scale': True,
        'use_pe': True,
        'use_gaussian_guidance': True,
        'use_refiner': True,
        'use_staged_training': True,
        'description': 'Full HI-NE-GBD + Refiner (measures refiner contribution)'
    },
    'A1_no_fine': {  # w/o Fine Branch
        'use_fine_branch': False,
        'use_coarse_branch': True,
        'use_alpha': True,
        'use_learnable_scale': True,
        'use_pe': True,
        'use_gaussian_guidance': True,
        'use_refiner': False,
        'use_staged_training': True,
        'description': 'w/o Fine Branch (Coarse MBD only)'
    },
    'A2_no_coarse': {  # w/o Coarse Branch + w/o PE = Siggraph Gauss+MLP baseline
        'use_fine_branch': True,
        'use_coarse_branch': False,
        'use_alpha': True,
        'use_learnable_scale': True,
        'use_pe': False,           # PE also disabled: matches Siggraph Gauss+MLP baseline
        'use_gaussian_guidance': True,
        'use_refiner': False,
        'use_staged_training': False,  # No staged training needed without coarse branch
        'description': 'Siggraph baseline: Gauss+MLP (no MBD, no PE, no staging)'
    },
    'A3_no_alpha': {  # w/o Learnable Alpha
        'use_fine_branch': True,
        'use_coarse_branch': True,
        'use_alpha': False,
        'use_learnable_scale': True,
        'use_pe': True,
        'use_gaussian_guidance': True,
        'use_refiner': False,
        'use_staged_training': True,
        'description': 'w/o Learnable Alpha (no opacity weighting)'
    },
    'A4_no_scale': {  # w/o Learnable Scale
        'use_fine_branch': True,
        'use_coarse_branch': True,
        'use_alpha': True,
        'use_learnable_scale': False,
        'use_pe': True,
        'use_gaussian_guidance': True,
        'use_refiner': False,
        'use_staged_training': True,
        'description': 'w/o Learnable MBD Scale (fixed scale=1)'
    },
    'A6_no_gauss_guide': {  # w/o Gaussian Guidance
        'use_fine_branch': True,
        'use_coarse_branch': True,
        'use_alpha': True,
        'use_learnable_scale': True,
        'use_pe': True,
        'use_gaussian_guidance': False,
        'use_refiner': False,
        'use_staged_training': True,
        'description': 'w/o Gaussian Guidance (MLP uses PE only, no fine gaussian weights)'
    },
    'A8_no_staged': {  # w/o Coarse-to-Fine Training
        'use_fine_branch': True,
        'use_coarse_branch': True,
        'use_alpha': True,
        'use_learnable_scale': True,
        'use_pe': True,
        'use_gaussian_guidance': True,
        'use_refiner': False,
        'use_staged_training': False,
        'description': 'w/o Staged Training (joint training from start)'
    },
}

# ==================== SELECT ABLATION HERE ====================
ABLATION_MODE = 'A8_no_staged'  # <-- Change this to run different ablation experiments
# Options: 'full', 'full_with_refiner', 'A1_no_fine', 'A2_no_coarse' (=Siggraph baseline),
#          'A3_no_alpha', 'A4_no_scale', 'A6_no_gauss_guide', 'A8_no_staged'

abl_cfg = ABLATION_CONFIGS[ABLATION_MODE]
print(f"\n{'='*60}")
print(f"ABLATION EXPERIMENT: {ABLATION_MODE}")
print(f"Description: {abl_cfg['description']}")
print(f"Config: {abl_cfg}")
print(f"{'='*60}\n")

class MBDCompressor3D(nn.Module):
    """
    Moving Basis Decomposition (MBD) with Hierarchical Decoding (Coarse-to-Fine)
    Ablation-ready version with configurable component switches.
    """
    def __init__(self, num_bases=6, coeff_res=12, basis_res=8, data_dim=3,
                 coeff_kernel_scale=0.15, basis_kernel_scale=0.2, mlp_hidden=64,
                 pe_num_freqs=4, fine_mlp_depth=2, fine_gaussian_res=16, fine_kernel_scale=0.08,
                 # ===== Ablation Switches =====
                 use_fine_branch=True,
                 use_coarse_branch=True,
                 use_alpha=True,
                 use_learnable_scale=True,
                 use_pe=True,
                 use_gaussian_guidance=True,
                 use_refiner=True):
        super().__init__()
        self.L = num_bases
        self.data_dim = data_dim
        self.mlp_hidden = mlp_hidden
        self.pe_num_freqs = pe_num_freqs
        self.fine_gaussian_res = fine_gaussian_res

        # ===== Store Ablation Switches =====
        self.use_fine_branch = use_fine_branch
        self.use_coarse_branch = use_coarse_branch
        self.use_alpha = use_alpha
        self.use_learnable_scale = use_learnable_scale
        self.use_pe = use_pe
        self.use_gaussian_guidance = use_gaussian_guidance
        self.use_refiner = use_refiner

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
        # Intensity/opacity alpha: [M] - Learnable intensity parameter (logit space)
        # Allows scale and intensity to jointly determine contribution weight, similar to 3DGS
        self.coeff_alpha = nn.Parameter(torch.zeros(coeff_res))  # sigmoid -> ~0.5

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
        # Intensity/opacity alpha: [N] - Learnable intensity parameter (logit space)
        self.basis_alpha = nn.Parameter(torch.zeros(basis_res))

        # ========== Fine Gaussian Parameters (NEW: for high-frequency anchoring) ==========
        # Sparse, small-scale Gaussians that learn to locate high-frequency regions
        # Position mu: [F, 3] - learns to anchor at detail regions (edges, shadows, etc.)
        self.fine_mu = nn.Parameter(torch.rand(fine_gaussian_res, 3))
        # Log-scale log_s: [F, 3] - smaller scale for local detail capture
        init_fine_log_scale = np.log(fine_kernel_scale)
        self.fine_log_s = nn.Parameter(
            torch.ones(fine_gaussian_res, 3) * init_fine_log_scale + torch.randn(fine_gaussian_res, 3) * 0.1
        )
        # Quaternion rotation q: [F, 4]
        self.fine_q = nn.Parameter(torch.zeros(fine_gaussian_res, 4))
        with torch.no_grad():
            self.fine_q[:, 0] = 1.0
        # Intensity/opacity alpha: [F] - learnable importance for each fine gaussian
        self.fine_alpha = nn.Parameter(torch.zeros(fine_gaussian_res))
        # Fine feature vectors: [F, D] - learnable features at each fine anchor
        # Small init for additive residual: fine branch starts near zero
        self.fine_features = nn.Parameter(torch.randn(fine_gaussian_res, data_dim) * 0.01)

        # ========== MBD Coefficient/Basis Tensors ==========
        # C: [M, L] - scalar coefficients at coefficient control points
        # B: [N, L, D] - basis vectors at basis control points
        self.C = nn.Parameter(torch.randn(coeff_res, self.L) * 0.1)
        self.B = nn.Parameter(torch.randn(basis_res, self.L, self.data_dim) * 0.1)
        
        # ========== MBD Learnable Scale ==========
        # Learnable scale factor for each basis l, allowing MBD scale to participate in training
        # f(x) = Σ_l scale_l * c_l(x) * b_l(x)
        # Initialized to 0 in log space (exp -> 1.0), ensuring initial behavior unchanged
        self.mbd_log_scale = nn.Parameter(torch.zeros(self.L))  # [L]

        # ========== Positional Encoding ==========
        if self.use_pe:
            self.pos_encoder = PositionalEncoding(num_freqs=pe_num_freqs)
            pe_dim = self.pos_encoder.get_output_dim(3)  # 3 + 3*2*num_freqs
        else:
            self.pos_encoder = None
            pe_dim = 3  # Raw coordinates only

        # ========== Fine Branch: Gaussian-Guided MLP ==========
        if self.use_fine_branch:
            # Input dimension depends on ablation switches
            if self.use_gaussian_guidance:
                fine_input_dim = pe_dim + fine_gaussian_res  # PE features + Gaussian weights
            else:
                fine_input_dim = pe_dim  # PE only, no Gaussian guidance
            self._fine_input_dim = fine_input_dim
            
            fine_layers = []
            fine_layers.append(nn.Linear(fine_input_dim, mlp_hidden))
            fine_layers.append(nn.ReLU())
            for _ in range(fine_mlp_depth - 1):
                fine_layers.append(nn.Linear(mlp_hidden, mlp_hidden))
                fine_layers.append(nn.ReLU())
            fine_layers.append(nn.Linear(mlp_hidden, data_dim))
            self.fine_mlp = nn.Sequential(*fine_layers)

            # Zero-init fine MLP last layer for additive residual
            with torch.no_grad():
                self.fine_mlp[-1].weight.mul_(0.01)
                self.fine_mlp[-1].bias.zero_()
        else:
            self.fine_mlp = None
            self._fine_input_dim = 0

        # ========== Residual Refinement ==========
        # NOTE: Refiner only takes blended output (no coords!)
        # If coords were included, Refiner becomes an independent function approximator,
        # which defeats the purpose of ablation experiments.
        if self.use_refiner:
            self.residual_refiner = nn.Sequential(
                nn.Linear(data_dim, mlp_hidden // 2),
                nn.ReLU(),
                nn.Linear(mlp_hidden // 2, data_dim)
            )
            with torch.no_grad():
                self.residual_refiner[-1].weight.mul_(0.01)
                self.residual_refiner[-1].bias.zero_()
        else:
            self.residual_refiner = None

        # Initialize statistics
        self.M = coeff_res
        self.N = basis_res
        self.F = fine_gaussian_res
        print(f"Ablation Model initialized:")
        print(f"  [Coarse Branch] {'ENABLED' if self.use_coarse_branch else 'DISABLED'}")
        if self.use_coarse_branch:
            print(f"    - Coefficient Gaussians: M={self.M}")
            print(f"    - Basis Gaussians: N={self.N}")
            print(f"    - Num bases: L={self.L}")
            print(f"    - Alpha: {'ON' if self.use_alpha else 'OFF'}")
            print(f"    - Learnable Scale: {'ON' if self.use_learnable_scale else 'OFF'}")
        print(f"  [Fine Branch] {'ENABLED' if self.use_fine_branch else 'DISABLED'}")
        if self.use_fine_branch:
            print(f"    - Fine Gaussians: F={self.F}")
            print(f"    - PE: {'ON (dim={})'.format(pe_dim) if self.use_pe else 'OFF (raw coords, dim=3)'}")
            print(f"    - Gaussian Guidance: {'ON' if self.use_gaussian_guidance else 'OFF'}")
            print(f"    - MLP input dim: {self._fine_input_dim}")
            print(f"    - MLP depth: {fine_mlp_depth} layers, hidden={mlp_hidden}")
        print(f"  [Residual Refiner] {'ENABLED' if self.use_refiner else 'DISABLED'}")

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

    def compute_gaussian_weights_3d(self, query_pts, mu, log_s, q, alpha=None):
        """
        计算3D高斯权重（归一化），支持可学习强度参数。

        query_pts: [Q, 3] - 查询位置
        mu: [K, 3] - 高斯中心位置
        log_s: [K, 3] - 对数尺度
        q: [K, 4] - 四元数旋转
        alpha: [K] - 强度参数（logit空间），可选
        返回: [Q, K] - 归一化高斯权重
        """
        s = torch.exp(log_s)  # [K, 3]
        gaussian_vals = self.gaussian_function_3d(query_pts, mu, s, q)  # [Q, K]

        # If alpha provided, apply intensity weighting (similar to 3DGS opacity)
        if alpha is not None:
            intensity = torch.sigmoid(alpha)  # [K] -> (0, 1)
            gaussian_vals = gaussian_vals * intensity.unsqueeze(0)  # [Q, K]

        # 归一化权重
        weights = gaussian_vals / (gaussian_vals.sum(dim=1, keepdim=True) + 1e-8)
        return weights

    def forward(self, coords, return_components=False):
        """
        Forward pass with ablation-aware hierarchical coarse-to-fine decoding.
        """
        Q = coords.shape[0]
        device = coords.device
        
        # ============ Coarse Branch: MBD Reconstruction ============
        if self.use_coarse_branch:
            # Alpha gating based on ablation switch
            coeff_alpha = self.coeff_alpha if self.use_alpha else None
            basis_alpha = self.basis_alpha if self.use_alpha else None
            
            phi_weights = self.compute_gaussian_weights_3d(
                coords, self.coeff_mu, self.coeff_log_s, self.coeff_q, coeff_alpha
            )  # [Q, M]
            psi_weights = self.compute_gaussian_weights_3d(
                coords, self.basis_mu, self.basis_log_s, self.basis_q, basis_alpha
            )  # [Q, N]

            moving_coeff = torch.matmul(phi_weights, self.C)  # [Q, L]
            B_flat = self.B.view(-1, self.L * self.data_dim)
            basis_interp_flat = torch.matmul(psi_weights, B_flat)
            moving_basis = basis_interp_flat.view(-1, self.L, self.data_dim)  # [Q, L, D]

            # Learnable scale ablation
            if self.use_learnable_scale:
                mbd_scale = torch.exp(self.mbd_log_scale)  # [L]
            else:
                mbd_scale = torch.ones(self.L, device=device)  # Fixed scale = 1
            
            scaled_coeff = moving_coeff * mbd_scale.unsqueeze(0)
            coarse_output = torch.sum(scaled_coeff.unsqueeze(-1) * moving_basis, dim=1)  # [Q, D]
        else:
            coarse_output = torch.zeros(Q, self.data_dim, device=device)
            moving_coeff = torch.zeros(Q, self.L, device=device)
            moving_basis = torch.zeros(Q, self.L, self.data_dim, device=device)

        # ============ Fine Branch: Gaussian-Guided MLP ============
        if self.use_fine_branch:
            fine_alpha = self.fine_alpha if self.use_alpha else None
            fine_weights = self.compute_gaussian_weights_3d(
                coords, self.fine_mu, self.fine_log_s, self.fine_q, fine_alpha
            )  # [Q, F]
            
            # Positional encoding ablation
            if self.use_pe:
                coords_encoded = self.pos_encoder(coords)  # [Q, pe_dim]
            else:
                coords_encoded = coords  # [Q, 3] raw coordinates
            
            # Gaussian guidance ablation
            if self.use_gaussian_guidance:
                fine_input = torch.cat([coords_encoded, fine_weights], dim=-1)  # [Q, pe_dim + F]
            else:
                fine_input = coords_encoded  # [Q, pe_dim] - no gaussian weights
            
            fine_mlp_output = self.fine_mlp(fine_input)  # [Q, D]
            fine_gaussian_interp = torch.matmul(fine_weights, self.fine_features)  # [Q, D]
            fine_output = fine_mlp_output + 0.2 * fine_gaussian_interp
        else:
            fine_output = torch.zeros(Q, self.data_dim, device=device)
            fine_weights = torch.zeros(Q, self.fine_gaussian_res, device=device)
            fine_gaussian_interp = torch.zeros(Q, self.data_dim, device=device)

        # ============ Additive Residual Blending ============
        blended = coarse_output + fine_output  # [Q, D]

        # ============ Residual Refinement (ablation) ============
        # Refiner only sees blended values, NOT coords (prevents it from becoming
        # an independent function approximator that bypasses ablation logic)
        if self.use_refiner and self.residual_refiner is not None:
            residual = self.residual_refiner(blended)
            reconstruction = blended + 0.1 * residual
        else:
            reconstruction = blended

        if return_components:
            return {
                'reconstruction': reconstruction,
                'coarse_output': coarse_output,
                'fine_output': fine_output,
                'fine_weights': fine_weights,
                'fine_gaussian_interp': fine_gaussian_interp,
                'blended': blended,
                'moving_coeff': moving_coeff,
                'moving_basis': moving_basis
            }
        
        return reconstruction, moving_coeff, moving_basis, coarse_output

    def get_compression_ratio(self, original_size, bytes_per_param=4):
        """Compute compression ratio (ablation-aware)"""
        info = self.get_branch_info()
        total_params = info['total']
        compressed_size = total_params * bytes_per_param
        ratio = original_size / compressed_size if compressed_size > 0 else float('inf')
        return ratio, compressed_size
    
    def get_branch_info(self):
        """Get detailed parameter count for each branch (ablation-aware)"""
        coarse_mbd = 0
        if self.use_coarse_branch:
            alpha_per_gauss = 1 if self.use_alpha else 0
            coeff_params = self.M * (3 + 3 + 4 + alpha_per_gauss + self.L)
            basis_params = self.N * (3 + 3 + 4 + alpha_per_gauss + self.L * self.data_dim)
            mbd_scale_params = self.L if self.use_learnable_scale else 0
            coarse_mbd = coeff_params + basis_params + mbd_scale_params
        
        fine_gaussian = 0
        fine_mlp_count = 0
        if self.use_fine_branch:
            alpha_per_fine = 1 if self.use_alpha else 0
            fine_gaussian = self.F * (3 + 3 + 4 + alpha_per_fine + self.data_dim)
            fine_mlp_count = sum(p.numel() for p in self.fine_mlp.parameters())
        
        refiner_count = 0
        if self.use_refiner and self.residual_refiner is not None:
            refiner_count = sum(p.numel() for p in self.residual_refiner.parameters())
        
        return {
            'coarse_mbd': coarse_mbd,
            'fine_gaussian': fine_gaussian,
            'fine_mlp': fine_mlp_count,
            'fine_total': fine_gaussian + fine_mlp_count,
            'gate_network': 0,
            'residual_refiner': refiner_count,
            'total': coarse_mbd + fine_gaussian + fine_mlp_count + refiner_count
        }

    def get_gaussian_params(self):
        """获取高斯参数用于可视化"""
        with torch.no_grad():
            coeff_mu = self.coeff_mu.cpu().numpy()
            coeff_s = torch.exp(self.coeff_log_s).cpu().numpy()
            coeff_q = self.coeff_q.cpu().numpy()
            coeff_q = coeff_q / (np.linalg.norm(coeff_q, axis=1, keepdims=True) + 1e-8)
            coeff_alpha = torch.sigmoid(self.coeff_alpha).cpu().numpy()

            basis_mu = self.basis_mu.cpu().numpy()
            basis_s = torch.exp(self.basis_log_s).cpu().numpy()
            basis_q = self.basis_q.cpu().numpy()
            basis_q = basis_q / (np.linalg.norm(basis_q, axis=1, keepdims=True) + 1e-8)
            basis_alpha = torch.sigmoid(self.basis_alpha).cpu().numpy()
            mbd_scale = torch.exp(self.mbd_log_scale).cpu().numpy()
            
            # Fine Gaussian parameters
            fine_mu = self.fine_mu.cpu().numpy()
            fine_s = torch.exp(self.fine_log_s).cpu().numpy()
            fine_q = self.fine_q.cpu().numpy()
            fine_q = fine_q / (np.linalg.norm(fine_q, axis=1, keepdims=True) + 1e-8)
            fine_alpha = torch.sigmoid(self.fine_alpha).cpu().numpy()
            fine_features = self.fine_features.cpu().numpy()
        return {
            'coeff_mu': coeff_mu, 'coeff_s': coeff_s, 'coeff_q': coeff_q, 'coeff_alpha': coeff_alpha,
            'basis_mu': basis_mu, 'basis_s': basis_s, 'basis_q': basis_q, 'basis_alpha': basis_alpha,
            'mbd_scale': mbd_scale,
            'fine_mu': fine_mu, 'fine_s': fine_s, 'fine_q': fine_q, 'fine_alpha': fine_alpha,
            'fine_features': fine_features
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
    Hierarchical MBD Solver with ablation-aware training strategy.
    """
    def __init__(self, model, lambda_reg=0.01, lambda_coarse=0.5, use_staged_training=True):
        self.model = model
        self.lambda_reg = lambda_reg
        self.lambda_coarse = lambda_coarse
        self.initial_lambda_coarse = lambda_coarse
        self.use_staged_training = use_staged_training

        # Build optimizer groups based on ablation switches
        all_params = []
        
        # Coarse branch params
        if model.use_coarse_branch:
            coarse_gaussian_params = [
                model.coeff_mu, model.coeff_log_s, model.coeff_q,
                model.basis_mu, model.basis_log_s, model.basis_q,
            ]
            if model.use_alpha:
                coarse_gaussian_params.extend([model.coeff_alpha, model.basis_alpha])
            mbd_params = [model.C, model.B]
            if model.use_learnable_scale:
                mbd_params.append(model.mbd_log_scale)
            
            self.optimizer_coarse_gaussian = optim.Adam(coarse_gaussian_params, lr=0.005)
            self.optimizer_mbd = optim.Adam(mbd_params, lr=0.01)
            self.scheduler_coarse_gaussian = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_coarse_gaussian, patience=50, factor=0.5)
            self.scheduler_mbd = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_mbd, patience=50, factor=0.5)
        else:
            self.optimizer_coarse_gaussian = None
            self.optimizer_mbd = None
            self.scheduler_coarse_gaussian = None
            self.scheduler_mbd = None

        # Fine branch params
        if model.use_fine_branch:
            fine_gaussian_params = [
                model.fine_mu, model.fine_log_s, model.fine_q,
                model.fine_features
            ]
            if model.use_alpha:
                fine_gaussian_params.append(model.fine_alpha)
            fine_mlp_params = list(model.fine_mlp.parameters())
            
            self.optimizer_fine_gaussian = optim.Adam(fine_gaussian_params, lr=0.008)
            self.optimizer_fine_mlp = optim.Adam(fine_mlp_params, lr=0.003)
            self.scheduler_fine_gaussian = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_fine_gaussian, patience=50, factor=0.5)
            self.scheduler_fine_mlp = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_fine_mlp, patience=50, factor=0.5)
        else:
            self.optimizer_fine_gaussian = None
            self.optimizer_fine_mlp = None
            self.scheduler_fine_gaussian = None
            self.scheduler_fine_mlp = None

        # Refiner params
        if model.use_refiner and model.residual_refiner is not None:
            refiner_params = list(model.residual_refiner.parameters())
            self.optimizer_refiner = optim.Adam(refiner_params, lr=0.003)
        else:
            self.optimizer_refiner = None

    def compute_loss(self, pred, target, coarse_output=None):
        """Compute hierarchical loss with intermediate supervision."""
        final_loss = torch.mean((pred - target) ** 2)
        
        coarse_loss = torch.tensor(0.0, device=pred.device)
        if coarse_output is not None and self.model.use_coarse_branch:
            coarse_loss = torch.mean((coarse_output - target) ** 2)
        
        # Regularization (only for enabled components)
        reg_terms = []
        if self.model.use_coarse_branch:
            coeff_s = torch.exp(self.model.coeff_log_s)
            basis_s = torch.exp(self.model.basis_log_s)
            reg_terms.append(torch.sum(self.model.C ** 2))
            reg_terms.append(torch.sum(coeff_s ** 2))
            reg_terms.append(torch.sum(basis_s ** 2))
        if self.model.use_fine_branch:
            fine_s = torch.exp(self.model.fine_log_s)
            reg_terms.append(0.5 * torch.sum(fine_s ** 2))
        
        reg_loss = self.lambda_reg * sum(reg_terms) if reg_terms else torch.tensor(0.0, device=pred.device)
        
        lambda_coarse = self.lambda_coarse
        total_loss = final_loss + lambda_coarse * coarse_loss + reg_loss
        
        return total_loss, final_loss, coarse_loss, reg_loss

    def train_step(self, coords_batch, target_batch, stage='main'):
        """Single training step with ablation-aware optimizer selection."""
        # Zero all active optimizers
        for opt in [self.optimizer_coarse_gaussian, self.optimizer_mbd,
                    self.optimizer_fine_gaussian, self.optimizer_fine_mlp,
                    self.optimizer_refiner]:
            if opt is not None:
                opt.zero_grad()

        pred, moving_coeff, moving_basis, coarse_output = self.model(coords_batch)
        total_loss, final_loss, coarse_loss, reg_loss = self.compute_loss(
            pred, target_batch, coarse_output
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Step optimizers based on stage and ablation config
        if stage == 'coarse':
            if self.optimizer_coarse_gaussian: self.optimizer_coarse_gaussian.step()
            if self.optimizer_mbd: self.optimizer_mbd.step()
        elif stage == 'fine':
            if self.optimizer_fine_gaussian: self.optimizer_fine_gaussian.step()
            if self.optimizer_fine_mlp: self.optimizer_fine_mlp.step()
            if self.optimizer_refiner: self.optimizer_refiner.step()
        else:  # 'main' - all
            if self.optimizer_coarse_gaussian: self.optimizer_coarse_gaussian.step()
            if self.optimizer_mbd: self.optimizer_mbd.step()
            if self.optimizer_fine_gaussian: self.optimizer_fine_gaussian.step()
            if self.optimizer_fine_mlp: self.optimizer_fine_mlp.step()
            if self.optimizer_refiner: self.optimizer_refiner.step()

        return {
            'total_loss': total_loss.item(),
            'final_loss': final_loss.item(),
            'coarse_loss': coarse_loss.item(),
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

    def train(self, coords, target, epochs_coarse=500, epochs_main=1500, epochs_fine=500, batch_size=2048):
        """Ablation-aware training: staged or joint."""
        losses = []
        num_samples = coords.shape[0]
        total_epochs = epochs_coarse + epochs_main + epochs_fine
        
        if not self.use_staged_training:
            # ============ A8: Joint Training (no staging) ============
            print(f"={'*'*60}")
            print(f"[ABLATION] Joint Training (no staging) - {total_epochs} epochs")
            print(f"={'*'*60}")
            self.lambda_coarse = self.initial_lambda_coarse * 0.3  # Fixed moderate weight
            
            for epoch in range(total_epochs):
                indices = torch.randperm(num_samples)[:batch_size]
                coords_batch = coords[indices]
                target_batch = target[indices]
                loss_dict = self.train_step(coords_batch, target_batch, 'main')
                losses.append(loss_dict)
                
                if epoch % 100 == 0:
                    for sch in [self.scheduler_coarse_gaussian, self.scheduler_mbd,
                                self.scheduler_fine_gaussian, self.scheduler_fine_mlp]:
                        if sch is not None:
                            sch.step(loss_dict['total_loss'])
                
                if epoch % 300 == 0 or epoch == total_epochs - 1:
                    print(f"  Epoch {epoch:4d}/{total_epochs} | "
                          f"Total: {loss_dict['total_loss']:.6f} | "
                          f"Final: {loss_dict['final_loss']:.6f}")
            
            print(f"\n[Training Complete] Total epochs: {len(losses)}")
            return losses
        
        # ============ Standard 3-Stage Training ============
        print(f"={'='*60}")
        print(f"Hierarchical Training Strategy (Coarse-to-Fine)")
        print(f"={'='*60}")
        print(f"  Stage 1 (Coarse): {epochs_coarse} epochs")
        print(f"  Stage 2 (Joint):  {epochs_main} epochs")
        print(f"  Stage 3 (Fine):   {epochs_fine} epochs")
        print(f"  Total: {total_epochs} epochs, batch_size={batch_size}")
        print(f"={'='*60}")
        
        # ============ Stage 1: Coarse Focus ============
        print(f"\n[Stage 1] Coarse Focus Training ({epochs_coarse} epochs)...")
        for epoch in range(epochs_coarse):
            indices = torch.randperm(num_samples)[:batch_size]
            coords_batch = coords[indices]
            target_batch = target[indices]
            loss_dict = self.train_step(coords_batch, target_batch, 'coarse')
            losses.append(loss_dict)
            if epoch % 100 == 0:
                if self.scheduler_coarse_gaussian: self.scheduler_coarse_gaussian.step(loss_dict['total_loss'])
                if self.scheduler_mbd: self.scheduler_mbd.step(loss_dict['total_loss'])
            if epoch % 200 == 0 or epoch == epochs_coarse - 1:
                print(f"  Epoch {epoch:4d}/{epochs_coarse} | "
                      f"Total: {loss_dict['total_loss']:.6f} | "
                      f"Coarse: {loss_dict['coarse_loss']:.6f}")

        # ============ Stage 2: Joint Training ============
        print(f"\n[Stage 2] Joint Training ({epochs_main} epochs)...")
        for epoch in range(epochs_main):
            progress = epoch / epochs_main
            self.lambda_coarse = self.initial_lambda_coarse * (1 - 0.8 * progress)
            indices = torch.randperm(num_samples)[:batch_size]
            coords_batch = coords[indices]
            target_batch = target[indices]
            loss_dict = self.train_step(coords_batch, target_batch, 'main')
            losses.append(loss_dict)
            if epoch % 100 == 0:
                for sch in [self.scheduler_coarse_gaussian, self.scheduler_mbd,
                            self.scheduler_fine_gaussian, self.scheduler_fine_mlp]:
                    if sch is not None:
                        sch.step(loss_dict['total_loss'])
            if epoch % 300 == 0 or epoch == epochs_main - 1:
                print(f"  Epoch {epoch:4d}/{epochs_main} | "
                      f"Total: {loss_dict['total_loss']:.6f} | "
                      f"Final: {loss_dict['final_loss']:.6f} | "
                      f"Coarse: {loss_dict['coarse_loss']:.6f} | "
                      f"λ_c: {self.lambda_coarse:.2f}")

        # ============ Stage 3: Fine Focus ============
        print(f"\n[Stage 3] Fine Focus Training ({epochs_fine} epochs)...")
        self.lambda_coarse = 0.1
        for epoch in range(epochs_fine):
            indices = torch.randperm(num_samples)[:batch_size]
            coords_batch = coords[indices]
            target_batch = target[indices]
            loss_dict = self.train_step(coords_batch, target_batch, 'fine')
            losses.append(loss_dict)
            if epoch % 200 == 0 or epoch == epochs_fine - 1:
                print(f"  Epoch {epoch:4d}/{epochs_fine} | "
                      f"Total: {loss_dict['total_loss']:.6f} | "
                      f"Final: {loss_dict['final_loss']:.6f}")

        print(f"\n[Training Complete] Total epochs: {len(losses)}")
        return losses

# ==================== Ablation Experiment: Model Instantiation ====================
print(f"\nCreating model with ablation config: {ABLATION_MODE}")
model = MBDCompressor3D(
    num_bases=8,
    coeff_res=16,
    basis_res=16,
    data_dim=C,
    coeff_kernel_scale=0.15,
    basis_kernel_scale=0.20,
    mlp_hidden=64,
    pe_num_freqs=6,
    fine_mlp_depth=2,
    fine_gaussian_res=16,
    fine_kernel_scale=0.05,
    # Ablation switches from config
    use_fine_branch=abl_cfg['use_fine_branch'],
    use_coarse_branch=abl_cfg['use_coarse_branch'],
    use_alpha=abl_cfg['use_alpha'],
    use_learnable_scale=abl_cfg['use_learnable_scale'],
    use_pe=abl_cfg['use_pe'],
    use_gaussian_guidance=abl_cfg['use_gaussian_guidance'],
    use_refiner=abl_cfg['use_refiner'],
)

# Print detailed architecture info
branch_info = model.get_branch_info()
print(f"\nModel Parameter Distribution:")
print(f"  Coarse (MBD):       {branch_info['coarse_mbd']:6d} params")
print(f"  Fine (Gaussian):    {branch_info['fine_gaussian']:6d} params")
print(f"  Fine (MLP):         {branch_info['fine_mlp']:6d} params")
print(f"  Fine (Total):       {branch_info['fine_total']:6d} params")
print(f"  Gate Network:       {branch_info['gate_network']:6d} params")
print(f"  Residual Refiner:   {branch_info['residual_refiner']:6d} params")
print(f"  Total:              {branch_info['total']:6d} params")

# Create solver with ablation-aware training strategy
solver = MBDSolver3D(
    model, lambda_reg=1e-5, lambda_coarse=0.5,
    use_staged_training=abl_cfg['use_staged_training']
)

# 计算原始数据大小
original_size = D * H * W * C * 4  # float32
print(f"Original size: {original_size/1024:.1f} KB")

# Train model with hierarchical strategy (Coarse -> Joint -> Fine)
print("\nStarting Hierarchical Training (Coarse-to-Fine)...")
losses = solver.train(
    coords, target_data, 
    epochs_coarse=500,    # Stage 1: Focus on MBD
    epochs_main=1500,     # Stage 2: Joint training
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
    
    reconstructed_vol = reconstructed.view(D, H, W, C).cpu().numpy()
    coarse_vol = coarse_recon.view(D, H, W, C).cpu().numpy()
    fine_vol = fine_recon.view(D, H, W, C).cpu().numpy()
    
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

# ==================== Visualization results (Hierarchical Decoding with Gaussian-Guided Fine) ====================
print("\nGenerating Hierarchical Visualization results...")

# Get middle slice for visualization
gt_slice = ground_truth[z_slice, :, :, :].numpy()  # [H, W, C]
rec_slice = reconstructed_vol[z_slice, :, :, :]    # [H, W, C]
coarse_slice = coarse_vol[z_slice, :, :, :]        # [H, W, C]
fine_slice = fine_vol[z_slice, :, :, :]            # [H, W, C]

fig = plt.figure(figsize=(28, 16))

# Row 1: Original, Final, Coarse, Fine, Fine Gaussian Positions
# 1. Original signal
ax1 = plt.subplot(3, 5, 1)
im1 = ax1.imshow(gt_slice, vmin=0, vmax=1)
ax1.set_title(f'Ground Truth (Z={z_slice})\n{D}x{H}x{W}x{C}')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.grid(False)

# 2. Final reconstruction (Coarse + Fine blended)
ax2 = plt.subplot(3, 5, 2)
im2 = ax2.imshow(rec_slice, vmin=0, vmax=1)
ax2.set_title(f'Final Output (Blended)\nPSNR: {psnr_value:.1f}dB, Ratio: {comp_ratio:.1f}:1')
ax2.set_xlabel('X')
ax2.grid(False)

# 3. Coarse branch output (MBD only)
coarse_psnr = compute_psnr(gt_slice.mean(axis=-1), coarse_slice.mean(axis=-1))
ax3 = plt.subplot(3, 5, 3)
im3 = ax3.imshow(coarse_slice, vmin=0, vmax=1)
ax3.set_title(f'Coarse Branch (MBD)\nPSNR: {coarse_psnr:.1f}dB')
ax3.set_xlabel('X')
ax3.grid(False)

# 4. Fine branch output (Gaussian + MLP)
fine_psnr = compute_psnr(gt_slice.mean(axis=-1), fine_slice.mean(axis=-1))
ax4 = plt.subplot(3, 5, 4)
im4 = ax4.imshow(fine_slice, vmin=0, vmax=1)
ax4.set_title(f'Fine Branch (Gauss+MLP)\nPSNR: {fine_psnr:.1f}dB')
ax4.set_xlabel('X')
ax4.grid(False)

# 5. Fine Gaussian positions overlay on error map
ax5 = plt.subplot(3, 5, 5)
error_for_overlay = np.abs(gt_slice - coarse_slice).mean(axis=-1)
ax5.imshow(error_for_overlay, cmap='hot', vmin=0, vmax=0.15)
# Overlay Fine Gaussian positions (projected to Z slice)
gaussian_params = model.get_gaussian_params()
fine_mu = gaussian_params['fine_mu']
fine_s = gaussian_params['fine_s']
fine_alpha = gaussian_params['fine_alpha']
# Filter gaussians near this Z slice
z_tolerance = 0.15
near_slice_mask = np.abs(fine_mu[:, 2] - z_slice / D) < z_tolerance
for i in range(len(fine_mu)):
    if near_slice_mask[i]:
        x_pos = fine_mu[i, 0] * W
        y_pos = fine_mu[i, 1] * H
        size = np.mean(fine_s[i]) * 500 * fine_alpha[i]
        ax5.scatter(x_pos, y_pos, s=size, c='cyan', alpha=0.7, edgecolors='white', linewidths=1)
ax5.set_title(f'Fine Gaussians on Coarse Error\n{near_slice_mask.sum()}/{len(fine_mu)} near Z={z_slice}')
ax5.set_xlabel('X')
ax5.grid(False)

# Row 2: Error maps and Gate visualization
# 6. Final error map
ax6 = plt.subplot(3, 5, 6)
error = np.abs(gt_slice - rec_slice)
error_img = ax6.imshow(error.mean(axis=-1), cmap='hot', vmin=0, vmax=0.15)
ax6.set_title(f'Final Error\nSSIM: {ssim_value:.4f}')
ax6.set_xlabel('X')
plt.colorbar(error_img, ax=ax6, fraction=0.046, pad=0.04)
ax6.grid(False)

# 7. Coarse error map
ax7 = plt.subplot(3, 5, 7)
coarse_error = np.abs(gt_slice - coarse_slice)
coarse_error_img = ax7.imshow(coarse_error.mean(axis=-1), cmap='hot', vmin=0, vmax=0.15)
ax7.set_title('Coarse Branch Error')
ax7.set_xlabel('X')
plt.colorbar(coarse_error_img, ax=ax7, fraction=0.046, pad=0.04)
ax7.grid(False)

# 8. Coarse vs Fine comparison
ax8 = plt.subplot(3, 5, 8)
diff_coarse_fine = np.abs(coarse_slice - fine_slice).mean(axis=-1)
diff_img = ax8.imshow(diff_coarse_fine, cmap='viridis', vmin=0, vmax=0.3)
ax8.set_title('|Coarse - Fine| Difference')
ax8.set_xlabel('X')
plt.colorbar(diff_img, ax=ax8, fraction=0.046, pad=0.04)
ax8.grid(False)

# 9. Fine branch output visualization (detailed view)
ax9 = plt.subplot(3, 5, 9)
# Normalize fine_slice for better visibility (fine output can be small residuals)
fine_display = fine_slice - fine_slice.min()
fine_max = fine_display.max()
if fine_max > 0:
    fine_display = fine_display / fine_max
fine_display = np.clip(fine_display, 0, 1)
im9 = ax9.imshow(fine_display, vmin=0, vmax=1)
ax9.set_title(f'Fine Branch (Enhanced)\nRange: [{fine_slice.min():.3f}, {fine_slice.max():.3f}]')
ax9.set_xlabel('X')
ax9.grid(False)
plt.colorbar(im9, ax=ax9, fraction=0.046, pad=0.04)

# 10. 3D Gaussian Transform Visualization (Ellipsoids)
ax10 = fig.add_subplot(3, 5, 10, projection='3d')

# Retrieve all gaussian params (coeff, basis, fine)
coeff_q_viz = gaussian_params['coeff_q']
basis_mu_viz = gaussian_params['basis_mu'] if 'basis_mu' not in dir() else gaussian_params['basis_mu']
basis_s_viz = gaussian_params['basis_s'] if 'basis_s' not in dir() else gaussian_params['basis_s']
basis_q_viz = gaussian_params['basis_q'] if 'basis_q' not in dir() else gaussian_params['basis_q']
fine_q_viz = gaussian_params['fine_q']

def draw_ellipsoid(ax, center, scale, quaternion, n_points=12, alpha=0.1, color='blue'):
    """Draw ellipsoid in 3D to represent Gaussian covariance shape"""
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points // 2)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    # Apply scale
    x = x * scale[0]
    y = y * scale[1]
    z = z * scale[2]
    # Quaternion to rotation matrix
    q_norm = quaternion / (np.linalg.norm(quaternion) + 1e-8)
    w_q, x_q, y_q, z_q = q_norm
    R = np.array([
        [1 - 2*y_q*y_q - 2*z_q*z_q, 2*x_q*y_q - 2*w_q*z_q, 2*x_q*z_q + 2*w_q*y_q],
        [2*x_q*y_q + 2*w_q*z_q, 1 - 2*x_q*x_q - 2*z_q*z_q, 2*y_q*z_q - 2*w_q*x_q],
        [2*x_q*z_q - 2*w_q*y_q, 2*y_q*z_q + 2*w_q*x_q, 1 - 2*x_q*x_q - 2*y_q*y_q]
    ])
    points = np.array([x.flatten(), y.flatten(), z.flatten()])
    rotated_points = R @ points
    x_rot = rotated_points[0, :].reshape(x.shape) + center[0]
    y_rot = rotated_points[1, :].reshape(y.shape) + center[1]
    z_rot = rotated_points[2, :].reshape(z.shape) + center[2]
    ax.plot_surface(x_rot, y_rot, z_rot, alpha=alpha, color=color, linewidth=0)

# Draw gaussian center points
coeff_mu_viz = gaussian_params['coeff_mu']
coeff_s_viz = gaussian_params['coeff_s']
basis_mu_viz = gaussian_params['basis_mu']
basis_s_viz = gaussian_params['basis_s']

ax10.scatter(coeff_mu_viz[:, 0], coeff_mu_viz[:, 1], coeff_mu_viz[:, 2],
            c='red', s=20, alpha=0.7, label=f'Coeff (M={model.M})')
ax10.scatter(basis_mu_viz[:, 0], basis_mu_viz[:, 1], basis_mu_viz[:, 2],
            c='blue', s=25, marker='s', alpha=0.7, label=f'Basis (N={model.N})')
ax10.scatter(fine_mu[:, 0], fine_mu[:, 1], fine_mu[:, 2],
            c='cyan', s=15, marker='^', alpha=0.7, label=f'Fine (F={model.F})')

# Draw ellipsoids for coeff gaussians
num_show = min(32, model.M)
indices_coeff = np.linspace(0, model.M - 1, num_show, dtype=int)
for idx in indices_coeff:
    draw_ellipsoid(ax10, coeff_mu_viz[idx], coeff_s_viz[idx], coeff_q_viz[idx], color='red', alpha=0.06)

# Draw ellipsoids for basis gaussians
num_show_b = min(32, model.N)
indices_basis = np.linspace(0, model.N - 1, num_show_b, dtype=int)
for idx in indices_basis:
    draw_ellipsoid(ax10, basis_mu_viz[idx], basis_s_viz[idx], basis_q_viz[idx], color='blue', alpha=0.06)

# Draw ellipsoids for fine gaussians
for idx in range(model.F):
    draw_ellipsoid(ax10, fine_mu[idx], fine_s[idx], fine_q_viz[idx], color='cyan', alpha=0.10)

ax10.set_xlim(0, 1)
ax10.set_ylim(0, 1)
ax10.set_zlim(0, 1)
ax10.set_title(f'3D Gaussian Ellipsoids\nM={model.M}, N={model.N}, F={model.F}')
ax10.set_xlabel('X')
ax10.set_ylabel('Y')
ax10.set_zlabel('Z')
ax10.legend(fontsize='xx-small', loc='upper left')

# Row 3: Training curves and statistics
# 11. Training loss curves
ax11 = plt.subplot(3, 5, 11)
total_losses = [l['total_loss'] for l in losses]
final_losses = [l['final_loss'] for l in losses]
coarse_losses = [l['coarse_loss'] for l in losses]

ax11.semilogy(total_losses, 'b-', linewidth=2, label='Total Loss')
ax11.semilogy(final_losses, 'g--', linewidth=1.5, alpha=0.7, label='Final Loss')
ax11.semilogy(coarse_losses, 'r:', linewidth=1.5, alpha=0.7, label='Coarse Loss')

# Mark training stages
epochs_coarse, epochs_main, epochs_fine = 500, 1500, 500
ax11.axvline(x=epochs_coarse, color='orange', linestyle=':', alpha=0.7, label='Stage 1→2')
ax11.axvline(x=epochs_coarse + epochs_main, color='purple', linestyle=':', alpha=0.7, label='Stage 2→3')
ax11.set_title('Training Loss (3 Stages)')
ax11.set_xlabel('Iterations')
ax11.set_ylabel('Loss Value')
ax11.legend(fontsize='x-small', loc='upper right')
ax11.grid(True, alpha=0.3)

# 12. Lambda coarse evolution during training
ax12 = plt.subplot(3, 5, 12)
# Reconstruct lambda_coarse values during training
epochs_coarse_stage, epochs_main_stage, epochs_fine_stage = 500, 1500, 500
lambda_values = []
initial_lambda = 0.5
for i in range(len(losses)):
    if i < epochs_coarse_stage:
        lambda_values.append(initial_lambda)
    elif i < epochs_coarse_stage + epochs_main_stage:
        progress = (i - epochs_coarse_stage) / epochs_main_stage
        lambda_values.append(initial_lambda * (1 - 0.8 * progress))
    else:
        lambda_values.append(0.1)

ax12.plot(lambda_values, 'b-', linewidth=2, label='λ_coarse')
ax12.axvline(x=epochs_coarse_stage, color='orange', linestyle=':', alpha=0.7)
ax12.axvline(x=epochs_coarse_stage + epochs_main_stage, color='purple', linestyle=':', alpha=0.7)
ax12.set_title('λ_coarse Evolution (Curriculum)')
ax12.set_xlabel('Iterations')
ax12.set_ylabel('λ_coarse')
ax12.legend(fontsize='small')
ax12.grid(True, alpha=0.3)

# 13. Channel comparison (1D line)
ax13 = plt.subplot(3, 5, 13)
y_line = H // 2
channel_colors = ['#E74C3C', '#27AE60', '#3498DB']
channel_names = ['R', 'G', 'B']

for c in range(C):
    ax13.plot(gt_slice[y_line, :, c],
              color=channel_colors[c], linestyle='-', alpha=0.6, linewidth=1.5,
              label=f'GT {channel_names[c]}')
    ax13.plot(rec_slice[y_line, :, c],
              color=channel_colors[c], linestyle='--', alpha=0.9, linewidth=1.5,
              label=f'Pred {channel_names[c]}')

ax13.set_title(f'1D Fitting (Z={z_slice}, Y={y_line})')
ax13.set_xlabel('X Coordinate')
ax13.set_ylabel('Value')
ax13.legend(loc='upper right', fontsize='x-small', ncol=2)
ax13.grid(True, alpha=0.3)

# 14. Fine Gaussian scale distribution
ax14 = plt.subplot(3, 5, 14)
fine_s_mean = fine_s.mean(axis=1)
coeff_s = gaussian_params['coeff_s']
coeff_s_mean = coeff_s.mean(axis=1)
ax14.hist(coeff_s_mean, bins=15, alpha=0.6, color='blue', label=f'Coarse (M={model.M})', edgecolor='black')
ax14.hist(fine_s_mean, bins=15, alpha=0.6, color='cyan', label=f'Fine (F={model.F})', edgecolor='black')
ax14.set_title('Gaussian Scale Distribution')
ax14.set_xlabel('Mean Scale')
ax14.set_ylabel('Count')
ax14.legend()
ax14.grid(True, alpha=0.3)

# 15. Summary information
ax15 = plt.subplot(3, 5, 15)
ax15.axis('off')
num_probes = D * H * W

coeff_alpha = gaussian_params['coeff_alpha']
basis_alpha = gaussian_params['basis_alpha']
mbd_scale = gaussian_params['mbd_scale']

info_text = f"""
Ablation: {ABLATION_MODE}
{abl_cfg['description']}
===============================================
Original Data:
  Volume: {D}x{H}x{W}x{C} = {num_probes} points
  Size: {original_size/1024:.1f} KB

Ablation Switches:
  Coarse: {'ON' if model.use_coarse_branch else 'OFF'}
  Fine: {'ON' if model.use_fine_branch else 'OFF'}
  Alpha: {'ON' if model.use_alpha else 'OFF'}
  Scale: {'ON' if model.use_learnable_scale else 'OFF'}
  PE: {'ON' if model.use_pe else 'OFF'}
  GaussGuide: {'ON' if model.use_gaussian_guidance else 'OFF'}
  Refiner: {'ON' if model.use_refiner else 'OFF'}
  Staged: {'ON' if abl_cfg['use_staged_training'] else 'OFF'}

Total params: {branch_info['total']}
Compression: {comp_ratio_fp16:.1f}:1 (FP16)

Reconstruction Quality:
  Final PSNR: {psnr_value:.1f} dB
  Final SSIM: {ssim_value:.4f}
  Coarse PSNR: {coarse_psnr:.1f} dB
"""
ax15.text(0.02, 0.5, info_text, fontsize=8,
          family='monospace', verticalalignment='center')

plt.suptitle(f'Ablation: {ABLATION_MODE} - {abl_cfg["description"]}', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()

print("\nAblation Experiment completed!")
print("="*70)
print(f"Ablation: {ABLATION_MODE}")
print(f"Description: {abl_cfg['description']}")
print("="*70)
print(f"Ablation Switches:")
for key, val in abl_cfg.items():
    if key != 'description':
        print(f"  {key}: {val}")
print(f"\nResults:")
print(f"  Total params: {branch_info['total']}")
print(f"  Compression (FP32): {comp_ratio_fp32:.1f}:1 ({comp_size_fp32/1024:.2f} KB)")
print(f"  Compression (FP16): {comp_ratio_fp16:.1f}:1 ({comp_size_fp16/1024:.2f} KB)")
print(f"  Final PSNR: {psnr_value:.1f} dB")
print(f"  Final SSIM: {ssim_value:.4f}")
print(f"  Coarse-only PSNR: {coarse_psnr:.1f} dB")
print("="*70)
