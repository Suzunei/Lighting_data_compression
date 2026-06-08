import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

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

# ==================== Step 1: Load ILCSampleData probe data ====================
print("Step 1: Loading ILCSampleData probe data...")

def load_ilc_probe_data(bin_path):
    """
    读取ILCSampleData二进制探针数据。
    
    每探针32 floats:
        - Position: 3 floats (x, y, z)
        - Radius: 1 float
        - SH coefficients: 27 floats (9 SH bands × 3 RGB channels, RGB交错存储)
          存储顺序: [SH0_R, SH0_G, SH0_B, SH1_R, SH1_G, SH1_B, ..., SH8_R, SH8_G, SH8_B]
        - Shadow: 1 float
    
    Returns:
        positions: [N, 3] 归一化到[0,1]的探针位置
        sh_coeffs: [N, 27] SH系数 (RGB交错)
        radii: [N] 探针半径
        shadows: [N] 阴影值
        pos_min, pos_max: 原始位置范围 (用于反归一化)
    """
    # 读取二进制数据
    raw_data = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 32)
    num_probes = raw_data.shape[0]
    
    # 解析各字段
    positions = raw_data[:, 0:3]     # [N, 3] 原始世界坐标
    radii = raw_data[:, 3]           # [N]
    sh_coeffs = raw_data[:, 4:31]    # [N, 27] SH系数 (RGB interleaved)
    shadows = raw_data[:, 31]        # [N]
    
    # 归一化位置到 [0, 1] 范围
    pos_min = positions.min(axis=0)
    pos_max = positions.max(axis=0)
    pos_range = pos_max - pos_min
    pos_range[pos_range < 1e-6] = 1.0  # 防止除零
    positions_normalized = (positions - pos_min) / pos_range
    
    print(f"  Loaded {num_probes} probes from {os.path.basename(bin_path)}")
    print(f"  Position range: X[{pos_min[0]:.1f}, {pos_max[0]:.1f}], "
          f"Y[{pos_min[1]:.1f}, {pos_max[1]:.1f}], Z[{pos_min[2]:.1f}, {pos_max[2]:.1f}]")
    print(f"  Radius range: [{radii.min():.2f}, {radii.max():.2f}]")
    print(f"  SH range: [{sh_coeffs.min():.6f}, {sh_coeffs.max():.6f}]")
    print(f"  SH mean: {sh_coeffs.mean():.6f}, std: {sh_coeffs.std():.6f}")
    print(f"  Shadow range: [{shadows.min():.4f}, {shadows.max():.4f}]")
    
    return positions_normalized, sh_coeffs, radii, shadows, pos_min, pos_max

# ===== 加载探针数据 =====
BIN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ILCSampleData_0.bin')
positions_np, sh_coeffs_np, radii_np, shadows_np, pos_min, pos_max = load_ilc_probe_data(BIN_PATH)

# 转为PyTorch张量
coords = torch.from_numpy(positions_np).float()           # [N, 3] 归一化坐标
target_data = torch.from_numpy(sh_coeffs_np).float()      # [N, 27] SH系数作为压缩目标

N = coords.shape[0]  # 探针数量
C = target_data.shape[1]  # 数据维度 = 27 (SH coefficients)

print(f"\n  Training data prepared:")
print(f"    Num probes (N): {N}")
print(f"    SH channels (C): {C} (9 SH bands × 3 RGB, interleaved)")
print(f"    Coords shape: {coords.shape}")
print(f"    Target shape: {target_data.shape}")

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
        
        Fine Branch (Gaussian + MLP): High-frequency details with spatial awareness
            φ_fine(x) = Gaussian weights from fine anchors
            f_fine(x) = MLP(PE(x), φ_fine(x)) - Gaussian-guided high-freq learning
        
        Adaptive Blending: Additive residual fusion
            f_final(x) = f_coarse(x) + f_fine(x)
    
    Key Innovation: Hierarchical decoding with Gaussian-guided Fine Branch:
        - MBD (large Gaussians): smooth, low-frequency components (compression)
        - Fine Gaussians (small, sparse): anchor high-frequency detail regions
        - MLP with Gaussian guidance: capture localized high-frequency details
        - Gate: adaptively blend based on local complexity
    """
    def __init__(self, num_bases=6, coeff_res=12, basis_res=8, data_dim=3,
                 coeff_kernel_scale=0.15, basis_kernel_scale=0.2, mlp_hidden=64,
                 pe_num_freqs=4, fine_mlp_depth=2, fine_gaussian_res=16, fine_kernel_scale=0.08):
        super().__init__()
        self.L = num_bases
        self.data_dim = data_dim
        self.mlp_hidden = mlp_hidden
        self.pe_num_freqs = pe_num_freqs
        self.fine_gaussian_res = fine_gaussian_res

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
        self.pos_encoder = PositionalEncoding(num_freqs=pe_num_freqs)
        pe_dim = self.pos_encoder.get_output_dim(3)  # 3 + 3*2*num_freqs

        # ========== Fine Branch: Gaussian-Guided MLP ==========
        # Input: PE(x) + fine_gaussian_weights -> learns high-frequency with spatial awareness
        # The gaussian weights provide "where am I relative to detail anchors" information
        fine_input_dim = pe_dim + fine_gaussian_res  # PE features + Gaussian weights
        fine_layers = []
        fine_layers.append(nn.Linear(fine_input_dim, mlp_hidden))
        fine_layers.append(nn.ReLU())
        for _ in range(fine_mlp_depth - 1):
            fine_layers.append(nn.Linear(mlp_hidden, mlp_hidden))
            fine_layers.append(nn.ReLU())
        fine_layers.append(nn.Linear(mlp_hidden, data_dim))
        self.fine_mlp = nn.Sequential(*fine_layers)

        # ========== Zero-init fine MLP last layer ==========
        # For additive residual: fine branch should start outputting ~0
        # so that initial output ≈ coarse_output, not coarse + random_noise
        with torch.no_grad():
            self.fine_mlp[-1].weight.mul_(0.01)
            self.fine_mlp[-1].bias.zero_()

        # ========== No Gate: Additive Residual Blending ==========
        # f_final = f_coarse + f_fine (fine learns residual)

        # ========== Residual Refinement (Optional Enhancement) ==========
        # Small MLP to refine the final output
        self.residual_refiner = nn.Sequential(
            nn.Linear(data_dim + 3, mlp_hidden // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden // 2, data_dim)
        )
        # Zero-init refiner last layer too
        with torch.no_grad():
            self.residual_refiner[-1].weight.mul_(0.01)
            self.residual_refiner[-1].bias.zero_()

        # Initialize statistics
        self.M = coeff_res
        self.N = basis_res
        self.F = fine_gaussian_res
        print(f"Hierarchical MBD+Gaussian+MLP model initialized (Coarse-to-Fine):")
        print(f"  [Coarse Branch] MBD with 3D Gaussians:")
        print(f"    - Coefficient Gaussians: M={self.M} (position + scale + rotation + alpha)")
        print(f"    - Basis Gaussians: N={self.N} (position + scale + rotation + alpha)")
        print(f"    - Num bases: L={self.L}")
        print(f"    - MBD Learnable Scale: [L={self.L}] per-basis scale factors")
        print(f"  [Fine Branch] Gaussian-Guided MLP:")
        print(f"    - Fine Gaussians: F={self.F} (small-scale anchors for high-freq)")
        print(f"    - Fine kernel scale: {fine_kernel_scale:.3f}")
        print(f"    - PE frequencies: {pe_num_freqs} -> dim {pe_dim}")
        print(f"    - MLP input: PE({pe_dim}) + GaussianWeights({fine_gaussian_res}) = {fine_input_dim}")
        print(f"    - MLP depth: {fine_mlp_depth} layers, hidden={mlp_hidden}")
        print(f"  [Blending] Additive Residual (coarse + fine, no gate)")
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
        # 1. Compute 3D Gaussian weights for C and B (with learnable alpha intensity)
        phi_weights = self.compute_gaussian_weights_3d(
            coords, self.coeff_mu, self.coeff_log_s, self.coeff_q, self.coeff_alpha
        )  # [Q, M]
        psi_weights = self.compute_gaussian_weights_3d(
            coords, self.basis_mu, self.basis_log_s, self.basis_q, self.basis_alpha
        )  # [Q, N]

        # 2. Compute moving coefficients c_l(x) = Σ_m φ_m(x) * C_{m,l}
        moving_coeff = torch.matmul(phi_weights, self.C)  # [Q, L]

        # 3. Compute moving bases b_l(x) = Σ_n ψ_n(x) * B_{n,l}
        B_flat = self.B.view(-1, self.L * self.data_dim)  # [N, L*D]
        basis_interp_flat = torch.matmul(psi_weights, B_flat)  # [Q, L*D]
        moving_basis = basis_interp_flat.view(-1, self.L, self.data_dim)  # [Q, L, D]

        # 4. MBD output (Coarse) with learnable scale: f_coarse(x) = Σ_l scale_l * c_l(x) * b_l(x)
        mbd_scale = torch.exp(self.mbd_log_scale)  # [L] - learnable scale factor
        scaled_coeff = moving_coeff * mbd_scale.unsqueeze(0)  # [Q, L] * [1, L] = [Q, L]
        coarse_output = torch.sum(scaled_coeff.unsqueeze(-1) * moving_basis, dim=1)  # [Q, D]

        # ============ Fine Branch: Gaussian-Guided MLP ============
        # 5. Compute Fine Gaussian weights (for high-frequency region awareness)
        fine_weights = self.compute_gaussian_weights_3d(
            coords, self.fine_mu, self.fine_log_s, self.fine_q, self.fine_alpha
        )  # [Q, F]
        
        # 6. Apply positional encoding to coordinates
        coords_encoded = self.pos_encoder(coords)  # [Q, pe_dim]
        
        # 7. Concatenate PE features with Fine Gaussian weights for spatial awareness
        fine_input = torch.cat([coords_encoded, fine_weights], dim=-1)  # [Q, pe_dim + F]
        
        # 8. Fine MLP output: high-frequency details with Gaussian guidance
        fine_mlp_output = self.fine_mlp(fine_input)  # [Q, D]
        
        # 9. Direct Gaussian interpolation for fine features (additional detail source)
        fine_gaussian_interp = torch.matmul(fine_weights, self.fine_features)  # [Q, D]
        
        # 10. Combine MLP output with Gaussian-interpolated features
        fine_output = fine_mlp_output + 0.2 * fine_gaussian_interp  # [Q, D]

        # ============ Additive Residual Blending (No Gate) ============
        # 11. Fine branch learns residual correction on top of coarse
        blended = coarse_output + fine_output  # [Q, D]

        # ============ Residual Refinement ============
        # 13. Small residual correction
        refine_input = torch.cat([blended, coords], dim=1)  # [Q, D+3]
        residual = self.residual_refiner(refine_input)  # [Q, D]
        
        # 14. Final output with residual connection
        reconstruction = blended + 0.1 * residual  # [Q, D]

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
        """Compute compression ratio (including all network parameters)
        
        Args:
            original_size: original data size in bytes
            bytes_per_param: bytes per parameter (4 for float32, 2 for float16)
        """
        # 3D Gaussian params: mu(3) + log_s(3) + q(4) + alpha(1) = 11 per gaussian
        coeff_params = self.M * (3 + 3 + 4 + 1 + self.L)
        basis_params = self.N * (3 + 3 + 4 + 1 + self.L * self.data_dim)
        mbd_scale_params = self.L  # MBD learnable scale
        
        # Fine Gaussian params: mu(3) + log_s(3) + q(4) + alpha(1) + features(D) = 11 + D per gaussian
        fine_gaussian_params = self.F * (3 + 3 + 4 + 1 + self.data_dim)

        # Count all network parameters
        fine_mlp_params = sum(p.numel() for p in self.fine_mlp.parameters())
        gate_params = 0  # No gate network
        refiner_params = sum(p.numel() for p in self.residual_refiner.parameters())

        total_params = coeff_params + basis_params + mbd_scale_params + fine_gaussian_params + fine_mlp_params + gate_params + refiner_params
        compressed_size = total_params * bytes_per_param  # 支持不同精度
        ratio = original_size / compressed_size
        return ratio, compressed_size
    
    def get_branch_info(self):
        """Get detailed parameter count for each branch"""
        # 3D Gaussian params: mu(3) + log_s(3) + q(4) + alpha(1) = 11 per gaussian
        coeff_params = self.M * (3 + 3 + 4 + 1 + self.L)
        basis_params = self.N * (3 + 3 + 4 + 1 + self.L * self.data_dim)
        mbd_scale_params = self.L  # MBD learnable scale
        
        # Fine Gaussian params: mu(3) + log_s(3) + q(4) + alpha(1) + features(D)
        fine_gaussian_params = self.F * (3 + 3 + 4 + 1 + self.data_dim)
        fine_mlp_params = sum(p.numel() for p in self.fine_mlp.parameters())
        
        gate_params = 0  # No gate network
        refiner_params = sum(p.numel() for p in self.residual_refiner.parameters())
        
        return {
            'coarse_mbd': coeff_params + basis_params + mbd_scale_params,
            'fine_gaussian': fine_gaussian_params,
            'fine_mlp': fine_mlp_params,
            'fine_total': fine_gaussian_params + fine_mlp_params,
            'gate_network': gate_params,
            'residual_refiner': refiner_params,
            'total': coeff_params + basis_params + mbd_scale_params + fine_gaussian_params + fine_mlp_params + gate_params + refiner_params
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
    Hierarchical MBD Solver with Coarse-to-Fine Training Strategy.
    
    Training Strategy:
        Stage 1 (Coarse Focus): Train MBD branch with higher weight on coarse loss
        Stage 2 (Fine Focus): Gradually shift focus to fine branch (Gaussian + MLP)
        Stage 3 (Joint Refinement): Fine-tune all branches together
    """
    def __init__(self, model, lambda_reg=0.01, lambda_coarse=0.5, target_data=None):
        self.model = model
        self.lambda_reg = lambda_reg
        self.lambda_coarse = lambda_coarse  # Weight for coarse supervision
        self.initial_lambda_coarse = lambda_coarse

        # ===== Per-channel loss weighting (variance-inverse) =====
        # Balances gradient across SH orders so weak channels (L1/L2) get fair training
        if target_data is not None and model.data_dim > 3:
            with torch.no_grad():
                ch_var = target_data.var(dim=0)  # [C] per-channel variance
                ch_weights = 1.0 / (ch_var + 1e-6)
                ch_weights = ch_weights / ch_weights.mean()  # normalize to mean=1
                self.channel_weights = ch_weights  # [C]
                print(f"  [Channel Weighting] Enabled for {model.data_dim}D data")
                print(f"    Variance range: [{ch_var.min():.6f}, {ch_var.max():.6f}]")
                print(f"    Weight range:   [{ch_weights.min():.2f}, {ch_weights.max():.2f}]")
        else:
            self.channel_weights = None

        # Separate parameter groups for different learning dynamics
        # Coarse Gaussian params (for MBD)
        coarse_gaussian_params = [
            self.model.coeff_mu, self.model.coeff_log_s, self.model.coeff_q, self.model.coeff_alpha,
            self.model.basis_mu, self.model.basis_log_s, self.model.basis_q, self.model.basis_alpha
        ]
        mbd_params = [self.model.C, self.model.B, self.model.mbd_log_scale]
        
        # Fine Gaussian params (NEW: for high-frequency anchoring)
        fine_gaussian_params = [
            self.model.fine_mu, self.model.fine_log_s, self.model.fine_q, 
            self.model.fine_alpha, self.model.fine_features
        ]
        
        # Fine MLP params
        fine_mlp_params = list(self.model.fine_mlp.parameters())
        refiner_params = list(self.model.residual_refiner.parameters())

        # Different learning rates for different components
        self.optimizer_coarse_gaussian = optim.Adam(coarse_gaussian_params, lr=0.005)
        self.optimizer_mbd = optim.Adam(mbd_params, lr=0.01)
        self.optimizer_fine_gaussian = optim.Adam(fine_gaussian_params, lr=0.008)  # Fine Gaussians
        self.optimizer_fine_mlp = optim.Adam(fine_mlp_params, lr=0.003)
        self.optimizer_refiner = optim.Adam(refiner_params, lr=0.003)

        self.scheduler_coarse_gaussian = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_coarse_gaussian, patience=50, factor=0.5
        )
        self.scheduler_mbd = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_mbd, patience=50, factor=0.5
        )
        self.scheduler_fine_gaussian = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_fine_gaussian, patience=50, factor=0.5
        )
        self.scheduler_fine_mlp = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_fine_mlp, patience=50, factor=0.5
        )

    def compute_loss(self, pred, target, coarse_output=None):
        """
        Compute hierarchical loss with intermediate supervision.
        Per-channel weighting ensures balanced gradients across SH orders.
        """
        # 1. Final reconstruction loss (weighted MSE)
        sq_err = (pred - target) ** 2  # [B, C]
        if self.channel_weights is not None:
            w = self.channel_weights.to(pred.device)  # [C]
            final_loss = torch.mean(sq_err * w.unsqueeze(0))
        else:
            final_loss = torch.mean(sq_err)
        
        # 2. Coarse branch intermediate supervision (same weighting)
        coarse_loss = torch.tensor(0.0, device=pred.device)
        if coarse_output is not None:
            coarse_sq_err = (coarse_output - target) ** 2
            if self.channel_weights is not None:
                coarse_loss = torch.mean(coarse_sq_err * w.unsqueeze(0))
            else:
                coarse_loss = torch.mean(coarse_sq_err)
        
        # 3. Regularization: prevent scale explosion
        coeff_s = torch.exp(self.model.coeff_log_s)
        basis_s = torch.exp(self.model.basis_log_s)
        fine_s = torch.exp(self.model.fine_log_s)
        reg_loss = self.lambda_reg * (
            torch.sum(self.model.C ** 2) +
            torch.sum(coeff_s ** 2) + torch.sum(basis_s ** 2) +
            0.5 * torch.sum(fine_s ** 2)  # Smaller regularization for fine (want small scales)
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
            'fine': Fine-tune with focus on fine branch (Gaussian + MLP)
        """
        # Zero gradients for all optimizers
        self.optimizer_coarse_gaussian.zero_grad()
        self.optimizer_mbd.zero_grad()
        self.optimizer_fine_gaussian.zero_grad()
        self.optimizer_fine_mlp.zero_grad()
        self.optimizer_refiner.zero_grad()

        # Forward pass with component outputs
        pred, moving_coeff, moving_basis, coarse_output = self.model(coords_batch)

        # Compute hierarchical loss
        total_loss, final_loss, coarse_loss, reg_loss = self.compute_loss(
            pred, target_batch, coarse_output
        )

        # Backward pass
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Optimization based on stage
        if stage == 'coarse':
            # Focus on coarse branch
            self.optimizer_coarse_gaussian.step()
            self.optimizer_mbd.step()
        elif stage == 'fine':
            # Focus on fine branch (Gaussian + MLP)
            self.optimizer_fine_gaussian.step()
            self.optimizer_fine_mlp.step()
            self.optimizer_refiner.step()
        else:  # 'main'
            # Train all branches
            self.optimizer_coarse_gaussian.step()
            self.optimizer_mbd.step()
            self.optimizer_fine_gaussian.step()
            self.optimizer_fine_mlp.step()
            self.optimizer_refiner.step()

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
        """
        Three-stage hierarchical training:
        
        Stage 1 (Coarse Focus): Train MBD to capture low-frequency structure
            - High λ_coarse weight
            - Only update Coarse Gaussian and MBD parameters
            
        Stage 2 (Joint Training): Train all branches together
            - Gradually decrease λ_coarse
            - Update all parameters (including Fine Gaussians)
            
        Stage 3 (Fine Focus): Refine high-frequency details
            - Low λ_coarse weight
            - Focus on Fine Gaussians + MLP + Gate
        """
        losses = []
        num_samples = coords.shape[0]
        total_epochs = epochs_coarse + epochs_main + epochs_fine
        
        print(f"="*60)
        print(f"Hierarchical Training Strategy (Coarse-to-Fine)")
        print(f"  with Gaussian-Guided Fine Branch")
        print(f"="*60)
        print(f"  Stage 1 (Coarse): {epochs_coarse} epochs - MBD focus")
        print(f"  Stage 2 (Joint):  {epochs_main} epochs - All branches")
        print(f"  Stage 3 (Fine):   {epochs_fine} epochs - Gaussian+MLP detail refinement")
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
                self.scheduler_coarse_gaussian.step(loss_dict['total_loss'])
                self.scheduler_mbd.step(loss_dict['total_loss'])

            if epoch % 200 == 0 or epoch == epochs_coarse - 1:
                print(f"  Epoch {epoch:4d}/{epochs_coarse} | "
                      f"Total: {loss_dict['total_loss']:.6f} | "
                      f"Coarse: {loss_dict['coarse_loss']:.6f}")

        # ============ Stage 2: Joint Training ============
        print(f"\n[Stage 2] Joint Training ({epochs_main} epochs)...")
        print(f"  Training Fine Gaussians to locate high-frequency regions...")
        
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
                self.scheduler_coarse_gaussian.step(loss_dict['total_loss'])
                self.scheduler_mbd.step(loss_dict['total_loss'])
                self.scheduler_fine_gaussian.step(loss_dict['total_loss'])
                self.scheduler_fine_mlp.step(loss_dict['total_loss'])

            if epoch % 300 == 0 or epoch == epochs_main - 1:
                print(f"  Epoch {epoch:4d}/{epochs_main} | "
                      f"Total: {loss_dict['total_loss']:.6f} | "
                      f"Final: {loss_dict['final_loss']:.6f} | "
                      f"Coarse: {loss_dict['coarse_loss']:.6f} | "
                      f"λ_c: {self.lambda_coarse:.2f}")

        # ============ Stage 3: Fine Focus ============
        print(f"\n[Stage 3] Fine Focus Training ({epochs_fine} epochs)...")
        print(f"  Refining Fine Gaussians + MLP for high-frequency details...")
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
                      f"Final: {loss_dict['final_loss']:.6f}")

        print(f"\n[Training Complete] Total epochs: {len(losses)}")
        return losses

# ==================== Hierarchical MBD + Gaussian + MLP with Coarse-to-Fine Decoding ====================
# Create model with hierarchical architecture for SH compression
# 27D SH data requires more capacity than 3D RGB
model = MBDCompressor3D(
    num_bases=16,              # Number of bases L (more for 27D)
    coeff_res=64,             # Coefficient 3D Gaussians M (Coarse)
    basis_res=64,             # Basis 3D Gaussians N (Coarse)
    data_dim=C,               # Data dimension D = 27 (SH coefficients)
    coeff_kernel_scale=0.12,  # Initial scale (Coarse - large, scattered probes)
    basis_kernel_scale=0.18,  # Initial scale (Coarse - large)
    mlp_hidden=128,            # MLP hidden size (larger for 27D output)
    pe_num_freqs=6,           # Positional encoding frequencies
    fine_mlp_depth=3,         # Fine branch MLP depth (deeper for SH detail)
    fine_gaussian_res=32,     # Fine Gaussians F (more anchors for scattered data)
    fine_kernel_scale=0.05    # Fine Gaussian scale (small for local detail)
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

# Create solver with hierarchical training strategy
solver = MBDSolver3D(model, lambda_reg=1e-5, lambda_coarse=0.5, target_data=target_data)

# 计算原始数据大小 (N个探针 × 27 SH系数 × 4 bytes)
original_size = N * C * 4  # float32
print(f"Original size: {original_size/1024:.1f} KB ({N} probes × {C} channels × 4B)")

# Train model with hierarchical strategy (Coarse -> Joint -> Fine)
print("\nStarting Hierarchical Training (Coarse-to-Fine)...")
losses = solver.train(
    coords, target_data, 
    epochs_coarse=500,    # Stage 1: Focus on MBD
    epochs_main=2500,     # Stage 2: Joint training
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
print("\nStep 3: Evaluating SH compression quality...")

# Reconstruct all probes using trained model
model.eval()
with torch.no_grad():
    # Process in batches to avoid OOM for large probe sets
    batch_size_eval = 8192
    reconstructed_list = []
    coarse_list = []
    fine_list = []
    
    for i in range(0, N, batch_size_eval):
        batch_coords = coords[i:i+batch_size_eval]
        results = model(batch_coords, return_components=True)
        reconstructed_list.append(results['reconstruction'].cpu())
        coarse_list.append(results['coarse_output'].cpu())
        fine_list.append(results['fine_output'].cpu())
    
    reconstructed = torch.cat(reconstructed_list, dim=0).numpy()  # [N, 27]
    coarse_recon = torch.cat(coarse_list, dim=0).numpy()          # [N, 27]
    fine_recon = torch.cat(fine_list, dim=0).numpy()              # [N, 27]

ground_truth_np = target_data.numpy()  # [N, 27]

# Compute per-channel PSNR
def compute_psnr_channel(gt, pred):
    """Compute PSNR for a single channel (1D arrays)"""
    mse = np.mean((gt - pred) ** 2)
    if mse == 0:
        return float('inf')
    # Use data range for PSNR calculation
    data_range = gt.max() - gt.min()
    if data_range < 1e-8:
        data_range = 1.0
    psnr = 20 * np.log10(data_range / np.sqrt(mse))
    return psnr

def compute_ssim_channel(gt, pred, window_size=101):
    """
    Compute SSIM for scattered probe data (1D signal per channel).
    Uses sliding window statistics along spatially-sorted probes.
    
    For scattered data, we sort probes by their spatial index (Morton/Z-order)
    and compute local SSIM statistics using a sliding window.
    
    gt, pred: [N] arrays - ground truth and prediction for one channel
    window_size: local window for statistics computation
    Returns: scalar SSIM value
    """
    gt = gt.astype(np.float64)
    pred = pred.astype(np.float64)
    
    n = len(gt)
    if n < window_size:
        # Fallback: compute global SSIM
        mu_x = np.mean(gt)
        mu_y = np.mean(pred)
        sigma_x_sq = np.var(gt)
        sigma_y_sq = np.var(pred)
        sigma_xy = np.mean((gt - mu_x) * (pred - mu_y))
        
        data_range = max(gt.max() - gt.min(), pred.max() - pred.min(), 1e-8)
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        
        ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x**2 + mu_y**2 + C1) * (sigma_x_sq + sigma_y_sq + C2))
        return ssim
    
    # Use cumulative sums for efficient sliding window
    # Pad for 'valid' convolution equivalent
    half_w = window_size // 2
    
    # Compute local means using uniform filter (sliding average)
    from scipy.ndimage import uniform_filter1d
    mu_x = uniform_filter1d(gt, size=window_size, mode='reflect')
    mu_y = uniform_filter1d(pred, size=window_size, mode='reflect')
    
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y
    
    sigma_x_sq = uniform_filter1d(gt ** 2, size=window_size, mode='reflect') - mu_x_sq
    sigma_y_sq = uniform_filter1d(pred ** 2, size=window_size, mode='reflect') - mu_y_sq
    sigma_xy = uniform_filter1d(gt * pred, size=window_size, mode='reflect') - mu_xy
    
    # Clamp negative variances (numerical)
    sigma_x_sq = np.maximum(sigma_x_sq, 0)
    sigma_y_sq = np.maximum(sigma_y_sq, 0)
    
    # Dynamic data range
    data_range = max(gt.max() - gt.min(), pred.max() - pred.min(), 1e-8)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
    
    return np.mean(ssim_map)

# Spatial sorting for SSIM: sort probes by Morton code (Z-order curve)
# This ensures spatially nearby probes are adjacent for local SSIM computation
def morton_code_3d(coords_np, bits=10):
    """Compute Morton (Z-order) code for 3D coordinates in [0,1]"""
    # Quantize to integer grid
    max_val = (1 << bits) - 1
    ix = np.clip((coords_np[:, 0] * max_val).astype(np.int64), 0, max_val)
    iy = np.clip((coords_np[:, 1] * max_val).astype(np.int64), 0, max_val)
    iz = np.clip((coords_np[:, 2] * max_val).astype(np.int64), 0, max_val)
    
    # Interleave bits (simplified for reasonable precision)
    code = np.zeros(len(ix), dtype=np.int64)
    for b in range(bits):
        code |= ((ix >> b) & 1).astype(np.int64) << (3 * b)
        code |= ((iy >> b) & 1).astype(np.int64) << (3 * b + 1)
        code |= ((iz >> b) & 1).astype(np.int64) << (3 * b + 2)
    return code

# Sort probes spatially for meaningful SSIM
morton_codes = morton_code_3d(coords.numpy())
spatial_sort_idx = np.argsort(morton_codes)
gt_sorted = ground_truth_np[spatial_sort_idx]      # [N, 27] spatially sorted
rec_sorted = reconstructed[spatial_sort_idx]        # [N, 27] spatially sorted
coarse_sorted = coarse_recon[spatial_sort_idx]      # [N, 27] spatially sorted

# Per-channel metrics (PSNR + SSIM)
psnr_values = []
ssim_values = []
for c in range(C):
    psnr_c = compute_psnr_channel(ground_truth_np[:, c], reconstructed[:, c])
    ssim_c = compute_ssim_channel(gt_sorted[:, c], rec_sorted[:, c])
    psnr_values.append(psnr_c)
    ssim_values.append(ssim_c)

psnr_value = np.mean(psnr_values)
ssim_value = np.mean(ssim_values)

# Coarse-only PSNR + SSIM
coarse_psnr_values = []
coarse_ssim_values = []
for c in range(C):
    psnr_c = compute_psnr_channel(ground_truth_np[:, c], coarse_recon[:, c])
    ssim_c = compute_ssim_channel(gt_sorted[:, c], coarse_sorted[:, c])
    coarse_psnr_values.append(psnr_c)
    coarse_ssim_values.append(ssim_c)
coarse_psnr = np.mean(coarse_psnr_values)
coarse_ssim = np.mean(coarse_ssim_values)

# Overall MSE and relative error
overall_mse = np.mean((ground_truth_np - reconstructed) ** 2)
relative_error = np.sqrt(overall_mse) / (np.std(ground_truth_np) + 1e-8)

print(f"\nReconstruction quality metrics ({N} probes, {C} channels):")
print(f"  Avg PSNR: {psnr_value:.2f} dB")
print(f"  Avg SSIM: {ssim_value:.4f}")
print(f"  Coarse PSNR: {coarse_psnr:.2f} dB")
print(f"  Coarse SSIM: {coarse_ssim:.4f}")
print(f"  Overall MSE: {overall_mse:.8f}")
print(f"  Relative Error: {relative_error:.6f}")
print(f"  Final loss: {losses[-1]['total_loss']:.6f}")

# ===== Per-SH-order PSNR/SSIM breakdown =====
# SH layout: RGB interleaved, 9 bands per color
# Channel order: [SH0_R, SH0_G, SH0_B, SH1_R, SH1_G, SH1_B, ..., SH8_R, SH8_G, SH8_B]
# SH order-0: 1 band (index 0), order-1: 3 bands (1-3), order-2: 5 bands (4-8)
sh_order_ranges = [
    (0, [0]),       # L0: 1 band
    (1, [1, 2, 3]), # L1: 3 bands
    (2, [4, 5, 6, 7, 8])  # L2: 5 bands
]

print(f"\n  Per-SH-Order Breakdown (RGB interleaved, 9 bands × 3 colors):")
print(f"  {'Order':<8} {'Bands':<12} {'Channels':<10} {'Avg PSNR':>10} {'Avg SSIM':>10}")
print(f"  {'-'*54}")
for sh_order, band_indices in sh_order_ranges:
    # RGB interleaved: channel index = band_idx * 3 + color_idx
    order_channels = []
    for band_idx in band_indices:
        for color_idx in range(3):  # R, G, B
            order_channels.append(band_idx * 3 + color_idx)
    order_psnr = np.mean([psnr_values[c] for c in order_channels])
    order_ssim = np.mean([ssim_values[c] for c in order_channels])
    print(f"  L{sh_order:<7} {len(band_indices)} bands     {len(order_channels):>2}ch       {order_psnr:>8.2f} dB  {order_ssim:>8.4f}")

# ==================== Visualization results (SH Compression for ILC Probes) ====================
print("\nGenerating SH Compression Visualization results...")

# Get Gaussian params for visualization
gaussian_params = model.get_gaussian_params()
fine_mu = gaussian_params['fine_mu']
fine_s = gaussian_params['fine_s']
fine_alpha = gaussian_params['fine_alpha']

fig = plt.figure(figsize=(24, 16))

# ===== Row 1: 3D Probe Distribution & Error Visualization =====

# 1. 3D Probe positions colored by reconstruction error
ax1 = fig.add_subplot(3, 4, 1, projection='3d')
per_probe_error = np.mean((ground_truth_np - reconstructed) ** 2, axis=1)  # [N]
# Subsample for visualization (too many points)
vis_indices = np.random.choice(N, min(5000, N), replace=False)
sc1 = ax1.scatter(coords[vis_indices, 0].numpy(), 
                  coords[vis_indices, 1].numpy(),
                  coords[vis_indices, 2].numpy(),
                  c=per_probe_error[vis_indices], cmap='hot', s=1, alpha=0.5)
ax1.set_title(f'Probe Error Distribution\n{N} probes (showing {len(vis_indices)})')
ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
plt.colorbar(sc1, ax=ax1, fraction=0.02, pad=0.1)

# 2. 3D Gaussian Ellipsoids (all types)
ax2 = fig.add_subplot(3, 4, 2, projection='3d')
coeff_mu_viz = gaussian_params['coeff_mu']
coeff_s_viz = gaussian_params['coeff_s']
basis_mu_viz = gaussian_params['basis_mu']
basis_s_viz = gaussian_params['basis_s']

ax2.scatter(coeff_mu_viz[:, 0], coeff_mu_viz[:, 1], coeff_mu_viz[:, 2],
            c='red', s=40, alpha=0.8, label=f'Coeff (M={model.M})')
ax2.scatter(basis_mu_viz[:, 0], basis_mu_viz[:, 1], basis_mu_viz[:, 2],
            c='blue', s=50, marker='s', alpha=0.8, label=f'Basis (N={model.N})')
ax2.scatter(fine_mu[:, 0], fine_mu[:, 1], fine_mu[:, 2],
            c='cyan', s=25, marker='^', alpha=0.8, label=f'Fine (F={model.F})')
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.set_zlim(0, 1)
ax2.set_title(f'3D Gaussian Anchors\nM={model.M}, N={model.N}, F={model.F}')
ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
ax2.legend(fontsize='x-small', loc='upper left')

# 3. Per-channel PSNR bar chart
ax3 = plt.subplot(3, 4, 3)
# Group by SH order and show as grouped bars
colors_order = ['#2ecc71', '#3498db', '#e74c3c']
bar_data = []
bar_labels = []
bar_colors = []
for sh_order, band_indices in sh_order_ranges:
    for band_idx in band_indices:
        for color_idx, color_name in enumerate(['R', 'G', 'B']):
            ch_idx = band_idx * 3 + color_idx
            bar_data.append(psnr_values[ch_idx])
            bar_labels.append(f'SH{band_idx}_{color_name}')
            bar_colors.append(colors_order[sh_order])

x_pos = np.arange(len(bar_data))
ax3.bar(x_pos, bar_data, color=bar_colors, alpha=0.7, width=0.8)
ax3.axhline(y=psnr_value, color='black', linestyle='--', alpha=0.5, label=f'Avg: {psnr_value:.1f}dB')
ax3.set_title(f'Per-Channel PSNR\nAvg: {psnr_value:.1f}dB')
ax3.set_xlabel('Channel')
ax3.set_ylabel('PSNR (dB)')
ax3.set_xticks(x_pos[::3])
ax3.set_xticklabels([f'SH{i}' for i in range(9)], fontsize=7)
ax3.legend(fontsize='small')
ax3.grid(True, alpha=0.3, axis='y')

# 4. SH coefficient distribution (GT vs Reconstructed)
ax4 = plt.subplot(3, 4, 4)
# Show DC (L0) channel distribution
for color_idx, (color, name) in enumerate(zip(['red', 'green', 'blue'], ['R', 'G', 'B'])):
    ch_idx = color_idx  # SH0_R=0, SH0_G=1, SH0_B=2
    ax4.hist(ground_truth_np[:, ch_idx], bins=50, alpha=0.3, color=color, 
             label=f'GT L0_{name}', density=True)
    ax4.hist(reconstructed[:, ch_idx], bins=50, alpha=0.3, color=color,
             linestyle='--', density=True, histtype='step', linewidth=2)
ax4.set_title('SH L0 (DC) Distribution\nSolid=GT, Line=Pred')
ax4.set_xlabel('SH Coefficient Value')
ax4.set_ylabel('Density')
ax4.legend(fontsize='x-small')
ax4.grid(True, alpha=0.3)

# ===== Row 2: Training curves and error analysis =====

# 5. Training loss curves
ax5 = plt.subplot(3, 4, 5)
total_losses = [l['total_loss'] for l in losses]
final_losses = [l['final_loss'] for l in losses]
coarse_losses = [l['coarse_loss'] for l in losses]

ax5.semilogy(total_losses, 'b-', linewidth=2, label='Total Loss')
ax5.semilogy(final_losses, 'g--', linewidth=1.5, alpha=0.7, label='Final Loss')
ax5.semilogy(coarse_losses, 'r:', linewidth=1.5, alpha=0.7, label='Coarse Loss')

# Mark training stages
epochs_coarse_stage, epochs_main_stage, epochs_fine_stage = 500, 2500, 500
ax5.axvline(x=epochs_coarse_stage, color='orange', linestyle=':', alpha=0.7, label='Stage 1\u21922')
ax5.axvline(x=epochs_coarse_stage + epochs_main_stage, color='purple', linestyle=':', alpha=0.7, label='Stage 2\u21923')
ax5.set_title('Training Loss (3 Stages)')
ax5.set_xlabel('Iterations')
ax5.set_ylabel('Loss Value')
ax5.legend(fontsize='x-small', loc='upper right')
ax5.grid(True, alpha=0.3)

# 6. Lambda coarse evolution
ax6 = plt.subplot(3, 4, 6)
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
ax6.plot(lambda_values, 'b-', linewidth=2, label='\u03bb_coarse')
ax6.axvline(x=epochs_coarse_stage, color='orange', linestyle=':', alpha=0.7)
ax6.axvline(x=epochs_coarse_stage + epochs_main_stage, color='purple', linestyle=':', alpha=0.7)
ax6.set_title('\u03bb_coarse Evolution (Curriculum)')
ax6.set_xlabel('Iterations')
ax6.set_ylabel('\u03bb_coarse')
ax6.legend(fontsize='small')
ax6.grid(True, alpha=0.3)

# 7. Error histogram
ax7 = plt.subplot(3, 4, 7)
ax7.hist(per_probe_error, bins=100, alpha=0.7, color='steelblue', edgecolor='none')
ax7.axvline(x=np.mean(per_probe_error), color='red', linestyle='--', 
            label=f'Mean: {np.mean(per_probe_error):.6f}')
ax7.axvline(x=np.median(per_probe_error), color='orange', linestyle='--',
            label=f'Median: {np.median(per_probe_error):.6f}')
ax7.set_title('Per-Probe MSE Distribution')
ax7.set_xlabel('MSE')
ax7.set_ylabel('Count')
ax7.legend(fontsize='small')
ax7.grid(True, alpha=0.3)
ax7.set_yscale('log')

# 8. Gaussian scale distribution
ax8 = plt.subplot(3, 4, 8)
fine_s_mean = fine_s.mean(axis=1)
coeff_s_viz_mean = coeff_s_viz.mean(axis=1)
basis_s_viz_mean = basis_s_viz.mean(axis=1)
ax8.hist(coeff_s_viz_mean, bins=15, alpha=0.6, color='red', label=f'Coeff (M={model.M})', edgecolor='black')
ax8.hist(basis_s_viz_mean, bins=15, alpha=0.6, color='blue', label=f'Basis (N={model.N})', edgecolor='black')
ax8.hist(fine_s_mean, bins=15, alpha=0.6, color='cyan', label=f'Fine (F={model.F})', edgecolor='black')
ax8.set_title('Gaussian Scale Distribution')
ax8.set_xlabel('Mean Scale')
ax8.set_ylabel('Count')
ax8.legend(fontsize='small')
ax8.grid(True, alpha=0.3)

# ===== Row 3: Detailed comparison and summary =====

# 9. Sample probes: GT vs Reconstructed (line plot for first few probes)
ax9 = plt.subplot(3, 4, 9)
sample_indices = np.random.choice(N, 5, replace=False)
for idx in sample_indices[:3]:
    ax9.plot(ground_truth_np[idx], 'o-', markersize=2, alpha=0.5, linewidth=1)
    ax9.plot(reconstructed[idx], 'x--', markersize=2, alpha=0.8, linewidth=1)
ax9.set_title('Sample Probes: GT(o) vs Pred(x)\n27 SH channels')
ax9.set_xlabel('Channel Index')
ax9.set_ylabel('SH Coefficient')
ax9.grid(True, alpha=0.3)

# 10. Coarse vs Fine contribution analysis
ax10 = plt.subplot(3, 4, 10)
coarse_magnitude = np.abs(coarse_recon).mean(axis=1)  # [N]
fine_magnitude = np.abs(fine_recon).mean(axis=1)      # [N]
ratio_fine_coarse = fine_magnitude / (coarse_magnitude + 1e-8)
ax10.hist(ratio_fine_coarse, bins=50, alpha=0.7, color='purple', edgecolor='none')
ax10.axvline(x=np.median(ratio_fine_coarse), color='red', linestyle='--',
             label=f'Median: {np.median(ratio_fine_coarse):.3f}')
ax10.set_title('Fine/Coarse Magnitude Ratio')
ax10.set_xlabel('|Fine| / |Coarse|')
ax10.set_ylabel('Count')
ax10.legend()
ax10.grid(True, alpha=0.3)

# 11. Per-SH-order PSNR/SSIM comparison
ax11 = plt.subplot(3, 4, 11)
order_names = []
order_psnrs_final = []
order_psnrs_coarse = []
order_ssims_final = []
order_ssims_coarse = []
for sh_order, band_indices in sh_order_ranges:
    order_channels = []
    for band_idx in band_indices:
        for color_idx in range(3):
            order_channels.append(band_idx * 3 + color_idx)
    order_psnrs_final.append(np.mean([psnr_values[c] for c in order_channels]))
    order_psnrs_coarse.append(np.mean([coarse_psnr_values[c] for c in order_channels]))
    order_ssims_final.append(np.mean([ssim_values[c] for c in order_channels]))
    order_ssims_coarse.append(np.mean([coarse_ssim_values[c] for c in order_channels]))
    order_names.append(f'L{sh_order}')

x_pos_order = np.arange(len(order_names))
width = 0.35
# Main axis: PSNR bars
ax11.bar(x_pos_order - width/2, order_psnrs_coarse, width, label='Coarse PSNR', color='orange', alpha=0.7)
ax11.bar(x_pos_order + width/2, order_psnrs_final, width, label='Final PSNR', color='green', alpha=0.7)
ax11.set_title('Per-SH-Order: PSNR & SSIM\nCoarse vs Final')
ax11.set_xticks(x_pos_order)
ax11.set_xticklabels(order_names)
ax11.set_ylabel('PSNR (dB)')
ax11.legend(loc='upper left', fontsize='x-small')
ax11.grid(True, alpha=0.3, axis='y')
# Secondary axis: SSIM line
ax11b = ax11.twinx()
ax11b.plot(x_pos_order, order_ssims_coarse, 'o--', color='darkorange', label='Coarse SSIM', markersize=6)
ax11b.plot(x_pos_order, order_ssims_final, 's-', color='darkgreen', label='Final SSIM', markersize=6)
ax11b.set_ylabel('SSIM')
ax11b.set_ylim(0, 1.05)
ax11b.legend(loc='upper right', fontsize='x-small')

# 12. Summary information
ax12 = plt.subplot(3, 4, 12)
ax12.axis('off')

coeff_alpha_viz = gaussian_params['coeff_alpha']
basis_alpha_viz = gaussian_params['basis_alpha']
mbd_scale = gaussian_params['mbd_scale']

info_text = f"""
HI-NE-GBD SH Compression (ILCSampleData)
===============================================
Input Data:
  Probes: {N} (scattered 3D positions)
  SH Channels: {C} (9 bands \u00d7 3 RGB, interleaved)
  Original Size: {original_size/1024:.1f} KB

Model Architecture:
  Coarse (MBD): M={model.M}, N={model.N}, L={model.L}
  Fine (Gaussian): F={model.F} anchors
  Fine (MLP): PE({model.pe_num_freqs}) + {model.mlp_hidden}h
  Blending: Additive Residual
  Total params: {branch_info['total']}

Compression (float16):
  FP32: {comp_size_fp32/1024:.1f} KB ({comp_ratio_fp32:.1f}:1)
  FP16: {comp_size_fp16/1024:.1f} KB ({comp_ratio_fp16:.1f}:1)

Reconstruction Quality:
  Final PSNR: {psnr_value:.1f} dB
  Final SSIM: {ssim_value:.4f}
  Coarse PSNR: {coarse_psnr:.1f} dB
  Coarse SSIM: {coarse_ssim:.4f}
  Overall MSE: {overall_mse:.8f}
  Relative Error: {relative_error:.6f}
"""
ax12.text(0.02, 0.5, info_text, fontsize=8,
          family='monospace', verticalalignment='center')

plt.suptitle('HI-NE-GBD: SH Compression for ILC Light Probes', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'HI-NE-GBD.png'), 
            dpi=150, bbox_inches='tight')
plt.show()

# ==================== Final Summary ====================
print("\nSH Compression completed!")
print("="*70)
print("HI-NE-GBD: SH Compression for ILC Light Probes")
print("="*70)
print(f"Input: {N} probes \u00d7 {C} SH channels (ILCSampleData_0.bin)")
print(f"  Position range: X[{pos_min[0]:.0f}, {pos_max[0]:.0f}], "
      f"Y[{pos_min[1]:.0f}, {pos_max[1]:.0f}], Z[{pos_min[2]:.0f}, {pos_max[2]:.0f}]")
print(f"\nArchitecture:")
print(f"  [Coarse Branch] MBD with 3D Gaussians")
print(f"    - Coefficient Gaussians: M={model.M}")
print(f"    - Basis Gaussians: N={model.N}")
print(f"    - Number of Bases: L={model.L}")
print(f"  [Fine Branch] Gaussian-Guided MLP")
print(f"    - Fine Gaussians: F={model.F}")
print(f"    - PE frequencies: {model.pe_num_freqs}")
print(f"    - Hidden size: {model.mlp_hidden}")
print(f"  [Blending] Additive Residual (coarse + fine)")
print(f"\nCompression Results:")
print(f"  Original:  {original_size/1024:.1f} KB")
print(f"  FP32:      {comp_size_fp32/1024:.1f} KB (ratio {comp_ratio_fp32:.1f}:1)")
print(f"  FP16:      {comp_size_fp16/1024:.1f} KB (ratio {comp_ratio_fp16:.1f}:1)")
print(f"\nReconstruction Quality:")
print(f"  Final PSNR: {psnr_value:.2f} dB")
print(f"  Final SSIM: {ssim_value:.4f}")
print(f"  Coarse PSNR: {coarse_psnr:.2f} dB")
print(f"  Coarse SSIM: {coarse_ssim:.4f}")
print(f"  Overall MSE: {overall_mse:.8f}")
print(f"  Relative Error: {relative_error:.6f}")
print(f"\nPer-SH-Order PSNR/SSIM:")
for i, (sh_order, _) in enumerate(sh_order_ranges):
    print(f"  L{sh_order}: PSNR={order_psnrs_final[i]:.1f}dB, SSIM={order_ssims_final[i]:.4f} | "
          f"Coarse: PSNR={order_psnrs_coarse[i]:.1f}dB, SSIM={order_ssims_coarse[i]:.4f}")
print("="*70)
