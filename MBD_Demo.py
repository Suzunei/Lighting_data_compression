import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from test_signal_3d import get_test_signal_by_name, get_all_test_signals

#运行指令：$env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + [Environment]::GetEnvironmentVariable("Path", "Machine")
#python MBD_Demo.py 2>&1

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
grid_size = 32  # 3D grid (32^3 = 32768 points)
ground_truth = create_test_signal_3d(grid_size, num_channels=3)
D, H, W, C = ground_truth.shape
print(f"Generated 3D test signal size: {D}x{H}x{W}x{C}")

# Prepare training data: flatten 3D grid to [N, 3] coordinates
x_coords = torch.linspace(0, 1, D)
y_coords = torch.linspace(0, 1, H)
z_coords = torch.linspace(0, 1, W)
X_grid, Y_grid, Z_grid = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
coords = torch.stack([X_grid.flatten(), Y_grid.flatten(), Z_grid.flatten()], dim=-1)  # [N, 3]
target_data = ground_truth.view(-1, C)  # [N, 3]

# ==================== Step 2: Implement MBD model and solver ====================
print("\nStep 2: Building MBD model and solver...")

# ==================== Helper: Quaternion to Rotation Matrix ====================
def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion to rotation matrix (3DGS standard implementation).
    q: [..., 4] quaternion (w, x, y, z)
    Returns: [..., 3, 3] rotation matrix
    """
    # Normalize quaternion
    q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Build rotation matrix
    R = torch.stack([
        torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y], dim=-1),
        torch.stack([2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x], dim=-1),
        torch.stack([2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y], dim=-1)
    ], dim=-2)
    return R

class MBDCompressor(nn.Module):
    """
    Moving Basis Decomposition (MBD) Compressor with Trainable 3D Gaussians
    Full 3DGS representation: Position + Scale + Rotation (Quaternion)
    Implements core formulas from the paper:
        c_l(x) = Σ_m φ_m(x) * c_{m,l}
        b_l(x) = Σ_n ψ_n(x) * B_{n,l}
        f̂(x) = Σ_l c_l(x) * b_l(x)

    Gaussian covariance: Σ = R @ S @ S^T @ R^T (full rotation support)
    """
    def __init__(self, num_bases=6, coeff_res=12, basis_res=8, data_dim=3,
                 coeff_kernel_scale_init=0.09, basis_kernel_scale_init=0.18):
        super().__init__()
        self.L = num_bases
        self.D = data_dim

        # ========== Improved initialization with structured grid + jitter ==========
        # Coefficient control points: use stratified sampling for better coverage
        coeff_grid_size = int(np.ceil(coeff_res ** (1/3)))
        coeff_grid = self._create_jittered_grid(coeff_res, coeff_grid_size, jitter=0.15)
        self.coeff_points = nn.Parameter(coeff_grid)

        # Basis control points: use stratified sampling
        basis_grid_size = int(np.ceil(basis_res ** (1/3)))
        basis_grid = self._create_jittered_grid(basis_res, basis_grid_size, jitter=0.12)
        self.basis_points = nn.Parameter(basis_grid)

        # Trainable log-scale parameters with adaptive initialization
        # Smaller initial scales for coeff (more local), larger for basis (smoother)
        self.coeff_log_scales = nn.Parameter(
            torch.full((coeff_res, 3), np.log(coeff_kernel_scale_init)) +
            torch.randn(coeff_res, 3) * 0.1  # Add small variation
        )
        self.basis_log_scales = nn.Parameter(
            torch.full((basis_res, 3), np.log(basis_kernel_scale_init)) +
            torch.randn(basis_res, 3) * 0.1
        )

        # Trainable quaternion rotation with small random perturbation
        self.coeff_q = nn.Parameter(torch.zeros(coeff_res, 4))
        self.basis_q = nn.Parameter(torch.zeros(basis_res, 4))
        with torch.no_grad():
            # Initialize close to identity with small noise for diversity
            self.coeff_q[:, 0] = 1.0 + torch.randn(coeff_res) * 0.02
            self.coeff_q[:, 1:] = torch.randn(coeff_res, 3) * 0.05
            self.basis_q[:, 0] = 1.0 + torch.randn(basis_res) * 0.02
            self.basis_q[:, 1:] = torch.randn(basis_res, 3) * 0.05
            # Normalize
            self.coeff_q.data = self.coeff_q.data / (torch.norm(self.coeff_q.data, dim=-1, keepdim=True) + 1e-8)
            self.basis_q.data = self.basis_q.data / (torch.norm(self.basis_q.data, dim=-1, keepdim=True) + 1e-8)

        # Learnable parameters with improved initialization
        # C: [M, L] - use Xavier initialization scaled by bases
        self.C = nn.Parameter(torch.randn(coeff_res, self.L) * np.sqrt(2.0 / (coeff_res + self.L)))
        # B: [N, L, D] - use Xavier initialization
        self.B = nn.Parameter(torch.randn(basis_res, self.L, self.D) * np.sqrt(2.0 / (basis_res + self.L * self.D)))

        # Store sizes
        self.M = coeff_res
        self.N = basis_res
        print(f"MBD 3D model initialized (Full 3DGS Transform):")
        print(f"  {self.M} coeff points, {self.N} basis points, {self.L} bases")
        print(f"  Trainable: Position(3D) + Scale(3D) + Rotation(Quaternion) + C + B")
        print(f"  Init scales: coeff={coeff_kernel_scale_init:.3f}, basis={basis_kernel_scale_init:.3f}")

    def _create_jittered_grid(self, num_points, grid_size, jitter=0.1):
        """Create stratified jittered grid points for better spatial coverage"""
        # Create base grid
        coords_1d = torch.linspace(0.05, 0.95, grid_size)
        xx, yy, zz = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')
        base_grid = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=-1)

        # Sample required number of points
        if len(base_grid) >= num_points:
            indices = torch.randperm(len(base_grid))[:num_points]
            grid = base_grid[indices]
        else:
            # If need more points, use random sampling
            grid = torch.rand(num_points, 3) * 0.9 + 0.05

        # Add jitter
        grid = grid + torch.randn_like(grid) * jitter / grid_size
        grid = torch.clamp(grid, 0.02, 0.98)  # Keep within bounds
        return grid

    def compute_kernel_weights(self, query_pts, control_pts, log_scales, quaternions):
        """
        Compute 3D Gaussian kernel weights with full covariance matrix.
        Uses Mahalanobis distance with Σ = R @ S @ S^T @ R^T

        query_pts: [Q, 3] - query 3D positions
        control_pts: [K, 3] - control point 3D positions
        log_scales: [K, 3] - log of anisotropic scales (s_x, s_y, s_z)
        quaternions: [K, 4] - rotation quaternions (w, x, y, z)

        Returns: [Q, K] normalized Gaussian weights
        """
        Q = query_pts.shape[0]
        K = control_pts.shape[0]

        # Get actual scales from log-scales
        scales = torch.exp(log_scales)  # [K, 3]

        # Build rotation matrices from quaternions
        R = quaternion_to_rotation_matrix(quaternions)  # [K, 3, 3]

        # Build precision matrix (inverse covariance): Σ^{-1} = R @ S^{-2} @ R^T
        s_inv_sq = 1.0 / (scales ** 2 + 1e-8)  # [K, 3]
        S_inv_sq = torch.diag_embed(s_inv_sq)  # [K, 3, 3]
        precision = R @ S_inv_sq @ R.transpose(-1, -2)  # [K, 3, 3]

        # Compute Mahalanobis distance: (p - μ)^T @ Σ^{-1} @ (p - μ)
        diff = query_pts.unsqueeze(1) - control_pts.unsqueeze(0)  # [Q, K, 3]
        diff_expanded = diff.unsqueeze(-1)  # [Q, K, 3, 1]
        precision_expanded = precision.unsqueeze(0)  # [1, K, 3, 3]

        mahalanobis_sq = (diff_expanded.transpose(-1, -2) @ precision_expanded @ diff_expanded)
        mahalanobis_sq = mahalanobis_sq.squeeze(-1).squeeze(-1)  # [Q, K]

        # Gaussian kernel: exp(-0.5 * d_mahalanobis²)
        weights = torch.exp(-0.5 * mahalanobis_sq)

        # Normalize weights
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        return weights

    def forward(self, coords):
        """Forward pass: reconstruct signal from coordinates"""
        # 1. Compute 3D Gaussian kernel weights with full covariance (Position + Scale + Rotation)
        phi_weights = self.compute_kernel_weights(coords, self.coeff_points,
                                                   self.coeff_log_scales,
                                                   self.coeff_q)  # [Q, M]
        psi_weights = self.compute_kernel_weights(coords, self.basis_points,
                                                   self.basis_log_scales,
                                                   self.basis_q)  # [Q, N]

        #B
        # 2. Compute moving coefficients c_l(x) = Σ_m φ_m(x) * C_{m,l}
        moving_coeff = torch.matmul(phi_weights, self.C)  # [Q, L] matmul will dot the vector and then sum the result

        #C interpolate the basis to reconstruct the sparse control points
        # 3. Compute moving bases b_l(x) = Σ_n ψ_n(x) * B_{n,l}
        B_flat = self.B.view(-1, self.L * self.D)  # [N, L*D]
        basis_interp_flat = torch.matmul(psi_weights, B_flat)  # [Q, L*D]
        #interpolate the basis vectors to get the moving basis
        moving_basis = basis_interp_flat.view(-1, self.L, self.D)  # [Q, L, D]

        # 4. Reconstruct signal f̂(x) = Σ_l c_l(x) * b_l(x)
        reconstruction = torch.sum(moving_coeff.unsqueeze(-1) * moving_basis, dim=1)  # [Q, D]

        return reconstruction, moving_coeff, moving_basis

    def get_compression_ratio(self, original_size):
        """Compute compression ratio for 3D data with full transform"""
        # Original data size: D*H*W*C*4 bytes (float32)
        # Compressed: M*L*4 + N*L*D*4 + Gaussian params
        compressed_size = (self.M * self.L + self.N * self.L * self.D) * 4
        # Add 3D Gaussian params: position(3) + scale(3) + quaternion(4) = 10 floats per point
        compressed_size += (self.M * 10 + self.N * 10) * 4
        ratio = original_size / compressed_size
        return ratio, compressed_size

    def get_gaussian_params(self):
        """Get Gaussian parameters for visualization"""
        with torch.no_grad():
            coeff_mu = self.coeff_points.cpu().numpy()
            coeff_s = torch.exp(self.coeff_log_scales).cpu().numpy()
            coeff_q = self.coeff_q.cpu().numpy()
            coeff_q = coeff_q / (np.linalg.norm(coeff_q, axis=1, keepdims=True) + 1e-8)

            basis_mu = self.basis_points.cpu().numpy()
            basis_s = torch.exp(self.basis_log_scales).cpu().numpy()
            basis_q = self.basis_q.cpu().numpy()
            basis_q = basis_q / (np.linalg.norm(basis_q, axis=1, keepdims=True) + 1e-8)
        return {
            'coeff': (coeff_mu, coeff_s, coeff_q),
            'basis': (basis_mu, basis_s, basis_q)
        }

class MBDSolver:
    """MBD Solver with enhanced optimization strategies"""
    def __init__(self, model, lambda_reg=0.01, lambda_tv=0.001, lambda_smooth=0.0005):
        self.model = model
        self.lambda_reg = lambda_reg
        self.lambda_tv = lambda_tv  # Total Variation regularization
        self.lambda_smooth = lambda_smooth  # Smoothness regularization

        # Separate parameter groups with finer control
        position_params = [model.coeff_points, model.basis_points]
        scale_params = [model.coeff_log_scales, model.basis_log_scales]
        rotation_params = [model.coeff_q, model.basis_q]
        feature_params = [model.C, model.B]

        self.optimizer = optim.AdamW([
            {'params': position_params, 'lr': 0.003, 'weight_decay': 1e-5},   # position
            {'params': scale_params, 'lr': 0.004, 'weight_decay': 1e-5},      # scale
            {'params': rotation_params, 'lr': 0.002, 'weight_decay': 0},      # rotation (no decay)
            {'params': feature_params, 'lr': 0.008, 'weight_decay': 1e-4},    # features
        ])
        self.scheduler = None
        self.warmup_epochs = 200  # Learning rate warmup
        self.base_lrs = [0.003, 0.004, 0.002, 0.008]

    def compute_tv_loss(self, pred, coords, grid_size=32):
        """Compute Total Variation loss for smoothness"""
        # Reshape prediction to 3D grid for TV computation
        pred_vol = pred.view(grid_size, grid_size, grid_size, -1)
        # TV in each dimension
        tv_d = torch.mean(torch.abs(pred_vol[1:, :, :, :] - pred_vol[:-1, :, :, :]))
        tv_h = torch.mean(torch.abs(pred_vol[:, 1:, :, :] - pred_vol[:, :-1, :, :]))
        tv_w = torch.mean(torch.abs(pred_vol[:, :, 1:, :] - pred_vol[:, :, :-1, :]))
        return (tv_d + tv_h + tv_w) / 3.0

    def compute_smoothness_loss(self):
        """Regularize Gaussian parameters for smoothness"""
        # Penalize extreme scales
        coeff_scales = torch.exp(self.model.coeff_log_scales)
        basis_scales = torch.exp(self.model.basis_log_scales)
        scale_var = torch.var(coeff_scales) + torch.var(basis_scales)
        return scale_var

    def compute_loss(self, pred, target, coeff_params, coords=None, use_tv=True):
        """Compute loss function with multiple regularizations"""
        # L1 + L2 hybrid reconstruction loss (better for sharp features)
        l2_loss = torch.mean((pred - target) ** 2)
        l1_loss = torch.mean(torch.abs(pred - target))
        recon_loss = 0.8 * l2_loss + 0.2 * l1_loss  # Hybrid

        # Frobenius norm regularization (prevents scale ambiguity)
        reg_loss = self.lambda_reg * (torch.sum(coeff_params ** 2) +
                                       0.1 * torch.sum(self.model.B ** 2))

        # Total variation regularization
        tv_loss = torch.tensor(0.0, device=pred.device)
        if use_tv and coords is not None and self.lambda_tv > 0:
            try:
                tv_loss = self.lambda_tv * self.compute_tv_loss(pred, coords)
            except:
                pass  # Skip if batch size doesn't match grid

        # Smoothness regularization
        smooth_loss = self.lambda_smooth * self.compute_smoothness_loss()

        total_loss = recon_loss + reg_loss + tv_loss + smooth_loss

        return total_loss, recon_loss, reg_loss, tv_loss

    def warmup_lr(self, epoch):
        """Linear learning rate warmup"""
        if epoch < self.warmup_epochs:
            warmup_factor = (epoch + 1) / self.warmup_epochs
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.base_lrs[i] * warmup_factor

    def train_step(self, coords_batch, target_batch, use_tv=False):
        """Single training step"""
        self.optimizer.zero_grad()

        # Forward pass
        pred, moving_coeff, _ = self.model(coords_batch)

        # Compute loss
        total_loss, recon_loss, reg_loss, tv_loss = self.compute_loss(
            pred, target_batch, self.model.C, coords_batch, use_tv=use_tv
        )

        # Backward pass
        total_loss.backward()

        # Gradient clipping (adaptive based on loss)
        max_norm = 1.0 if total_loss.item() < 0.1 else 2.0
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)

        # Optimization
        self.optimizer.step()

        # Normalize quaternions after update
        with torch.no_grad():
            self.model.coeff_q.data = self.model.coeff_q.data / (
                torch.norm(self.model.coeff_q.data, dim=-1, keepdim=True) + 1e-8)
            self.model.basis_q.data = self.model.basis_q.data / (
                torch.norm(self.model.basis_q.data, dim=-1, keepdim=True) + 1e-8)

        return {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'reg_loss': reg_loss.item(),
            'tv_loss': tv_loss.item() if isinstance(tv_loss, torch.Tensor) else tv_loss
        }

    def train(self, coords, target, epochs=2000, batch_size=2048):
        """Training loop with enhanced strategies"""
        print(f"Starting training, {epochs} epochs, batch_size={batch_size}")
        print(f"  Warmup: {self.warmup_epochs} epochs")
        print(f"  Regularization: lambda_reg={self.lambda_reg}, lambda_tv={self.lambda_tv}")

        # Create scheduler with warm restarts for better convergence
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=epochs//4, T_mult=2, eta_min=1e-6
        )

        losses = []
        num_samples = coords.shape[0]
        best_loss = float('inf')
        best_state = None

        # Exponential moving average for smoother loss monitoring
        ema_loss = None
        ema_alpha = 0.98

        for epoch in range(epochs):
            # Learning rate warmup
            self.warmup_lr(epoch)

            # Use larger batch in later stages for stability
            current_batch = min(batch_size, batch_size // 2 + (batch_size // 2) * epoch // (epochs // 2))
            current_batch = max(2048, current_batch)

            # Random batch sampling
            indices = torch.randperm(num_samples)[:current_batch]
            coords_batch = coords[indices]
            target_batch = target[indices]

            # Use TV loss after warmup
            use_tv = epoch > self.warmup_epochs

            # Training step
            loss_dict = self.train_step(coords_batch, target_batch, use_tv=use_tv)
            losses.append(loss_dict)

            # Update EMA loss
            if ema_loss is None:
                ema_loss = loss_dict['total_loss']
            else:
                ema_loss = ema_alpha * ema_loss + (1 - ema_alpha) * loss_dict['total_loss']
            loss_dict['ema_loss'] = ema_loss

            # Save best model
            if ema_loss < best_loss and epoch > self.warmup_epochs:
                best_loss = ema_loss
                best_state = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }

            # Scheduler step (after warmup)
            if epoch >= self.warmup_epochs:
                self.scheduler.step()

            # Print progress
            if epoch % 300 == 0 or epoch == epochs - 1:
                lrs = [pg['lr'] for pg in self.optimizer.param_groups]
                print(f"Epoch {epoch:5d}/{epochs} | "
                      f"Loss: {loss_dict['total_loss']:.6f} | "
                      f"EMA: {ema_loss:.6f} | "
                      f"Recon: {loss_dict['recon_loss']:.6f} | "
                      f"TV: {loss_dict['tv_loss']:.6f} | "
                      f"lr: {lrs[0]:.1e}/{lrs[3]:.1e}")

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"Restored best model with EMA loss: {best_loss:.6f}")

        return losses

# Create model and solver with optimized configuration
model = MBDCompressor(
    num_bases=24,                    # Increased bases for better representation
    coeff_res=192,                   # More coefficient control points
    basis_res=96,                    # Moderate basis points (asymmetric for efficiency)
    data_dim=3,                      # Data dimension D (RGB)
    coeff_kernel_scale_init=0.12,    # Slightly smaller for finer control
    basis_kernel_scale_init=0.18     # Larger for smoother basis interpolation
)

solver = MBDSolver(
    model,
    lambda_reg=0.003,      # Reduced regularization for better fitting
    lambda_tv=0.0008,      # TV regularization for smoothness
    lambda_smooth=0.0003   # Gaussian parameter smoothness
)

# 计算压缩比 (3D体积数据)
original_size = D * H * W * C * 4  # float32
comp_ratio, comp_size = model.get_compression_ratio(original_size)
print(f"Original size: {original_size/1024:.1f} KB")
print(f"Compressed: {comp_size/1024:.1f} KB")
print(f"Compression ratio: {comp_ratio:.1f}:1")

# Training with optimized configuration
print("\nStarting compression (training)...")
losses = solver.train(coords, target_data, epochs=6000, batch_size=8192)

# ==================== Step 3: Evaluation and visualization ====================
print("\nStep 3: Evaluating 3D compression and reconstruction quality...")

# Reconstruct entire 3D volume using trained model
model.eval()
with torch.no_grad():
    reconstructed, _, _ = model(coords)
    reconstructed_vol = reconstructed.view(D, H, W, C).cpu().numpy()
    # Clip to valid range [0, 1]
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

# 计算指标 (3D体积平均)
psnr_values = []
ssim_values = []
for c in range(C):
    psnr_c = compute_psnr(ground_truth[..., c].numpy(), reconstructed_vol[..., c])
    psnr_values.append(psnr_c)
for c in range(C):
    # 对每个通道取中间切片计算SSIM
    z_mid = D // 2
    ssim_c = compute_ssim(ground_truth[z_mid, :, :, c].numpy(), reconstructed_vol[z_mid, :, :, c])
    ssim_values.append(ssim_c)

psnr_value = np.mean(psnr_values)
ssim_value = np.mean(ssim_values)

print(f"Reconstruction quality metrics (3D volume):")
print(f"  Avg PSNR: {psnr_value:.2f} dB")
print(f"  Avg SSIM: {ssim_value:.4f}")
print(f"  Final loss: {losses[-1]['total_loss']:.6f}")

# ==================== Visualization results (3D slices) ====================
print("\nGenerating 3D visualization results...")

# Take middle slice for visualization
z_slice = D // 2
gt_slice = ground_truth[z_slice, :, :, :].numpy()  # [H, W, C]
rec_slice = reconstructed_vol[z_slice, :, :, :]    # [H, W, C]

fig = plt.figure(figsize=(20, 12))

# 1. Original 3D slice
ax1 = plt.subplot(2, 4, 1)
ax1.imshow(gt_slice, vmin=0, vmax=1)
ax1.set_title(f'Original (Z={z_slice} slice)\n3D Volume: {D}x{H}x{W}x{C}')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.grid(False)

# 2. MBD reconstructed slice
ax2 = plt.subplot(2, 4, 2)
ax2.imshow(rec_slice, vmin=0, vmax=1)
ax2.set_title(f'MBD 3D Reconstruction\nPSNR: {psnr_value:.1f}dB, SSIM: {ssim_value:.4f}')
ax2.set_xlabel('X')
ax2.grid(False)

# 3. Error map
ax3 = plt.subplot(2, 4, 3)
error = np.abs(gt_slice - rec_slice)
error_img = ax3.imshow(error.mean(axis=-1), cmap='hot', vmin=0, vmax=0.15)
ax3.set_title('Reconstruction Error (Z slice)')
ax3.set_xlabel('X')
plt.colorbar(error_img, ax=ax3, fraction=0.046, pad=0.04)
ax3.grid(False)

# 4. 3D Control point distribution with rotation info
ax4 = fig.add_subplot(2, 4, 4, projection='3d')
gauss_params = model.get_gaussian_params()
coeff_mu, coeff_s, coeff_q = gauss_params['coeff']
basis_mu, basis_s, basis_q = gauss_params['basis']

ax4.scatter(coeff_mu[:, 0], coeff_mu[:, 1], coeff_mu[:, 2],
           c='red', s=30, alpha=0.6, label=f'Coeff (M={model.M})')
ax4.scatter(basis_mu[:, 0], basis_mu[:, 1], basis_mu[:, 2],
           c='blue', s=50, marker='s', alpha=0.6, label=f'Basis (N={model.N})')
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.set_zlim(0, 1)
ax4.set_title('3D Gaussian Positions')
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_zlabel('Z')
ax4.legend(fontsize='small')

# 5. Loss curves (updated for new loss components)
ax5 = plt.subplot(2, 4, 5)
total_losses = [l['total_loss'] for l in losses]
recon_losses = [l['recon_loss'] for l in losses]
reg_losses = [l['reg_loss'] for l in losses]
tv_losses = [l.get('tv_loss', 0) for l in losses]
ema_losses = [l.get('ema_loss', l['total_loss']) for l in losses]

ax5.semilogy(total_losses, 'b-', linewidth=0.5, alpha=0.3, label='Raw Loss')
ax5.semilogy(ema_losses, 'orange', linewidth=2, label='EMA Loss')
ax5.semilogy(recon_losses, 'g--', linewidth=1.2, alpha=0.7, label='Reconstruction')
ax5.semilogy([r + t for r, t in zip(reg_losses, tv_losses)], 'r:', linewidth=1, alpha=0.5, label='Reg+TV')
ax5.axvline(x=200, color='gray', linestyle='--', alpha=0.5, label='Warmup End')
ax5.set_title('Training Loss Curves')
ax5.set_xlabel('Iterations')
ax5.set_ylabel('Loss Value')
ax5.legend(fontsize='x-small')
ax5.grid(True, alpha=0.3)

# 6. Channel comparison (along X axis at Z middle slice)
ax6 = plt.subplot(2, 4, 6)
y_line = H // 2
for c in range(3):
    ax6.plot(gt_slice[y_line, :, c],
            color=['r', 'g', 'b'][c],
            linestyle='-', alpha=0.7, label=f'Original C{c}')
    ax6.plot(rec_slice[y_line, :, c],
            color=['r', 'g', 'b'][c],
            linestyle='--', alpha=0.9, label=f'Recon C{c}')
ax6.set_title(f'Channel Comparison (Z={z_slice}, Y={y_line})')
ax6.set_xlabel('X Coordinate')
ax6.set_ylabel('Intensity')
ax6.legend(loc='upper right', fontsize='small')
ax6.grid(True, alpha=0.3)

# 7. Rotation angle distribution
ax7 = plt.subplot(2, 4, 7)
# Compute rotation angles from quaternions
coeff_angles = 2 * np.arccos(np.clip(np.abs(coeff_q[:, 0]), 0, 1)) * 180 / np.pi
basis_angles = 2 * np.arccos(np.clip(np.abs(basis_q[:, 0]), 0, 1)) * 180 / np.pi
all_angles = np.concatenate([coeff_angles, basis_angles])

ax7.hist(coeff_angles, bins=20, alpha=0.6, color='red', edgecolor='darkred', label='Coeff')
ax7.hist(basis_angles, bins=20, alpha=0.6, color='blue', edgecolor='darkblue', label='Basis')
ax7.axvline(x=all_angles.mean(), color='green', linestyle='--',
            label=f'Mean: {all_angles.mean():.1f}\u00b0')
ax7.set_title('Rotation Angle Distribution')
ax7.set_xlabel('Rotation Angle (degrees)')
ax7.set_ylabel('Count')
ax7.legend(fontsize='small')
ax7.grid(True, alpha=0.3)

# Scale anisotropy analysis
coeff_scale_ratios = coeff_s.max(axis=1) / (coeff_s.min(axis=1) + 1e-8)
basis_scale_ratios = basis_s.max(axis=1) / (basis_s.min(axis=1) + 1e-8)

# 8. Compression info with transform stats
ax8 = plt.subplot(2, 4, 8)
ax8.axis('off')
num_probes = D * H * W
final_recon_loss = losses[-1]['recon_loss']
info_text = f"""
3D MBD with Full Transform
===========================
Original Volume: {D}×{H}×{W}×{C}
Total Probes: {num_probes}
Original: {original_size/1024:.1f} KB
Compressed: {comp_size/1024:.1f} KB
Ratio: {comp_ratio:.1f}:1

Optimization:
  Epochs: 6000 (warmup: 200)
  Batch: 8192 (adaptive)
  Loss: L1+L2 + Reg + TV

Gaussian Transform:
  Position: Jittered grid init
  Scale: 3D anisotropic
  Rotation: Quaternion

Transform Stats:
  Coeff Anisotropy: {coeff_scale_ratios.mean():.2f}±{coeff_scale_ratios.std():.2f}
  Basis Anisotropy: {basis_scale_ratios.mean():.2f}±{basis_scale_ratios.std():.2f}
  Avg Rotation: {all_angles.mean():.1f}°±{all_angles.std():.1f}°

Reconstruction:
  PSNR: {psnr_value:.1f} dB
  SSIM: {ssim_value:.4f}
  Final Recon Loss: {final_recon_loss:.6f}
"""
ax8.text(0.02, 0.5, info_text, fontsize=7,
        family='monospace', verticalalignment='center')

plt.suptitle('3D MBD with Trainable Gaussians (Position + Scale + Rotation)', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

print("\nDemo completed!")
print("="*70)
print("Optimization Summary:")
print("1. Enhanced training: 6000 epochs with 200-epoch warmup")
print("2. Improved loss: L1+L2 hybrid + Frobenius reg + TV smoothness")
print("3. Better init: Jittered grid positions + Xavier weights")
print("4. AdamW optimizer with separate lr for position/scale/rotation/features")
print("5. Cosine annealing with warm restarts + best model checkpoint")
print("")
print("Model Configuration:")
print(f"  Bases: {model.L} | Coeff points: {model.M} | Basis points: {model.N}")
print(f"  Compression ratio: {comp_ratio:.1f}:1")
print("")
print("Results:")
print(f"  PSNR: {psnr_value:.2f} dB | SSIM: {ssim_value:.4f}")
print(f"  Avg rotation angle: {all_angles.mean():.1f}deg")
print(f"  Scale anisotropy - Coeff: {coeff_scale_ratios.mean():.2f}, Basis: {basis_scale_ratios.mean():.2f}")
print("="*70)
