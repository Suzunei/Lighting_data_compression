"""
HI-NE-GBD 实时解压模块 (Real-time Decoder)

功能：
    - 加载训练好的 HI-NE-GBD 压缩模型参数
    - 对任意输入坐标 x (世界坐标或归一化坐标) 实时解压，输出 27 维 SH 系数

使用方式：
    1. 训练完成后，HI-NE-GBD.py 会自动保存模型到 compressed_model.pth
    2. 使用本模块加载并实时解压：
    
        from decoder import HINEGBDDecoder
        
        decoder = HINEGBDDecoder("compressed_model.pth")
        
        # 单点解压 (世界坐标)
        sh_coeffs = decoder.decode(x=100.0, y=50.0, z=-200.0)
        
        # 批量解压 (numpy array)
        coords = np.array([[100, 50, -200], [110, 60, -180]], dtype=np.float32)
        sh_batch = decoder.decode_batch(coords)
        
    3. 命令行交互模式：
        python decoder.py --model compressed_model.pth --interactive
"""

import numpy as np
import torch
import torch.nn as nn
import os
import argparse
import json
import time

# ==================== 全局配置 ====================
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'compressed_model.pth')  # 模型文件路径
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 推理设备: 'cuda' 或 'cpu'


# ==================== 模型定义（与训练代码保持一致） ====================

def quaternion_to_rotation_matrix(q):
    """将四元数转换为旋转矩阵。q: [..., 4] (w, x, y, z)"""
    q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = torch.stack([
        torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y], dim=-1),
        torch.stack([2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x], dim=-1),
        torch.stack([2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y], dim=-1)
    ], dim=-2)
    return R


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    def __init__(self, num_freqs=6):
        super().__init__()
        self.num_freqs = num_freqs
        self.freq_bands = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)

    def forward(self, x):
        device = x.device
        freq_bands = self.freq_bands.to(device)
        encoded = [x]
        for freq in freq_bands:
            encoded.append(torch.sin(freq * np.pi * x))
            encoded.append(torch.cos(freq * np.pi * x))
        return torch.cat(encoded, dim=-1)

    def get_output_dim(self, input_dim=3):
        return input_dim * (1 + 2 * self.num_freqs)


class MBDCompressor3D(nn.Module):
    """HI-NE-GBD 模型结构（用于推理/解压）"""

    def __init__(self, num_bases=6, coeff_res=12, basis_res=8, data_dim=3,
                 coeff_kernel_scale=0.15, basis_kernel_scale=0.2, mlp_hidden=64,
                 pe_num_freqs=4, fine_mlp_depth=2, fine_gaussian_res=16, fine_kernel_scale=0.08,
                 latent_dim=32):
        super().__init__()
        self.L = num_bases
        self.data_dim = data_dim
        self.mlp_hidden = mlp_hidden
        self.pe_num_freqs = pe_num_freqs
        self.fine_gaussian_res = fine_gaussian_res
        self.latent_dim = latent_dim

        # Coefficient 3D Gaussian Parameters
        self.coeff_mu = nn.Parameter(torch.rand(coeff_res, 3))
        self.coeff_log_s = nn.Parameter(torch.ones(coeff_res, 3) * np.log(coeff_kernel_scale))
        self.coeff_q = nn.Parameter(torch.zeros(coeff_res, 4))
        self.coeff_alpha = nn.Parameter(torch.zeros(coeff_res))

        # Basis 3D Gaussian Parameters
        self.basis_mu = nn.Parameter(torch.rand(basis_res, 3))
        self.basis_log_s = nn.Parameter(torch.ones(basis_res, 3) * np.log(basis_kernel_scale))
        self.basis_q = nn.Parameter(torch.zeros(basis_res, 4))
        self.basis_alpha = nn.Parameter(torch.zeros(basis_res))

        # Fine Gaussian Parameters
        self.fine_mu = nn.Parameter(torch.rand(fine_gaussian_res, 3))
        self.fine_log_s = nn.Parameter(torch.ones(fine_gaussian_res, 3) * np.log(fine_kernel_scale))
        self.fine_q = nn.Parameter(torch.zeros(fine_gaussian_res, 4))
        self.fine_alpha = nn.Parameter(torch.zeros(fine_gaussian_res))
        # Per-anchor latent V: [F, K] (paper Eq.5)
        self.fine_features = nn.Parameter(torch.randn(fine_gaussian_res, latent_dim) * (1.0 / np.sqrt(latent_dim)))

        # MBD Coefficient/Basis Tensors
        self.C = nn.Parameter(torch.randn(coeff_res, self.L) * 0.1)
        self.B = nn.Parameter(torch.randn(basis_res, self.L, self.data_dim) * 0.1)
        self.mbd_log_scale = nn.Parameter(torch.zeros(self.L))

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(num_freqs=pe_num_freqs)
        pe_dim = self.pos_encoder.get_output_dim(3)

        # Fine Branch MLP — input: PE(x) + mixed latent F(x) [K]
        fine_input_dim = pe_dim + latent_dim
        fine_layers = []
        fine_layers.append(nn.Linear(fine_input_dim, mlp_hidden))
        fine_layers.append(nn.ReLU())
        for _ in range(fine_mlp_depth - 1):
            fine_layers.append(nn.Linear(mlp_hidden, mlp_hidden))
            fine_layers.append(nn.ReLU())
        fine_layers.append(nn.Linear(mlp_hidden, data_dim))
        self.fine_mlp = nn.Sequential(*fine_layers)

        # Residual Refiner
        self.residual_refiner = nn.Sequential(
            nn.Linear(data_dim + 3, mlp_hidden // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden // 2, data_dim)
        )

        self.M = coeff_res
        self.N = basis_res
        self.F = fine_gaussian_res

    def gaussian_function_3d(self, p, mu, s, q):
        """计算 3D 高斯函数值。p:[N,3], mu:[K,3], s:[K,3], q:[K,4] -> [N,K]"""
        R = quaternion_to_rotation_matrix(q)
        s_inv_sq = 1.0 / (s ** 2 + 1e-8)
        S_inv_sq = torch.diag_embed(s_inv_sq)
        precision = R @ S_inv_sq @ R.transpose(-1, -2)
        diff = p.unsqueeze(1) - mu.unsqueeze(0)
        diff_expanded = diff.unsqueeze(-1)
        precision_expanded = precision.unsqueeze(0)
        mahalanobis_sq = (diff_expanded.transpose(-1, -2) @ precision_expanded @ diff_expanded).squeeze(-1).squeeze(-1)
        return torch.exp(-0.5 * mahalanobis_sq)

    def compute_gaussian_weights_3d(self, query_pts, mu, log_s, q, alpha=None):
        """计算归一化 3D 高斯权重"""
        s = torch.exp(log_s)
        gaussian_vals = self.gaussian_function_3d(query_pts, mu, s, q)
        if alpha is not None:
            intensity = torch.sigmoid(alpha)
            gaussian_vals = gaussian_vals * intensity.unsqueeze(0)
        weights = gaussian_vals / (gaussian_vals.sum(dim=1, keepdim=True) + 1e-8)
        return weights

    def forward(self, coords):
        """
        解压前向传播：输入归一化坐标，输出重建信号。
        
        Args:
            coords: [Q, 3] 归一化坐标 (范围 [0, 1])
            
        Returns:
            reconstruction: [Q, D] 重建的 SH 系数
        """
        # Coarse Branch: MBD
        phi_weights = self.compute_gaussian_weights_3d(
            coords, self.coeff_mu, self.coeff_log_s, self.coeff_q, self.coeff_alpha
        )
        psi_weights = self.compute_gaussian_weights_3d(
            coords, self.basis_mu, self.basis_log_s, self.basis_q, self.basis_alpha
        )

        moving_coeff = torch.matmul(phi_weights, self.C)
        B_flat = self.B.view(-1, self.L * self.data_dim)
        basis_interp_flat = torch.matmul(psi_weights, B_flat)
        moving_basis = basis_interp_flat.view(-1, self.L, self.data_dim)

        mbd_scale = torch.exp(self.mbd_log_scale)
        scaled_coeff = moving_coeff * mbd_scale.unsqueeze(0)
        coarse_output = torch.sum(scaled_coeff.unsqueeze(-1) * moving_basis, dim=1)

        # Fine Branch: paper-aligned mixed-latent MLP (GPC Eq.5/7)
        fine_weights = self.compute_gaussian_weights_3d(
            coords, self.fine_mu, self.fine_log_s, self.fine_q, self.fine_alpha
        )                                                              # [Q, F]
        mixed_latent = torch.matmul(fine_weights, self.fine_features)  # [Q, K]   Eq.5
        coords_encoded = self.pos_encoder(coords)                      # [Q, pe_dim]
        fine_input = torch.cat([coords_encoded, mixed_latent], dim=-1) # [Q, pe_dim + K]
        fine_output = self.fine_mlp(fine_input)                        # [Q, D]   Eq.7

        # Additive Blending
        blended = coarse_output + fine_output

        # Residual Refinement
        refine_input = torch.cat([blended, coords], dim=1)
        residual = self.residual_refiner(refine_input)
        reconstruction = blended + 0.1 * residual

        return reconstruction


# ==================== 实时解压器 ====================

class HINEGBDDecoder:
    """
    HI-NE-GBD 实时解压器
    
    加载训练好的压缩模型，根据输入坐标实时重建 SH 系数。
    
    Usage:
        decoder = HINEGBDDecoder("compressed_model.pth")
        
        # 单点查询（世界坐标）
        sh = decoder.decode(x=100.0, y=50.0, z=-200.0)
        
        # 批量查询
        coords = np.array([[100, 50, -200], [110, 60, -180]])
        sh_batch = decoder.decode_batch(coords)
    """

    def __init__(self, model_path, device='cpu'):
        """
        初始化解压器。
        
        Args:
            model_path: 模型文件路径 (.pth)
            device: 推理设备 ('cpu' 或 'cuda')
        """
        self.device = torch.device(device)
        
        # 加载 checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # 提取模型配置
        self.config = checkpoint['model_config']
        
        # 提取位置归一化参数（世界坐标 <-> 归一化坐标）
        self.pos_min = np.array(checkpoint['pos_min'], dtype=np.float32)
        self.pos_max = np.array(checkpoint['pos_max'], dtype=np.float32)
        self.pos_range = self.pos_max - self.pos_min
        self.pos_range[self.pos_range < 1e-6] = 1.0
        
        # 数据维度信息
        self.data_dim = self.config['data_dim']
        self.num_probes = checkpoint.get('num_probes', 0)
        
        # 构建模型并加载权重
        self.model = MBDCompressor3D(**self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # 统计信息
        total_params = sum(p.numel() for p in self.model.parameters())
        storage_bytes = checkpoint.get('storage_bytes', total_params * 2)  # 默认 FP16
        
        print(f"[HI-NE-GBD Decoder] 模型加载成功")
        print(f"  模型路径: {model_path}")
        print(f"  输出维度: {self.data_dim} (SH coefficients)")
        print(f"  模型参数: {total_params:,}")
        print(f"  存储大小: {storage_bytes/1024:.1f} KB")
        print(f"  位置范围: X[{self.pos_min[0]:.1f}, {self.pos_max[0]:.1f}], "
              f"Y[{self.pos_min[1]:.1f}, {self.pos_max[1]:.1f}], "
              f"Z[{self.pos_min[2]:.1f}, {self.pos_max[2]:.1f}]")
        print(f"  设备: {self.device}")

    def _normalize_coords(self, coords_world):
        """
        将世界坐标归一化到 [0, 1]。
        
        Args:
            coords_world: [N, 3] numpy array, 世界坐标
        Returns:
            [N, 3] numpy array, 归一化坐标
        """
        return (coords_world - self.pos_min) / self.pos_range

    def decode(self, x, y, z, normalized=False):
        """
        单点实时解压：输入一个 3D 坐标，返回该位置的 SH 系数。
        
        Args:
            x, y, z: 查询点的坐标
            normalized: 若 True，表示输入已是归一化坐标 [0,1]；否则为世界坐标
            
        Returns:
            np.ndarray: [27] SH 系数 (RGB 交错存储)
        """
        coord = np.array([[x, y, z]], dtype=np.float32)
        if not normalized:
            coord = self._normalize_coords(coord)
        
        coord_tensor = torch.from_numpy(coord).to(self.device)
        
        with torch.no_grad():
            result = self.model(coord_tensor)
        
        return result.cpu().numpy().squeeze(0)

    def decode_batch(self, coords, normalized=False, batch_size=8192):
        """
        批量实时解压：输入多个 3D 坐标，返回对应的 SH 系数。
        
        Args:
            coords: [N, 3] numpy array, 查询坐标
            normalized: 若 True，输入已是归一化坐标；否则为世界坐标
            batch_size: 推理时的批大小（避免 OOM）
            
        Returns:
            np.ndarray: [N, 27] SH 系数
        """
        coords = np.asarray(coords, dtype=np.float32)
        if coords.ndim == 1:
            coords = coords.reshape(1, 3)
        
        if not normalized:
            coords = self._normalize_coords(coords)
        
        N = coords.shape[0]
        results = []
        
        with torch.no_grad():
            for i in range(0, N, batch_size):
                batch = torch.from_numpy(coords[i:i+batch_size]).to(self.device)
                output = self.model(batch)
                results.append(output.cpu().numpy())
        
        return np.concatenate(results, axis=0)

    def decode_grid(self, resolution=16):
        """
        在归一化空间 [0,1]^3 中生成均匀网格并解压全部 SH 系数。
        适合可视化或验证。
        
        Args:
            resolution: 每个轴的采样点数
            
        Returns:
            coords: [R^3, 3] 归一化网格坐标
            sh_values: [R^3, 27] 对应 SH 系数
        """
        lin = np.linspace(0, 1, resolution)
        xx, yy, zz = np.meshgrid(lin, lin, lin, indexing='ij')
        coords = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1).astype(np.float32)
        
        sh_values = self.decode_batch(coords, normalized=True)
        return coords, sh_values

    def benchmark(self, num_queries=10000, num_runs=10):
        """
        性能基准测试：测量解压速度。
        
        Args:
            num_queries: 每次查询的点数
            num_runs: 运行次数
            
        Returns:
            dict: 包含延迟和吞吐量统计
        """
        coords = np.random.rand(num_queries, 3).astype(np.float32)
        coord_tensor = torch.from_numpy(coords).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                self.model(coord_tensor)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                self.model(coord_tensor)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append(end - start)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        throughput = num_queries / avg_time
        
        results = {
            'num_queries': num_queries,
            'avg_time_ms': avg_time * 1000,
            'std_time_ms': std_time * 1000,
            'throughput_pts_per_sec': throughput,
            'per_point_us': (avg_time / num_queries) * 1e6
        }
        
        print(f"\n[Benchmark] 解压性能测试")
        print(f"  查询点数: {num_queries:,}")
        print(f"  平均耗时: {results['avg_time_ms']:.2f} ± {results['std_time_ms']:.2f} ms")
        print(f"  吞吐量: {results['throughput_pts_per_sec']:,.0f} pts/sec")
        print(f"  单点延迟: {results['per_point_us']:.2f} μs")
        
        return results


# ==================== 模型保存工具函数 ====================

def save_compressed_model(model, save_path, model_config, pos_min, pos_max, 
                          num_probes=0, use_fp16=True):
    """
    保存训练完成的压缩模型。
    
    Args:
        model: 训练好的 MBDCompressor3D 模型
        save_path: 保存路径 (.pth)
        model_config: 模型配置字典（构造函数参数）
        pos_min: [3] 位置最小值（世界坐标）
        pos_max: [3] 位置最大值（世界坐标）
        num_probes: 原始探针数量
        use_fp16: 是否以 FP16 精度存储参数
    """
    # 准备 state_dict
    state_dict = model.state_dict()
    
    # FP16 量化存储
    if use_fp16:
        state_dict_save = {k: v.half() for k, v in state_dict.items()}
        bytes_per_param = 2
    else:
        state_dict_save = state_dict
        bytes_per_param = 4
    
    total_params = sum(v.numel() for v in state_dict_save.values())
    storage_bytes = total_params * bytes_per_param
    
    checkpoint = {
        'model_config': model_config,
        'model_state_dict': state_dict_save,
        'pos_min': pos_min.tolist() if hasattr(pos_min, 'tolist') else list(pos_min),
        'pos_max': pos_max.tolist() if hasattr(pos_max, 'tolist') else list(pos_max),
        'num_probes': num_probes,
        'storage_bytes': storage_bytes,
        'precision': 'fp16' if use_fp16 else 'fp32',
    }
    
    torch.save(checkpoint, save_path)
    
    print(f"\n[Save] 压缩模型已保存")
    print(f"  路径: {save_path}")
    print(f"  参数数量: {total_params:,}")
    print(f"  存储精度: {'FP16' if use_fp16 else 'FP32'}")
    print(f"  文件大小: {os.path.getsize(save_path)/1024:.1f} KB")
    print(f"  有效压缩数据: {storage_bytes/1024:.1f} KB")


# ==================== 命令行交互入口 ====================

def interactive_mode(decoder):
    """交互式解压模式"""
    print("\n" + "="*60)
    print("HI-NE-GBD 实时解压 - 交互模式")
    print("="*60)
    print("输入格式: x y z （世界坐标，空格分隔）")
    print("特殊命令:")
    print("  'n x y z' - 输入归一化坐标 [0,1]")
    print("  'bench'   - 运行性能基准测试")
    print("  'info'    - 显示模型信息")
    print("  'quit'    - 退出")
    print("="*60)
    
    while True:
        try:
            user_input = input("\n> 输入坐标: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ('quit', 'exit', 'q'):
                print("退出解压器。")
                break
            
            if user_input.lower() == 'bench':
                decoder.benchmark()
                continue
            
            if user_input.lower() == 'info':
                print(f"  数据维度: {decoder.data_dim}")
                print(f"  探针数量: {decoder.num_probes}")
                print(f"  位置范围: X[{decoder.pos_min[0]:.1f}, {decoder.pos_max[0]:.1f}]")
                print(f"            Y[{decoder.pos_min[1]:.1f}, {decoder.pos_max[1]:.1f}]")
                print(f"            Z[{decoder.pos_min[2]:.1f}, {decoder.pos_max[2]:.1f}]")
                continue
            
            # 解析坐标
            normalized = False
            if user_input.startswith('n '):
                normalized = True
                user_input = user_input[2:]
            
            parts = user_input.replace(',', ' ').split()
            if len(parts) != 3:
                print("  [错误] 请输入3个坐标值，例如: 100 50 -200")
                continue
            
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            
            # 执行解压
            start_time = time.perf_counter()
            sh_coeffs = decoder.decode(x, y, z, normalized=normalized)
            elapsed = (time.perf_counter() - start_time) * 1000
            
            # 显示结果
            coord_type = "归一化" if normalized else "世界"
            print(f"  [{coord_type}坐标] ({x:.2f}, {y:.2f}, {z:.2f})")
            print(f"  [解压耗时] {elapsed:.3f} ms")
            print(f"  [SH 系数] ({decoder.data_dim} 维):")
            
            # 按 SH order 分组显示
            # 9 bands × 3 RGB, interleaved: [SH0_R, SH0_G, SH0_B, SH1_R, ...]
            for band in range(9):
                r, g, b = sh_coeffs[band*3], sh_coeffs[band*3+1], sh_coeffs[band*3+2]
                print(f"    SH{band}: R={r:+.6f}  G={g:+.6f}  B={b:+.6f}")
                
        except ValueError as e:
            print(f"  [错误] 无法解析输入: {e}")
        except KeyboardInterrupt:
            print("\n退出解压器。")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HI-NE-GBD 实时解压器")
    parser.add_argument("--model", type=str, default=MODEL_PATH,
                        help=f"压缩模型文件路径 (默认: {MODEL_PATH})")
    parser.add_argument("--device", type=str, default=DEVICE,
                        choices=["cpu", "cuda"],
                        help=f"推理设备 (默认: {DEVICE})")
    parser.add_argument("--interactive", action="store_true",
                        help="进入交互式解压模式")
    parser.add_argument("--benchmark", action="store_true",
                        help="运行性能基准测试")
    parser.add_argument("--coords", type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                        help="直接指定查询坐标 (世界坐标)")
    parser.add_argument("--normalized", action="store_true",
                        help="输入坐标为归一化坐标 [0,1]")
    
    args = parser.parse_args()
    
    # 解析模型路径
    model_path = args.model
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)
    
    if not os.path.exists(model_path):
        print(f"[错误] 模型文件不存在: {model_path}")
        print(f"  请先运行 buildtime/HI-NE-GBD.py 完成训练并保存模型。")
        exit(1)
    
    # 初始化解压器
    decoder = HINEGBDDecoder(model_path, device=args.device)
    
    # 执行相应操作
    if args.benchmark:
        decoder.benchmark()
    elif args.coords:
        x, y, z = args.coords
        sh = decoder.decode(x, y, z, normalized=args.normalized)
        print(f"\n查询坐标: ({x}, {y}, {z})")
        print(f"SH 系数 ({decoder.data_dim}D):")
        for band in range(9):
            r, g, b = sh[band*3], sh[band*3+1], sh[band*3+2]
            print(f"  SH{band}: R={r:+.6f}  G={g:+.6f}  B={b:+.6f}")
    elif args.interactive:
        interactive_mode(decoder)
    else:
        # 默认进入交互模式
        interactive_mode(decoder)
