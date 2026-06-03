"""
HI-NE-GBD 模型导出工具

将 PyTorch .pth 模型导出为 UCommon FFileArchive 兼容的二进制格式 (.uasset)，
供 C++ 运行时加载并执行前向传播。

使用方式：
    python export_model.py --model compressed_model.pth --output hinegbd_model.uasset

二进制格式说明（与 UCommon::FFileArchive 完全兼容）：
    [FFileArchive Header - 16 bytes]
        Magic: "Ubpa" (4 bytes)
        NumVersionKeys: uint32 (4 bytes)
        VersionMapOffset: uint64 (8 bytes)
    [Model Config - metadata]
        data_dim, num_bases, coeff_res, basis_res, fine_gaussian_res: uint32 x5
        pe_num_freqs, mlp_hidden, fine_mlp_depth: uint32 x3
        num_fine_mlp_linear_layers, num_residual_linear_layers: uint32 x2
        pos_min[3], pos_max[3]: float32 x6
    [Weight Tensors - sequential]
        每个 tensor: uint64(元素数量) + float32[N](数据)
    [Version Map at VersionMapOffset]
        Key: uint64, Version: int64
"""

import os
import sys
import struct
import argparse
import numpy as np
import torch


def write_tensor(f, tensor, use_fp16=True):
    """写入一个 tensor：先写元素数量 (uint64)，再写数据
    
    use_fp16=True 时写 float16 (2 bytes/element)，否则写 float32 (4 bytes/element)
    """
    if use_fp16:
        data = tensor.detach().cpu().half().numpy().flatten()
    else:
        data = tensor.detach().cpu().float().numpy().flatten()
    num_elements = len(data)
    f.write(struct.pack('<Q', num_elements))  # uint64 元素数量
    f.write(data.tobytes())  # float16 or float32 data


def export_model(model_path, output_path):
    """将 .pth 模型导出为 FFileArchive 兼容的二进制文件"""
    
    # 加载 checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    config = checkpoint['model_config']
    state_dict = checkpoint['model_state_dict']
    pos_min = np.array(checkpoint['pos_min'], dtype=np.float32)
    pos_max = np.array(checkpoint['pos_max'], dtype=np.float32)
    
    # 保持原始精度（不再强制转 FP32）
    # state_dict 中的权重可能是 FP16 或 FP32
    
    # 提取配置参数
    data_dim = config['data_dim']
    num_bases = config['num_bases']
    coeff_res = config['coeff_res']
    basis_res = config['basis_res']
    fine_gaussian_res = config['fine_gaussian_res']
    pe_num_freqs = config['pe_num_freqs']
    mlp_hidden = config['mlp_hidden']
    fine_mlp_depth = config['fine_mlp_depth']
    
    # 计算 fine_mlp 的层数信息
    pe_dim = 3 * (1 + 2 * pe_num_freqs)  # positional encoding output dim
    fine_input_dim = pe_dim + fine_gaussian_res
    
    # 收集 MLP 层的 weight/bias keys (只取 Linear 层)
    fine_mlp_keys = []
    residual_refiner_keys = []
    for k in sorted(state_dict.keys()):
        if k.startswith('fine_mlp.') and ('weight' in k or 'bias' in k):
            fine_mlp_keys.append(k)
        elif k.startswith('residual_refiner.') and ('weight' in k or 'bias' in k):
            residual_refiner_keys.append(k)
    
    # 按层序号排序（weight 在 bias 前面）
    def sort_key(k):
        parts = k.split('.')
        return (int(parts[1]), 0 if 'weight' in k else 1)
    
    fine_mlp_keys.sort(key=sort_key)
    residual_refiner_keys.sort(key=sort_key)
    
    num_fine_mlp_linear_layers = len(fine_mlp_keys) // 2
    num_residual_linear_layers = len(residual_refiner_keys) // 2
    
    # FP16 标志
    use_fp16 = True  # 默认使用 FP16 存储以减小文件体积
    
    print(f"[Export] 模型配置:")
    print(f"  data_dim={data_dim}, num_bases(L)={num_bases}")
    print(f"  coeff_res(M)={coeff_res}, basis_res(N)={basis_res}, fine_gaussian_res(F)={fine_gaussian_res}")
    print(f"  pe_num_freqs={pe_num_freqs}, mlp_hidden={mlp_hidden}, fine_mlp_depth={fine_mlp_depth}")
    print(f"  pe_dim={pe_dim}, fine_input_dim={fine_input_dim}")
    print(f"  pos_min={pos_min}, pos_max={pos_max}")
    print(f"  Fine MLP Linear layers ({num_fine_mlp_linear_layers}): {fine_mlp_keys}")
    print(f"  Residual refiner layers ({num_residual_linear_layers}): {residual_refiner_keys}")
    print(f"  存储精度: {'FP16' if use_fp16 else 'FP32'}")
    
    # === 写入二进制文件 ===
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        # --- FFileArchive Header (16 bytes) ---
        f.write(b'Ubpa')                        # Magic
        f.write(struct.pack('<I', 1))           # NumVersionKeys = 1
        f.write(struct.pack('<Q', 0))           # VersionMapOffset (placeholder)
        
        # --- Model Config Metadata ---
        f.write(struct.pack('<I', data_dim))
        f.write(struct.pack('<I', num_bases))
        f.write(struct.pack('<I', coeff_res))
        f.write(struct.pack('<I', basis_res))
        f.write(struct.pack('<I', fine_gaussian_res))
        f.write(struct.pack('<I', pe_num_freqs))
        f.write(struct.pack('<I', mlp_hidden))
        f.write(struct.pack('<I', fine_mlp_depth))
        f.write(struct.pack('<I', num_fine_mlp_linear_layers))
        f.write(struct.pack('<I', num_residual_linear_layers))
        f.write(struct.pack('<I', 1 if use_fp16 else 0))  # UseFP16 flag
        
        # Position normalization params
        f.write(pos_min.tobytes())  # float32 x 3 = 12 bytes
        f.write(pos_max.tobytes())  # float32 x 3 = 12 bytes
        
        # --- Gaussian Parameters ---
        write_tensor(f, state_dict['coeff_mu'], use_fp16)         # [M, 3]
        write_tensor(f, state_dict['coeff_log_s'], use_fp16)      # [M, 3]
        write_tensor(f, state_dict['coeff_q'], use_fp16)          # [M, 4]
        write_tensor(f, state_dict['coeff_alpha'], use_fp16)      # [M]
        
        write_tensor(f, state_dict['basis_mu'], use_fp16)         # [N, 3]
        write_tensor(f, state_dict['basis_log_s'], use_fp16)      # [N, 3]
        write_tensor(f, state_dict['basis_q'], use_fp16)          # [N, 4]
        write_tensor(f, state_dict['basis_alpha'], use_fp16)      # [N]
        
        write_tensor(f, state_dict['fine_mu'], use_fp16)          # [F, 3]
        write_tensor(f, state_dict['fine_log_s'], use_fp16)       # [F, 3]
        write_tensor(f, state_dict['fine_q'], use_fp16)           # [F, 4]
        write_tensor(f, state_dict['fine_alpha'], use_fp16)       # [F]
        write_tensor(f, state_dict['fine_features'], use_fp16)    # [F, D]
        
        # MBD Tensors
        write_tensor(f, state_dict['C'], use_fp16)                # [M, L]
        write_tensor(f, state_dict['B'], use_fp16)                # [N, L, D]
        write_tensor(f, state_dict['mbd_log_scale'], use_fp16)    # [L]
        
        # --- MLP Weights (weight then bias for each Linear layer) ---
        for key in fine_mlp_keys:
            write_tensor(f, state_dict[key], use_fp16)
        
        for key in residual_refiner_keys:
            write_tensor(f, state_dict[key], use_fp16)
        
        # --- Version Map ---
        version_map_offset = f.tell()
        # Version Key = "HINEGBD\0" as uint64
        version_key = 0x48494E4547424400
        version_value = 1
        f.write(struct.pack('<Q', version_key))   # uint64 key
        f.write(struct.pack('<q', version_value)) # int64 version
        
        # Overwrite header with correct VersionMapOffset
        f.seek(8)
        f.write(struct.pack('<Q', version_map_offset))
    
    file_size = os.path.getsize(output_path)
    print(f"\n[Export] 导出成功!")
    print(f"  输出路径: {output_path}")
    print(f"  文件大小: {file_size / 1024:.1f} KB")


def verify_export(output_path):
    """验证导出的二进制文件格式正确性"""
    with open(output_path, 'rb') as f:
        magic = f.read(4)
        assert magic == b'Ubpa', f"Magic mismatch: {magic}"
        
        num_version_keys = struct.unpack('<I', f.read(4))[0]
        version_map_offset = struct.unpack('<Q', f.read(8))[0]
        
        print(f"\n[Verify] Header:")
        print(f"  Magic: {magic}")
        print(f"  NumVersionKeys: {num_version_keys}")
        print(f"  VersionMapOffset: {version_map_offset}")
        
        # Read config
        data_dim = struct.unpack('<I', f.read(4))[0]
        num_bases = struct.unpack('<I', f.read(4))[0]
        coeff_res = struct.unpack('<I', f.read(4))[0]
        basis_res = struct.unpack('<I', f.read(4))[0]
        fine_gaussian_res = struct.unpack('<I', f.read(4))[0]
        pe_num_freqs = struct.unpack('<I', f.read(4))[0]
        mlp_hidden = struct.unpack('<I', f.read(4))[0]
        fine_mlp_depth = struct.unpack('<I', f.read(4))[0]
        num_fine_mlp_layers = struct.unpack('<I', f.read(4))[0]
        num_residual_layers = struct.unpack('<I', f.read(4))[0]
        use_fp16 = struct.unpack('<I', f.read(4))[0]
        
        pos_min = np.frombuffer(f.read(12), dtype=np.float32)
        pos_max = np.frombuffer(f.read(12), dtype=np.float32)
        
        print(f"  Config: data_dim={data_dim}, L={num_bases}, M={coeff_res}, N={basis_res}, F={fine_gaussian_res}")
        print(f"  PE freqs={pe_num_freqs}, MLP hidden={mlp_hidden}, depth={fine_mlp_depth}")
        print(f"  Fine MLP layers: {num_fine_mlp_layers}, Residual layers: {num_residual_layers}")
        print(f"  UseFP16: {use_fp16}")
        print(f"  pos_min={pos_min}, pos_max={pos_max}")
        
        # 读取第一个 tensor 验证
        bytes_per_elem = 2 if use_fp16 else 4
        tensor_dtype = np.float16 if use_fp16 else np.float32
        n = struct.unpack('<Q', f.read(8))[0]
        data = np.frombuffer(f.read(n * bytes_per_elem), dtype=tensor_dtype)
        print(f"  First tensor (coeff_mu): shape=({coeff_res}, 3), elements={n}, dtype={tensor_dtype.__name__}, sample={data[:6].astype(np.float32)}")
        
        # 验证 version map
        f.seek(version_map_offset)
        for i in range(num_version_keys):
            key = struct.unpack('<Q', f.read(8))[0]
            version = struct.unpack('<q', f.read(8))[0]
            print(f"  Version[{i}]: key=0x{key:016X}, version={version}")
    
    print(f"[Verify] 格式验证通过!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HI-NE-GBD 模型导出为 UCommon FFileArchive 格式")
    parser.add_argument("--model", type=str, default="compressed_model.pth",
                        help="输入模型路径 (.pth)")
    parser.add_argument("--output", type=str, default="hinegbd_model.uasset",
                        help="输出二进制路径 (.uasset)")
    parser.add_argument("--verify", action="store_true",
                        help="验证导出文件格式")
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = args.model if os.path.isabs(args.model) else os.path.join(script_dir, args.model)
    output_path = args.output if os.path.isabs(args.output) else os.path.join(script_dir, args.output)
    
    if not os.path.exists(model_path):
        print(f"[错误] 模型文件不存在: {model_path}")
        sys.exit(1)
    
    export_model(model_path, output_path)
    
    if args.verify:
        verify_export(output_path)
