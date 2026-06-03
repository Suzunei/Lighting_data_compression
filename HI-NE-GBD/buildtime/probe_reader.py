"""
ILCSampleData 探针数据读取器 (Probe Reader)

功能：
    - 读取 ILCSampleData 二进制探针文件
    - 随机返回一个或多个探针的坐标与 SH 系数
    - 支持按索引查询

数据格式：
    每探针 32 个 float32：
        - Position: 3 floats (x, y, z) — 世界坐标
        - Radius:   1 float
        - SH Coeff: 27 floats (9 bands × 3 RGB, 交错存储)
        - Shadow:   1 float

使用方式：
    from probe_reader import ProbeReader
    
    reader = ProbeReader("ILCSampleData_0.bin")
    
    # 随机返回一个探针
    pos, sh = reader.random_probe()
    
    # 随机返回 N 个探针
    positions, sh_coeffs = reader.random_probes(n=10)
    
    # 按索引查询
    pos, sh = reader.get_probe(index=42)
    
    # 命令行使用
    python probe_reader.py --bin ILCSampleData_0.bin --random 5
"""

import numpy as np
import os
import argparse


class ProbeReader:
    """
    ILCSampleData 探针数据读取器
    
    读取二进制探针文件，提供随机/索引查询接口，
    返回探针的世界坐标和 27 维 SH 系数。
    """

    # 每探针的 float 数量
    FLOATS_PER_PROBE = 32
    # 各字段偏移
    POS_OFFSET = 0       # [0:3]   位置 xyz
    RADIUS_OFFSET = 3    # [3]     半径
    SH_OFFSET = 4        # [4:31]  SH 系数 (27维)
    SHADOW_OFFSET = 31   # [31]    阴影值

    def __init__(self, bin_path):
        """
        初始化探针读取器。
        
        Args:
            bin_path: ILCSampleData 二进制文件路径
        """
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"探针文件不存在: {bin_path}")
        
        # 读取全部数据
        raw = np.fromfile(bin_path, dtype=np.float32)
        if raw.size % self.FLOATS_PER_PROBE != 0:
            raise ValueError(
                f"文件大小不是 {self.FLOATS_PER_PROBE} floats 的整数倍，"
                f"可能不是有效的 ILCSampleData 文件"
            )
        
        self.data = raw.reshape(-1, self.FLOATS_PER_PROBE)
        self.num_probes = self.data.shape[0]
        
        # 预解析各字段
        self.positions = self.data[:, self.POS_OFFSET:self.POS_OFFSET+3]      # [N, 3]
        self.radii = self.data[:, self.RADIUS_OFFSET]                          # [N]
        self.sh_coeffs = self.data[:, self.SH_OFFSET:self.SH_OFFSET+27]       # [N, 27]
        self.shadows = self.data[:, self.SHADOW_OFFSET]                        # [N]
        
        # 位置范围（用于归一化）
        self.pos_min = self.positions.min(axis=0)
        self.pos_max = self.positions.max(axis=0)
        
        print(f"[ProbeReader] 加载完成: {os.path.basename(bin_path)}")
        print(f"  探针数量: {self.num_probes:,}")
        print(f"  位置范围: X[{self.pos_min[0]:.1f}, {self.pos_max[0]:.1f}], "
              f"Y[{self.pos_min[1]:.1f}, {self.pos_max[1]:.1f}], "
              f"Z[{self.pos_min[2]:.1f}, {self.pos_max[2]:.1f}]")

    def get_probe(self, index):
        """
        按索引获取单个探针的数据。
        
        Args:
            index: 探针索引 (0 ~ num_probes-1)
            
        Returns:
            position: [3] 世界坐标 (x, y, z)
            sh_coeffs: [27] SH 系数 (RGB 交错)
        """
        if index < 0 or index >= self.num_probes:
            raise IndexError(f"索引 {index} 超出范围 [0, {self.num_probes-1}]")
        return self.positions[index].copy(), self.sh_coeffs[index].copy()

    def get_probe_full(self, index):
        """
        按索引获取探针的完整数据（含半径和阴影）。
        
        Args:
            index: 探针索引
            
        Returns:
            dict: {position, radius, sh_coeffs, shadow}
        """
        if index < 0 or index >= self.num_probes:
            raise IndexError(f"索引 {index} 超出范围 [0, {self.num_probes-1}]")
        return {
            'position': self.positions[index].copy(),
            'radius': float(self.radii[index]),
            'sh_coeffs': self.sh_coeffs[index].copy(),
            'shadow': float(self.shadows[index])
        }

    def random_probe(self):
        """
        随机返回一个探针的坐标和 SH 系数。
        
        Returns:
            position: [3] 世界坐标 (x, y, z)
            sh_coeffs: [27] SH 系数 (RGB 交错)
        """
        idx = np.random.randint(0, self.num_probes)
        return self.get_probe(idx)

    def random_probes(self, n=1):
        """
        随机返回 N 个探针的坐标和 SH 系数。
        
        Args:
            n: 返回的探针数量
            
        Returns:
            positions: [N, 3] 世界坐标
            sh_coeffs: [N, 27] SH 系数
        """
        n = min(n, self.num_probes)
        indices = np.random.choice(self.num_probes, size=n, replace=False)
        return self.positions[indices].copy(), self.sh_coeffs[indices].copy()

    def get_normalized_position(self, index):
        """
        获取探针的归一化坐标 [0, 1]（与 decoder 输入一致）。
        
        Args:
            index: 探针索引
            
        Returns:
            [3] 归一化坐标
        """
        pos = self.positions[index]
        pos_range = self.pos_max - self.pos_min
        pos_range[pos_range < 1e-6] = 1.0
        return (pos - self.pos_min) / pos_range

    def print_probe(self, index=None):
        """
        打印一个探针的详细信息。
        
        Args:
            index: 探针索引，None 则随机选取
        """
        if index is None:
            index = np.random.randint(0, self.num_probes)
        
        probe = self.get_probe_full(index)
        pos = probe['position']
        norm_pos = self.get_normalized_position(index)
        sh = probe['sh_coeffs']
        
        print(f"\n{'='*50}")
        print(f"探针 #{index}")
        print(f"{'='*50}")
        print(f"  世界坐标: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
        print(f"  归一化坐标: ({norm_pos[0]:.6f}, {norm_pos[1]:.6f}, {norm_pos[2]:.6f})")
        print(f"  半径: {probe['radius']:.4f}")
        print(f"  阴影: {probe['shadow']:.4f}")
        print(f"  SH 系数 (27维, RGB交错):")
        for band in range(9):
            r, g, b = sh[band*3], sh[band*3+1], sh[band*3+2]
            print(f"    SH{band}: R={r:+.6f}  G={g:+.6f}  B={b:+.6f}")
        print(f"{'='*50}")


# ==================== 命令行入口 ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ILCSampleData 探针数据读取器")
    parser.add_argument("--bin", type=str, default="ILCSampleData_0.bin",
                        help="探针二进制文件路径 (默认: ../probedata/ILCSampleData_0.bin)")
    parser.add_argument("--random", type=int, default=1,
                        help="随机返回 N 个探针 (默认: 1)")
    parser.add_argument("--index", type=int, default=None,
                        help="按索引查询指定探针")
    parser.add_argument("--stats", action="store_true",
                        help="显示数据集统计信息")
    
    args = parser.parse_args()
    
    # 解析文件路径
    bin_path = args.bin
    if not os.path.isabs(bin_path):
        bin_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'probedata', bin_path)
    
    # 加载数据
    reader = ProbeReader(bin_path)
    
    # 显示统计信息
    if args.stats:
        print(f"\n[数据集统计]")
        print(f"  探针数量: {reader.num_probes:,}")
        print(f"  位置范围:")
        print(f"    X: [{reader.pos_min[0]:.2f}, {reader.pos_max[0]:.2f}]")
        print(f"    Y: [{reader.pos_min[1]:.2f}, {reader.pos_max[1]:.2f}]")
        print(f"    Z: [{reader.pos_min[2]:.2f}, {reader.pos_max[2]:.2f}]")
        print(f"  半径范围: [{reader.radii.min():.4f}, {reader.radii.max():.4f}]")
        print(f"  SH 系数范围: [{reader.sh_coeffs.min():.6f}, {reader.sh_coeffs.max():.6f}]")
        print(f"  SH 均值: {reader.sh_coeffs.mean():.6f}, 标准差: {reader.sh_coeffs.std():.6f}")
        print(f"  阴影范围: [{reader.shadows.min():.4f}, {reader.shadows.max():.4f}]")
    
    # 按索引查询
    if args.index is not None:
        reader.print_probe(args.index)
    else:
        # 随机查询
        print(f"\n随机选取 {args.random} 个探针:")
        for i in range(args.random):
            reader.print_probe()
