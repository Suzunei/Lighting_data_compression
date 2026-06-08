/**
 * HI-NE-GBD C++ 推理模块
 * 
 * 使用 UCommon::FFileArchive 加载由 export_model.py 导出的模型权重，
 * 实现纯 CPU 前向传播，输出 27 维 SH 系数。
 * 
 * 用法:
 *   FHINEGBDModel model;
 *   model.LoadFromFile("hinegbd_model.uasset");
 *   
 *   float coords[3] = {100.0f, 50.0f, -200.0f};
 *   std::vector<float> sh = model.Decode(coords);  // 27 维
 */

#pragma once

#include <UCommon/Archive.h>
#include <UCommon/Half.h>
#include <vector>
#include <cstdint>
#include <cmath>
#include <string>

namespace HINEGBD
{
    // ======================== 数据结构 ========================

    /// 线性层 (全连接层)
    struct FLinearLayer
    {
        uint32_t InFeatures = 0;
        uint32_t OutFeatures = 0;
        std::vector<float> Weight;  // [OutFeatures * InFeatures], row-major
        std::vector<float> Bias;    // [OutFeatures]

        void Serialize(UCommon::IArchive& Archive, bool bFP16);

        /// 前向: output[OutFeatures] = Weight * input[InFeatures] + Bias
        void Forward(const float* Input, float* Output) const;

        /// 批量前向: 对 BatchSize 个输入分别计算
        void ForwardBatch(const float* Input, float* Output, uint32_t BatchSize) const;
    };

    /// MLP (多层感知机)
    struct FMLP
    {
        std::vector<FLinearLayer> Layers;
        bool UseReLU = true;  // 中间层是否使用 ReLU

        /// 前向传播 (含 ReLU 激活，最后一层无激活)
        void Forward(const float* Input, float* Output, std::vector<float>& Scratch) const;

        /// 批量前向
        void ForwardBatch(const float* Input, float* Output, uint32_t BatchSize) const;

        uint32_t GetInputDim() const { return Layers.empty() ? 0 : Layers[0].InFeatures; }
        uint32_t GetOutputDim() const { return Layers.empty() ? 0 : Layers.back().OutFeatures; }
    };

    /// 3D 高斯参数组
    struct FGaussianParams
    {
        uint32_t NumCenters = 0;  // K
        std::vector<float> Mu;     // [K * 3] 中心坐标
        std::vector<float> LogS;   // [K * 3] log 尺度
        std::vector<float> Q;      // [K * 4] 旋转四元数 (w, x, y, z)
        std::vector<float> Alpha;  // [K] 强度

        void Serialize(UCommon::IArchive& Archive, bool bFP16);
    };

    /// 模型配置
    struct FModelConfig
    {
        uint32_t DataDim = 27;           // 输出维度 (SH 系数)
        uint32_t NumBases = 6;           // L: 基函数数量
        uint32_t CoeffRes = 12;          // M: 系数高斯数量
        uint32_t BasisRes = 8;           // N: 基函数高斯数量
        uint32_t FineGaussianRes = 16;   // F: 细高斯数量
        uint32_t PENumFreqs = 4;         // 位置编码频率数
        uint32_t MLPHidden = 64;         // MLP 隐藏层宽度
        uint32_t FineMlpDepth = 2;       // Fine MLP 深度
        uint32_t NumFineMLPLayers = 0;   // Fine MLP 线性层数量
        uint32_t NumResidualLayers = 0;  // Residual 线性层数量
        uint32_t UseFP16 = 0;              // 是否使用 FP16 存储权重
        uint32_t LatentDim = 32;           // K: per-anchor latent width V_{[F,K]} (paper Eq.5/7)

        float PosMin[3] = {0, 0, 0};    // 世界坐标最小值
        float PosMax[3] = {1, 1, 1};    // 世界坐标最大值

        uint32_t GetPEDim() const { return 3 * (1 + 2 * PENumFreqs); }
        // Paper-aligned fine MLP input: PE(x) ⊕ mixed latent F(x) ∈ R^K
        uint32_t GetFineInputDim() const { return GetPEDim() + LatentDim; }
    };

    // ======================== 模型主类 ========================

    class FHINEGBDModel
    {
    public:
        FHINEGBDModel() = default;
        ~FHINEGBDModel() = default;

        /// 从 .uasset 文件加载模型 (FFileArchive 格式)
        bool LoadFromFile(const char* FilePath);

        /// 从内存加载模型
        bool LoadFromMemory(const uint8_t* Data, uint64_t Size);

        /// 单点解压: 输入世界坐标 [3]，输出 SH 系数 [DataDim]
        std::vector<float> Decode(const float* WorldCoords) const;

        /// 单点解压 (归一化坐标): 输入归一化坐标 [0,1] 范围
        std::vector<float> DecodeNormalized(const float* NormCoords) const;

        /// 批量解压: 输入 N 个世界坐标 [N*3]，输出 [N*DataDim]
        std::vector<float> DecodeBatch(const float* WorldCoords, uint32_t NumPoints) const;

        /// 批量解压 (归一化坐标)
        std::vector<float> DecodeBatchNormalized(const float* NormCoords, uint32_t NumPoints) const;

        /// 获取模型配置
        const FModelConfig& GetConfig() const { return Config; }

        /// 是否加载成功
        bool IsValid() const { return bLoaded; }

    private:
        /// 将世界坐标归一化到 [0, 1]
        void NormalizeCoords(const float* WorldCoords, float* OutNorm, uint32_t NumPoints) const;

        /// 前向传播核心 (输入为归一化坐标 [Q, 3])
        void Forward(const float* NormCoords, float* Output, uint32_t NumPoints) const;

        /// 计算 3D 高斯权重: [Q, K]
        void ComputeGaussianWeights(const float* QueryPts, uint32_t Q,
                                     const FGaussianParams& Params,
                                     float* OutWeights) const;

        /// 位置编码: input [Q, 3] -> output [Q, PEDim]
        void PositionalEncoding(const float* Coords, float* Output, uint32_t Q) const;

        /// 从 Archive 中加载 Serialize
        void Serialize(UCommon::IArchive& Archive);

    private:
        bool bLoaded = false;
        FModelConfig Config;

        // 高斯参数
        FGaussianParams CoeffGaussian;   // 系数高斯
        FGaussianParams BasisGaussian;   // 基函数高斯
        FGaussianParams FineGaussian;    // 细分支高斯

        // 细分支特征
        std::vector<float> FineFeatures;  // [F, DataDim]

        // MBD 张量
        std::vector<float> C;             // [M, L] 系数矩阵
        std::vector<float> B;             // [N, L, DataDim] 基函数张量
        std::vector<float> MBDLogScale;   // [L] 缩放因子

        // MLP
        FMLP FineMLP;                     // Fine Branch MLP
        FMLP ResidualRefiner;             // Residual Refinement MLP
    };

    // ======================== 工具函数 ========================

    /// 矩阵乘法: C[M,N] = A[M,K] * B[K,N]
    void MatMul(const float* A, const float* B, float* C,
                uint32_t M, uint32_t K, uint32_t N);

    /// ReLU 激活 (in-place)
    void ReLU(float* Data, uint32_t Size);

    /// Sigmoid
    inline float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

} // namespace HINEGBD
