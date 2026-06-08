/**
 * HI-NE-GBD C++ 推理实现
 * 
 * 完整实现模型前向传播:
 *   1. 粗分支 (MBD): 高斯权重 -> 系数插值 -> 基函数插值 -> 加权求和
 *   2. 细分支: 位置编码 + 高斯权重 -> MLP -> 高斯特征插值
 *   3. 残差细化: blended + coords -> MLP -> residual
 */

#include "HINEGBD.h"
#include <cstring>
#include <algorithm>
#include <cassert>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace HINEGBD
{

// ======================== 工具函数实现 ========================

/// 读取 tensor，根据 UseFP16 标志决定读 FP16 还是 FP32，最终存储为 FP32
static void ReadTensorAuto(UCommon::IArchive& Archive, std::vector<float>& Vec, bool bFP16)
{
    uint64_t NumElements = 0;
    Archive.ByteSerialize(NumElements);
    Vec.resize(NumElements);
    if (NumElements == 0) return;

    if (bFP16)
    {
        // 读取 FP16 数据，转换为 FP32
        std::vector<UCommon::FHalf> HalfBuf(NumElements);
        Archive.Serialize(HalfBuf.data(), NumElements * sizeof(UCommon::FHalf));
        for (uint64_t i = 0; i < NumElements; ++i)
        {
            Vec[i] = static_cast<float>(HalfBuf[i]);
        }
    }
    else
    {
        // 直接读取 FP32
        Archive.Serialize(Vec.data(), NumElements * sizeof(float));
    }
}

// ======================== 工具函数实现 ========================

void MatMul(const float* A, const float* B, float* C, uint32_t M, uint32_t K, uint32_t N)
{
    // C[M,N] = A[M,K] * B[K,N]
    for (uint32_t i = 0; i < M; ++i)
    {
        for (uint32_t j = 0; j < N; ++j)
        {
            float Sum = 0.0f;
            for (uint32_t k = 0; k < K; ++k)
            {
                Sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = Sum;
        }
    }
}

void ReLU(float* Data, uint32_t Size)
{
    for (uint32_t i = 0; i < Size; ++i)
    {
        if (Data[i] < 0.0f) Data[i] = 0.0f;
    }
}

// ======================== FLinearLayer ========================

void FLinearLayer::Serialize(UCommon::IArchive& Archive, bool bFP16)
{
    // Weight: [OutFeatures, InFeatures]
    ReadTensorAuto(Archive, Weight, bFP16);
    OutFeatures = 0;
    InFeatures = 0;

    // Bias: [OutFeatures]
    ReadTensorAuto(Archive, Bias, bFP16);

    // 推导维度
    OutFeatures = (uint32_t)Bias.size();
    if (OutFeatures > 0 && Weight.size() > 0)
    {
        InFeatures = (uint32_t)(Weight.size() / OutFeatures);
    }
}

void FLinearLayer::Forward(const float* Input, float* Output) const
{
    // Output[o] = sum_i(Weight[o * InFeatures + i] * Input[i]) + Bias[o]
    for (uint32_t o = 0; o < OutFeatures; ++o)
    {
        float Sum = Bias[o];
        const float* Row = Weight.data() + o * InFeatures;
        for (uint32_t i = 0; i < InFeatures; ++i)
        {
            Sum += Row[i] * Input[i];
        }
        Output[o] = Sum;
    }
}

void FLinearLayer::ForwardBatch(const float* Input, float* Output, uint32_t BatchSize) const
{
    for (uint32_t b = 0; b < BatchSize; ++b)
    {
        Forward(Input + b * InFeatures, Output + b * OutFeatures);
    }
}

// ======================== FMLP ========================

void FMLP::Forward(const float* Input, float* Output, std::vector<float>& Scratch) const
{
    if (Layers.empty()) return;

    // 找出最大中间维度来分配 scratch
    uint32_t MaxDim = 0;
    for (auto& L : Layers)
    {
        MaxDim = std::max(MaxDim, std::max(L.InFeatures, L.OutFeatures));
    }
    Scratch.resize(MaxDim * 2);

    float* Buf0 = Scratch.data();
    float* Buf1 = Scratch.data() + MaxDim;

    // 第一层
    Layers[0].Forward(Input, Buf0);
    if (UseReLU && Layers.size() > 1) ReLU(Buf0, Layers[0].OutFeatures);

    // 中间层
    for (size_t i = 1; i < Layers.size() - 1; ++i)
    {
        Layers[i].Forward(Buf0, Buf1);
        if (UseReLU) ReLU(Buf1, Layers[i].OutFeatures);
        std::swap(Buf0, Buf1);
    }

    // 最后一层 (无激活)
    if (Layers.size() > 1)
    {
        Layers.back().Forward(Buf0, Output);
    }
    else
    {
        std::memcpy(Output, Buf0, Layers[0].OutFeatures * sizeof(float));
    }
}

void FMLP::ForwardBatch(const float* Input, float* Output, uint32_t BatchSize) const
{
    std::vector<float> Scratch;
    uint32_t InDim = GetInputDim();
    uint32_t OutDim = GetOutputDim();
    for (uint32_t b = 0; b < BatchSize; ++b)
    {
        Forward(Input + b * InDim, Output + b * OutDim, Scratch);
    }
}

// ======================== FGaussianParams ========================

void FGaussianParams::Serialize(UCommon::IArchive& Archive, bool bFP16)
{
    ReadTensorAuto(Archive, Mu, bFP16);
    ReadTensorAuto(Archive, LogS, bFP16);
    ReadTensorAuto(Archive, Q, bFP16);
    ReadTensorAuto(Archive, Alpha, bFP16);

    NumCenters = (uint32_t)Alpha.size();
}

// ======================== FHINEGBDModel 序列化 ========================

void FHINEGBDModel::Serialize(UCommon::IArchive& Archive)
{
    // Config metadata
    Archive.ByteSerialize(Config.DataDim);
    Archive.ByteSerialize(Config.NumBases);
    Archive.ByteSerialize(Config.CoeffRes);
    Archive.ByteSerialize(Config.BasisRes);
    Archive.ByteSerialize(Config.FineGaussianRes);
    Archive.ByteSerialize(Config.PENumFreqs);
    Archive.ByteSerialize(Config.MLPHidden);
    Archive.ByteSerialize(Config.FineMlpDepth);
    Archive.ByteSerialize(Config.NumFineMLPLayers);
    Archive.ByteSerialize(Config.NumResidualLayers);
    Archive.ByteSerialize(Config.UseFP16);
    Archive.ByteSerialize(Config.LatentDim);

    // Position normalization
    Archive.Serialize(Config.PosMin, 3 * sizeof(float));
    Archive.Serialize(Config.PosMax, 3 * sizeof(float));

    const bool bFP16 = (Config.UseFP16 != 0);

    // Gaussian parameters
    CoeffGaussian.Serialize(Archive, bFP16);
    BasisGaussian.Serialize(Archive, bFP16);
    FineGaussian.Serialize(Archive, bFP16);

    // Fine features
    ReadTensorAuto(Archive, FineFeatures, bFP16);

    // MBD tensors
    ReadTensorAuto(Archive, C, bFP16);
    ReadTensorAuto(Archive, B, bFP16);
    ReadTensorAuto(Archive, MBDLogScale, bFP16);

    // Fine MLP layers
    FineMLP.Layers.resize(Config.NumFineMLPLayers);
    for (auto& Layer : FineMLP.Layers)
    {
        Layer.Serialize(Archive, bFP16);
    }

    // Residual Refiner layers
    ResidualRefiner.Layers.resize(Config.NumResidualLayers);
    for (auto& Layer : ResidualRefiner.Layers)
    {
        Layer.Serialize(Archive, bFP16);
    }
}

bool FHINEGBDModel::LoadFromFile(const char* FilePath)
{
    UCommon::FFileArchive Archive(UCommon::IArchive::EState::Loading, FilePath);
    if (!Archive.IsValid())
    {
        return false;
    }

    Serialize(Archive);
    bLoaded = true;
    return true;
}

bool FHINEGBDModel::LoadFromMemory(const uint8_t* Data, uint64_t Size)
{
    UCommon::TSpan<const uint8_t> Span(Data, Size);
    UCommon::FMemoryArchive Archive(Span);

    // 跳过 FFileArchive header (16 bytes) — 内存中需要手动跳过
    // 实际上如果用 FMemoryArchive，数据应该不含 header
    // 这里假设传入的是裸数据（不含 Ubpa header）
    Serialize(Archive);
    bLoaded = true;
    return true;
}

// ======================== 前向传播实现 ========================

void FHINEGBDModel::NormalizeCoords(const float* WorldCoords, float* OutNorm, uint32_t NumPoints) const
{
    for (uint32_t i = 0; i < NumPoints; ++i)
    {
        for (uint32_t d = 0; d < 3; ++d)
        {
            float Range = Config.PosMax[d] - Config.PosMin[d];
            if (Range < 1e-6f) Range = 1.0f;
            OutNorm[i * 3 + d] = (WorldCoords[i * 3 + d] - Config.PosMin[d]) / Range;
        }
    }
}

std::vector<float> FHINEGBDModel::Decode(const float* WorldCoords) const
{
    float NormCoords[3];
    NormalizeCoords(WorldCoords, NormCoords, 1);
    return DecodeNormalized(NormCoords);
}

std::vector<float> FHINEGBDModel::DecodeNormalized(const float* NormCoords) const
{
    std::vector<float> Output(Config.DataDim);
    Forward(NormCoords, Output.data(), 1);
    return Output;
}

std::vector<float> FHINEGBDModel::DecodeBatch(const float* WorldCoords, uint32_t NumPoints) const
{
    std::vector<float> NormCoords(NumPoints * 3);
    NormalizeCoords(WorldCoords, NormCoords.data(), NumPoints);
    return DecodeBatchNormalized(NormCoords.data(), NumPoints);
}

std::vector<float> FHINEGBDModel::DecodeBatchNormalized(const float* NormCoords, uint32_t NumPoints) const
{
    std::vector<float> Output(NumPoints * Config.DataDim);
    Forward(NormCoords, Output.data(), NumPoints);
    return Output;
}

void FHINEGBDModel::ComputeGaussianWeights(
    const float* QueryPts, uint32_t Q,
    const FGaussianParams& Params,
    float* OutWeights) const
{
    const uint32_t K = Params.NumCenters;

    // 对每个查询点和每个高斯中心计算马氏距离
    for (uint32_t q = 0; q < Q; ++q)
    {
        const float* P = QueryPts + q * 3;
        float WeightSum = 0.0f;

        for (uint32_t k = 0; k < K; ++k)
        {
            const float* Mu = Params.Mu.data() + k * 3;
            const float* LogS = Params.LogS.data() + k * 3;
            const float* QVal = Params.Q.data() + k * 4;

            // 四元数 -> 旋转矩阵
            float qw = QVal[0], qx = QVal[1], qy = QVal[2], qz = QVal[3];
            float norm = std::sqrt(qw*qw + qx*qx + qy*qy + qz*qz + 1e-8f);
            qw /= norm; qx /= norm; qy /= norm; qz /= norm;

            // 旋转矩阵 R (row-major 3x3)
            float R[9] = {
                1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qw*qz, 2*qx*qz + 2*qw*qy,
                2*qx*qy + 2*qw*qz, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qw*qx,
                2*qx*qz - 2*qw*qy, 2*qy*qz + 2*qw*qx, 1 - 2*qx*qx - 2*qy*qy
            };

            // 尺度: s = exp(log_s), s_inv_sq = 1 / s^2
            float s0 = std::exp(LogS[0]), s1 = std::exp(LogS[1]), s2 = std::exp(LogS[2]);
            float sInvSq0 = 1.0f / (s0*s0 + 1e-8f);
            float sInvSq1 = 1.0f / (s1*s1 + 1e-8f);
            float sInvSq2 = 1.0f / (s2*s2 + 1e-8f);

            // diff = P - Mu
            float dx = P[0] - Mu[0];
            float dy = P[1] - Mu[1];
            float dz = P[2] - Mu[2];

            // rotated_diff = R^T * diff (将diff旋转到高斯的局部坐标系)
            float rx = R[0]*dx + R[3]*dy + R[6]*dz;  // R^T 第一行
            float ry = R[1]*dx + R[4]*dy + R[7]*dz;
            float rz = R[2]*dx + R[5]*dy + R[8]*dz;

            // Mahalanobis distance: diff^T * Precision * diff
            // Precision = R * diag(s_inv_sq) * R^T
            // 等价于 rotated_diff^T * diag(s_inv_sq) * rotated_diff
            float MahalanobisSq = rx*rx*sInvSq0 + ry*ry*sInvSq1 + rz*rz*sInvSq2;

            float GaussVal = std::exp(-0.5f * MahalanobisSq);

            // 乘以强度 sigmoid(alpha)
            float Intensity = Sigmoid(Params.Alpha[k]);
            GaussVal *= Intensity;

            OutWeights[q * K + k] = GaussVal;
            WeightSum += GaussVal;
        }

        // 归一化
        if (WeightSum > 1e-8f)
        {
            float InvSum = 1.0f / WeightSum;
            for (uint32_t k = 0; k < K; ++k)
            {
                OutWeights[q * K + k] *= InvSum;
            }
        }
    }
}

void FHINEGBDModel::PositionalEncoding(const float* Coords, float* Output, uint32_t Q) const
{
    const uint32_t PEDim = Config.GetPEDim();
    const uint32_t NumFreqs = Config.PENumFreqs;

    for (uint32_t q = 0; q < Q; ++q)
    {
        const float* P = Coords + q * 3;
        float* Out = Output + q * PEDim;
        uint32_t Idx = 0;

        // 原始坐标
        Out[Idx++] = P[0];
        Out[Idx++] = P[1];
        Out[Idx++] = P[2];

        // sin/cos 编码 - 每个频率先输出所有维度的 sin，再输出所有维度的 cos
        for (uint32_t f = 0; f < NumFreqs; ++f)
        {
            float Freq = std::pow(2.0f, (float)f);  // 2^f
            for (uint32_t d = 0; d < 3; ++d)
            {
                float Val = Freq * (float)M_PI * P[d];
                Out[Idx++] = std::sin(Val);
            }
            for (uint32_t d = 0; d < 3; ++d)
            {
                float Val = Freq * (float)M_PI * P[d];
                Out[Idx++] = std::cos(Val);
            }
        }
    }
}

void FHINEGBDModel::Forward(const float* NormCoords, float* Output, uint32_t Q) const
{
    const uint32_t L = Config.NumBases;
    const uint32_t M = Config.CoeffRes;
    const uint32_t N = Config.BasisRes;
    const uint32_t F = Config.FineGaussianRes;
    const uint32_t D = Config.DataDim;
    const uint32_t PEDim = Config.GetPEDim();

    // ==================== 粗分支 (Coarse Branch: MBD) ====================

    // 计算系数高斯权重: phi_weights [Q, M]
    std::vector<float> PhiWeights(Q * M);
    ComputeGaussianWeights(NormCoords, Q, CoeffGaussian, PhiWeights.data());

    // 计算基函数高斯权重: psi_weights [Q, N]
    std::vector<float> PsiWeights(Q * N);
    ComputeGaussianWeights(NormCoords, Q, BasisGaussian, PsiWeights.data());

    // moving_coeff = phi_weights [Q, M] * C [M, L] -> [Q, L]
    std::vector<float> MovingCoeff(Q * L);
    MatMul(PhiWeights.data(), C.data(), MovingCoeff.data(), Q, M, L);

    // basis_interp = psi_weights [Q, N] * B_flat [N, L*D] -> [Q, L*D]
    std::vector<float> BasisInterp(Q * L * D);
    MatMul(PsiWeights.data(), B.data(), BasisInterp.data(), Q, N, L * D);

    // mbd_scale = exp(mbd_log_scale) [L]
    std::vector<float> MBDScale(L);
    for (uint32_t l = 0; l < L; ++l)
    {
        MBDScale[l] = std::exp(MBDLogScale[l]);
    }

    // coarse_output [Q, D] = sum_l(scaled_coeff[q,l] * moving_basis[q,l,d])
    std::vector<float> CoarseOutput(Q * D, 0.0f);
    for (uint32_t q = 0; q < Q; ++q)
    {
        for (uint32_t l = 0; l < L; ++l)
        {
            float ScaledCoeff = MovingCoeff[q * L + l] * MBDScale[l];
            for (uint32_t d = 0; d < D; ++d)
            {
                CoarseOutput[q * D + d] += ScaledCoeff * BasisInterp[q * (L * D) + l * D + d];
            }
        }
    }

    // ==================== 细分支 (Fine Branch, paper-aligned Eq.5/7) ====================

    const uint32_t K = Config.LatentDim;

    // 计算细高斯权重 φ_fine: [Q, F]
    std::vector<float> FineWeights(Q * F);
    ComputeGaussianWeights(NormCoords, Q, FineGaussian, FineWeights.data());

    // 混合 per-anchor latent: F(x) = φ_fine(x) · V_{[F,K]} -> [Q, K]   (Eq.5)
    std::vector<float> MixedLatent(Q * K);
    MatMul(FineWeights.data(), FineFeatures.data(), MixedLatent.data(), Q, F, K);

    // 位置编码: [Q, PEDim]
    std::vector<float> PEOutput(Q * PEDim);
    PositionalEncoding(NormCoords, PEOutput.data(), Q);

    // Fine MLP 输入: concat(PE [Q, PEDim], MixedLatent [Q, K]) -> [Q, PEDim + K]
    const uint32_t FineInputDim = PEDim + K;
    std::vector<float> FineInput(Q * FineInputDim);
    for (uint32_t q = 0; q < Q; ++q)
    {
        std::memcpy(FineInput.data() + q * FineInputDim,
                    PEOutput.data() + q * PEDim, PEDim * sizeof(float));
        std::memcpy(FineInput.data() + q * FineInputDim + PEDim,
                    MixedLatent.data() + q * K, K * sizeof(float));
    }

    // Fine MLP forward: [Q, FineInputDim] -> [Q, D]   (Eq.7)
    std::vector<float> FineOutput(Q * D);
    FineMLP.ForwardBatch(FineInput.data(), FineOutput.data(), Q);

    // ==================== 混合 + 残差 ====================

    // blended = coarse_output + fine_output
    std::vector<float> Blended(Q * D);
    for (uint32_t i = 0; i < Q * D; ++i)
    {
        Blended[i] = CoarseOutput[i] + FineOutput[i];
    }

    // residual refiner input: concat(blended [Q, D], coords [Q, 3]) -> [Q, D+3]
    uint32_t RefineInputDim = D + 3;
    std::vector<float> RefineInput(Q * RefineInputDim);
    for (uint32_t q = 0; q < Q; ++q)
    {
        std::memcpy(RefineInput.data() + q * RefineInputDim,
                    Blended.data() + q * D, D * sizeof(float));
        std::memcpy(RefineInput.data() + q * RefineInputDim + D,
                    NormCoords + q * 3, 3 * sizeof(float));
    }

    // Residual MLP forward: [Q, D+3] -> [Q, D]
    std::vector<float> Residual(Q * D);
    ResidualRefiner.ForwardBatch(RefineInput.data(), Residual.data(), Q);

    // reconstruction = blended + 0.1 * residual
    for (uint32_t i = 0; i < Q * D; ++i)
    {
        Output[i] = Blended[i] + 0.1f * Residual[i];
    }
}

} // namespace HINEGBD
