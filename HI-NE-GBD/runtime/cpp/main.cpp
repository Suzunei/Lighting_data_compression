/**
 * HI-NE-GBD C++ 推理示例
 * 
 * 使用方式:
 *   HINEGBD.exe <model_path.uasset> [x y z]
 * 
 * 示例:
 *   HINEGBD.exe hinegbd_model.uasset                  # 交互模式
 *   HINEGBD.exe hinegbd_model.uasset 100 50 -200     # 单点查询
 */

#include "HINEGBD.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#endif

static void PrintModelInfo(const HINEGBD::FModelConfig& Config)
{
    printf("[HI-NE-GBD C++ Decoder]\n");
    printf("  输出维度: %u (SH coefficients)\n", Config.DataDim);
    printf("  基函数数: L=%u, 系数高斯: M=%u, 基函数高斯: N=%u, 细高斯: F=%u\n",
           Config.NumBases, Config.CoeffRes, Config.BasisRes, Config.FineGaussianRes);
    printf("  PE频率: %u, MLP隐藏: %u, Fine深度: %u\n",
           Config.PENumFreqs, Config.MLPHidden, Config.FineMlpDepth);
    printf("  位置范围: X[%.1f, %.1f], Y[%.1f, %.1f], Z[%.1f, %.1f]\n",
           Config.PosMin[0], Config.PosMax[0],
           Config.PosMin[1], Config.PosMax[1],
           Config.PosMin[2], Config.PosMax[2]);
}

static void PrintSHCoeffs(const std::vector<float>& SH, uint32_t DataDim)
{
    printf("  [SH 系数] (%u 维):\n", DataDim);
    uint32_t NumBands = DataDim / 3;
    for (uint32_t band = 0; band < NumBands; ++band)
    {
        printf("    SH%u: R=%+.6f  G=%+.6f  B=%+.6f\n",
               band, SH[band * 3], SH[band * 3 + 1], SH[band * 3 + 2]);
    }
}

static void Benchmark(const HINEGBD::FHINEGBDModel& Model, uint32_t NumQueries = 10000, uint32_t NumRuns = 10)
{
    printf("\n[Benchmark] 性能测试: %u 点 x %u 次\n", NumQueries, NumRuns);

    // 生成随机归一化坐标
    std::vector<float> Coords(NumQueries * 3);
    for (size_t i = 0; i < Coords.size(); ++i)
    {
        Coords[i] = (float)rand() / (float)RAND_MAX;
    }

    // Warmup
    for (int i = 0; i < 3; ++i)
    {
        Model.DecodeBatchNormalized(Coords.data(), NumQueries);
    }

    // Benchmark
    double TotalMs = 0.0;
    for (uint32_t run = 0; run < NumRuns; ++run)
    {
        auto Start = std::chrono::high_resolution_clock::now();
        Model.DecodeBatchNormalized(Coords.data(), NumQueries);
        auto End = std::chrono::high_resolution_clock::now();
        double Ms = std::chrono::duration<double, std::milli>(End - Start).count();
        TotalMs += Ms;
    }

    double AvgMs = TotalMs / NumRuns;
    double Throughput = NumQueries / (AvgMs / 1000.0);
    double PerPointUs = (AvgMs / NumQueries) * 1000.0;

    printf("  平均耗时: %.2f ms\n", AvgMs);
    printf("  吞吐量: %.0f pts/sec\n", Throughput);
    printf("  单点延迟: %.2f μs\n", PerPointUs);
}

static void InteractiveMode(const HINEGBD::FHINEGBDModel& Model)
{
    printf("\n==================================================\n");
    printf("HI-NE-GBD C++ 实时解压 - 交互模式\n");
    printf("==================================================\n");
    printf("输入格式: x y z （世界坐标，空格分隔）\n");
    printf("特殊命令:\n");
    printf("  'n x y z' - 输入归一化坐标 [0,1]\n");
    printf("  'bench'   - 运行性能基准测试\n");
    printf("  'info'    - 显示模型信息\n");
    printf("  'quit'    - 退出\n");
    printf("==================================================\n");

    char Line[256];
    while (true)
    {
        printf("\n> 输入坐标: ");
        if (!fgets(Line, sizeof(Line), stdin)) break;

        // 去除换行
        size_t Len = strlen(Line);
        while (Len > 0 && (Line[Len-1] == '\n' || Line[Len-1] == '\r'))
            Line[--Len] = '\0';

        if (Len == 0) continue;

        if (strcmp(Line, "quit") == 0 || strcmp(Line, "exit") == 0 || strcmp(Line, "q") == 0)
        {
            printf("退出。\n");
            break;
        }

        if (strcmp(Line, "bench") == 0)
        {
            Benchmark(Model);
            continue;
        }

        if (strcmp(Line, "info") == 0)
        {
            PrintModelInfo(Model.GetConfig());
            continue;
        }

        // 解析坐标
        bool Normalized = false;
        const char* ParsePtr = Line;
        if (Line[0] == 'n' && Line[1] == ' ')
        {
            Normalized = true;
            ParsePtr = Line + 2;
        }

        float x, y, z;
        if (sscanf(ParsePtr, "%f %f %f", &x, &y, &z) != 3)
        {
            printf("  [错误] 请输入3个坐标值，例如: 100 50 -200\n");
            continue;
        }

        float Coords[3] = {x, y, z};

        auto Start = std::chrono::high_resolution_clock::now();
        std::vector<float> SH;
        if (Normalized)
        {
            SH = Model.DecodeNormalized(Coords);
        }
        else
        {
            SH = Model.Decode(Coords);
        }
        auto End = std::chrono::high_resolution_clock::now();
        double Ms = std::chrono::duration<double, std::milli>(End - Start).count();

        printf("  [%s坐标] (%.2f, %.2f, %.2f)\n", Normalized ? "归一化" : "世界", x, y, z);
        printf("  [解压耗时] %.3f ms\n", Ms);
        PrintSHCoeffs(SH, Model.GetConfig().DataDim);
    }
}

int main(int argc, char* argv[])
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    if (argc < 2)
    {
        printf("用法: %s <model.uasset> [x y z]\n", argv[0]);
        printf("  无坐标参数时进入交互模式\n");
        return 1;
    }

    const char* ModelPath = argv[1];

    // 加载模型
    printf("加载模型: %s\n", ModelPath);
    HINEGBD::FHINEGBDModel Model;
    if (!Model.LoadFromFile(ModelPath))
    {
        printf("[错误] 无法加载模型文件: %s\n", ModelPath);
        return 1;
    }

    PrintModelInfo(Model.GetConfig());

    // 如果提供了坐标参数，直接解压
    if (argc >= 5)
    {
        float x = (float)atof(argv[2]);
        float y = (float)atof(argv[3]);
        float z = (float)atof(argv[4]);
        float Coords[3] = {x, y, z};

        printf("\n查询坐标: (%.2f, %.2f, %.2f)\n", x, y, z);

        auto Start = std::chrono::high_resolution_clock::now();
        std::vector<float> SH = Model.Decode(Coords);
        auto End = std::chrono::high_resolution_clock::now();
        double Ms = std::chrono::duration<double, std::milli>(End - Start).count();

        printf("解压耗时: %.3f ms\n", Ms);
        PrintSHCoeffs(SH, Model.GetConfig().DataDim);
    }
    else
    {
        InteractiveMode(Model);
    }

    return 0;
}
