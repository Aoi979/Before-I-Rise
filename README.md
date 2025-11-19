# **Before I Rise**

> “Before I Rise” 来源于游戏 *Heaven Burns Red* 的同名主题曲。

一个基于 **CUDA** 的算子库，主要是整理和记录自己的学习过程

---
## 构建

使用 **CMake + Ninja** 构建：

```bash
mkdir build
cd build
cmake ..
ninja
```


## Profiling 与优化记录

重要算子的性能分析与优化过程摘要会存放在docs路径下，相关kernel的ncu报告在ncu路径下

硬件平台为RTX 4060 Laptop, 算子会根据ncu报告做出调整并与cuBLAS对比，不保证在其他机器下高效

可使用以下命令打开 `.ncu-rep` 报告查看详细性能数据：

```bash
ncu-ui ncu/xxxx.ncu-rep
```
### SGEMM
- 📘 [优化记录：docs/sgemm.md](docs/sgemm.md)
### HGEMM
- 📘 [优化记录 :docs/hgemm.md](docs/hgemm.md)


