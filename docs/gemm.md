# GEMM Profiling记录

## GEMM kernel 1:
仅仅使用shared memory做分块矩阵乘
数据规模为1024x1024x1024,一个块处理32x32个元素
### Throughput
![alt text](image.png)
可见任务负载已经让SM很繁忙了，但这不能说明问题，我们可以通过Roofline去分析实际算力如何
#### Roofline Analysis

![alt text](image-1.png)

分块形状是32x32，因为计算是按Thread Block隔离的,一个Block内有1024 threads，一个thread的计算量为1024 * 2 FLOP, 整个块的AI为1024 * 1024 * 2 / 2 * 32 * 1024 * 4 + 1024 * 4 = 7.87 
ncu给出的L1 AI是7.77, 差不多，从图上也能看出，算力被完全浪费了，这指出了优化方向，应该让线程的计算任务更重一些，GEMM本身是三级算子，利用数据的局部性可以让运算更密集

### Compute Workload Analysis

![alt text](image-2.png)
可以看到计算单元非常空闲，只有LSU(Load/Store Uint)繁忙，符合Roofline部分分析的

### Memory Workload Analysis
![alt text](image-3.png)
L2的命中率很高，这一定程度上会让DRAM的AI变大，此外smem无bank conflict，other那一栏不归程序员管，目前的smem访存模式是高效的
L1难以分析，因为硬件上L1的策略是需要权衡的，而不是L2那样较为固定

### 总结
没用充分发挥硬件计算能力，应该利用GEMM的数据规律加强计算负载

## GEMM kernel 2:
