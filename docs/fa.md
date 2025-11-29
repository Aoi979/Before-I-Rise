# FlashAttention Profiling记录
由于算子复杂，新增draw文件夹保留描述计算的示意图，使用excalidraw
![alt text](res/fa.svg)
使用MMA PTX
## FA kernel 1:
单个kernel直接负责一段Q的计算，Q一次载入smem，K和V序列会循环载入smem，warp内通信较复杂


