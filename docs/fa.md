# FlashAttention Profiling记录
由于算子复杂，新增draw文件夹保留描述计算的示意图，使用excalidraw，使用MMA PTX

excalidraw文件保存在docs/draw下，以下是导出的两张图

### Light Theme
![light](res/fa_light.svg)
### Dark Theme
![dark](res/fa_dark.svg)


## FA kernel 1:
单个kernel直接负责一段Q的计算，Q一次载入smem，K和V序列会循环载入smem，warp内通信较复杂，且需要block level reduce

当前是图方便的写法，自然引出两个优化点
- 当前要求Q可完全装进smem,可优化同一块空间循环加载
- 如图，每个warp负责一块，所以需要block level reduce,可设计为每个warp负责一行S
- $S$ 为 $16 * 8$ 的矩阵，后续和 $V$ 的乘法中需要装入 $16*16$ 的寄存器片段中，形状方便的情况下可走寄存器而非走smem
- 寄存器使用过多，可考虑是否能最大程度复用
