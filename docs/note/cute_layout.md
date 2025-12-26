# CuTe Layout笔记
> 并不严谨
## *Layout*
一个布局$L$是由一对具有相同维度的正整数元组$S$和$D$组成的，其中$S$被称为*shape*,$D$为*stride*，写作$L = S:D$

扁平化的布局意味着shape和stride中没有内部括号。例如, $L=(1,2,3):(1,1,2)$是一个扁平化布局，而$L=((1,2),3):((1,1),2)$不是，当然，两者含义完全相等

*Layout*本质上是一个函数，且满足双射，输入$x$可以是坐标$(x_0,x_1)$也可以是等价的线性坐标$x=x_0+x_1{S[0]}$

### 布局的大小,长度,模式
$L=S:D=(M_0,M_1,...,M_\alpha):(d_0,d_1,...,d_\alpha)$是一个布局，其中$\alpha$是一个满足$\alpha \ge 0$的整数，那么
- $L$的*size*为$M=M_0⋅M_1⋅...⋅M_\alpha$,即$\prod_{i=0}^{\alpha} M_i$
- $L$的*length*为$\alpha+1$
- $L$的一个*mode*为 $(M_0,M_1,...,M_\alpha):(d_0,d_1,...,d_\alpha)$ 中的一对 $(M_k):(d_k),0 \le k \le \alpha$

为方便书写，$(x_0,x_1,...,x_\alpha)$记为
$$
(x_i)_{i=0}^{\alpha}
$$
### 连接
给定两个布局$L=S:D， L'=S':D'$,则
$$(L,L')=(S,S'):(D,D')$$

### 同构
设 $S=(M_i)_{i=0}^{\alpha}$ 和 $D=(d_i)_{i=0}^{\alpha}$ 组成 $L = S : D$，$M=\prod_{i=0}^{\alpha} M_i$为$L$的大小，对于任意$x\in[0,M)$有
$$
x\mapsto(x \bmod M_0, \lfloor \frac{x}{M_0} \rfloor \bmod M_1, ..., \lfloor \frac{x}{\prod_{i=0}^{\alpha-1} M_i} \rfloor \bmod M_\alpha)
$$

### 布局函数

对于一个布局$L$，其布局函数$f_L:[0,M)\rightarrow\mathbb{N}$

$M$为布局$L$的大小，比如内存连续矩阵$A:(5,6)$，其大小为$30$,输入可以是坐标$(0...4,0...5)$，换到线性坐标上即$[0,30)$

给定输入$x$,$f_L$的输出为
$$
x\mapsto(x \bmod M_0, \lfloor \frac{x}{M_0} \rfloor \bmod M_1, ..., \lfloor \frac{x}{\prod_{i=0}^{\alpha-1} M_i} \rfloor \bmod M_\alpha) = (x_i)_{i=0}^{\alpha}
$$
$$
f_L(x) = f_L((x_i)_{i=0}^{\alpha}) = \vec{x} \cdot \vec{d}, \quad \text{其中 } \vec{x} = (x_i)_{i=0}^{\alpha}, \vec{d} = (d_i)_{i=0}^{\alpha}
$$
可以看出
$$
f_L(x) =f_L((x_i)_{i=0}^{\alpha}) = \sum_{i=0}^{\alpha}f_L(x_i)
$$


## 合并
合并不会改变布局
### 合并规则
考虑一个只有2个*mode*的布局$A=(N_0,N_1):(d_0,d_1)$，当$N_0$或$N_1$为$1$时，这一个*mode*可以直接去除,如

$$N_0=1 \rightarrow M_0 \in [0,1) \rightarrow 0$$

$$
A=(N_1):(d_!)
$$

另一种情况是紧密的排布，即$d_1 = N_0d_0$，对于$A=(N_0,N_1):(d_0,N_0d_0)$,我们有$x\mapsto(x_0,x_1)$,
$$
f_L(x) = x_0d_0 + x_1N_0d_0,其中x_0 = x  \bmod N_0, x_1 = \lfloor \frac{x}{N_0} \rfloor
$$
$$
f_L(x) = x_0d_0 + \lfloor \frac{x}{N_0} \rfloor N_0d_0 = x_0d_0 + [x-(x \bmod N_0)]d_0 = x_0d_0 + (x-x_0)d_0 = xd_0
$$
所以可以合并为$A=(N_0N_1):(d_0)$，其他情况不可合并

具有两个以上*mode*的布局，我们可以递归地应用上述情况，每次尝试合并两个相邻的积分模式，直到不能再合并为止。这保证了布局函数保持不变。

设$L=(N_i)_{i=0}^{\alpha}:(d_i)_{i=0}^{\alpha}$,其输入$x \mapsto (x_i)_{i=0}^{\alpha}$,
$$
f_L(x) =f_L((x_i)_{i=0}^{\alpha}) = \sum_{i=0}^{\alpha}f_L(x_i)
$$
考虑两个连续的*mode*可合并，即


$$
(x_0,...,x_i,x_{i+1},...,x_\alpha) \rightarrow (x_0,...,x',...,x_\alpha)，x'=x_i+x_{i+1}N_i
$$

$$
(d_0,...,d_i,d_{i+1},...,d_\alpha) \rightarrow (d_0,...,d_i,...,d_\alpha)
$$

$$
f_L(x_i) + f_L(x_{i+1}) = x_id_i + x_{i+1}d_{i+1} = x_id_i+x_{i+1}N_id_i=f_L(x') =  x'd_i = (x_i+x_{i+1}N_i)d_0
$$
$$
f_L(x) = \sum_{i=0}^{\alpha}f_L(x_i) = \sum_{i=0}^{i'-1}f_L(x_i) + f_L(x') + \sum_{i=i'+2}^{\alpha}f_L(x_i)
$$
这个写法不严谨，因为*mode*改变了，$f_L$也发生了变化，应该分解$f_L$到各个*mode*得到$f_{L_i}$，总之就这个意思
## 补集
