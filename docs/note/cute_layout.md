# CuTe Layout笔记
> 并不严谨

cutlass3引入cute,cute的layout抽象很强大，但由于资料介绍不到位导致这部分不是那么好懂，为了理解其本质，这里记录一些相关内容
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
f_L(x) =f_L((x_i)_{i=0}^{\alpha}) = \sum_{i=0}^{\alpha}f_{L_i}(x_i)
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
### 排序布局
设$L=(N_i)_{i=0}^{\alpha}:(d_i)_{i=0}^{\alpha}$, 如果$d_0 \le d_1 \le ...\le d_\alpha$且如果$d_i=d_j$,那么$N_i \le N_j$

排序后的布局改变了布局的意义，因为*mode*的顺序被交换了,但其点积的本质导致函数的签名不会改变

### 可补
设$A=(N_i)_{i=0}^{\alpha}:(d_i)_{i=0}^{\alpha}$, $M$为一个正整数， 如果$A$未排序那么将$A$替换为其排序版本$A'$,若满足以下条件则称$\{A,M\}$可补

- $d_i \bmod N_{i-1}d_{i-1} = 0$
- $M \bmod N_\alpha d_\alpha = 0$


### 补集
设$A=(N_i)_{i=0}^{\alpha}:(d_i)_{i=0}^{\alpha}$, $M$为一个正整数， 如果$\{A,M\}$是可补的，则其补集定义为
$$
complement(A,M) = (d_0,\frac{d_1}{N_0d_0},...,\frac{d_\alpha}{N_{\alpha-1}d_{\alpha-1}},\frac{M}{N_\alpha d_\alpha}):(1,N_0d_0,...,N_\alpha d_\alpha)
$$
不难看出其是排序好的，严格递增的,当$x+1$,可能会在新的一个维度上进一，而$N_{i+1}d_{i+1} >(\frac{d_{i+1}}{N_id_i}-1)N_id_i$,即$N_{i+1} > 1-\frac{N_id_i}{d_{i+1}}$ , 由于$d_i \bmod N_{i-1}d_{i-1} = 0$，故$\frac{N_id_i}{d_{i+1}} \le 1$,不等式成立

$complement(A,M)$的大小为$\frac{M}{size(A)}=\frac{M}{\prod_{i=0}^{\alpha} N_i}$

当我们将$complement(A,M)$与$A$拼(*连接并按某种规则交换mode*)在一起会得到
$$
(d_0,N_0,\frac{d_1}{N_0d_0},N_1,...,\frac{d_\alpha}{N_{\alpha-1}d_{\alpha-1}},N_\alpha,\frac{M}{N_\alpha d_\alpha}):(1,d_0,N_0d_0,d_1,...,N_{\alpha-1} d_{\alpha-1},d_\alpha,N_\alpha d_\alpha)
$$
可合并为
$$
(M)：(1)
$$
虽然不一定按这种方便的情况去排布但我们知道*mode*的交换不会影响$f_L：[0,M) \rightarrow [0,M)的性质$

## 组合
因为是函数，自然可以$g(f(x))$这样使用，记$f*g(x)$为$f与g的组合$，有
$$
y=g(f(x)) = f*g(x)
$$
显然$f(x)的值域需要是g(x)定义域的子集$，且$f*g(x)$的定义域和$f(x)$相同，考虑一个简单的情况$x \in [0,M), f(x)=x, g(f(x)) = g(x) = g((x_i)_{i=0}^{\alpha}) = \sum_{i=0}^{\alpha}g(x_i)$，其过程为
$$
x \mapsto (x_0,x_1,...,x_\alpha)
$$
$$
y = f((x_i)_{i=0}^{\alpha}) = x_0d_0 + x_1d_1 + ... + x_\alpha d_\alpha
$$
$$
y \mapsto (x_0',x_1',...,x_{\alpha'}') 
$$
$$
z = g((x_i')_{i=0}^{\alpha'}) = x_0'd_0' + x_1'd_1' + ... + x_{\alpha'}' d_{\alpha'}'
$$
若我们能够找出一个$L$能直接描述这种映射关系，则是可组合的，在上面的例子中
$$
(x_0',x_1',...,x_{\alpha'}') = (x_0,x_1,...,x_\alpha)
$$
$$
g((x_i')_{i=0}^{\alpha'}) = x_0d_0' + x_1d_1' + ... + x_{\alpha} d_{\alpha}'
$$
$$
f*g(x) = g(x)
$$
同理可以有
$$
(x_0',x_1',...,x_{\alpha'}') = a(x_0,x_1,...,x_\alpha)
$$
$$
g((x_i')_{i=0}^{\alpha'}) = ax_0d_0' + ax_1d_1' + ... + ax_{\alpha} d_{\alpha}'
$$
$$
f*g(x) = ag(x)
$$
### 左可除性
设$M,d >0$,为正整数，设$M=M_0M_1...M_\alpha$为一种可能的分解，我们说$M$对于$d$是左可除的，如果存在$0 \le i\le \alpha$满足


- $d \bmod M_0M_1...M_{i-1} == 0$  
- 上一个条件下，$c=\frac{d}{M_0M_1...M_{i-1}}$, 如果$i<\alpha$,还需要满足$1 \le c \le M_i$
- 上一个条件下，如果$i<\alpha$ ,还需要$M_i \bmod c==0$



若满足前两个条件而不满足第三条，则称为弱左可除


对于一个$M=M_0M_1...M_\alpha$,可表示为$M=M_0M_1...M_{i-1}c \frac{M_i}{c} M_{i+1}...M_\alpha$

### 约束
上面提到$f(x)$的值域需要是$g(x)$定义域的子集,这是一种约束，考虑简单情况，一般来说可以这样定义这个约束

设$S=(M_0,M_1,...,M_\alpha)$，$M=M_0M_1...M_\alpha$，$B=(N):(r)$,说$\{S,B\}$是可组合的，如果满足 

- $M$能被$r$左除，记$M=rM'$,其中$r=M_0M_1...M_{i-1}c,M'=\frac{M_i}{c} M_{i+1}...M_\alpha$
- $M'$可被$N$弱左除

如此约束的原因是
$$
x \mapsto (x_0,x_1,...,x_\alpha)
$$
$$
f*g(x) = x_0d_0'' + x_1d_1'' + ... + x_{\alpha} d_{\alpha}''
$$
我们渴望找到一种$D$直接能够提供等价变换，但我们知道
$$
x \mapsto (x_0,x_1,...,x_\alpha)
$$
$$
y = f((x_i)_{i=0}^{\alpha}) = x_0d_0 + x_1d_1 + ... + x_\alpha d_\alpha
$$
$$
y \mapsto (x_0',x_1',...,x_{\alpha'}') 
$$
$$
z = g((x_i')_{i=0}^{\alpha'}) = x_0'd_0' + x_1'd_1' + ... + x_{\alpha'}' d_{\alpha'}'
$$
显然我们要寻找的$(d_i'')_{i=0}^{\alpha}$来源于$(d_i')_{i=0}^{\alpha'}$，换句话说$(d_i'')_{i=0}^{\alpha}$是$(d_i')_{i=0}^{\alpha'}$的*切片*，同理$(x_i)_{i=0}^{\alpha}$也是$(x_i')_{i=0}^{\alpha'}$的*切片*,实际上$M_0M_1...M_{i-1}$就是一个在$i$位置上为$1$的向量坐标：$(0,...,1,...,0)$，要求左除本质是找到*切片*开始的起始坐标,弱左除则是约束*切片*开始位置加上*切片*长度不会超过总长

### 组合
#### 特殊情况
设 $S=(M_i)_{i=0}^{\alpha}$ 和 $D=(d_i)_{i=0}^{\alpha}$ 组成 $A = S : D$,$B=(N):(r)$,且$\{S,B\}$是满足上述约束的，$A*B$的定义分两种情况

假设$0\le i \le \alpha$,有
$$
r=M_0M_1...M_{i-1}c
$$
$$
M'=\frac{M_i}{c} M_{i+1}...M_\alpha
$$
若$N \le \frac{M_i}{c}$，则$A*B=(N):(cd_i)$

否则，存在一个$j\in[i+1,\alpha]$,使得$N=\frac{M_i}{c} M_{i+1}...M_{j-1}c'$,其中$i \le c' < M_j$如果$j \ne \alpha$
$$
A*B=
\begin{cases}
(\frac{M_i}{c}, M_{i+1},...,M_{j-1},c'):(cd_i,d_{i+1},...,d_{j-1},d_j) & c' > 1 \\
(\frac{M_i}{c}, M_{i+1},...,M_{j-1}):(cd_i,d_{i+1},...,d_{j-1}),  & c' = 1
\end{cases}
$$

#### 区间

设布局 $B$ 的映射函数为 $f_B : [0,N) \to \mathbb{N}$。
记 $I = [\min f_B([1,N)),\ \max f_B([1,N))]$ 为 $f_B$ 在区间 $[1,N)$ 上的值域的最小连续整数区间。
再设 $M' = M_0 M_1 \cdots M_{\alpha-1}$，则$S$ 允许访问的位置范围为 $[1,M')$。于是，$\{S,B\}$ 的 区间 为 $J = I \cap [1,M')$。
若 $\alpha = 0$，则 $M' = 1$，于是 $J = \varnothing$。

#### 一般情况

设形状元组  
$$
S = (M_0, M_1, \ldots, M_\alpha)
$$

以及布局  
$$
B = (N_0, N_1, \ldots, N_\beta) : (r_0, r_1, \ldots, r_\beta)
$$

并记  
$$
B_k = (N_k):(r_k), \quad 0 \le k \le \beta
$$

我们称二元组 $\{S,B\}$ 是可组合的，若满足：

1. 对所有 $0 \le k \le \beta$，二元组 $\{S,B_k\}$ 在上述约束下都是可组合的

2. 对所有二元组 $\{S,B_k\}$（$0 \le k \le \beta$）的区间两两互不相交。

---

在这种情况下，如果  
$$
\mathbf{D} = (d_0, d_1, \ldots, d_\alpha)
$$
是一个步长元组，并且  
$$
A = S : \mathbf{D}
$$
那么我们将组合 $A * B$ 定义为连接得到的布局：

$$
A * B := (A * B_0,\ A * B_1,\ \ldots,\ A * B_\beta)
$$

其中每个 $A \circ B_k$ 的定义如特殊情况相同


我们知道,
$$
f_A(f_B(x)) = f_A(f_{B_0}(x_0)+f_{B_1}(x_1)+...+f_{B_{\beta}}(x_\beta))
$$
根据定义
$$
f_A*f_B(x) = f_A(f_{B_0}(x_0)) +  f_A(f_{B_1}(x_1)) +...+ f_A(f_{B_\beta}(x_\beta)) 
$$
一般情况下二者是不等的，但是要求了$\{S,B_k\}$不许相交则满足了
$$
f_A(f_B(x)) = f_A*f_B(x) 
$$
不严谨但是比较好理解的说，每个$f_{B_i}(x_i)$分解为向量坐标后都处于不同的有效维度中，通过点积公式很容易就发现二者相等
