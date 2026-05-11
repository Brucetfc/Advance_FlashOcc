# *<font color='blue'>FlashOcc改进工作的目前状态</font>*

## <font color='green'>Baseline： FlashOcc</font>

速度快 参数少 36.9FPS，mIoU：32.08

```
# with det pretrain; use_mask=True;

# ===> per class IoU of 6019 samples:

# ===> others - IoU = 6.74

# ===> barrier - IoU = 37.65

# ===> bicycle - IoU = 10.26

# ===> bus - IoU = 39.55

# ===> car - IoU = 44.36

# ===> construction_vehicle - IoU = 14.88

# ===> motorcycle - IoU = 13.4

# ===> pedestrian - IoU = 15.79

# ===> traffic_cone - IoU = 15.38

# ===> trailer - IoU = 27.44

# ===> truck - IoU = 31.73

# ===> driveable_surface - IoU = 78.82

# ===> other_flat - IoU = 37.98

# ===> sidewalk - IoU = 48.7

# ===> terrain - IoU = 52.5

# ===> manmade - IoU = 37.89

# ===> vegetation - IoU = 32.24

# ===> mIoU of 6019 samples: 32.08
```

## <font color='green'>FlashOcc_HGAC_GeoLoss</font>>**

第一次改进加入HGAC(Height-Grouped Attention Convolution)高度分组注意力卷积，目的是解决FlashOcc在Channel-to-Height映射过程中存在的高度语义混叠问题，原始的FlashOcc将高度维度直接压缩到BEV通道维度之后，后续使用标准的2D卷积进行特征建模，但是传统的2D卷积默认所有通道具有均匀语义属性，没有显式区分不同通道对应的物理高度信息，这会导致不同高度层的语义特征在卷积过程中被无差别混合，尤其是对于细粒度目标如（行人，交通锥，自行车，摩托车等）容易产生高度结构模糊，目标断裂，以及垂直拓扑不连续的问题，因此HGAC的出发点是通过高度感知特征解耦来增强网络对垂直空间结构的建模能力

```
FlashOcc_HGAC_GeoLoss, Training 24epoch:
===> per class IoU of 6019 samples:
===> **others - IoU = 7.11**  （异性障碍物）+0.37
===> **barrier - IoU = 39.72**    +2.07
===> bicycle - IoU = 10.38
===> bus - IoU = 38.44
===> car - IoU = 44.27
===> construction_vehicle - IoU = 13.67
===> motorcycle - IoU = 13.14
===> **pedestrian - IoU = 16.08**    +0.29
===> **traffic_cone - IoU = 15.97**    +0.59
===> trailer - IoU = 26.87
===> truck - IoU = 30.85
===> driveable_surface - IoU = 78.95
===> **other_flat - IoU = 39.22**  +1.24
===> sidewalk - IoU = 48.86
===> terrain - IoU = 52.22
===> **manmade - IoU = 38.55**   +0.66
===> **vegetation - IoU = 33.15**   +0.91
===> **mIoU of 6019 samples: 32.2**   +0.12
```

但是存在的问题就是Car，Truck， Bus这些大目标类别没有涨幅，略降

## *<font color='green'>Flash_Occ_AdvSoftHGAC_GeoLoss_Routing Diversity Loss</font>*

目前的整体改进，本质上是在针对 FlashOcc 中 “2D BEV 特征在 Channel-to-Height 映射过程中缺乏显式高度结构建模” 这一核心问题，构建一个动态高度解耦的三维占据表示学习框架。

原始 FlashOcc 的核心思想，是通过将 3D Occupancy Volume 沿高度维度 (Z) 展平到 channel 维度，从而将传统高开销的 3D 卷积问题转化为高效的 2D 卷积问题。其本质可以表示为：

$$
\mathbf{F}*{BEV}\in\mathbb{R}^{C\times H\times W}\rightarrow\mathbf{V}*{occ}\in\mathbb{R}^{(C\times D_z)\times H\times W}
$$
其中：(H,W) 表示 BEV 空间尺寸；(D_z) 表示垂直高度层数；高度维度被隐式编码到 channel 中。这种设计虽然极大降低了推理开销，但存在一个非常明显的问题：不同物理高度的语义信息在 2D 卷积过程中被强制混合。

由于标准卷积对 channel 一视同仁，因此：地面区域（driveable surface）中层结构（car body）高层结构（traffic light / vegetation）会共享同一套卷积核。这种高度语义混叠（Height Semantic Entanglement）会导致：

1. 小目标高度结构模糊；
2. 细长物体出现断裂；
3. 垂直拓扑连续性缺失；
4. 远距离目标边界退化。

因此，目前提出的核心目标，就是在不恢复昂贵 3D 卷积的前提下，为 2D BEV 特征建立显式的高度结构感知能力。为了解决这个问题，设计了 Advanced Soft Height-Grouped Attention Convolution（Advanced Soft HGAC）。与最初的静态 HGAC 不同，当前已经从“固定高度分组”演化为“动态空间高度路由”。

核心思想是：让网络在 BEV 空间的每一个位置，自主决定当前特征更应该属于：低层结构；中层结构；高层结构，中的哪一种高度语义。

具体而言，首先通过 Routing Generator 生成空间动态路由权重：

$$
\mathbf{R}=\operatorname{Softmax}(f_{router}(\mathbf{X}))
$$
其中：

$$
(\mathbf{X}\in\mathbb{R}^{C\times H\times W})
$$
为输入 BEV 特征；
$$
(\mathbf{R}\in\mathbb{R}^{3\times H\times W})
$$
表示三个高度平面的概率分布；Softmax 保证同一空间位置的高度权重满足概率归一化：
$$
\sum_{k=1}^{3}R_k(h,w)=1
$$
随后，网络基于动态路由权重，对原始特征进行软分配：

$$
\mathbf{X}*{low}=\mathbf{X}\odot R*{low},\quad \mathbf{X}*{mid}=\mathbf{X}\odot R*{mid},\quad \mathbf{X}*{high}=\mathbf{X}\odot R*{high}
$$
这里的物理意义非常关键。与传统静态分组不同，现在的方法不再假设：“固定 channel 对应固定高度”。

而是允许：同一个 channel 在不同空间区域动态承担不同高度语义。

例如：近距离区域可能主要属于 ground-level；远距离区域可能主要属于 high-level sparse structure。

因此，现在的方法本质上建立了一种，空间自适应的高度语义解耦机制。在得到三个高度分支后，进一步利用独立的高度特异性卷积进行建模：

$$
\mathbf{F}*{low}=f*{low}(\mathbf{X}*{low}),\quad \mathbf{F}*{mid}=f_{mid}(\mathbf{X}*{mid}),\quad \mathbf{F}*{high}=f_{high}(\mathbf{X}_{high})
$$
其中每个分支对应不同高度结构；采用 group convolution 保持轻量化；不同高度拥有独立参数空间。这一阶段的核心目标是避免不同高度特征在同一卷积核中发生语义冲突。随后提出了 Cross-Height Interaction Module. 因为真实世界中的物体并不是高度独立的。例如车辆通常具有：lower-level wheel structure；middle-level body structure；upper-level roof structure。

因此，仅进行高度解耦是不够的，还必须建立高度层之间的垂直拓扑关联。

$$
\mathbf{F}*{cross}=f*{fusion}([\mathbf{F}*{low};\mathbf{F}*{mid};\mathbf{F}_{high}])
$$
通过显式跨高度交互，实现：低层几何；中层语义；高层结构；之间的信息协同。

这一部分实际上是在显式建模：Vertical Structural Dependency。这是当前很多纯视觉 Occupancy 方法缺乏的。但是过度高度解耦会导致大目标出现性能下降。

例如：Bus；Truck；Trailer；等大型目标，本身依赖完整的全局上下文。如果完全进行局部分裂，会破坏其整体语义一致性。

因此额外引入了：Global Residual Branch：

$$
\mathbf{F}*{global}=f*{global}(\mathbf{X})
$$
并最终进行联合融合：

$$
\mathbf{F}*{out}=f*{proj}(\mathbf{F}*{cross}+\mathbf{F}*{global})
$$
本质上是：在细粒度高度建模之外，保留原始 BEV 的全局上下文信息。从而避免大目标语义断裂；长距离结构不连续；全局几何退化。

在训练层面，设计了两类约束机制。第一类是 ***<u>Geometry-Aware Vertical Consistency Loss</u>***。其目标是约束 Occupancy Prediction 在高度方向上的连续性。具体而言，沿 (Z) 轴计算相邻体素概率差分：

$$
\mathcal{L}_{vert}=\frac{1}{N}\sum\left|P(z+1)-P(z)\right|
$$
它能够有效缓解：vertical hole；fragmented prediction；discontinuous occupancy structure。

尤其对：pedestrian；pole；traffic cone；等细长结构具有明显作用。

第二类是 ***<u>Routing Diversity Loss</u>***。

由于 soft routing 容易出现：Routing Collaps，也就是所有像素全部偏向同一个高度，因此通过余弦相似度约束：

$$
\mathcal{L}*{div}=\sum*{i\neq j}\operatorname{CosSim}(R_i,R_j)
$$
强制不同高度路由学习不同的语义模式。其目标是增强：高度分支差异性；解耦能力；空间语义多样性。

整体目前方法的一个较完整的框架是：

1. 动态高度软路由；
2. 高度特异性特征建模；
3. 跨高度垂直拓扑交互；
4. 全局语义补偿；
5. 几何一致性约束；
6. 路由多样性正则化。

核心贡献：在保持 FlashOcc 原始高效 2D 卷积范式的基础上，为纯视觉 Occupancy Prediction 引入显式高度结构建模能力，从而提升复杂场景下的三维空间语义表达能力。

