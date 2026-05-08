你是一个国际人工智能顶级学术会议的写作助手，请帮我完成写作任务。

# SharQ Agent Brief

SharQ 是一种面向低比特推理的激活分解方法：它用硬件友好的 N:M 半结构化稀疏 FP4 提取由离群值主导的主干信息，用低比特补偿路径编码更规整的剩余部分，并通过共享 FP4 权重与路径相关的 scale 设计解决 dense–sparse 的 scale 粒度不匹配问题。

本文档用于帮助写作/代码代理准确理解 SharQ 的研究目标、方法设计、实现边界与论文结构。重点是**避免对方法本身产生理解偏差**。SharQ 不是一个泛泛的“稀疏 + 量化”拼接方案，也不是单纯把已有半结构化稀疏与低比特量化串联起来；它的核心是一个面向 activation 的 **decomposition strategy**，并围绕这一点展开算法、kernel 与实验设计。

永远不要擅自改动论文的模板格式，每次写入前需确认。
目录中存在 `eng.tex` 与 `chinese.tex` ，二者内容一致，仅有语言不同，每次修改内容时同步修改两个文件，确保内容同步一致。

---

> 由于时效性，以下内容可参考理解论文写作思路，以及 SharQ 算法思路，但并非严格写作大纲

## 1. 项目定位与目标

### 1.1 论文主体目标

SharQ 的主体目标已经收敛为：

- 以 **LLM inference** 为主场景；
- 提出一种面向 Blackwell 等新硬件环境的 **activation decomposition low-bit inference method**；
- 在主线实验中聚焦标准 LLM；
- 在扩展性实验中额外测试 MoE、DiT 等模型架构，说明方法并不局限于单一 dense LLM 结构。

因此，SharQ 论文的主叙事不是“我们做了一个通用稀疏框架”，而是：

> 在低比特推理中，activation 的不同组成部分具有不同统计性质，因此应先进行分解，再分别采用更适合的压缩与计算路径。

### 1.2 不是在做什么

为了避免代理误解，下面几点必须明确：

1. **SharQ 不是简单的稀疏与量化串联。**
   不是“先做个 top-k 稀疏，再把结果量化一下”。

2. **SharQ 不是在主张纯稀疏优于量化。**
   恰恰相反，现有观察表明，对 activation 而言，纯稀疏带来的损失通常明显大于纯量化。

3. **SharQ 不是高精度 sparse branch + 低比特 residual branch。**
   sparse branch 本身也允许是低比特的，甚至正因为 residual path 会补偿，sparse branch 没必要追求高精度独立重建。

4. **SharQ 不是普通意义上的残差量化。**
   这里的 residual 不是简单的“原激活减去某个近似值”后直接做二次量化这么朴素；论文里可以使用 residual 这一术语，但必须明确它在 SharQ 中承担的是一个统一的补偿角色。

---

## 2. 核心方法理解：SharQ 的正确叙事

### 2.1 一句话概括

SharQ 的核心不是把稀疏和量化拼接起来，而是先对 activation 做分解，让：

- **sparse path** 负责提取由 outlier 主导的 backbone；
- **low-bit dense/residual path** 负责表示更规整的剩余信息；
- 并通过统一的 compensation 组织 sparse approximation error 与 quantization error。

### 2.2 为什么需要 decomposition

SharQ 的出发点来自如下观察：

1. **activation 存在明显离群值与一定稀疏性。**
   这意味着激活中少数高幅值位置承载了很强、很关键的信息。

2. **但直接稀疏化 activation 通常效果很差。**
   即使激活“看起来稀疏”，直接采用 naive sparse approximation（例如简单 top-k / 半结构化选择）往往会造成远大于量化的精度损失。

3. **低比特量化又很怕 outlier。**
   尤其是 4bit activation quantization 中，outlier 会显著拉大动态范围，使主体分布更难被低比特均匀量化良好表示。

4. **直接把稀疏和量化串联，误差会进一步放大。**
   稀疏已经引入一轮信息损失，再叠加量化，误差通常并非简单线性叠加，而是存在明显交互作用。

因此，SharQ 的关键切入点不是“选择稀疏还是量化”，而是：

> 能否先把 activation 中最难量化、但又最关键的部分分离出来，再让剩余更规整的部分去接受低比特表示？

### 2.3 decomposition 的直观图景

可将 activation 理解成两个统计性质不同的部分：

- **outlier-dominated backbone**：少数高幅值、显著扰动量化分布、但又很关键的成分；
- **more regular remainder**：去除这些主干后，剩余成分的分布更平滑、更规整，更适合低比特量化。

SharQ 的核心正是围绕这两个部分的分工展开。

### 2.4 硬件支持与稀疏模式

SharQ 所依赖的硬件支持，不是泛化意义上的“稀疏”，而是 **N:M 半结构化稀疏**。

并且，针对不同低比特格式，SharQ 对应采用不同的半结构化稀疏模式：

- 对 **NVFP4**，使用 **4:8（in pairs）** 半结构化稀疏；
- 对 **HiF4**，使用 **2:4（标准）** 半结构化稀疏。

这点必须保持准确，因为它直接关系到 kernel 映射与 scale 组织，不能被笼统写成“structured sparsity”而不加区分。

---

## 3. 方法主线：两条路径的功能分工

### 3.1 sparse path 做什么

sparse path 的职责不是高精度地完整重建 activation，而是：

- 优先保留由 outlier 主导的 backbone；
- 把最扰动低比特量化分布的成分先显式提取出来；
- 以硬件友好的 **N:M 半结构化稀疏** 形式参与后续计算。

这一点很关键。代理在写作时，不能把 sparse path 描述成“负责高保真表示激活”，因为那会偏离 SharQ 的方法思想。

更准确的说法是：

> sparse path 负责抓住最显著、最关键、同时也是最难量化的主干信息，而不是独立承担完整表示。

### 3.2 为什么 sparse path 也可以是 4bit

由于 residual / compensation path 会承担剩余信息和误差补偿职责，因此 sparse path 并不需要高精度。

这直接导出一个重要设计选择：

- sparse path 本身也可以直接使用 **4bit sparse representation**；
- 无需走“高精度稀疏 + 低比特 residual”的路线；
- 这不是单纯为了节省存储或计算，而是由路径功能分工自然得到的。

这一点必须保持一致。不要把 SharQ 错写成“高精 sparse backbone + low-bit residual refinement”。

### 3.3 residual / dense path 做什么

在 outlier backbone 被 sparse path 提取之后，剩余部分会更规整。这时，低比特量化面对的不再是原始 heavy-tail activation，而是经过“去离群化”的剩余信息。

因此，residual path 的任务是：

- 表示剩余更规整的信息；
- 承担 sparse approximation 引入的信息缺失补偿；
- 承担 sparse branch 低比特表示产生的相关量化误差补偿；
- 以统一的低比特形式参与最终计算。

这里尤其要注意：SharQ 中使用“residual”这一词是允许的，但应始终强调其**统一补偿通道**的含义，而不能让读者误解为只是一个朴素的数值残差。

---

## 4. residual 的准确含义

这是最容易被误解的地方之一，必须单独说明。

### 4.1 不要把 residual 理解得过窄

SharQ 中的 residual 不能只被理解为：

- “稀疏之后剩下的那些值”；或
- “原始激活减去 sparse activation” 后直接得到的普通余项。

这样理解太窄，也无法体现 SharQ 的统一补偿思想。

### 4.2 更准确的理解

SharQ 中的 residual 更适合被理解为一个 **unified compensation carrier**。

它里面至少包含两类信息：

1. **sparse selection 没有覆盖到的信息**，也就是 sparse approximation 本身留下的误差；
2. **sparse path 保留部分在低比特表示后产生的量化误差相关信息**。

也就是说，SharQ 不是分别处理“稀疏误差”和“量化误差”，而是希望把这两类误差统一组织进一个补偿分量中。

### 4.3 写作时的措辞建议

可用以下表达：

- residual serves as a unified compensation path;
- residual absorbs both sparse approximation loss and quantization-induced distortion;
- residual is not merely the leftover activation, but the carrier of jointly compensated information.

避免使用下列表述：

- residual is simply the discarded part after sparsification;
- we first sparsify and then quantize the residual in a straightforward residual quantization manner.

这些说法会把方法讲偏。

---

## 5. 为什么 SharQ 有效：方法层面的合理性

SharQ 的有效性不应被解释为“稀疏本身就很好”，而应从 decomposition 的角度去理解。

### 5.1 对量化友好性的改善

原始 activation 可能具有 heavy-tail / outlier-dominated 分布。对这类分布直接做低比特量化时，少数 outlier 会拉大整体动态范围，使主体区域的低幅值信息被迫落入较粗的量化区间中。

SharQ 中，sparse path 先把 outlier backbone 提取出来，相当于显式剥离掉最破坏量化友好性的成分。这样，residual path 面对的是更规整的对象，因此更适合 4bit 等低比特表示。

### 5.2 对纯稀疏不足的修正

纯稀疏的问题在于：即使抓住了大值，也不等于完整保留了关键语义信息。对 activation 而言，直接 sparse approximation 往往损失过大。

SharQ 不把 sparse path 当作最终近似，而是让它只承担 backbone extraction 的角色，并通过 residual path 补偿剩余信息与相关误差。因此，SharQ 的目标不是证明纯稀疏有效，而是证明：

> 若先用稀疏提取最关键但最难量化的主干，再用低比特路径统一补偿剩余信息与误差，则可以获得比 naive sparse / naive stacked sparse+quantization 更合理的精度-效率折中。

### 5.3 对“稀疏 + 量化交互误差”的处理

已知观察表明：

- 纯量化损失通常较温和；
- 纯稀疏损失往往更大；
- 稀疏后再量化会进一步变差，说明两者存在交互误差。

SharQ 的价值不在于消除所有误差，而在于：

- 通过 decomposition 重新组织误差结构；
- 把最难处理的 outlier 主干与更温和的剩余部分分离；
- 让 residual path 吸收 sparse approximation error 与 quantization error；
- 从而避免 naive 稀疏或 naive 串联方案中的灾难性退化。

---

## 6. 具体实现方向：kernel 与权重/scale 设计

### 6.1 kernel 设计在论文中的位置

kernel 细节是方法的重要组成部分，但不应抢占核心方法论叙事。更合适的组织方式是在 Methodology 中设置一个单独的 **Kernel Design** 小节。

这个 subsection 主要回答的是：

- decomposition 后的两条路径如何映射到实际 kernel；
- 稀疏与 dense/residual 分量如何高效组织计算；
- 如何避免朴素实现造成额外访存或调度开销。

### 6.2 需要讲清的 kernel 视角问题

代理在展开 kernel design 时，应围绕以下主题：

1. **两条路径的计算组织方式**
   - sparse branch 如何以硬件友好的 **N:M 半结构化稀疏** 形式参与计算；
   - residual/dense branch 如何与之配合。

2. **误差补偿如何在 kernel 层面落地**
   - compensation 不是抽象概念，需要通过具体数据排布、拼接或融合策略实现。

3. **为什么要强调 kernel-aware design**
   - 如果 decomposition 只是算法层分解，却需要两套低效 kernel 或过多 memory movement，则整体价值会被削弱；
   - SharQ 的设计目标之一就是让 decomposition 和实际硬件执行路径匹配起来。

### 6.3 shared weight 与 scale 问题的准确来源

这是实现层面最容易被写偏的地方，必须准确描述。

SharQ 中所谓的 shared weight 问题，并不只是抽象层面的“不同分量统计性质不同”，而是来自非常具体的 **scale granularity mismatch**。

具体来说：

- 对 **dense NVFP4** 路径，通常是 **16 个 FP4 权重共享一个 FP8 scale**；
- 对 **N:M sparse** 路径，通常是 **32 个 sparse FP4 权重共享一个 FP8 scale**，也就是实际非零的 **16 个元素** 共享一个 FP8 scale。

因此，dense path 与 sparse path 在权重量化的 scale 分组粒度上天然不匹配。这正是 SharQ 需要专门解决的问题。

### 6.4 SharQ 的解决方向

SharQ 的方向不是分别维护两套独立权重，而是：

- **构造 shared FP4 weight**，避免 dense path 和 sparse path 各自维护一套完整权重；
- 同时为两条路径保留**各自匹配的 scale**，以解决 dense NVFP4 与 N:M sparse 在 scale granularity 上的不一致。

也就是说，SharQ 最终需要解决的是一个非常具体的实现矛盾：

> 如何在共享同一份 FP4 weight 主体的前提下，让 dense path 和 sparse path 使用各自匹配的 scale 体系。

### 6.5 对代理的提醒

如果代理需要写更加具体的实现描述，必须遵守以下原则：

- 可以写 kernel design 的意图、组织方式、数据流；
- 可以写 shared FP4 weight 与 path-specific scales 的必要性；
- 不要凭空杜撰未确认的底层实现细节、公式、layout 名称或具体 CUDA 技巧；
- 若缺少确认信息，应保持抽象层面的准确，而不是编造具体实现。

---

## 7. 实验目标与评测设计

实验部分已经相对明确，分为主线效果实验、扩展性实验与性能实验。

### 7.1 主线效果实验：以 LLM 为主

主体模型：

- **Llama 3.1-8B**
- **Qwen3-30B-A3B**

主体任务：

- 若干 0-shot tasks
- WikiText PPL
- MMLU

这部分构成论文主结果，服务于 SharQ 作为 LLM 量化算法的主叙事。

### 7.2 扩展性实验：跨架构验证

扩展模型与任务包括：

- **Qwen2.5-7B-Coder-Instruct**：HumanEval、MBPP
- **Wan2.2-A14B**：VBench
- MoE、DiT 等架构作为扩展验证对象

这部分的定位不是抢占主线，而是说明 SharQ 的 decomposition 思想在不同架构和任务上具有可迁移性。

### 7.3 性能实验：三层结构

性能实验分为三层：

1. **单个 kernel 的测速**
   - 用于观察 micro-level efficiency；
   - 说明所设计 kernel 在局部计算层面的吞吐表现。

2. **prefill 阶段的 profile**
   - 分析不同组件的时间占比；
   - 说明瓶颈是否来自 GEMM、quantize、compensation、memory movement 等。

3. **vLLM end-to-end benchmark**
   - 用于系统层验证；
   - 说明 SharQ 在真实推理框架中的端到端价值，而不仅是 isolated kernel 的提升。

代理在撰写实验部分时，应保持这三层结构清晰，不要把 kernel benchmark 与 e2e benchmark 混为一谈。

---

## 8. baseline 选择与比较逻辑

### 8.1 量化 baseline

主线量化 baseline 包括：

- **W4A4**
- **W4A8**

必要时可围绕低比特 activation quantization 的常见代表方法组织比较。

这些 baseline 用来回答：

> 为什么不直接采用常规 low-bit quantization，而要采用 SharQ 这种 decomposition 策略？

### 8.2 稀疏 baseline

稀疏对比方向包括：

- **FP16 下的半结构化稀疏等算法**

同时，若论文主张需要说明“激活虽有稀疏性，但直接利用并不成立”，则应保留 naive activation sparsity 或相关对照，以支撑这一观察。

### 8.3 比较逻辑

对代理来说，baseline 不是简单罗列，而应体现两条主线：

1. 与纯量化比较：SharQ 相比传统 low-bit quantization 的收益；
2. 与纯稀疏/结构化稀疏比较：SharQ 相比直接利用 sparsity 的优势。

这样，SharQ 的贡献才能被准确放在“decomposition improves the trade-off”这一框架下理解。

---

## 9. 可视化与理论分析应该服务什么

### 9.1 可视化的作用

可视化不是装饰，而是直接支撑方法故事。

重点建议围绕三类图：

1. **原始 activation 与 residual 的分布对比**
   - 说明 sparse extraction 后 residual 更规整、更适合低比特量化。

2. **activation / backbone / residual 的热图**
   - 说明 sparse path 的确抓住了高幅值 backbone；
   - residual 更接近细粒度补偿。

3. **误差图**
   - 对比纯量化、纯稀疏、SharQ 等方案的误差分布；
   - 说明 SharQ 在重新组织误差结构，而不是简单叠加两个压缩模块。

### 9.2 理论分析的作用

理论分析的重点不一定是一开始就给出极其复杂的大定理，而是帮助解释 SharQ 为什么合理、为什么有效。

可以围绕两个方向：

1. **分布角度**
   - outlier 如何破坏量化友好性；
   - sparse extraction 如何让 residual 更规整。

2. **误差分解角度**
   - 最终误差如何由 sparse approximation、sparse path quantization、residual path quantization 共同构成；
   - SharQ 如何改变这些误差的整体结构。

代理写作时，应把理论分析视为“解释性支撑”，而不是脱离方法主线独立展开。

---

## 10. 论文写作时必须保持的统一表述

为避免风格漂移或方法理解偏差，全文应尽量保持以下统一表述：

### 10.1 推荐反复强调的关键词

- activation decomposition
- outlier backbone
- more regular residual
- unified compensation
- sparse approximation error and quantization error
- hardware-friendly N:M sparse kernel design
- shared FP4 weights with path-specific scale adaptation

### 10.2 应尽量避免的误导性表述

- simply combining sparsity and quantization
- straightforward residual quantization
- sparse branch precisely reconstructs activations
- sparsity alone already works well for activations
- SharQ mainly relies on a high-precision sparse path

### 10.3 全文的一致核心

全文都应围绕同一个核心：

> SharQ works by decomposing activations into components with different statistical properties and assigning them to different low-bit computation paths, instead of applying a single compression scheme uniformly to the whole activation.

---

# 论文主干结构大纲

下面给出一版较为稳定的论文主干结构，用于后续写作展开。

## 1. Introduction

### Background

- 低比特推理与 N:M 半结构化稀疏在新硬件上的重要性；
- Blackwell 等硬件提供了对 low-bit 与 N:M 半结构化稀疏协同加速的支持；
- NVFP4 与 HiF4 在稀疏映射上分别对应 4:8（in pairs）与 2:4（标准）半结构化稀疏；
- 现有的 W4A4、W4A8 量化效果较好但精度仍有下降，而权重的半结构化稀疏不仅精度下降显著，而且还有大量训练和微调开销
- 但围绕 activation 在线半结构化稀疏、稀疏与量化结合的相关探索仍然不足。

### Problem

- activation 存在 outlier 与稀疏性；
- 直接稀疏通常造成很大损失；
- 低比特量化又容易受 outlier 影响；
- 稀疏与量化直接串联时误差会进一步放大。

### Method Overview

- 提出 SharQ；
- 核心是在线的 activation decomposition：sparse path 负责 outlier backbone，low-bit path 负责 more regular residual；
- 通过 unified compensation 组织误差，同时用统一 FP4 数值格式利用 Tensor Core 的高吞吐实现推理加速；
- 与 CUDA kernel co-design 配合，降低在线操作的冗余访存开销，实现高效推理。

### Contributions

建议概括为三类贡献：

1. 利用激活稀疏性，提出一种面向 activation 的在线 decomposition-based FP4 inference strategy；
2. 设计统一 compensation ，创新性的同时在线补偿稀疏与量化的两部分误差
3. 设计高效的 fused kernel ，实现高效推理和端到端部署
4. 在跨模型架构、跨场景下表现优异，无需校准和微调，即插即用

## 2. Related Works

### 2.1 Quantization

- LLM low-bit quantization；
- weight-activation quantization；
- W4A4 / W4A8 等相关路线。

### 2.2 Sparsity

- 半结构化稀疏；
- activation sparsity；
- 稀疏在推理中的应用与局限。

### 2.3 Fine-grained Numeric Formats / Hardware Support

- 细粒度数值格式，NVFP4，MXFP4，HiF4；
- 新硬件对 low-bit 与 N:M 半结构化稀疏的支持；
- NVFP4 与 HiF4 对应的稀疏映射方式（分别对应 4:8 in pairs 与 2:4 标准半结构化稀疏）；
- 为什么这给 SharQ 提供了现实背景。

## 3. Methodology

### 3.1 Preliminary

随想：
```
1.problem definition，提出目标：对于模型中的一个线性层 Y=X*W，我们通过weight activation quantization、sparsity 去降低显存占用、加速模型推理。其中量化（默认对称量化、group wise）是如何计算缩放因子、得到低精度矩阵、反量化的；稀疏的话讲半结构化稀疏，即先按规则确定mask再通过稀疏gemm kernel即可实现加速。这里mask一般是离线学习、训练出固定的；但我们研究的是利用激活值，在线动态确定 mask。算法设计的目标是，原本的Y-量化/稀疏的X*同样的W的F范数最小2.block-scaled quantization与N:M稀疏，二者都是 blackwell 架构上对应ptx支持的，提高 tensor core吞吐量的策略。其中块缩放格式包括NVFP4、HiF4，前者已经在 blackwell 上支持，通过16一组的E4M3缩放因子和FP32二级缩放比标准FP4显著精度提升（细节见附录）而N:M半结构化稀疏则是规定连续M个元素中至少有N个0的情况下的稀疏，其中在blackwell上支持2:4和4:8 in pairs 情况的加速（详见附录）同时稀疏并量化，比如4:8 in pairs 加NVFP4可带来进一步的吞吐量提升，此时的NVFP4的scale会从16一组变成32一组（由于0的存在）
```


- activation 的 outlier 与稀疏性观察；
- 纯量化、纯稀疏、稀疏后再量化的现象；
- 为什么需要 decomposition。

### 3.2 Motivation

随想：
```
每一点先用一句话简短概括，再几句话去讨论：1. 激活的离群值影响量化产生量化误差，但引入混合精度对硬件不友好 2. 激活具有一定稀疏性，但在线直接做激活的半结构稀疏效果很差，动态mask又有性能风险，现有工作都有微调或者训练开销 3.FP4量化+N:M稀疏能有效加速，但两部分误差会叠加放大。question：我们能否在FP4或N:M稀疏约束下，利用激活稀疏性，实现有效加速，同时保持精度，做到场景通用、即插即用
```

### 3.3 Decomposition Strategy

- activation 如何被划分为 backbone 与 residual；
- sparse path 与 residual path 的功能分工；
- 为什么 sparse path 也可以直接使用 4bit；
- NVFP4 与 HiF4 分别对应的 N:M 半结构化稀疏模式。
- residual 的准确含义；
- sparse approximation error 与 quantization error 如何被统一组织；
- 为什么这不同于朴素 residual quantization。

### 3.4 Kernel Design

- 两条路径的实际计算组织；
- compensation 的落地方式；
- shared FP4 weights + path-specific scales 的必要性与实现思路；
- dense NVFP4 与 N:M sparse 在 scale 粒度上为何不匹配，以及如何解决。

### 3.5 Theoretical Analysis

- 分布角度：outlier、dynamic range、量化友好性；
- 误差角度：误差组成与 SharQ 的重组作用。

## 4. Experiments

### 4.1 Experimental Setup

- 模型：Llama 3.1-8B、Qwen3-30B-A3B 等；
- 扩展模型：Qwen2.5-7B-Coder-Instruct、Wan2.2-A14B、MoE / DiT；
- 任务：0-shot、Wiki PPL、MMLU、HumanEval、MBPP、VBench；
- baseline：W4A4、W4A8、FP16 半结构化稀疏等。

### 4.2 Main Results

- LLM 主线模型上的主要精度结果；
- 与量化 baseline、稀疏 baseline 的比较；
- 体现 SharQ 在精度-效率 trade-off 上的位置。

### 4.3 Generalization

- 在 coder model、MoE、DiT 等上的扩展验证；
- 说明 decomposition 思路的迁移性。

### 4.4 Efficiency

- kernel benchmark；
- prefill profile；
- vLLM end-to-end benchmark。

### 4.5 Ablation Study

建议围绕：

- decomposition 各组成部分作用；
- sparse/residual 的效果（稀疏+稀疏<<稀疏+稠密≈稠密+稠密）；
- shared weights / scale 设计影响；
- kernel 设计相关消融。

## 5. Conclusion

- 重申 SharQ 的核心：activation decomposition；
- 重申其在低比特推理中的意义；
- 点出其在新硬件环境与跨架构场景中的潜力。

