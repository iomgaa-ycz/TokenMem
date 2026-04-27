# Query Pack (auto-generated, max 8000 chars)

**Last updated**: 2026-04-26

## Project Direction (300 chars)
TokenMem: 面向冻结LLM的即插即用记忆pipeline。检索(FAISS)→融合(Cross-Attention,借鉴DecoupledRAG)→更新(token编辑)。6模型×3家族验证。News 50K SFT→MedQA/ARC/MMLU跨域泛化。ExplicitLM(ICLR26,本组)后继。

## Top Gaps (1200 chars)
G1🔴 多模型通用性: 现有最多3模型(KBLaM)，无6+模型验证
G2🔴 冻结LLM即插即用: ExplicitLM需预训练，DecoupledRAG无持久记忆
G3🟡 完整pipeline: 无系统同时做检索+融合+更新
G4🟡 动态知识管理: ExplicitLM固定容量，无运行时增删编辑
G5🟡 跨领域泛化: 现有系统每任务独立SFT

## Paper Clusters (1600 chars)
**Cluster 1: Token记忆** — ExplicitLM(ICLR26,本组)证明token存储+PKM有效(+43.67%)但需预训练。TokMem(ICLR26)做过程记忆soft token(不可读,不同方向)
**Cluster 2: Cross-Attention注入** — DecoupledRAG(WWW25)证明冻结LLM+cross-attn+零初始化LoRA可行(只需SFT,~4.19M参数)。KBLaM(ICLR25)用矩形注意力注入KB三元组(3模型)。K-Capsules(Apr26)用KVI注入结构化capsule
**Cluster 3: 隐式记忆** — MemoryLLM(ICML24)隐式hidden states自更新(不可读不可编辑)。M+(ICML25)加retriever扩容到160K
**Cluster 4: 高效检索** — PKM(NeurIPS19)product key O(√N)。FwPKM(Jan26)PKM+在线梯度更新

## Failed Ideas (1400 chars)
idea:002 MemoryBridge — 纯跨模型方案，消除因单点贡献不足
idea:003 EditMem — 纯编辑方案，消除因与ROME/MEMIT领域期望有gap
idea:004 ConvoMem — 对话自增长记忆，消除因知识提取质量是混淆变量
⚠️ PKM方案A(内容导出型key)也被否决：PKM key需要聚类训练才有效，随机分组导致平均key退化为噪声。当前用FAISS替代。

## Top Papers (1800 chars)
1. paper:explicitlm2025 🔴核心 — 本组prior work。token记忆+PKM+预训练。TokenMem改为SFT即插即用
2. paper:decoupledrag2025 🔴核心 — 融合机制来源。代码验证:基座冻结+只训gate_crossattention(~4.19M)。无Pretrain只需SFT
3. paper:kblam2025 🟡竞品 — 结构化三元组KV注入，3模型。不支持自然语言
4. paper:kcapsules2026 🟡竞品 — 最新(Apr26)。结构化capsule+KVI。运行时不可读
5. paper:memoryllm2024 🟢对比 — 隐式自更新。不可读不可编辑
6. paper:lample2019_pkm 🟢参考 — PKM理论来源。训练方式不适用(联合预训练LM)
7. paper:fwpkm2026 🟢参考 — PKM+在线更新可行但不可读
8. paper:tokmem2025 ⚠️命名 — 名称可能混淆,方向不同(procedural vs factual)

## Active Chains (900 chars)
ExplicitLM(需预训练) →limitation→ TokenMem(冻结LLM+SFT) →addresses→ G2
DecoupledRAG(无持久记忆) →limitation→ TokenMem(持久bank+编辑) →addresses→ G3,G4
KBLaM(3模型) →limitation→ TokenMem(6模型) →addresses→ G1
DecoupledRAG(每任务独立SFT) →limitation→ TokenMem(一次SFT跨域) →addresses→ G5

## Open Unknowns (500 chars)
- OOD泛化效果(News训练→MedQA/ARC/MMLU)到底有多少？→ E1验证
- 大模型(8B)的adapter是否和小模型同样有效？→ E1验证
- 知识编辑的cascade效应如何？→ E2验证(简化版)
- 论文命名是否需要避开TokMem？→ 待决定
