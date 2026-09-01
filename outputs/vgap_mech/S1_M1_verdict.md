# S1(判死活/立项)+ M1(归因首配对)判读 — 2026-08-31

独立项目 VGAP,R-Judge(n=571),env qwen,node 115。遵循冻结的预注册 `S1_logitlens_预注册方案_20260831.md`
(Step 0 先行、末层承重、跨≥3 宿主同向、标签置换基线)。**结果只在本项目。**

## 特异性确认(前置,已过)
`confirm_rjudge_qwen7b.json`:注入通道 verbal 0.551 / 探针 0.955–0.966(gap +0.415);
对照通道(判是否含 URL,客观标签)verbal **0.962** / gap −0.047。→ verbal 通道本身没坏,注入失败是特异的"知而不言"。

## S1:断链真实且跨 3 宿主同向 → 机制篇立项(Go)

| 宿主 | Step0(末层真 logits) | 探针末层 | lens末层 | 判定 |
|---|---|---|---|---|
| Qwen2.5-7B | 0.539 | 0.974 | 0.552 | Go |
| llama-3-8b-instruct | 0.535 | 0.976 | 0.536 | Go |
| gemma-3-4b-it | 0.526 | 0.962 | 0.526 | Go |

- **Step0 三宿主全 ~0.53(随机)→ No-Go A 否决**:注入信息**没进末层 logits**,gap 不是解码/校准假象。
- 探针末层 ~0.96–0.98(复现存档 0.96 ✓,内置正确性校验过);lens 末层 ~0.53 ≤ 0.75、CI 与探针不重叠。
- 预注册判据 §2.6 全满足且**三宿主同向**(过本篇铁律,非 M-C 那种翻车)。
- 形态:注入概念在中层一度进输出基(Qwen L9 lens 0.94),到末三分之一被转出可读空间归零;探针始终 0.97。
  (中层 lens 属补强三的不可靠区,承重点在末层;中层"短暂对齐"若要写需 tuned-lens 复核。)

## M1(归因,base vs instruct):**不支持"对齐制造 gap"的强故事**

仅有 llama 一个 base/instruct 配对(Qwen2.5-7B base 不在盘上)。用 Step0(末层真 logits,对称伪口头):

| llama-3-8b | Step0 | 探针末层 | lens末层 | Δgap=探针−lens |
|---|---|---|---|---|
| instruct | 0.535 [.487,.583] | 0.976 | 0.536 | 0.440 |
| **base** | 0.606 [.560,.654] | 0.962 | 0.378 | 0.584 |

- base 的 Step0(0.606)只比 instruct(0.535)略高、CI 勉强重叠;base 的 gap 反而**更大**。
- **两者都近随机 → 更像结构性读出瓶颈(注入方向不在 unembedding 可读子空间),不是指令微调把它压下去。**
- 混淆:base 无 chat 模板、不是称职 judge,其低读出可能是"不会做这个任务"而非"知而不言";单配对不足以下结论(预注册 §M1 要 ≥2 配对同向)。

## M2:排除廉价解释(gap 不可 prompt 出来)—— 支持论文

`M2_intervention_rjudge_qwen7b.json`,Qwen2.5-7B·R-Judge n=571,verbal AUROC(探针参照 ~0.96):

| 干预 | verbal AUROC |
|---|---|
| zeroshot | 0.551 [.513,.587] |
| CoT(先分析再打分) | 0.432 [.384,.478] |
| few-shot(2 示范) | 0.475 [.429,.518] |

- **三种干预都没把 verbal 拉近探针**;CoT/few-shot 反而**比 zeroshot 更低**(越让它推理越偏)。→ **NOT-FIXABLE-BY-PROMPTING=True**。
- (n=20 冒烟时 few-shot 曾 0.98,是仅 3 良性的小样本噪声,全量证伪。)
- 待铺开:M2 目前单宿主,预注册要求四宿主 × R-Judge+CSTM;CoT 下若要同时报探针需把探针位置重对齐到"吐分前"(§3.2)。

## 结论与框架建议

- **机制篇的脊柱(S1)成立且稳健**:探针读得到、模型自己读不出、非解码假象、跨 3 宿主同向。这是可发的机制结果。
- **但主打"对齐压住已知知识"这个天花板故事,当前证据不支持**——首配对反而指向**结构性读出瓶颈**。
  建议把论文框架从"alignment hides it"改到更稳的"**injection 方向结构性地不被 unembedding 读出**",
  alignment 归因作为**待定/需更多配对**(要坐实需 Qwen2.5-7B base 权重[需下载]+ §3.1 对称伪口头 + ≥2 配对同向)。
- 未做:M2(CoT/few-shot 排除廉价解释)、tuned-lens 中层复核、CSTM 一致性附跑。

工件:`confirm_rjudge_qwen7b.json`、`S1_logitlens_rjudge_{qwen7b,llama8b,gemma4b}.json`、`M1_S1_rjudge_llama8b_base.json`。
