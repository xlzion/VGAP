# Verbalization-gap 确认(判死活)判读 — 2026-08-31

**独立项目 VGAP,结果不入 NSCC。** 宿主 Qwen2.5-7B · R-Judge(n=571,注入 301 / URL 对照 92),
`run_vgap_confirm.py`,env qwen,node 115。方案原 §S1(logit-lens 定位)有平凡通过风险
(logit-lens 对几乎所有概念中层都读不出、末层又只重述已知 gap),故改为**特异性对照**判死活。

## 结果:注入特异的 know-say gap 成立(CONFIRM=True)

| 通道 | verbal AUROC | 探针 AUROC(dom/logit) | gap=探针−verbal |
|---|---|---|---|
| **注入**(判是否有注入) | 0.551 [.513,.587] | 0.955 / 0.966 | **+0.415** |
| **对照**(判是否含 URL,客观标签) | **0.962** [.941,.979] | 0.706 / 0.915 | −0.047 |

- 注入 verbal 0.551 与 M-A 存档值(RESULTS_SUMMARY 的 Qwen2.5-7B R-Judge verbal 0.551)**完全吻合** → 复刻可信。
- **verbal 通道本身没坏**(对照 0.962):同模型、同上下文、同 0-100 输出格式,模型能完美 verbal 报出 URL,
  却对注入近乎随机——而激活明明把注入分到 0.96。**排除了"读出通道通用性损坏 / logit-lens 伪影"这个廉价解释**,
  gap 是注入特异的"知而不言"。

## 结论:值得铺开(方案的机制篇有真地基)

确认通过 → 按方案进入铺开(**先得确认、已确认**):
1. **归因 Q2(最高价值,权重已就位)**:llama3-8b(base)vs llama3-8b-instruct(两权重都在集群)——
   若 instruct 的 gap 显著大于 base,则"对齐训练压住了模型已知的注入知识";base 也一样大则是结构性读出瓶颈。
2. **多宿主同向**(本篇铁律,防 M-C 翻车):gemma-3-4b-it / qwen3.5-4b 复现特异 gap。
3. Q3 排除廉价解释(CoT/few-shot 关不掉 gap)、logit-lens 作补充定位(降级为佐证,不作主判据)。

诚实边界:单宿主、单对照概念(URL)、探针 in-sample 乐观(相对 gap 才是有效信号)。铺开须跨≥3 宿主同向才写机制主张。
