# PGMC logits 融合选择

原始代码同时计算了三种候选：

1. 直接加权和（原变量 `final_logits`）；
2. 仅保留相对零样本 logits 正增量的截断和（原变量 `some`）；
3. 去掉目标全局与目标重建分量的和（原变量 `other`）。

从原项目 `logs/train.log` 中，只保留“最终上报准确率与该轮 `final_logits` 准确率一致”的完整记录，共匹配 102 次实验：

| 方案 | 平均准确率 | 单次最优次数 |
| --- | ---: | ---: |
| 直接加权和 | 52.94 | 55 |
| 截断增量和 | 52.09 | 27 |
| 去掉目标全局分量 | 48.53 | 20 |

因此 PGMC 默认只保留直接加权和：

```text
zero_shot
+ target_global
+ target_reconstructed
+ 2 * source_global
+ 2 * source_reconstructed
+ source_local
+ target_local
```

所有权重集中在 `configs/pgmc.yaml`，便于后续消融，但运行代码中不再维护三套互相分叉的公式。

