"""以国家为整群(block)的自助法。

为什么不按窗口抽: 滚动窗口高度重叠(相邻10年窗口共享9年), 且各国房价受同一轮
全球周期驱动。按窗口有放回抽样等于把强相关的样本当独立样本, 置信区间会窄得
失真。把整个国家作为不可分的 block 重抽, 同时吸收窗口重叠与国别内相关。

代价: block 只有 16 个(有租金收益率序列的国家数), 区间本身也有噪声, 且偏保守。
这正是本项目愿意接受的方向——宁可把不确定性说大, 不可说小。
"""
import numpy as np

N_BOOT = 2000


def country_block_ci(iso, x, fn=np.mean, n_boot: int = N_BOOT, seed: int = 0,
                     qs=(2.5, 97.5)) -> list[float]:
    """fn 在"重抽国家后拼接的样本"上的取值分布, 返回其 qs 分位数。

    iso: 与 x 等长的国家标签; x: 逐窗口的取值; fn: 作用于一维数组的统计量。
    """
    iso = np.asarray(iso)
    x = np.asarray(x)
    blocks = [x[iso == c] for c in np.unique(iso)]
    G = len(blocks)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, G, G)
        draws[b] = fn(np.concatenate([blocks[j] for j in pick]))
    return [float(v) for v in np.percentile(draws, qs)]
