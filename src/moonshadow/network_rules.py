"""网络赋值的规则版本对照。

`v1`（基线）在 `compress.baseline_network`：三个分支、依赖 `tier`、**永不输出 observation**。
`v2` 是本模块：**不依赖 tier**（修掉 tier 泄漏），四类各有类级词汇，按判别力排序。

规则改进只允许看 `split == "dev"` 的用例；`holdout` 在改进完成后只跑一次。
词汇表按**类别语义**建立（"这类话通常长什么样"），不是从某条用例的字符串里抄的。
"""

from __future__ import annotations

import re

#: 判别顺序即优先级。次序依据：
#: 一次性/寒暄信号最强 → 主观评价 → 做法与教训 → 客观属性 → 兜底。
OBSERVATION_RE = re.compile(
    r"(嗯|好的|收到|谢谢|多谢|辛苦|……|\.\.\.|明白|了解|稍等|先看|"
    r"今天|昨天|明天|刚才|刚刚|这会儿|早上|下午|晚上|开会|偶发|暂时|顺手)"
)
OPINION_RE = re.compile(
    r"(觉得|认为|看法|倾向|宁愿|宁可|建议|希望|喜欢|不喜欢|最好|"
    r"以后都|不用|别再|挺.{0,4}的|太.{0,4}了|不值得)"
)
EXPERIENCE_RE = re.compile(
    r"(决定|采用|方案|理由|做法|约定|规范|流程|步骤|踩|坑|教训|"
    r"失败|回滚|灰度|压测|兼容|不要|别用|注意|经验|以前|上次)"
)
WORLD_RE = re.compile(
    r"(版本|字符集|地址|端口|域名|邮箱|IP|QPS|延迟|内存|节点|集群|"
    r"金额|价格|预算|费用|配置为|端口是|叫作|名字|叫|号$|元。?$|%|"
    r"是 [A-Za-z0-9]|PostgreSQL|Redis|MySQL|UTF8)"
)


def classify_network_v2(text: str) -> str:
    """规则版网络赋值 v2。**不接收 tier**——这是与 v1 的关键差别。"""
    if OBSERVATION_RE.search(text):
        return "observation"
    if OPINION_RE.search(text):
        return "opinion"
    if EXPERIENCE_RE.search(text):
        return "experience"
    if WORLD_RE.search(text):
        return "world"
    return "experience"
