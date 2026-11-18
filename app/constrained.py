"""受限续写：禁用片段规避 + 指定尾句收尾的贪心解码。

规则（与需求逐项对应）：
  - 每步候选仅限可打印 ASCII 字符（词表 id 3..97）与 EOS（id 2）；
  - 生成文本、以及生成文本与提示词的拼接处，都不得出现任一禁用片段
    （Aho-Corasick 多模自动机保证跨边界检测；状态只依赖最近 m-1 个
    字符，m = 最长禁用片段长度，故可作为记忆化 DFS 的状态）；
  - 新文本必须以指定尾句紧接 EOS 结束：EOS 仅在「已生成文本的末尾匹配
    完整尾句」时合法，且 EOS 计入新 token 上限；尾句匹配从生成起点
    开始计算（约束作用于新文本本身，而非提示词后缀）；
  - 每步在合法候选中取目标模型 logit 最高者，平局取词表序号小者
    （屏蔽非法候选后对张量 argmax，天然平局取小）；
  - 「合法」不仅要求当前不立刻产生禁用片段，还要求从该候选出发仍能在
    剩余步数内合法结束（可行性 DFS，记忆化），避免走进死路；
  - 主路径复用目标模型 KV 缓存，按实际选中的 token 逐 token 增量续算；
  - 另跑一条独立的「无缓存、逐步对完整前缀重算」参考路径，两条路径
    的每步候选集、平局规则、可行性判定完全相同；只有逐 token 完全一致
    且终态合规复核通过才返回续写，否则判定 verify_failed，不展示任何
    部分结论。

该模块的全部推理都在模型状态快照/恢复保护下运行：函数返回后模型权重、
KV 缓存与 forward_count 与调用前完全一致，不影响原解码流程。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import torch

from . import tokenizer
from .model import CausalTransformer, greedy_argmax

# 每步候选：EOS + 95 个可打印 ASCII 字符（PAD/UNK 永远不可选）
EOS_ID = tokenizer.EOS_ID
CHAR_IDS: list[int] = list(
    range(tokenizer._FIRST_CHAR_ID, tokenizer._FIRST_CHAR_ID + len(tokenizer._CHARS))
)
CANDIDATE_IDS: list[int] = sorted([EOS_ID] + CHAR_IDS)

PRINTABLE_MIN, PRINTABLE_MAX = 0x20, 0x7E


def is_printable_ascii(text: str) -> bool:
    return len(text) > 0 and all(PRINTABLE_MIN <= ord(ch) <= PRINTABLE_MAX for ch in text)


# ----------------------------------------------------------------------
# 禁用片段：Aho-Corasick 多模自动机（按字符 id 建转移表）
# ----------------------------------------------------------------------
class ForbiddenAutomaton:
    """AC 自动机：状态 = 当前文本后缀与任一模式前缀的最长匹配。

    - goto[state][char] 为追加字符后的新状态（BFS 阶段补全，O(1) 转移）；
    - bad[state] 表示该状态下已有某条禁用片段在文本末尾完整命中
      （含 fail 链上的输出传播）。
    模式均为 2..6 个可打印字符；总状态数 ≤ 1 + 模式长度之和 ≤ 19。
    """

    def __init__(self, patterns: list[str]):
        assert patterns
        self.patterns = list(patterns)
        self.max_len = max(len(p) for p in patterns)
        pat_ids = [tokenizer.encode(p) for p in patterns]

        children: list[dict[int, int]] = [{}]   # trie 子边
        fail = [0]
        is_end = [False]                        # 该节点本身是否为某模式终点
        for pat in pat_ids:
            s = 0
            for cid in pat:
                nxt = children[s].get(cid)
                if nxt is None:
                    nxt = len(children)
                    children[s][cid] = nxt
                    children.append({})
                    fail.append(0)
                    is_end.append(False)
                s = nxt
            is_end[s] = True

        # BFS 求 fail 链并补全 goto 转移（经典 AC 构造：每个状态对每个
        # 字符都有确定转移，缺边沿 fail 链回退，运行期 O(1)）
        n_nodes = len(children)
        self.goto: list[dict[int, int]] = [{} for _ in range(n_nodes)]
        self.bad = [False] * n_nodes
        queue: deque[int] = deque()
        for cid in CHAR_IDS:
            child = children[0].get(cid)
            if child is not None:
                fail[child] = 0
                self.goto[0][cid] = child
                queue.append(child)
            else:
                self.goto[0][cid] = 0
        while queue:
            v = queue.popleft()
            # 输出传播：v 命中 ⇔ v 自身是模式终点，或其 fail 链上存在终点
            self.bad[v] = is_end[v] or self.bad[fail[v]]
            for cid in CHAR_IDS:
                child = children[v].get(cid)
                if child is not None:
                    fail[child] = self.goto[fail[v]][cid]
                    self.goto[v][cid] = child
                    queue.append(child)
                else:
                    self.goto[v][cid] = self.goto[fail[v]][cid]

        # 扁平转移表，解码热路径只做数组索引
        self._n_nodes = n_nodes
        self.table: list[list[int]] = [
            [self.goto[s][cid] for cid in CHAR_IDS]
            for s in range(n_nodes)
        ]

    def initial_state(self, prompt_ids: list[int]) -> int:
        """提示词喂入后的自动机状态；提示词已含禁用片段时抛错。

        状态编码：低位为 AC 节点 id，高位锁存「历史是否已命中」。
        禁用片段一旦在任意位置出现即永久违规，故命中后状态保持锁存。
        """
        s = 0
        valid = set(CHAR_IDS)
        for cid in prompt_ids:
            if cid not in valid:
                continue
            s = self.step(s, cid)
            if self.is_match(s):
                raise ValueError("prompt_forbidden")
        return s

    def step(self, state: int, char_id: int) -> int:
        node = state % self._n_nodes
        hit = state >= self._n_nodes
        nxt = self.table[node][char_id - tokenizer._FIRST_CHAR_ID]
        return nxt + (self._n_nodes if (hit or self.bad[nxt]) else 0)

    def is_match(self, state: int) -> bool:
        if state >= self._n_nodes:
            return True
        return self.bad[state]


# ----------------------------------------------------------------------
# 尾句匹配：已生成文本末尾与尾句前缀的最长公共长度（KMP）
# ----------------------------------------------------------------------
class TailMatcher:
    """状态 k ∈ [0, L]：生成文本后缀匹配尾句前 k 个字符。k==L 方可发 EOS。"""

    def __init__(self, tail: str):
        self.tail = tail
        self.tail_ids = tokenizer.encode(tail)
        self.L = len(self.tail_ids)
        pi = [0] * self.L
        j = 0
        for i in range(1, self.L):
            while j and self.tail_ids[i] != self.tail_ids[j]:
                j = pi[j - 1]
            if self.tail_ids[i] == self.tail_ids[j]:
                j += 1
            pi[i] = j
        self.pi = pi

    def step(self, k: int, char_id: int) -> int:
        if k == self.L:
            # 完整匹配后又追加字符：尾句不再贴尾，按前缀函数回退再匹配
            k = self.pi[k - 1]
        while k and char_id != self.tail_ids[k]:
            k = self.pi[k - 1]
        if char_id == self.tail_ids[k]:
            k += 1
        return k


# ----------------------------------------------------------------------
# 可行性：从 (forb_state, tail_state, budget) 出发能否合法收尾
# ----------------------------------------------------------------------
class Feasibility:
    """记忆化 DFS。budget = 剩余可发新 token 数（含本次与 EOS）。"""

    def __init__(self, forb: ForbiddenAutomaton, tail: TailMatcher):
        self.forb = forb
        self.tail = tail
        self.memo: dict[tuple[int, int, int], bool] = {}
        self.calls = 0

    def can_finish(self, forb_state: int, tail_state: int, budget: int) -> bool:
        # 当前状态已命中禁用片段则绝无合法出路（调用方本应只传入非命中状态，
        # 此处自防御，保证该判定器独立使用时也自洽）
        if self.forb.is_match(forb_state):
            return False
        if tail_state == self.tail.L and budget >= 1:
            return True  # 直接发 EOS 即可（EOS 不在字符自动机上）
        key = (forb_state, tail_state, budget)
        cached = self.memo.get(key)
        if cached is not None:
            return cached
        self.calls += 1
        # 还需补齐 L-k 个字符 + 1 个 EOS
        if budget < self.tail.L - tail_state + 1:
            self.memo[key] = False
            return False
        ok = False
        for cid in CHAR_IDS:
            nf = self.forb.step(forb_state, cid)
            if self.forb.is_match(nf):
                continue
            nt = self.tail.step(tail_state, cid)
            if self.can_finish(nf, nt, budget - 1):
                ok = True
                break
        self.memo[key] = ok
        return ok


# ----------------------------------------------------------------------
# 输入校验（网页/API 共用，返回中文错误原因；None 表示通过）
# ----------------------------------------------------------------------
def validate_inputs(prompt: str, forbidden: list[str], tail: str,
                    max_new: int) -> str | None:
    if not isinstance(prompt, str) or not (1 <= len(prompt) <= 32):
        return "提示词长度须为 1～32 个字符。"
    if not is_printable_ascii(prompt):
        return "提示词只能包含可打印 ASCII 字符（0x20～0x7E）。"
    if not isinstance(forbidden, list) or not (1 <= len(forbidden) <= 3):
        return "禁用片段须为 1～3 条。"
    seen: set[str] = set()
    for frag in forbidden:
        if not isinstance(frag, str) or not (2 <= len(frag) <= 6):
            return "每条禁用片段长度须为 2～6 个字符。"
        if not is_printable_ascii(frag):
            return "禁用片段只能包含可打印 ASCII 字符。"
        if frag in seen:
            return f"禁用片段重复：{frag!r}。"
        seen.add(frag)
    if not isinstance(tail, str) or not (2 <= len(tail) <= 4):
        return "尾句长度须为 2～4 个字符。"
    if not is_printable_ascii(tail):
        return "尾句只能包含可打印 ASCII 字符。"
    if not isinstance(max_new, int) or not (6 <= max_new <= 20):
        return "新 token 上限须为 6～20。"
    # 提示词已含禁用片段：任何续写都无法消除拼接处的命中
    if any(frag in prompt for frag in forbidden):
        return "提示词本身已包含禁用片段，拼接处必然违规，无解。"
    # 尾句自身含禁用片段：任何成功收尾都必然违规
    if any(frag in tail for frag in forbidden):
        return "尾句自身包含禁用片段，无法以该尾句合法收尾，无解。"
    return None


# ----------------------------------------------------------------------
# 结果结构
# ----------------------------------------------------------------------
@dataclass
class ConstrainedStep:
    index: int                 # 第几个新 token（0 起）
    chosen_id: int
    chosen_text: str
    logit: float
    legal_ids: list[int]       # 本步全部合法候选（通过即时违规与可行性两道门）
    rejected_immediate: list[int]   # 因立刻产生禁用片段而被拒
    rejected_dead_end: list[int]    # 即时合法但剩余步数内无法收尾
    budget_before: int
    forb_state_before: int
    tail_state_before: int
    target_forward_input_len: int  # 本步目标前向输入长度（主路径首步为提示词长，其后为 1）


@dataclass
class ConstrainedResult:
    status: str                # "ok" | "no_solution" | "verify_failed"
    reason: str | None
    prompt: str
    prompt_ids: list[int]
    forbidden: list[str]
    tail: str
    max_new: int
    new_tokens: list[int] = field(default_factory=list)
    steps: list[ConstrainedStep] = field(default_factory=list)
    target_forward_count_main: int = 0
    target_forward_count_reference: int = 0
    feasibility_calls: int = 0
    mismatch_index: int | None = None
    final_check: dict = field(default_factory=dict)

    @property
    def generated_text(self) -> str:
        return tokenizer.decode(self.new_tokens)


# ----------------------------------------------------------------------
# 共享决策器：两条路径调用同一份逻辑，仅「如何取下一位置 logits」不同
# ----------------------------------------------------------------------
def _legal_candidates(logits: torch.Tensor, forb: ForbiddenAutomaton,
                      tail: TailMatcher, feas: Feasibility,
                      forb_state: int, tail_state: int,
                      budget: int) -> tuple[list[int], list[int], list[int]]:
    """返回 (合法候选, 即时违规候选, 死路候选)，id 均按升序。"""
    legal: list[int] = [EOS_ID] if tail_state == tail.L else []
    immediate_bad: list[int] = []
    dead_end: list[int] = []
    for cid in CHAR_IDS:
        nf = forb.step(forb_state, cid)
        if forb.is_match(nf):
            immediate_bad.append(cid)
            continue
        nt = tail.step(tail_state, cid)
        if feas.can_finish(nf, nt, budget - 1):
            legal.append(cid)
        else:
            dead_end.append(cid)
    return legal, immediate_bad, dead_end


def _choose(logits: torch.Tensor, legal: list[int]) -> int:
    """合法候选中 logit 最高，平局取最小 id。

    legal 已按 id 升序（EOS=2 在最前）；对屏蔽后的张量 argmax 天然平局取小。
    """
    masked = torch.full_like(logits, float("-inf"))
    masked[legal] = logits[legal]
    return greedy_argmax(masked)


def _drive(target: CausalTransformer, prompt_ids: list[int],
           forb: ForbiddenAutomaton, tail: TailMatcher,
           max_new: int, use_cache: bool
           ) -> tuple[list[int], list[ConstrainedStep], int, str | None, Feasibility]:
    """受限贪心主循环。use_cache=True 走 KV 增量，False 走无缓存逐步重算。

    两条路径仅前向方式不同，候选/平局/可行性逻辑完全共用。
    """
    feas = Feasibility(forb, tail)
    forb_state = forb.initial_state(prompt_ids)
    tail_state = 0  # 尾句匹配从生成起点开始（约束作用于新文本）
    seq = list(prompt_ids)
    new_tokens: list[int] = []
    steps: list[ConstrainedStep] = []
    no_reason: str | None = None

    if use_cache:
        target.init_cache()
        # 首轮一次性灌入提示词，之后每步仅喂选中的 1 个 token
        first_logits = target.forward(prompt_ids, use_cache=True)[-1]
    forward_count = 0

    for budget in range(max_new, 0, -1):
        if use_cache:
            logits = first_logits if not steps else \
                target.forward([new_tokens[-1]], use_cache=True)[-1]
        else:
            logits = target.forward(seq, use_cache=False)[-1]
        forward_count += 1

        legal, imm, dead = _legal_candidates(
            logits, forb, tail, feas, forb_state, tail_state, budget)
        if not legal:
            no_reason = (
                f"第 {len(new_tokens) + 1} 步无合法 token：剩余 {budget} 个新 token 额度，"
                f"尾句匹配 {tail_state}/{tail.L}，所有候选或立刻命中禁用片段，"
                f"或无法在剩余步数内以尾句 {tail!r} 紧接 EOS 收尾。"
            )
            break
        chosen = _choose(logits, legal)
        steps.append(ConstrainedStep(
            index=len(new_tokens),
            chosen_id=int(chosen),
            chosen_text=tokenizer.render_token(int(chosen)),
            logit=float(logits[chosen].item()),
            legal_ids=list(legal),
            rejected_immediate=list(imm),
            rejected_dead_end=list(dead),
            budget_before=budget,
            forb_state_before=forb_state,
            tail_state_before=tail_state,
            target_forward_input_len=(len(prompt_ids) if not steps else 1)
            if use_cache else len(seq),
        ))
        new_tokens.append(int(chosen))
        if chosen == EOS_ID:
            break
        cid = int(chosen)
        forb_state = forb.step(forb_state, cid)
        tail_state = tail.step(tail_state, cid)
        seq.append(cid)
    else:
        # 正常耗尽预算却未发 EOS：防御性分支（可行性门控理论上不会让其发生）
        no_reason = f"达到 {max_new} 个新 token 上限仍未能以尾句紧接 EOS 结束。"

    return new_tokens, steps, forward_count, no_reason, feas


# ----------------------------------------------------------------------
# 对外入口：主路径（KV）+ 独立无缓存参考路径，逐 token 核对后才放行
# ----------------------------------------------------------------------
class _ModelSnapshot:
    """保存/恢复目标模型的 KV 缓存、缓存长度与前向计数（权重全程只读）。"""

    def __init__(self, target: CausalTransformer):
        self.target = target
        self.cache = target._cache
        self.cache_len = target.cache_len
        self.forward_count = target.forward_count

    def restore(self) -> None:
        t = self.target
        t._cache = self.cache
        t.cache_len = self.cache_len
        t.forward_count = self.forward_count


def constrained_generate(target: CausalTransformer, prompt_text: str,
                         forbidden: list[str], tail: str,
                         max_new: int) -> ConstrainedResult:
    err = validate_inputs(prompt_text, forbidden, tail, max_new)
    if err is not None:
        raise ValueError(err)
    prompt_ids = tokenizer.encode(prompt_text)
    forb = ForbiddenAutomaton(forbidden)
    tail_m = TailMatcher(tail)
    result = ConstrainedResult(
        status="ok", reason=None, prompt=prompt_text,
        prompt_ids=list(prompt_ids), forbidden=list(forbidden),
        tail=tail, max_new=max_new,
    )

    snap = _ModelSnapshot(target)
    try:
        with torch.no_grad():
            # ---- 主路径：复用 KV，按实际选中 token 逐 token 增量续算 ----
            main_tokens, main_steps, main_fwd, main_reason, feas = _drive(
                target, prompt_ids, forb, tail_m, max_new, use_cache=True)
            # ---- 参考路径：独立无缓存，逐步对完整前缀重算 ----
            ref_tokens, _ref_steps, ref_fwd, ref_reason, _ = _drive(
                target, prompt_ids, forb, tail_m, max_new, use_cache=False)
    finally:
        # 无论成功失败，模型缓存与计数器必须还原，权重全程只读
        snap.restore()

    result.target_forward_count_main = main_fwd
    result.target_forward_count_reference = ref_fwd
    result.feasibility_calls = feas.calls

    # ---- 无解判定（两条路径必须一致地无解）----
    if main_reason is not None or ref_reason is not None:
        if main_reason != ref_reason or main_tokens != ref_tokens:
            result.status = "verify_failed"
            result.reason = (
                "主路径与无缓存参考路径的无解判定不一致，拒绝展示任何部分结论。"
                f" 主路径：{main_reason or '有解'}；参考路径：{ref_reason or '有解'}。"
            )
            return result
        result.status = "no_solution"
        result.reason = main_reason
        return result

    # ---- 逐 token 核对：完全一致才展示 ----
    mismatch: int | None = None
    for i, (a, b) in enumerate(zip(main_tokens, ref_tokens)):
        if a != b:
            mismatch = i
            break
    if mismatch is None and len(main_tokens) != len(ref_tokens):
        mismatch = min(len(main_tokens), len(ref_tokens))
    if mismatch is not None:
        result.status = "verify_failed"
        result.mismatch_index = mismatch
        a = main_tokens[mismatch] if mismatch < len(main_tokens) else None
        b = ref_tokens[mismatch] if mismatch < len(ref_tokens) else None
        result.reason = (
            f"KV 缓存主路径与独立无缓存逐步参考在新 token 第 {mismatch} 个出现分歧"
            f"（主路径={_fmt(a)}，参考={_fmt(b)}），按约定不展示续写。"
        )
        return result

    # ---- 终态合规复核（独立于解码过程的最后一道检查）----
    check = _final_check(prompt_text, main_tokens, forbidden, tail, max_new)
    result.final_check = check
    if not check["passed"]:
        result.status = "verify_failed"
        result.reason = "终态合规复核失败：" + "；".join(check["violations"])
        return result

    result.new_tokens = main_tokens
    result.steps = main_steps
    result.status = "ok"
    return result


def _fmt(token_id: int | None) -> str:
    if token_id is None:
        return "∅"
    return f"{tokenizer.render_token(token_id)}(id {token_id})"


def _final_check(prompt_text: str, new_tokens: list[int],
                 forbidden: list[str], tail: str, max_new: int) -> dict:
    """对最终产物做独立字符串级复核。"""
    violations: list[str] = []
    gen = tokenizer.decode(new_tokens)
    joined = prompt_text + gen
    for frag in forbidden:
        if frag in gen:
            violations.append(f"生成文本出现禁用片段 {frag!r}")
        if frag in joined:
            violations.append(f"提示词拼接处出现禁用片段 {frag!r}")
    if not new_tokens or new_tokens[-1] != EOS_ID:
        violations.append("新文本未以 EOS 结束")
    if not gen.endswith(tail):
        violations.append(f"新文本未以尾句 {tail!r} 收尾")
    if new_tokens and new_tokens[-1] == EOS_ID:
        before_eos = tokenizer.decode(new_tokens[:-1])
        if not before_eos.endswith(tail):
            violations.append("EOS 未紧接在尾句之后")
    if len(new_tokens) > max_new:
        violations.append(f"新 token 数 {len(new_tokens)} 超过上限 {max_new}")
    if EOS_ID in new_tokens[:-1]:
        violations.append("EOS 出现在末尾之前")
    return {"passed": not violations, "violations": violations,
            "generated_len": len(gen), "token_len": len(new_tokens)}
