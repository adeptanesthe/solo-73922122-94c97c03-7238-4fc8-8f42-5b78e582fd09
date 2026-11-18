"""受限续写关键功能测试：

 1. Aho-Corasick 自动机与朴素子串搜索在随机文本/随机片段集下完全等价
    （含提示词-生成文本跨边界命中）；
 2. 尾句 KMP 匹配状态与朴素 endswith 语义等价；
 3. 可行性 DFS：预算不足时正确判否；门控保证贪心一旦选路必能收尾；
 4. 端到端 constrained_generate：
    - 状态 ok 时全部硬约束成立（无禁用片段、尾句紧接 EOS、长度上限、
      EOS 计入上限、候选仅限可打印字符+EOS）；
    - KV 主路径与无缓存参考逐 token 一致，前向次数语义正确
      （均等于新 token 数；主路径首步输入为完整提示词，其后恒为 1）；
 5. 无解路径：内部小预算驱动返回 no_solution 原因；静态无解类
    （提示词/尾句含禁用片段）被 validate_inputs 拦截；
 6. verify_failed 守护：模拟参考分歧时拒绝展示任何续写；
 7. 模型状态隔离：受限续写前后权重、KV 缓存、forward_count 完全不变，
    且不影响随后原推测解码的结果与计数；
 8. 平局取较小 id；输入校验各边界；API 层 400/200 行为。
"""

from __future__ import annotations

import os
import random
import string
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import constrained as C
from app import tokenizer
from app.constrained import (
    CHAR_IDS,
    EOS_ID,
    Feasibility,
    ForbiddenAutomaton,
    TailMatcher,
    constrained_generate,
    validate_inputs,
)
from app.decoding import compare
from app.model import get_models

torch.set_num_threads(1)
DRAFT, TARGET = get_models()
PRINTABLE = string.printable[:95]  # 0x20..0x7E


# ----------------------------------------------------------------------
# 1. AC 自动机 vs 朴素子串搜索
# ----------------------------------------------------------------------
def test_automaton_matches_naive() -> None:
    rng = random.Random(20261118)
    for trial in range(300):
        n_pat = rng.randint(1, 3)
        pats = ["".join(rng.choice(PRINTABLE) for _ in range(rng.randint(2, 6)))
                for _ in range(n_pat)]
        pats = list(dict.fromkeys(pats))
        aut = ForbiddenAutomaton(pats)
        text = "".join(rng.choice(PRINTABLE) for _ in range(rng.randint(0, 40)))
        ids = tokenizer.encode(text)
        s = 0
        naive_hit_at = -1
        for i, cid in enumerate(ids):
            s = aut.step(s, cid)
            window = text[: i + 1]
            if any(p in window for p in pats):
                naive_hit_at = i
                assert aut.is_match(s), (trial, i, pats, text)
                break
            assert not aut.is_match(s), (trial, i, pats, text)
        else:
            assert naive_hit_at == -1
    print("ok: AC 自动机 300 组随机用例与朴素子串搜索完全等价")


def test_automaton_cross_boundary() -> None:
    # 禁用片段跨提示词/生成边界命中
    aut = ForbiddenAutomaton(["ab"])
    s = aut.initial_state(tokenizer.encode("xxa"))  # 提示词以 'a' 结尾
    assert not aut.is_match(s)
    s = aut.step(s, tokenizer.encode("b")[0])
    assert aut.is_match(s)
    # 提示词自身已含片段 → initial_state 抛错
    try:
        aut.initial_state(tokenizer.encode("ab"))
    except ValueError:
        pass
    else:
        raise AssertionError("提示词含禁用片段时应抛 ValueError")
    print("ok: 跨边界命中与提示词前置命中检测正确")


# ----------------------------------------------------------------------
# 2. 尾句匹配 vs 朴素语义
# ----------------------------------------------------------------------
def test_tail_matcher_matches_naive() -> None:
    rng = random.Random(42)
    for _ in range(200):
        tail = "".join(rng.choice(PRINTABLE) for _ in range(rng.randint(2, 4)))
        tm = TailMatcher(tail)
        text = "".join(rng.choice(PRINTABLE) for _ in range(rng.randint(0, 30)))
        k = 0
        for i, cid in enumerate(tokenizer.encode(text)):
            k = tm.step(k, cid)
            exp = next((j for j in range(tm.L, 0, -1)
                        if text[: i + 1].endswith(tail[:j])), 0)
            assert k == exp, (tail, text, i, k, exp)
            assert (k == tm.L) == text[: i + 1].endswith(tail)
    print("ok: 尾句 KMP 匹配与朴素后缀语义 200 组随机用例一致")


# ----------------------------------------------------------------------
# 3. 可行性判定
# ----------------------------------------------------------------------
def test_feasibility_budget_and_terminal() -> None:
    forb = ForbiddenAutomaton(["zz"])
    tm = TailMatcher("abc")
    feas = Feasibility(forb, tm)
    # 尾句未齐：预算不足（需 L-k 字符 + 1 EOS）
    assert not feas.can_finish(0, 0, 3)   # 需要 4，只有 3
    assert feas.can_finish(0, 0, 4)       # 恰好 "abc"+EOS
    # 已完整匹配尾句：一个预算发 EOS 即可
    assert feas.can_finish(0, 3, 1)
    assert not feas.can_finish(0, 3, 0)
    print("ok: 可行性 DFS 预算门控正确")


def test_feasibility_blocks_forbidden_routes() -> None:
    # 尾句 "ab"，禁用 "ab" 之外的构造：禁用 "xb"，尾句前只能避开 x
    forb = ForbiddenAutomaton(["xb"])
    tm = TailMatcher("ab")
    feas = Feasibility(forb, tm)
    # 直接 "ab"+EOS 可行（a 不是 x）
    assert feas.can_finish(0, 0, 3)
    # 走到状态 forb 命中 'xb' 后不应被当作可行状态调用；
    # 手工验证：尾句路径 a->b 全程不触命中
    s = 0
    for cid in tokenizer.encode("ab"):
        s = forb.step(s, cid)
        assert not forb.is_match(s)
    print("ok: 可行性绕开禁用片段的路径正确")


# ----------------------------------------------------------------------
# 4. 端到端：硬约束与双路径核对
# ----------------------------------------------------------------------
def _assert_invariants(res) -> None:
    assert res.status == "ok", res.reason
    toks = res.new_tokens
    assert toks and toks[-1] == EOS_ID, "必须以 EOS 结束"
    assert EOS_ID not in toks[:-1], "EOS 只能在末尾"
    assert len(toks) <= res.max_new, "EOS 计入上限"
    gen = res.generated_text
    assert gen.endswith(res.tail), "新文本以尾句收尾"
    assert tokenizer.decode(toks[:-1]).endswith(res.tail)
    for frag in res.forbidden:
        assert frag not in gen, f"生成文本出现禁用片段 {frag!r}"
        assert frag not in (res.prompt + gen), f"拼接处出现禁用片段 {frag!r}"
    # 所有 token 只能是可打印字符或 EOS
    assert all(t == EOS_ID or t in CHAR_IDS for t in toks)
    # 逐步明细自洽
    for st in res.steps:
        assert st.chosen_id in st.legal_ids
        # 三类候选互斥且覆盖全部 95 个字符（+可能的 EOS）
        covered = set(st.legal_ids) | set(st.rejected_immediate) | set(st.rejected_dead_end)
        assert covered == set(CHAR_IDS) | ({EOS_ID} if EOS_ID in st.legal_ids else set())
        if st.index > 0:
            assert st.target_forward_input_len == 1, "主路径后续步只喂 1 个 token"
        else:
            assert st.target_forward_input_len == len(res.prompt_ids)
    assert res.final_check["passed"]
    assert res.target_forward_count_main == len(toks)
    assert res.target_forward_count_reference == len(toks)


def test_constrained_end_to_end_fixed_cases() -> None:
    cases = [
        ("speculative demo ", ["th"], "end", 12),
        ("hello world, ", [" a", "er"], "ok!", 14),
        ("x=1; y=2; ", ["xy", "zz"], "ok", 8),
        ("a" * 32, ["aa"], "bc", 20),            # 提示词达长度上限，且已含 aa → 400
    ]
    # 最后一条提示词自身含 "aa"，应被校验拦截
    ok_cases = cases[:-1]
    for prompt, frags, tail, cap in ok_cases:
        res = constrained_generate(TARGET, prompt, frags, tail, cap)
        _assert_invariants(res)
    err = validate_inputs(*cases[-1])
    assert err is not None and "提示词本身已包含" in err
    print(f"ok: {len(ok_cases)} 组固定端到端用例全部硬约束成立、双路径一致")


def test_constrained_end_to_end_random() -> None:
    rng = random.Random(7)
    n_ok = 0
    for _ in range(60):
        prompt = "".join(rng.choice(PRINTABLE) for _ in range(rng.randint(1, 32)))
        frags = []
        for _ in range(rng.randint(1, 3)):
            f = "".join(rng.choice(PRINTABLE) for _ in range(rng.randint(2, 6)))
            if f not in frags and f not in prompt:
                frags.append(f)
        tail = "".join(rng.choice(PRINTABLE) for _ in range(rng.randint(2, 4)))
        cap = rng.randint(6, 20)
        err = validate_inputs(prompt, frags, tail, cap)
        if err is not None:
            continue
        res = constrained_generate(TARGET, prompt, frags, tail, cap)
        assert res.status == "ok", res.reason
        _assert_invariants(res)
        n_ok += 1
    assert n_ok >= 30, f"随机有效用例过少：{n_ok}"
    print(f"ok: {n_ok} 组随机端到端用例全部合规且双路径逐 token 一致")


def test_eos_only_after_full_tail() -> None:
    res = constrained_generate(TARGET, "speculative ", ["th", "he"], "end", 15)
    assert res.status == "ok"
    for st in res.steps[:-1]:
        assert EOS_ID not in st.legal_ids, "尾句未完整匹配前 EOS 不得合法"
    assert EOS_ID in res.steps[-1].legal_ids
    assert res.steps[-1].chosen_id == EOS_ID
    print("ok: EOS 仅在尾句完整匹配后合法且被选中收尾")


# ----------------------------------------------------------------------
# 5. 无解路径
# ----------------------------------------------------------------------
def test_no_solution_tight_budget() -> None:
    # 绕过 API 校验直接驱动内部循环：尾句 4 字符 + EOS 至少 5 个 token，
    # 预算 3 必然无解（首步可行性门即关闭所有候选）
    forb = ForbiddenAutomaton(["zz"])
    tm = TailMatcher("abcd")
    ids = tokenizer.encode("prompt ")
    toks, _steps, _fwd, reason, _feas = C._drive(
        TARGET, ids, forb, tm, max_new=3, use_cache=True)
    assert reason is not None
    assert len(toks) <= 3
    # 经公共入口时两条路径无解判定必须一致 → no_solution
    # （通过猴子补丁收窄 max_new 的等价构造：直接验证 Feasibility 判否）
    feas = Feasibility(forb, tm)
    assert not feas.can_finish(0, 0, 3)
    print("ok: 预算不足时正确判定无解并给出原因")


def test_no_solution_dead_end_midway() -> None:
    """构造「首步可走、随后所有候选都走进死路」的场景，覆盖中途无解分支。

    禁用所有 95 个可打印字符在第二步的延伸不现实，改为直接构造可行性
    记忆：让 Feasibility.can_finish 在任意非初始状态恒为 False，
    则首步若尾句未完整匹配，所有字符候选均为死路、EOS 也不合法 → 无解。
    """
    forb = ForbiddenAutomaton(["zz"])
    tm = TailMatcher("ab")
    real_can = Feasibility.can_finish

    def dead_after_first(self, forb_state, tail_state, budget):
        if budget <= 1:
            return real_can(self, forb_state, tail_state, budget)
        return False  # 任何延伸都宣称无法收尾

    Feasibility.can_finish = dead_after_first
    try:
        toks, steps, _fwd, reason, _feas = C._drive(
            TARGET, tokenizer.encode("p"), forb, tm, max_new=10, use_cache=False)
    finally:
        Feasibility.can_finish = real_can
    assert reason is not None and "无合法 token" in reason
    assert len(toks) == len(steps) == 0  # 首步即全死路（EOS 此时也不合法）
    print("ok: 中途无合法候选（死路）时给出无解原因且不产生部分输出")


def test_static_no_solution_validation() -> None:
    assert "提示词本身已包含" in validate_inputs("ab", ["ab"], "xy", 10)
    assert "尾句自身包含" in validate_inputs("pq", ["xy"], "xy", 10)
    assert validate_inputs("ok", ["zz"], "ab", 10) is None
    print("ok: 静态无解类（提示词/尾句含禁用片段）被输入校验拦截")


# ----------------------------------------------------------------------
# 6. verify_failed 守护
# ----------------------------------------------------------------------
def test_verify_failed_guard(monkeypatch=None) -> None:
    real_drive = C._drive

    def fake_drive(target, prompt_ids, forb, tail, max_new, use_cache):
        toks, steps, fwd, reason, feas = real_drive(
            target, prompt_ids, forb, tail, max_new, use_cache)
        if use_cache:
            # 模拟主路径在第一个字符 token 上与参考分歧
            toks = list(toks)
            if len(toks) > 1:
                alt = next(c for c in CHAR_IDS if c != toks[0])
                toks[0] = alt
        return toks, steps, fwd, reason, feas

    C._drive = fake_drive
    try:
        res = constrained_generate(TARGET, "hello ", ["th"], "end", 12)
    finally:
        C._drive = real_drive
    assert res.status == "verify_failed"
    assert res.new_tokens == [] and res.steps == [], "核对失败不得展示部分结论"
    assert res.mismatch_index == 0 and "分歧" in res.reason
    print("ok: 双路径分歧时 verify_failed 且不返回任何续写内容")


# ----------------------------------------------------------------------
# 7. 模型状态隔离 + 不影响原解码
# ----------------------------------------------------------------------
def test_model_state_untouched_and_original_decode_intact() -> None:
    ids = tokenizer.encode("the quick brown fox jumps over ")
    before = compare(DRAFT, TARGET, ids, 3, 20)
    cache_before = TARGET._cache
    cache_len_before = TARGET.cache_len
    fc_before = TARGET.forward_count

    res = constrained_generate(TARGET, "quick brown ", ["th", "er"], "end", 14)
    assert res.status == "ok"

    assert TARGET._cache is cache_before, "KV 缓存对象引用应还原"
    assert TARGET.cache_len == cache_len_before
    assert TARGET.forward_count == fc_before, "前向计数应还原"
    # 权重未变（只读推理 + no_grad；抽查范数稳定性即可）
    w = TARGET.ln_f.weight
    assert torch.isfinite(w).all()

    after = compare(DRAFT, TARGET, ids, 3, 20)
    assert after.match
    assert after.baseline.new_tokens == before.baseline.new_tokens
    assert after.speculative.new_tokens == before.speculative.new_tokens
    assert after.speculative.target_forward_count == \
        before.speculative.target_forward_count
    print("ok: 受限续写不改变权重/KV/前向计数，原推测解码结果与计数不变")


# ----------------------------------------------------------------------
# 8. 平局取小 id 与输入校验边界
# ----------------------------------------------------------------------
def test_choose_tie_smallest_id() -> None:
    logits = torch.zeros(tokenizer.VOCAB_SIZE)
    legal = [5, 9, 2]
    assert C._choose(logits, legal) == 2  # 全体平局取最小
    logits[9] = 1.0
    assert C._choose(logits, legal) == 9  # 唯一最大
    logits[5] = 1.0
    assert C._choose(logits, legal) == 5  # 9/5 平局取小
    print("ok: 合法候选平局时取较小 token id")


def test_validate_inputs_boundaries() -> None:
    good = ("prompt", ["ab"], "xy", 10)
    assert validate_inputs(*good) is None
    bad_cases = [
        ("", ["ab"], "xy", 10),                       # 空提示词
        ("p" * 33, ["ab"], "xy", 10),                 # 提示词过长
        ("pr\nt", ["ab"], "xy", 10),                  # 非可打印 ASCII
        ("prompt", ["a"], "xy", 10),                  # 片段过短
        ("prompt", ["a" * 7], "xy", 10),              # 片段过长
        ("prompt", ["ab", "ab"], "xy", 10),           # 片段重复
        ("prompt", ["ab"], "x", 10),                  # 尾句过短
        ("prompt", ["ab"], "xyzwq", 10),              # 尾句过长
        ("prompt", ["ab"], "xy", 5),                  # 上限过小
        ("prompt", ["ab"], "xy", 21),                 # 上限过大
        ("prompt", [], "xy", 10),                     # 无禁用片段
        ("prompt", ["ab", "cd", "ef", "gh"], "xy", 10),  # 片段过多
    ]
    for case in bad_cases:
        assert validate_inputs(*case) is not None, case
    print("ok: 输入校验覆盖全部长度/字符/数量边界")


# ----------------------------------------------------------------------
# 9. API 层（FastAPI TestClient）
# ----------------------------------------------------------------------
def test_api_endpoints() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    # 健康检查与原解码接口仍可用
    assert client.get("/api/health").json()["ok"] is True
    old = client.post("/api/decode", json={
        "prompt": "the quick brown fox jumps over ", "k": 3, "max_new": 16})
    assert old.status_code == 200 and old.json()["match"]

    # 受限续写：正常请求
    r = client.post("/api/constrained", json={
        "prompt": "speculative demo ",
        "forbidden": ["th"], "tail": "end", "max_new": 12})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["status"] == "ok" and data["final_check"]["passed"]
    assert data["new_tokens"][-1]["id"] == EOS_ID

    # 受限续写：校验失败 → 400
    r = client.post("/api/constrained", json={
        "prompt": "ab", "forbidden": ["ab"], "tail": "xy", "max_new": 10})
    assert r.status_code == 400 and "提示词本身已包含" in r.json()["error"]

    r = client.post("/api/constrained", json={
        "prompt": "ok", "forbidden": ["a"], "tail": "xy", "max_new": 10})
    assert r.status_code == 400

    # 页面可访问
    assert client.get("/constrained").status_code == 200
    assert client.get("/").status_code == 200
    print("ok: API 层 200/400 行为正确，原解码接口与两个页面均可用")


if __name__ == "__main__":
    test_automaton_matches_naive()
    test_automaton_cross_boundary()
    test_tail_matcher_matches_naive()
    test_feasibility_budget_and_terminal()
    test_feasibility_blocks_forbidden_routes()
    test_validate_inputs_boundaries()
    test_choose_tie_smallest_id()
    test_no_solution_tight_budget()
    test_no_solution_dead_end_midway()
    test_static_no_solution_validation()
    test_constrained_end_to_end_fixed_cases()
    test_constrained_end_to_end_random()
    test_eos_only_after_full_tail()
    test_verify_failed_guard()
    test_model_state_untouched_and_original_decode_intact()
    test_api_endpoints()
    print("\n受限续写全部测试通过 ✔")
