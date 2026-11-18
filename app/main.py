"""FastAPI 后端：推测解码 vs 无缓存贪心基线的对照服务，
以及受限续写（禁用片段规避 + 指定尾句收尾）验证服务。"""

from __future__ import annotations

import os
import threading
from dataclasses import asdict

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import tokenizer
from .constrained import constrained_generate, validate_inputs
from .decoding import compare
from .model import get_models

app = FastAPI(title="Speculative Decoding Verifier")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")

# 模型为进程内单例且两条解码路径都会改写 KV 缓存/前向计数，
# 用进程级锁串行化所有推理请求，避免并发请求互相污染缓存状态。
_MODEL_LOCK = threading.Lock()

# 候选提示词（纯 ASCII，固定词表全部可编码）；启动时扫描出
# 分别触发"拒绝"和"整块接受"的两条预设。
_CANDIDATE_PROMPTS = [
    "the quick brown fox jumps over the lazy dog while ",
    "hello world, this is a tiny transformer demo of ",
    "speculative decoding uses a small draft model to ",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ",
    "in the beginning there was only silence and then ",
    "print(\"sum =\")  # ",
    "the answer to life the universe and everything is ",
    "one two three four five six seven eight nine ten ",
    "python java rust go c plus plus ruby swift scala ",
    "a stitch in time saves nine, and early birds get ",
    "the rain in spain stays mainly on the plain while ",
    "to be or not to be that is the question whether ",
    "x = 1\ny = 2\nz = x + y\nprint(z)\n# output is ",
    "all work and no play makes jack a dull boy, so ",
    "once upon a time in a small village lived an old ",
    "the capital of france is paris and the capital of ",
    "0 1 1 2 3 5 8 13 21 34 55 89 144 233 next number ",
    "she sells sea shells by the sea shore, the shells ",
    "machine learning models can be compressed with ",
    "alpha beta gamma delta epsilon zeta eta theta iota ",
]

# 固定预设：经实测与测试套件锁定，分别稳定触发"拒绝"与"整块接受"。
# 拒绝预设：草稿与目标在该上下文上分歧，首轮接受后持续被拒，
# 用于验证"被拒草稿不会污染后续生成"；
# 整块接受预设：每轮草稿 token 全部通过，展示目标前向节省的上限。
PRESET_REJECT = "python java rust go c plus plus ruby swift scala "
PRESET_ALL_ACCEPTED = "the quick brown fox jumps over the lazy dog while "


class DecodeRequest(BaseModel):
    prompt: str = Field(default="", max_length=200)
    k: int = Field(default=3, ge=1, le=4)
    max_new: int = Field(default=32, ge=1, le=32)


class ConstrainedRequest(BaseModel):
    # 语义校验（长度区间、可打印 ASCII、重复、前置无解等）在
    # constrained.validate_inputs 内完成并返回 400 + 中文原因；
    # Pydantic 只做最外层形状兜底。
    prompt: str = Field(default="", max_length=64)
    forbidden: list[str] = Field(default_factory=list, max_length=3)
    tail: str = Field(default="", max_length=8)
    max_new: int = Field(default=12, ge=1, le=64)


def _token_view(ids: list[int]) -> list[dict]:
    return [{"id": int(t), "text": tokenizer.render_token(int(t))} for t in ids]


def _serialize(cmp_result, prompt_text: str) -> dict:
    base, spec = cmp_result.baseline, cmp_result.speculative
    return {
        "prompt": prompt_text,
        "prompt_ids": base.prompt_ids,
        "match": cmp_result.match,
        "status": "PASS" if cmp_result.match else "FAIL",
        "first_mismatch_index": cmp_result.first_mismatch_index,
        "baseline": {
            "new_tokens": _token_view(base.new_tokens),
            "text": tokenizer.decode(base.new_tokens),
            "target_forward_count": base.target_forward_count,
            "stopped_by_eos": base.stopped_by_eos,
        },
        "speculative": {
            "new_tokens": _token_view(spec.new_tokens),
            "text": tokenizer.decode(spec.new_tokens),
            "target_forward_count": spec.target_forward_count,
            "draft_forward_count": spec.draft_forward_count,
            "stopped_by_eos": spec.stopped_by_eos,
            "rounds": [asdict(r) | {
                "draft_tokens_view": _token_view(r.draft_tokens),
                "target_predictions_view": _token_view(r.target_predictions),
                "target_tokens_view": _token_view(r.target_tokens),
            } for r in spec.rounds],
        },
        "savings": {
            "baseline_forwards": base.target_forward_count,
            "spec_forwards": spec.target_forward_count,
            "saved": base.target_forward_count - spec.target_forward_count,
        },
    }


@app.get("/api/health")
def health() -> dict:
    draft, target = get_models()
    return {
        "ok": True,
        "vocab_size": tokenizer.VOCAB_SIZE,
        "target_layers": target.n_layers,
        "draft_layers": draft.n_layers,
    }


@app.post("/api/decode")
def decode(req: DecodeRequest) -> JSONResponse:
    prompt_ids = tokenizer.encode(req.prompt)
    if not prompt_ids:
        return JSONResponse(
            status_code=400,
            content={"error": "提示词编码后为空，请输入可打印 ASCII 字符。"},
        )
    draft, target = get_models()
    with _MODEL_LOCK:
        result = compare(draft, target, prompt_ids, req.k, req.max_new)
    return JSONResponse(_serialize(result, req.prompt))


def _constrained_token_view(ids: list[int]) -> list[dict]:
    return [{"id": int(t), "text": tokenizer.render_token(int(t))} for t in ids]


def _serialize_constrained(res) -> dict:
    return {
        "status": res.status,
        "reason": res.reason,
        "prompt": res.prompt,
        "prompt_ids": res.prompt_ids,
        "forbidden": res.forbidden,
        "tail": res.tail,
        "max_new": res.max_new,
        "generated_text": res.generated_text,
        "joined_text": res.prompt + res.generated_text,
        "new_tokens": _constrained_token_view(res.new_tokens),
        "steps": [asdict(s) | {
            "chosen_view": _constrained_token_view([s.chosen_id])[0],
            "legal_view": _constrained_token_view(s.legal_ids),
            "immediate_view": _constrained_token_view(s.rejected_immediate),
            "dead_end_view": _constrained_token_view(s.rejected_dead_end),
        } for s in res.steps],
        "target_forward_count_main": res.target_forward_count_main,
        "target_forward_count_reference": res.target_forward_count_reference,
        "feasibility_calls": res.feasibility_calls,
        "mismatch_index": res.mismatch_index,
        "final_check": res.final_check,
    }


@app.post("/api/constrained")
def constrained(req: ConstrainedRequest) -> JSONResponse:
    # 输入语义校验失败 → 400；no_solution / verify_failed 是正常业务结果 → 200
    verr = validate_inputs(req.prompt, req.forbidden, req.tail, req.max_new)
    if verr is not None:
        return JSONResponse(status_code=400, content={"error": verr})
    _draft, target = get_models()
    with _MODEL_LOCK:
        res = constrained_generate(
            target, req.prompt, req.forbidden, req.tail, req.max_new)
    return JSONResponse(_serialize_constrained(res))


@app.get("/api/presets")
def presets() -> dict:
    """返回固定的两条预设（由测试套件锁定其判定类型，避免请求时全量扫描）。"""
    return {"reject": PRESET_REJECT, "all_accepted": PRESET_ALL_ACCEPTED}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/constrained")
def constrained_page() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "constrained.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
