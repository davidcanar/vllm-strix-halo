# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Determinism regression tests for the Python ``grouped_topk`` fallback.

Intended to live in ``tests/kernels/moe/test_grouped_topk.py``; the imports
below duplicate the ones at the top of that file, so drop them when merging.
The file also runs standalone:

    pytest tests/kernels/moe/test_grouped_topk_determinism.py

Every existing test in ``test_grouped_topk.py`` is gated with
``@pytest.mark.skipif(not current_platform.is_cuda(), ...)`` because it
exercises the fused CUDA kernel. These tests deliberately are NOT gated: the
code under test is the pure-PyTorch fallback in
``grouped_topk_router.py::grouped_topk``, which is the live path on ROCm
(whenever AITER MoE is off) and is also reached from CUDA whenever the fused
kernel is not taken (``VLLM_USE_FUSED_MOE_GROUPED_TOPK=0``,
``e_score_correction_bias is None``, or a shape outside the kernel's tier
table). The fused-kernel gate is forced closed below, so the same Python
code runs on CUDA, ROCm and CPU CI alike.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    grouped_topk,
)

# The fallback under test is device-agnostic; run on whatever this runner has.
# (On ROCm builds torch.cuda.is_available() is True and "cuda" is the right
# device string -- HIP presents itself through the CUDA API.)
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_NUM_REPEAT = 20


def _run_python_grouped_topk(
    logits: torch.Tensor,
    bias: torch.Tensor | None,
    topk: int,
    *,
    num_expert_group: int = 1,
    topk_group: int = 1,
    scoring_func: str = "sigmoid",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Call the Python ``grouped_topk`` exactly as the router does when the
    fused kernel is not taken."""
    return grouped_topk(
        hidden_states=torch.empty(
            (logits.shape[0], 0), dtype=logits.dtype, device=logits.device
        ),
        gating_output=logits,
        topk=topk,
        renormalize=False,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        scoring_func=scoring_func,
        e_score_correction_bias=bias,
    )


def _scores(logits: torch.Tensor, scoring_func: str) -> torch.Tensor:
    if scoring_func == "sigmoid":
        return logits.sigmoid()
    return torch.softmax(logits, dim=-1)


def _biased_scores(
    logits: torch.Tensor, bias: torch.Tensor | None, scoring_func: str
) -> torch.Tensor:
    scores = _scores(logits, scoring_func)
    if bias is not None:
        scores = scores + bias.unsqueeze(0)
    return scores


def _make_tie_logits(
    num_experts: int, k: int, tie_lo: int, tie_hi: int
) -> torch.Tensor:
    """One row with a deliberate, bitwise-exact tie at the k-boundary.

    Experts ``0 .. k-2`` get well-separated descending logits, experts
    ``tie_lo`` (= k-1) and ``tie_hi`` get the *same* logit value, and every
    other expert sits strictly below the pair. Because the tied pair has
    identical input bits and the scoring activation is elementwise, the two
    biased scores are bitwise-equal on every backend, eager or compiled --
    the tie is constructed, not a rounding accident. The pair occupies
    positions k-1 and k of the value-descending order, i.e. the last selected
    slot and the first dropped one.
    """
    assert tie_lo == k - 1 and tie_hi > tie_lo and tie_hi < num_experts
    logits = torch.zeros(num_experts, dtype=torch.float32)
    logits[: k - 1] = torch.linspace(8.0, 4.0, k - 1)
    logits[tie_lo] = 3.0
    logits[tie_hi] = 3.0
    rest = torch.ones(num_experts, dtype=torch.bool)
    rest[: k - 1] = False
    rest[tie_lo] = False
    rest[tie_hi] = False
    logits[rest] = torch.linspace(2.9, 0.1, int(rest.sum()))
    return logits


@pytest.mark.parametrize("scoring_func", ["sigmoid", "softmax"])
@pytest.mark.parametrize("bias_is_none", [False, True])
def test_grouped_topk_single_group_deterministic_ties(
    monkeypatch: pytest.MonkeyPatch, scoring_func: str, bias_is_none: bool
):
    # Force the Python path on every platform (including CUDA) and make it
    # explicit that determinism must not require VLLM_BATCH_INVARIANT, which
    # is documented NVIDIA-SM90-only.
    monkeypatch.setenv("VLLM_USE_FUSED_MOE_GROUPED_TOPK", "0")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")

    num_experts, topk, tie_lo, tie_hi = 32, 8, 7, 21
    logits = _make_tie_logits(num_experts, topk, tie_lo, tie_hi)[None].to(
        _DEVICE
    )
    bias = None if bias_is_none else torch.zeros(num_experts, device=_DEVICE)

    biased = _biased_scores(logits, bias, scoring_func)
    # Self-validate the construction before asserting anything about the op.
    tie_val = biased[0, tie_lo]
    assert biased[0, tie_lo].item() == biased[0, tie_hi].item()
    assert int((biased[0] > tie_val).sum()) == topk - 1
    assert int((biased[0] == tie_val).sum()) == 2

    # (a) Repeat determinism: identical inputs (fresh clone per call so the
    # input buffer address varies, as it does in a real forward) must give
    # bitwise-identical outputs.
    results = [
        _run_python_grouped_topk(
            logits.clone(), bias, topk, scoring_func=scoring_func
        )
        for _ in range(_NUM_REPEAT)
    ]
    first_w, first_i = results[0]
    for w, i in results[1:]:
        assert torch.equal(i, first_i)
        # Bitwise, not merely value-wise: all scores are strictly positive,
        # and comparing the raw bits also catches -0.0/0.0 flips.
        assert w.view(torch.int32).equal(first_w.view(torch.int32))

    # (b) The exact tie at the k-boundary is won by the lower expert index,
    # matching the fused kernel's contract (value desc, index asc).
    assert int(first_i[0, topk - 1]) == tie_lo
    assert tie_hi not in first_i[0].tolist()

    # The full selection is the value-descending one: experts 0..k-2 by
    # construction, then the tie winner tie_lo = k-1.
    expected_ids = torch.arange(topk, dtype=torch.int32, device=_DEVICE)[None]
    torch.testing.assert_close(first_i, expected_ids)

    # (c) The returned order is value-descending (ties allowed to repeat).
    order = biased.gather(1, first_i.to(torch.long))
    assert torch.all(order[:, :-1] >= order[:, 1:])

    # Weights are the unbiased scores of the selected experts. Allow a small
    # tolerance vs the eager reference: the compiled elementwise activation
    # may differ from eager by an ULP on some backends.
    expected_w = _scores(logits, scoring_func).gather(
        1, first_i.to(torch.long)
    )
    torch.testing.assert_close(first_w, expected_w, atol=2e-6, rtol=0)


@pytest.mark.parametrize("scoring_func", ["sigmoid", "softmax"])
@pytest.mark.parametrize("bias_is_none", [False, True])
def test_grouped_topk_single_group_tie_free_control(
    monkeypatch: pytest.MonkeyPatch, scoring_func: str, bias_is_none: bool
):
    monkeypatch.setenv("VLLM_USE_FUSED_MOE_GROUPED_TOPK", "0")

    num_experts, topk, tie_lo, tie_hi = 32, 8, 7, 21
    logits = _make_tie_logits(num_experts, topk, tie_lo, tie_hi)[None].to(
        _DEVICE
    )
    # Break the tie: the boundary value now has a unique owner, so the top-k
    # has exactly one correct answer.
    logits[0, tie_hi] = 2.95
    bias = None if bias_is_none else torch.zeros(num_experts, device=_DEVICE)

    biased = _biased_scores(logits, bias, scoring_func)
    assert int((biased[0] == biased[0, tie_lo]).sum()) == 1

    results = [
        _run_python_grouped_topk(
            logits.clone(), bias, topk, scoring_func=scoring_func
        )
        for _ in range(_NUM_REPEAT)
    ]
    first_w, first_i = results[0]
    for w, i in results[1:]:
        assert torch.equal(i, first_i)
        assert w.view(torch.int32).equal(first_w.view(torch.int32))

    expected_ids = torch.arange(topk, dtype=torch.int32, device=_DEVICE)[None]
    torch.testing.assert_close(first_i, expected_ids)

    # Control: with no tie, the deterministic selection is exactly what
    # torch.topk(..., sorted=True) already returns, i.e. the fix changes
    # nothing on tie-free input.
    ref_ids = biased.topk(topk, dim=-1, sorted=True)[1].to(torch.int32)
    torch.testing.assert_close(first_i, ref_ids)


@pytest.mark.parametrize("scoring_func", ["sigmoid", "softmax"])
def test_grouped_topk_multi_group_deterministic_group_tie(
    monkeypatch: pytest.MonkeyPatch, scoring_func: str
):
    """A bitwise-exact tie at the *group* boundary (topk_group > 1) is also
    broken by ascending group index."""
    monkeypatch.setenv("VLLM_USE_FUSED_MOE_GROUPED_TOPK", "0")

    # 4 groups of 8 experts. Group 0 is clearly best (its top-2 dominate),
    # groups 1 and 2 are bitwise-identical (a deliberate exact tie for the
    # second of the two selected groups), group 3 is clearly worst. The
    # non-top logits inside each group are kept low so that the top-4
    # individuals of the union {group 0, group 1} are exactly experts
    # 0, 1, 8, 9 -- had group 2 won the tie, the ids would be 0, 1, 16, 17.
    g0 = [8.0, 7.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4]
    g12 = [4.0, 3.5, 1.3, 1.2, 1.1, 1.0, 0.95, 0.9]
    g3 = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    logits = torch.tensor(g0 + g12 + g12 + g3, dtype=torch.float32)[None].to(
        _DEVICE
    )
    num_experts = logits.shape[1]
    bias = torch.zeros(num_experts, device=_DEVICE)

    # Group score (top-2 sum within the group) of groups 1 and 2 must be
    # bitwise-equal: identical inputs, elementwise activation, values-only
    # reduction. (Ties inside this top-2-by-value reduction are between
    # equal values, so the sum is unaffected -- which is why that reduction
    # itself needs no tie-break.)
    def group_scores(row: torch.Tensor) -> torch.Tensor:
        return row.view(1, 4, -1).topk(2, dim=-1)[0].sum(dim=-1)

    gs = group_scores(_biased_scores(logits, bias, scoring_func))
    assert gs[0, 1].item() == gs[0, 2].item()
    assert gs[0, 0] > gs[0, 1] and gs[0, 3] < gs[0, 1]

    results = [
        _run_python_grouped_topk(
            logits.clone(),
            bias,
            4,
            num_expert_group=4,
            topk_group=2,
            scoring_func=scoring_func,
        )
        for _ in range(_NUM_REPEAT)
    ]
    first_w, first_i = results[0]
    for w, i in results[1:]:
        assert torch.equal(i, first_i)
        assert w.view(torch.int32).equal(first_w.view(torch.int32))

    # Group 1 (lower index) wins the group tie, so the experts are drawn
    # from groups 0 and 1 only -- had group 2 won, ids would be 16/17.
    expected_ids = torch.tensor(
        [[0, 1, 8, 9]], dtype=torch.int32, device=_DEVICE
    )
    torch.testing.assert_close(first_i, expected_ids)

    order = _biased_scores(logits, bias, scoring_func).gather(
        1, first_i.to(torch.long)
    )
    assert torch.all(order[:, :-1] >= order[:, 1:])
