from dataclasses import dataclass

from custom_components.ha_ragent.src.homeassistant.helpers.retrieval_helper import RetrievalHelper
from custom_components.ha_ragent.src.models.device import Device
from custom_components.ha_ragent.src.models.scored_result import ScoredResult
from custom_components.ha_ragent.src.models.tool import LlmTool
from custom_components.ha_ragent.src.models.tool_metadata import ToolMetadata


def test_unrelated_history_is_not_embedded() -> None:
    text = RetrievalHelper.build_retrieval_text("turn on the kitchen lights")

    assert text == "turn on the kitchen lights"


@dataclass
class Candidate:
    name: str


def test_exact_match_can_recover_candidate_outside_vector_results() -> None:
    candidates = [
        Candidate("Bedroom lamp"),
        Candidate("Kitchen ceiling light"),
        Candidate("Patio light"),
    ]

    result = RetrievalHelper.rank_scored_candidates(
        [
            ScoredResult(candidates[0], 0.9, 1),
            ScoredResult(candidates[2], 0.8, 2),
        ],
        candidates,
        "turn on the kitchen ceiling light",
        lambda candidate: candidate.name,
        lambda candidate: (candidate.name,),
        2,
    )

    assert result[0] == candidates[1]
    assert len(result) == 2


def test_fuzzy_match_is_fused_with_vector_rank() -> None:
    candidates = [Candidate("Bedroom lamp"), Candidate("Kitchen ceiling light")]

    result = RetrievalHelper.rank_scored_candidates(
        [ScoredResult(candidates[0], 0.05, 1)],
        candidates,
        "kithen ceiling light",
        lambda candidate: candidate.name,
        lambda candidate: (candidate.name,),
        1,
    )

    assert result == [candidates[1]]


def test_normalization_supports_unicode_without_language_patterns() -> None:
    assert RetrievalHelper._normalize("KÜCHE") == "küche"


def test_reciprocal_rank_fusion_rewards_agreement() -> None:
    scores = RetrievalHelper.reciprocal_rank_fusion(
        (["a", "b"], ["b", "c"], ["b", "a"]),
    )

    assert scores["b"] > scores["a"] > scores["c"]


def test_metadata_rank_breaks_equal_text_match() -> None:
    candidates = [Candidate("sensor"), Candidate("sensor")]

    result = RetrievalHelper.rank_scored_candidates(
        [
            ScoredResult(candidates[0], 0.8, 1),
            ScoredResult(candidates[1], 0.8, 2),
        ],
        candidates,
        "kitchen sensor",
        lambda candidate: str(id(candidate)),
        lambda candidate: (candidate.name,),
        1,
        metadata_score=lambda candidate: 1.0 if candidate is candidates[1] else 0.0,
    )

    assert result == [candidates[1]]


def test_device_compatibility_promotes_matching_tool_schema() -> None:
    device = Device(
        id="light.kitchen",
        friendly_name="Kitchen light",
        area_name="Kitchen",
        floor_name="Ground floor",
        domain=["light"],
    )
    incompatible = LlmTool(
        name="CoverControl",
        description="Control cover",
        parameters={"properties": {"domain": {"enum": ["cover"]}}},
        metadata=ToolMetadata(is_domain_aware=True),
    )
    compatible = LlmTool(
        name="LightControl",
        description="Control light",
        parameters={"properties": {"domain": {"enum": ["light"]}}},
        metadata=ToolMetadata(is_domain_aware=True),
    )

    result = RetrievalHelper.rerank_tools_for_devices(
        [incompatible, compatible],
        [device],
        1,
    )

    assert result == [compatible]


def test_adaptive_candidate_limit_is_bounded() -> None:
    assert RetrievalHelper.adaptive_candidate_limit(4) == 16
    assert RetrievalHelper.adaptive_candidate_limit(100) == 40
    assert RetrievalHelper.adaptive_candidate_limit(0) == 0
