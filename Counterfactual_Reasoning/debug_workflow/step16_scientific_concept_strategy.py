"""Step 16: 测试 ScientificConceptDiscoveryStrategy 的单元逻辑。

验证点:
- normalize_claim 的过滤逻辑（空值、Unknown、缺 action 等）
- render_claim 的输出格式
- build_judge_user_prompt 包含交叉验证指令
- Judgment 新增字段 concept_true_understanding

前置条件:
  - 无外部依赖，纯单元测试
"""

import sys
from pathlib import Path

REPO_DIR = Path("/mnt/shared-storage-user/fengxinshun/AISci/AgentDebug/agdebugger")
sys.path.insert(0, str(REPO_DIR))

from external_agent.schemas import EvidenceBundle, Judgment
from external_agent.strategies import ScientificConceptDiscoveryStrategy, get_strategy


def test_strategy_registration():
    """get_strategy should return ScientificConceptDiscoveryStrategy."""
    print("=" * 60)
    print("Test: strategy registration")
    print("=" * 60)
    strategy = get_strategy("scientific_concept_discovery")
    assert isinstance(strategy, ScientificConceptDiscoveryStrategy), (
        f"Expected ScientificConceptDiscoveryStrategy, got {type(strategy)}"
    )
    assert strategy.name == "scientific_concept_discovery"
    print("[OK]")


def test_normalize_claim_valid():
    """Valid raw dict should produce a Claim."""
    print("\n" + "=" * 60)
    print("Test: normalize_claim with valid input")
    print("=" * 60)
    strategy = ScientificConceptDiscoveryStrategy()
    raw = {
        "scientific_concept": "Le Chatelier's Principle",
        "concept_understanding": "When a system at equilibrium is disturbed, it shifts to counteract the disturbance.",
        "corresponding_action": "Increase temperature to shift equilibrium toward endothermic products.",
        "context_snippet": "Based on Le Chatelier's principle, raising the temperature...",
    }
    claim = strategy.normalize_claim(raw, conversation_id=1, turn_number=0)
    assert claim is not None, "Expected a Claim, got None"
    assert claim.category == "scientific_concept"
    assert claim.text == raw["concept_understanding"]
    assert claim.source_ref == raw["scientific_concept"]
    assert claim.source_type == "scientific_concept"
    assert claim.data.get("corresponding_action") == raw["corresponding_action"]
    print(f"  claim_id: {claim.claim_id}")
    print(f"  category: {claim.category}")
    print(f"  source_ref: {claim.source_ref}")
    print("[OK]")


def test_normalize_claim_empty_concept():
    """Empty scientific_concept should be filtered out."""
    print("\n" + "=" * 60)
    print("Test: normalize_claim filters empty concept")
    print("=" * 60)
    strategy = ScientificConceptDiscoveryStrategy()
    raw = {
        "scientific_concept": "",
        "concept_understanding": "something",
        "corresponding_action": "do something",
    }
    claim = strategy.normalize_claim(raw, conversation_id=1, turn_number=0)
    assert claim is None, f"Expected None for empty concept, got {claim}"
    print("[OK]")


def test_normalize_claim_unknown_concept():
    """scientific_concept == 'Unknown' should be filtered out."""
    print("\n" + "=" * 60)
    print("Test: normalize_claim filters Unknown concept")
    print("=" * 60)
    strategy = ScientificConceptDiscoveryStrategy()
    raw = {
        "scientific_concept": "Unknown",
        "concept_understanding": "something",
        "corresponding_action": "do something",
    }
    claim = strategy.normalize_claim(raw, conversation_id=1, turn_number=0)
    assert claim is None, f"Expected None for Unknown concept, got {claim}"
    print("[OK]")


def test_normalize_claim_missing_action():
    """Missing corresponding_action should be filtered out."""
    print("\n" + "=" * 60)
    print("Test: normalize_claim filters missing action")
    print("=" * 60)
    strategy = ScientificConceptDiscoveryStrategy()
    raw = {
        "scientific_concept": "Ohm's Law",
        "concept_understanding": "V = IR",
        "corresponding_action": "",
    }
    claim = strategy.normalize_claim(raw, conversation_id=1, turn_number=0)
    assert claim is None, f"Expected None for empty action, got {claim}"
    print("[OK]")


def test_render_claim():
    """render_claim should produce the expected format."""
    print("\n" + "=" * 60)
    print("Test: render_claim output format")
    print("=" * 60)
    strategy = ScientificConceptDiscoveryStrategy()
    raw = {
        "scientific_concept": "Ideal Gas Law",
        "concept_understanding": "PV = nRT relates pressure, volume, and temperature.",
        "corresponding_action": "Calculate the pressure at the new temperature.",
        "context_snippet": "Using the ideal gas law, we can find...",
    }
    claim = strategy.normalize_claim(raw, conversation_id=1, turn_number=0)
    assert claim is not None
    rendered = strategy.render_claim(claim)
    assert "Scientific Concept: Ideal Gas Law" in rendered
    assert "Agent's Understanding: PV = nRT" in rendered
    assert "Corresponding Action: Calculate the pressure" in rendered
    assert "Original Context: Using the ideal gas law" in rendered
    print(f"  Rendered:\n{rendered}")
    print("[OK]")


def test_build_judge_user_prompt():
    """build_judge_user_prompt should include cross-validation instructions."""
    print("\n" + "=" * 60)
    print("Test: build_judge_user_prompt cross-validation instructions")
    print("=" * 60)
    strategy = ScientificConceptDiscoveryStrategy()
    raw = {
        "scientific_concept": "Newton's Third Law",
        "concept_understanding": "Every action has an equal and opposite reaction.",
        "corresponding_action": "Apply equal force in opposite direction.",
        "context_snippet": "By Newton's third law...",
    }
    claim = strategy.normalize_claim(raw, conversation_id=1, turn_number=0)
    assert claim is not None
    evidence = EvidenceBundle(
        search_results="Source 1: ... Source 2: ...",
        filtered_content="Filtered evidence text",
    )
    prompt = strategy.build_judge_user_prompt(claim, evidence)
    assert "Cross-validate using MULTIPLE independent sources" in prompt
    assert "mark hallucination as N/A" in prompt
    assert "concept_true_understanding" in prompt
    assert "Newton's Third Law" in prompt
    assert "Source 1:" in prompt
    assert "Filtered evidence text" in prompt
    print(f"  Prompt length: {len(prompt)} chars")
    print("[OK]")


def test_judge_system_prompt():
    """Judge system prompt should mention concept_true_understanding field."""
    print("\n" + "=" * 60)
    print("Test: judge_system_prompt contains required fields")
    print("=" * 60)
    strategy = ScientificConceptDiscoveryStrategy()
    prompt = strategy.judge_system_prompt
    assert "concept_true_understanding" in prompt
    assert "reference_name" in prompt
    assert "reference_grounding" in prompt
    assert "hallucination" in prompt
    print("[OK]")


def test_judgment_concept_true_understanding_field():
    """Judgment dataclass should have concept_true_understanding field."""
    print("\n" + "=" * 60)
    print("Test: Judgment has concept_true_understanding field")
    print("=" * 60)
    j = Judgment(
        claim_id="test",
        conversation_id=1,
        turn_number=0,
        concept_true_understanding="The correct understanding is ...",
    )
    assert j.concept_true_understanding == "The correct understanding is ..."
    # default value
    j2 = Judgment(claim_id="test2", conversation_id=1, turn_number=0)
    assert j2.concept_true_understanding == ""
    print("[OK]")


def main():
    print("Step 16: ScientificConceptDiscoveryStrategy Unit Tests")
    print("=" * 60)
    test_strategy_registration()
    test_normalize_claim_valid()
    test_normalize_claim_empty_concept()
    test_normalize_claim_unknown_concept()
    test_normalize_claim_missing_action()
    test_render_claim()
    test_build_judge_user_prompt()
    test_judge_system_prompt()
    test_judgment_concept_true_understanding_field()
    print("\n" + "=" * 60)
    print("[PASS] All Step 16 tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
