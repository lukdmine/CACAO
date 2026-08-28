"""Per-problem rules.

The reference implementation defines correctness, not intent. Nothing else stops a
branch from winning by dropping to fp16 accumulation whenever the tolerance is loose
enough, so intent has to be stated and — where it matters — enforced.
"""

import textwrap

from utils.rules import ForbiddenPattern, Rules, parse_rules


def test_no_rules_is_falsy_and_contributes_nothing():
    rules = parse_rules({})
    assert not rules
    assert rules.prompt_block() == ""
    assert rules.violations("anything") == []


def test_bare_list_shorthand():
    # Prose rules are the common case; making the simple form require nesting invites
    # it being skipped entirely.
    rules = parse_rules({"rules": ["Accumulate in fp32."]})
    assert rules.text == ["Accumulate in fp32."]


def test_bare_string_shorthand():
    assert parse_rules({"rules": "One rule."}).text == ["One rule."]


def test_full_form():
    rules = parse_rules(
        {
            "rules": {
                "text": ["Accumulate in fp32."],
                "forbid": [{"pattern": "wmma::", "reason": "no tensor cores"}],
            }
        }
    )
    assert rules.text == ["Accumulate in fp32."]
    assert rules.forbid[0].pattern == "wmma::"
    assert rules.forbid[0].reason == "no tensor cores"


def test_forbid_accepts_a_bare_pattern():
    rules = parse_rules({"rules": {"forbid": ["__half"]}})
    assert rules.forbid[0].pattern == "__half"


def test_malformed_entries_are_skipped_not_fatal():
    # A typo in a rules block must not take a run down.
    rules = parse_rules({"rules": {"forbid": [{"no_pattern": 1}, "ok"], "text": "t"}})
    assert [p.pattern for p in rules.forbid] == ["ok"]


def test_malformed_rules_block_is_ignored():
    assert not parse_rules({"rules": 42})


def test_violation_reports_the_line_and_the_reason():
    rules = Rules(forbid=[ForbiddenPattern(r"\b__half\b", "reduced precision not permitted")])
    kernel = "line one\nline two\n__half x;\n"
    found = rules.violations(kernel)
    assert len(found) == 1
    assert "line 3" in found[0]
    assert "reduced precision not permitted" in found[0]


def test_clean_kernel_has_no_violations():
    rules = Rules(forbid=[ForbiddenPattern("wmma::")])
    assert rules.violations("float acc = 0.0f;") == []


def test_invalid_pattern_is_ignored_rather_than_raising():
    rules = Rules(forbid=[ForbiddenPattern("(unclosed")])
    assert rules.violations("anything") == []


def test_prompt_block_states_both_kinds():
    rules = parse_rules(
        {"rules": {"text": ["Use fp32."], "forbid": [{"pattern": "wmma::", "reason": "why"}]}}
    )
    block = rules.prompt_block()
    assert "Use fp32." in block
    assert "/wmma::/" in block and "why" in block
    # Telling the model the check exists gets conforming code the first time instead
    # of via a failed compile.
    assert "checked automatically" in block
