from prompts._portability import DEVICE_RULES, TENSOR_CORE_FALLBACK, TOGGLE_RULES
import prompts.author, prompts.propose, prompts.plan, prompts.framework_configure


def test_propose_asks_for_toggles():
    system, _ = prompts.propose.build({})
    assert TOGGLE_RULES in system


def test_author_carries_toggle_and_device_rules():
    system, _ = prompts.author.build({})
    assert TOGGLE_RULES in system and DEVICE_RULES in system


def test_tensor_core_reference_carries_the_fallback_pattern():
    from prompts._tensor_core_reference import TENSOR_CORE_REFERENCE
    assert TENSOR_CORE_FALLBACK in TENSOR_CORE_REFERENCE


def test_plan_keeps_the_48kb_rule_and_mentions_the_launcher():
    system, _ = prompts.plan.build({})
    assert "48KB" in system and "GetMaxComputeUnits" in system


def test_legacy_configure_path_gets_the_device_rules():
    assert DEVICE_RULES in prompts.framework_configure.SYSTEM


def test_no_em_dash_in_the_new_rules():
    assert chr(0x2014) not in TOGGLE_RULES + DEVICE_RULES + TENSOR_CORE_FALLBACK


def test_the_persistent_grid_example_names_its_global_size_type():
    """The example passes a total thread count, which is right only under opencl; under
    cuda (the default) RunKernel's global size is a block count."""
    assert "`global_size_type: opencl`" in DEVICE_RULES
    assert "ktt::DimensionVector(sms * bps)" in DEVICE_RULES
