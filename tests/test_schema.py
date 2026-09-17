"""The JSON Schema subset the tool boundary relies on."""

from __future__ import annotations

from typing import Any

import pytest

from abacus.schema import SchemaError, Violation, validate


def paths(instance: Any, schema: dict[str, Any]) -> list[str]:
    return [v.path for v in validate(instance, schema)]


def keywords(instance: Any, schema: dict[str, Any]) -> list[str]:
    return [v.keyword for v in validate(instance, schema)]


# -- types ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "wanted"),
    [
        (1, "integer"),
        (1.0, "integer"),  # an integral float is an integer in JSON
        (1, "number"),
        (1.5, "number"),
        ("x", "string"),
        (True, "boolean"),
        (None, "null"),
        ([], "array"),
        ({}, "object"),
    ],
)
def test_types_that_match(value: Any, wanted: str) -> None:
    assert validate(value, {"type": wanted}) == []


@pytest.mark.parametrize(
    ("value", "wanted"),
    [
        (True, "integer"),  # a JSON boolean is not a number
        (True, "number"),
        (1.5, "integer"),
        (1, "string"),
        ("1", "number"),
        (None, "string"),
        ([], "object"),
        ({}, "array"),
    ],
)
def test_types_that_do_not_match(value: Any, wanted: str) -> None:
    assert keywords(value, {"type": wanted}) == ["type"]


def test_boolean_is_not_an_integer_despite_python() -> None:
    # `isinstance(True, int)` is true in Python and must not leak into the
    # validator, or a schema expecting a count would accept `true`.
    violations = validate(True, {"type": "integer"})
    assert "got boolean" in violations[0].message


def test_a_union_of_types_is_accepted() -> None:
    schema = {"type": ["string", "null"]}
    assert validate("x", schema) == []
    assert validate(None, schema) == []
    assert keywords(1, schema) == ["type"]


def test_an_unknown_type_name_is_a_schema_error() -> None:
    with pytest.raises(SchemaError, match="unknown type name"):
        validate(1, {"type": "integerish"})


# -- numbers --------------------------------------------------------------


@pytest.mark.parametrize(
    ("schema", "bad", "good"),
    [
        ({"minimum": 0}, -1, 0),
        ({"maximum": 10}, 11, 10),
        ({"exclusiveMinimum": 0}, 0, 0.0001),
        ({"exclusiveMaximum": 1}, 1, 0.999),
        ({"multipleOf": 0.5}, 0.7, 1.5),
    ],
)
def test_numeric_bounds(schema: dict[str, Any], bad: Any, good: Any) -> None:
    assert validate(good, schema) == []
    assert len(validate(bad, schema)) == 1


def test_bound_messages_quote_the_offending_value() -> None:
    violation = validate(-3.0, {"exclusiveMinimum": 0})[0]
    assert "greater than 0" in violation.message
    assert "-3" in violation.message


def test_numeric_keywords_ignore_non_numbers() -> None:
    assert validate("x", {"minimum": 0}) == []
    assert validate(True, {"minimum": 0}) == []


def test_multipleof_must_be_positive() -> None:
    with pytest.raises(SchemaError, match="multipleOf"):
        validate(1, {"multipleOf": 0})


# -- strings --------------------------------------------------------------


def test_string_length_and_pattern() -> None:
    schema = {"type": "string", "minLength": 2, "maxLength": 4, "pattern": "^[a-z]+$"}
    assert validate("abc", schema) == []
    assert keywords("a", schema) == ["minLength"]
    assert keywords("abcde", schema) == ["maxLength"]
    assert keywords("AB", schema) == ["pattern"]


# -- enum and const -------------------------------------------------------


def test_enum_accepts_a_listed_value() -> None:
    assert validate("call", {"enum": ["call", "put"]}) == []


def test_enum_lists_the_choices_when_it_rejects() -> None:
    violation = validate("straddle", {"enum": ["call", "put"]})[0]
    assert "'call', 'put'" in violation.message


def test_enum_distinguishes_booleans_from_the_integers_they_equal() -> None:
    # `True == 1` in Python, so a naive membership test would accept `true`
    # against an enum of [1, 2].
    assert keywords(True, {"enum": [1, 2]}) == ["enum"]
    assert validate(1, {"enum": [1, 2]}) == []


def test_const() -> None:
    assert validate("x", {"const": "x"}) == []
    assert keywords("y", {"const": "x"}) == ["const"]


# -- objects --------------------------------------------------------------


def test_required_properties_are_reported_individually() -> None:
    schema = {"type": "object", "required": ["a", "b", "c"]}
    assert paths({"a": 1}, schema) == ["/b", "/c"]


def test_every_violation_is_collected_not_just_the_first() -> None:
    # A caller with three problems should learn about three problems, rather
    # than having to fix them one round trip at a time.
    schema = {
        "type": "object",
        "properties": {"a": {"type": "number"}, "b": {"type": "string"}},
        "required": ["a", "b", "c"],
    }
    assert len(validate({"a": "wrong", "b": 2}, schema)) == 3


def test_unknown_properties_are_named_along_with_what_is_accepted() -> None:
    schema = {
        "type": "object",
        "properties": {"vol": {"type": "number"}, "spot": {"type": "number"}},
        "additionalProperties": False,
    }
    violation = validate({"volatility": 0.2}, schema)[0]
    assert violation.path == "/volatility"
    assert "spot, vol" in violation.message


def test_additional_properties_are_permitted_by_default() -> None:
    assert validate({"extra": 1}, {"type": "object", "properties": {}}) == []


def test_additional_properties_may_carry_a_schema() -> None:
    schema = {"type": "object", "properties": {}, "additionalProperties": {"type": "number"}}
    assert validate({"a": 1}, schema) == []
    assert paths({"a": "x"}, schema) == ["/a"]


def test_nested_paths_point_at_the_offending_value() -> None:
    schema = {
        "type": "object",
        "properties": {
            "legs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"vol": {"type": "number", "exclusiveMinimum": 0}},
                },
            }
        },
    }
    assert paths({"legs": [{"vol": 0.2}, {"vol": 0.3}, {"vol": -1}]}, schema) == ["/legs/2/vol"]


def test_pointer_tokens_are_escaped() -> None:
    schema = {"type": "object", "required": ["a/b", "c~d"]}
    assert paths({}, schema) == ["/a~1b", "/c~0d"]


def test_min_properties() -> None:
    assert keywords({}, {"type": "object", "minProperties": 1}) == ["minProperties"]


# -- arrays ---------------------------------------------------------------


def test_array_length_bounds() -> None:
    schema = {"type": "array", "minItems": 1, "maxItems": 2}
    assert validate([1], schema) == []
    assert keywords([], schema) == ["minItems"]
    assert keywords([1, 2, 3], schema) == ["maxItems"]


def test_unique_items() -> None:
    schema = {"type": "array", "uniqueItems": True}
    assert validate([1, 2], schema) == []
    assert keywords([1, 1], schema) == ["uniqueItems"]


def test_items_are_validated_elementwise() -> None:
    assert paths([1, "x", 3], {"type": "array", "items": {"type": "number"}}) == ["/1"]


# -- composition ----------------------------------------------------------


def test_any_of() -> None:
    schema = {"anyOf": [{"type": "string"}, {"type": "number"}]}
    assert validate("x", schema) == []
    assert validate(1, schema) == []
    assert keywords(None, schema) == ["anyOf"]


def test_one_of_requires_exactly_one_branch() -> None:
    schema = {"oneOf": [{"type": "number"}, {"type": "string"}]}
    assert validate(1, schema) == []
    assert keywords(None, schema) == ["oneOf"]


def test_one_of_reports_an_ambiguous_instance_distinctly() -> None:
    # Matching two branches is a different mistake from matching none, and the
    # fix is different too, so the message says which happened.
    schema = {"oneOf": [{"type": "number"}, {"minimum": 0}]}
    violation = validate(1, schema)[0]
    assert "ambiguous" in violation.message
    assert "matches 2" in violation.message


def test_all_of_reports_each_failing_branch() -> None:
    schema = {"allOf": [{"type": "number"}, {"minimum": 10}]}
    assert validate(11, schema) == []
    assert keywords(1, schema) == ["minimum"]


def test_not() -> None:
    assert validate(1, {"not": {"type": "string"}}) == []
    assert keywords("x", {"not": {"type": "string"}}) == ["not"]


def test_empty_any_of_is_a_schema_error() -> None:
    with pytest.raises(SchemaError, match="anyOf"):
        validate(1, {"anyOf": []})


# -- references -----------------------------------------------------------


def test_local_ref_is_resolved() -> None:
    schema = {
        "$defs": {"positive": {"type": "number", "exclusiveMinimum": 0}},
        "type": "object",
        "properties": {"spot": {"$ref": "#/$defs/positive"}},
    }
    assert validate({"spot": 1}, schema) == []
    assert paths({"spot": -1}, schema) == ["/spot"]


def test_a_network_ref_is_refused_rather_than_fetched() -> None:
    # The specification forbids dereferencing a network `$ref` automatically. A
    # validator that quietly fetched one would let a schema drive requests from
    # inside the server.
    schema = {"$ref": "https://example.com/schema.json"}
    with pytest.raises(SchemaError, match="same-document"):
        validate(1, schema)


def test_an_unresolvable_ref_is_a_schema_error() -> None:
    with pytest.raises(SchemaError, match="does not resolve"):
        validate(1, {"$ref": "#/$defs/missing"})


def test_ref_siblings_are_applied_as_well_as_the_reference() -> None:
    # 2020-12 changed this: under draft-07 a `$ref` replaced its siblings, so
    # the `maximum` here would have been ignored and 500 would have passed.
    schema = {
        "$defs": {"positive": {"type": "number", "exclusiveMinimum": 0}},
        "$ref": "#/$defs/positive",
        "maximum": 100,
    }
    assert validate(50, schema) == []
    assert keywords(500, schema) == ["maximum"]
    assert keywords(-1, schema) == ["exclusiveMinimum"]


def test_a_recursive_definition_terminates_on_finite_data() -> None:
    schema = {
        "$defs": {
            "node": {
                "type": "object",
                "properties": {"value": {"type": "number"}, "child": {"$ref": "#/$defs/node"}},
                "required": ["value"],
            }
        },
        "$ref": "#/$defs/node",
    }
    assert validate({"value": 1, "child": {"value": 2, "child": {"value": 3}}}, schema) == []
    assert paths({"value": 1, "child": {"value": "x"}}, schema) == ["/child/value"]


def test_a_self_referential_schema_is_stopped_by_the_depth_bound() -> None:
    # `{"$ref": "#"}` refers to itself, so resolving it forever is a way to hang
    # the validator. Because resolving a reference spends depth, the bound
    # catches it instead.
    with pytest.raises(SchemaError, match="maximum validation depth"):
        validate(1, {"$ref": "#"})


# -- resource bounds ------------------------------------------------------


def test_deep_nesting_is_refused_rather_than_recursed_into() -> None:
    # A schema is untrusted input in the general case, and unbounded recursion
    # over one is a denial-of-service vector against the validator.
    schema: dict[str, Any] = {"type": "object"}
    node = schema
    for _ in range(40):
        child: dict[str, Any] = {"type": "object"}
        node["properties"] = {"next": child}
        node = child

    instance: dict[str, Any] = {}
    cursor = instance
    for _ in range(40):
        cursor["next"] = {}
        cursor = cursor["next"]

    with pytest.raises(SchemaError, match="maximum validation depth"):
        validate(instance, schema)


def test_the_depth_bound_is_adjustable() -> None:
    schema = {"type": "object", "properties": {"a": {"type": "object"}}}
    with pytest.raises(SchemaError):
        validate({"a": {}}, schema, max_depth=1)


# -- misc -----------------------------------------------------------------


def test_unknown_keywords_are_ignored() -> None:
    # Required by the specification: an unrecognised keyword is an annotation,
    # not a reason to fail the instance.
    assert validate(1, {"type": "number", "title": "x", "$comment": "y", "deprecated": True}) == []


def test_a_non_object_schema_is_refused() -> None:
    with pytest.raises(SchemaError, match="must be an object"):
        validate(1, ["not", "a", "schema"])  # type: ignore[arg-type]


def test_violation_renders_readably() -> None:
    assert str(Violation("/vol", "type", "expected number, got string")) == (
        "/vol: expected number, got string"
    )
    assert str(Violation("", "type", "expected object, got array")).startswith("(root)")
