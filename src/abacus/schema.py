"""A JSON Schema 2020-12 validator, scoped to what the tool schemas use.

Validation is the load-bearing part of a tool boundary that a language model is
on the other end of. A model composes a call from context and gets it wrong in
particular, recoverable ways — a field left out, a volatility handed over in
percent, an expiry in days rather than years — and what comes back decides
whether it repairs the call or abandons it. So the validator collects *every*
violation rather than stopping at the first, and reports each one as a location,
a keyword and a sentence, which is enough for a caller to fix the call without
having to guess what the server wanted.

Why not a validation library: the server needs to control this boundary
precisely, including the specification's rules that ``$ref`` must not be
dereferenced across the network and that composition keywords need a resource
bound. Those are small to implement and awkward to retrofit onto someone else's
validator, and the alternative is a dependency for something the server cares
about more than its dependency would.

Not implemented, because no tool schema here uses them: ``if``/``then``/
``else``, ``dependentSchemas``, ``patternProperties``, ``unevaluatedProperties``
and format assertions. An unknown keyword is ignored, as the specification
requires, rather than quietly failing the instance.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

#: Bound on how deep validation will recurse. A schema is untrusted input in the
#: general case, and composition keywords make it cheap to write one that is
#: expensive to validate against; the specification asks implementations to put
#: a bound somewhere, so it is here.
MAX_DEPTH = 32

_TYPE_NAMES = {
    "null": type(None),
    "boolean": bool,
    "string": str,
    "array": list,
    "object": dict,
}


class SchemaError(ValueError):
    """The schema itself is malformed, which is a bug in the server, not the call."""


@dataclass(frozen=True)
class Violation:
    """One thing wrong with an instance, and where.

    ``path`` is a JSON Pointer into the instance, so ``/legs/2/vol`` points at
    exactly the offending value rather than at the call as a whole.
    """

    path: str
    keyword: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "keyword": self.keyword, "message": self.message}

    def __str__(self) -> str:
        where = self.path or "(root)"
        return f"{where}: {self.message}"


def _pointer(path: str, token: str | int) -> str:
    """Extend a JSON Pointer by one token, escaping per RFC 6901."""
    text = str(token).replace("~", "~0").replace("/", "~1")
    return f"{path}/{text}"


def _type_of(value: Any) -> str:
    """Name the JSON type of a decoded value.

    ``bool`` is checked before ``int`` because Python makes booleans a subclass
    of int, and a schema asking for an integer should not quietly accept
    ``true``.
    """
    if isinstance(value, bool):
        return "boolean"
    if value is None:
        return "null"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        # An integral float is an integer as far as JSON Schema is concerned:
        # 1.0 and 1 are the same JSON number.
        return "integer" if value.is_integer() and math.isfinite(value) else "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


def _matches_type(value: Any, wanted: str) -> bool:
    actual = _type_of(value)
    if wanted == "number":
        return actual in ("number", "integer")
    return actual == wanted


def validate(
    instance: Any, schema: dict[str, Any], *, max_depth: int = MAX_DEPTH
) -> list[Violation]:
    """Validate ``instance`` against ``schema``, returning every violation found.

    An empty list means the instance is valid. Violations come back in document
    order so a caller reading them top to bottom walks the instance.
    """
    if not isinstance(schema, dict):
        raise SchemaError("a schema must be an object")
    found: list[Violation] = []
    _validate(instance, schema, "", schema, found, max_depth)
    return found


def _resolve(ref: str, root: dict[str, Any]) -> dict[str, Any]:
    """Resolve a local ``$ref``.

    Only same-document pointers are supported. A ``$ref`` naming a network URI
    is refused rather than fetched: the specification requires that
    implementations not dereference one automatically, and a schema that
    silently reaches out over the network during validation is a request
    forgery waiting to happen.
    """
    if not ref.startswith("#"):
        raise SchemaError(f"$ref must be a same-document pointer, got {ref!r}")
    pointer = ref[1:]
    if not pointer:
        return root
    if not pointer.startswith("/"):
        raise SchemaError(f"unsupported $ref form: {ref!r}")
    node: Any = root
    for raw in pointer.lstrip("/").split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or token not in node:
            raise SchemaError(f"$ref does not resolve: {ref!r}")
        node = node[token]
    if not isinstance(node, dict):
        raise SchemaError(f"$ref does not point at a schema: {ref!r}")
    return node


def _validate(
    value: Any,
    schema: dict[str, Any],
    path: str,
    root: dict[str, Any],
    out: list[Violation],
    depth: int,
) -> None:
    if depth <= 0:
        raise SchemaError("schema nesting exceeded the maximum validation depth")

    if "$ref" in schema:
        ref = schema["$ref"]
        if not isinstance(ref, str):
            raise SchemaError("$ref must be a string")
        _validate(value, _resolve(ref, root), path, root, out, depth - 1)
        return

    _check_type(value, schema, path, out)
    _check_enum(value, schema, path, out)
    _check_numeric(value, schema, path, out)
    _check_string(value, schema, path, out)
    _check_array(value, schema, path, root, out, depth)
    _check_object(value, schema, path, root, out, depth)
    _check_composition(value, schema, path, root, out, depth)


def _check_type(value: Any, schema: dict[str, Any], path: str, out: list[Violation]) -> None:
    if "type" not in schema:
        return
    wanted = schema["type"]
    names = [wanted] if isinstance(wanted, str) else wanted
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise SchemaError("type must be a string or a list of strings")
    for name in names:
        if name not in _TYPE_NAMES and name not in ("number", "integer"):
            raise SchemaError(f"unknown type name: {name!r}")
    if any(_matches_type(value, name) for name in names):
        return
    expected = " or ".join(names)
    out.append(
        Violation(path, "type", f"expected {expected}, got {_type_of(value)}")
    )


def _check_enum(value: Any, schema: dict[str, Any], path: str, out: list[Violation]) -> None:
    if "const" in schema and value != schema["const"]:
        out.append(Violation(path, "const", f"must be {schema['const']!r}"))
    if "enum" not in schema:
        return
    choices = schema["enum"]
    if not isinstance(choices, list):
        raise SchemaError("enum must be a list")
    # `1 == True` in Python, so compare types alongside values; otherwise an
    # enum of [1, 2] would accept `true`.
    if not any(value == choice and _type_of(value) == _type_of(choice) for choice in choices):
        rendered = ", ".join(repr(c) for c in choices)
        out.append(Violation(path, "enum", f"must be one of: {rendered}"))


def _check_numeric(value: Any, schema: dict[str, Any], path: str, out: list[Violation]) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return
    for keyword, op, phrase in (
        ("minimum", "<", "at least"),
        ("maximum", ">", "at most"),
        ("exclusiveMinimum", "<=", "greater than"),
        ("exclusiveMaximum", ">=", "less than"),
    ):
        if keyword not in schema:
            continue
        bound = schema[keyword]
        if isinstance(bound, bool) or not isinstance(bound, int | float):
            raise SchemaError(f"{keyword} must be a number")
        failed = {
            "<": value < bound,
            ">": value > bound,
            "<=": value <= bound,
            ">=": value >= bound,
        }[op]
        if failed:
            out.append(Violation(path, keyword, f"must be {phrase} {bound}, got {value}"))

    if "multipleOf" in schema:
        divisor = schema["multipleOf"]
        if isinstance(divisor, bool) or not isinstance(divisor, int | float) or divisor <= 0:
            raise SchemaError("multipleOf must be a positive number")
        quotient = value / divisor
        if not math.isclose(quotient, round(quotient), rel_tol=0.0, abs_tol=1e-9):
            out.append(Violation(path, "multipleOf", f"must be a multiple of {divisor}"))


def _check_string(value: Any, schema: dict[str, Any], path: str, out: list[Violation]) -> None:
    if not isinstance(value, str):
        return
    if "minLength" in schema and len(value) < schema["minLength"]:
        out.append(
            Violation(path, "minLength", f"must be at least {schema['minLength']} characters")
        )
    if "maxLength" in schema and len(value) > schema["maxLength"]:
        out.append(
            Violation(path, "maxLength", f"must be at most {schema['maxLength']} characters")
        )
    if "pattern" in schema:
        pattern = schema["pattern"]
        if not isinstance(pattern, str):
            raise SchemaError("pattern must be a string")
        try:
            matches = re.search(pattern, value) is not None
        except re.error as exc:  # pragma: no cover - a bad pattern is a server bug
            raise SchemaError(f"invalid pattern {pattern!r}: {exc}") from exc
        if not matches:
            out.append(Violation(path, "pattern", f"must match {pattern}"))


def _check_array(
    value: Any,
    schema: dict[str, Any],
    path: str,
    root: dict[str, Any],
    out: list[Violation],
    depth: int,
) -> None:
    if not isinstance(value, list):
        return
    if "minItems" in schema and len(value) < schema["minItems"]:
        out.append(Violation(path, "minItems", f"must have at least {schema['minItems']} items"))
    if "maxItems" in schema and len(value) > schema["maxItems"]:
        out.append(Violation(path, "maxItems", f"must have at most {schema['maxItems']} items"))
    if schema.get("uniqueItems") is True:
        seen: list[Any] = []
        for item in value:
            if item in seen:
                out.append(Violation(path, "uniqueItems", "items must be unique"))
                break
            seen.append(item)
    items = schema.get("items")
    if isinstance(items, dict):
        for index, item in enumerate(value):
            _validate(item, items, _pointer(path, index), root, out, depth - 1)


def _check_object(
    value: Any,
    schema: dict[str, Any],
    path: str,
    root: dict[str, Any],
    out: list[Violation],
    depth: int,
) -> None:
    if not isinstance(value, dict):
        return

    properties = schema.get("properties")
    if properties is not None and not isinstance(properties, dict):
        raise SchemaError("properties must be an object")

    for name in schema.get("required", []):
        if name not in value:
            out.append(Violation(_pointer(path, name), "required", "is required but was not given"))

    if "minProperties" in schema and len(value) < schema["minProperties"]:
        out.append(
            Violation(
                path, "minProperties", f"must have at least {schema['minProperties']} properties"
            )
        )

    additional = schema.get("additionalProperties")
    known = set(properties or {})
    if additional is False:
        for name in value:
            if name not in known:
                # Naming the accepted properties turns "that is wrong" into
                # something the caller can act on, which matters most when the
                # mistake is a near miss such as `volatility` for `vol`.
                allowed = ", ".join(sorted(known)) or "(none)"
                out.append(
                    Violation(
                        _pointer(path, name),
                        "additionalProperties",
                        f"is not a recognised property; accepted properties are: {allowed}",
                    )
                )

    if properties:
        for name, subschema in properties.items():
            if name in value:
                if not isinstance(subschema, dict):
                    raise SchemaError(f"schema for property {name!r} must be an object")
                _validate(value[name], subschema, _pointer(path, name), root, out, depth - 1)

    if isinstance(additional, dict):
        for name, item in value.items():
            if name not in known:
                _validate(item, additional, _pointer(path, name), root, out, depth - 1)


def _check_composition(
    value: Any,
    schema: dict[str, Any],
    path: str,
    root: dict[str, Any],
    out: list[Violation],
    depth: int,
) -> None:
    for subschema in schema.get("allOf", []):
        _validate(value, subschema, path, root, out, depth - 1)

    if "anyOf" in schema:
        branches = schema["anyOf"]
        if not isinstance(branches, list) or not branches:
            raise SchemaError("anyOf must be a non-empty list")
        if not any(_is_valid(value, branch, root, depth - 1) for branch in branches):
            out.append(Violation(path, "anyOf", "does not match any accepted form"))

    if "oneOf" in schema:
        branches = schema["oneOf"]
        if not isinstance(branches, list) or not branches:
            raise SchemaError("oneOf must be a non-empty list")
        matched = sum(1 for branch in branches if _is_valid(value, branch, root, depth - 1))
        if matched != 1:
            detail = "does not match any accepted form" if matched == 0 else (
                f"is ambiguous: it matches {matched} accepted forms, and exactly one must match"
            )
            out.append(Violation(path, "oneOf", detail))

    if "not" in schema and _is_valid(value, schema["not"], root, depth - 1):
        out.append(Violation(path, "not", "matches a form that is not allowed"))


def _is_valid(value: Any, schema: Any, root: dict[str, Any], depth: int) -> bool:
    if not isinstance(schema, dict):
        raise SchemaError("a subschema must be an object")
    probe: list[Violation] = []
    _validate(value, schema, "", root, probe, depth)
    return not probe
