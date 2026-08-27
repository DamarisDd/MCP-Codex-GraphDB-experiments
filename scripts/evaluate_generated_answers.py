"""Evaluate generated BPMN answers against their gold answers."""

# WHAT DOES THIS SCRIPT DO
#
# 1. It loads the answer schema and the gold answers.
# 2. It either gets the generated answers from the existing prediction file or extracts
#    them from the transcript files (in the transcripts directory).
# 3. During transcript extraction, it identifies the question and run, keeps the
#    first answer to the experiment question and ignores later follow-ups.
# 4. It normalizes the predictions and saves them in one JSONL file. query_id
#    connects an answer to its question and run_id identifies the attempt.
# 5. It checks whether every expected question and run is present.
# 6. It validates each answer and compares it with the corresponding gold answer.
# 7. It scores every individual answer, then averages the scores overall and by
#    question, category (C1-C6), response type (binary or structured) and run (1-3).
# 8. It writes a detailed JSON report and the CSV table of per-run scores.
#
# The script does not generate answers or determine the truth from the BPMN model.
# It assumes that the supplied gold answers are correct.
#
#
# ANSWER FORMAT
#
# Yes/no questions require only "Yes" or "No". All other answers follow the JSON
# schema and contain three main fields:
# - answer_type: the kind of answer, such as a list, value, ranking or path;
# - status: whether an answer was found or why it could not be provided;
# - results: the items returned as the answer.
#
# Each result contains identifier, label, element_type, rank, attributes and
# related_elements. rank, attributes and related_elements are used only when the
# question needs them; otherwise they are null or empty. element_type records the
# BPMN kind but is not scored for the current questions.
#
#
# SCORING
#
# - Lists: compare the returned items/entities and when requested, their attributes. 
#   Order is ignored unless the question asks for an ordered answer.
# - Values: compare the requested values and their units. Durations are converted to seconds 
#   before comparison.
# - Rankings: rank is an entity's position; rank 1 is the top position,
#   rank 2 is the next; tied entities share the same rank.
#   For example, individuals X and Y both have rank 1 because they
#   are tied for the largest number of assigned activities.
# - Paths: compare each path/route as an ordered sequence of BPMN elements. Correct
#   elements receive credit when they appear in the same relative order. For
#   example, if the gold path is A -> B -> C and the generated path is A -> C,
#   the two elements count as matches, while the missing one counts as a false negative.
#   Missing elements reduce recall, extra elements reduce precision and 
#   either can lower F1. Requested path costs or durations are also compared.
# - Yes/no answers: accept only "Yes" or "No" and check whether it matches the
#   gold answer.
#
# Entities are matched by identifier when possible, with labels as a fallback.
# related_elements are ignored for ordinary answers and used only when they
# represent an ordered path. A structured answer must be valid JSON and follow
# the schema; otherwise, the run is marked invalid and its scores are zero.
#
# Some questions accept more than one gold representation. For example, C3-023
# accepts a compact path and the same path with expanded subprocesses. A prediction
# is scored against every accepted version, and the best result is kept.
#
#
# METRICS AND OUTPUTS
#
# A true positive is a generated item or value that matches the gold answer. A
# false positive is a generated item or value with no gold match. A false negative
# is a gold item or value with no match in the generated answer. For example, if
# the gold list is A, B, C and the generated list is A, C, D, then A and C are true
# positives, D is a false positive and B is a false negative.
#
# Precision measures how much of the generated answer is correct; recall measures
# how much of the gold answer was recovered; F1 balances precision and recall.
#
# exact_answer shows whether the whole answer is correct. For a structured answer,
# it is true only when all expected items and values are present, nothing extra is
# returned, and answer_type and status match the gold answer.
#
# Scores are first calculated separately for every run and then averaged. Each run
# has the same influence on the final average, whether its answer contains one
# item or fifty.
#
# Three runs are expected for every question. By default, evaluation stops when a
# run is missing. With "--allow-incomplete", missing runs are listed in the
# coverage report and excluded from score calculations.
#
# The script writes normalized predictions, a detailed JSON report and a CSV file
# containing one row of scores for every evaluated run.


from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


JSONValue = Any
Entity = dict[str, Any]


@dataclass(frozen=True)
class Counts:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        if self.tp + self.fp == 0:
            return 1.0 if self.fn == 0 else 0.0
        return self.tp / (self.tp + self.fp)

    @property
    def recall(self) -> float:
        if self.tp + self.fn == 0:
            return 1.0 if self.fp == 0 else 0.0
        return self.tp / (self.tp + self.fn)

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0

    @property
    def exact(self) -> bool:
        return self.fp == 0 and self.fn == 0


@dataclass(frozen=True)
class RootFact:
    entity: Entity
    rank: int | None


@dataclass(frozen=True)
class RelationFact:
    root: Entity
    relation: str
    target: Entity
    order: int | None


@dataclass(frozen=True)
class AttributeFact:
    root: Entity
    name: str
    value: JSONValue
    unit: str | None


@dataclass(frozen=True)
class ScalarFact:
    name: str
    value: JSONValue
    unit: str | None


@dataclass(frozen=True)
class PathView:
    steps: tuple[Entity, ...]
    attributes: tuple[ScalarFact, ...] = ()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number}: each record must be an object.")
        records.append(record)
    return records


def resolve_json_pointer(root: JSONValue, pointer: str) -> JSONValue:
    if pointer == "#":
        return root
    if not pointer.startswith("#/"):
        raise ValueError(f"Only internal schema references are supported: {pointer}")
    current = root
    for token in pointer[2:].split("/"):
        current = current[token.replace("~1", "/").replace("~0", "~")]
    return current


def type_matches(value: JSONValue, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return False


def validate_instance(
    value: JSONValue,
    schema: dict[str, Any],
    root_schema: dict[str, Any],
    location: str = "$",
) -> list[str]:
    """Validate the JSON Schema subset used by the answer schema."""
    if "$ref" in schema:
        target = resolve_json_pointer(root_schema, schema["$ref"])
        return validate_instance(value, target, root_schema, location)

    if "anyOf" in schema:
        branch_errors = [
            validate_instance(value, branch, root_schema, location)
            for branch in schema["anyOf"]
        ]
        if all(errors for errors in branch_errors):
            return [f"{location}: does not satisfy any anyOf branch."]
        return []

    errors: list[str] = []
    expected = schema.get("type")
    if expected is not None:
        expected_types = expected if isinstance(expected, list) else [expected]
        if not any(type_matches(value, item) for item in expected_types):
            return [
                f"{location}: expected {expected_types}, got {type(value).__name__}."
            ]

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{location}: {value!r} is not an allowed enum value.")

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for required in schema.get("required", []):
            if required not in value:
                errors.append(f"{location}: missing required property {required!r}.")
        for name, child in value.items():
            if name in properties:
                errors.extend(
                    validate_instance(
                        child, properties[name], root_schema, f"{location}.{name}"
                    )
                )
            elif schema.get("additionalProperties") is False:
                errors.append(f"{location}: additional property {name!r} is forbidden.")

    if isinstance(value, list) and "items" in schema:
        for index, child in enumerate(value):
            errors.extend(
                validate_instance(
                    child, schema["items"], root_schema, f"{location}[{index}]"
                )
            )

    if isinstance(value, str) and "pattern" in schema:
        if re.search(schema["pattern"], value) is None:
            errors.append(f"{location}: does not match {schema['pattern']!r}.")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{location}: is below minimum {schema['minimum']}.")

    return errors


def normalized_text(value: Any, *, casefold: bool = False) -> str:
    text = unicodedata.normalize("NFKC", "" if value is None else str(value))
    text = " ".join(text.split())
    return text.casefold() if casefold else text


def canonical_attribute_name(value: Any) -> str:
    name = normalized_text(value, casefold=True)
    if name in {
        "duration",
        "execution_time",
        "expected_duration",
        "expected_execution_time",
        "path_execution_time",
        "total_duration",
        "total_execution_time",
    }:
        return "duration"
    if name in {"cost", "expected_cost", "path_cost", "total_cost"}:
        return "cost"
    if name in {"assigned_task_count", "task_count"}:
        return "task_count"
    return name


def canonical_communication_attribute_name(value: Any) -> str | None:
    """Return the atomic communication direction encoded by an attribute.

    Generated answers may serialize the same requested message facts as
    "sends", "sent_messages" or a counterpart-qualified name such as
    "sends_to_court". The counterpart remains represented by the root's
    communication relation; the attribute itself records whether the root
    sends or receives the message.
    """
    name = canonical_attribute_name(value)
    if name in {"sends", "sent_message", "sent_messages"} or name.startswith(
        "sends_to_"
    ):
        return "sent_message"
    if name in {
        "receives",
        "received_message",
        "received_messages",
    } or name.startswith("receives_from_"):
        return "received_message"
    return None


def atomic_attribute_values(name: Any, value: JSONValue) -> list[tuple[str, JSONValue]]:
    """Expand only known communication-list attributes into atomic facts."""
    communication_name = canonical_communication_attribute_name(name)
    if communication_name is None:
        return [(canonical_attribute_name(name), value)]
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(";") if part.strip()]
        if parts:
            return [(communication_name, part) for part in parts]
    return [(communication_name, value)]


def canonical_unit(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    unit = normalized_text(value, casefold=True)
    aliases = {
        "task": "tasks",
        "tasks": "tasks",
        "second": "seconds",
        "seconds": "seconds",
        "minute": "minutes",
        "minutes": "minutes",
        "hour": "hours",
        "hours": "hours",
        "day": "days",
        "days": "days",
    }
    return aliases.get(unit, unit)


def duration_seconds(value: JSONValue, unit: str | None) -> float | None:
    """Normalize supported duration serializations to seconds."""
    if isinstance(value, str):
        serialized = re.fullmatch(
            r"\s*(\d+):(\d+):(\d+):(\d+):(\d+)\s*", value
        )
        if serialized:
            years, days, hours, minutes, seconds = map(int, serialized.groups())
            return float(
                years * 365 * 24 * 3600
                + days * 24 * 3600
                + hours * 3600
                + minutes * 60
                + seconds
            )
        labelled = re.fullmatch(
            r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s+"
            r"(seconds?|minutes?|hours?|days?)\s*",
            value,
            flags=re.IGNORECASE,
        )
        if labelled:
            value = float(labelled.group(1))
            unit = canonical_unit(labelled.group(2))

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    factors = {
        "seconds": 1.0,
        "minutes": 60.0,
        "hours": 3600.0,
        "days": 86400.0,
    }
    canonical = canonical_unit(unit)
    if canonical not in factors:
        return None
    return float(value) * factors[canonical]


def scalar_fact(name: Any, value: JSONValue, unit: Any) -> ScalarFact:
    return ScalarFact(
        name=canonical_attribute_name(name),
        value=value,
        unit=canonical_unit(unit),
    )


def scalar_fact_matches(
    gold: ScalarFact,
    predicted: ScalarFact,
    tolerance: float,
) -> bool:
    if gold.name != predicted.name:
        return False
    if gold.name == "duration":
        gold_seconds = duration_seconds(gold.value, gold.unit)
        predicted_seconds = duration_seconds(predicted.value, predicted.unit)
        if gold_seconds is None or predicted_seconds is None:
            return False
        return math.isclose(
            gold_seconds,
            predicted_seconds,
            rel_tol=0.0,
            abs_tol=tolerance,
        )
    return gold.unit == predicted.unit and scalar_matches(
        gold.value, predicted.value, tolerance
    )


def identifier(entity: Entity) -> str | None:
    value = entity.get("identifier")
    return normalized_text(value) if isinstance(value, str) and value.strip() else None


def label(entity: Entity) -> str | None:
    value = entity.get("label")
    return (
        normalized_text(value, casefold=True)
        if isinstance(value, str) and value.strip()
        else None
    )


def element_type(entity: Entity) -> str | None:
    value = entity.get("element_type")
    return (
        normalized_text(value, casefold=True)
        if isinstance(value, str) and value.strip()
        else None
    )


def entity_matches(gold: Entity, predicted: Entity, mode: str) -> bool:
    gold_id, predicted_id = identifier(gold), identifier(predicted)
    gold_label, predicted_label = label(gold), label(predicted)

    if mode == "identifier":
        return (
            gold_id is not None
            and predicted_id is not None
            and gold_id == predicted_id
        )
    if mode == "hybrid" and gold_id is not None and predicted_id is not None:
        return gold_id == predicted_id

    labels_equal = gold_label is not None and gold_label == predicted_label
    if not labels_equal:
        return False
    gold_type, predicted_type = element_type(gold), element_type(predicted)
    return gold_type is None or predicted_type is None or gold_type == predicted_type


def path_entity_matches(gold: Entity, predicted: Entity, mode: str) -> bool:
    """Match path elements by identifier, falling back to label only.

    Element type is deliberately ignored for the label fallback because path
    scoring concerns the identity of the returned BPMN element rather than the
    way its type was serialized.
    """
    gold_id, predicted_id = identifier(gold), identifier(predicted)
    if mode == "identifier":
        return (
            gold_id is not None
            and predicted_id is not None
            and gold_id == predicted_id
        )
    if mode == "hybrid" and gold_id is not None and predicted_id is not None:
        return gold_id == predicted_id
    gold_label, predicted_label = label(gold), label(predicted)
    return (
        gold_label is not None
        and predicted_label is not None
        and gold_label == predicted_label
    )


def answer_results(answer: dict[str, Any]) -> list[Entity]:
    results = answer.get("results", [])
    if not isinstance(results, list):
        return []
    return [item for item in results if isinstance(item, dict)]


def answer_is_order_sensitive(answer: dict[str, Any]) -> bool:
    """Return whether rank/order carries answer semantics in the gold answer.

    "path" and "ranking" are ordered by definition. A gold answer of
    another type may still explicitly encode an ordered result, in which case
    a non-null root rank enables order-sensitive matching. Related-element
    order is metadata outside normalized path scoring. Prediction-only ordering
    never changes the scoring policy.
    """
    if answer.get("answer_type") in {"path", "ranking"}:
        return True
    for root in answer_results(answer):
        if isinstance(root.get("rank"), int):
            return True
    return False


def answer_facts(
    answer: dict[str, Any], *, order_sensitive: bool = False
) -> list[AnswerFact]:
    """Extract scored roots and attributes from a non-path answer.

    Related elements are intentionally omitted. They remain available in the
    original JSON for qualitative inspection but cannot create true positives,
    false positives, or false negatives.
    """
    facts: list[AnswerFact] = []
    for root in answer_results(answer):
        rank = root.get("rank")
        facts.append(
            RootFact(
                root,
                rank if order_sensitive and isinstance(rank, int) else None,
            )
        )

        attributes = root.get("attributes", [])
        if isinstance(attributes, list):
            for attribute in attributes:
                if not isinstance(attribute, dict):
                    continue
                unit = attribute.get("unit")
                for atomic_name, atomic_value in atomic_attribute_values(
                    attribute.get("name"), attribute.get("value")
                ):
                    facts.append(
                        AttributeFact(
                            root=root,
                            name=atomic_name,
                            value=atomic_value,
                            unit=canonical_unit(unit),
                        )
                    )
    return facts


def scalar_matches(gold: JSONValue, predicted: JSONValue, tolerance: float) -> bool:
    gold_numeric = isinstance(gold, (int, float)) and not isinstance(gold, bool)
    predicted_numeric = isinstance(predicted, (int, float)) and not isinstance(
        predicted, bool
    )
    if gold_numeric and predicted_numeric:
        return math.isclose(
            float(gold), float(predicted), rel_tol=0.0, abs_tol=tolerance
        )
    if isinstance(gold, str) and isinstance(predicted, str):
        return normalized_text(gold, casefold=True) == normalized_text(
            predicted, casefold=True
        )
    return gold == predicted


def fact_matches(
    gold: AnswerFact,
    predicted: AnswerFact,
    mode: str,
    tolerance: float,
) -> bool:
    if isinstance(gold, RootFact) and isinstance(predicted, RootFact):
        return gold.rank == predicted.rank and entity_matches(
            gold.entity, predicted.entity, mode
        )
    if isinstance(gold, RelationFact) and isinstance(predicted, RelationFact):
        return (
            gold.order == predicted.order
            and entity_matches(gold.root, predicted.root, mode)
            and entity_matches(gold.target, predicted.target, mode)
        )
    if isinstance(gold, AttributeFact) and isinstance(predicted, AttributeFact):
        gold_scalar = ScalarFact(gold.name, gold.value, gold.unit)
        predicted_scalar = ScalarFact(
            predicted.name, predicted.value, predicted.unit
        )
        return (
            entity_matches(gold.root, predicted.root, mode)
            and scalar_fact_matches(gold_scalar, predicted_scalar, tolerance)
        )
    return False


def maximum_matching(
    gold: Sequence[AnswerFact],
    predicted: Sequence[AnswerFact],
    matches: Callable[[AnswerFact, AnswerFact], bool],
) -> list[tuple[int, int]]:
    """Return a maximum one-to-one matching between gold and predicted facts."""
    predicted_to_gold: dict[int, int] = {}

    def augment(gold_index: int, visited: set[int]) -> bool:
        for predicted_index, predicted_item in enumerate(predicted):
            if predicted_index in visited or not matches(
                gold[gold_index], predicted_item
            ):
                continue
            visited.add(predicted_index)
            previous_gold = predicted_to_gold.get(predicted_index)
            if previous_gold is None or augment(previous_gold, visited):
                predicted_to_gold[predicted_index] = gold_index
                return True
        return False

    for gold_index in range(len(gold)):
        augment(gold_index, set())
    return sorted(
        (gold_index, predicted_index)
        for predicted_index, gold_index in predicted_to_gold.items()
    )


def score_fact_sets(
    gold: Sequence[AnswerFact],
    predicted: Sequence[AnswerFact],
    mode: str,
    tolerance: float,
) -> Counts:
    pairs = maximum_matching(
        gold,
        predicted,
        lambda left, right: fact_matches(left, right, mode, tolerance),
    )
    tp = len(pairs)
    return Counts(tp=tp, fp=len(predicted) - tp, fn=len(gold) - tp)


def scalar_facts(answer: dict[str, Any]) -> list[ScalarFact]:
    """Extract the logical scalar values from a value answer.

    A value may be serialized as an attribute or, when no attribute is
    present, directly in a derived result label such as "4 hours".
    """
    facts: list[ScalarFact] = []
    for root in answer_results(answer):
        raw_attributes = root.get("attributes", [])
        attributes = (
            [item for item in raw_attributes if isinstance(item, dict)]
            if isinstance(raw_attributes, list)
            else []
        )
        if attributes:
            facts.extend(
                scalar_fact(
                    item.get("name"), item.get("value"), item.get("unit")
                )
                for item in attributes
            )
            continue

        root_label = root.get("label")
        root_type = element_type(root)
        if root_type in {"duration", "time"} and isinstance(root_label, str):
            if duration_seconds(root_label, None) is not None:
                facts.append(ScalarFact("duration", root_label, None))
    return facts


def score_scalar_answers(
    gold: dict[str, Any],
    predicted: dict[str, Any],
    tolerance: float,
) -> Counts:
    gold_facts = scalar_facts(gold)
    predicted_facts = scalar_facts(predicted)
    pairs = maximum_matching(
        gold_facts,
        predicted_facts,
        lambda left, right: scalar_fact_matches(left, right, tolerance),
    )
    tp = len(pairs)
    return Counts(
        tp=tp,
        fp=len(predicted_facts) - tp,
        fn=len(gold_facts) - tp,
    )


def root_scalar_facts(root: Entity) -> tuple[ScalarFact, ...]:
    raw_attributes = root.get("attributes", [])
    if not isinstance(raw_attributes, list):
        return ()
    return tuple(
        scalar_fact(item.get("name"), item.get("value"), item.get("unit"))
        for item in raw_attributes
        if isinstance(item, dict)
    )


def is_start_path_element(root: Entity) -> bool:
    root_type = element_type(root) or ""
    if "start_event" in root_type:
        return True
    root_id = identifier(root) or ""
    return "#Start_Event" in root_id


def is_end_path_element(root: Entity) -> bool:
    root_type = element_type(root) or ""
    if "end_event" in root_type:
        return True
    root_id = identifier(root) or ""
    return "#End_Event" in root_id


def flat_path_views(roots: Sequence[Entity]) -> list[PathView]:
    """Split concatenated flat routes without splitting nested/cross-pool starts."""
    groups: list[list[Entity]] = []
    current: list[Entity] = []
    for root in roots:
        repeated_route_start = (
            current
            and is_start_path_element(root)
            and path_entity_matches(current[0], root, "hybrid")
        )
        starts_after_completed_route = (
            current
            and is_start_path_element(root)
            and is_end_path_element(current[-1])
        )
        if repeated_route_start or starts_after_completed_route:
            groups.append(current)
            current = []
        current.append(root)
    if current:
        groups.append(current)
    return [
        PathView(
            steps=tuple(group),
            attributes=tuple(
                scalar
                for root in group
                for scalar in root_scalar_facts(root)
            ),
        )
        for group in groups
    ]


def path_views(answer: dict[str, Any]) -> list[PathView]:
    """Normalize containerized, flat, and concatenated-flat path answers."""
    roots = answer_results(answer)
    if not roots:
        return []

    has_containerized_path = any(
        isinstance(root.get("related_elements"), list)
        and any(isinstance(item, dict) for item in root["related_elements"])
        for root in roots
    )
    if not has_containerized_path:
        return flat_path_views(roots)

    views: list[PathView] = []
    for root in roots:
        raw_related = root.get("related_elements", [])
        related = (
            [item for item in raw_related if isinstance(item, dict)]
            if isinstance(raw_related, list)
            else []
        )
        if not related:
            views.append(
                PathView(
                    steps=(root,),
                    attributes=root_scalar_facts(root),
                )
            )
            continue

        indexed = list(enumerate(related))
        indexed.sort(
            key=lambda pair: (
                0 if isinstance(pair[1].get("order"), int) else 1,
                pair[1].get("order")
                if isinstance(pair[1].get("order"), int)
                else pair[0],
                pair[0],
            )
        )
        ordered_related = tuple(item for _, item in indexed)
        root_type = element_type(root)
        derived_container = identifier(root) is None or root_type in {
            "path",
            "route",
        }
        if derived_container:
            views.append(
                PathView(
                    steps=ordered_related,
                    attributes=root_scalar_facts(root),
                )
            )
        else:
            views.append(
                PathView(
                    steps=(root, *ordered_related),
                    attributes=root_scalar_facts(root),
                )
            )
    return views


def lcs_entity_pairs(
    gold: Sequence[Entity], predicted: Sequence[Entity], mode: str
) -> list[tuple[int, int]]:
    """Return an order-preserving maximum entity matching."""
    rows, columns = len(gold), len(predicted)
    table = [[0] * (columns + 1) for _ in range(rows + 1)]
    for gold_index in range(rows - 1, -1, -1):
        for predicted_index in range(columns - 1, -1, -1):
            if path_entity_matches(
                gold[gold_index], predicted[predicted_index], mode
            ):
                table[gold_index][predicted_index] = (
                    1 + table[gold_index + 1][predicted_index + 1]
                )
            else:
                table[gold_index][predicted_index] = max(
                    table[gold_index + 1][predicted_index],
                    table[gold_index][predicted_index + 1],
                )

    pairs: list[tuple[int, int]] = []
    gold_index = predicted_index = 0
    while gold_index < rows and predicted_index < columns:
        if (
            path_entity_matches(
                gold[gold_index], predicted[predicted_index], mode
            )
            and table[gold_index][predicted_index]
            == 1 + table[gold_index + 1][predicted_index + 1]
        ):
            pairs.append((gold_index, predicted_index))
            gold_index += 1
            predicted_index += 1
        elif table[gold_index + 1][predicted_index] >= table[gold_index][
            predicted_index + 1
        ]:
            gold_index += 1
        else:
            predicted_index += 1
    return pairs


def path_pair_score(
    gold: PathView,
    predicted: PathView,
    mode: str,
    tolerance: float,
    score_attributes: bool,
) -> int:
    step_matches = len(lcs_entity_pairs(gold.steps, predicted.steps, mode))
    if not score_attributes:
        return step_matches
    attribute_matches = maximum_matching(
        gold.attributes,
        predicted.attributes,
        lambda left, right: scalar_fact_matches(left, right, tolerance),
    )
    return step_matches + len(attribute_matches)


def score_path_answers(
    gold: dict[str, Any],
    predicted: dict[str, Any],
    mode: str,
    tolerance: float,
) -> Counts:
    """Score ordered routes and any route metric requested by the gold answer."""
    gold_paths = path_views(gold)
    predicted_paths = path_views(predicted)
    score_attributes = any(path.attributes for path in gold_paths)
    gold_total = sum(
        len(path.steps) + (len(path.attributes) if score_attributes else 0)
        for path in gold_paths
    )
    predicted_total = sum(
        len(path.steps) + (len(path.attributes) if score_attributes else 0)
        for path in predicted_paths
    )
    pair_scores = [
        [
            path_pair_score(
                left,
                right,
                mode,
                tolerance,
                score_attributes,
            )
            for right in predicted_paths
        ]
        for left in gold_paths
    ]

    if len(predicted_paths) <= 18:
        @lru_cache(maxsize=None)
        def best(gold_index: int, used_mask: int) -> int:
            if gold_index == len(gold_paths):
                return 0
            score = best(gold_index + 1, used_mask)
            for predicted_index in range(len(predicted_paths)):
                if used_mask & (1 << predicted_index):
                    continue
                score = max(
                    score,
                    pair_scores[gold_index][predicted_index]
                    + best(
                        gold_index + 1,
                        used_mask | (1 << predicted_index),
                    ),
                )
            return score

        tp = best(0, 0)
    else:
        # Avoid exponential state growth for unusually large route collections.
        candidates = sorted(
            (
                (score, gold_index, predicted_index)
                for gold_index, row in enumerate(pair_scores)
                for predicted_index, score in enumerate(row)
            ),
            reverse=True,
        )
        used_gold: set[int] = set()
        used_predicted: set[int] = set()
        tp = 0
        for score, gold_index, predicted_index in candidates:
            if score == 0:
                break
            if gold_index in used_gold or predicted_index in used_predicted:
                continue
            used_gold.add(gold_index)
            used_predicted.add(predicted_index)
            tp += score

    return Counts(tp=tp, fp=predicted_total - tp, fn=gold_total - tp)


def empty_answer() -> dict[str, Any]:
    return {"answer_type": None, "status": None, "results": []}


def coerce_answer(value: Any) -> tuple[dict[str, Any], list[str]]:
    if isinstance(value, dict):
        return value, []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            return empty_answer(), [
                f"Answer is not a standalone JSON object: {exc}"
            ]
        if isinstance(parsed, dict):
            return parsed, []
    return empty_answer(), ["Answer is missing or is not a JSON object."]


def score_answer(
    gold: dict[str, Any],
    predicted: dict[str, Any],
    mode: str,
    tolerance: float,
    schema_valid: bool,
) -> dict[str, Any]:
    order_sensitive = answer_is_order_sensitive(gold)
    path_scoring = gold.get("answer_type") == "path"
    value_scoring = gold.get("answer_type") == "value"
    if path_scoring:
        gold_paths = path_views(gold)
        predicted_paths = path_views(predicted)
        score_path_attributes = any(path.attributes for path in gold_paths)
        gold_fact_count = sum(
            len(path.steps)
            + (len(path.attributes) if score_path_attributes else 0)
            for path in gold_paths
        )
        predicted_fact_count = sum(
            len(path.steps)
            + (len(path.attributes) if score_path_attributes else 0)
            for path in predicted_paths
        )
    elif value_scoring:
        gold_value_facts = scalar_facts(gold)
        predicted_value_facts = scalar_facts(predicted)
        gold_fact_count = len(gold_value_facts)
        predicted_fact_count = len(predicted_value_facts)
    else:
        gold_facts = answer_facts(gold, order_sensitive=order_sensitive)
        predicted_facts = answer_facts(
            predicted, order_sensitive=order_sensitive
        )
        gold_fact_count = len(gold_facts)
        predicted_fact_count = len(predicted_facts)
    answer_type_correct = gold.get("answer_type") == predicted.get("answer_type")
    status_correct = gold.get("status") == predicted.get("status")

    if schema_valid:
        if path_scoring:
            counts = score_path_answers(gold, predicted, mode, tolerance)
        elif value_scoring:
            counts = score_scalar_answers(gold, predicted, tolerance)
        else:
            counts = score_fact_sets(
                gold_facts, predicted_facts, mode, tolerance
            )
        precision = counts.precision
        recall = counts.recall
        f1 = counts.f1
        exact_answer = counts.exact and answer_type_correct and status_correct
    else:
        counts = Counts(
            tp=0,
            fp=predicted_fact_count,
            fn=gold_fact_count,
        )
        precision = recall = f1 = 0.0
        exact_answer = False

    return {
        "fact_counts": {
            "gold": gold_fact_count,
            "predicted": predicted_fact_count,
            "tp": counts.tp,
            "fp": counts.fp,
            "fn": counts.fn,
        },
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_answer": exact_answer,
        "answer_type_correct": answer_type_correct,
        "status_correct": status_correct,
        "order_sensitive": order_sensitive,
        "path_serialization_normalized": path_scoring,
        "value_serialization_normalized": value_scoring,
    }


def best_ground_truth(
    candidates: Sequence[dict[str, Any]],
    predicted: dict[str, Any],
    mode: str,
    tolerance: float,
    schema_valid: bool,
) -> tuple[int, dict[str, Any]]:
    scores = [
        score_answer(candidate, predicted, mode, tolerance, schema_valid)
        for candidate in candidates
    ]
    index = max(
        range(len(scores)),
        key=lambda item: (
            scores[item]["exact_answer"],
            scores[item]["f1"],
            scores[item]["precision"],
            scores[item]["recall"],
        ),
    )
    return index, scores[index]


AnswerFact = RootFact | RelationFact | AttributeFact


FILENAME_PATTERN = re.compile(
    r"^(?P<query_id>C\d+-\d{3})_(?P<run_id>\d+)_"
)
QUOTE_TRANSLATION = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})
FOLLOW_UP_PROMPT = re.compile(r"\n\s*\n(?=> [^\n]+\?\s*(?:\n|$))")
QUERY_ID_PATTERN = re.compile(r"^C(?P<category>\d+)-(?P<number>\d+)$")


def normalized_question(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).translate(QUOTE_TRANSLATION)
    return " ".join(value.split()).strip()


def gold_question_map(records: list[dict[str, Any]]) -> tuple[dict[str, str], set[str]]:
    by_question: dict[str, str] = {}
    query_ids: set[str] = set()
    for record in records:
        query_id = record.get("query_id")
        question = record.get("question")
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError("Every gold record requires a non-empty query_id.")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{query_id}: gold record requires a non-empty question.")
        normalized = normalized_question(question)
        if normalized in by_question:
            raise ValueError(f"Duplicate normalized gold question: {question!r}")
        by_question[normalized] = query_id
        query_ids.add(query_id)
    return by_question, query_ids


def transcript_question(text: str, by_question: dict[str, str]) -> tuple[str, str]:
    candidates: list[tuple[str, str]] = []
    for line in text.splitlines():
        if not line.startswith("> "):
            continue
        raw = line[2:].strip()
        normalized = normalized_question(raw)
        query_id = by_question.get(normalized)
        if query_id:
            candidates.append((query_id, raw))
    unique = list(dict.fromkeys(candidates))
    if not unique:
        raise ValueError("No transcript question matches a gold question.")
    if len({query_id for query_id, _ in unique}) != 1:
        raise ValueError(f"Transcript contains multiple gold questions: {unique!r}")
    return unique[0]


def unwrap_entire_code_fence(text: str) -> str:
    lines = text.splitlines()
    if len(lines) >= 2 and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


def final_response(text: str) -> tuple[Any, str, str | None]:
    """Return (answer value, extraction mode, JSON parsing error)."""
    tail = text.rsplit("</details>", 1)[-1].strip()
    follow_up = re.search(r"(?m)^>\s+\S", tail)
    has_follow_up = follow_up is not None
    if follow_up is not None:
        tail = tail[: follow_up.start()].rstrip()
    tail = unwrap_entire_code_fence(tail)
    if not tail:
        return "", "empty_text", "No final response found after the transcript details."
    try:
        mode = "standalone_json_before_followup" if has_follow_up else "standalone_json"
        return json.loads(tail), mode, None
    except json.JSONDecodeError as exc:
        mode = "raw_text_before_followup" if has_follow_up else "raw_text"
        return tail, mode, str(exc)


def discover_paths(items: Iterable[Path]) -> list[Path]:
    paths: set[Path] = set()
    for item in items:
        if item.is_file():
            paths.add(item.resolve())
        elif item.is_dir():
            paths.update(path.resolve() for path in item.rglob("*.txt") if path.is_file())
        else:
            raise ValueError(f"Transcript source does not exist: {item}")
    return sorted(paths)


def manifest_selections(path: Path) -> list[tuple[str, str, Path]]:
    selections: list[tuple[str, str, Path]] = []
    for record in load_jsonl(path):
        query_id = record.get("query_id")
        run_id = record.get("run_id")
        raw_path = record.get("path")
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError("Every manifest record requires query_id.")
        if run_id is None or not str(run_id).strip():
            raise ValueError(f"{query_id}: manifest record requires run_id.")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"{query_id}/{run_id}: manifest record requires path.")
        transcript = Path(raw_path)
        if not transcript.is_absolute():
            transcript = path.parent / transcript
        selections.append((query_id, str(run_id), transcript.resolve()))
    return selections


def automatic_selections(
    paths: list[Path], by_question: dict[str, str]
) -> tuple[list[tuple[str, str, Path]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[Path]] = defaultdict(list)
    skipped: list[dict[str, Any]] = []
    for path in paths:
        match = FILENAME_PATTERN.match(path.name)
        if not match:
            skipped.append({"path": str(path), "reason": "filename_has_no_query_and_run"})
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        try:
            question_query_id, _ = transcript_question(text, by_question)
        except ValueError as exc:
            skipped.append({"path": str(path), "reason": str(exc)})
            continue
        filename_query_id = match.group("query_id")
        if filename_query_id != question_query_id:
            skipped.append(
                {
                    "path": str(path),
                    "reason": "filename_question_mismatch",
                    "filename_query_id": filename_query_id,
                    "question_query_id": question_query_id,
                }
            )
            continue
        grouped[(question_query_id, match.group("run_id"))].append(path)

    duplicates = {key: values for key, values in grouped.items() if len(values) > 1}
    if duplicates:
        details = "; ".join(
            f"{query_id}/run {run_id}: {[path.name for path in paths]}"
            for (query_id, run_id), paths in sorted(duplicates.items())
        )
        raise ValueError(
            "Automatic discovery found duplicate transcript candidates. Use a manifest "
            f"to select the intended experiment files. {details}"
        )
    return [(query_id, run_id, paths[0]) for (query_id, run_id), paths in grouped.items()], skipped


def primary_response(text: str) -> tuple[str, bool]:
    """Return the first visible answer and whether a later user turn exists."""
    tail = text.rsplit("</details>", 1)[-1].strip()
    follow_up = FOLLOW_UP_PROMPT.search(tail)
    response = tail[: follow_up.start()] if follow_up else tail
    return unwrap_entire_code_fence(response).strip(), follow_up is not None


def parse_binary(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"yes", "yes."}:
            return True
        if normalized in {"no", "no."}:
            return False
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    gold = parser.add_mutually_exclusive_group(required=True)
    gold.add_argument("--gold", type=Path, help="Unified gold JSONL file.")
    gold.add_argument(
        "--gold-sources",
        nargs="+",
        type=Path,
        help="Gold JSONL files to merge before evaluation.",
    )
    parser.add_argument(
        "--gold-out",
        type=Path,
        default=Path("unified_evaluation/gold_answers_all.jsonl"),
        help="Unified gold output when --gold-sources is used.",
    )

    predictions = parser.add_mutually_exclusive_group(required=True)
    predictions.add_argument(
        "--predictions",
        type=Path,
        help="Existing unified predictions JSONL.",
    )
    predictions.add_argument(
        "--prediction-sources",
        nargs="+",
        type=Path,
        help="Prediction JSONL files to merge before evaluation.",
    )
    predictions.add_argument(
        "--transcripts",
        nargs="+",
        type=Path,
        help="Transcript files or directories from which to extract predictions.",
    )
    predictions.add_argument(
        "--manifest",
        type=Path,
        help="JSONL manifest with query_id, run_id, and transcript path.",
    )

    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=Path("unified_evaluation/predictions_all.jsonl"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("unified_evaluation/evaluation_all_report.json"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("unified_evaluation/evaluation_all_per_run.csv"),
    )
    parser.add_argument("--expected-runs", type=int, default=3)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Evaluate available runs while reporting missing query/run pairs. "
            "Missing runs are not silently converted into model errors."
        ),
    )
    parser.add_argument(
        "--match",
        choices=("identifier", "label", "hybrid"),
        default="hybrid",
    )
    parser.add_argument("--numeric-tolerance", type=float, default=0.0)
    return parser.parse_args()


def query_sort_key(query_id: str) -> tuple[int, int, str]:
    match = QUERY_ID_PATTERN.fullmatch(query_id)
    if not match:
        return (10**9, 10**9, query_id)
    return (
        int(match.group("category")),
        int(match.group("number")),
        query_id,
    )


def run_sort_key(run_id: Any) -> tuple[int, str]:
    value = str(run_id)
    return (int(value), "") if value.isdigit() else (10**9, value)


def category_of(query_id: str) -> str:
    match = QUERY_ID_PATTERN.fullmatch(query_id)
    return f"C{match.group('category')}" if match else "unclassified"


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def infer_response_mode(value: JSONValue) -> str:
    if isinstance(value, bool):
        return "binary"
    if isinstance(value, dict):
        return "structured"
    if isinstance(value, str):
        parsed_binary = parse_binary(value)
        if parsed_binary is not None:
            return "binary"
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return "structured"
    raise ValueError(
        "A gold answer must be a Boolean/Yes-No value or a structured JSON object."
    )


def normalize_binary_gold(value: JSONValue) -> bool:
    parsed = parse_binary(value)
    if parsed is None:
        raise ValueError(f"Invalid binary gold answer: {value!r}")
    return parsed


def normalize_structured_gold(
    value: JSONValue,
    schema: dict[str, Any],
    *,
    location: str,
) -> dict[str, Any]:
    answer, parse_errors = coerce_answer(value)
    errors = parse_errors + validate_instance(answer, schema, schema)
    if errors:
        raise ValueError(f"{location}: structured gold violates the schema: {errors}")
    return answer


def normalize_gold_records(
    records: Sequence[dict[str, Any]], schema: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    normalized: list[dict[str, Any]] = []
    by_query: dict[str, dict[str, Any]] = {}

    for raw in records:
        query_id = raw.get("query_id")
        question = raw.get("question")
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError("Every gold record requires a non-empty query_id.")
        if query_id in by_query:
            raise ValueError(f"Duplicate gold answer for {query_id}.")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{query_id}: question must be a non-empty string.")

        primary_raw = raw.get("ground_truth", raw.get("gold_answer"))
        mode = infer_response_mode(primary_raw)
        declared_mode = raw.get("response_mode")
        if declared_mode is not None and declared_mode != mode:
            raise ValueError(
                f"{query_id}: declared response_mode {declared_mode!r} conflicts "
                f"with the {mode} gold answer."
            )

        alternatives_raw = raw.get(
            "acceptable_ground_truths", raw.get("acceptable_answers", [])
        )
        if not isinstance(alternatives_raw, list):
            raise ValueError(f"{query_id}: acceptable_ground_truths must be an array.")

        if mode == "binary":
            primary = normalize_binary_gold(primary_raw)
            alternatives = [normalize_binary_gold(item) for item in alternatives_raw]
            expected = "Yes" if primary else "No"
        else:
            primary = normalize_structured_gold(
                primary_raw, schema, location=f"{query_id}/primary"
            )
            alternatives = [
                normalize_structured_gold(
                    item, schema, location=f"{query_id}/alternative-{index}"
                )
                for index, item in enumerate(alternatives_raw, start=1)
            ]
            expected = None

        record: dict[str, Any] = {
            "query_id": query_id,
            "question": question,
            "response_mode": mode,
            "ground_truth": primary,
        }
        if expected is not None:
            record["expected_answer"] = expected
        record["acceptable_ground_truths"] = alternatives
        normalized.append(record)
        by_query[query_id] = record

    normalized.sort(key=lambda item: query_sort_key(item["query_id"]))
    return normalized, by_query


def load_gold(
    args: argparse.Namespace, schema: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], Path]:
    paths = args.gold_sources if args.gold_sources else [args.gold]
    records = [record for path in paths for record in load_jsonl(path)]
    normalized, by_query = normalize_gold_records(records, schema)
    if args.gold_sources:
        write_jsonl(args.gold_out, normalized)
        gold_path = args.gold_out
    else:
        gold_path = args.gold
    return normalized, by_query, gold_path


def normalize_prediction_records(
    records: Sequence[dict[str, Any]],
    gold_by_query: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for sequence, raw in enumerate(records, start=1):
        query_id = raw.get("query_id")
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError("Every prediction requires a non-empty query_id.")
        if query_id not in gold_by_query:
            raise ValueError(f"Prediction uses unknown query_id: {query_id}")
        run_id_raw = raw.get("run_id", sequence)
        run_id = str(run_id_raw)
        key = (query_id, run_id)
        if key in seen:
            raise ValueError(f"Duplicate prediction key: {query_id}/run {run_id}")
        seen.add(key)

        mode = gold_by_query[query_id]["response_mode"]
        declared_mode = raw.get("response_mode")
        if declared_mode is not None and declared_mode != mode:
            raise ValueError(
                f"{query_id}/run {run_id}: prediction response_mode "
                f"{declared_mode!r} conflicts with gold mode {mode!r}."
            )
        normalized.append(
            {
                "query_id": query_id,
                "run_id": int(run_id) if run_id.isdigit() else run_id,
                "response_mode": mode,
                "answer": raw.get("answer"),
            }
        )

    normalized.sort(
        key=lambda item: (
            query_sort_key(item["query_id"]),
            run_sort_key(item["run_id"]),
        )
    )
    return normalized


def extract_predictions(
    args: argparse.Namespace,
    gold_records: list[dict[str, Any]],
    gold_by_query: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_question, _ = gold_question_map(gold_records)
    if args.manifest:
        selections = manifest_selections(args.manifest)
        skipped: list[dict[str, Any]] = []
    else:
        selections, skipped = automatic_selections(
            discover_paths(args.transcripts), by_question
        )

    records: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for declared_query_id, run_id, path in selections:
        if declared_query_id not in gold_by_query:
            raise ValueError(f"Manifest uses unknown query_id: {declared_query_id}")
        if not path.is_file():
            raise ValueError(f"Transcript does not exist: {path}")
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        actual_query_id, question = transcript_question(text, by_question)
        if actual_query_id != declared_query_id:
            raise ValueError(
                f"{path.name}: declared as {declared_query_id}, but its question "
                f"maps to {actual_query_id}."
            )

        mode = gold_by_query[declared_query_id]["response_mode"]
        if mode == "binary":
            answer, omitted_follow_up = primary_response(text)
            extraction_mode = (
                "bare_binary_before_followup" if omitted_follow_up else "bare_binary"
            )
            extraction_error = None if parse_binary(answer) is not None else (
                "The primary response is not a bare Yes/No answer."
            )
        else:
            answer, extraction_mode, extraction_error = final_response(text)
            omitted_follow_up = "before_followup" in extraction_mode

        records.append(
            {
                "query_id": declared_query_id,
                "run_id": int(run_id) if run_id.isdigit() else run_id,
                "answer": answer,
            }
        )
        sources.append(
            {
                "query_id": declared_query_id,
                "run_id": run_id,
                "response_mode": mode,
                "question": question,
                "source_file": str(path),
                "extraction_mode": extraction_mode,
                "extraction_error": extraction_error,
                "omitted_later_follow_up": omitted_follow_up,
            }
        )

    return normalize_prediction_records(records, gold_by_query), {
        "source": "transcripts" if args.transcripts else "manifest",
        "sources": sources,
        "skipped_files": skipped,
    }


def load_predictions(
    args: argparse.Namespace,
    gold_records: list[dict[str, Any]],
    gold_by_query: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if args.transcripts or args.manifest:
        predictions, build = extract_predictions(
            args, gold_records, gold_by_query
        )
    else:
        paths = (
            args.prediction_sources
            if args.prediction_sources
            else [args.predictions]
        )
        records = [record for path in paths for record in load_jsonl(path)]
        predictions = normalize_prediction_records(records, gold_by_query)
        build = {
            "source": "merged_prediction_files"
            if args.prediction_sources
            else "unified_prediction_file",
            "source_files": [str(path) for path in paths],
        }
    write_jsonl(args.predictions_out, predictions)
    return predictions, build


def expected_prediction_keys(
    query_ids: Iterable[str], expected_runs: int
) -> set[tuple[str, str]]:
    return {
        (query_id, str(run_id))
        for query_id in query_ids
        for run_id in range(1, expected_runs + 1)
    }


def coverage(
    gold_by_query: dict[str, dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    expected_runs: int,
) -> dict[str, Any]:
    expected = expected_prediction_keys(gold_by_query, expected_runs)
    actual = {
        (item["query_id"], str(item["run_id"])) for item in predictions
    }
    missing = sorted(
        expected - actual,
        key=lambda item: (query_sort_key(item[0]), run_sort_key(item[1])),
    )
    unexpected = sorted(
        actual - expected,
        key=lambda item: (query_sort_key(item[0]), run_sort_key(item[1])),
    )
    mode_counts = defaultdict(int)
    for record in gold_by_query.values():
        mode_counts[record["response_mode"]] += 1
    return {
        "gold_queries": len(gold_by_query),
        "structured_queries": mode_counts["structured"],
        "binary_queries": mode_counts["binary"],
        "expected_runs_per_query": expected_runs,
        "expected_predictions": len(expected),
        "available_predictions": len(actual.intersection(expected)),
        "coverage_rate": (
            len(actual.intersection(expected)) / len(expected) if expected else None
        ),
        "missing_query_runs": [
            {"query_id": query_id, "run_id": run_id}
            for query_id, run_id in missing
        ],
        "unexpected_query_runs": [
            {"query_id": query_id, "run_id": run_id}
            for query_id, run_id in unexpected
        ],
    }


def score_binary(
    candidates: Sequence[bool], prediction: JSONValue
) -> tuple[dict[str, Any], dict[str, Any]]:
    parsed = parse_binary(prediction)
    valid = parsed is not None
    correct = valid and parsed in candidates
    counts = Counts(
        tp=1 if correct else 0,
        fp=0 if correct or not valid else 1,
        fn=0 if correct else 1,
    )
    score = {
        "precision": counts.precision,
        "recall": counts.recall,
        "f1": counts.f1,
        "exact_answer": correct,
    }
    diagnostics = {
        "format_errors": [] if valid else ["Expected a bare Yes or No answer."],
        "fact_counts": {
            "gold": 1,
            "predicted": 1 if valid else 0,
            "tp": counts.tp,
            "fp": counts.fp,
            "fn": counts.fn,
        },
        "gold_answers": ["Yes" if item else "No" for item in candidates],
        "parsed_prediction": (
            None if parsed is None else ("Yes" if parsed else "No")
        ),
    }
    return {"format_valid": valid, **score}, diagnostics


def score_structured(
    candidates: Sequence[dict[str, Any]],
    prediction: JSONValue,
    schema: dict[str, Any],
    match: str,
    tolerance: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    predicted, parse_errors = coerce_answer(prediction)
    format_errors = parse_errors + validate_instance(predicted, schema, schema)
    valid = not format_errors
    selected, score = best_ground_truth(
        candidates, predicted, match, tolerance, valid
    )
    public_score = {
        "format_valid": valid,
        "precision": score["precision"],
        "recall": score["recall"],
        "f1": score["f1"],
        "exact_answer": score["exact_answer"],
    }
    diagnostics = {
        "selected_ground_truth": selected,
        "format_errors": format_errors,
        "fact_counts": score["fact_counts"],
        "answer_type_correct": score["answer_type_correct"],
        "status_correct": score["status_correct"],
        "order_sensitive": score["order_sensitive"],
        "path_serialization_normalized": score[
            "path_serialization_normalized"
        ],
        "value_serialization_normalized": score[
            "value_serialization_normalized"
        ],
    }
    return public_score, diagnostics


def evaluate(
    gold_by_query: dict[str, dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    schema: dict[str, Any],
    match: str,
    tolerance: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        query_id = prediction["query_id"]
        gold = gold_by_query[query_id]
        mode = gold["response_mode"]
        candidates = [gold["ground_truth"], *gold["acceptable_ground_truths"]]
        if mode == "binary":
            score, diagnostics = score_binary(candidates, prediction.get("answer"))
        else:
            score, diagnostics = score_structured(
                candidates,
                prediction.get("answer"),
                schema,
                match,
                tolerance,
            )
        rows.append(
            {
                "query_id": query_id,
                "category": category_of(query_id),
                "run_id": prediction["run_id"],
                "question": gold["question"],
                "response_mode": mode,
                **score,
                "diagnostics": diagnostics,
            }
        )
    rows.sort(
        key=lambda row: (
            query_sort_key(row["query_id"]),
            run_sort_key(row["run_id"]),
        )
    )
    return rows


def mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return sum(items) / len(items) if items else None


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    valid = sum(bool(row["format_valid"]) for row in rows)
    total = len(rows)
    return {
        "evaluated_runs": total,
        "format_validation": {
            "valid_predictions": valid,
            "invalid_predictions": total - valid,
            "validity_rate": valid / total if total else None,
        },
        "metrics": {
            "macro_precision": mean(float(row["precision"]) for row in rows),
            "macro_recall": mean(float(row["recall"]) for row in rows),
            "macro_f1": mean(float(row["f1"]) for row in rows),
            "exact_answer_accuracy": mean(
                float(row["exact_answer"]) for row in rows
            ),
        },
    }


def aggregate_by(
    rows: Sequence[dict[str, Any]], field: str
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    return {
        key: aggregate(items)
        for key, items in sorted(grouped.items(), key=lambda item: item[0])
    }


def aggregate_by_query(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["query_id"]].append(row)
    return {
        query_id: aggregate(grouped[query_id])
        for query_id in sorted(grouped, key=query_sort_key)
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = [
        "query_id",
        "category",
        "run_id",
        "question",
        "response_mode",
        "format_valid",
        "precision",
        "recall",
        "f1",
        "exact_answer",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def main() -> int:
    args = parse_args()
    if args.expected_runs < 1:
        raise ValueError("--expected-runs must be at least 1.")
    if args.numeric_tolerance < 0:
        raise ValueError("--numeric-tolerance must be non-negative.")

    schema = json.loads(args.schema.read_text(encoding="utf-8-sig"))
    gold_records, gold_by_query, gold_path = load_gold(args, schema)
    predictions, prediction_build = load_predictions(
        args, gold_records, gold_by_query
    )
    coverage_report = coverage(gold_by_query, predictions, args.expected_runs)
    missing = coverage_report["missing_query_runs"]
    unexpected = coverage_report["unexpected_query_runs"]
    if unexpected:
        raise ValueError(
            f"Found {len(unexpected)} prediction(s) outside the expected run set."
        )
    if missing and not args.allow_incomplete:
        preview = ", ".join(
            f"{item['query_id']}/run {item['run_id']}" for item in missing[:12]
        )
        suffix = " ..." if len(missing) > 12 else ""
        raise ValueError(
            f"Missing {len(missing)} expected prediction(s): {preview}{suffix}. "
            "Add the transcripts or use --allow-incomplete to evaluate only the "
            "available runs while retaining the coverage warning."
        )

    rows = evaluate(
        gold_by_query,
        predictions,
        schema,
        args.match,
        args.numeric_tolerance,
    )
    report = {
        "configuration": {
            "gold": str(gold_path),
            "predictions": str(args.predictions_out),
            "schema": str(args.schema),
            "match": args.match,
            "numeric_tolerance": args.numeric_tolerance,
            "response_mode_policy": (
                "Boolean gold answers are binary; JSON-object gold answers are "
                "structured. Question wording is not used for classification."
            ),
            "structured_scoring_unit": (
                "requested root entities and requested attributes, with normalized "
                "path/value handling; related_elements and relation strings ignored"
            ),
            "binary_scoring_unit": "one Yes/No fact per run",
            "missing_run_policy": (
                "reported but excluded from metrics when --allow-incomplete is used"
            ),
            "principal_metrics": [
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "exact_answer_accuracy",
            ],
        },
        "prediction_build": prediction_build,
        "coverage": coverage_report,
        "aggregate": aggregate(rows),
        "by_response_mode": aggregate_by(rows, "response_mode"),
        "by_category": aggregate_by(rows, "category"),
        "by_query": aggregate_by_query(rows),
        "by_run": aggregate_by(rows, "run_id"),
        "per_run": rows,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(args.csv, rows)

    print(
        json.dumps(
            {
                "coverage": coverage_report,
                "aggregate": report["aggregate"],
                "by_response_mode": report["by_response_mode"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"Gold: {gold_path}")
    print(f"Predictions: {args.predictions_out}")
    print(f"Report: {args.report}")
    print(f"Per-run CSV: {args.csv}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
