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
#   For questions marked with score_related_elements, the linked item is compared
#   as well. The relation name can be ignored when the question asks only which
#   items belong together. A gold record can also leave root entities or attributes
#   unscored when they are only context. Order is ignored unless the question asks
#   for it.
#   A communication question may keep participants as the root items while
#   treating matching "sent" and "received" descriptions as one directed message
#   flow. This prevents the same communication from being scored twice.
#   A message-flow answer may also identify an interaction point through its
#   unique non-pool "from" or "to" endpoint. When enabled by the gold record,
#   that endpoint is compared with the requested activity or event.
# - Values: compare the requested values and their units. Durations are converted to seconds 
#   before comparison.
# - Rankings: rank is an entity's position; rank 1 is the top position,
#   rank 2 is the next; tied entities share the same rank.
#   For example, individuals X and Y both have rank 1 because they
#   are tied for the largest number of assigned activities.
# - Paths: normally compare each path/route as an ordered sequence of BPMN elements.
#   A gold path can mark an ad-hoc section as unordered while retaining any
#   explicitly required order inside that section. For collaboration routes,
#   each pool keeps its own sequence, while elements from different pools may
#   interleave; a message send must still precede its matching receipt.
#   Inclusive-gateway branches may likewise be interleaved between their split
#   and join when the model does not order one branch before the other. Correct
#   elements receive credit when they respect the order that the model actually
#   defines. For example, if the gold path is A -> B -> C and the generated path is A -> C,
#   the two elements count as matches, while the missing one counts as a false negative.
#   Missing elements reduce recall, extra elements reduce precision and 
#   either can lower F1. Requested path costs or durations are also compared.
#   A path policy may keep selected events and gateways as optional context. Their
#   presence or absence then has no effect on the score, while the required order
#   between the remaining activities is still checked.
# - Yes/no answers: accept only "Yes" or "No" and check whether it matches the
#   gold answer.
#
# Entities are matched by identifier when possible, with labels as a fallback.
# For a scored relationship to a process or subprocess, matching labels can also
# bridge the subprocess object and its corresponding process-diagram identifier.
# related_elements are scored for explicitly marked gold questions and when they
# represent an ordered path. They are ignored for other ordinary answers. A
# marked gold record may also name contextual relation types that should remain
# unscored. It may also list plausible but optional related associations. Returning
# one of these is not an error, while leaving it out does not reduce recall. A
# marked question may use an attribute value as a fallback related-target label
# when that target was not already returned in related_elements. The value may
# equal the label or begin with it before a short explanation. The attribute name
# and unit are ignored by this fallback.
# Equivalent default-branch conditions are normalized. For example, "default
# path", "default process path" and "default path; no explicit condition is
# represented" express the same condition. A Boolean default_path attribute is
# ignored when it only clarifies a condition already given for the same result;
# other unexpected attributes remain extra scored facts.
# A structured answer must be valid JSON and follow the schema; otherwise, the run
# is marked invalid and its scores are zero.
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
# An allowed association is reported separately. It is a plausible optional
# answer: returning it is not a false positive, and omitting it is not a false
# negative.
#
# Precision measures how much of the generated answer is correct; recall measures
# how much of the gold answer was recovered; F1 balances precision and recall.
#
# exact_answer shows whether the whole answer is correct. For a structured answer,
# it is true only when all expected items and values are present, nothing extra is
# returned, and answer_type and status match the gold answer. When an attribute
# value supplies a related-element label, added explanatory wording still earns
# ordinary matching credit but prevents an exact answer.
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
PathPolicy = dict[str, Any]

PROCESS_CONTAINER_TYPES = frozenset(
    {
        "business_process_diagram_bpmn_2_0",
        "business_process",
        "process",
        "sub_process_bpmn",
        "subprocess_bpmn",
        "subprocess",
    }
)

POOL_TYPES = frozenset(
    {
        "pool_bpmn",
        "pool_collapsed_bpmn",
        "collapsed_pool_bpmn",
    }
)


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
class CommunicationFact:
    sender: Entity | None
    message: JSONValue
    receiver: Entity | None


@dataclass(frozen=True)
class CommunicationStatement:
    participant: Entity
    direction: str
    message: JSONValue
    counterpart_hint: str | None


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
        ("sends_to_", "sent_to_")
    ):
        return "sent_message"
    if name in {
        "receives",
        "received_message",
        "received_messages",
    } or name.startswith(("receives_from_", "received_from_")):
        return "received_message"
    return None


def communication_attribute_parts(value: Any) -> tuple[str | None, str | None]:
    """Return a communication direction and any participant named in the field."""
    name = canonical_attribute_name(value)
    if name.startswith(("sends_to_", "sent_to_")):
        prefix = "sends_to_" if name.startswith("sends_to_") else "sent_to_"
        return "sent_message", normalized_text(
            name.removeprefix(prefix).replace("_", " "), casefold=True
        )
    if name.startswith(("receives_from_", "received_from_")):
        prefix = (
            "receives_from_"
            if name.startswith("receives_from_")
            else "received_from_"
        )
        return "received_message", normalized_text(
            name.removeprefix(prefix).replace("_", " "), casefold=True
        )
    return canonical_communication_attribute_name(name), None


def canonical_condition_value(value: JSONValue) -> JSONValue:
    """Normalize equivalent wording for a default BPMN branch condition."""
    if not isinstance(value, str):
        return value
    normalized = normalized_text(value, casefold=True)
    if re.match(r"^default(?: process)? path(?:\b|$)", normalized):
        return "default path"
    return value


def atomic_attribute_values(name: Any, value: JSONValue) -> list[tuple[str, JSONValue]]:
    """Expand only known communication-list attributes into atomic facts."""
    communication_name = canonical_communication_attribute_name(name)
    if communication_name is None:
        canonical_name = canonical_attribute_name(name)
        if canonical_name == "condition":
            value = canonical_condition_value(value)
        return [(canonical_name, value)]
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
    factors = {
        "seconds": 1.0,
        "minutes": 60.0,
        "hours": 3600.0,
        "days": 86400.0,
    }
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
        component_pattern = re.compile(
            r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s+"
            r"(seconds?|minutes?|hours?|days?)",
            flags=re.IGNORECASE,
        )
        components = list(component_pattern.finditer(value))
        if components:
            cursor = 0
            complete = True
            for component in components:
                if value[cursor : component.start()].strip():
                    complete = False
                    break
                cursor = component.end()
            if complete and not value[cursor:].strip():
                return sum(
                    float(component.group(1))
                    * factors[canonical_unit(component.group(2))]
                    for component in components
                )

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
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


def related_target_matches(gold: Entity, predicted: Entity, mode: str) -> bool:
    """Match a related target, treating matching process labels as equivalent.

    Bee-Up can represent one subprocess both as a subprocess object and as its
    process diagram. Their identifiers and element types differ even though they
    name the same process. Identifier-only mode remains strict.
    """
    if entity_matches(gold, predicted, mode):
        return True
    if mode == "identifier":
        return False
    gold_type, predicted_type = element_type(gold), element_type(predicted)
    if (
        gold_type not in PROCESS_CONTAINER_TYPES
        or predicted_type not in PROCESS_CONTAINER_TYPES
    ):
        return False
    gold_label, predicted_label = label(gold), label(predicted)
    return (
        gold_label is not None
        and predicted_label is not None
        and gold_label == predicted_label
    )


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
    answer: dict[str, Any],
    *,
    order_sensitive: bool = False,
    include_root_entities: bool = True,
    include_attributes: bool = True,
    include_relations: bool = False,
    score_relation_names: bool = True,
    ignored_relations: frozenset[str] = frozenset(),
) -> list[AnswerFact]:
    """Extract the requested facts from a non-path answer.

    Relationships are included only when the gold record explicitly enables
    them. This keeps contextual related elements from affecting other questions.
    """
    facts: list[AnswerFact] = []
    for root in answer_results(answer):
        rank = root.get("rank")
        if include_root_entities:
            facts.append(
                RootFact(
                    root,
                    rank if order_sensitive and isinstance(rank, int) else None,
                )
            )

        if include_relations:
            related_elements = root.get("related_elements", [])
            if isinstance(related_elements, list):
                for related in related_elements:
                    if not isinstance(related, dict):
                        continue
                    relation = related.get("relation")
                    if not isinstance(relation, str):
                        continue
                    if relation in ignored_relations:
                        continue
                    order = related.get("order")
                    facts.append(
                        RelationFact(
                            root=root,
                            relation=relation if score_relation_names else "",
                            target=related,
                            order=order if isinstance(order, int) else None,
                        )
                    )

        attributes = root.get("attributes", [])
        if include_attributes and isinstance(attributes, list):
            has_condition = any(
                isinstance(attribute, dict)
                and canonical_attribute_name(attribute.get("name")) == "condition"
                for attribute in attributes
            )
            for attribute in attributes:
                if not isinstance(attribute, dict):
                    continue
                attribute_name = canonical_attribute_name(attribute.get("name"))
                if (
                    has_condition
                    and attribute_name == "default_path"
                    and isinstance(attribute.get("value"), bool)
                ):
                    # This Boolean only clarifies the condition already given
                    # for the same outcome; it is not a second condition fact.
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


def allowed_relation_facts(
    associations: Sequence[dict[str, Any]],
    *,
    score_relation_names: bool,
) -> list[RelationFact]:
    """Convert optional allowed associations into matchable relation facts."""
    return [
        RelationFact(
            root=association["root"],
            relation=(association["relation"] if score_relation_names else ""),
            target=association["target"],
            order=association["order"],
        )
        for association in associations
    ]


def attribute_value_relation_facts(
    answer: dict[str, Any],
    gold_relations: Sequence[RelationFact],
    predicted_relations: Sequence[RelationFact],
    mode: str,
    tolerance: float,
) -> tuple[list[RelationFact], int]:
    """Use matching attribute values as missing related-target labels."""
    covered_gold = {
        gold_index
        for gold_index, _ in maximum_matching(
            gold_relations,
            predicted_relations,
            lambda left, right: fact_matches(left, right, mode, tolerance),
        )
    }
    fallbacks: list[RelationFact] = []
    non_exact_labels = 0
    for root in answer_results(answer):
        attributes = root.get("attributes", [])
        if not isinstance(attributes, list):
            continue
        for attribute in attributes:
            if not isinstance(attribute, dict):
                continue
            value = attribute.get("value")
            if not isinstance(value, str) or not value.strip():
                continue
            normalized_value = normalized_text(value, casefold=True)
            for gold_index, gold_relation in enumerate(gold_relations):
                expected_label = label(gold_relation.target)
                if (
                    gold_index in covered_gold
                    or expected_label is None
                    or not entity_matches(gold_relation.root, root, mode)
                    or not (
                        normalized_value == expected_label
                        or normalized_value.startswith(f"{expected_label} ")
                    )
                ):
                    continue
                candidate = RelationFact(
                    root=root,
                    relation="",
                    target={
                        "identifier": None,
                        "label": gold_relation.target.get("label"),
                        "element_type": None,
                    },
                    order=None,
                )
                fallbacks.append(candidate)
                if normalized_value != expected_label:
                    non_exact_labels += 1
                covered_gold.add(gold_index)
                break
    return fallbacks, non_exact_labels


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
            gold.relation == predicted.relation
            and (gold.order is None or gold.order == predicted.order)
            and entity_matches(gold.root, predicted.root, mode)
            and related_target_matches(gold.target, predicted.target, mode)
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
    if isinstance(gold, CommunicationFact) and isinstance(
        predicted, CommunicationFact
    ):
        return (
            gold.sender is not None
            and predicted.sender is not None
            and gold.receiver is not None
            and predicted.receiver is not None
            and entity_matches(gold.sender, predicted.sender, mode)
            and entity_matches(gold.receiver, predicted.receiver, mode)
            and scalar_matches(gold.message, predicted.message, tolerance)
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
    allowed: Sequence[AnswerFact] = (),
) -> tuple[Counts, int]:
    required_pairs = maximum_matching(
        gold,
        predicted,
        lambda left, right: fact_matches(left, right, mode, tolerance),
    )
    matched_prediction_indexes = {
        predicted_index for _, predicted_index in required_pairs
    }
    unmatched_predictions = [
        fact
        for index, fact in enumerate(predicted)
        if index not in matched_prediction_indexes
    ]
    allowed_pairs = maximum_matching(
        allowed,
        unmatched_predictions,
        lambda left, right: fact_matches(left, right, mode, tolerance),
    )
    tp = len(required_pairs)
    allowed_count = len(allowed_pairs)
    counts = Counts(
        tp=tp,
        fp=len(predicted) - tp - allowed_count,
        fn=len(gold) - tp,
    )
    return counts, allowed_count


def communication_statements(answer: dict[str, Any]) -> list[CommunicationStatement]:
    """Read sent and received message descriptions without scoring them twice."""
    statements: list[CommunicationStatement] = []
    for participant in answer_results(answer):
        attributes = participant.get("attributes", [])
        if not isinstance(attributes, list):
            continue
        for attribute in attributes:
            if not isinstance(attribute, dict):
                continue
            direction, counterpart_hint = communication_attribute_parts(
                attribute.get("name")
            )
            if direction is None:
                continue
            for _, message in atomic_attribute_values(
                attribute.get("name"), attribute.get("value")
            ):
                statements.append(
                    CommunicationStatement(
                        participant=participant,
                        direction=direction,
                        message=message,
                        counterpart_hint=counterpart_hint,
                    )
                )
    return statements


def resolve_participant_hint(
    hint: str | None, participants: Sequence[Entity]
) -> Entity | None:
    if hint is None:
        return None
    matches = [participant for participant in participants if label(participant) == hint]
    return matches[0] if len(matches) == 1 else None


def communication_peers(
    participant: Entity,
    participants: Sequence[Entity],
    mode: str,
) -> list[Entity]:
    peers: list[Entity] = []
    related_elements = participant.get("related_elements", [])
    if not isinstance(related_elements, list):
        return peers
    for related in related_elements:
        if (
            not isinstance(related, dict)
            or canonical_attribute_name(related.get("relation"))
            != "communicates_with"
        ):
            continue
        for candidate in participants:
            if entity_matches(related, candidate, mode) and not any(
                entity_matches(candidate, existing, mode) for existing in peers
            ):
                peers.append(candidate)
    return peers


def communication_facts(
    answer: dict[str, Any],
    mode: str,
    tolerance: float,
) -> list[CommunicationFact]:
    """Collapse mirrored send/receive descriptions into directed message facts."""
    participants = answer_results(answer)
    statements = communication_statements(answer)
    sent = [item for item in statements if item.direction == "sent_message"]
    received = [
        item for item in statements if item.direction == "received_message"
    ]

    peer_cache = {
        id(participant): communication_peers(participant, participants, mode)
        for participant in participants
    }

    def statement_pair_matches(
        sent_item: CommunicationStatement,
        received_item: CommunicationStatement,
    ) -> bool:
        if entity_matches(
            sent_item.participant, received_item.participant, mode
        ) or not scalar_matches(
            sent_item.message, received_item.message, tolerance
        ):
            return False
        sent_hint = resolve_participant_hint(
            sent_item.counterpart_hint, participants
        )
        received_hint = resolve_participant_hint(
            received_item.counterpart_hint, participants
        )
        if sent_hint is not None and not entity_matches(
            sent_hint, received_item.participant, mode
        ):
            return False
        if received_hint is not None and not entity_matches(
            received_hint, sent_item.participant, mode
        ):
            return False
        sent_peers = peer_cache[id(sent_item.participant)]
        received_peers = peer_cache[id(received_item.participant)]
        if sent_peers and not any(
            entity_matches(received_item.participant, peer, mode)
            for peer in sent_peers
        ):
            return False
        if received_peers and not any(
            entity_matches(sent_item.participant, peer, mode)
            for peer in received_peers
        ):
            return False
        return True

    pairs = maximum_matching(sent, received, statement_pair_matches)
    matched_sent = {left for left, _ in pairs}
    matched_received = {right for _, right in pairs}
    facts = [
        CommunicationFact(
            sender=sent[left].participant,
            message=sent[left].message,
            receiver=received[right].participant,
        )
        for left, right in pairs
    ]

    def inferred_counterpart(statement: CommunicationStatement) -> Entity | None:
        hinted = resolve_participant_hint(statement.counterpart_hint, participants)
        if hinted is not None:
            return hinted
        peers = peer_cache[id(statement.participant)]
        return peers[0] if len(peers) == 1 else None

    for index, item in enumerate(sent):
        if index not in matched_sent:
            facts.append(
                CommunicationFact(
                    sender=item.participant,
                    message=item.message,
                    receiver=inferred_counterpart(item),
                )
            )
    for index, item in enumerate(received):
        if index not in matched_received:
            facts.append(
                CommunicationFact(
                    sender=inferred_counterpart(item),
                    message=item.message,
                    receiver=item.participant,
                )
            )
    return facts


def communication_answer_facts(
    answer: dict[str, Any],
    mode: str,
    tolerance: float,
    *,
    include_root_entities: bool,
) -> list[AnswerFact]:
    facts: list[AnswerFact] = []
    if include_root_entities:
        facts.extend(RootFact(participant, None) for participant in answer_results(answer))
    facts.extend(communication_facts(answer, mode, tolerance))
    return facts


def is_pool_entity(entity: Entity) -> bool:
    if element_type(entity) in POOL_TYPES:
        return True
    entity_identifier = identifier(entity)
    if entity_identifier is None:
        return False
    local_name = entity_identifier.rsplit("#", maxsplit=1)[-1].casefold()
    return local_name.startswith(("pool_bpmn-", "pool_collapsed_bpmn-"))


def is_message_flow_entity(entity: Entity) -> bool:
    if element_type(entity) == "message_flow_bpmn":
        return True
    entity_identifier = identifier(entity)
    if entity_identifier is None:
        return False
    local_name = entity_identifier.rsplit("#", maxsplit=1)[-1].casefold()
    return local_name.startswith("message_flow_bpmn-")


def message_flow_endpoint_root_facts(
    answer: dict[str, Any],
    *,
    order_sensitive: bool,
) -> list[AnswerFact]:
    """Use a message flow's single non-pool from/to endpoint as its root fact."""
    facts: list[AnswerFact] = []
    for root in answer_results(answer):
        effective_root = root
        if is_message_flow_entity(root):
            related_elements = root.get("related_elements", [])
            endpoints = [
                related
                for related in related_elements
                if isinstance(related, dict)
                and canonical_attribute_name(related.get("relation"))
                in {"from", "to"}
                and not is_pool_entity(related)
            ] if isinstance(related_elements, list) else []
            if len(endpoints) == 1:
                effective_root = endpoints[0]
        rank = root.get("rank")
        facts.append(
            RootFact(
                effective_root,
                rank if order_sensitive and isinstance(rank, int) else None,
            )
        )
    return facts


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


def normalize_path_policy(
    raw: Any,
    answer: dict[str, Any],
    *,
    location: str,
) -> PathPolicy:
    """Validate evaluator-only ordering rules for one gold path answer."""
    if raw is None or raw == {}:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{location}: path policy must be an object.")
    if answer.get("answer_type") != "path":
        raise ValueError(f"{location}: path policy requires answer_type='path'.")

    allowed_keys = {
        "single_path_prediction",
        "ignored_predicted_elements",
        "optional_path_elements",
        "optional_path_element_types",
        "unordered_groups",
    }
    unknown = sorted(set(raw) - allowed_keys)
    if unknown:
        raise ValueError(f"{location}: unknown path-policy fields: {unknown}.")

    single_path = raw.get("single_path_prediction", False)
    if not isinstance(single_path, bool):
        raise ValueError(
            f"{location}/single_path_prediction: value must be true or false."
        )

    ignored_raw = raw.get("ignored_predicted_elements", [])
    if not isinstance(ignored_raw, list) or not all(
        isinstance(item, str) and item.strip() for item in ignored_raw
    ):
        raise ValueError(
            f"{location}/ignored_predicted_elements: expected an array of "
            "non-empty identifiers."
        )
    ignored = [normalized_text(item) for item in ignored_raw]
    if len(ignored) != len(set(ignored)):
        raise ValueError(
            f"{location}/ignored_predicted_elements: duplicate identifier."
        )

    optional_raw = raw.get("optional_path_elements", [])
    if not isinstance(optional_raw, list) or not all(
        isinstance(item, str) and item.strip() for item in optional_raw
    ):
        raise ValueError(
            f"{location}/optional_path_elements: expected an array of "
            "non-empty identifiers."
        )
    optional = [normalized_text(item) for item in optional_raw]
    if len(optional) != len(set(optional)):
        raise ValueError(
            f"{location}/optional_path_elements: duplicate identifier."
        )

    optional_types_raw = raw.get("optional_path_element_types", [])
    if not isinstance(optional_types_raw, list) or not all(
        isinstance(item, str) and item.strip() for item in optional_types_raw
    ):
        raise ValueError(
            f"{location}/optional_path_element_types: expected an array of "
            "non-empty element types."
        )
    optional_types = [normalized_text(item) for item in optional_types_raw]
    if len(optional_types) != len(set(optional_types)):
        raise ValueError(
            f"{location}/optional_path_element_types: duplicate element type."
        )

    answer_paths = path_views(answer)
    positions: dict[str, list[tuple[int, int]]] = defaultdict(list)
    matched_optional_types: set[str] = set()
    for path_index, path in enumerate(answer_paths):
        for step_index, step in enumerate(path.steps):
            step_id = identifier(step)
            if step_id is not None:
                positions[step_id].append((path_index, step_index))
                step_type = element_type(step)
                if step_type in optional_types:
                    optional.append(step_id)
                    matched_optional_types.add(step_type)

    unmatched_optional_types = set(optional_types) - matched_optional_types
    if unmatched_optional_types:
        raise ValueError(
            f"{location}/optional_path_element_types: no gold path element "
            f"uses {sorted(unmatched_optional_types)}."
        )
    optional = list(dict.fromkeys(optional))
    overlap = set(ignored) & set(optional)
    if overlap:
        raise ValueError(
            f"{location}: ignored and optional path elements overlap at "
            f"{sorted(overlap)}."
        )

    def unique_position(step_id: str, item_location: str) -> tuple[int, int]:
        found = positions.get(step_id, [])
        if len(found) != 1:
            detail = "not found" if not found else "not unique"
            raise ValueError(
                f"{item_location}: identifier is {detail} in this gold answer: "
                f"{step_id!r}."
            )
        return found[0]

    for step_id in ignored:
        if step_id in positions:
            raise ValueError(
                f"{location}/ignored_predicted_elements: an expected gold step "
                f"cannot be ignored: {step_id!r}."
            )

    for step_id in optional:
        unique_position(step_id, f"{location}/optional_path_elements")

    groups_raw = raw.get("unordered_groups", [])
    if not isinstance(groups_raw, list):
        raise ValueError(f"{location}/unordered_groups: expected an array.")

    groups: list[dict[str, Any]] = []
    all_group_members: set[str] = set()
    for group_index, group_raw in enumerate(groups_raw, start=1):
        group_location = f"{location}/unordered_groups/{group_index}"
        if not isinstance(group_raw, dict):
            raise ValueError(f"{group_location}: group must be an object.")
        group_unknown = sorted(
            set(group_raw)
            - {
                "after",
                "before",
                "members",
                "ordered_subgroups",
                "required_precedence",
            }
        )
        if group_unknown:
            raise ValueError(
                f"{group_location}: unknown fields: {group_unknown}."
            )

        after_raw = group_raw.get("after")
        before_raw = group_raw.get("before")
        if not isinstance(after_raw, str) or not after_raw.strip():
            raise ValueError(f"{group_location}/after: identifier is required.")
        if before_raw is not None and (
            not isinstance(before_raw, str) or not before_raw.strip()
        ):
            raise ValueError(
                f"{group_location}/before: expected an identifier or null."
            )
        after = normalized_text(after_raw)
        before = normalized_text(before_raw) if before_raw is not None else None

        members_raw = group_raw.get("members")
        if not isinstance(members_raw, list) or len(members_raw) < 2 or not all(
            isinstance(item, str) and item.strip() for item in members_raw
        ):
            raise ValueError(
                f"{group_location}/members: expected at least two identifiers."
            )
        members = [normalized_text(item) for item in members_raw]
        member_set = set(members)
        if len(members) != len(member_set):
            raise ValueError(f"{group_location}/members: duplicate identifier.")
        overlap = all_group_members & member_set
        if overlap:
            raise ValueError(
                f"{group_location}/members: groups overlap at {sorted(overlap)}."
            )

        member_positions = [
            unique_position(item, f"{group_location}/members") for item in members
        ]
        path_indexes = {path_index for path_index, _ in member_positions}
        if len(path_indexes) != 1:
            raise ValueError(
                f"{group_location}: all members must belong to the same gold path."
            )
        path_index = next(iter(path_indexes))
        step_indexes = {step_index for _, step_index in member_positions}
        skipped_indexes = (
            set(range(min(step_indexes), max(step_indexes) + 1)) - step_indexes
        )
        skipped_ids = {
            identifier(answer_paths[path_index].steps[step_index])
            for step_index in skipped_indexes
        }
        if skipped_ids - set(optional):
            raise ValueError(
                f"{group_location}: members must form one gold section; only "
                "optional path elements may occur between them."
            )

        after_position = unique_position(after, f"{group_location}/after")
        before_position = (
            unique_position(before, f"{group_location}/before")
            if before is not None
            else None
        )
        invalid_bounds = (
            after_position[0] != path_index
            or after_position[1] >= min(step_indexes)
            or (
                before_position is not None
                and (
                    before_position[0] != path_index
                    or before_position[1] <= max(step_indexes)
                )
            )
            or (
                before_position is None
                and any(
                    identifier(answer_paths[path_index].steps[step_index])
                    not in set(optional)
                    for step_index in range(
                        max(step_indexes) + 1,
                        len(answer_paths[path_index].steps),
                    )
                )
            )
        )
        if invalid_bounds:
            raise ValueError(
                f"{group_location}: after and before must bound the unordered section."
            )

        subgroups_raw = group_raw.get("ordered_subgroups", [])
        if not isinstance(subgroups_raw, list):
            raise ValueError(
                f"{group_location}/ordered_subgroups: expected an array."
            )
        ordered_subgroups: list[list[str]] = []
        subgroup_members: set[str] = set()
        for subgroup_index, subgroup_raw in enumerate(subgroups_raw, start=1):
            subgroup_location = (
                f"{group_location}/ordered_subgroups/{subgroup_index}"
            )
            if not isinstance(subgroup_raw, list) or len(subgroup_raw) < 2 or not all(
                isinstance(item, str) and item.strip() for item in subgroup_raw
            ):
                raise ValueError(
                    f"{subgroup_location}: expected at least two identifiers."
                )
            subgroup = [normalized_text(item) for item in subgroup_raw]
            subgroup_set = set(subgroup)
            if len(subgroup) != len(subgroup_set):
                raise ValueError(f"{subgroup_location}: duplicate identifier.")
            if not subgroup_set <= member_set:
                raise ValueError(
                    f"{subgroup_location}: every identifier must also be a group member."
                )
            subgroup_overlap = subgroup_members & subgroup_set
            if subgroup_overlap:
                raise ValueError(
                    f"{subgroup_location}: ordered subgroups overlap at "
                    f"{sorted(subgroup_overlap)}."
                )
            subgroup_members.update(subgroup_set)
            ordered_subgroups.append(subgroup)

        precedence_raw = group_raw.get("required_precedence", [])
        if not isinstance(precedence_raw, list):
            raise ValueError(
                f"{group_location}/required_precedence: expected an array."
            )
        required_precedence: list[dict[str, str]] = []
        seen_precedence: set[tuple[str, str]] = set()
        for precedence_index, precedence_item in enumerate(
            precedence_raw, start=1
        ):
            precedence_location = (
                f"{group_location}/required_precedence/{precedence_index}"
            )
            if not isinstance(precedence_item, dict) or set(
                precedence_item
            ) != {"before", "after"}:
                raise ValueError(
                    f"{precedence_location}: expected before and after identifiers."
                )
            earlier_raw = precedence_item["before"]
            later_raw = precedence_item["after"]
            if not isinstance(earlier_raw, str) or not earlier_raw.strip():
                raise ValueError(
                    f"{precedence_location}/before: identifier is required."
                )
            if not isinstance(later_raw, str) or not later_raw.strip():
                raise ValueError(
                    f"{precedence_location}/after: identifier is required."
                )
            earlier = normalized_text(earlier_raw)
            later = normalized_text(later_raw)
            if earlier == later or earlier not in member_set or later not in member_set:
                raise ValueError(
                    f"{precedence_location}: both different identifiers must be "
                    "members of this unordered group."
                )
            pair = (earlier, later)
            if pair in seen_precedence:
                raise ValueError(f"{precedence_location}: duplicate constraint.")
            seen_precedence.add(pair)
            required_precedence.append({"before": earlier, "after": later})

        all_group_members.update(member_set)
        groups.append(
            {
                "after": after,
                "before": before,
                "members": members,
                "ordered_subgroups": ordered_subgroups,
                "required_precedence": required_precedence,
            }
        )

    return {
        "single_path_prediction": single_path,
        "ignored_predicted_elements": ignored,
        "optional_path_elements": optional,
        "optional_path_element_types": optional_types,
        "unordered_groups": groups,
    }


def path_views_for_policy(
    answer: dict[str, Any], policy: PathPolicy, *, predicted: bool
) -> list[PathView]:
    """Build path views, optionally keeping one flat prediction as one route."""
    if not predicted or not policy.get("single_path_prediction", False):
        return path_views(answer)
    roots = answer_results(answer)
    if not roots:
        return []
    if any(
        isinstance(root.get("related_elements"), list)
        and any(isinstance(item, dict) for item in root["related_elements"])
        for root in roots
    ):
        return path_views(answer)
    return [
        PathView(
            steps=tuple(roots),
            attributes=tuple(
                scalar
                for root in roots
                for scalar in root_scalar_facts(root)
            ),
        )
    ]


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


def matches_any_path_entity(
    candidate: Entity, expected: Sequence[Entity], mode: str
) -> bool:
    return any(path_entity_matches(item, candidate, mode) for item in expected)


def policy_filtered_steps(
    steps: Sequence[Entity], policy: PathPolicy, *, gold: bool = False
) -> tuple[Entity, ...]:
    excluded = set(policy.get("optional_path_elements", []))
    if not gold:
        excluded.update(policy.get("ignored_predicted_elements", []))
    return tuple(step for step in steps if identifier(step) not in excluded)


def projected_path_policy(
    gold_steps: Sequence[Entity], policy: PathPolicy
) -> PathPolicy:
    """Remove optional nodes from ordering rules while preserving task order."""
    optional = set(policy.get("optional_path_elements", []))
    if not optional:
        return policy

    gold_ids = [identifier(step) for step in gold_steps]
    positions = {
        step_id: index
        for index, step_id in enumerate(gold_ids)
        if step_id is not None
    }
    scored_ids = {
        step_id
        for step_id in gold_ids
        if step_id is not None and step_id not in optional
    }
    projected_groups: list[dict[str, Any]] = []

    for group in policy.get("unordered_groups", []):
        original_members = group["members"]
        members = [
            step_id for step_id in original_members if step_id in scored_ids
        ]
        if len(members) < 2:
            continue

        first = min(positions[step_id] for step_id in original_members)
        last = max(positions[step_id] for step_id in original_members)
        previous = [
            step_id
            for step_id in gold_ids[:first]
            if step_id is not None and step_id in scored_ids
        ]
        following = [
            step_id
            for step_id in gold_ids[last + 1 :]
            if step_id is not None and step_id in scored_ids
        ]

        ordered_subgroups = []
        for subgroup in group["ordered_subgroups"]:
            projected = [step_id for step_id in subgroup if step_id in scored_ids]
            if len(projected) >= 2:
                ordered_subgroups.append(projected)

        edges: dict[str, set[str]] = {
            step_id: set() for step_id in original_members
        }
        for subgroup in group["ordered_subgroups"]:
            for earlier, later in zip(subgroup, subgroup[1:]):
                edges[earlier].add(later)
        for precedence in group.get("required_precedence", []):
            edges[precedence["before"]].add(precedence["after"])

        reachable: dict[str, set[str]] = {}
        for start in original_members:
            seen: set[str] = set()
            stack = list(edges[start])
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                stack.extend(edges[current] - seen)
            reachable[start] = seen

        subgroup_order = {
            (earlier, later)
            for subgroup in ordered_subgroups
            for earlier_index, earlier in enumerate(subgroup)
            for later in subgroup[earlier_index + 1 :]
        }
        required_precedence = []
        for earlier in members:
            for later in members:
                if later not in reachable[earlier] or (earlier, later) in subgroup_order:
                    continue
                has_scored_middle = any(
                    middle not in {earlier, later}
                    and middle in reachable[earlier]
                    and later in reachable[middle]
                    for middle in members
                )
                if not has_scored_middle:
                    required_precedence.append(
                        {"before": earlier, "after": later}
                    )

        projected_groups.append(
            {
                "after": previous[-1] if previous else None,
                "before": following[0] if following else None,
                "members": members,
                "ordered_subgroups": ordered_subgroups,
                "required_precedence": required_precedence,
            }
        )

    return {
        **policy,
        "unordered_groups": projected_groups,
    }


def partial_order_step_score(
    gold_steps: Sequence[Entity],
    predicted_steps: Sequence[Entity],
    mode: str,
    policy: PathPolicy,
) -> int:
    """Match one path against its declared partial-order constraints."""
    gold_by_id = {
        step_id: step
        for step in gold_steps
        if (step_id := identifier(step)) is not None
    }
    relevant_groups = [
        group
        for group in policy.get("unordered_groups", [])
        if set(group["members"]) <= set(gold_by_id)
    ]
    if not relevant_groups:
        return len(lcs_entity_pairs(gold_steps, predicted_steps, mode))

    group_entities = [
        gold_by_id[step_id]
        for group in relevant_groups
        for step_id in group["members"]
    ]
    outside_gold = [
        step
        for step in gold_steps
        if identifier(step)
        not in {
            step_id
            for group in relevant_groups
            for step_id in group["members"]
        }
    ]
    outside_predicted = [
        step
        for step in predicted_steps
        if not matches_any_path_entity(step, group_entities, mode)
    ]
    score = len(lcs_entity_pairs(outside_gold, outside_predicted, mode))

    for group in relevant_groups:
        members = [gold_by_id[step_id] for step_id in group["members"]]
        after_id = group["after"]
        after = gold_by_id[after_id] if after_id is not None else None
        before_id = group["before"]
        before = gold_by_id[before_id] if before_id is not None else None

        after_index = (
            next(
                (
                    index
                    for index, step in enumerate(predicted_steps)
                    if path_entity_matches(after, step, mode)
                ),
                -1,
            )
            if after is not None
            else -1
        )
        before_index = (
            next(
                (
                    index
                    for index, step in enumerate(predicted_steps)
                    if index > after_index
                    and path_entity_matches(before, step, mode)
                ),
                len(predicted_steps),
            )
            if before is not None
            else len(predicted_steps)
        )
        eligible = [
            step
            for index, step in enumerate(predicted_steps)
            if after_index < index < before_index
            and matches_any_path_entity(step, members, mode)
        ]

        ordered_ids = {
            step_id
            for subgroup in group["ordered_subgroups"]
            for step_id in subgroup
        }
        ordered_entities = [gold_by_id[step_id] for step_id in ordered_ids]
        group_score = 0
        for subgroup in group["ordered_subgroups"]:
            subgroup_gold = [gold_by_id[step_id] for step_id in subgroup]
            subgroup_predicted = [
                step
                for step in eligible
                if matches_any_path_entity(step, subgroup_gold, mode)
            ]
            group_score += len(
                lcs_entity_pairs(subgroup_gold, subgroup_predicted, mode)
            )

        unordered_gold = [
            gold_by_id[step_id]
            for step_id in group["members"]
            if step_id not in ordered_ids
        ]
        unordered_predicted = [
            step
            for step in eligible
            if not matches_any_path_entity(step, ordered_entities, mode)
        ]
        group_score += len(
            maximum_matching(
                unordered_gold,
                unordered_predicted,
                lambda left, right: path_entity_matches(left, right, mode),
            )
        )
        for precedence in group.get("required_precedence", []):
            earlier = gold_by_id[precedence["before"]]
            later = gold_by_id[precedence["after"]]
            earlier_positions = [
                index
                for index, step in enumerate(eligible)
                if path_entity_matches(earlier, step, mode)
            ]
            later_positions = [
                index
                for index, step in enumerate(eligible)
                if path_entity_matches(later, step, mode)
            ]
            if (
                earlier_positions
                and later_positions
                and not any(
                    earlier_index < later_index
                    for earlier_index in earlier_positions
                    for later_index in later_positions
                )
            ):
                group_score = max(0, group_score - 1)
        score += group_score
    return score


def path_pair_score(
    gold: PathView,
    predicted: PathView,
    mode: str,
    tolerance: float,
    score_attributes: bool,
    policy: PathPolicy,
) -> int:
    gold_steps = policy_filtered_steps(gold.steps, policy, gold=True)
    predicted_steps = policy_filtered_steps(predicted.steps, policy)
    effective_policy = projected_path_policy(gold.steps, policy)
    step_matches = partial_order_step_score(
        gold_steps, predicted_steps, mode, effective_policy
    )
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
    policy: PathPolicy,
) -> Counts:
    """Score ordered routes and any route metric requested by the gold answer."""
    gold_paths = path_views_for_policy(gold, policy, predicted=False)
    predicted_paths = path_views_for_policy(predicted, policy, predicted=True)
    score_attributes = any(path.attributes for path in gold_paths)
    gold_total = sum(
        len(policy_filtered_steps(path.steps, policy, gold=True))
        + (len(path.attributes) if score_attributes else 0)
        for path in gold_paths
    )
    predicted_total = sum(
        len(policy_filtered_steps(path.steps, policy))
        + (len(path.attributes) if score_attributes else 0)
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
                policy,
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
    score_root_entities: bool,
    score_attributes: bool,
    score_related_elements: bool,
    score_related_relation_names: bool,
    ignored_related_relations: frozenset[str],
    allowed_related_associations: Sequence[dict[str, Any]],
    attribute_values_as_related_labels: bool,
    score_communication_flows_once: bool,
    message_flow_endpoints_as_roots: bool,
    path_policy: PathPolicy,
) -> dict[str, Any]:
    order_sensitive = answer_is_order_sensitive(gold)
    path_scoring = gold.get("answer_type") == "path"
    value_scoring = gold.get("answer_type") == "value"
    attribute_fallback_count = 0
    non_exact_attribute_label_count = 0
    if path_scoring:
        gold_paths = path_views_for_policy(gold, path_policy, predicted=False)
        predicted_paths = path_views_for_policy(
            predicted, path_policy, predicted=True
        )
        score_path_attributes = any(path.attributes for path in gold_paths)
        gold_fact_count = sum(
            len(policy_filtered_steps(path.steps, path_policy, gold=True))
            + (len(path.attributes) if score_path_attributes else 0)
            for path in gold_paths
        )
        predicted_fact_count = sum(
            len(policy_filtered_steps(path.steps, path_policy))
            + (len(path.attributes) if score_path_attributes else 0)
            for path in predicted_paths
        )
    elif value_scoring:
        gold_value_facts = scalar_facts(gold)
        predicted_value_facts = scalar_facts(predicted)
        gold_fact_count = len(gold_value_facts)
        predicted_fact_count = len(predicted_value_facts)
    else:
        if score_communication_flows_once:
            gold_facts = communication_answer_facts(
                gold,
                mode,
                tolerance,
                include_root_entities=score_root_entities,
            )
            predicted_facts = communication_answer_facts(
                predicted,
                mode,
                tolerance,
                include_root_entities=score_root_entities,
            )
            allowed_facts = []
        elif message_flow_endpoints_as_roots:
            gold_facts = message_flow_endpoint_root_facts(
                gold, order_sensitive=order_sensitive
            )
            predicted_facts = message_flow_endpoint_root_facts(
                predicted, order_sensitive=order_sensitive
            )
            allowed_facts = []
        else:
            gold_facts = answer_facts(
                gold,
                order_sensitive=order_sensitive,
                include_root_entities=score_root_entities,
                include_attributes=score_attributes,
                include_relations=score_related_elements,
                score_relation_names=score_related_relation_names,
                ignored_relations=ignored_related_relations,
            )
            predicted_facts = answer_facts(
                predicted,
                order_sensitive=order_sensitive,
                include_root_entities=score_root_entities,
                include_attributes=score_attributes,
                include_relations=score_related_elements,
                score_relation_names=score_related_relation_names,
                ignored_relations=ignored_related_relations,
            )
            allowed_facts = allowed_relation_facts(
                allowed_related_associations,
                score_relation_names=score_related_relation_names,
            )
        if attribute_values_as_related_labels:
            fallback_facts, non_exact_attribute_label_count = (
                attribute_value_relation_facts(
                    predicted,
                    [
                        fact
                        for fact in gold_facts
                        if isinstance(fact, RelationFact)
                    ],
                    [
                        fact
                        for fact in predicted_facts
                        if isinstance(fact, RelationFact)
                    ],
                    mode,
                    tolerance,
                )
            )
            predicted_facts.extend(fallback_facts)
            attribute_fallback_count = len(fallback_facts)
        gold_fact_count = len(gold_facts)
        predicted_fact_count = len(predicted_facts)
    answer_type_correct = gold.get("answer_type") == predicted.get("answer_type")
    status_correct = gold.get("status") == predicted.get("status")

    if schema_valid:
        if path_scoring:
            counts = score_path_answers(
                gold, predicted, mode, tolerance, path_policy
            )
        elif value_scoring:
            counts = score_scalar_answers(gold, predicted, tolerance)
            allowed_count = 0
        else:
            counts, allowed_count = score_fact_sets(
                gold_facts,
                predicted_facts,
                mode,
                tolerance,
                allowed_facts,
            )
        if path_scoring:
            allowed_count = 0
        precision = counts.precision
        recall = counts.recall
        f1 = counts.f1
        exact_answer = (
            counts.exact
            and answer_type_correct
            and status_correct
            and non_exact_attribute_label_count == 0
        )
    else:
        counts = Counts(
            tp=0,
            fp=predicted_fact_count,
            fn=gold_fact_count,
        )
        precision = recall = f1 = 0.0
        exact_answer = False
        allowed_count = 0

    return {
        "fact_counts": {
            "gold": gold_fact_count,
            "predicted": predicted_fact_count,
            "tp": counts.tp,
            "fp": counts.fp,
            "fn": counts.fn,
            "allowed": allowed_count,
        },
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_answer": exact_answer,
        "answer_type_correct": answer_type_correct,
        "status_correct": status_correct,
        "order_sensitive": order_sensitive,
        "root_entities_scored": score_root_entities,
        "attributes_scored": score_attributes,
        "related_elements_scored": score_related_elements,
        "related_relation_names_scored": (
            score_related_elements and score_related_relation_names
        ),
        "ignored_related_relations": sorted(ignored_related_relations),
        "allowed_related_associations": len(allowed_related_associations),
        "attribute_value_label_fallbacks": attribute_fallback_count,
        "non_exact_attribute_value_labels": non_exact_attribute_label_count,
        "path_serialization_normalized": path_scoring,
        "partial_order_path_groups": len(
            path_policy.get("unordered_groups", [])
        ),
        "required_path_precedence": sum(
            len(group.get("required_precedence", []))
            for group in path_policy.get("unordered_groups", [])
        ),
        "ignored_predicted_path_elements": len(
            path_policy.get("ignored_predicted_elements", [])
        ),
        "optional_path_elements": len(
            path_policy.get("optional_path_elements", [])
        ),
        "value_serialization_normalized": value_scoring,
        "communication_flows_scored_once": score_communication_flows_once,
        "message_flow_endpoints_as_roots": message_flow_endpoints_as_roots,
    }


def best_ground_truth(
    candidates: Sequence[dict[str, Any]],
    path_policies: Sequence[PathPolicy],
    predicted: dict[str, Any],
    mode: str,
    tolerance: float,
    schema_valid: bool,
    score_root_entities: bool,
    score_attributes: bool,
    score_related_elements: bool,
    score_related_relation_names: bool,
    ignored_related_relations: frozenset[str],
    allowed_related_associations: Sequence[dict[str, Any]],
    attribute_values_as_related_labels: bool,
    score_communication_flows_once: bool,
    message_flow_endpoints_as_roots: bool,
) -> tuple[int, dict[str, Any]]:
    if len(path_policies) != len(candidates):
        raise ValueError("Each gold candidate requires one path policy.")
    scores = [
        score_answer(
            candidate,
            predicted,
            mode,
            tolerance,
            schema_valid,
            score_root_entities,
            score_attributes,
            score_related_elements,
            score_related_relation_names,
            ignored_related_relations,
            allowed_related_associations,
            attribute_values_as_related_labels,
            score_communication_flows_once,
            message_flow_endpoints_as_roots,
            path_policies[index],
        )
        for index, candidate in enumerate(candidates)
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


AnswerFact = RootFact | RelationFact | AttributeFact | CommunicationFact


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


def normalize_allowed_related_associations(
    value: Any,
    schema: dict[str, Any],
    *,
    location: str,
) -> list[dict[str, Any]]:
    """Validate optional root-to-target associations using the answer schema."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{location}: must be an array.")

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, association in enumerate(value, start=1):
        item_location = f"{location}/{index}"
        if not isinstance(association, dict):
            raise ValueError(f"{item_location}: must be an object.")
        if set(association) - {"root", "relation", "target", "order"}:
            raise ValueError(
                f"{item_location}: only root, relation, target and order are allowed."
            )
        root = association.get("root")
        target = association.get("target")
        relation = association.get("relation")
        order = association.get("order")
        if not isinstance(root, dict) or not isinstance(target, dict):
            raise ValueError(f"{item_location}: root and target must be objects.")
        if set(root) != {"identifier", "label", "element_type"}:
            raise ValueError(
                f"{item_location}/root: identifier, label and element_type are required."
            )
        if set(target) != {"identifier", "label", "element_type"}:
            raise ValueError(
                f"{item_location}/target: identifier, label and element_type are required."
            )

        synthetic_answer = {
            "answer_type": "list",
            "status": "ok",
            "results": [
                {
                    **root,
                    "rank": None,
                    "attributes": [],
                    "related_elements": [
                        {
                            "relation": relation,
                            **target,
                            "order": order,
                        }
                    ],
                }
            ],
        }
        checked = normalize_structured_gold(
            synthetic_answer, schema, location=item_location
        )
        checked_root = checked["results"][0]
        checked_target = checked_root["related_elements"][0]
        normalized_association = {
            "root": {
                "identifier": checked_root["identifier"],
                "label": checked_root["label"],
                "element_type": checked_root["element_type"],
            },
            "relation": checked_target["relation"],
            "target": {
                "identifier": checked_target["identifier"],
                "label": checked_target["label"],
                "element_type": checked_target["element_type"],
            },
            "order": checked_target["order"],
        }
        key = json.dumps(normalized_association, sort_keys=True)
        if key in seen:
            raise ValueError(f"{item_location}: duplicate allowed association.")
        seen.add(key)
        normalized.append(normalized_association)
    return normalized


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

        score_root_entities = raw.get("score_root_entities", True)
        if not isinstance(score_root_entities, bool):
            raise ValueError(
                f"{query_id}: score_root_entities must be true or false."
            )
        score_attributes = raw.get("score_attributes", True)
        if not isinstance(score_attributes, bool):
            raise ValueError(
                f"{query_id}: score_attributes must be true or false."
            )
        score_communication_flows_once = raw.get(
            "score_communication_flows_once", False
        )
        if not isinstance(score_communication_flows_once, bool):
            raise ValueError(
                f"{query_id}: score_communication_flows_once must be true or false."
            )
        message_flow_endpoints_as_roots = raw.get(
            "message_flow_endpoints_as_roots", False
        )
        if not isinstance(message_flow_endpoints_as_roots, bool):
            raise ValueError(
                f"{query_id}: message_flow_endpoints_as_roots must be true or false."
            )
        attribute_values_as_related_labels = raw.get(
            "attribute_values_as_related_labels", False
        )
        if not isinstance(attribute_values_as_related_labels, bool):
            raise ValueError(
                f"{query_id}: attribute_values_as_related_labels must be true or false."
            )
        score_related_elements = raw.get("score_related_elements", False)
        if not isinstance(score_related_elements, bool):
            raise ValueError(
                f"{query_id}: score_related_elements must be true or false."
            )
        score_related_relation_names = raw.get(
            "score_related_relation_names", True
        )
        if not isinstance(score_related_relation_names, bool):
            raise ValueError(
                f"{query_id}: score_related_relation_names must be true or false."
            )
        if not score_related_relation_names and not score_related_elements:
            raise ValueError(
                f"{query_id}: score_related_relation_names=false requires "
                "score_related_elements=true."
            )
        ignored_related_relations_raw = raw.get("ignored_related_relations", [])
        if not isinstance(ignored_related_relations_raw, list) or not all(
            isinstance(item, str) and item for item in ignored_related_relations_raw
        ):
            raise ValueError(
                f"{query_id}: ignored_related_relations must be an array of "
                "non-empty relation names."
            )
        ignored_related_relations = sorted(set(ignored_related_relations_raw))
        if ignored_related_relations and not score_related_elements:
            raise ValueError(
                f"{query_id}: ignored_related_relations requires "
                "score_related_elements=true."
            )
        allowed_related_associations = normalize_allowed_related_associations(
            raw.get("allowed_related_associations", []),
            schema,
            location=f"{query_id}/allowed_related_associations",
        )
        if allowed_related_associations and not score_related_elements:
            raise ValueError(
                f"{query_id}: allowed_related_associations requires "
                "score_related_elements=true."
            )
        if attribute_values_as_related_labels and (
            not score_related_elements
            or score_related_relation_names
            or score_attributes
        ):
            raise ValueError(
                f"{query_id}: attribute_values_as_related_labels=true requires "
                "score_related_elements=true, score_related_relation_names=false "
                "and score_attributes=false."
            )

        primary_raw = raw.get("ground_truth", raw.get("gold_answer"))
        mode = infer_response_mode(primary_raw)
        declared_mode = raw.get("response_mode")
        if declared_mode is not None and declared_mode != mode:
            raise ValueError(
                f"{query_id}: declared response_mode {declared_mode!r} conflicts "
                f"with the {mode} gold answer."
            )
        if score_related_elements and mode != "structured":
            raise ValueError(
                f"{query_id}: score_related_elements is only valid for "
                "structured answers."
            )
        if score_communication_flows_once and (
            mode != "structured"
            or not score_attributes
            or score_related_elements
            or allowed_related_associations
            or attribute_values_as_related_labels
        ):
            raise ValueError(
                f"{query_id}: score_communication_flows_once=true requires a "
                "structured answer with score_attributes=true and unscored "
                "related elements."
            )
        if message_flow_endpoints_as_roots and (
            mode != "structured"
            or not score_root_entities
            or score_attributes
            or score_related_elements
            or score_communication_flows_once
        ):
            raise ValueError(
                f"{query_id}: message_flow_endpoints_as_roots=true requires a "
                "structured answer that scores only root entities."
            )
        if mode != "structured" and (
            not score_root_entities or not score_attributes
        ):
            raise ValueError(
                f"{query_id}: score_root_entities and score_attributes are only "
                "configurable for structured answers."
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
            if raw.get("path_policy") not in (None, {}) or raw.get(
                "acceptable_path_policies"
            ) not in (None, []):
                raise ValueError(
                    f"{query_id}: path policies are only valid for structured "
                    "path answers."
                )
            path_policy: PathPolicy = {}
            acceptable_path_policies: list[PathPolicy] = [
                {} for _ in alternatives
            ]
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
            path_policy = normalize_path_policy(
                raw.get("path_policy"),
                primary,
                location=f"{query_id}/path_policy",
            )
            alternative_policies_raw = raw.get("acceptable_path_policies")
            if alternative_policies_raw is None:
                alternative_policies_raw = [{} for _ in alternatives]
            if not isinstance(alternative_policies_raw, list) or len(
                alternative_policies_raw
            ) != len(alternatives):
                raise ValueError(
                    f"{query_id}: acceptable_path_policies must contain one "
                    "entry for each acceptable_ground_truth."
                )
            acceptable_path_policies = [
                normalize_path_policy(
                    item,
                    alternatives[index - 1],
                    location=f"{query_id}/acceptable_path_policies/{index}",
                )
                for index, item in enumerate(
                    alternative_policies_raw, start=1
                )
            ]
            expected = None

        record: dict[str, Any] = {
            "query_id": query_id,
            "question": question,
            "response_mode": mode,
            "score_root_entities": score_root_entities,
            "score_attributes": score_attributes,
            "score_communication_flows_once": score_communication_flows_once,
            "message_flow_endpoints_as_roots": message_flow_endpoints_as_roots,
            "attribute_values_as_related_labels": (
                attribute_values_as_related_labels
            ),
            "score_related_elements": score_related_elements,
            "score_related_relation_names": score_related_relation_names,
            "ignored_related_relations": ignored_related_relations,
            "allowed_related_associations": allowed_related_associations,
            "path_policy": path_policy,
            "ground_truth": primary,
        }
        if expected is not None:
            record["expected_answer"] = expected
        record["acceptable_ground_truths"] = alternatives
        record["acceptable_path_policies"] = acceptable_path_policies
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
    path_policies: Sequence[PathPolicy],
    prediction: JSONValue,
    schema: dict[str, Any],
    match: str,
    tolerance: float,
    score_root_entities: bool,
    score_attributes: bool,
    score_related_elements: bool,
    score_related_relation_names: bool,
    ignored_related_relations: frozenset[str],
    allowed_related_associations: Sequence[dict[str, Any]],
    attribute_values_as_related_labels: bool,
    score_communication_flows_once: bool,
    message_flow_endpoints_as_roots: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    predicted, parse_errors = coerce_answer(prediction)
    format_errors = parse_errors + validate_instance(predicted, schema, schema)
    valid = not format_errors
    selected, score = best_ground_truth(
        candidates,
        path_policies,
        predicted,
        match,
        tolerance,
        valid,
        score_root_entities,
        score_attributes,
        score_related_elements,
        score_related_relation_names,
        ignored_related_relations,
        allowed_related_associations,
        attribute_values_as_related_labels,
        score_communication_flows_once,
        message_flow_endpoints_as_roots,
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
        "root_entities_scored": score["root_entities_scored"],
        "attributes_scored": score["attributes_scored"],
        "related_elements_scored": score["related_elements_scored"],
        "related_relation_names_scored": score[
            "related_relation_names_scored"
        ],
        "ignored_related_relations": score["ignored_related_relations"],
        "allowed_related_associations": score["allowed_related_associations"],
        "attribute_value_label_fallbacks": score[
            "attribute_value_label_fallbacks"
        ],
        "non_exact_attribute_value_labels": score[
            "non_exact_attribute_value_labels"
        ],
        "path_serialization_normalized": score[
            "path_serialization_normalized"
        ],
        "partial_order_path_groups": score["partial_order_path_groups"],
        "required_path_precedence": score["required_path_precedence"],
        "ignored_predicted_path_elements": score[
            "ignored_predicted_path_elements"
        ],
        "optional_path_elements": score["optional_path_elements"],
        "value_serialization_normalized": score[
            "value_serialization_normalized"
        ],
        "communication_flows_scored_once": score[
            "communication_flows_scored_once"
        ],
        "message_flow_endpoints_as_roots": score[
            "message_flow_endpoints_as_roots"
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
            path_policies = [
                gold["path_policy"], *gold["acceptable_path_policies"]
            ]
            score, diagnostics = score_structured(
                candidates,
                path_policies,
                prediction.get("answer"),
                schema,
                match,
                tolerance,
                gold["score_root_entities"],
                gold["score_attributes"],
                gold["score_related_elements"],
                gold["score_related_relation_names"],
                frozenset(gold["ignored_related_relations"]),
                gold["allowed_related_associations"],
                gold["attribute_values_as_related_labels"],
                gold["score_communication_flows_once"],
                gold["message_flow_endpoints_as_roots"],
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
                "requested root entities and attributes, plus related-element "
                "relationships for explicitly marked questions; a gold record "
                "may disable root-entity or attribute facts; paths and values use "
                "their normalized scoring"
            ),
            "related_target_match_policy": (
                "For scored process/subprocess relationships, matching labels may "
                "bridge a subprocess object and its process-diagram identifier; "
                "identifier-only mode remains strict."
            ),
            "ignored_related_relation_policy": (
                "A gold record may list contextual relation names that are ignored "
                "in both the gold and predicted answers."
            ),
            "related_relation_name_policy": (
                "A gold record may disable relation-name scoring when only the "
                "linked root and target entities are required by the question."
            ),
            "allowed_related_association_policy": (
                "A gold record may list plausible but optional related-element "
                "associations. Matched allowed associations are excluded from "
                "false positives and are not required for recall."
            ),
            "attribute_value_related_label_policy": (
                "An explicitly marked gold record may use a string attribute "
                "value as a fallback for a required related target that was not "
                "already returned in related_elements. The value must equal the "
                "target label or begin with that label before an explanation; "
                "the latter receives ordinary credit but is not an exact label."
            ),
            "default_path_condition_policy": (
                "Condition values beginning with 'default path' and the value "
                "'default process path' are normalized to 'default path'. A "
                "Boolean default_path attribute is ignored when the same result "
                "already supplies a condition; other unexpected attributes remain "
                "scored."
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
