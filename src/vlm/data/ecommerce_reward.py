"""Normalization and label utilities for Stage 4 e-commerce rewards."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any


GENERATION_TASKS = {
    "product_title_generation",
    "product_attribute_summary",
}

SHORT_ANSWER_TASKS = {
    "product_brand_qa",
    "product_type_qa",
    "product_color_qa",
    "product_style_qa",
}


COLOR_CANONICAL = {
    "black": "black",
    "negro": "black",
    "nero": "black",
    "noir": "black",
    "schwarz": "black",
    "svart": "black",
    "white": "white",
    "wit": "white",
    "blanc": "white",
    "bianco": "white",
    "weiss": "white",
    "weiß": "white",
    "blue": "blue",
    "blu": "blue",
    "bleu": "blue",
    "azul": "blue",
    "navy": "blue",
    "red": "red",
    "rojo": "red",
    "rouge": "red",
    "rot": "red",
    "orange": "orange",
    "anaranjado": "orange",
    "yellow": "yellow",
    "gelb": "yellow",
    "green": "green",
    "gruen": "green",
    "grün": "green",
    "verde": "green",
    "grey": "gray",
    "gray": "gray",
    "gris": "gray",
    "grau": "gray",
    "brown": "brown",
    "braun": "brown",
    "brun": "brown",
    "beige": "beige",
    "tan": "beige",
    "silver": "silver",
    "silber": "silver",
    "gold": "gold",
    "golden": "gold",
    "pink": "pink",
    "rose": "pink",
    "purple": "purple",
    "violet": "purple",
    "violett": "purple",
    "platinum": "platinum",
    "platino": "platinum",
    "clear": "clear",
    "transparent": "clear",
    "multicolor": "multicolor",
    "multi color": "multicolor",
    "multi colored": "multicolor",
    "multicolored": "multicolor",
    "multi coloured": "multicolor",
    "others": "other",
    "other": "other",
}


BRAND_CANONICAL = {
    "amazon basics": "amazonbasics",
    "amazonbasics": "amazonbasics",
    "amazon brand solimo": "amazon brand solimo",
    "solimo": "amazon brand solimo",
    "amazon brand symbol": "amazon brand symbol",
    "symbol": "amazon brand symbol",
    "stone beam": "stone and beam",
    "stone and beam": "stone and beam",
    "pinzon by amazoncom": "pinzon by amazon",
    "pinzon by amazon": "pinzon by amazon",
    "365 by whole foods market": "365 by whole foods market",
    "365 everyday value": "365 everyday value",
}


STYLE_CANONICAL = {
    "morden": "modern",
    "modern": "modern",
    "low top": "low top",
    "lowtop": "low top",
    "ankle strap": "ankle strap",
    "anklestrap": "ankle strap",
    "sneaker": "sneakers",
    "sneakers": "sneakers",
    "running shoe": "running shoes",
    "running shoes": "running shoes",
}


def normalize_match_text(text: Any) -> str:
    """Lowercase text and keep separator boundaries for token overlap."""

    if text is None:
        return ""
    value = str(text).lower().strip()
    value = value.replace("_", " ")
    value = re.sub(r"[-/|]+", " ", value)
    value = value.replace("&", " and ")
    value = value.translate(str.maketrans("", "", string.punctuation.replace("_", "")))
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", normalize_match_text(text))


def canonicalize(text: Any, task: str = "") -> str:
    normalized = normalize_match_text(text)
    if not normalized:
        return ""

    if task == "product_color_qa":
        if normalized in COLOR_CANONICAL:
            return COLOR_CANONICAL[normalized]
        for token in normalized.split():
            if token in COLOR_CANONICAL:
                return COLOR_CANONICAL[token]
        return normalized

    if task == "product_brand_qa":
        compact = compact_text(normalized)
        if compact == "amazonbasics":
            return "amazonbasics"
        return BRAND_CANONICAL.get(normalized, normalized)

    if task == "product_style_qa":
        return STYLE_CANONICAL.get(normalized, normalized)

    return normalized


def normalized_exact_match(prediction: str, references: list[str], task: str = "") -> float:
    pred = canonicalize(prediction, task)
    return float(any(pred and pred == canonicalize(ref, task) for ref in references))


def token_f1(prediction: str, references: list[str], task: str = "") -> float:
    pred_tokens = canonicalize(prediction, task).split()
    if not pred_tokens:
        return 0.0
    best = 0.0
    for ref in references:
        ref_tokens = canonicalize(ref, task).split()
        if not ref_tokens:
            continue
        common = Counter(pred_tokens) & Counter(ref_tokens)
        same = sum(common.values())
        if same == 0:
            continue
        precision = same / len(pred_tokens)
        recall = same / len(ref_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def answer_aliases(answer: str, task: str) -> list[str]:
    aliases: list[str] = []

    def add(value: str) -> None:
        value = re.sub(r"\s+", " ", str(value)).strip()
        if value and value not in aliases:
            aliases.append(value)

    add(answer)
    normalized = normalize_match_text(answer)
    canonical = canonicalize(answer, task)
    add(normalized)
    add(canonical)

    if task == "product_type_qa":
        add(answer.replace("_", " "))
        add(answer.replace("_", "-"))
        add(answer.title().replace("_", " "))
    elif task == "product_brand_qa":
        if canonical == "amazonbasics":
            add("Amazon Basics")
            add("AmazonBasics")
        if canonical == "amazon brand solimo":
            add("Amazon Brand - Solimo")
            add("Solimo")
    elif task == "product_color_qa":
        if canonical == "multicolor":
            add("Multicolor")
            add("multi-colored")
            add("multicolored")
        if canonical == "gray":
            add("Grey")
            add("Gray")
        if canonical == "black":
            add("Black")
            add("Negro")
            add("Nero")
            add("Schwarz")
        if canonical == "white":
            add("White")
            add("Wit")
        if canonical == "other":
            add("Others")

    return aliases


def ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    ascii_chars = sum(1 for char in text if ord(char) < 128)
    return ascii_chars / max(1, len(text))


def looks_like_noisy_short_label(answer: str) -> bool:
    value = normalize_match_text(answer)
    if not value:
        return True
    if len(value) > 64:
        return True
    if re.search(r"\d", value):
        return True
    return False


def reliable_for_grpo(sample: dict[str, Any], *, top_types: set[str], top_brands: set[str]) -> bool:
    task = str(sample.get("task", ""))
    answers = sample.get("answers") or []
    answer = str(answers[0] if answers else sample.get("answer", "")).strip()
    if not answer:
        return False

    if task == "product_type_qa":
        return answer in top_types and len(answer) <= 48

    if task == "product_brand_qa":
        return answer in top_brands and ascii_ratio(answer) >= 0.85 and len(answer) <= 48

    if task == "product_color_qa":
        if len(answer) > 48:
            return False
        if re.search(r"\d", answer):
            return False
        return ascii_ratio(answer) >= 0.6

    if task == "product_style_qa":
        if looks_like_noisy_short_label(answer):
            return False
        return ascii_ratio(answer) >= 0.85 and len(normalize_match_text(answer).split()) <= 4

    if task == "product_title_generation":
        words = normalize_match_text(answer).split()
        return ascii_ratio(answer) >= 0.9 and 3 <= len(words) <= 24

    if task == "product_attribute_summary":
        words = normalize_match_text(answer).split()
        return ascii_ratio(answer) >= 0.85 and 6 <= len(words) <= 96

    return False
