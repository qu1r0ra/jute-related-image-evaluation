"""Versioned human-rating definitions shared by the applet and audit tools."""

from __future__ import annotations

from typing import Any

RATING_RUBRIC_VERSION = "rating-bundle-v2"
PAIR_RUBRIC_VERSION = "related-image-v2"

PAIR_RUBRIC: dict[str, Any] = {
    "version": PAIR_RUBRIC_VERSION,
    "decision_order": "first matching category",
    "identity_first": True,
    "scale_note": (
        "The categories form an identity-first decision sequence, not a "
        "validated interval or ordinal similarity scale."
    ),
    "categories": [
        {
            "value": "exact",
            "label": "Visually identical",
            "merge": True,
            "definition": (
                "The images have indistinguishable visible content, framing and "
                "orientation under inspection. A copy or re-save qualifies if "
                "it introduces no perceptible change. This does not establish "
                "byte identity."
            ),
        },
        {
            "value": "near-duplicate",
            "label": "Near-duplicate",
            "merge": True,
            "definition": (
                "The images visibly differ, but corresponding distinctive "
                "details establish that they depict the same physical leaf or "
                "overlapping content from the same captured scene. Differences "
                "may include another capture, a crop, rotation, reflection or "
                "image processing. A shared generic background alone is "
                "insufficient."
            ),
        },
        {
            "value": "very similar",
            "label": "Visually similar",
            "merge": False,
            "definition": (
                "Neither identity category applies, but the images strongly "
                "resemble one another overall. For leaf-focused images, judge "
                "resemblance primarily from the leaf's shape, venation, "
                "markings and damage; background is secondary."
            ),
        },
        {
            "value": "different",
            "label": "Visually dissimilar",
            "merge": False,
            "definition": (
                "Neither identity category applies, and overall resemblance is "
                "limited. Shared generic features, such as green color or "
                "similar placement, do not by themselves establish strong "
                "resemblance."
            ),
        },
    ],
    "merge_semantics": {
        "merge": ["exact", "near-duplicate"],
        "no_merge": ["very similar", "different"],
    },
    "file_hash_identity": (
        "File-hash exact-copy grouping is separate from visual identity."
    ),
    "review_rule": (
        "Flag identity uncertainty for review when resolving it could change "
        "the merge decision."
    ),
}

MASK_RUBRIC_VERSION = "leaf-mask-v2"
MASK_RUBRIC: dict[str, Any] = {
    "version": MASK_RUBRIC_VERSION,
    "categories": ["pass", "minor", "fail"],
    "labels": {"pass": "Pass", "minor": "Minor defect", "fail": "Fail"},
    "target_leaf": (
        "The visually prominent leaf that appears to be the intended subject. "
        "Position and apparent closeness may help identify it, but neither "
        "determines the target."
    ),
    "background": (
        "Everything outside the target leaf, including other leaves, stems "
        "and surrounding objects."
    ),
    "criteria": [
        "target-leaf tissue coverage",
        "visible-lesion coverage",
        "background inclusion (mask leakage)",
    ],
    "rule": (
        "Consider tissue coverage, lesion coverage and background inclusion "
        "separately, then apply the most severe applicable defect. A small "
        "clipped region containing most visible lesion evidence may warrant Fail."
    ),
    "anchors": {
        "pass": (
            "The target leaf and its visible lesions are covered, with no more "
            "than negligible clipping or background inclusion."
        ),
        "minor": (
            "Localized clipping or background inclusion is visible, but most "
            "target-leaf tissue and visible lesions remain covered."
        ),
        "fail": (
            "Substantial target-leaf or lesion clipping, or substantial "
            "background inclusion, is present."
        ),
    },
    "review_rule": (
        "Flag ambiguous target leaves for review instead of forcing a mask rating."
    ),
}

SCENE_RUBRIC_VERSION = "scene-tag-v2"
SCENE_RUBRIC: dict[str, Any] = {
    "version": SCENE_RUBRIC_VERSION,
    "categories": ["clean-background", "field", "unknown"],
    "anchors": {
        "clean-background": (
            "Most visible background is a uniform, uncluttered surface."
        ),
        "field": (
            "The background contains substantial outdoor or field structure, "
            "including other leaves or plants."
        ),
        "unknown": (
            "The image provides insufficient readable background, for example "
            "because of a tight crop or blur."
        ),
    },
}
SCENE_GUIDANCE: dict[str, str] = SCENE_RUBRIC["anchors"]

CALIBRATION_EXAMPLES: list[dict[str, str]] = [
    {
        "case": "Same visible pixels, framing and orientation after a re-save.",
        "decision": "Visually identical",
    },
    {
        "case": (
            "A crop differs, while matching distinctive venation and damage "
            "identify the same leaf."
        ),
        "decision": "Near-duplicate",
    },
    {
        "case": (
            "The focal leaf changes, but distinctive overlapping scene details "
            "establish the same capture."
        ),
        "decision": "Near-duplicate",
    },
    {
        "case": "A plain background matches, while leaf shape and markings differ.",
        "decision": (
            "The background alone does not establish identity; rate resemblance "
            "from visible content."
        ),
    },
    {
        "case": (
            "Different leaves have strongly matching shape, venation, markings "
            "and damage, with no identity evidence."
        ),
        "decision": "Visually similar",
    },
    {
        "case": "Images share only green color and a similar placement.",
        "decision": "Visually dissimilar",
    },
    {
        "case": (
            "Distinctive details are unreadable and identity could change merge status."
        ),
        "decision": "Flag identity uncertainty for review.",
    },
    {
        "case": (
            "Several leaves are equally prominent and the intended mask target "
            "is unclear."
        ),
        "decision": "Flag the ambiguous target for review.",
    },
    {
        "case": "A small clipped area contains most of the visible lesion evidence.",
        "decision": (
            "Fail may apply because lesion loss is considered separately from "
            "total area."
        ),
    },
]

_PAIR_INSTRUCTIONS = [
    f"- **{item['label']}** (`{item['value']}`): {item['definition']}"
    for item in PAIR_RUBRIC["categories"]
]
_MASK_INSTRUCTIONS = [
    f"- **{MASK_RUBRIC['labels'][value]}** (`{value}`): {definition}"
    for value, definition in MASK_RUBRIC["anchors"].items()
]
_SCENE_INSTRUCTIONS = [
    f"- **{value}**: {definition}" for value, definition in SCENE_GUIDANCE.items()
]
_CALIBRATION_INSTRUCTIONS = [
    f"- {example['case']} → {example['decision']}" for example in CALIBRATION_EXAMPLES
]

AGENT_INSTRUCTIONS = "\n".join(
    (
        f"# Independent rating instructions — {RATING_RUBRIC_VERSION}",
        "",
        (
            "Use only the definitions that apply to the assigned item. For "
            "pairs, apply the first matching category in order. The values "
            "are storage codes; record the code, not a new label."
        ),
        "",
        f"## Pair identity and resemblance ({PAIR_RUBRIC_VERSION})",
        "",
        *_PAIR_INSTRUCTIONS,
        "",
        PAIR_RUBRIC["scale_note"],
        PAIR_RUBRIC["review_rule"],
        PAIR_RUBRIC["file_hash_identity"],
        "",
        f"## Mask quality ({MASK_RUBRIC_VERSION})",
        "",
        f"Target leaf: {MASK_RUBRIC['target_leaf']}",
        f"Background: {MASK_RUBRIC['background']}",
        MASK_RUBRIC["rule"],
        "",
        *_MASK_INSTRUCTIONS,
        "",
        MASK_RUBRIC["review_rule"],
        "",
        "## Scene tags (unchanged: scene-tag-v2)",
        "",
        *_SCENE_INSTRUCTIONS,
        "",
        (
            "Do not use or receive earlier owner or agent answers, grouping "
            "decisions, predicted labels, algorithm outcomes, rating "
            "distributions, or gate outcomes. Keep item order and the supplied "
            "cohort fixed. Record the applicable rubric and instrument "
            "versions with the pass. If identity uncertainty could change "
            "merge status or the intended mask target is ambiguous, flag the "
            "case for review and leave it unresolved; do not silently code "
            "uncertainty as no-merge or mask failure."
        ),
        "",
        "## Synthetic calibration examples",
        "",
        *_CALIBRATION_INSTRUCTIONS,
    )
)
