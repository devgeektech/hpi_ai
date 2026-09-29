from typing import Any, Dict


def explain(structured: Dict[str, Any]) -> Dict[str, str]:
    """Optional NLP stub — never invents traits; only narrates approved structure."""
    strength = structured.get("standout_strength") or "your profile strengths"
    confidence = structured.get("confidence", 0)
    return {
        "summary": (
            f"Based on the approved analysis, the standout theme is {strength} "
            f"(confidence {confidence:.0f}%)."
        ),
        "explanation": (
            "This text is generated from structured HPI findings only. "
            "It does not reinterpret handwriting features."
        ),
        "nlp_version": "nlp-stub-v1",
    }
