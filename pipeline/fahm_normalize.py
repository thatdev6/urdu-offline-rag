"""
fahm_normalize.py — Lightweight Roman Urdu normalization (FAHM Arm 1).

Not a full phonetic algorithm (UrduPhone/Lex-Var are out of scope for this
timeframe) -- this is a small, manually-curated lexical variant map built
from patterns observed in this project's own eval queries. Explicitly a
minimal/honest version, not a production normalizer. See LIMITATIONS.md.
"""

import re

# Common Roman Urdu spelling variants -> one canonical form.
# Built from patterns observed in this project's actual query set --
# NOT a comprehensive dictionary. Expand as more queries are collected.
VARIANT_MAP = {
    # kya / kia / kiya -> kya
    r"\bkia\b": "kya",
    r"\bkiya\b": "kya",
    # ho sakti / ho skti / hosakti -> ho sakti
    r"\bskti\b": "sakti",
    r"\bskta\b": "sakta",
    r"\bhoskti\b": "ho sakti",
    r"\bhoskta\b": "ho sakta",
    # kar / kr -> kar
    r"\bkr\b": "kar",
    r"\bkro\b": "karo",
    # agar / agr -> agar
    r"\bagr\b": "agar",
    # milegi / mile gi / milegy -> milegi
    r"\bmile gi\b": "milegi",
    r"\bmilegy\b": "milegi",
    # kahan / kidr / kaha / kidhar -> kahan
    r"\bkidr\b": "kahan",
    r"\bkidhar\b": "kahan",
    r"\bkaha\b": "kahan",
    # kese / kaise -> kaise
    r"\bkese\b": "kaise",
    # hota / hota he / hota hy -> hota hai
    r"\bhota he\b": "hota hai",
    r"\bhota hy\b": "hota hai",
    r"\bhe\b": "hai",
    r"\bhy\b": "hai",
}


def normalize_roman_urdu(text: str) -> str:
    """Apply variant normalization. Case-insensitive, word-boundary matching."""
    normalized = text
    for pattern, replacement in VARIANT_MAP.items():
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    return normalized


if __name__ == "__main__":
    test_queries = [
        "Kia us din friday ki Chutti mile gi?",
        "Agr paper miss ho jae, repeat kr skte paper ko?",
        "Fees installment me ho skti?",
        "Raging ya bullying ka case kidr ho skta?",
        "Teacher ki complaint kese ho skti? Or kaha ho skti",
    ]
    for q in test_queries:
        print(f"RAW:  {q}")
        print(f"NORM: {normalize_roman_urdu(q)}")
        print()
