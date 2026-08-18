"""
fahm_translate.py — Translate Roman Urdu query -> English (FAHM Arm 2, adapted).

Original plan called for transliteration to Urdu script + monolingual
retrieval, but this project's corpus was confirmed to be entirely
English-language (no Urdu-script content found via grep). Transliterating to
Urdu script would move the query further from the corpus's actual language,
not closer. Adapted Arm 2 instead: translate to English, the corpus's real
language, and test whether closing that gap improves retrieval.

Uses llama-server directly (short prompt, short output -- fast, not a full
RAG generation call).
"""

import requests

LLAMA_SERVER_URL = "http://localhost:8080/v1/chat/completions"

TRANSLATE_SYSTEM_PROMPT = (
    "Translate the following Roman Urdu (or mixed Roman Urdu/English) question "
    "into clear, formal English. Output ONLY the translated question, nothing else "
    "-- no explanation, no quotes."
)


def translate_to_english(query: str, timeout: int = 120) -> str:
    payload = {
        "messages": [
            {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        "temperature": 0.1,
        "max_tokens": 60,
    }
    resp = requests.post(LLAMA_SERVER_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


if __name__ == "__main__":
    test_queries = [
        "Kia us din friday ki Chutti mile gi?",
        "Agr paper miss ho jae, repeat kr skte paper ko?",
        "Fees installment me ho skti?",
    ]
    for q in test_queries:
        print(f"RAW: {q}")
        print(f"EN:  {translate_to_english(q)}")
        print()
