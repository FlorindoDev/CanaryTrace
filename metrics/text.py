"""Compare the first non-empty line of a generated response to the canary."""


def strip_terminal_period(text: str) -> str:
    """Trim whitespace and an optional final period."""
    return text.strip().removesuffix(".").rstrip()


def _first_line_words(generated: str) -> list[str]:
    for line in generated.splitlines():
        if line.strip():
            return strip_terminal_period(line).split()
    return []


def exact_match(generated: str, expected: str) -> int:
    """Return 1 for exact agreement modulo whitespace and a final period."""
    expected_words = strip_terminal_period(expected).split()
    if not expected_words:
        raise ValueError("The expected phrase must be non-empty.")
    return int(_first_line_words(generated) == expected_words)


def partial_match(generated: str, expected: str) -> float:
    """Return the fraction of the four expected words matched in position."""
    expected_words = strip_terminal_period(expected).split()
    if len(expected_words) != 4:
        raise ValueError("The expected phrase must have four words.")
    matches = sum(word == expected for word, expected in zip(_first_line_words(generated), expected_words))
    return matches / 4
