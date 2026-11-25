"""Compare a generated response to the canary using line-based metrics."""


def strip_terminal_period(text: str) -> str:
    """Trim whitespace and an optional final period."""
    return text.strip().removesuffix(".").rstrip()


def _first_nonempty_line_words(generated: str) -> list[str]:
    for line in generated.splitlines():
        if line.strip():
            return strip_terminal_period(line).split()
    return []


def exact_match(generated: str, expected: str) -> int:
    """Compare the first line, ignoring whitespace differences and a final period."""
    expected_words = strip_terminal_period(expected).split()
    if not expected_words:
        raise ValueError("The expected phrase must be non-empty.")
    lines = generated.splitlines()
    first_line = lines[0] if lines else ""
    return int(strip_terminal_period(first_line).split() == expected_words)


def partial_match(generated: str, expected: str) -> float:
    """Return the fraction of the four expected words matched in position."""
    expected_words = strip_terminal_period(expected).split()
    if len(expected_words) != 4:
        raise ValueError("The expected phrase must have four words.")
    generated_words = _first_nonempty_line_words(generated)
    matches = sum(word == target for word, target in zip(generated_words, expected_words))
    return matches / 4
