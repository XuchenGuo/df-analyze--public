from __future__ import annotations

import shlex


COLUMN_LIST_OPTIONS = frozenset(
    {"--targets", "--categoricals", "--ordinals", "--drops"}
)


def split_cli_args(args: str) -> list[str]:
    """Split CLI text without damaging Windows paths or column names."""
    lexer = shlex.shlex(args, posix=True)
    lexer.commenters = ""
    lexer.whitespace_split = True
    lexer.escape = ""
    lexer.quotes = '"\''
    tokens = list(lexer)

    normalized: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        normalized.append(token)
        index += 1
        if token not in COLUMN_LIST_OPTIONS:
            continue

        columns: list[str] = []
        while index < len(tokens) and not tokens[index].startswith("--"):
            columns.extend(column for column in tokens[index].split(",") if column)
            index += 1
        if columns:
            normalized.append(",".join(columns))

    return normalized
