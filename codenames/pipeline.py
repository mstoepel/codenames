"""End-to-end: words + key card -> spymaster view."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .key_card import KeyCardResult, Team, extract_colors
from .word_cards import WordCardsResult, extract_words


@dataclass
class SpymasterView:
    grid: list[list[tuple[str | None, Team]]]
    words_result: WordCardsResult = field(repr=False)
    key_result: KeyCardResult = field(repr=False)

    def _by_team(self, team: Team) -> list[str]:
        return [
            w for row in self.grid for (w, t) in row if t == team and w is not None
        ]

    @property
    def red(self) -> list[str]:
        return self._by_team("red")

    @property
    def blue(self) -> list[str]:
        return self._by_team("blue")

    @property
    def neutral(self) -> list[str]:
        return self._by_team("neutral")

    @property
    def assassin(self) -> list[str]:
        return self._by_team("assassin")


def analyze(words_img: np.ndarray, key_img: np.ndarray) -> SpymasterView:
    wr = extract_words(words_img)
    kr = extract_colors(key_img)
    grid: list[list[tuple[str | None, Team]]] = [
        [(wr.grid[r][c], kr.grid[r][c]) for c in range(5)] for r in range(5)
    ]
    return SpymasterView(grid=grid, words_result=wr, key_result=kr)
