"""Core data types for Hanoi Crossing.

All state is immutable: `GameState` is a frozen dataclass holding tuples, so a
state value can be freely shared across concurrent games, snapshotted for
reinforcement learning, or constructed directly in a test.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class Player(str, Enum):
    """The two players. A holds odd-sized disks, B holds even-sized disks."""

    A = "A"
    B = "B"

    @property
    def opponent(self) -> Player:
        return Player.B if self is Player.A else Player.A


class Pole(str, Enum):
    """The five poles. `SHARED` is the middle pole, visible to both players."""

    P1A = "1a"
    P3A = "3a"
    P1B = "1b"
    P3B = "3b"
    SHARED = "2"


class ActionKind(str, Enum):
    """The three actions a player may take on their turn."""

    LIFT = "lift"
    PLACE = "place"
    SKIP = "skip"


class Outcome(str, Enum):
    """Result of a game. `DRAW` covers the turn limit being reached."""

    A_WINS = "a_wins"
    B_WINS = "b_wins"
    DRAW = "draw"
    IN_PROGRESS = "in_progress"


# Which poles each player can see and interact with. The shared pole appears
# in both sets, which is why a disk left there blocks both players from
# winning (see README, Decision 1).
VISIBLE_POLES: dict[Player, tuple[Pole, ...]] = {
    Player.A: (Pole.P1A, Pole.SHARED, Pole.P3A),
    Player.B: (Pole.P1B, Pole.SHARED, Pole.P3B),
}

# Each player's home pole (where their disks start) and target pole.
HOME_POLE: dict[Player, Pole] = {Player.A: Pole.P1A, Player.B: Pole.P1B}
TARGET_POLE: dict[Player, Pole] = {Player.A: Pole.P3A, Player.B: Pole.P3B}


@dataclass(frozen=True, slots=True)
class Action:
    """A single turn's action.

    `pole` is the pole to lift from or place onto, and is `None` for a skip.
    """

    kind: ActionKind
    pole: Pole | None = None

    @staticmethod
    def lift(pole: Pole) -> Action:
        return Action(ActionKind.LIFT, pole)

    @staticmethod
    def place(pole: Pole) -> Action:
        return Action(ActionKind.PLACE, pole)

    @staticmethod
    def skip() -> Action:
        return Action(ActionKind.SKIP)

    def __str__(self) -> str:
        if self.kind is ActionKind.SKIP:
            return "skip"
        assert self.pole is not None
        return f"{self.kind.value} {self.pole.value}"


@dataclass(frozen=True, slots=True)
class GameState:
    """A complete game position.

    Poles are stored in a single mapping rather than five named fields so that
    `step` can replace one entry generically, without branching on which pole
    was touched.

    Each stack is a tuple ordered bottom-to-top, so the last element is the
    only disk that can be lifted.
    """

    poles: dict[Pole, tuple[int, ...]]
    hands: dict[Player, int | None]
    starting_disks: dict[Player, frozenset[int]]
    turn: int = 0

    def top(self, pole: Pole) -> int | None:
        """Return the topmost disk on `pole`, or None if it is empty."""
        stack = self.poles[pole]
        return stack[-1] if stack else None

    def with_poles(self, **changes: tuple[int, ...]) -> GameState:
        """Return a new state with the named poles replaced."""
        poles = dict(self.poles)
        for name, stack in changes.items():
            poles[Pole[name]] = stack
        return replace(self, poles=poles)


@dataclass(frozen=True, slots=True)
class Observation:
    """What a single player can see.

    Deliberately excludes the opponent's poles and hand. Agents consume this
    rather than `GameState`, so hidden information is structurally
    unreachable rather than merely conventionally respected.
    """

    player: Player
    poles: dict[Pole, tuple[int, ...]]
    hand: int | None
    turn: int

    def top(self, pole: Pole) -> int | None:
        stack = self.poles[pole]
        return stack[-1] if stack else None