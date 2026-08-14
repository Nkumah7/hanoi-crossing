"""Command-line frontends.

Two entry points: `replay` reads a recorded game and reports the final
position, `random-play` runs two random agents. Both are thin — all rules live
in the engine, and everything here is input parsing and formatting.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from hanoi_crossing.agents import Agent, RandomAgent
from hanoi_crossing.engine import check_outcome, initial_state, observe, step
from hanoi_crossing.model import Action, ActionKind, GameState, Outcome, Player, Pole

DEFAULT_MAX_TURNS = 500


def parse_action(raw: str | dict) -> Action:
    """Parse one recorded action.

    Accepts either a compact string (`"lift 1a"`, `"skip"`) or an object
    (`{"kind": "lift", "pole": "1a"}`). The string form keeps hand-written
    game files readable; the object form is easier to generate.
    """
    if isinstance(raw, dict):
        kind = ActionKind(raw["kind"])
        pole = Pole(raw["pole"]) if raw.get("pole") else None
        return Action(kind, pole)

    parts = raw.split()
    kind = ActionKind(parts[0])
    if kind is ActionKind.SKIP:
        return Action.skip()
    return Action(kind, Pole(parts[1]))


def render(state: GameState, outcome: Outcome) -> str:
    """Format the full position. Used by the frontends, never by the engine."""
    lines = [
        f"turn {state.turn}",
        "",
        f"  1a  {list(state.poles[Pole.P1A])}",
        f"  3a  {list(state.poles[Pole.P3A])}",
        f"   2  {list(state.poles[Pole.SHARED])}   (shared)",
        f"  1b  {list(state.poles[Pole.P1B])}",
        f"  3b  {list(state.poles[Pole.P3B])}",
        "",
        f"  hand A  {state.hands[Player.A]}",
        f"  hand B  {state.hands[Player.B]}",
        "",
        f"outcome: {outcome.value}",
    ]
    return "\n".join(lines)


def run_replay(path: Path, require_all_disks: bool = False) -> tuple[GameState, Outcome]:
    """Replay a recorded game and return the final position.

    Input is JSON:

        {
          "n": 2,
          "turn_order": ["A", "B", "A"],
          "moves": ["lift 1a", "lift 1b", "place 3a"]
        }

    `turn_order` and `moves` are parallel: the i-th move is played by the i-th
    player. Turn order is external input, as the spec requires — the engine
    assumes no pattern.

    The replay stops early once a game is decided, so trailing moves in a
    recording do not overwrite a result.
    """
    data = json.loads(path.read_text())
    turn_order = [Player(p) for p in data["turn_order"]]
    moves = [parse_action(m) for m in data["moves"]]

    if len(turn_order) != len(moves):
        raise ValueError(f"turn_order has {len(turn_order)} entries but moves has {len(moves)}")

    state = initial_state(data["n"])
    outcome = check_outcome(state, require_all_disks)

    for player, action in zip(turn_order, moves, strict=True):
        if outcome is not Outcome.IN_PROGRESS:
            break
        state = step(state, player, action)
        outcome = check_outcome(state, require_all_disks)

    return state, outcome


def run_random(
    n: int,
    seed: int | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    require_all_disks: bool = False,
    agent: Agent | None = None,
) -> tuple[GameState, Outcome]:
    """Play a game, defaulting to both sides choosing uniformly at random.

    `agent` accepts any callable satisfying the `Agent` protocol, so an
    external policy can drive the engine without modifying it. Defaults to a
    `RandomAgent` seeded from `seed`.

    Turn order alternates here only because the frontend has to pick
    something; the engine imposes no pattern.

    `max_turns` is required, not optional: the rules permit games that never
    end (a player may hold an opponent's disk and skip indefinitely), so an
    external bound is the only guarantee of termination. This is the same
    truncation any RL environment applies.
    """
    rng = random.Random(seed)
    if agent is None:
        agent = RandomAgent(rng)

    state = initial_state(n)
    outcome = check_outcome(state, require_all_disks)
    players = (Player.A, Player.B)

    while outcome is Outcome.IN_PROGRESS and state.turn < max_turns:
        player = players[state.turn % 2]
        action = agent(observe(state, player))
        state = step(state, player, action)
        outcome = check_outcome(state, require_all_disks)

    return state, outcome


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hanoi-crossing")
    parser.add_argument(
        "--require-all-disks",
        action="store_true",
        help="stricter win condition: every starting disk must be on pole 3",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    replay = sub.add_parser("replay", help="replay a recorded game from JSON")
    replay.add_argument("path", type=Path)

    rand = sub.add_parser("random-play", help="play a game with two random agents")
    rand.add_argument("-n", type=int, default=3, help="disks per player")
    rand.add_argument("--seed", type=int, default=None)
    rand.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)

    args = parser.parse_args(argv)

    if args.command == "replay":
        state, outcome = run_replay(args.path, args.require_all_disks)
    else:
        state, outcome = run_random(args.n, args.seed, args.max_turns, args.require_all_disks)

    print(render(state, outcome))
    return 0


if __name__ == "__main__":
    sys.exit(main())
