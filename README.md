# Hanoi Crossing

A game engine for Hanoi Crossing, with a replay CLI and a random-play mode.

The engine is a set of pure functions over immutable state, so it can be
embedded unchanged in a reinforcement-learning loop or a service running many
concurrent games. The random agent consumes it through exactly the interface
such an external agent would use.

---

## Running it

```bash
uv sync
uv run pytest
```

**Replay a recorded game:**

```bash
uv run hanoi-crossing replay examples/spec_example.json
```

**Play a random game:**

```bash
uv run hanoi-crossing random-play -n 3 --seed 42
uv run hanoi-crossing random-play -n 3 --seed 42 --max-turns 200
```

**Stricter win condition** (see Decision 4):

```bash
uv run hanoi-crossing --require-all-disks random-play -n 3 --seed 42
```

---

## Input format

Replay input is JSON:

```json
{
  "n": 1,
  "turn_order": ["A", "B", "A"],
  "moves": ["lift 1a", "lift 1b", "place 3a"]
}
```

`turn_order` and `moves` are parallel arrays — the i-th move is played by the
i-th player. Turn order is external input, as the spec requires; the engine
assumes no pattern and the two frontends supply it differently.

Moves accept either a compact string (`"lift 1a"`, `"skip"`) or an object
(`{"kind": "lift", "pole": "1a"}`). The string form keeps hand-written game
files readable; the object form is easier to generate.

Poles are named as in the specification: `1a`, `3a`, `1b`, `3b`, and `2` for
the shared pole.

---

## Architecture

```
src/hanoi_crossing/
    model.py     types: Player, Pole, Action, GameState, Observation, Outcome
    engine.py    rules: initial_state, observe, legal_actions, step, check_outcome
    agents.py    RandomAgent, SkipAgent
    cli.py       replay and random-play frontends
```

The brief asks that the engine later serve, unchanged, as the core of an RL
training loop or a concurrent simulation service. Three choices follow from
that.

### Immutable state, pure step function

`step(state, player, action) -> GameState` returns a new state rather than
mutating.

- **Concurrent games:** no shared mutable state, so no locking and no
  cross-contamination between simulations. A service holding many games holds
  many independent values.
- **RL:** snapshot and restore are free, because a state *is* a snapshot. Tree
  search and replay buffers need no engine support.
- **Testing:** each test constructs the exact position it needs, with no
  setup, teardown, or ordering dependencies.

The cost is an allocation per action instead of an in-place update. At this
scale it doesn't matter, and if profiling ever said otherwise the fix would be
a mutable inner layer behind the same pure interface — the public API would
not change.

Implemented with frozen dataclasses and tuples. One honest gap: `frozen=True`
prevents rebinding `state.poles`, not mutating the dict it points at. Engine
code never mutates, and `with_poles` provides the correct path, but a
`MappingProxyType` would enforce rather than rely on discipline.

### A separate observation type

The rules give each player sight of only three poles and their own hand.
`observe(state, player)` returns an `Observation` containing exactly that.

It is a distinct type, not a `GameState` with fields blanked, so the
opponent's poles are structurally absent — there is no attribute to read.

The load-bearing property: **a player's legal actions depend only on their
visible poles and their own hand**, which is precisely an observation's
contents. So `legal_actions(observation)` is well-defined, and the random
agent computes its own options without the engine handing it privileged
information. That is the interface an external policy expects.

### Legality has one definition

`is_legal` is defined as membership in `legal_actions`. There is no second
validation path inside `step` that could accept an action the action list
never offered, or reject one it did — a class of bug removed by construction
rather than by care.

### The engine does no I/O

No printing, no file access, no CLI concerns in `model.py` or `engine.py`. An
engine that prints cannot be embedded in a training loop without producing
noise, nor tested without capturing stdout. The frontends own presentation.

---

## Time spent

Roughly two hours on implementation. Additional time went into analysing the
specification for ambiguities and writing up the decisions — the seven
interpretive questions in the log below took longer to settle than the code
took to write, and a final review pass followed.

---

## Decision log

The spec leaves several points open. Each is resolved here with the reading
taken and the alternative rejected.

Fuller reasoning — including the alternatives considered and where the initial
analysis went wrong — is in [`docs/design-notes.md`](docs/design-notes.md).

### 1. The shared pole blocks both players

The win condition is *"their hand is empty and, among their visible poles,
only pole 3 has disks on it."* Player A's visible poles are `1a`, `2`, `3a`.
So `2` must be empty for A to win — and since `2` is visible to both players,
a disk left there blocks **both**.

*Alternative rejected:* reading "their visible poles" as private poles only.
The spec says visible, and defines the middle pole as visible to both.

This makes the shared pole a genuine risk rather than a free dumping ground,
and it is the source of most of the game's structure.

### 2. The outcome is checked for both players after every action

A win can be completed by the **opponent's** move. If A has finished but a
disk sits on the shared pole, A is blocked; when B lifts that disk for their
own reasons, A's condition becomes true without A acting.

`check_outcome` therefore takes only the state, not the player who moved.
Whose turn it was is irrelevant to whether anyone has won.

### 3. A simultaneous win is a draw

This was got wrong twice before the tests settled it, which is worth
recording.

The first assumption was that simultaneous wins needed a tie-break. The second
was that they were unreachable — reasoning that the action clearing the shared
pole is a lift, which leaves the actor holding a disk, so their own hand isn't
empty.

A test written for something else disproved it:

```
1a: ()      1b: ()
3a: (1,)    3b: (4, 3, 2)      shared empty, both hands free
```

Both conditions hold, and the position arises naturally: B finishes and waits
— nothing obliges them to keep acting — and A later places their final disk.
The second argument only considered how the shared pole gets cleared; it
missed that two wins need not be caused by the same action, only be true at
the same time.

A draw is the fair result. The opponent finished first and waited, and nothing
in the rules penalises waiting.

### 4. A player may win with an incomplete tower (configurable)

The condition says "only pole 3 has disks on it." It does **not** say *all* of
that player's disks must be there.

This matters because disks can be lost permanently: A places disk 5 on the
shared pole, B lifts it and puts it on `3b`, and A can never reach it again.
Under the literal reading A can still win with what remains.

*Decision:* take the literal reading by default. Adding an unstated
requirement is a larger interpretive step than following the text.

*Known consequence:* this admits a degenerate strategy — move one disk to pole
3, dump the rest on the shared pole, and win once the opponent clears it.

*Resolution:* `--require-all-disks` supports the stricter reading, so the
ambiguity is a documented option rather than a coin flip.

### 5. A turn limit, and no rule fix for the hostage strategy

**The rules permit games that never end.** If B lifts one of A's disks and
then skips indefinitely, that disk can never reach `3a` — and B cannot win
either, since B's hand is never empty. B trades their own victory for a
denial. Nothing in the spec prevents this.

**A qualification the tests produced.** The strategy is weaker than it first
appears under the default win reading. Because a player may win with an
incomplete tower (Decision 4), taking one disk hostage only blocks the victim
if it is their *only* route to a non-empty pole 3. Worse for the hostage-taker:
lifting a disk *off* the shared pole clears the very thing that was blocking
the opponent, so the move can hand them the win outright.

The strategy is decisive only when the victim has nothing else on their target
pole, or under `--require-all-disks`, where every disk must come home. Both
cases have tests
(`test_hostage_strategy_prevents_both_players_winning`,
`test_hostage_is_decisive_when_all_disks_required`).

This was found by writing the test, not by reading the rules — the first
version asserted a deadlock and instead reported `A_WINS`.

**Decision:** the engine takes a turn limit. Not a rule change — a practical
bound, the same thing an RL environment calls episode truncation. Replay ends
when the turn sequence is exhausted; random-play ends at `--max-turns`.

**Deliberately not done:** no skip cap, no forced-placement rule. Non-
termination is a genuine property of the rules as written, not an ambiguity.
Patching it would be a design change beyond the spec.

### 6. Replay stops once the game is decided

Trailing moves in a recording do not overwrite a result already reached. The
alternative — playing the whole sequence and reporting the final position —
would let a won game continue.

### 7. Reading of the 500-line constraint

I read "core engine" as the rules layer: `model.py` and `engine.py`, **325
lines** (243 excluding blanks and comments).

`agents.py` and `cli.py` are consumers built on that engine, and including
them brings the package to 531 lines. Stating both numbers since the term is
open to either reading.

Their separability is itself evidence the engine is properly scoped —
inlining the CLI to satisfy a line budget would have violated the reuse
requirement.

---

## How this serves the stated futures

The brief asks that the engine support an RL loop and a concurrent simulation
service without modification.

| RL concept | Here |
|---|---|
| `reset()` | `initial_state(n)` |
| `step(action)` | `step(state, player, action)` |
| observation | `observe(state, player)` |
| action mask | `legal_actions(observation)` |
| termination | `check_outcome(state)` |
| truncation | `state.turn` against a limit |

Reward is deliberately absent — it is a training concern rather than a rule,
and it is derivable from `check_outcome`.

For concurrent games, states are values: independent, immutable, and safe to
hold in a dict keyed by game id, with no locking. `RandomAgent` takes an
injected `random.Random`, so games do not contend on shared generator state
and any game is reproducible from its seed.

---

## Testing

45 tests. The engine is exercised directly rather than through the frontends.

**Engine and rules**

- The worked example from the specification, as the primary ground truth
- Initial layout, including largest-at-bottom ordering
- Partial observability — an observation excludes the opponent's poles
- Legal actions: lift and place are mutually exclusive, skip is always legal,
  placement respects the size rule, lifting ignores it
- Illegal actions waste the turn without changing the position
- `step` does not mutate its input
- Each rule interpretation above has a test naming it
- The hostage deadlock, and the case where it backfires under the literal
  win reading

**Agents**

- Only legal actions are ever returned, over a full game
- Reproducible from a seed; different seeds diverge
- An agent receives only its own observation

**Frontends**

- Both move formats parse, and an unknown action kind is rejected
- Replay reproduces the spec example
- Replay stops once the game is decided
- Mismatched `turn_order` and `moves` lengths are rejected
- Random play respects `--max-turns` and is reproducible from a seed

Not tested: property-based invariants (that no reachable state has a stack out
of order), and performance at scale.

---

## AI assistance

Used throughout, as permitted.

**Tool:** Claude (Anthropic), as a design collaborator and to draft
implementation once decisions were settled.

**What it produced:** the module implementations, test bodies, and CLI, drafted
from decisions and a list of cases I specified. I reviewed every line before
committing and can account for each.

**What I did:** analysed the spec and identified the ambiguities, made all
seven decisions above, set the architecture, decided what to test, and
verified the behaviour against the worked example in the brief.

**What I rejected, and what the review caught.** Three things, all found by
testing or reviewing rather than by reading:

- The claim that simultaneous wins were unreachable, and that `DRAW` was only
  a defensive guard. I kept the branch and wrote a test around it; the test
  disproved the claim (Decision 3).
- The analysis of when the hostage strategy actually blocks a player. The
  first test asserted a deadlock and reported `A_WINS` instead — taking a disk
  hostage clears the shared pole, which can hand the victim the win
  (Decision 5).
- Two speculative helpers (`Player.opponent`, `GameState.with_poles`) that
  nothing called, and an outdated enum idiom that ruff flagged. All removed on
  a review pass.

Decision 3 and Decision 5 record the corrections rather than tidying them
away, because they are the clearest evidence the tests are doing real work
rather than confirming what was already believed. Where the model's reasoning
conflicted with the worked example or my reading of the rules, I went with the
spec.

---

## What I would do next

- Enforce deep immutability on the pole and hand mappings
- A `Game` object carrying config (`n`, `require_all_disks`, turn limit) so
  settings travel with the game rather than being threaded through calls
- Property-based tests asserting that no reachable state has an out-of-order
  stack
- A greedy or search-based agent, to check the `Agent` protocol holds for a
  policy that needs history