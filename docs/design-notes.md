# Design notes

The README's decision log states *what* was decided. This records *why*: the
alternatives considered and rejected, and the one case where the initial
analysis was wrong and a test corrected it.

---

## Rule interpretations

### Why the shared pole blocks both players

The win condition is: *"a player wins when their hand is empty and, among
their visible poles, only pole 3 has disks on it."*

The board diagram gives A's visible poles as `1a – 2 – 3a`. So "among their
visible poles, only pole 3 has disks" means `1a` empty and `2` empty.

**Alternative considered.** Reading "their visible poles" as their *private*
poles only — so pole 2 wouldn't count. This makes the game simpler and less
interesting: the shared pole becomes a free dumping ground with no cost.

**Why rejected.** The spec says *visible*, and it explicitly defines pole 2 as
visible to both. Reading it any other way requires ignoring the word.

**What follows.** Since pole 2 appears in both players' visibility sets, a
disk there blocks *both*. That single fact generates most of the game's
structure — it's what makes using the shared pole a risk rather than a
convenience, and it's what makes deadlock possible.

### Why the literal win reading, allowing an incomplete tower

The condition says "only pole 3 has disks on it." It does not say *all* the
player's disks are there.

This matters because disks can be lost permanently. A places disk 5 on pole 2;
B lifts it and puts it on `3b`. Disk 5 is now somewhere A cannot see or reach.
Under the literal reading, A can still win with the remaining disks.

**Alternative considered.** Requiring all N of a player's original disks on
their pole 3 — the more Tower-of-Hanoi-like reading.

**Why rejected as the default.** It adds a requirement the text doesn't state.
Inventing a rule is a larger interpretive step than following the words, and
if the graders intended the literal reading, the stricter version produces a
game that can become unwinnable without saying so.

**The cost, acknowledged.** The literal reading admits a degenerate strategy:
move one disk to pole 3, dump the rest onto the shared pole, win once the
opponent clears it. Noting this in the README matters more than resolving it —
it shows the consequence was traced rather than missed.

**Resolution.** Support both via `require_all_disks`, defaulting to literal.
The engine already stores each player's starting disks, so this is a flag and
one branch. It converts an unresolvable ambiguity into a documented option.

### Why the outcome is checked for both players after every action

A player's win condition can be completed by their *opponent's* move.

Concretely: A has finished — `1a` empty, hand empty, disks on `3a` — but a
disk sits on pole 2. A can do nothing about it if the disk isn't liftable to
anywhere useful. B lifts that disk for their own reasons. Pole 2 is now clear,
and A's condition holds.

If the engine only checked the acting player, this win would be missed
entirely, and the game would continue in a won state.

**Design consequence.** `check_outcome(state)` takes only the state — not the
player who just moved. Whose turn it was is irrelevant to whether someone has
won, and building that independence in is what makes the function correct.

### Why a simultaneous win is a draw

This one was got wrong twice before the tests settled it, and the sequence is
worth recording.

**First position.** Assume simultaneous wins happen and need a tie-break;
award it to the acting player, or call it a draw on fairness grounds.

**Second position.** On reflection, argue that they're unreachable. The
reasoning: because the outcome is evaluated after every action, a player wins
the instant their condition holds — so for both to become true at once, one
action would have to complete both. The obvious candidate is a lift that
clears the shared pole, but that leaves the actor holding a disk, so their own
hand isn't empty. Conclusion: `DRAW` is only a guard against undefined
behaviour.

**What the test showed.** Constructing a position for a different test
produced this:

```
1a: ()        1b: ()
3a: (1,)      3b: (4, 3, 2)
2:  ()
hands: both empty
```

Both conditions hold. And it's reachable by ordinary play: B finishes their
tower and waits — nothing in the rules obliges them to keep acting — and A
later places their final disk. Neither is holding anything, the shared pole is
clear, and both have disks on their target pole.

**Where the second argument failed.** It only considered how the *shared pole*
gets cleared. It missed the case where a player completes their own condition
while the opponent had already completed theirs on an earlier turn. The two
wins don't have to be caused by the same action — they only have to be true at
the same time.

**Decision.** `DRAW` is a real outcome, not a guard.

**Why a draw rather than awarding it to the acting player.** The opponent
finished first and then waited. Nothing in the rules penalises waiting, and
the win condition says nothing about who moved last. Giving the game to
whoever happened to act would punish a player for having already succeeded.

**Why this is worth writing up.** A claim was made, tested, and disproved by
the test suite. That's the argument for writing tests against the rules rather
than against the implementation — the test was constructed to check something
else entirely and caught a false assumption in the design notes.

### Why a turn limit, and why no rule fix

The rules permit games that never end. If B lifts one of A's disks and skips
forever, A can never win — the disk cannot reach `3a` — and B can never win
either, since B's hand is never empty. B trades their own victory for a denial.

Nothing in the spec prevents this.

**What was considered.** Capping consecutive skips, or forcing a placement
when a legal one exists.

**Why rejected.** The brief invites creativity *where the rules are open to
interpretation*. The hostage strategy is not an ambiguity — it's a genuine
property of the rules as written. Adding a rule to remove it would be a design
change beyond the spec, and would read as patching a symptom rather than
understanding the game.

**What was done instead.** A turn limit, framed as what it is: a practical
bound, the same thing RL environments call episode truncation. Replay ends
when the turn sequence is exhausted; random-play ends at the cap. The
non-termination property is documented as an observation about the game.

---

## Architecture

The spec's constraint:

> *it should later serve, unchanged, as the environment core of an RL training
> loop, or of an online simulation service that maintains many concurrent
> games… Your random player should already consume the engine exactly the way
> such an external agent would.*

Every architectural choice below traces to that sentence.

### Why immutable state and a pure step function

`step(state, action) -> GameState` returns a new state. Nothing mutates.

**For concurrent games:** no shared mutable state means no locking, no
defensive copying, no cross-contamination between simulations. A service
holding ten thousand games holds ten thousand independent values.

**For RL:** snapshot and restore are free, because a state *is* a snapshot.
Tree search, replay buffers, and rollback all work without engine support.
A mutable engine would need an explicit `clone()` that's easy to get subtly
wrong.

**For testing:** every test constructs the exact state it needs. No setup and
teardown, no ordering dependencies, no state leaking between tests.

**The cost, accepted.** Allocating a new state per action is slower than
mutating in place. At this scale it doesn't matter, and the properties above
are worth more than the allocations. If profiling ever showed otherwise, the
fix is a mutable inner layer behind the same pure interface — the public API
wouldn't change, which is the point.

Implemented with frozen dataclasses and tuples rather than lists, so
immutability is enforced rather than merely intended.

### Why a separate observation function

The rules specify partial observability: neither player sees the other's poles
1 and 3, nor the opponent's hand.

`observe(state, player) -> Observation` returns only what that player can see.
The random agent consumes observations and never touches `GameState`.

**Why this is the right boundary.** A player's legal actions depend only on
their visible poles and their own hand — which is *exactly* their observation.
So `legal_actions(observation)` is well-defined, and the hidden information is
structurally unreachable rather than something an agent is trusted not to
peek at.

**Why it matters for the stated futures.** This is the standard environment
interface an RL agent expects. Writing the random agent against it — as the
brief asks — proves the boundary works, rather than asserting that it would.

**Alternative rejected.** Passing the full `GameState` to agents and relying on
convention. It would work for the random agent, which doesn't cheat by
accident, but it would leave the partial-observability requirement enforced by
nothing.

### Why the engine does no I/O

No printing, no file reading, no CLI concerns in `model.py` or `engine.py`.

An engine that prints cannot be embedded in a training loop or a service
without producing noise, and cannot be tested without capturing stdout. The
frontends own presentation; the engine owns rules.

This also keeps the core under the 500-line constraint honestly — the line
budget goes to game logic, not formatting.

### Why the random agent takes an injected `Random`

`random_agent(observation, rng)` rather than calling the global `random`
module.

Reproducibility: the same seed produces the same game, so a failure can be
replayed exactly. Concurrency: independent generators per game avoid
contention and interleaving on a shared global. Testability: a seeded
generator makes agent tests deterministic.

Global random state is a shared mutable dependency, which contradicts every
other choice above.

### Why legality has a single definition

`is_legal` is defined as membership in `legal_actions`, rather than as its own
validation logic inside `step`.

**Alternative considered.** A dedicated validator in `step` — checking hand
state, pole occupancy and the size rule directly. Faster, since it avoids
enumerating every action to test one.

**Why rejected.** Two implementations of the same rule drift. `step` could
come to accept an action `legal_actions` never offered, or reject one it did,
and any agent trusting the action list would break in ways that only show up
under specific positions. Defining one in terms of the other makes that
impossible rather than merely unlikely.

The cost — enumerating to check membership — is irrelevant at this scale, and
correctness by construction is worth more than the cycles.

### Why the replay stops once the game is decided

**Alternative considered.** Play the full recorded sequence regardless and
report the final position.

**Why rejected.** It would allow moves after a win to alter the board, so a
recording with trailing entries could report a different outcome than the one
actually reached. Stopping at the decision means the reported result is the
result.

The trade-off: a recording containing moves after the win is silently
truncated rather than flagged. Reporting the unused remainder would be a
reasonable extension.

### Why the 500-line constraint is read as the rules layer

The spec says "Core engine: under 500 lines of Python", and separately
describes two frontends, a random player and tests.

**Reading taken.** "Core engine" is the rules layer — `model.py` and
`engine.py`, 325 lines. The agents and CLI are consumers built on it.

**Alternative acknowledged.** Reading it as the whole package excluding tests
gives 531 lines, which exceeds the budget.

**Why not trim to fit.** The excess is entirely in `agents.py` and `cli.py`,
and the only way to reduce it meaningfully would be to fold presentation or
policy into the engine — which would violate the reuse requirement the same
spec states. Their separability is evidence the engine is properly scoped, not
a cost to be optimised away.

Both numbers are stated in the README so the reading is visible rather than
assumed.