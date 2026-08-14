from hanoi_crossing.engine import check_outcome, initial_state, step
from hanoi_crossing.model import Action, Outcome, Player, Pole

def test_spec_example_player_a_wins():
    state = initial_state(n=1)
    state = step(state, Player.A, Action.lift(Pole.P1A))
    state = step(state, Player.B, Action.lift(Pole.P1B))
    state = step(state, Player.A, Action.place(Pole.P3A))
    assert check_outcome(state) is Outcome.A_WINS