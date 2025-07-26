# tests/test_fsm.py
from autoboat.fsm import StateMachine

def test_fsm_initial_state_and_update():
    timeouts = {'search': 5, 'avoid': 5, 'forward': 5}
    sm = StateMachine(timeouts)
    # Initial state
    assert sm.state == 'SEARCH'
    # update returns None for stub and state stays unchanged
    result = sm.update({'dummy': True})
    assert result is None
    assert sm.state == 'SEARCH'
