class StateMachine:
    """Handles SEARCH, FORWARD, AVOID, ERROR_STOP states."""
    def __init__(self, timeouts):
        self.timeouts = timeouts
        self.state = 'SEARCH'

    def update(self, event):
        """
        Process vision event and return a motor command (stub).
        """
        return None
