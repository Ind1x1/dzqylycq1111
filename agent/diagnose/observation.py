class DiagnosisObservation:
    """
    DiagnosisObservation is to describe the problem observed
    by ErrorCodeDiagnostician.observe
    """

    def __init__(self, observation: str = ""):
        # The simple description info for the problem.
        self._observation: str = observation
        self.extra_infos: dict = {}

    @property
    def observation(self):
        return self._observation