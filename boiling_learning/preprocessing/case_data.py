import abc

from boiling_learning.preprocessing.cases import Case
from boiling_learning.preprocessing.experimental_data import ExperimentalData


class CaseData(abc.ABC):
    def __init__(
        self, case: Case, experimental_data: ExperimentalData, *, synced: bool = False
    ) -> None:
        self.case = case
        self.experimental_data = experimental_data
        self.synced = synced

    @abc.abstractmethod
    def sync(self, *, force: bool = False) -> None:
        pass
