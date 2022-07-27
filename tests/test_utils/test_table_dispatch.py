from boiling_learning.utils.table_dispatch import TableDispatcher


class X:
    def __init__(self, value: int) -> None:
        self.value = value


def test_instance_dispatch() -> None:
    dispatcher = TableDispatcher()

    @dispatcher.dispatch(int)
    def _dispatch_int(obj: str) -> int:
        return int(obj)

    @dispatcher.dispatch(X)
    def _dispatch_x(obj: str) -> X:
        return X(dispatcher[int](obj))

    assert dispatcher[int]('2022') == 2022

    dispatched = dispatcher[X]('2022')
    assert isinstance(dispatched, X)
    assert dispatched.value == 2022
