from collections.abc import Callable

class BaseClass:
    a: int

    def __init__(self):
        self.a = 1

    def b(self) -> int:
        return self.a


class ChildClass(BaseClass):
    def c(self) -> int:
        return self.a

    def d(self) -> int:
        return self.b()


__all__ = ['ChildClass', 'BaseClass', 'Callable']


if __name__ == '__main__':
    print(ChildClass().d())
