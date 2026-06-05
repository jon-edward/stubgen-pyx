"""
Normalize spacing between class/module members.

Removes all existing blank lines, then adds a single blank line whenever the member type changes
(class -> function, function -> assignment, etc.). Members of the same type are kept together without
blank lines between them. After exiting a class, a blank line is always added.

For example:
    class A:
        '''
        This is a docstring.
        '''

        class B:
            pass

        def __init__(self): ...

        def b(self): ...
    C: int = ...


is converted to:

    class A:
        '''
        This is a docstring.
        '''

        class B:
            pass

        def __init__(self): ...
        def b(self): ...

    C: int = ...
"""

import ast
from dataclasses import dataclass, field
import enum
import tokenize
from typing import Union, TypeAlias

from ..parsing.utils import LineColConverter, tokenize_py, remove_indices


def normalize_member_spacing(code: str) -> str:
    for start, end in _get_blank_line_indices_to_remove(code):
        code = remove_indices(code, start, end, replace_with="")
    for start, end in _get_member_type_change_indices(code):
        code = remove_indices(code, start, end, replace_with="\n")
    return code


def _get_blank_line_indices_to_remove(code: str) -> list[tuple[int, int]]:
    results = []

    line_converter = LineColConverter(code)
    tokens = list(tokenize_py(code))

    for token in tokens:
        if token.type == tokenize.NL:
            start = line_converter.line_col_to_offset(token.start)
            end = line_converter.line_col_to_offset(token.end)
            results.append((start, end))

    results.sort(reverse=True)
    return results


def _get_member_type_change_indices(code: str) -> list[tuple[int, int]]:
    results = []

    visitor = _MemberVisitor()
    visitor.visit(ast.parse(code))

    line_converter = LineColConverter(code)
    for lineno in visitor.lines_needing_newlines:
        start = line_converter.line_col_to_offset((lineno, 0))
        results.append((start, start))

    results.sort(reverse=True)
    return results


class _MemberTypeState(enum.Enum):
    CLASS = "class"
    FUNCTION = "function"
    ASSIGNMENT = "assignment"
    PREVIOUS_CLASS = "previous_class"


_VISIT_TYPE: TypeAlias = Union[
    ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef, ast.AnnAssign, ast.Assign
]


@dataclass
class _MemberVisitor(ast.NodeVisitor):
    lines_needing_newlines: list[int] = field(default_factory=list, init=False)
    member_type_state: _MemberTypeState | None = field(default=None, init=False)

    def _visit_member(self, node: _VISIT_TYPE, member_type_state: _MemberTypeState):
        if (
            self.member_type_state is not None
            and self.member_type_state != member_type_state
        ):
            nodes = ()
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                nodes = node.decorator_list
            nodes = (*nodes, node)
            self.lines_needing_newlines.append(min(node.lineno for node in nodes))
        self.member_type_state = member_type_state

    def visit_ClassDef(self, node: ast.ClassDef):
        self._visit_member(node, _MemberTypeState.CLASS)
        self.member_type_state = None  # Do not add newline at first member
        self.generic_visit(node)
        self.member_type_state = (
            _MemberTypeState.PREVIOUS_CLASS
        )  # Force newline no matter what follows

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._visit_member(node, _MemberTypeState.FUNCTION)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._visit_member(node, _MemberTypeState.FUNCTION)

    def visit_Assign(self, node: ast.Assign):
        self._visit_member(node, _MemberTypeState.ASSIGNMENT)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        self._visit_member(node, _MemberTypeState.ASSIGNMENT)
