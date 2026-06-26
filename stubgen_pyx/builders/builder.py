"""
Generates Python code from PyiElements.
"""

from __future__ import annotations

from dataclasses import dataclass
import textwrap

from ..models.pyi_elements import (
    PyiModule,
    PyiClass,
    PyiFunction,
    PyiAssignment,
    PyiImport,
    PyiScope,
    PyiSignature,
    PyiArgument,
    PyiEnum,
)


@dataclass
class Builder:
    """Generates Python .pyi stub code from PyiElements.

    Converts intermediate PyiElement representations into properly formatted
    Python stub file text, handling indentation, decorators, docstrings,
    and type annotations.
    """

    include_private: bool = False

    def _is_private(self, name: str) -> bool:
        """Check if a name is private (starts with _ but doesn't end with _)."""
        return name.startswith("_") and not name.endswith("_")

    def build_argument(self, argument: PyiArgument) -> str:
        """Build a string representation of a function argument.

        Includes name, annotation, and default value as applicable.

        Args:
            argument: The PyiArgument to build.

        Returns:
            String like "x", "x: int", or "x: int = 5".
        """
        parts = [argument.name]
        if argument.annotation is not None:
            parts.append(f": {argument.annotation}")
        if argument.default is not None:
            parts.append(f" = {argument.default}")
        return "".join(parts)

    def build_signature(self, signature: PyiSignature) -> str:
        """Build a function signature string.

        Handles positional-only, regular, *args, keyword-only, **kwargs,
        and return type annotations.

        Args:
            signature: The PyiSignature to build.

        Returns:
            String like "(x: int, y: str) -> bool".
        """
        arg_strings: list[str] = []
        kwonly_args = (
            signature.args[-signature.num_kwonly_args :]
            if signature.num_kwonly_args > 0
            else []
        )
        positional_args = signature.args[: len(signature.args) - len(kwonly_args)]

        if signature.num_posonly_args > 0:
            for i in range(min(signature.num_posonly_args, len(positional_args))):
                arg_strings.append(self.build_argument(positional_args[i]))
            arg_strings.append("/")

        for i in range(signature.num_posonly_args, len(positional_args)):
            arg_strings.append(self.build_argument(positional_args[i]))

        if signature.var_arg is not None:
            arg_strings.append("*" + self.build_argument(signature.var_arg))

        if signature.num_kwonly_args > 0:
            if signature.var_arg is None:
                arg_strings.append("*")
            for arg in kwonly_args:
                arg_strings.append(self.build_argument(arg))

        if signature.kw_arg is not None:
            arg_strings.append(f"**{signature.kw_arg.name}")
        sig = f"({', '.join(arg_strings)})"
        if signature.return_type is not None:
            sig += f" -> {signature.return_type}"
        return sig

    def build_class(self, class_: PyiClass) -> str | None:
        """Build a class definition string.

        Args:
            class_: The PyiClass to build.

        Returns:
            Class definition with docstring, decorators, bases, and body.
        """
        if not self.include_private and self._is_private(class_.name):
            return None
        parts = ["".join(f"{d}\n" for d in class_.decorators)]
        parts.append(f"class {class_.name}")

        if class_.bases or class_.metaclass is not None:
            inheritance_parts = list(class_.bases)
            if class_.metaclass is not None:
                inheritance_parts.append(f"metaclass={class_.metaclass}")
            parts.append(f"({', '.join(inheritance_parts)})")

        parts.append(": ")
        content = ""
        if class_.doc is not None:
            content += f"{textwrap.indent(class_.doc, '    ')}\n"
        content += textwrap.indent(self.build_scope(class_.scope) or "", "    ")
        if content.rstrip():
            parts.append(f"\n{content}")
        else:
            parts.append("...")
        return "".join(parts)

    def build_function(self, function: PyiFunction) -> str | None:
        """Build a function definition string.

        Args:
            function: The PyiFunction to build.

        Returns:
            Function signature with docstring and decorators.
        """
        if not self.include_private and self._is_private(function.name):
            return None
        parts = ["".join(f"{d}\n" for d in function.decorators)]
        async_prefix = "async " if function.is_async else ""
        parts.append(
            f"{async_prefix}def {function.name}{self.build_signature(function.signature)}:"
        )
        if function.type_comment:
            parts.append(f"  {function.type_comment}")
        if function.doc is not None:
            parts.append(f"\n{textwrap.indent(function.doc, '    ')}")
        elif function.type_comment:
            parts.append("\n    ...")
        else:
            parts.append(" ...")
        return "".join(parts)

    def build_assignment(self, assignment: PyiAssignment) -> str | None:
        """Build an assignment statement string."""
        if not self.include_private:
            name = assignment.statement.partition("=")[0].partition(":")[0].strip()
            if self._is_private(name):
                return None
        return assignment.statement

    def build_import(self, import_statement: PyiImport) -> str | None:
        """Build an import statement string."""
        return import_statement.statement

    def build_scope(self, scope: PyiScope) -> str | None:
        """Build a scope (module or class body) string.

        Combines enums, classes, assignments, and functions in proper order.

        Args:
            scope: The PyiScope to build.

        Returns:
            Formatted scope contents, or None if empty.
        """
        chunks: list[str] = []

        enum_lines: list[str] = []
        for element in scope.enums:
            enum_content = self.build_enum(element)
            if enum_content:
                enum_lines.append(f"{enum_content}\n\n")
        if enum_lines:
            chunks.append("".join(enum_lines) + "\n")

        assignment_lines: list[str] = []
        for element in scope.assignments:
            assignment_content = self.build_assignment(element)
            if assignment_content:
                assignment_lines.append(f"{assignment_content}\n")
        if assignment_lines:
            chunks.append("".join(assignment_lines) + "\n")

        class_lines: list[str] = []
        for element in scope.classes:
            class_content = self.build_class(element)
            if class_content:
                class_lines.append(f"{class_content}\n\n")
        if class_lines:
            chunks.append("".join(class_lines) + "\n")

        function_lines: list[str] = []
        for element in scope.functions:
            function_content = self.build_function(element)
            if function_content:
                function_lines.append(f"\n{function_content}\n")
        if function_lines:
            chunks.append("".join(function_lines) + "\n")

        output = "".join(chunks)
        return output or None

    def build_module(self, module: PyiModule) -> str:
        """Build a complete module .pyi file string.

        Combines module docstring, imports, and scope in proper order.

        Args:
            module: The PyiModule to build.

        Returns:
            Complete .pyi file content.
        """
        parts: list[str] = []
        if module.doc:
            parts.append(f"{module.doc}\n\n")

        import_lines = [
            f"{self.build_import(imp)}\n"
            for imp in module.imports
            if self.build_import(imp)
        ]
        if import_lines:
            parts.append("".join(import_lines) + "\n")

        parts.append(self.build_scope(module.scope) or "")
        return "".join(parts)

    def build_enum(self, enum: PyiEnum | PyiAssignment) -> str | None:
        """Build an enum definition.

        Named enums are converted to classes with int attributes.
        Unnamed enums are generated as bare int assignments.
        """

        if isinstance(enum, PyiAssignment):
            return enum.statement

        if not enum.names:
            return None

        annotations = [f"{name}: int" for name in enum.names]

        if enum.enum_name is not None:
            class_ = PyiClass(
                name=enum.enum_name,
                scope=PyiScope(
                    assignments=[
                        PyiAssignment(annotation) for annotation in annotations
                    ],
                ),
            )
            return self.build_class(class_)

        return "\n".join(annotations)
