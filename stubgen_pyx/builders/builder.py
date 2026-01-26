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
    """
    Generates Python code from PyiElements.
    """

    def build_argument(
        self,
        argument: PyiArgument,
    ) -> str:
        output = f"{argument.name}"
        if argument.annotation is not None:
            output += f": {argument.annotation}"
        if argument.default is not None:
            output += f" = {argument.default}"
        return output

    def build_signature(
        self,
        signature: PyiSignature,
    ) -> str:
        arg_strings: list[str] = []
        if signature.num_posonly_args > 0:
            for i in range(signature.num_posonly_args):
                arg_strings.append(self.build_argument(signature.args[i]))
            arg_strings.append("/")
        for i in range(signature.num_posonly_args, len(signature.args)):
            arg_strings.append(self.build_argument(signature.args[i]))
        if signature.var_arg is not None:
            arg_strings.append("*" + self.build_argument(signature.var_arg))
        if signature.num_kwonly_args > 0 and signature.var_arg is None:
            arg_strings.insert(-signature.num_kwonly_args, "*")
        if signature.kw_arg is not None:
            arg_strings.append(f"**{signature.kw_arg.name}")
        output = f"({', '.join(arg_strings)})"
        if signature.return_type is not None:
            output += f" -> {signature.return_type}"
        return output

    def build_class(
        self,
        class_: PyiClass,
    ) -> str | None:
        output = "".join(f"{decorator}\n" for decorator in class_.decorators)
        output += f"class {class_.name}"

        if class_.bases or class_.metaclass is not None:
            output += "("
            for base in class_.bases:
                output += f"{base}, "
            if class_.metaclass is not None:
                output += f"metaclass={class_.metaclass}"
            output += ")"

        output += ": "
        content = ""
        if class_.doc is not None:
            content += f"{textwrap.indent(class_.doc, '    ')}\n"
        content += textwrap.indent(self.build_scope(class_.scope) or "", "    ")
        if content.rstrip():
            output += f"\n{content}"
        else:
            output += "..."
        return output

    def build_function(self, function: PyiFunction) -> str | None:
        output = "".join(f"{decorator}\n" for decorator in function.decorators)
        output += f"{'async ' if function.is_async else ''}def {function.name}{self.build_signature(function.signature)}: "
        if function.doc is not None:
            output += f"\n{textwrap.indent(function.doc, '    ')}"
        else:
            output += "..."
        return output

    def build_assignment(
        self,
        assignment: PyiAssignment,
    ) -> str | None:
        return assignment.statement

    def build_import(
        self,
        import_statement: PyiImport,
    ) -> str | None:
        return import_statement.statement

    def build_scope(self, scope: PyiScope) -> str | None:
        output = ""

        for element in scope.enums:
            enum_content = self.build_enum(element)
            if not enum_content:
                continue
            output += f"{enum_content}\n\n"

        for element in scope.assignments:
            assignment_content = self.build_assignment(element)
            if not assignment_content:
                continue
            output += f"{assignment_content}\n"
        if scope.assignments:
            output += "\n"

        for element in scope.functions:
            function_content = self.build_function(element)
            if not function_content:
                continue
            output += f"\n{function_content}\n"
        if scope.functions:
            output += "\n"

        for element in scope.classes:
            class_content = self.build_class(element)
            if not class_content:
                continue
            output += f"\n{class_content}\n"
        if scope.classes:
            output += "\n"

        return output or None

    def build_module(self, module: PyiModule) -> str:
        output = module.doc + "\n\n" if module.doc else ""

        for import_statement in module.imports:
            import_content = self.build_import(import_statement)
            if not import_content:
                continue
            output += f"{self.build_import(import_statement)}\n"
        if module.imports:
            output += "\n"

        output += f"{self.build_scope(module.scope) or ''}" or ""
        return output

    def build_enum(self, enum: PyiEnum) -> str | None:
        if not enum.names:
            return None

        annotations = [f"{name}: int" for name in enum.names]

        if enum.enum_name is not None:
            # Named enums are converted to classes
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
