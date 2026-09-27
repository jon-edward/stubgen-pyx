"""
Creates a call graph of the stubgen-pyx package
"""

import argparse
import pathlib

import pyan

CG_FORMATS = ["html", "svg", "tgf", "yed", "text"]


def main():
    parser = argparse.ArgumentParser(
        description="Creates a call graph of the stubgen-pyx package"
    )
    parser.add_argument("output_path", type=pathlib.Path, help="Output path")

    args = parser.parse_args()
    output_path: pathlib.Path = args.output_path

    format_ = output_path.suffix[1:]
    format_ = "text" if format_ == "txt" else format_

    assert format_ in CG_FORMATS, (
        f"Invalid format {format_} (must be one of {CG_FORMATS})"
    )

    output_str = pyan.create_callgraph(
        filenames="stubgen_pyx",
        format=format_,
        colored=True,
        nested_groups=True,
        draw_defines=True,
        draw_uses=True,
        depth=2,
        direction="both",
        concentrate=True,
        layout="fdp",
    )

    output_str = output_str.replace(
        "https://cdn.rawgit.com/jquery/jquery-mousewheel/master/jquery.mousewheel.min.js",
        "https://cdnjs.cloudflare.com/ajax/libs/jquery-mousewheel/3.2.2/jquery.mousewheel.min.js",
    )

    output_path.write_text(output_str)


if __name__ == "__main__":
    main()
