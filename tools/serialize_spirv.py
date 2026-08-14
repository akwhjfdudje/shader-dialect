#!/usr/bin/env python3
"""Serialize SPIR-V from shader-opt pipeline output.

Pipes: shader-opt -> mlir-opt-23 -> shader-opt -> extract-spirv-module -> mlir-translate

Usage:
  python3 serialize_spirv.py input.mlir -o output.spv
  build/tools/shader-opt ... | python3 serialize_spirv.py --stdin -o output.spv
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHADER_OPT = REPO / "build" / "tools" / "shader-opt" / "shader-opt"
MLIR_OPT = "/usr/bin/mlir-opt-23"
MLIR_TRANSLATE = "/usr/bin/mlir-translate-23"


def run_pipeline(input_path, output_path):
    if str(input_path) == "--stdin":
        stage1_input = sys.stdin.read()
    else:
        stage1_input = Path(input_path).read_text()

    # Stage 1: shader-opt (proxy backend + SPIR-V type/op conversion)
    p1 = subprocess.run(
        [str(SHADER_OPT), str(input_path) if str(input_path) != "--stdin" else "-",
         "--shader-lower-to-proxy-backend",
         "--shader-prune-dead-resources",
         "--shader-to-spirv"],
        input=stage1_input if str(input_path) == "--stdin" else None,
        capture_output=True, text=True, cwd=REPO)
    if p1.returncode != 0:
        print(p1.stderr, file=sys.stderr)
        return 1

    # Stage 2: mlir-opt-23 (arith + vector + func -> spirv)
    p2 = subprocess.run(
        [MLIR_OPT, "--convert-arith-to-spirv",
         "--convert-vector-to-spirv", "--convert-func-to-spirv"],
        input=p1.stdout, capture_output=True, text=True)
    if p2.returncode != 0:
        print(p2.stderr, file=sys.stderr)
        return 1

    # Stage 3: shader-opt (wrap in spirv.module)
    p3 = subprocess.run(
        [str(SHADER_OPT), "--shader-wrap-spirv-module"],
        input=p2.stdout, capture_output=True, text=True)
    if p3.returncode != 0:
        print(p3.stderr, file=sys.stderr)
        return 1

    # Extract spirv.module body (strip outer builtin.module wrapper)
    ir = p3.stdout
    start = ir.find("spirv.module")
    if start < 0:
        print("ERROR: no spirv.module found in output", file=sys.stderr)
        return 1
    # Find the matching closing brace for the spirv.module
    depth = 0
    end = start
    for i in range(start, len(ir)):
        if ir[i] == '{':
            depth += 1
        elif ir[i] == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    spirv_ir = ir[start:end]

    # Stage 4: serialize to SPIR-V binary
    p4 = subprocess.run(
        [MLIR_TRANSLATE, "--no-implicit-module", "--serialize-spirv",
         "-o", str(output_path)],
        input=spirv_ir, capture_output=True, text=True)
    if p4.returncode != 0:
        print(p4.stderr, file=sys.stderr)
        return 1

    binary_size = Path(output_path).stat().st_size
    print(f"SUCCESS: {binary_size} bytes -> {output_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Shader IR -> SPIR-V binary pipeline")
    parser.add_argument("input", help="Input .mlir file or --stdin")
    parser.add_argument("-o", "--output", default="/tmp/shader_output.spv",
                        help="Output .spv file")
    args = parser.parse_args()
    return run_pipeline(args.input, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
