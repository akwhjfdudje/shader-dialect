#!/usr/bin/env python3
"""
@file benchmark_shader.py
@brief Run structural Shader-MLIR benchmarks.

This script is intentionally lightweight: it times shader-opt pipelines and extracts
textual IR metrics that are meaningful before a backend exists. Future benchmark
kinds can reuse the manifest and add richer metric collectors.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


RESOURCE_TYPES = ("!shader.texture", "!shader.sampler", "!shader.buffer")
SHADER_OPS = (
    "shader.texture_sample",
    "shader.texture_sample_reusable",
    "shader.channel",
    "shader.saturate",
    "shader.derivative_hint",
)
SPIRV_OPS = (
    "spirv.SampledImage",
    "spirv.ImageSampleImplicitLod",
    "spirv.CompositeExtract",
    "spirv.FMul",
    "spirv.FAdd",
    "spirv.FSub",
    "spirv.GL.FMax",
    "spirv.GL.FMin",
    "spirv.Constant",
    "spirv.Select",
    "spirv.IsNan",
)
PROXY_BACKEND_PASS = "--shader-lower-to-proxy-backend"
DEFAULT_GRAPH_METRICS = (
    "time_ms_median",
    "time_ms_p95",
    "sample_ops_eliminated",
    "resource_args_pruned",
    "remaining_shader_op_count",
    "spirv_op_count",
    "proxy_backend_op_count",
    "output_bytes_delta",
)


@dataclass(frozen=True)
class BenchmarkCase:
    """
    Represents a single benchmark configuration.

    Attributes:
        input: Path to a fixed MLIR file, or None if using a generator.
        generator: The name of a synthetic IR generator.
        pipeline: List of shader-opt passes to execute.
    """
    name: str
    description: str
    input: Path | None
    generator: str | None
    params: dict[str, int]
    pipeline: tuple[str, ...]


def run_command(command: list[str], *, cwd: Path) -> tuple[str, str, float]:
    """Executes a subprocess command and returns (stdout, stderr, elapsed_ms)."""
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return completed.stdout, completed.stderr, elapsed_ms


def load_manifest(path: Path, repo_root: Path) -> list[BenchmarkCase]:
    data = json.loads(path.read_text())
    cases: list[BenchmarkCase] = []
    for item in data.get("benchmarks", []):
        cases.append(
            BenchmarkCase(
                name=item["name"],
                description=item.get("description", ""),
                input=repo_root / item["input"] if "input" in item else None,
                generator=item.get("generator"),
                params={key: int(value) for key, value in item.get("params", {}).items()},
                pipeline=tuple(item.get("pipeline", [])),
            )
        )
    return cases


def param(params: dict[str, int], name: str, default: int) -> int:
    """Helper to extract integer parameters for IR generators."""
    value = params.get(name, default)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def sample_op(op: str, result: str, texture: str, sampler: str, uv: str) -> str:
    """Generates a raw string representation of a shader.texture_sample op."""
    return (
        f'    {result} = "shader.{op}"({texture}, {sampler}, {uv}) '
        f': (!shader.texture, !shader.sampler, vector<2xf32>) -> vector<4xf32>'
    )


def channel_op(result: str, sample: str, channel: int) -> str:
    """Generates a raw string representation of a shader.channel op."""
    return (
        f'    {result} = "shader.channel"({sample}) '
        f'{{channel = {channel} : i32}} : (vector<4xf32>) -> f32'
    )


def generate_duplicate_reusable(params: dict[str, int]) -> str:
    """Generates IR with multiple identical reusable samples to stress CSE."""
    samples = max(param(params, "samples", 16), 1)
    channels = max(param(params, "channels", 4), 1)
    lines = [
        "module {",
        "  func.func @generated_duplicate_reusable(%tex: !shader.texture, %sampler: !shader.sampler, %uv: vector<2xf32>) -> f32 {",
    ]
    for index in range(samples):
        lines.append(sample_op("texture_sample_reusable", f"%s{index}", "%tex", "%sampler", "%uv"))
        lines.append(channel_op(f"%c{index}", f"%s{index}", index % min(channels, 4)))
    lines.append("    return %c0 : f32")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def generate_ordinary_derivative(params: dict[str, int]) -> str:
    """Generates IR with ordinary samples that should NOT be merged by CSE."""
    samples = max(param(params, "samples", 16), 1)
    lines = [
        "module {",
        "  func.func @generated_ordinary_derivative(%tex: !shader.texture, %sampler: !shader.sampler, %uv: vector<2xf32>) -> f32 {",
        '    %duv = "shader.derivative_hint"(%uv) : (vector<2xf32>) -> vector<2xf32>',
    ]
    for index in range(samples):
        lines.append(sample_op("texture_sample", f"%s{index}", "%tex", "%sampler", "%duv"))
        lines.append(channel_op(f"%c{index}", f"%s{index}", index % 4))
    lines.append("    return %c0 : f32")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def generate_dead_resources(params: dict[str, int]) -> str:
    """Generates IR with many unused function arguments to stress pruning."""
    live_textures = max(param(params, "live_textures", 1), 1)
    dead_textures = param(params, "dead_textures", 8)
    lines = ["module {"]
    args = [f"%live{i}: !shader.texture" for i in range(live_textures)]
    args += [f"%dead{i}: !shader.texture" for i in range(dead_textures)]
    args += ["%sampler: !shader.sampler", "%uv: vector<2xf32>"]
    lines.append(f"  func.func @generated_dead_resources({', '.join(args)}) -> f32 {{")
    for index in range(live_textures):
        lines.append(sample_op("texture_sample", f"%s{index}", f"%live{index}", "%sampler", "%uv"))
    lines.append(channel_op("%c0", "%s0", 0))
    lines.append("    return %c0 : f32")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def generate_call_chain(params: dict[str, int]) -> str:
    """Generates a deep call graph where resources are passed down but unused."""
    functions = max(param(params, "functions", 4), 1)
    dead_textures = param(params, "dead_textures", 2)
    lines = ["module {"]
    for index in range(functions):
        args = ["%tex: !shader.texture"]
        args += [f"%dead{dead}: !shader.texture" for dead in range(dead_textures)]
        args += ["%sampler: !shader.sampler", "%uv: vector<2xf32>"]
        lines.append(f"  func.func @chain{index}({', '.join(args)}) -> f32 {{")
        if index + 1 < functions:
            operands = ["%tex"] + [f"%dead{dead}" for dead in range(dead_textures)] + ["%sampler", "%uv"]
            types = ["!shader.texture"] * (1 + dead_textures) + ["!shader.sampler", "vector<2xf32>"]
            lines.append(
                f"    %out = func.call @chain{index + 1}({', '.join(operands)}) "
                f": ({', '.join(types)}) -> f32"
            )
            lines.append("    return %out : f32")
        else:
            lines.append(sample_op("texture_sample", "%s0", "%tex", "%sampler", "%uv"))
            lines.append(channel_op("%c0", "%s0", 0))
            lines.append("    return %c0 : f32")
        lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def generate_mixed_variants(params: dict[str, int]) -> str:
    """Generates many function variants to stress resource summary analysis."""
    variants = max(param(params, "variants", 4), 1)
    max_textures = max(param(params, "max_textures", 4), 1)
    lines = ["module {"]
    for variant in range(variants):
        texture_count = 1 + (variant % max_textures)
        args = [f"%tex{i}: !shader.texture" for i in range(texture_count)]
        args += ["%sampler: !shader.sampler", "%uv: vector<2xf32>"]
        lines.append(f"  func.func @variant{variant}({', '.join(args)}) -> f32 {{")
        for texture in range(texture_count):
            op = "texture_sample_reusable" if texture % 2 else "texture_sample"
            lines.append(sample_op(op, f"%s{texture}", f"%tex{texture}", "%sampler", "%uv"))
        lines.append(channel_op("%c0", "%s0", 0))
        lines.append("    return %c0 : f32")
        lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


GENERATORS = {
    "duplicate_reusable_samples": generate_duplicate_reusable,
    "ordinary_derivative_samples": generate_ordinary_derivative,
    "dead_resource_args": generate_dead_resources,
    "call_chain_dead_resources": generate_call_chain,
    "mixed_material_variants": generate_mixed_variants,
}


def materialize_case_input(
    case: BenchmarkCase, *, repo_root: Path, generated_dir: Path | None
) -> tuple[Path, str]:
    """Ensures input MLIR exists on disk, either by locating it or generating it."""
    if case.input is not None:
        return case.input, str(case.input.relative_to(repo_root))
    if not case.generator:
        raise ValueError(f"benchmark {case.name} needs input or generator")
    try:
        generator = GENERATORS[case.generator]
    except KeyError as exc:
        raise ValueError(f"unknown generator {case.generator!r}") from exc
    if generated_dir is None:
        raise ValueError("generated benchmarks require a generated_dir")
    generated_dir.mkdir(parents=True, exist_ok=True)
    path = generated_dir / f"{case.name}.mlir"
    path.write_text(generator(case.params))
    return path, f"<generated:{case.generator}>"


def split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char in "(<[":
            depth += 1
        elif char in ")>]":
            depth = max(depth - 1, 0)
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def count_func_args(ir: str) -> int:
    total = 0
    for match in re.finditer(r"func\.func\s+@[^(]+\((.*?)\)\s*(?:->|\{)", ir, re.S):
        args = match.group(1).strip()
        if args:
            total += len(split_top_level_commas(args))
    return total


def iter_func_args(ir: str) -> Iterable[str]:
    for match in re.finditer(r"func\.func\s+@[^(]+\((.*?)\)\s*(?:->|\{)", ir, re.S):
        args = match.group(1).strip()
        if args:
            yield from split_top_level_commas(args)


def count_resource_func_args(ir: str) -> int:
    return sum(
        1
        for arg in iter_func_args(ir)
        if any(resource_type in arg for resource_type in RESOURCE_TYPES)
    )


def count_call_operands(ir: str) -> int:
    total = 0
    for match in re.finditer(r"(?:func\.)?call\s+@[^(]+\((.*?)\)", ir):
        operands = match.group(1).strip()
        if operands:
            total += len(split_top_level_commas(operands))
    return total


def collect_metrics(ir: str) -> dict[str, int]:
    metrics: dict[str, int] = {}
    metrics["func_count"] = len(re.findall(r"\bfunc\.func\s+@", ir))
    metrics["func_arg_count"] = count_func_args(ir)
    metrics["call_operand_count"] = count_call_operands(ir)
    metrics["resource_arg_count"] = count_resource_func_args(ir)
    metrics["resource_type_occurrence_count"] = sum(
        ir.count(resource) for resource in RESOURCE_TYPES
    )
    metrics["texture_type_count"] = ir.count("!shader.texture")
    metrics["sampler_type_count"] = ir.count("!shader.sampler")
    metrics["buffer_type_count"] = ir.count("!shader.buffer")
    metrics["arith_op_count"] = len(re.findall(r"\barith\.[A-Za-z0-9_]+", ir))
    metrics["vector_op_count"] = len(re.findall(r"\bvector\.[A-Za-z0-9_]+", ir))
    metrics["return_count"] = len(re.findall(r"\breturn\b", ir))
    for op in SHADER_OPS:
        key = op.replace("shader.", "").replace(".", "_") + "_count"
        metrics[key] = ir.count(f'"{op}"')
    metrics["shader_op_count"] = sum(
        metrics[op.replace("shader.", "").replace(".", "_") + "_count"]
        for op in SHADER_OPS
    )
    metrics["spirv_op_count"] = sum(ir.count(op) for op in SPIRV_OPS)
    return metrics


def parse_resource_summary(output: str) -> dict[str, int]:
    resources = re.findall(
        r"resource\(texture=([^,]+), sampler=([^)]+)\) "
        r"ordinary_samples=(\d+) reusable_samples=(\d+) channels=\[([^\]]*)\]",
        output,
    )
    ordinary = 0
    reusable = 0
    channels = 0
    for _, _, ordinary_text, reusable_text, channel_text in resources:
        ordinary += int(ordinary_text)
        reusable += int(reusable_text)
        if channel_text.strip():
            channels += len([part for part in channel_text.split(",") if part.strip()])
    return {
        "summary_resource_count": len(resources),
        "summary_ordinary_samples": ordinary,
        "summary_reusable_samples": reusable,
        "summary_channel_count": channels,
    }


def metric_delta(before: dict[str, int], after: dict[str, int], key: str) -> int:
    return after.get(key, 0) - before.get(key, 0)


def summarize_timing(times: Iterable[float]) -> dict[str, float]:
    values = list(times)
    sorted_values = sorted(values)
    p95_index = min(math.ceil(len(sorted_values) * 0.95) - 1, len(sorted_values) - 1)
    return {
        "time_ms_min": min(values),
        "time_ms_mean": statistics.fmean(values),
        "time_ms_median": statistics.median(values),
        "time_ms_p95": sorted_values[p95_index],
        "time_ms_max": max(values),
    }


def parse_mlir_timing(stderr_output: str) -> dict[str, float]:
    """Parse --mlir-timing stderr output.  Returns dict of pass_name -> wall_time_ms."""
    timings: dict[str, float] = {}
    in_table = False
    for line in stderr_output.splitlines():
        if "---Wall Time---" in line:
            in_table = True
            continue
        if not in_table or not line.strip():
            continue
        if "Total" in line:
            break
        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                wall_sec = float(parts[0])
                name = parts[-1]
                # Strip C++ namespace / anonymous-namespace prefixes
                if "::" in name:
                    name = name.split("::")[-1]
                timings[name] = wall_sec * 1000.0
            except ValueError:
                pass
    return timings


def run_benchmark(
    case: BenchmarkCase,
    *,
    shader_opt: Path,
    repo_root: Path,
    repeats: int,
    warmups: int,
    generated_dir: Path | None,
) -> dict[str, object]:
    input_path, input_label = materialize_case_input(
        case, repo_root=repo_root, generated_dir=generated_dir
    )
    baseline_ir, _, _ = run_command([str(shader_opt), str(input_path)], cwd=repo_root)

    times: list[float] = []
    output = ""
    timing_output = ""
    command = [str(shader_opt), str(input_path), *case.pipeline, "--mlir-timing"]
    for _ in range(warmups):
        run_command(command, cwd=repo_root)
    for _ in range(repeats):
        output, timing_output, elapsed_ms = run_command(command, cwd=repo_root)
        times.append(elapsed_ms)

    pass_timings = parse_mlir_timing(timing_output)

    before = collect_metrics(baseline_ir)
    after = collect_metrics(output)
    summary = parse_resource_summary(output)
    external_sample_count = (
        after["texture_sample_count"] + after["texture_sample_reusable_count"]
    )
    lowered_math_op_count = after["arith_op_count"]
    lowered_vector_op_count = after["vector_op_count"]

    result: dict[str, object] = {
        "name": case.name,
        "description": case.description,
        "input": input_label,
        "generator": case.generator or "",
        "params": json.dumps(case.params, sort_keys=True),
        "pipeline": " ".join(case.pipeline) if case.pipeline else "<none>",
        **summarize_timing(times),
        "input_bytes": len(baseline_ir.encode()),
        "output_bytes": len(output.encode()),
        "output_bytes_delta": len(output.encode()) - len(baseline_ir.encode()),
        "proxy_backend_enabled": PROXY_BACKEND_PASS in case.pipeline,
        "lowered_ir_bytes": len(output.encode()),
        "lowered_ir_bytes_delta": len(output.encode()) - len(baseline_ir.encode()),
        "remaining_shader_op_count": after["shader_op_count"],
        "spirv_op_count": after.get("spirv_op_count", 0),
        "external_sample_op_count": external_sample_count,
        "lowered_math_op_count": lowered_math_op_count,
        "lowered_vector_op_count": lowered_vector_op_count,
        "proxy_backend_op_count": (
            external_sample_count + lowered_math_op_count + lowered_vector_op_count
        ),
        "func_args_before": before["func_arg_count"],
        "func_args_after": after["func_arg_count"],
        "func_args_delta": metric_delta(before, after, "func_arg_count"),
        "resource_args_before": before["resource_arg_count"],
        "resource_args_after": after["resource_arg_count"],
        "resource_args_delta": metric_delta(before, after, "resource_arg_count"),
        "shader_ops_before": before["shader_op_count"],
        "shader_ops_after": after["shader_op_count"],
        "shader_ops_delta": metric_delta(before, after, "shader_op_count"),
        "ordinary_samples_before": before["texture_sample_count"],
        "ordinary_samples_after": after["texture_sample_count"],
        "ordinary_samples_delta": metric_delta(before, after, "texture_sample_count"),
        "reusable_samples_before": before["texture_sample_reusable_count"],
        "reusable_samples_after": after["texture_sample_reusable_count"],
        "reusable_samples_delta": metric_delta(
            before, after, "texture_sample_reusable_count"
        ),
        "sample_ops_eliminated": -(
            metric_delta(before, after, "texture_sample_count")
            + metric_delta(before, after, "texture_sample_reusable_count")
        ),
        "resource_args_pruned": -metric_delta(before, after, "resource_arg_count"),
        "channels_before": before["channel_count"],
        "channels_after": after["channel_count"],
        "channels_delta": metric_delta(before, after, "channel_count"),
        "call_operands_before": before["call_operand_count"],
        "call_operands_after": after["call_operand_count"],
        "call_operands_delta": metric_delta(before, after, "call_operand_count"),
        **summary,
        "pass_timing_total_ms": sum(pass_timings.values()),
    }
    # Emit one field per pass so it shows in JSON/CSV output
    for pass_name, wall_ms in pass_timings.items():
        result[f"pass_time_{pass_name}_ms"] = wall_ms
    return result


def print_markdown(results: list[dict[str, object]]) -> None:
    columns = [
        "name",
        "time_ms_median",
        "time_ms_p95",
        "resource_args_delta",
        "resource_args_pruned",
        "reusable_samples_delta",
        "sample_ops_eliminated",
        "ordinary_samples_delta",
        "output_bytes_delta",
        "summary_resource_count",
        "summary_channel_count",
        "remaining_shader_op_count",
        "external_sample_op_count",
        "lowered_math_op_count",
        "lowered_vector_op_count",
        "lowered_ir_bytes_delta",
    ]
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join("---" for _ in columns) + " |")
    for result in results:
        row = []
        for column in columns:
            value = result.get(column, "")
            if isinstance(value, float):
                value = f"{value:.3f}"
            row.append(str(value))
        print("| " + " | ".join(row) + " |")


def numeric_value(result: dict[str, object], key: str) -> float | None:
    value = result.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def format_number(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:.3f}"


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", text.strip()).strip("-").lower()
    return slug or "graph"


def render_bar_chart_svg(
    results: list[dict[str, object]], metric: str, *, title: str | None = None
) -> str:
    values: list[tuple[str, float]] = []
    for result in results:
        value = numeric_value(result, metric)
        if value is not None:
            values.append((str(result["name"]), value))

    if not values:
        raise ValueError(f"metric {metric!r} has no numeric values")

    label_width = 280
    value_width = 96
    chart_width = 520
    row_height = 26
    top = 54
    bottom = 26
    width = label_width + chart_width + value_width + 48
    height = top + row_height * len(values) + bottom
    min_value = min(0.0, *(value for _, value in values))
    max_value = max(0.0, *(value for _, value in values))
    span = max_value - min_value
    if span == 0:
        span = max(abs(max_value), 1.0)
        min_value = 0.0
        max_value = span

    def x_for(value: float) -> float:
        return label_width + 24 + ((value - min_value) / span) * chart_width

    zero_x = x_for(0.0)
    title_text = title or metric
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:12px;fill:#172033}",
        ".title{font-size:16px;font-weight:700}",
        ".axis{stroke:#8892a6;stroke-width:1}",
        ".bar-positive{fill:#2f7d64}",
        ".bar-negative{fill:#b55345}",
        ".tick{fill:#596275}",
        "</style>",
        f'<text class="title" x="16" y="28">{html.escape(title_text)}</text>',
        f'<text class="tick" x="{label_width + 24}" y="48">{html.escape(format_number(min_value))}</text>',
        f'<text class="tick" text-anchor="middle" x="{zero_x:.1f}" y="48">0</text>',
        f'<text class="tick" text-anchor="end" x="{label_width + 24 + chart_width}" y="48">{html.escape(format_number(max_value))}</text>',
        f'<line class="axis" x1="{zero_x:.1f}" y1="{top - 10}" x2="{zero_x:.1f}" y2="{height - bottom + 4}"/>',
    ]

    for index, (name, value) in enumerate(values):
        y = top + index * row_height
        value_x = x_for(value)
        bar_x = min(zero_x, value_x)
        bar_width = max(abs(value_x - zero_x), 1.0)
        bar_class = "bar-positive" if value >= 0 else "bar-negative"
        parts.extend(
            [
                f'<text x="16" y="{y + 15}">{html.escape(name)}</text>',
                f'<rect class="{bar_class}" x="{bar_x:.1f}" y="{y + 4}" width="{bar_width:.1f}" height="16" rx="2"/>',
                f'<text x="{label_width + chart_width + 36}" y="{y + 15}">{html.escape(format_number(value))}</text>',
            ]
        )

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_graphs(
    graph_dir: Path, results: list[dict[str, object]], metrics: Iterable[str]
) -> list[Path]:
    graph_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for metric in metrics:
        metric = metric.strip()
        if not metric:
            continue
        svg = render_bar_chart_svg(results, metric, title=metric.replace("_", " "))
        path = graph_dir / f"{slugify(metric)}.svg"
        path.write_text(svg)
        written.append(path)

    index = graph_dir / "index.md"
    lines = ["# Shader-MLIR Benchmark Graphs", ""]
    for path in written:
        lines.append(f"![{path.stem}]({path.name})")
        lines.append("")
    index.write_text("\n".join(lines))
    written.append(index)
    return written


def write_output(path: Path, output_format: str, results: list[dict[str, object]]) -> None:
    if output_format == "json":
        path.write_text(json.dumps(results, indent=2) + "\n")
        return

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="benchmarks/benchmark_manifest.json")
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--shader-opt")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--generated-dir")
    parser.add_argument("--format", choices=("markdown", "json", "csv"), default="markdown")
    parser.add_argument("--output")
    parser.add_argument(
        "--graph-dir",
        help="Write dependency-free SVG bar charts for selected benchmark metrics.",
    )
    parser.add_argument(
        "--graph-metrics",
        default=",".join(DEFAULT_GRAPH_METRICS),
        help="Comma-separated numeric result fields to graph.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    manifest = repo_root / args.manifest
    shader_opt = (
        Path(args.shader_opt)
        if args.shader_opt
        else repo_root / args.build_dir / "tools" / "shader-opt" / "shader-opt"
    )

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if not shader_opt.exists():
        parser.error(f"shader-opt not found: {shader_opt}")

    cases = load_manifest(manifest, repo_root)
    generated_dir = (
        Path(args.generated_dir)
        if args.generated_dir
        else repo_root / args.build_dir / "benchmarks" / "generated"
    )
    results = [
        run_benchmark(
            case,
            shader_opt=shader_opt,
            repo_root=repo_root,
            repeats=args.repeats,
            warmups=args.warmups,
            generated_dir=generated_dir,
        )
        for case in cases
    ]

    if args.output:
        write_output(Path(args.output), args.format, results)
    elif args.format == "markdown":
        print_markdown(results)
    else:
        print(json.dumps(results, indent=2))

    if args.graph_dir:
        metrics = [metric.strip() for metric in args.graph_metrics.split(",")]
        written = write_graphs(Path(args.graph_dir), results, metrics)
        if not args.output and args.format == "markdown":
            print()
            print("Graphs:")
            for path in written:
                print(f"- {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
