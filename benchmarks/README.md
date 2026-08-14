# Shader-MLIR Benchmarks

**Scope note:** These benchmarks measure **compiler-analysis throughput** (wall-clock time for `shader-opt` to parse, transform, and print MLIR IR) and **static IR structural metrics** (op counts, argument counts, byte sizes). No GPU shader-execution performance is evaluated. The prototype has no executable GPU backend yet.

Benchmarks are listed in `benchmark_manifest.json`. Each case names an input
MLIR file and a `shader-opt` pass pipeline. The runner records wall-clock timing
and structural metrics before and after the pipeline.

Run from the repository root:

```sh
python3 tools/benchmark_shader.py --build-dir build --repeats 5
```

The manifest supports fixed inputs and synthetic generated inputs. Generated
cases use a `generator` name and integer `params`, then the runner writes the
MLIR to `build/benchmarks/generated/` before timing the pipeline.

Current generated benchmark families:

- `duplicate_reusable_samples`: stresses CSE of reusable texture samples.
- `ordinary_derivative_samples`: checks that ordinary samples remain conservative.
- `dead_resource_args`: stresses function-interface pruning.
- `call_chain_dead_resources`: measures today's direct-call pruning behavior.
- `mixed_material_variants`: stresses resource summary across many variants.

The call-chain case is intentionally useful as a limitation probe: the current
pruning pass removes dead resource args only when they are unused in that
function, so forwarded-but-eventually-dead resources are not fully propagated
through the call graph yet.

Write machine-readable output:

```sh
python3 tools/benchmark_shader.py --format json --output build/shader-benchmarks.json
```

Write SVG graphs for the default timing and structural metrics:

```sh
python3 tools/benchmark_shader.py --build-dir build --repeats 5 \
  --graph-dir build/shader-benchmark-graphs
```

The graph directory contains one dependency-free SVG bar chart per metric plus
an `index.md` file that embeds them. Choose specific numeric result fields with
`--graph-metrics`:

```sh
python3 tools/benchmark_shader.py --build-dir build \
  --graph-dir build/shader-benchmark-graphs \
  --graph-metrics time_ms_median,sample_ops_eliminated,resource_args_pruned
```

Add a new benchmark by adding an entry:

```json
{
  "name": "my_case",
  "description": "What this case is meant to prove.",
  "input": "examples/mlir/my_case.mlir",
  "pipeline": ["--cse", "--shader-resource-summary"]
}
```

Generated example:

```json
{
  "name": "many_reusable_samples",
  "generator": "duplicate_reusable_samples",
  "params": {"samples": 64, "channels": 4},
  "pipeline": ["--cse"]
}
```
