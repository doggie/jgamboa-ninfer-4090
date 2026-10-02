# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

NInfer is a from-scratch C++20/CUDA inference engine for a single GPU: one resident model,
1–8 concurrent "lanes", bounded FIFO ingress, no preemption, no multi-GPU. This branch is the
**RTX 4090 (`sm_89`) port**, developed and validated on Windows 11 (MSVC + CUDA, no WSL/Docker);
the Linux build (CMake + vcpkg, `Dockerfile`) is inherited from upstream and less re-validated.
It serves the Qwen3.5/3.6/3.8 family (27B dense: 48 Gated DeltaNet linear-attention + 16
full-attention layers; the code also supports the 35B-A3B MoE) and the ternary "Bonsai 2 27B"
model, through three apps: `ninfer` (CLI), `ninfer-serve` (OpenAI + Anthropic HTTP),
`ninfer-perplexity`.

`AGENTS.md` is the project rules file, written for the upstream RTX 5090 (`sm_120a`) product.
Where it disagrees with this branch (GPU target, artifact names, Python path), `README.md` and
`WINDOWS_PORT.md` describe the current 4090 branch.

## Build

```bash
# Linux
cmake --preset dev      # Release + tests + benchmarks into build/
# or: cmake --preset release   # apps only
cmake --build build -j
```

```bat
:: Windows (primary dev platform; see WINDOWS_PORT.md "Building on Windows")
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake ^
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
```

- Outputs: `build/apps/{ninfer,ninfer-serve,ninfer-perplexity}`, `build/tests/ninfer_*_test`,
  `build/bench/ninfer_bench` plus per-Op `build/bench/ninfer_*_bench`.
- vcpkg deps (curl, ffmpeg, pkgconf) install automatically from `vcpkg.json` on first configure;
  on Windows, put `build/vcpkg_installed/x64-windows/bin` and the CUDA `bin` dir on `PATH` at runtime.
- Tests/benchmarks are OFF by default; the `dev` preset enables them (`BUILD_TESTING`,
  `NINFER_BUILD_BENCHMARKS`). If an existing `build/` was configured without them, reconfigure with
  `cmake --preset dev` (check `build/CMakeCache.txt` for what is active).

## Tests

```bash
ctest --test-dir build --output-on-failure                              # full suite
ctest --test-dir build -R ninfer_sampling_test --output-on-failure      # one target
build/tests/ninfer_softmax_attention_test --dflash2-only                # some binaries take custom flags
NINFER_OP_REPORT_STATS=1 ctest --test-dir build -V -R '^ninfer_softmax_attention_test$'  # per-case float error stats
```

- Every CUDA Op test checks against an independent FP32/FP64 oracle (exact oracles for exact
  transforms/codecs). The qualification contract is `docs/maintainer/op-development.md`; do not
  relax a test criterion to make a failure pass — find the cause.
- Real-artifact Engine tests are opt-in and are skipped in CTest otherwise:
  `NINFER_TEST_ARTIFACT=$PWD/out/<model>.ninfer ctest --test-dir build -R ninfer_qwen3_5_prefix_real_test`.
  Run GPU integration tests serially. `NINFER_PREFIX_REAL_SCENARIO` focuses prefix scenarios.
- Python suites (conversion, artifact framing, eval, spec-sim — no GPU inference):
  `python3 -m pytest tests/artifact tests/convert tests/test_serve_corpus.py tests/test_eval.py tests/test_spec_sim.py`.
  The maintained environment is Python 3.11 (conversion also runs on 3.12); `Python3_EXECUTABLE`
  selects the interpreter for CMake interop tests.

## Benchmarks and profiling

- Product-level, through the public Engine route:
  `build/bench/ninfer_bench --weights <m.ninfer> -p 512,2048 -n 128 -r 3`
  (`pp{P}` = prefill, `tg{G}` = decode; default matrix `pp512` + `tg128`; `-o json --output-file …`).
- Op microbenchmarks: `build/bench/ninfer_<op>_bench` (e.g.
  `ninfer_linear_bench --qtype q4 --policy a16 --n 4096 --k 5120 --t 8`); `--profile` brackets one
  public call for `ncu --profile-from-start off`.
- `--profile-measured` on `ninfer_bench` pairs with an nsys `cudaProfilerApi` capture range.
- Raw reports/captures belong in `profiles/{bench,ncu,nsys}/`.
- A/B performance work: alternate old and new binaries within one session (thermal + display noise
  is ~±1 ms per decode round; report "within the noise" when it is). An Op microbenchmark
  establishes an Op result, not an end-to-end improvement.

## Python tools

All run with the maintained Python env; there is no Python model-inference path — the C++ Engine is
the only execution route.

- `python -m tools.convert --model <dir> --recipe qwen3_8_27b …` — build `.ninfer` artifacts from
  checkpoints (recipes incl. `qwen3_8_27b`, `qwen3_8_27b_a8` int8-prefill, `bonsai2_27b_mtp_*`;
  full guide: `docs/weight-conversion.md`).
- `python -m tools.eval run --base-url http://127.0.0.1:8080/v1 --label <name> --out <f>.json` —
  45-task deterministic quality eval against a running server (`python -m tools.eval compare …`).
- `python -m tools.spec_sim <transcripts.jsonl> --artifact <m.ninfer>` — n-gram speculation
  simulator (no GPU).
- `python tools/bench/run_serve_concurrency.py --serve build/apps/ninfer-serve …` — concurrent-lane
  decode measurements.
- `python3 tools/smoke/serve_contract.py --base-url …`, `python3 tools/smoke/serve_thinking_preservation.py
  --artifact … --backend mtp` — manual serving checks; intentionally not CTest (need a live server
  and a real artifact).
- `tools/artifact/inspect.py` — inspect `.ninfer` contents.

## Architecture (big picture)

Four layers with unique ownership; the authority is `docs/maintainer/engine-architecture.md`:

1. **Gateway** — `src/product` (CLI) and `src/serve` (HTTP: OpenAI Chat/Responses + Anthropic
   Messages, `/monitor`, `/metrics`, `/slots`, slot save/restore).
2. **Frontend** — `src/models/qwen3_5/frontend`, `src/text` (Jinja), `src/media`: tokenizer, chat
   template, vision preprocessing, detokenization.
3. **Engine** — `src/runtime/engine`: FIFO queue, admission (1–8 lanes), Scheduler (who runs when),
   ResourceManager (which logical contexts to keep), context cache/checkpoints. A single Engine
   worker thread is the only mutation owner; each model unit is one atomic `PendingBatch` commit
   (KV, recurrent state, RNG, speculative state all commit or roll back together).
4. **Program** — `src/models/qwen3_5/program`: state/KV stores, allocators, prefill/decode/
   speculative execution, CUDA Graphs. Every decode round replays as one graph per exact batch
   size `B` (compact batch, no padding); request identity and KV page IDs are graph input data,
   not graph keys.

Supporting: `src/core` (device primitives, paged KV, transfers, graph RAII — never captures),
`src/artifact` (`.ninfer` v3 framing/materialization — the only artifact format; no extension
sniffing, no C++ SDK), `src/ops` (closed semantic Ops; production dispatch selects the kernel).

Model code has **no generic graph**: each layer calls explicit kernels selected for its weight
format (Q4/Q5 groupwise, ternary `t5`, BF16/FP8/NVFP4). Speculative decoding is a Program-internal
backend: MTP (K=1..5), DFlash/DFlash2 (K≤15), n-gram chain (host draft pool, `--ngram chain`);
GDN linear-attention state rolls back rejected tokens through ReplaySSM records. KV cache is paged
with selectable dtype (`bf16`, `int8`, `rk4v4-e8`, `rk2v4-e8`, …); request prefixes are reused from
device/host checkpoints.

Public C++ API: opaque `ninfer::Engine` in `include/ninfer/engine.h` (`prepare` /
`prepare_tokens`, `submit` / `generate`, `score_tokens`, slot save/restore); host value types in
`include/ninfer/types.h` (`EngineOptions`, `RequestOptions`, `kMaximumConcurrency = 8`).

## RTX 4090 / `sm_89` gotchas

- 128 SMs (upstream code was tuned for the 5090's 170) — check grid sizes and wave counts.
- Measured read ceiling is ~845 GB/s, not the datasheet 1008.
- 48 KiB static shared memory per kernel: with `-rdc=true`, an overflow is detected only at device
  link (`nvlink … uses too much shared data`) — compute the total per template instantiation.
- `bar.sync` with a register barrier ID reserves all 16 hardware barriers and drops occupancy to
  1 CTA/SM; use immediate IDs.
- If the 4090 also drives the display, the Windows compositor steals ~15–18 % of decode; compare
  tok/s only between runs with the same display setup.
- NVFP4/W4A4 execution needs Blackwell tensor cores and is unavailable on `sm_89`.

## Documentation

| Topic | Where |
|---|---|
| Rules, ownership boundaries, verification policy | `AGENTS.md` (upstream 5090 wording) |
| Product, exact commands, performance tables | `README.md`, `docs/cli.md`, `docs/serving.md` |
| Windows build + Qwen3.8 measurements/VRAM budget | `WINDOWS_PORT.md` |
| Engine architecture | `docs/maintainer/engine-architecture.md` |
| Bonsai ternary design/conversion; live state + measurements in §9.1 | `docs/maintainer/bonsai-ternary-design.md`, `docs/maintainer/bonsai-ternary-conversion.md` |
| Op development and qualification | `docs/maintainer/op-development.md` |
| Paged KV, scheduling, context cache | `docs/maintainer/paged-kv-cache.md`, `docs/maintainer/resource-scheduling-and-context-cache.md` |
| Test / benchmark commands | `tests/README.md`, `bench/README.md` |
| Weight conversion and recipes | `docs/weight-conversion.md` |
| Full doc map | `docs/README.md` |

Record measured results where the project keeps them: Bonsai figures in
`docs/maintainer/bonsai-ternary-design.md` §9.1, Qwen3.8 figures in `WINDOWS_PORT.md`.
`HANDOFF.md` (Spanish) is the temporary work plan for the active 4090 performance work — treat it
as current context while it exists; it is deleted when its items close, not a permanent authority.
