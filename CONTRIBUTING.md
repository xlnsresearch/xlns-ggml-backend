# Contributing

Thanks for your interest in the LNS backends. This page covers how to build
the project, run the tests, and get a change merged. For the background on
what the backends do, start with the [README](README.md); for how the code is
put together, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Getting set up

```bash
git clone https://github.com/xlnsresearch/xlns-ggml-backend
cd xlns-ggml-backend
git submodule update --init
```

You need `git`, `cmake`, `g++`, `make` and `perl`. The full list of
prerequisites, including the CUDA toolkit for the GPU backends, is in
[SETUP.md](SETUP.md#prerequisites).

`setup.sh` does everything else: it clones ggml and llama.cpp at the commits
pinned in `patches/`, hooks the LNS backends into them, and builds one binary
per xlns16 summation mode. The flags you will use most:

| Command | What it does |
|---|---|
| `./setup.sh --unit-test --skip-inference` | Build and run the unit tests for all four sum modes and LNS32. No model needed. |
| `./setup.sh --unit-test --skip-inference --sum-mode=sum32` | The same for one sum mode (`base`, `monte`, `lpvip` or `sum32`). Much faster. |
| `./setup.sh --cuda ...` | Also build LNSCUDA16/LNSCUDA32 and run the GPU parity tests. |
| `./setup.sh` | Build llama.cpp and run SmolLM2 per sum mode. Needs the model, see [SETUP.md](SETUP.md#2-models-what-to-download). |
| `./run.sh --sum-mode=sum32` | Run inference again without rebuilding. |

On a machine with little RAM, limit the build with `JOBS=4 ./setup.sh ...`.

Everything under `build/` is generated. Do not edit the ggml or llama.cpp
trees there as a way to change the backends: the sources are copied in from
`lns-backend/` on every run, and `--clean` or `--update` resets those trees.

## Running the tests

| Test | How to run it |
|---|---|
| `test-lns-backend`, each backend against float references | `./setup.sh --unit-test --skip-inference` |
| `test-lns-cuda-parity`, GPU bit-identical to CPU | `./setup.sh --cuda --unit-test --skip-inference` |
| `test-lns16-sum32`, sum32 accumulator invariants (no ggml) | `g++ -O2 -std=c++17 -I xlnscpp -I lns-backend/src lns-backend/tests/test-lns16-sum32.cpp -o /tmp/test-lns16-sum32 && /tmp/test-lns16-sum32` |
| End to end against a pure-CPU llama.cpp | `./scripts/lns-vs-cpu-quality.sh --skip-ppl` |

The GPU tests are skipped when there is no usable GPU. Set
`LNS_REQUIRE_GPU=1` to make that a failure instead, which is what you want
when you are testing a CUDA change.

If your change touches the arithmetic (a kernel, a summation mode, a
conversion), run the unit tests for every sum mode it affects, and for
anything that changes numbers, include before and after results in the PR:
the unit test output, a `./run.sh` generation, or the
`lns-vs-cpu-quality.sh` report.

CI runs the CPU unit tests for all four sum modes, builds the CUDA backends,
and lints the scripts with `shellcheck --severity=warning` and `ruff check`.
You can run the same linters locally before pushing.

## Where things go

- **Backend code** lives in `lns-backend/`: `src/` (LNS16), `src32/`
  (LNS32), `cuda/` (LNSCUDA16/32), `common/` (shared framework and op
  predicates), `include/` (public headers) and `tests/`.
- **A new op or a new backend**: follow
  [Adding a backend](docs/ARCHITECTURE.md#adding-a-backend) in the
  architecture doc. A new backend is mostly a new directory plus one row in
  the manifest, `scripts/lns-backends.sh`.
- **Experiments** go in their own folder under `experiments/`, with a
  `README.md` that states the question and the commands, a `run.sh` that
  reproduces the results, and the results themselves (small text files and
  SVG figures only). Reuse `experiments/lib.sh`, `runlib.py` and `plot.py`, and
  add a row to the table in [experiments/README.md](experiments/README.md).
- **Models and build outputs** are never committed. `*.gguf` and `build/`
  are already gitignored.

## Upstream pins and patches

Nothing in ggml or llama.cpp is forked. `scripts/integrate-lns.sh` edits the
upstream trees with regexes, and `patches/*.patch` are only a fallback for
when an anchor stops matching. Each dependency is pinned in
`patches/*-base-commit.txt`.

- **Moving ggml, llama.cpp or xlnscpp to the latest upstream**: run
  `./setup.sh --update --unit-test` (with a model, so llama.cpp is built and
  run as well). When it succeeds, it rewrites the pin files and regenerates
  both patches. Commit the `patches/` diff together with any fixes it needed.
- **Changing the hooks** (`integrate-lns.sh` or the manifest): regenerate the
  fallback patches at the current pins so they match. With the model in
  place, `./setup.sh --clean --unit-test` resets both trees to their pins and
  integrates them again. Then:

  ```bash
  git -C build/llama.cpp diff -- ggml/ > patches/llama-core.patch
  git -C build/ggml diff -- CMakeLists.txt include/ggml.h src/CMakeLists.txt \
      src/ggml-backend-reg.cpp src/ggml.c > patches/ggml-core.patch
  ```

  `--clean` throws away any edits you made inside `build/`.
- **Bumping a submodule** (xlnscpp or xlnscuda): update the submodule and
  its `patches/*-base-commit.txt` in the same commit. `setup.sh` checks out
  the commit in the pin file, so the two must match.

## Commits and pull requests

- Keep each PR to one change. A bug fix, a new mode and an experiment are
  three PRs.
- Use a short subject line with a scope, the way the history does:
  `fix(cuda): build CUDA sources as C++14 for older nvcc`,
  `experiments: add dot-product error scaling`.
- In the PR description, say which backends and sum modes you tested, and on
  what hardware if it involves the GPU.
- Update the docs (README, SETUP.md, docs/) in the same PR when you change a
  flag, an environment variable or the build.

## Reporting bugs and asking questions

Use the issue templates. For a bug, the backend, sum mode, model, the exact
command and its output are what we need to reproduce it. Problems in the LNS
arithmetic itself may belong in
[xlnscpp](https://github.com/xlnsresearch/xlnscpp) or
[xlnscuda](https://github.com/xlnsresearch/xlnscuda).

Security issues should not go in a public issue; see [SECURITY.md](SECURITY.md).

By taking part in this project you agree to follow the
[Code of Conduct](CODE_OF_CONDUCT.md).
