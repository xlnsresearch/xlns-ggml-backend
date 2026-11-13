# Experiments

Measurements made with the LNS backends, kept with the tools and commands
that produced them so they can be rerun and extended.

| Folder | What it answers |
|---|---|
| [layer-by-layer/](layer-by-layer/) | Where does LNS error enter and how does it grow through an LLM, per backend and summation mode? Does the GPU compute exactly what the CPU computes, at every layer? |

## Configurations

Every experiment compares the same set of configurations. Each xlns16
summation mode is a separate build (`build/llama.cpp/build-lns-<mode>`, from
`./setup.sh --cuda`); `GGML_LNS_BACKEND` picks the device inside it.

| Label | Build | `GGML_LNS_BACKEND` | Arithmetic |
|---|---|---|---|
| `f32` | `build-ref` | (none) | plain F32 on the CPU: the reference |
| `cpu16-<mode>` | `build-lns-<mode>` | `16` | xlns16 on the CPU, summation `<mode>` |
| `cuda16-<mode>` | `build-lns-<mode>` | `cuda16` | xlns16 on the GPU, summation `<mode>` |
| `cpu32` | `build-lns-base` | `32` | xlns32 on the CPU |
| `cuda32` | `build-lns-base` | `cuda32` | xlns32 on the GPU |

`<mode>` is `base`, `monte`, `lpvip` or `sum32`.

## Library versions

The CUDA monte/lpvip modes need xlnscuda#11 and xlnscpp#64. Until those are
merged and the submodule pins moved, the results here were produced with the
submodules temporarily holding the **post-merge contents** (upstream `main`
plus each PR), so they describe the state the repo will have after the merge:

- xlnscpp: `main` 4f5e75f + PR #64 (`xlns16.cpp` fixes, and the `xlns32lpvip.cpp`
  fixes: signed `xlns32_mitch` shift, missing parentheses)
- xlnscuda: `main` f0834bb merged with PR #11 (cf16dd4)
