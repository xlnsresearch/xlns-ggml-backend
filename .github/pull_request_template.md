## What this changes

<!-- What the change does and why. Link the issue if there is one. -->

## How it was tested

Backends:
- [ ] LNS16
- [ ] LNS32
- [ ] LNSCUDA16
- [ ] LNSCUDA32
- [ ] Not applicable (docs, experiments or tooling only)

Summation modes: <!-- base / monte / lpvip / sum32, or n/a -->

Hardware: <!-- CPU, and GPU + CUDA version for GPU changes -->

- [ ] `./setup.sh --unit-test --skip-inference` passes for the modes above
- [ ] GPU parity tests pass (`./setup.sh --cuda --unit-test --skip-inference` with `LNS_REQUIRE_GPU=1`)
- [ ] For changes to the numbers: before/after results are below or linked

## Upstream pins and patches

- [ ] No change to `patches/`
- [ ] Pins moved with `./setup.sh --update`, and the regenerated patches are included
- [ ] Hooks changed, and `patches/*.patch` regenerated at the current pins

## Docs

- [ ] README / SETUP.md / docs updated, or nothing user-facing changed
