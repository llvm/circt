# Plan: Add `esiaccel.cosim` pytest integration library

## Goals
- Move cosim lifecycle boilerplate out of tests and into a dedicated library under `esiaccel.cosim`.
- Provide function/class decorators to run PyCDE hardware-generation scripts and launch cosim automatically.
- Support argument injection into test callables:
  - host/port (from `cosim.cfg` / `SimProcess.port`)
  - `Accelerator` object (auto-connect to cosim when requested by signature)
- Enforce process isolation: every decorated test invocation runs in a separate Python process.
- Enable class-level compile reuse: compile once per class, run each method in a fresh simulator run without recompiling.

## API shape (MVP)
- New module: `lib/Dialect/ESI/runtime/python/esiaccel/cosim/pytest.py`
- Decorator:
  - `@cosim_test(pycde_script, args=("{tmp_dir}",), simulator="verilator", ...)`
  - Also usable on classes: every test method gets its own simulation run.
- Config object:
  - script path
  - script args template
  - simulator kind
  - timeout
  - failure/warning matchers
  - debug / top-module overrides

## Execution model
1. Create temporary run workspace and `chdir` into it.
2. Run PyCDE script to emit hardware.
3. Build `SourceFiles` and `Simulator` using existing classes in `simulator.py`.
4. Start simulator with `run_proc` and wait for RPC readiness.
5. Invoke the test function.
   - If function accepts host/port-like parameters, inject them.
   - If function accepts `Accelerator` annotation (or `accelerator` argument name), connect via cosim and inject.
6. Tear down simulator robustly and preserve logs.

## Class decoration behavior
- Compile hardware once per class in a compile cache directory.
- For each test method:
  - create unique temporary run workspace and run directory
  - start simulator without recompiling (reuse compiled artifacts)
  - run the method in a child process
- Use a per-class lock around cache initialization to avoid races under xdist.

## Process isolation
- Each decorated test call runs in `multiprocessing.Process`.
- Parent process receives structured result via queue/pipe:
  - pass/fail
  - traceback
  - log summary
- Optional timeout (`timeout_s`) terminates the child process and fails the test.
- Parent re-raises failures as assertion errors with useful diagnostics.

## Log scanning hooks (updated)
- Support two matcher forms for both failure and warning detection:
  - callback functions (line/stream aware)
  - arrays of regex patterns
- Defaults:
  - failure regex detects `error` case-insensitively
  - excludes summary lines like `<number> errors` (to avoid false positives)
- Warning reporting:
  - warning matches are emitted through pytest logging (`logging` module), e.g. logger `esiaccel.cosim.pytest`.
- Scan both stdout and stderr with stream tagging.

## Open questions with proposed answers
### 1) Detecting simulation failure beyond exit code
**Proposal**: layered checks:
- startup invariants (cfg appears, port opens, process alive)
- process liveness during test
- log scan hooks (regex + callback)
- optional user-provided health callback for custom checks

**Why**: catches silent simulator failures and backend-specific errors.

### 2) C++ compilation strategy
**Proposal**:
- Keep CMake as primary and supported path.
- Add thin helper utilities for CMake configure/build invocation and diagnostics.
- Defer auto-generated one-file CMake support to follow-up (optional convenience layer).

**Why**: avoids over-design and preserves existing tested build path.

### 3) How much pytest infra to use
**Proposal**:
- Use standard pytest-compatible decorators + logging + fixtures where helpful.
- Avoid deep custom collection hooks initially.
- Keep internals modular so behavior can be changed if decorator ergonomics get ugly.

**Why**: lower complexity and easier iteration.

## Implementation phases
1. Add pytest module scaffolding + config dataclasses.
2. Add function decorator path with process isolation + signature injection.
3. Add log scanners (callback + regex) and pytest warning logging.
4. Add class decorator path with compile reuse + unique run dirs.
5. Incrementally migrate `test_cosim.py` to use decorators.
6. Validate with normal pytest and xdist.
