# ESI Accelerator Runtime

The ESI Accelerator Runtime (`esiaccel`) provides software APIs and tools for
interacting with accelerators built using CIRCT's Elastic Silicon Interconnect
(ESI) dialect. It includes Python bindings and C++ libraries for discovering an
accelerator's hierarchy, communicating over typed channels, and accessing
services such as MMIO and host memory.

## Installation

The ESI Accelerator Runtime requires Python 3.8 or newer.

```sh
pip install esiaccel
```

## Getting started

See the [ESI documentation](https://circt.llvm.org/docs/Dialects/ESI/) for an
overview of ESI. The package includes support for connecting to accelerators
through RTL cosimulation and Xilinx Runtime (XRT) backends.

For command-line accelerator discovery and inspection, run `esiquery --help`.
The ESI Accelerator Runtime is developed as part of the
[CIRCT repository](https://github.com/llvm/circt/tree/main/lib/Dialect/ESI/runtime).
