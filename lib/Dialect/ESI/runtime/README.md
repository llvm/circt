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

## Cosim HostMem width

The cosim BSP accepts a `hostmem_width` parameter in bits, shared by reads and
writes:

```python
CosimBSP(MyDesign, hostmem_width=512)
```

The default is 64 bits. Supported widths are powers of two, at least 8 bits.
`CosimBSP_DMA` accepts the same parameter. The host runtime discovers the width
from the design's channel types; no corresponding software setting is needed.

Wider beats reduce transport messages for wide writes: a 4096-byte write takes
512 beats at 64 bits, but only 64 beats at 512 bits. Each beat is still
acknowledged, and the client receives completion after all its beats complete.
Reads similarly return fewer, wider response beats. Short reads access only the
requested bytes; unused bytes in the final response beat are zero-filled.
Separate narrow client requests are not combined. The cosim read-request and
write-payload caps remain 256 bytes unless one configured beat is larger, in
which case the caps grow to accommodate that beat.
