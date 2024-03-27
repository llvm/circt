# Emission (Emit) Dialect Rationale

This document describes various design points of the `emit` dialect, why it is
the way it is, and current status. This follows in the spirit of other [MLIR
Rationale docs](https://mlir.llvm.org/docs/Rationale/).

## Introduction

The `emit` dialects controls the structure and formatting of the files emitted
from CIRCT. It captures information about both SystemVerilog output files and
generic collateral files. The ops are translated to output files in
`ExportVerilog`. Presently, the dialect is intertwined with SystemVerilog -
it can reference items in a design through symbols to emit references to
them through SystemVerilog names in the output.

## Operations

The dialect is centred around the `emit.file` operation which groups a list
of statements in its body, responsible for producing the contents of the file.
The `emit.file_list` operation pins down a list of references to emitted files
and outputs a file list file enumerating the paths to them.
Together, these operations represent the SV and collateral output of CIRCT.
