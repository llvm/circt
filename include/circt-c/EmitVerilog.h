//===-- circt-c/SVDialect.h - C API for emitting Verilog ----------*- C -*-===//
//
// This header declares the C interface for emitting Verilog from a CIRCT MLIR
// module
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_EMITVERILOG_H
#define CIRCT_C_EMITVERILOG_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Emits verilog for the specified module using the provided callback and user
/// data
MlirLogicalResult mlirEmitVerilog(MlirModule, MlirStringCallback,
                                  void *userData);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_EMITVERILOG_H
