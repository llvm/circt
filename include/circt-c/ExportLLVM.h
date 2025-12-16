//===-- circt-c/ExportLLVM.h - C API for exporting LLVM IR --------*- C -*-===//
//
// This header declares the C interface for exporting LLVM IR from a CIRCT MLIR
// module.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_EXPORTLLVM_H
#define CIRCT_C_EXPORTLLVM_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Exports LLVM IR for the specified module using the provided callback and
/// user data
MLIR_CAPI_EXPORTED MlirLogicalResult mlirExportLLVMIR(MlirModule,
                                                      MlirStringCallback,
                                                      void *userData);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_EXPORTLLVM_H
