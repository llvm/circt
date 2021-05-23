//===-- circt-c/Dialect/MSFT.h - C API for MSFT dialect -----------*- C -*-===//
//
// This header declares the C interface for registering and accessing the
// MSFT dialect. A dialect should be registered with a context to make it
// available to users of the context. These users must load the dialect
// before using any of its attributes, operations or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_DIALECT_MSFT_H
#define CIRCT_C_DIALECT_MSFT_H

#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
#include "mlir-c/Registration.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(MSFT, msft);

/// Emits tcl for the specified module using the provided callback and user
/// data
MlirLogicalResult mlirMSFTExportTcl(MlirModule, MlirStringCallback,
                                    void *userData);

typedef struct {
  MlirOperation (*callback)(MlirOperation, void *userData);
  void *userData;
} mlirMSFTGeneratorCallback;

/// Register a generator callback (function pointer, user data pointer). Returns
/// the callback it replaced, if any.
void mlirMSFTRegisterGenerator(const char *opName, const char *generatorName,
                               mlirMSFTGeneratorCallback cb);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_MSFT_H
