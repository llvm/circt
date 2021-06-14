//===-- circt-c/SVDialect.h - C API for SV dialect ----------------*- C -*-===//
//
// This header declares the C interface for registering and accessing the
// SV dialect. A dialect should be registered with a context to make it
// available to users of the context. These users must load the dialect
// before using any of its attributes, operations or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_DIALECT_SV_H
#define CIRCT_C_DIALECT_SV_H

#include "mlir-c/Registration.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(SystemVerilog, sv);
MLIR_CAPI_EXPORTED void registerSVPasses();

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_SV_H
