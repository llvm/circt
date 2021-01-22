//===-- circt-c/SVDialect.h - C API for SV dialect ----------------*- C -*-===//
//
// This header declares the C interface for registering and accessing the
// SV dialect. A dialect should be registered with a context to make it
// available to users of the context. These users must load the dialect
// before using any of its attributes, operations or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_SVDIALECT_H
#define CIRCT_C_SVDIALECT_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Registers the SV dialect with the given context. This allows the dialect to
/// be loaded dynamically if needed when parsing.
void mlirContextRegisterSVDialect(MlirContext context);

/// Loads the SV dialect into the given context. The dialect does _not_ have to
/// be registered in advance.
MlirDialect mlirContextLoadSVDialect(MlirContext context);

/// Returns the namespace of the SV dialect, suitable for loading it.
MlirStringRef mlirSVDialectGetNamespace();

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_SVDIALECT_H
