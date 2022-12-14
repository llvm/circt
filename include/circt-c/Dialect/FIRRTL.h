//===-- circt-c/Dialect/FIRRTL.h - C API for FIRRTL dialect -------*- C -*-===//
//
// This header declares the C interface for registering and accessing the
// FIRRTL dialect. A dialect should be registered with a context to make it
// available to users of the context. These users must load the dialect
// before using any of its attributes, operations or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_DIALECT_FIRRTL_H
#define CIRCT_C_DIALECT_FIRRTL_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(FIRRTL, firrtl);

struct FirrtlContext {
  void *ptr;
};
// NOLINTNEXTLINE(modernize-use-using)
typedef struct FirrtlContext FirrtlContext;

MLIR_CAPI_EXPORTED FirrtlContext firrtlCreateContext(void);

MLIR_CAPI_EXPORTED void firrtlDestroyContext(FirrtlContext ctx);

// NOLINTNEXTLINE(modernize-use-using)
typedef void (*FirrtlErrorHandler)(MlirStringRef message, void *userData);
MLIR_CAPI_EXPORTED void firrtlSetErrorHandler(FirrtlContext ctx,
                                              FirrtlErrorHandler handler,
                                              void *userData);

MLIR_CAPI_EXPORTED void firrtlVisitCircuit(FirrtlContext ctx,
                                           MlirStringRef name);

MLIR_CAPI_EXPORTED void firrtlVisitModule(FirrtlContext ctx,
                                          MlirStringRef name);

MLIR_CAPI_EXPORTED MlirStringRef firrtlExportFirrtl(FirrtlContext ctx);

MLIR_CAPI_EXPORTED void firrtlDestroyString(FirrtlContext ctx,
                                            MlirStringRef string);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_FIRRTL_H
