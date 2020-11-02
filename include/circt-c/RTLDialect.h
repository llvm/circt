/*===-- circt-c/RTLDialect.h - C API for RTL dialect --------------*- C -*-===*\
|*                                                                            *|
*|
\*===----------------------------------------------------------------------===*/

#ifndef CIRCT_C_RTLDIALECT_H
#define CIRCT_C_RTLDIALECT_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Registers the RTL dialect with the given context. This allows the
 * dialect to be loaded dynamically if needed when parsing. */
void mlirContextRegisterRTLDialect(MlirContext context);

/** Loads the RTL dialect into the given context. The dialect does _not_
 * have to be registered in advance. */
MlirDialect mlirContextLoadRTLDialect(MlirContext context);

/** Returns the namespace of the RTL dialect, suitable for loading it. */
MlirStringRef mlirRTLDialectGetNamespace();

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_RTLDIALECT_H
