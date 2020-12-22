//===-- circt-c/RTLDialect.h - C API for RTL dialect --------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface for registering and accessing the
// RTL dialect. A dialect should be registered with a context to make it
// available to users of the context. These users must load the dialect
// before using any of its attributes, operations or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_RTLDIALECT_H
#define CIRCT_C_RTLDIALECT_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Registers the RTL dialect with the given context. This allows the dialect to
/// be loaded dynamically if needed when parsing.
void mlirContextRegisterRTLDialect(MlirContext context);

/// Loads the RTL dialect into the given context. The dialect does _not_ have to
/// be registered in advance.
MlirDialect mlirContextLoadRTLDialect(MlirContext context);

/// Returns the namespace of the RTL dialect, suitable for loading it.
MlirStringRef mlirRTLDialectGetNamespace();

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_RTLDIALECT_H
