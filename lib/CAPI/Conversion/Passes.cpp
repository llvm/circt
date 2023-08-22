//===- Conversion.cpp - C API for Conversion Passes -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/Passes.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/Pass/Pass.h"

// Must include the declarations as they carry important visibility attributes.
#include "circt/Conversion/Conversion.capi.h.inc"

using namespace circt;

#ifdef __cplusplus
extern "C" {
#endif

#include "circt/Conversion/Conversion.capi.cpp.inc"

#ifdef __cplusplus
}
#endif
