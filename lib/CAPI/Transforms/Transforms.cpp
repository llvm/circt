//===- Transforms.cpp - C API for Transforms Passes -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Transforms.h"
#include "circt/Transforms/Passes.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/Pass/Pass.h"

using namespace circt;

#ifdef __cplusplus
extern "C" {
#endif

#include "circt/Transforms/Transforms.capi.cpp.inc"

#ifdef __cplusplus
}
#endif
