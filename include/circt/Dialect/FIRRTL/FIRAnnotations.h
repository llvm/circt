//===- FIRAnnotations.h - .fir to FIRRTL dialect parser -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provide utilities related to dealing with Scala FIRRTL Compiler Annotations.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRANNOTATIONS_H
#define CIRCT_DIALECT_FIRRTL_FIRANNOTATIONS_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/JSON.h"

namespace circt {
namespace firrtl {

bool fromJSON(llvm::json::Value &value,
              llvm::StringMap<ArrayAttr> &annotationMap, llvm::json::Path path,
              MLIRContext *context);

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_FIRANNOTATIONS_H
