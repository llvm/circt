//===- FIRAnnotations.h - FIRRTL Annotation Utilities ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provide utilities related to dealing with FIRRTL Annotations.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRANNOTATIONS_H
#define CIRCT_DIALECT_FIRRTL_FIRANNOTATIONS_H

#include "circt/Support/LLVM.h"
#include "llvm/ADT/StringMap.h"

namespace llvm {
namespace json {
class Path;
class Value;
} // namespace json
} // namespace llvm

namespace mlir {
class ArrayAttr;
}

namespace circt {
namespace firrtl {

bool fromJSON(llvm::json::Value &value,
              llvm::StringMap<ArrayAttr> &annotationMap, llvm::json::Path path,
              MLIRContext *context, unsigned &annotationID);

void scatterCustomAnnotations(llvm::StringMap<ArrayAttr> &annotationMao,
                              MLIRContext *context, unsigned &annotationID);

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_FIRANNOTATIONS_H
