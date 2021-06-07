//===- FIRAnnotations.h - .fir file annotation interface --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains private APIs the parser uses to load annotations.
//
//===----------------------------------------------------------------------===//

#ifndef FIRANNOTATIONS_H
#define FIRANNOTATIONS_H

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
              MLIRContext *context);

bool scatterCustomAnnotations(llvm::StringMap<ArrayAttr> &annotationMao,
                              MLIRContext *context, unsigned &annotationID,
                              Location loc);

} // namespace firrtl
} // namespace circt

#endif // FIRANNOTATIONS_H
