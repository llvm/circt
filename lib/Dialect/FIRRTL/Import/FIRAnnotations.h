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

namespace llvm {
namespace json {
class Path;
class Value;
} // namespace json
} // namespace llvm

namespace circt {
namespace firrtl {

bool fromJSONRaw(llvm::json::Value &value, StringRef circuitTarget,
              SmallVectorImpl<Attribute> &attrs, llvm::json::Path path,
              MLIRContext *context);

} // namespace firrtl
} // namespace circt

#endif // FIRANNOTATIONS_H
