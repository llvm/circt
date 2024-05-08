//===- FIRAnnotations.h - .fir file annotation interface --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains public APIs for loading annotations.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_IMPORT_FIRANNOTATIONS_H
#define CIRCT_DIALECT_FIRRTL_IMPORT_FIRANNOTATIONS_H

#include "circt/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {
namespace json {
class Path;
class Value;
} // namespace json
} // namespace llvm

namespace circt {
namespace firrtl {

/// Deserialize a JSON value into FIRRTL Annotations.  Annotations are
/// represented as a Target-keyed arrays of attributes.  The input JSON value is
/// checked, at runtime, to be an array of objects.  Returns true if successful,
/// false if unsuccessful.
bool importAnnotationsFromJSONRaw(llvm::json::Value &value,
                                  SmallVectorImpl<Attribute> &annotations,
                                  llvm::json::Path path, MLIRContext *context);

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_IMPORT_FIRANNOTATIONS_H
