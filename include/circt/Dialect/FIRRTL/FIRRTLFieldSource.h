//===- FIRRTLFieldSource.h - Field Source (points-to) -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the FIRRTL Field Source Analysis, which is a points-to
// analysis which identifies the source field and storage of any value derived
// from a storage location.  Values derived from reads of storage locations do
// not appear in this analysis.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLFIELDSOURCE_H
#define CIRCT_DIALECT_FIRRTL_FIRRTLFIELDSOURCE_H

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Support/LLVM.h"

namespace circt {
namespace firrtl {

/// To use this class, retrieve a cached copy from the analysis manager:
///   auto &fieldsource = getAnalysis<FieldSource>(getOperation());
class FieldSource {

public:
  explicit FieldSource(Operation *) {}

  struct PathNode {
    Value src;
    size_t fieldID;
  };

  PathNode nodeForValue(Value v);

private:
  PathNode computeSrc(Value v);
  DenseMap<Value, PathNode> paths;
};

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLFIELDSOURCE_H
