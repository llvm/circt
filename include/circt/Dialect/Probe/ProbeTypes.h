//===- ProbeTypes.h - Probe dialect types -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_PROBE_PROBETYPES_H
#define CIRCT_DIALECT_PROBE_PROBETYPES_H

#include "circt/Dialect/Seq/SeqTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/Probe/ProbeTypes.h.inc"

namespace circt {
namespace probe {

/// Return true if this type is a valid probe payload. Probe payloads may be HW
/// aggregate types whose leaves are HW value types or seq.clock values, but
/// must not contain any inout types.
bool isProbeElementType(mlir::Type type);

} // namespace probe
} // namespace circt

#endif // CIRCT_DIALECT_PROBE_PROBETYPES_H
