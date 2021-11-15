//===-- LLHD.h - C API for LLHD dialect -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/LLHD.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/LLHD/IR/LLHDTypes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

using namespace circt;
using namespace circt::llhd;

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(LLHD, llhd, circt::llhd::LLHDDialect)

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

/// Check if a type is a time type.
bool llhdTypeIsATimeType(MlirType type) { return unwrap(type).isa<TimeType>(); }

/// Check if a type is a signal type.
bool llhdTypeIsASignalType(MlirType type) {
  return unwrap(type).isa<SigType>();
}

/// Check if a type is a pointer type.
bool llhdTypeIsAPointerType(MlirType type) {
  return unwrap(type).isa<PtrType>();
}

/// Create a time type.
MlirType llhdTimeTypeGet(MlirContext ctx) {
  return wrap(TimeType::get(unwrap(ctx)));
}

/// Create a signal type.
MlirType llhdSignalTypeGet(MlirType element) {
  return wrap(SigType::get(unwrap(element)));
}

/// Create a pointer type.
MlirType llhdPointerTypeGet(MlirType element) {
  return wrap(PtrType::get(unwrap(element)));
}

/// Get the inner type of a signal.
MlirType llhdSignalTypeGetElementType(MlirType type) {
  return wrap(unwrap(type).cast<SigType>().getUnderlyingType());
}

/// Get the inner type of a pointer.
MlirType llhdPointerTypeGetElementType(MlirType type) {
  return wrap(unwrap(type).cast<PtrType>().getUnderlyingType());
}
