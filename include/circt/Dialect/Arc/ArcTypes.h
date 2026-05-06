//===- ArcTypes.h - Arc dialect types ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_ARCTYPES_H
#define CIRCT_DIALECT_ARC_ARCTYPES_H

#include "circt/Dialect/HW/HWTypeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/Arc/ArcTypes.h.inc"

/// Compute the bit width a type will have when allocated as part of the
/// simulator's storage. This includes any padding and alignment that may be
/// necessary once the type has been mapped to LLVM. The idea is for this
/// function to be conservative, such that we provide sufficient storage bytes
/// for any type.
std::optional<uint64_t> computeLLVMBitWidth(mlir::Type type);

#endif // CIRCT_DIALECT_ARC_ARCTYPES_H
