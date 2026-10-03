//===- DPITypeInfo.h - SV DPI type mapping ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SV_DPITYPEINFO_H
#define CIRCT_DIALECT_SV_DPITYPEINFO_H

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/LogicalResult.h"

namespace circt::sv {

/// The effective SV type, rather than the signedness of the source MLIR type.
struct DPITypeInfo {
  enum class Kind { Bit, Logic, Byte, ShortInt, Int, LongInt, Enum };

  Kind kind;
  unsigned width = 0;
  bool isSigned = false;

  llvm::StringRef getIntegerKeyword() const;
  bool isIntegerAtom() const;
  bool isValidReturn() const;
};

/// Integer printing contexts used by the SV exporter. Typedef integers are
/// four-state vectors; packed dimensions suppress native integer atoms.
enum class DPIIntegerContext { Import, Packed, Typedef };

/// Preserve current SV printing: source integer signedness is not consulted.
DPITypeInfo getDPIIntegerTypeInfo(unsigned width, DPIIntegerContext context);
DPITypeInfo getDPIEnumTypeInfo(unsigned width);

/// Resolve a nonzero integer as it is printed in a DPI import.
llvm::FailureOr<DPITypeInfo> resolveDPIType(mlir::Type type);

} // namespace circt::sv

#endif // CIRCT_DIALECT_SV_DPITYPEINFO_H
