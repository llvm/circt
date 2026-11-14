//===- DebugOps.h - Debug dialect operations ====----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_DEBUG_DEBUGOPS_H
#define CIRCT_DIALECT_DEBUG_DEBUGOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "circt/Dialect/Debug/DebugDialect.h"
#include "circt/Dialect/Debug/DebugTypes.h"
#include "llvm/ADT/StringExtras.h"

// Enum definitions generated from `DebugOps.td`
#include "circt/Dialect/Debug/DebugEnums.h.inc"

// Operation definitions generated from `Debug.td`
#define GET_OP_CLASSES
#include "circt/Dialect/Debug/Debug.h.inc"

namespace circt::debug {

/// Whether `lang` can key an `ext` dictionary: a lowercase language name.
inline bool isExtLanguage(llvm::StringRef lang) {
  return !lang.empty() && llvm::isLower(lang.front()) &&
         llvm::all_of(
             lang, [](char c) { return llvm::isLower(c) || llvm::isDigit(c); });
}

// Names of the discardable attributes declared in `DebugDialect.td`.
inline constexpr llvm::StringLiteral kDbgMergedFromAttr =
    DebugDialect::MergedFromAttrHelper::getNameStr();
inline constexpr llvm::StringLiteral kDbgModuleLocsAttr =
    DebugDialect::ModuleLocsAttrHelper::getNameStr();
inline constexpr llvm::StringLiteral kDbgLoweredAttr =
    DebugDialect::LoweredAttrHelper::getNameStr();
inline constexpr llvm::StringLiteral kDbgArtificialAttr =
    DebugDialect::ArtificialAttrHelper::getNameStr();
inline constexpr llvm::StringLiteral kDbgMemoryAttr =
    DebugDialect::MemoryAttrHelper::getNameStr();
inline constexpr llvm::StringLiteral kDbgModuleInfoAttr =
    DebugDialect::ModuleinfoAttrHelper::getNameStr();
inline constexpr llvm::StringLiteral kDbgInstanceAttr =
    DebugDialect::InstanceAttrHelper::getNameStr();

} // namespace circt::debug

#endif // CIRCT_DIALECT_DEBUG_DEBUGOPS_H
