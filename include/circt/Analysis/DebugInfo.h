//===- DebugInfo.h - Debug info analysis ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_ANALYSIS_DEBUGINFO_H
#define CIRCT_ANALYSIS_DEBUGINFO_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"

namespace circt {

struct DIInstance;
struct DIVariable;

namespace detail {
struct DebugInfoBuilder;
} // namespace detail

/// Per-module integer identifier for a `dbg.enumdef` op
using DIEnumDefId = uint32_t;

/// Map from an enum's raw integer value to its variant name
using DIEnumValMap = llvm::MapVector<int64_t, StringAttr>;

/// Module-scoped table of all enum definitions: id -> variant map
using DIEnumDefMap = llvm::MapVector<DIEnumDefId, DIEnumValMap>;

/// Source-language type information lifted off
/// `dbg.moduleinfo` and `dbg.variable` ops
struct DISourceLang {
  // Fields are null when absent; check before use.
  StringAttr typeName;
  ArrayAttr params;
};

struct DIModule {
  /// The operation that generated this level of hierarchy.
  Operation *op = nullptr;
  /// The name of this level of hierarchy.
  StringAttr name;
  /// Levels of hierarchy nested under this module.
  SmallVector<DIInstance *, 0> instances;
  /// Variables declared within this module.
  SmallVector<DIVariable *, 0> variables;
  /// If this is an extern declaration.
  bool isExtern = false;
  /// If this is an inline scope created by a `dbg.scope` operation.
  bool isInline = false;

  /// Source-language type, from `dbg.moduleinfo`.
  DISourceLang sourceLangType;

  /// Enum defs declared inside this module (not inherited from parent scopes)
  DIEnumDefMap enumDefinitions;

  /// Maps each `dbg.enumdef` op to its id in `enumDefinitions`. Duplicates
  /// (same enumTypeName + fqn + variants) share the first-seen id.
  llvm::DenseMap<mlir::Operation *, DIEnumDefId> enumDefIds;
};

struct DIInstance {
  /// The operation that generated this instance.
  Operation *op = nullptr;
  /// The name of this instance.
  StringAttr name;
  /// The instantiated module.
  DIModule *module;
};

struct DIVariable {
  /// The name of this variable.
  StringAttr name;
  /// The location of the variable's declaration.
  LocationAttr loc;
  /// The SSA value representing the value of this variable.
  Value value = nullptr;
  /// Enum definition for scalar-typed variables; null for aggregate
  mlir::Value enumDef = nullptr;

  /// Source-language type, from `dbg.variable` attributes.
  DISourceLang sourceLangType;

  /// Per-module integer id for `enumDef`. nullopt when `enumDef` is null, when
  /// its defining op is not a `dbg.enumdef`, or when that op was not registered
  /// in the enclosing module's `enumDefIds` (e.g. it lives in another module).
  std::optional<DIEnumDefId> enumDefRef = std::nullopt;
};

/// Debug information attached to an operation and the operations nested within.
///
/// This is an analysis that gathers debug information for a piece of IR, either
/// from attributes attached to operations or the general structure of the IR.
struct DebugInfo {
  /// Collect the debug information nested under the given operation.
  DebugInfo(Operation *op);

  /// The operation that was passed to the constructor.
  Operation *operation;
  /// A mapping from module name to module debug info.
  llvm::MapVector<StringAttr, DIModule *> moduleNodes;

protected:
  friend struct detail::DebugInfoBuilder;
  llvm::SpecificBumpPtrAllocator<DIModule> moduleAllocator;
  llvm::SpecificBumpPtrAllocator<DIInstance> instanceAllocator;
  llvm::SpecificBumpPtrAllocator<DIVariable> variableAllocator;
};

} // namespace circt

#endif // CIRCT_ANALYSIS_DEBUGINFO_H
