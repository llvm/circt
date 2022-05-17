//===- FIRRTLAnnotationHelper.h - FIRRTL Annotation Lookup ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares helpers mapping annotations to operations.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLANNOTATIONHELPER_H
#define CIRCT_DIALECT_FIRRTL_FIRRTLANNOTATIONHELPER_H

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Namespace.h"

namespace circt {
namespace firrtl {

/// Stores an index into an aggregate.
struct TargetToken {
  StringRef name;
  bool isIndex;
};

/// The parsed annotation path.
struct TokenAnnoTarget {
  StringRef circuit;
  SmallVector<std::pair<StringRef, StringRef>> instances;
  StringRef module;
  // The final name of the target
  StringRef name;
  // Any aggregates indexed.
  SmallVector<TargetToken> component;
};

// The potentially non-local resolved annotation.
struct AnnoPathValue {
  SmallVector<InstanceOp> instances;
  AnnoTarget ref;
  unsigned fieldIdx = 0;

  AnnoPathValue() = default;
  AnnoPathValue(CircuitOp op) : ref(OpAnnoTarget(op)) {}
  AnnoPathValue(Operation *op) : ref(OpAnnoTarget(op)) {}
  AnnoPathValue(const SmallVectorImpl<InstanceOp> &insts, AnnoTarget b,
                unsigned fieldIdx)
      : instances(insts.begin(), insts.end()), ref(b), fieldIdx(fieldIdx) {}

  bool isLocal() const { return instances.empty(); }

  template <typename... T>
  bool isOpOfType() const {
    if (auto opRef = ref.dyn_cast<OpAnnoTarget>())
      return isa<T...>(opRef.getOp());
    return false;
  }
};

/// Return an input \p target string in canonical form.  This converts a Legacy
/// Annotation (e.g., A.B.C) into a modern annotation (e.g., ~A|B>C).  Trailing
/// subfield/subindex references are preserved.
std::string canonicalizeTarget(StringRef target);

/// Parse a FIRRTL annotation path into its constituent parts.
Optional<TokenAnnoTarget> tokenizePath(StringRef origTarget);

/// Convert a parsed target string to a resolved target structure.  This
/// resolves all names and aggregates from a parsed target.
Optional<AnnoPathValue> resolveEntities(TokenAnnoTarget path, CircuitOp circuit,
                                        SymbolTable &symTbl);

/// Resolve a string path to a named item inside a circuit.
Optional<AnnoPathValue> resolvePath(StringRef rawPath, CircuitOp circuit,
                                    SymbolTable &symTbl);

/// Return true if an Annotation's class name is handled by the LowerAnnotations
/// pass.
bool isAnnoClassLowered(StringRef className);

/// State threaded through functions for resolving and applying annotations.
struct ApplyState {
  using AddToWorklistFn = llvm::function_ref<void(DictionaryAttr)>;
  ApplyState(CircuitOp circuit, SymbolTable &symTbl,
             AddToWorklistFn addToWorklistFn)
      : circuit(circuit), symTbl(symTbl), addToWorklistFn(addToWorklistFn) {}

  CircuitOp circuit;
  SymbolTable &symTbl;
  AddToWorklistFn addToWorklistFn;

  ModuleNamespace &getNamespace(FModuleLike module) {
    auto &ptr = namespaces[module];
    if (!ptr)
      ptr = std::make_unique<ModuleNamespace>(module);
    return *ptr;
  }

private:
  DenseMap<Operation *, std::unique_ptr<ModuleNamespace>> namespaces;
};

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLANNOTATIONHELPER_H
