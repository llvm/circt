//===- InstanceChoiceInfo.h - Instance choice analysis ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the InstanceChoiceInfo analysis, which computes dominating
// instance choice options for each (public module, destination module) pair.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_INSTANCECHOICEINFO_H
#define CIRCT_DIALECT_FIRRTL_INSTANCECHOICEINFO_H

#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"

namespace circt {
namespace firrtl {

/// Represents a choice option and case.
struct ChoiceKey {
  /// The option name.
  mlir::SymbolRefAttr option;
  /// The case name (nullable for default case).
  mlir::SymbolRefAttr caseAttr;

  bool operator==(const ChoiceKey &other) const {
    return option == other.option && caseAttr == other.caseAttr;
  }
  bool operator!=(const ChoiceKey &other) const { return !(*this == other); }
};

} // namespace firrtl
} // namespace circt

namespace llvm {
template <>
struct DenseMapInfo<circt::firrtl::ChoiceKey, void> {
  using ChoiceKey = circt::firrtl::ChoiceKey;
  using SymbolRefInfo = DenseMapInfo<mlir::SymbolRefAttr>;

  static ChoiceKey getEmptyKey() {
    return {SymbolRefInfo::getEmptyKey(), SymbolRefInfo::getEmptyKey()};
  }

  static ChoiceKey getTombstoneKey() {
    return {SymbolRefInfo::getTombstoneKey(), SymbolRefInfo::getTombstoneKey()};
  }

  static unsigned getHashValue(const ChoiceKey &key) {
    return llvm::hash_combine(SymbolRefInfo::getHashValue(key.option),
                              SymbolRefInfo::getHashValue(key.caseAttr));
  }

  static bool isEqual(const ChoiceKey &lhs, const ChoiceKey &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

namespace circt {
namespace firrtl {

/// This class checks for nested instance choices with different options
/// and computes the dominating choices for each (public module, destination)
/// pair.
///
/// The restriction is: on any single path from a public module, you cannot
/// go through instance choices with different options (= nested instance is not
/// allowed). However, reaching the same module through different options on
/// different paths is allowed.
class InstanceChoiceInfo {
public:
  InstanceChoiceInfo(CircuitOp circuit, InstanceGraph &instanceGraph)
      : circuit(circuit), instanceGraph(instanceGraph) {}

  /// Run the analysis. Returns failure if there is a nested instance choice.
  LogicalResult run();

  /// Get choices for a destination module from a public module.
  /// Failure is returned if the destination is not reachable from the public
  /// module. Empty array is returned if the destination is reachable but there
  /// is no instance choice on any path.
  FailureOr<ArrayRef<ChoiceKey>> getChoices(FModuleLike publicModule,
                                            FModuleLike destination) const {
    assert(publicModule.isPublic() &&
           "the first argument must be a public module");
    auto publicIt = moduleChoices.find(publicModule);
    if (publicIt == moduleChoices.end())
      return failure();
    auto it = publicIt->second.find(destination);
    // It means the destination is not reachable from the public module.
    if (it == publicIt->second.end())
      return failure();
    return it->second.getArrayRef();
  }

  /// Check if a destination module is always reachable from a public module
  /// (i.e., has at least one path with no instance choices).
  bool isAlwaysReachable(FModuleLike publicModule,
                         FModuleLike destination) const {
    assert(publicModule.isPublic() &&
           "the first argument must be a public module");
    auto &it = alwaysReachable.at(publicModule);
    return it.contains(destination);
  }

  void dump(raw_ostream &os) const;

private:
  void computeAlwaysReachable(igraph::InstanceGraphNode *node,
                              FModuleLike publicModule);

  LogicalResult processNode(igraph::InstanceGraphNode *node,
                            FModuleLike publicModule);

  CircuitOp circuit;
  InstanceGraph &instanceGraph;

  /// Map from (public module, destination module) to all choices that can reach
  /// it. This tracks ALL choices from ALL paths (union of choices).
  /// Uses MapVector to preserve insertion order for deterministic output.
  llvm::MapVector<FModuleLike,
                  llvm::MapVector<FModuleLike, SetVector<ChoiceKey>>>
      moduleChoices;

  /// Set of (public module, destination module) pairs where the destination
  /// is always reachable (has at least one path with no instance choices).
  DenseMap<FModuleLike, DenseSet<FModuleLike>> alwaysReachable;
};

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_INSTANCECHOICEINFO_H
