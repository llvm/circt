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
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"

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

  /// Append the annotation path to the given `SmallString` or `SmallVector`.
  void toVector(SmallVectorImpl<char> &out) const;

  /// Convert the annotation path to a string.
  std::string str() const {
    SmallString<32> out;
    toVector(out);
    return std::string(out);
  }
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

/// Cache AnnoTargets for a module's named things.
struct AnnoTargetCache {
  AnnoTargetCache() = delete;
  AnnoTargetCache(const AnnoTargetCache &other) = default;
  AnnoTargetCache(AnnoTargetCache &&other)
      : targets(std::move(other.targets)){};

  AnnoTargetCache(FModuleLike mod) { gatherTargets(mod); };

  /// Lookup the target for 'name', empty if not found.
  /// (check for validity using operator bool()).
  AnnoTarget getTargetForName(StringRef name) const {
    return targets.lookup(name);
  }

private:
  /// Walk the module and add named things to 'targets'.
  void gatherTargets(FModuleLike mod);

  llvm::DenseMap<StringRef, AnnoTarget> targets;
};

/// Cache AnnoTargets for a circuit's modules, walked as needed.
struct CircuitTargetCache {
  /// Get cache for specified module, creating it as needed.
  /// Returned reference may become invalidated by future calls.
  const AnnoTargetCache &getOrCreateCacheFor(FModuleLike module) {
    auto it = targetCaches.find(module);
    if (it == targetCaches.end())
      it = targetCaches.try_emplace(module, module).first;
    return it->second;
  }

  /// Lookup the target for 'name' in 'module'.
  AnnoTarget lookup(FModuleLike module, StringRef name) {
    return getOrCreateCacheFor(module).getTargetForName(name);
  }

private:
  DenseMap<Operation *, AnnoTargetCache> targetCaches;
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
                                        SymbolTable &symTbl,
                                        CircuitTargetCache &cache);

/// Resolve a string path to a named item inside a circuit.
Optional<AnnoPathValue> resolvePath(StringRef rawPath, CircuitOp circuit,
                                    SymbolTable &symTbl,
                                    CircuitTargetCache &cache);

/// Return true if an Annotation's class name is handled by the LowerAnnotations
/// pass.
bool isAnnoClassLowered(StringRef className);

/// Stores the information content of an ExtractGrandCentralAnnotation.
struct GCExtractionInfo {
  /// The directory where Grand Central generated collateral (modules,
  /// interfaces, etc.) will be written.
  StringAttr directory = {};

  /// The name of the file where any binds will be written.  This will be placed
  /// in the same output area as normal compilation output, e.g., output
  /// Verilog.  This has no relation to the `directory` member.
  StringAttr bindFilename = {};
};

/// State threaded through functions for resolving and applying annotations.
struct ApplyState {
  using AddToWorklistFn = llvm::function_ref<void(DictionaryAttr)>;
  ApplyState(CircuitOp circuit, SymbolTable &symTbl,
             AddToWorklistFn addToWorklistFn)
      : circuit(circuit), symTbl(symTbl), addToWorklistFn(addToWorklistFn) {
        circuitNamespace = CircuitNamespace(circuit);
      }

  CircuitOp circuit;
  SymbolTable &symTbl;
  CircuitTargetCache targetCaches;
  CircuitNamespace circuitNamespace;
  AddToWorklistFn addToWorklistFn;

  /// Inforamtion about how the circuit should be extracted.  This will be
  /// non-empty if an extraction annotation is found.
  Optional<GCExtractionInfo> gcMaybeExtractInfo = None;

  /// A filename describing where to put a YAML representation of the
  /// interfaces generated by this pass.
  StringAttr gcMaybeHierarchyFileYAML ;

  /// An optional prefix applied to all interfaces in the design.  This is set
  /// based on a PrefixInterfacesAnnotation.
  StringAttr gcInterfacePrefix ;
  
  StringAttr testbenchDir;
  
  SmallVector<DictionaryAttr> gctViewAnnoList;

  ModuleNamespace &getNamespace(FModuleLike module) {
    auto &ptr = namespaces[module];
    if (!ptr)
      ptr = std::make_unique<ModuleNamespace>(module);
    return *ptr;
  }

  IntegerAttr newID() {
    return IntegerAttr::get(IntegerType::get(circuit.getContext(), 64),
                            annotationID++);
  };

  LogicalResult setGCextractionInfo(StringAttr dir, StringAttr fName) {

    if (gcMaybeExtractInfo)
      return failure();
    gcMaybeExtractInfo = {dir, fName};
    return success();
  }

  LogicalResult setGChierFileYaml(StringAttr fName) {

    if (gcMaybeHierarchyFileYAML)
      return failure();
    gcMaybeHierarchyFileYAML = fName;
    return success();
  }

  LogicalResult setGCinterfacePrefix(StringAttr prefix) {
    if (gcInterfacePrefix)
      return failure();
    gcInterfacePrefix = prefix;
    return success();
  }

private:
  DenseMap<Operation *, std::unique_ptr<ModuleNamespace>> namespaces;
  unsigned annotationID = 0;
};

LogicalResult applyGCinterfacePrefixInfo(const AnnoPathValue &target, DictionaryAttr anno, ApplyState &state);
LogicalResult applyGCHierFileInfo(const AnnoPathValue &target, DictionaryAttr anno, ApplyState &state);
LogicalResult applyExtractionInfo(const AnnoPathValue &target, DictionaryAttr anno, ApplyState &state);
LogicalResult applyGCTView(const AnnoPathValue &target, DictionaryAttr anno,
                           ApplyState &state);
LogicalResult handleGCTViews(ApplyState &state, InstanceGraph &instanceGraph);

LogicalResult applyGCTDataTaps(const AnnoPathValue &target, DictionaryAttr anno,
                               ApplyState &state);

LogicalResult applyGCTMemTaps(const AnnoPathValue &target, DictionaryAttr anno,
                              ApplyState &state);

LogicalResult applyGCTSignalMappings(const AnnoPathValue &target,
                                     DictionaryAttr anno, ApplyState &state);

LogicalResult applyOMIR(const AnnoPathValue &target, DictionaryAttr anno,
                        ApplyState &state);

/// Implements the same behavior as DictionaryAttr::getAs<A> to return the
/// value of a specific type associated with a key in a dictionary. However,
/// this is specialized to print a useful error message, specific to custom
/// annotation process, on failure.
template <typename A>
A tryGetAs(DictionaryAttr &dict, const Attribute &root, StringRef key,
           Location loc, Twine className, Twine path = Twine()) {
  // Check that the key exists.
  auto value = dict.get(key);
  if (!value) {
    SmallString<128> msg;
    if (path.isTriviallyEmpty())
      msg = ("Annotation '" + className + "' did not contain required key '" +
             key + "'.")
                .str();
    else
      msg = ("Annotation '" + className + "' with path '" + path +
             "' did not contain required key '" + key + "'.")
                .str();
    mlir::emitError(loc, msg).attachNote()
        << "The full Annotation is reproduced here: " << root << "\n";
    return nullptr;
  }
  // Check that the value has the correct type.
  auto valueA = value.dyn_cast_or_null<A>();
  if (!valueA) {
    SmallString<128> msg;
    if (path.isTriviallyEmpty())
      msg = ("Annotation '" + className +
             "' did not contain the correct type for key '" + key + "'.")
                .str();
    else
      msg = ("Annotation '" + className + "' with path '" + path +
             "' did not contain the correct type for key '" + key + "'.")
                .str();
    mlir::emitError(loc, msg).attachNote()
        << "The full Annotation is reproduced here: " << root << "\n";
    return nullptr;
  }
  return valueA;
}

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLANNOTATIONHELPER_H
