//===- GrandCentralTaps.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the GrandCentralTaps pass.
//
//===----------------------------------------------------------------------===//

#include "./PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/FieldRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "gct"

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

namespace llvm {
// IdentifierOps hash just like Operation pointers.
template <>
struct DenseMapInfo<InstanceOp> {
  static InstanceOp getEmptyKey() {
    return InstanceOp(llvm::DenseMapInfo<Operation *>::getEmptyKey());
  }
  static InstanceOp getTombstoneKey() {
    return InstanceOp(llvm::DenseMapInfo<Operation *>::getTombstoneKey());
  }
  static unsigned getHashValue(InstanceOp op) {
    return DenseMapInfo<Operation *>::getHashValue(op);
  }
  static bool isEqual(InstanceOp lhs, InstanceOp rhs) {
    return lhs.getOperation() == rhs.getOperation();
  }
};

template <>
struct DenseMapInfo<DictionaryAttr> {
  static DictionaryAttr getEmptyKey() {
    return llvm::DenseMapInfo<Attribute>::getEmptyKey().cast<DictionaryAttr>();
  }
  static DictionaryAttr getTombstoneKey() {
    return llvm::DenseMapInfo<Attribute>::getTombstoneKey()
        .cast<DictionaryAttr>();
  }
  static unsigned getHashValue(DictionaryAttr op) {
    return DenseMapInfo<Attribute>::getHashValue(op);
  }
  static bool isEqual(DictionaryAttr lhs, DictionaryAttr rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

//===----------------------------------------------------------------------===//
// Static information gathered once upfront
//===----------------------------------------------------------------------===//

namespace {
/// Attributes used throughout the annotations.
struct Strings {
  MLIRContext *const cx;
  Strings(MLIRContext *cx) : cx(cx) {}

  Identifier annos = Identifier::get("annotations", cx);
  Identifier fannos = Identifier::get("firrtl.annotations", cx);

  StringAttr dataTapsClass =
      StringAttr::get(cx, "sifive.enterprise.grandcentral.DataTapsAnnotation");
  StringAttr memTapClass =
      StringAttr::get(cx, "sifive.enterprise.grandcentral.MemTapAnnotation");

  StringAttr deletedKeyClass =
      StringAttr::get(cx, "sifive.enterprise.grandcentral.DeletedDataTapKey");
  StringAttr literalKeyClass =
      StringAttr::get(cx, "sifive.enterprise.grandcentral.LiteralDataTapKey");
  StringAttr referenceKeyClass =
      StringAttr::get(cx, "sifive.enterprise.grandcentral.ReferenceDataTapKey");
  StringAttr internalKeyClass = StringAttr::get(
      cx, "sifive.enterprise.grandcentral.DataTapModuleSignalKey");
};
} // namespace

//===----------------------------------------------------------------------===//
// Data Taps Implementation
//===----------------------------------------------------------------------===//

// static LogicalResult processDataTapAnnotation(FExtModuleOp module,
//                                               DictionaryAttr anno) {}

/// A port annotated with a data tap key or mem tap.
struct AnnotatedPort {
  unsigned portNum;
  DictionaryAttr anno;
};

/// An extmodule that has annotated ports.
struct AnnotatedExtModule {
  FExtModuleOp extModule;
  SmallVector<AnnotatedPort, 4> portAnnos;
};

/// A value annotated to be tapped.
struct TappedValue {
  Value value;
  DictionaryAttr anno;
};

/// An absolute instance path.
using InstancePath = ArrayRef<InstanceOp>;

template <typename T>
T &operator<<(T &os, const InstancePath &path) {
  os << "$root";
  for (auto inst : path)
    os << "." << inst.name();
  // if (path.empty())
  //   return os << "$root";
  // llvm::interleave(
  //     path.begin(), path.end(), [&](InstanceOp op) { os << op.name(); },
  //     [&] { os << "."; });
  return os;
}

/// A data structure that tracks the instances of modules and can provide
/// absolute paths to these instances.
struct InstanceGraph {
  /// The root circuit.
  CircuitOp circuitOp;
  /// The main module in the circuit.
  Operation *mainModule;
  /// A mapping from a module to all its instances in the design.
  DenseMap<Operation *, DenseSet<InstanceOp>> moduleInstances;

  InstanceGraph(CircuitOp circuitOp);
  ArrayRef<InstancePath> getAbsolutePaths(Operation *op);

private:
  /// An allocator for individual instance paths and entire path lists.
  llvm::BumpPtrAllocator allocator;

  /// Cached absolute instance paths.
  DenseMap<Operation *, ArrayRef<InstancePath>> absolutePathsCache;

  /// Append an instance to a path.
  InstancePath appendInstance(InstancePath path, InstanceOp inst);
};

InstanceGraph::InstanceGraph(CircuitOp circuitOp) : circuitOp(circuitOp) {
  mainModule = circuitOp.getMainModule();

  // Gather all instances in the circuit.
  circuitOp.walk([&](Operation *op) {
    if (auto instOp = dyn_cast<InstanceOp>(op))
      moduleInstances[instOp.getReferencedModule()].insert(instOp);
  });

  LLVM_DEBUG(llvm::dbgs() << "Instance graph:\n");
  for (auto it : moduleInstances) {
    LLVM_DEBUG(llvm::dbgs()
               << "- " << it.first->getName() << " "
               << it.first->getAttrOfType<StringAttr>("sym_name") << "\n");
    for (auto inst : it.second)
      LLVM_DEBUG(llvm::dbgs()
                 << "  - " << inst.nameAttr() << " in \""
                 << inst->getParentOfType<FModuleOp>().getName() << "\"\n");
  }
}

ArrayRef<InstancePath> InstanceGraph::getAbsolutePaths(Operation *op) {
  if (!isa<FModuleOp, FExtModuleOp>(op))
    return {};

  // If we have reached the circuit root, we're done.
  if (mainModule == op)
    return InstancePath{}; // array with single empty path

  // Otherwise look up the instances of this module.
  auto it = moduleInstances.find(op);
  if (it == moduleInstances.end())
    return {};

  // Fast path: hit the cache.
  auto cached = absolutePathsCache.find(op);
  if (cached != absolutePathsCache.end())
    return cached->second;

  // For each instance, collect the instance paths to its parent and append the
  // instance itself to each.
  SmallVector<InstancePath, 8> extendedPaths;
  for (auto inst : it->second) {
    auto instPaths = getAbsolutePaths(inst->getParentOfType<FModuleOp>());
    extendedPaths.reserve(instPaths.size());
    for (auto path : instPaths) {
      extendedPaths.push_back(appendInstance(path, inst));
    }
  }

  // Move the list of paths into the bump allocator for later quick retrieval.
  ArrayRef<InstancePath> pathList;
  if (!extendedPaths.empty()) {
    auto paths = allocator.Allocate<InstancePath>(extendedPaths.size());
    std::copy(extendedPaths.begin(), extendedPaths.end(), paths);
    pathList = ArrayRef<InstancePath>(paths, extendedPaths.size());
  }

  absolutePathsCache.insert({op, pathList});
  return pathList;
}

InstancePath InstanceGraph::appendInstance(InstancePath path, InstanceOp inst) {
  size_t n = path.size() + 1;
  auto newPath = allocator.Allocate<InstanceOp>(n);
  std::copy(path.begin(), path.end(), newPath);
  newPath[path.size()] = inst;
  return InstancePath(newPath, n);
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

class GrandCentralTapsPass : public GrandCentralTapsBase<GrandCentralTapsPass> {
  void runOnOperation() override;
};

void GrandCentralTapsPass::runOnOperation() {
  auto circuitOp = getOperation();
  LLVM_DEBUG(llvm::dbgs() << "Running the GCT Data Taps pass\n");

  // Here's a rough idea of what the Scala code is doing:
  // - Gather the `source` of all `keys` of all `DataTapsAnnotation`s throughout
  //   the design.
  // - Convert the sources, which are specified on modules, to the absolute
  //   paths in all instances. E.g. module M tap x will produce a.x and b.x if
  //   there are two instances a and b of module M in the design.
  // - All data tap keys are specified on black box ports.
  // - The code then processes every DataTapsAnnotation separately as follows
  //   (with the targeted blackbox and keys):
  // - Check for collisions between SV keywords and key port names (skip this).
  // - Find all instances of the blackbox, but then just pick the first one (we
  //   should probably do this for each?)
  // - Process each key independently as follows:
  // - Look up the absolute path of the source in the map. Ensure it exists and
  //   is unambiguous. Make it relative to the blackbox instance path.
  // - Look up the port on the black box.
  // - Create a hierarchical SV name and store this as an assignment to the
  //   blackbox port.
  //   - DeletedDataTapKey: skip and don't create a wiring
  //   - LiteralDataTapKey: just the literal
  //   - ReferenceDataTapKey: relative path with "." + source name
  //   - DataTapModuleSignalKey: relative path with "." + internal path
  // - Generate a body for the blackbox module with the signal mapping

  // Gather some string attributes in the context to simplify working with the
  // annotations.
  Strings strings(&getContext());

  // Gather a list of extmodules that have data or mem tap annotations to be
  // expanded.
  SmallVector<AnnotatedExtModule, 4> modules;
  for (auto &op : *circuitOp.getBody()) {
    auto extModule = dyn_cast<FExtModuleOp>(&op);
    if (!extModule)
      continue;

    // Go through the module ports and collect the annotated ones.
    AnnotatedExtModule result{extModule, {}};
    for (unsigned argNum = 0; argNum < extModule.getNumArguments(); ++argNum) {
      auto attrs =
          extModule.getArgAttrOfType<ArrayAttr>(argNum, strings.fannos);
      if (!attrs)
        continue;

      // Go through all annotations on this port and add the data tap key and
      // mem tap ones to the list.
      for (auto attr : attrs) {
        auto anno = attr.dyn_cast<DictionaryAttr>();
        if (!anno)
          continue;
        auto cls = anno.getAs<StringAttr>("class");
        if (cls == strings.memTapClass || cls == strings.deletedKeyClass ||
            cls == strings.literalKeyClass ||
            cls == strings.referenceKeyClass || cls == strings.internalKeyClass)
          result.portAnnos.push_back({argNum, anno});
      }
    }
    if (!result.portAnnos.empty())
      modules.push_back(std::move(result));
  }

  for (auto m : modules) {
    LLVM_DEBUG(llvm::dbgs() << "Extmodule " << m.extModule.getName() << " has "
                            << m.portAnnos.size() << " port annotations\n");
  }

  // Fast path if there's nothing to do.
  if (modules.empty())
    return;

  // Build an instance graph.
  InstanceGraph instanceGraph(circuitOp);

  // Gather the annotated ports and operations throughout the design that we are
  // supposed to tap in one way or another.
  DenseMap<Attribute, Value> tappedData;
  DenseMap<Attribute, Operation *> tappedMems;
  circuitOp.walk([&](Operation *op) {
    if (auto module = dyn_cast<FModuleOp>(op)) {
      // Go through the module ports and collect the annotated ones.
      for (unsigned argNum = 0; argNum < module.getNumArguments(); ++argNum) {
        auto attrs = module.getArgAttrOfType<ArrayAttr>(argNum, strings.fannos);
        if (!attrs)
          continue;

        // Go through all annotations on this port and extract the interesting
        // ones.
        for (auto attr : attrs) {
          auto anno = attr.dyn_cast<DictionaryAttr>();
          if (!anno)
            continue;
          auto cls = anno.getAs<StringAttr>("class");
          if (cls == strings.deletedKeyClass ||
              cls == strings.literalKeyClass ||
              cls == strings.referenceKeyClass ||
              cls == strings.internalKeyClass)
            assert(
                tappedData.insert({anno, module.getArgument(argNum)}).second &&
                "ambiguous data tap annotation");
        }
      }
    } else {
      // We only support tapping single result operations.
      if (op->getNumResults() != 1)
        return;
      auto attrs = op->getAttrOfType<ArrayAttr>(strings.annos);
      if (!attrs)
        return;

      // Go through all annotations on this op and extract the interesting ones.
      // Note that the way tap annotations are scattered to their targets, we
      // should never see multiple values or memories annotated with the exact
      // same annotation (hence the asserts).
      for (auto attr : attrs) {
        auto anno = attr.dyn_cast<DictionaryAttr>();
        if (!anno)
          continue;
        auto cls = anno.getAs<StringAttr>("class");
        if (cls == strings.memTapClass)
          assert(tappedMems.insert({anno, op}).second &&
                 "ambiguous mem tap annotation");
        else if (cls == strings.deletedKeyClass ||
                 cls == strings.literalKeyClass ||
                 cls == strings.referenceKeyClass ||
                 cls == strings.internalKeyClass)
          assert(tappedData.insert({anno, op->getResult(0)}).second &&
                 "ambiguous data tap annotation");
      }
    }
  });

  LLVM_DEBUG(llvm::dbgs() << "Tapped data:\n");
  for (auto it : tappedData)
    LLVM_DEBUG(llvm::dbgs() << "- " << it.first << ": " << it.second << "\n");
  LLVM_DEBUG(llvm::dbgs() << "Tapped mems:\n");
  for (auto it : tappedMems)
    LLVM_DEBUG(llvm::dbgs() << "- " << it.first << ": " << *it.second << "\n");

  // Process each black box independently.
  for (auto blackBox : modules) {
    LLVM_DEBUG(llvm::dbgs() << "Generating impls for "
                            << blackBox.extModule.getName() << "\n");

    // As a first step, gather a list of all absolute paths to instances of this
    // black box.
    auto paths = instanceGraph.getAbsolutePaths(blackBox.extModule);
    for (auto path : paths)
      LLVM_DEBUG(llvm::dbgs() << "- " << path << "\n");
  }

  // Gather the absolute paths to all instances of every annotated extmodule.
  // Since the contents of the generated taps module are sensitive to the where
  // the module is instantiated, this will inform what different variations of
  // the taps module need to be generated.
}

std::unique_ptr<mlir::Pass> circt::firrtl::createGrandCentralTapsPass() {
  return std::make_unique<GrandCentralTapsPass>();
}
