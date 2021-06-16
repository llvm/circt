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

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/InstanceGraph.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "gct"

using namespace circt;
using namespace firrtl;
using mlir::function_like_impl::getArgAttrDict;
using mlir::function_like_impl::setAllArgAttrDicts;

//===----------------------------------------------------------------------===//
// PointerLikeTypeTraits
//===----------------------------------------------------------------------===//

namespace llvm {
template <>
struct PointerLikeTypeTraits<InstanceOp> : PointerLikeTypeTraits<Operation *> {
public:
  static inline void *getAsVoidPointer(InstanceOp val) {
    return const_cast<void *>(val.getAsOpaquePointer());
  }
  static inline InstanceOp getFromVoidPointer(void *p) {
    return InstanceOp::getFromOpaquePointer(p);
  }
};
} // end namespace llvm

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

/// An absolute instance path.
using InstancePath = ArrayRef<InstanceOp>;

template <typename T>
static T &operator<<(T &os, const InstancePath &path) {
  os << "$root";
  for (auto inst : path)
    os << "." << inst.name();
  return os;
}

namespace {

/// A port annotated with a data tap key or mem tap.
struct AnnotatedPort {
  unsigned portNum;
  Annotation anno;
};

/// An extmodule that has annotated ports.
struct AnnotatedExtModule {
  FExtModuleOp extModule;
  SmallVector<AnnotatedPort, 4> portAnnos;
  // Module annotations without data tap stuff.
  ArrayAttr filteredModuleAnnos;
  /// Port annotations without data tap stuff.
  SmallVector<AnnotationSet, 8> filteredPortAnnos;
};

/// A value annotated to be tapped.
struct TappedValue {
  Value value;
  Annotation anno;
};

/// A data structure that caches and provides absolute paths to module instances
/// in the IR.
struct InstancePaths {
  /// The instance graph of the IR.
  InstanceGraph &instanceGraph;

  explicit InstancePaths(InstanceGraph &instanceGraph)
      : instanceGraph(instanceGraph) {}
  ArrayRef<InstancePath> getAbsolutePaths(Operation *op);

private:
  /// An allocator for individual instance paths and entire path lists.
  llvm::BumpPtrAllocator allocator;

  /// Cached absolute instance paths.
  DenseMap<Operation *, ArrayRef<InstancePath>> absolutePathsCache;

  /// Append an instance to a path.
  InstancePath appendInstance(InstancePath path, InstanceOp inst);
};

/// Necessary information to wire up a port with tapped data or memory location.
struct PortWiring {
  unsigned portNum;
  ArrayRef<InstancePath> prefices;
  SmallString<16> suffix;
};

} // namespace

ArrayRef<InstancePath> InstancePaths::getAbsolutePaths(Operation *op) {
  assert((isa<FModuleOp, FExtModuleOp>(op))); // extra parens makes parser smile

  // If we have reached the circuit root, we're done.
  if (op == instanceGraph.getTopLevelNode()->getModule()) {
    static InstancePath empty{};
    return empty; // array with single empty path
  }

  // Fast path: hit the cache.
  auto cached = absolutePathsCache.find(op);
  if (cached != absolutePathsCache.end())
    return cached->second;

  // For each instance, collect the instance paths to its parent and append the
  // instance itself to each.
  SmallVector<InstancePath, 8> extendedPaths;
  for (auto inst : instanceGraph[op]->uses()) {
    auto instPaths = getAbsolutePaths(inst->getParent()->getModule());
    extendedPaths.reserve(instPaths.size());
    for (auto path : instPaths) {
      extendedPaths.push_back(appendInstance(path, inst->getInstance()));
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

InstancePath InstancePaths::appendInstance(InstancePath path, InstanceOp inst) {
  size_t n = path.size() + 1;
  auto newPath = allocator.Allocate<InstanceOp>(n);
  std::copy(path.begin(), path.end(), newPath);
  newPath[path.size()] = inst;
  return InstancePath(newPath, n);
}

/// Return a version of `path` that skips all front instances it has in common
/// with `other`.
static InstancePath stripCommonPrefix(InstancePath path, InstancePath other) {
  while (!path.empty() && !other.empty() && path.front() == other.front()) {
    path = path.drop_front();
    other = other.drop_front();
  }
  return path;
}

//===----------------------------------------------------------------------===//
// Static class names
//===----------------------------------------------------------------------===//

static constexpr const char *dataTapsClass =
    "sifive.enterprise.grandcentral.DataTapsAnnotation";
static constexpr const char *memTapClass =
    "sifive.enterprise.grandcentral.MemTapAnnotation";
static constexpr const char *deletedKeyClass =
    "sifive.enterprise.grandcentral.DeletedDataTapKey";
static constexpr const char *literalKeyClass =
    "sifive.enterprise.grandcentral.LiteralDataTapKey";
static constexpr const char *referenceKeyClass =
    "sifive.enterprise.grandcentral.ReferenceDataTapKey";
static constexpr const char *internalKeyClass =
    "sifive.enterprise.grandcentral.DataTapModuleSignalKey";

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

class GrandCentralTapsPass : public GrandCentralTapsBase<GrandCentralTapsPass> {
  void runOnOperation() override;
  void gatherAnnotations(Operation *op);
  void processAnnotation(AnnotatedPort &portAnno, AnnotatedExtModule &blackBox,
                         InstancePaths &instancePaths);

private:
  DenseMap<Attribute, BlockArgument> tappedArgs;
  DenseMap<Attribute, Operation *> tappedOps;
  SmallDenseMap<Attribute, unsigned, 2> memPortIdx;
  SmallVector<PortWiring, 8> portWiring;
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

  memPortIdx.clear();

  // Gather a list of extmodules that have data or mem tap annotations to be
  // expanded.
  SmallVector<AnnotatedExtModule, 4> modules;
  for (auto &op : *circuitOp.getBody()) {
    auto extModule = dyn_cast<FExtModuleOp>(&op);
    if (!extModule)
      continue;

    // Go through the module ports and collect the annotated ones.
    AnnotatedExtModule result{extModule, {}, {}, {}};
    result.filteredPortAnnos.reserve(extModule.getNumArguments());
    for (unsigned argNum = 0, e = extModule.getNumArguments(); argNum < e;
         ++argNum) {
      // Go through all annotations on this port and add the data tap key and
      // mem tap ones to the list.
      auto annos = AnnotationSet::forPort(extModule, argNum);
      annos.removeAnnotations([&](Annotation anno) {
        if (anno.isClass(memTapClass, deletedKeyClass, literalKeyClass,
                         referenceKeyClass, internalKeyClass)) {
          result.portAnnos.push_back({argNum, anno});
          return true;
        }
        return false;
      });
      result.filteredPortAnnos.push_back(annos);
    }

    // If there are data tap annotations on the module, which is likely the
    // case, create a filtered array of annotations with them removed.
    AnnotationSet annos(extModule.getOperation());
    annos.removeAnnotations(
        [&](Annotation anno) { return anno.isClass(dataTapsClass); });
    result.filteredModuleAnnos = annos.getArrayAttr();

    if (!result.portAnnos.empty())
      modules.push_back(std::move(result));
  }

#ifndef NDEBUG
  for (auto m : modules) {
    LLVM_DEBUG(llvm::dbgs() << "Extmodule " << m.extModule.getName() << " has "
                            << m.portAnnos.size() << " port annotations\n");
  }
#endif

  // Fast path if there's nothing to do.
  if (modules.empty()) {
    markAllAnalysesPreserved();
    return;
  }

  // Build a generator for absolute module and instance paths in the design.
  InstancePaths instancePaths(getAnalysis<InstanceGraph>());

  // Gather the annotated ports and operations throughout the design that we are
  // supposed to tap in one way or another.
  tappedArgs.clear();
  tappedOps.clear();
  circuitOp.walk([&](Operation *op) { gatherAnnotations(op); });

#ifndef NDEBUG
  LLVM_DEBUG(llvm::dbgs() << "Tapped values:\n");
  for (auto it : tappedArgs)
    LLVM_DEBUG(llvm::dbgs() << "- " << it.first << ": " << it.second << "\n");
  LLVM_DEBUG(llvm::dbgs() << "Tapped ops:\n");
  for (auto it : tappedOps)
    LLVM_DEBUG(llvm::dbgs() << "- " << it.first << ": " << *it.second << "\n");
#endif

  // Process each black box independently.
  for (auto blackBox : modules) {
    LLVM_DEBUG(llvm::dbgs() << "Generating impls for "
                            << blackBox.extModule.getName() << "\n");

    // As a first step, gather a list of all absolute paths to instances of this
    // black box.
    auto paths = instancePaths.getAbsolutePaths(blackBox.extModule);
#ifndef NDEBUG
    for (auto path : paths)
      LLVM_DEBUG(llvm::dbgs() << "- " << path << "\n");
#endif

    // Go through the port annotations of the tap module and generate a
    // hierarchical path for each.
    portWiring.clear();
    portWiring.reserve(blackBox.portAnnos.size());
    for (auto portAnno : blackBox.portAnnos) {
      processAnnotation(portAnno, blackBox, instancePaths);
    }

#ifndef NDEBUG
    LLVM_DEBUG(llvm::dbgs() << "- Wire up as follows:\n");
    for (auto wiring : portWiring) {
      LLVM_DEBUG(llvm::dbgs() << "- Port " << wiring.portNum << ":\n");
      for (auto path : wiring.prefices) {
        LLVM_DEBUG(llvm::dbgs()
                   << "  - " << path << "." << wiring.suffix << "\n");
      }
    }
#endif

    // Now we have an awkward mapping problem. We have multiple data tap module
    // instances, which reference things in modules that in turn have multiple
    // instances. This is a side-effect of how Grand Central annotates things on
    // modules rather than instances. (However in practice these will have a
    // one-to-one correspondence due to CHIRRTL having fully uniquified
    // instances.) To solve this issue, create a dedicated implementation for
    // every data tap instance, and among the possible targets for the data taps
    // choose the one with the shortest relative path to the data tap instance.
    ImplicitLocOpBuilder builder(blackBox.extModule->getLoc(),
                                 blackBox.extModule);
    unsigned implIdx = 0;
    for (auto path : paths) {
      builder.setInsertionPointAfter(blackBox.extModule);

      // Get the list of ports from the original extmodule, and update the
      // annotations such that they no longer contain any data/mem taps.
      auto ports = blackBox.extModule.getPorts();
      for (auto port : llvm::zip(ports, blackBox.filteredPortAnnos)) {
        std::get<0>(port).annotations = std::get<1>(port);
      }

      // Create a new firrtl.module that implements the data tap.
      auto name =
          StringAttr::get(&getContext(), Twine(blackBox.extModule.getName()) +
                                             "_impl_" + Twine(implIdx++));
      LLVM_DEBUG(llvm::dbgs()
                 << "Implementing " << name << " ("
                 << blackBox.extModule.getName() << " for " << path << ")\n");
      auto impl =
          builder.create<FModuleOp>(name, ports, blackBox.filteredModuleAnnos);
      builder.setInsertionPointToEnd(impl.getBodyBlock());

      // Connect the output ports to the appropriate tapped object.
      for (auto port : portWiring) {
        LLVM_DEBUG(llvm::dbgs() << "- Wiring up port " << port.portNum << "\n");
        // Determine the shortest hierarchical prefix from this black box
        // instance to the tapped object.
        Optional<InstancePath> shortestPrefix;
        for (auto prefix : port.prefices) {
          auto relative = stripCommonPrefix(prefix, path);
          if (!shortestPrefix.hasValue() ||
              relative.size() < shortestPrefix->size())
            shortestPrefix = relative;
        }
        if (!shortestPrefix.hasValue()) {
          LLVM_DEBUG(llvm::dbgs() << "  - Has no prefix, skipping\n");
          continue;
        }
        LLVM_DEBUG(llvm::dbgs()
                   << "  - Shortest prefix " << *shortestPrefix << "\n");

        // Concatenate the prefix into a proper full hierarchical name.
        SmallString<128> hname;
        for (auto inst : shortestPrefix.getValue()) {
          if (!hname.empty())
            hname += '.';
          hname += inst.name();
        }
        if (!hname.empty())
          hname += '.';
        hname += port.suffix;
        LLVM_DEBUG(llvm::dbgs() << "  - Connecting as " << hname << "\n");

        // Add a verbatim op that assigns this module port.
        auto arg = impl.getArgument(port.portNum);
        auto hnameExpr = builder.create<VerbatimExprOp>(
            arg.getType().cast<FIRRTLType>(), hname);
        builder.create<ConnectOp>(arg, hnameExpr);
      }

      // Switch the instance from the original extmodule to this implementation.
      // CAVEAT: If the same black box data tap module is instantiated in a
      // parent module that itself is instantiated in different locations, this
      // will pretty arbitrarily pick one of those locations.
      path.back()->setAttr("moduleName",
                           builder.getSymbolRefAttr(name.getValue()));
    }

    // Drop the original black box module.
    blackBox.extModule.erase();
  }
}

/// Gather the annotations on ports and operations into the `tappedOps` and
/// `tappedArgs` maps.
void GrandCentralTapsPass::gatherAnnotations(Operation *op) {
  if (auto module = dyn_cast<FModuleOp>(op)) {
    // Go through the module ports and collect the annotated ones.
    SmallVector<DictionaryAttr, 8> filteredArgAttrs;
    filteredArgAttrs.reserve(module.getNumArguments());
    for (unsigned argNum = 0, e = module.getNumArguments(); argNum < e;
         ++argNum) {
      auto annos = AnnotationSet::forPort(module, argNum);
      if (annos.empty()) {
        filteredArgAttrs.push_back(getArgAttrDict(module, argNum));
        continue;
      }

      // Go through all annotations on this port and extract the interesting
      // ones.
      annos.removeAnnotations([&](Annotation anno) {
        if (anno.isClass(referenceKeyClass)) {
          assert(tappedArgs.insert({anno.getDict(), module.getArgument(argNum)})
                     .second &&
                 "ambiguous tap annotation");
          return true;
        }
        return false;
      });
      filteredArgAttrs.push_back(
          annos.applyToPortDictionaryAttr(getArgAttrDict(module, argNum)));
    }
    setAllArgAttrDicts(module, filteredArgAttrs);
  } else {
    AnnotationSet annos(op);
    if (annos.empty())
      return;

    // Go through all annotations on this op and extract the interesting ones.
    // Note that the way tap annotations are scattered to their targets, we
    // should never see multiple values or memories annotated with the exact
    // same annotation (hence the asserts).
    annos.removeAnnotations([&](Annotation anno) {
      if (anno.isClass(memTapClass, referenceKeyClass, internalKeyClass)) {
        assert(tappedOps.insert({anno.getDict(), op}).second &&
               "ambiguous tap annotation");
        return true;
      }
      return false;
    });
    annos.applyToOperation(op);
  }
}

void GrandCentralTapsPass::processAnnotation(AnnotatedPort &portAnno,
                                             AnnotatedExtModule &blackBox,
                                             InstancePaths &instancePaths) {
  LLVM_DEBUG(llvm::dbgs() << "- Processing port " << portAnno.portNum
                          << " anno " << portAnno.anno.getDict() << "\n");
  auto portName = getModulePortName(blackBox.extModule, portAnno.portNum);
  PortWiring wiring = {portAnno.portNum, {}, {}};

  // Handle data taps on signals and ports.
  if (portAnno.anno.isClass(referenceKeyClass)) {
    // Handle block arguments.
    if (auto blockArg = tappedArgs.lookup(portAnno.anno.getDict())) {
      auto parentModule = blockArg.getOwner()->getParentOp();
      wiring.prefices = instancePaths.getAbsolutePaths(parentModule);
      wiring.suffix =
          getModulePortName(parentModule, blockArg.getArgNumber()).getValue();
      portWiring.push_back(std::move(wiring));
      return;
    }

    // Handle operations.
    if (auto op = tappedOps.lookup(portAnno.anno.getDict())) {
      // We currently require the target to be a wire.
      // TODO: This should probably also allow other things?
      auto wire = dyn_cast<WireOp>(op);
      if (!wire) {
        auto diag = blackBox.extModule.emitError("ReferenceDataTapKey on port ")
                    << portName << " must be a wire";
        diag.attachNote(op->getLoc()) << "referenced operation is here:";
        signalPassFailure();
        return;
      }

      // We currently require the target to be named.
      // TODO: If we were to use proper cross-module reference ops in the IR
      // then this could be anonymous, with ExportVerilog resolving the name
      // at the last moment.
      auto name = wire.nameAttr();
      if (!name) {
        auto diag =
            wire.emitError("wire targeted by data tap must have a name");
        diag.attachNote(blackBox.extModule->getLoc())
            << "used by ReferenceDataTapKey on port " << portName << " here:";
        signalPassFailure();
        return;
      }

      wiring.prefices =
          instancePaths.getAbsolutePaths(wire->getParentOfType<FModuleOp>());
      wiring.suffix = name.getValue();
      portWiring.push_back(std::move(wiring));
      return;
    }

    // The annotation scattering must have placed this annotation on some
    // target operation or block argument, which we should have picked up in
    // the tapped args or ops maps.
    blackBox.extModule.emitOpError(
        "ReferenceDataTapKey annotation was not scattered to "
        "an operation: ")
        << portAnno.anno.getDict();
    signalPassFailure();
    return;
  }

  // Handle data taps on black boxes.
  if (portAnno.anno.isClass(internalKeyClass)) {
    auto op = tappedOps.lookup(portAnno.anno.getDict());
    if (!op) {
      blackBox.extModule.emitOpError(
          "DataTapModuleSignalKey annotation was not scattered to "
          "an operation: ")
          << portAnno.anno.getDict();
      signalPassFailure();
      return;
    }

    // Extract the internal path we're supposed to append.
    auto internalPath = portAnno.anno.getMember<StringAttr>("internalPath");
    if (!internalPath) {
      blackBox.extModule.emitError("DataTapModuleSignalKey annotation on port ")
          << portName << " missing \"internalPath\" attribute";
      signalPassFailure();
      return;
    }

    wiring.prefices = instancePaths.getAbsolutePaths(op);
    wiring.suffix = internalPath.getValue();
    portWiring.push_back(std::move(wiring));
    return;
  }

  // Handle data taps with literals.
  if (portAnno.anno.isClass(literalKeyClass)) {
    blackBox.extModule.emitError(
        "LiteralDataTapKey annotations not yet supported (on port ")
        << portName << ")";
    signalPassFailure();
    return;
  }

  // Handle memory taps.
  if (portAnno.anno.isClass(memTapClass)) {
    auto op = tappedOps.lookup(portAnno.anno.getDict());
    if (!op) {
      blackBox.extModule.emitOpError(
          "DataTapModuleSignalKey annotation was not scattered to "
          "an operation: ")
          << portAnno.anno.getDict();
      signalPassFailure();
      return;
    }

    // Extract the name of the memory.
    // TODO: This would be better handled through a proper cross-module
    // reference preserved in the IR, such that ExportVerilog can insert a
    // proper name here at the last moment.
    auto name = op->getAttrOfType<StringAttr>("name");
    if (!name) {
      auto diag = op->emitError("target of memory tap must have a name");
      diag.attachNote(blackBox.extModule->getLoc())
          << "used by MemTapAnnotation on port " << portName << " here:";
      signalPassFailure();
      return;
    }

    // Formulate a hierarchical reference into the memory.
    // CAVEAT: This just assumes that the memory will map to something that
    // can be indexed in the final Verilog. If the memory gets turned into
    // an instance of some sort, we lack the information necessary to go in
    // and access individual elements of it. This will break horribly since
    // we generate memory impls out-of-line already, and memories coming
    // from an external generator are even worse. This needs a special node
    // in the IR that can properly inject the memory array on emission.
    wiring.prefices =
        instancePaths.getAbsolutePaths(op->getParentOfType<FModuleOp>());
    (Twine(name.getValue()) + "[" +
     llvm::utostr(memPortIdx[portAnno.anno.getDict()]++) + "]")
        .toVector(wiring.suffix);
    portWiring.push_back(std::move(wiring));
    return;
  }

  // We never arrive here since the above code that populates the portAnnos
  // list only adds annotations that we handle in one of the if statements
  // above.
  llvm_unreachable("portAnnos is never populated with unsupported annos");
}

std::unique_ptr<mlir::Pass> circt::firrtl::createGrandCentralTapsPass() {
  return std::make_unique<GrandCentralTapsPass>();
}
