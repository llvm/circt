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
#include "circt/Dialect/SV/SVOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "gct"

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

/// An absolute instance path.
using InstancePath = ArrayRef<InstanceOp>;

template <typename T>
T &operator<<(T &os, const InstancePath &path) {
  os << "$root";
  for (auto inst : path)
    os << "." << inst.name();
  return os;
}

namespace {

/// A port annotated with a data tap key or mem tap.
struct AnnotatedPort {
  unsigned portNum;
  DictionaryAttr anno;
};

/// An extmodule that has annotated ports.
struct AnnotatedExtModule {
  FExtModuleOp extModule;
  SmallVector<AnnotatedPort, 4> portAnnos;
  ArrayAttr filteredModuleAnnos; // module annotations without data tap stuff
};

/// A value annotated to be tapped.
struct TappedValue {
  Value value;
  DictionaryAttr anno;
};

/// A data structure that tracks the instances of modules and can provide
/// absolute paths to these instances.
struct InstanceGraph {
  /// The root circuit.
  CircuitOp circuitOp;
  /// The main module in the circuit.
  Operation *mainModule;
  /// A mapping from a module to all its instances in the design.
  DenseMap<Operation *, SmallVector<InstanceOp, 1>> moduleInstances;

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

/// Necessary information to wire up a port with tapped data or memory location.
struct PortWiring {
  unsigned portNum;
  ArrayRef<InstancePath> prefices;
  SmallString<16> suffix;
};

} // namespace

InstanceGraph::InstanceGraph(CircuitOp circuitOp) : circuitOp(circuitOp) {
  mainModule = circuitOp.getMainModule();

  // Gather all instances in the circuit.
  circuitOp.walk([&](Operation *op) {
    if (auto instOp = dyn_cast<InstanceOp>(op))
      moduleInstances[instOp.getReferencedModule()].push_back(instOp);
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
// Static information gathered once upfront
//===----------------------------------------------------------------------===//

namespace {
/// Attributes used throughout the annotations.
struct Strings {
  MLIRContext *const context;
  Strings(MLIRContext *context) : context(context) {}

  Identifier annos = Identifier::get("annotations", context);
  Identifier fannos = Identifier::get("firrtl.annotations", context);

  StringAttr dataTapsClass = StringAttr::get(
      context, "sifive.enterprise.grandcentral.DataTapsAnnotation");
  StringAttr memTapClass = StringAttr::get(
      context, "sifive.enterprise.grandcentral.MemTapAnnotation");

  StringAttr deletedKeyClass = StringAttr::get(
      context, "sifive.enterprise.grandcentral.DeletedDataTapKey");
  StringAttr literalKeyClass = StringAttr::get(
      context, "sifive.enterprise.grandcentral.LiteralDataTapKey");
  StringAttr referenceKeyClass = StringAttr::get(
      context, "sifive.enterprise.grandcentral.ReferenceDataTapKey");
  StringAttr internalKeyClass = StringAttr::get(
      context, "sifive.enterprise.grandcentral.DataTapModuleSignalKey");
};
} // namespace

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
    AnnotatedExtModule result{extModule, {}, {}};
    for (unsigned argNum = 0, e = extModule.getNumArguments(); argNum < e;
         ++argNum) {
      auto annos =
          extModule.getArgAttrOfType<ArrayAttr>(argNum, strings.fannos);
      if (!annos)
        continue;

      // Go through all annotations on this port and add the data tap key and
      // mem tap ones to the list.
      for (auto anno : annos.getAsRange<DictionaryAttr>()) {
        auto cls = anno.getAs<StringAttr>("class");
        if (cls == strings.memTapClass || cls == strings.deletedKeyClass ||
            cls == strings.literalKeyClass ||
            cls == strings.referenceKeyClass || cls == strings.internalKeyClass)
          result.portAnnos.push_back({argNum, anno});
      }
    }

    // If there are data tap annotations on the module, which is likely the
    // case, create a filtered array of annotations with them removed.
    if (auto attrs = extModule->getAttrOfType<ArrayAttr>(strings.annos)) {
      auto isBad = [&](Attribute attr) {
        if (attr.cast<DictionaryAttr>().getAs<StringAttr>("class") ==
            strings.dataTapsClass)
          return true;
        return false;
      };
      bool anyBad = llvm::any_of(attrs, isBad);
      if (anyBad) {
        SmallVector<Attribute, 8> filteredAttrs;
        filteredAttrs.reserve(attrs.size());
        for (auto attr : attrs) {
          if (!isBad(attr))
            filteredAttrs.push_back(attr);
        }
        result.filteredModuleAnnos =
            ArrayAttr::get(&getContext(), filteredAttrs);
      } else {
        result.filteredModuleAnnos = attrs;
      }
    }

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

  // Build an instance graph.
  InstanceGraph instanceGraph(circuitOp);

  // Gather the annotated ports and operations throughout the design that we are
  // supposed to tap in one way or another.
  DenseMap<Attribute, BlockArgument> tappedArgs;
  DenseMap<Attribute, Operation *> tappedOps;
  circuitOp.walk([&](Operation *op) {
    if (auto module = dyn_cast<FModuleOp>(op)) {
      // Go through the module ports and collect the annotated ones.
      for (unsigned argNum = 0, e = module.getNumArguments(); argNum < e;
           ++argNum) {
        auto annos = module.getArgAttrOfType<ArrayAttr>(argNum, strings.fannos);
        if (!annos)
          continue;

        // Go through all annotations on this port and extract the interesting
        // ones.
        for (auto anno : annos.getAsRange<DictionaryAttr>()) {
          auto cls = anno.getAs<StringAttr>("class");
          if (cls == strings.referenceKeyClass)
            assert(
                tappedArgs.insert({anno, module.getArgument(argNum)}).second &&
                "ambiguous tap annotation");
        }
      }
    } else {
      auto annos = op->getAttrOfType<ArrayAttr>(strings.annos);
      if (!annos)
        return;

      // Go through all annotations on this op and extract the interesting ones.
      // Note that the way tap annotations are scattered to their targets, we
      // should never see multiple values or memories annotated with the exact
      // same annotation (hence the asserts).
      for (auto anno : annos.getAsRange<DictionaryAttr>()) {
        auto cls = anno.getAs<StringAttr>("class");
        if (cls == strings.memTapClass || cls == strings.referenceKeyClass ||
            cls == strings.internalKeyClass)
          assert(tappedOps.insert({anno, op}).second &&
                 "ambiguous tap annotation");
      }
    }
  });

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
    auto paths = instanceGraph.getAbsolutePaths(blackBox.extModule);
    for (auto path : paths)
      LLVM_DEBUG(llvm::dbgs() << "- " << path << "\n");

    // Go through the port annotations of the tap module and generate a
    // hierarchical path for each.
    SmallVector<PortWiring, 8> portWiring;
    SmallDenseMap<Attribute, unsigned, 2> memPortIdx;
    portWiring.reserve(blackBox.portAnnos.size());
    for (auto portAnno : blackBox.portAnnos) {
      LLVM_DEBUG(llvm::dbgs() << "- Processing port " << portAnno.portNum
                              << " anno " << portAnno.anno << "\n");
      auto portName = getModulePortName(blackBox.extModule, portAnno.portNum);
      auto cls = portAnno.anno.getAs<StringAttr>("class");
      PortWiring wiring = {portAnno.portNum, {}, {}};

      // Handle data taps on signals and ports.
      if (cls == strings.referenceKeyClass) {
        // Handle block arguments.
        if (auto blockArg = tappedArgs.lookup(portAnno.anno)) {
          auto parentModule = blockArg.getOwner()->getParentOp();
          wiring.prefices = instanceGraph.getAbsolutePaths(parentModule);
          wiring.suffix =
              getModulePortName(parentModule, blockArg.getArgNumber())
                  .getValue();
          portWiring.push_back(std::move(wiring));
          continue;
        }

        // Handle operations.
        if (auto op = tappedOps.lookup(portAnno.anno)) {
          // We currently require the target to be a wire.
          // TODO: This should probably also allow other things?
          auto wire = dyn_cast<WireOp>(op);
          if (!wire) {
            auto diag =
                blackBox.extModule.emitError("ReferenceDataTapKey on port ")
                << portName << " must be a wire";
            diag.attachNote(op->getLoc()) << "referenced operation is here:";
            signalPassFailure();
            continue;
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
                << "used by ReferenceDataTapKey on port " << portName
                << " here:";
            signalPassFailure();
            continue;
          }

          wiring.prefices = instanceGraph.getAbsolutePaths(
              wire->getParentOfType<FModuleOp>());
          wiring.suffix = name.getValue();
          portWiring.push_back(std::move(wiring));
          continue;
        }

        // The annotation scattering must have placed this annotation on some
        // target operation or block argument, which we should have picked up in
        // the tapped args or ops maps.
        blackBox.extModule.emitOpError(
            "ReferenceDataTapKey annotation was not scattered to "
            "an operation: ")
            << portAnno.anno;
        signalPassFailure();
        continue;
      }

      // Handle data taps on black boxes.
      if (cls == strings.internalKeyClass) {
        auto op = tappedOps.lookup(portAnno.anno);
        if (!op) {
          blackBox.extModule.emitOpError(
              "DataTapModuleSignalKey annotation was not scattered to "
              "an operation: ")
              << portAnno.anno;
          signalPassFailure();
          continue;
        }

        // Extract the internal path we're supposed to append.
        auto internalPath = portAnno.anno.getAs<StringAttr>("internalPath");
        if (!internalPath) {
          blackBox.extModule.emitError(
              "DataTapModuleSignalKey annotation on port ")
              << portName << " missing \"internalPath\" attribute";
          signalPassFailure();
          continue;
        }

        wiring.prefices = instanceGraph.getAbsolutePaths(op);
        wiring.suffix = internalPath.getValue();
        portWiring.push_back(std::move(wiring));
        continue;
      }

      // Handle data taps with literals.
      if (cls == strings.literalKeyClass) {
        blackBox.extModule.emitError(
            "LiteralDataTapKey annotations not yet supported (on port ")
            << portName << ")";
        signalPassFailure();
        continue;
      }

      // Handle memory taps.
      if (cls == strings.memTapClass) {
        auto op = tappedOps.lookup(portAnno.anno);
        if (!op) {
          blackBox.extModule.emitOpError(
              "DataTapModuleSignalKey annotation was not scattered to "
              "an operation: ")
              << portAnno.anno;
          signalPassFailure();
          continue;
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
          continue;
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
            instanceGraph.getAbsolutePaths(op->getParentOfType<FModuleOp>());
        wiring.suffix = name.getValue();
        wiring.suffix += '[';
        wiring.suffix += llvm::utostr(memPortIdx[portAnno.anno]++);
        wiring.suffix += ']';
        portWiring.push_back(std::move(wiring));
        continue;
      }

      llvm_unreachable("only accepted annos added to portAnnos list");
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
    OpBuilder builder(blackBox.extModule);
    unsigned implIdx = 0;
    for (auto path : paths) {
      builder.setInsertionPointAfter(blackBox.extModule);

      // Create a new firrtl.module that implements the data tap.
      SmallString<32> nameBuffer(blackBox.extModule.getName());
      nameBuffer += "_impl_";
      nameBuffer += llvm::utostr(implIdx++);
      auto name = StringAttr::get(&getContext(), nameBuffer);
      LLVM_DEBUG(llvm::dbgs()
                 << "Implementing " << name << " ("
                 << blackBox.extModule.getName() << " for " << path << ")\n");
      auto impl = builder.create<FModuleOp>(blackBox.extModule->getLoc(), name,
                                            blackBox.extModule.getPorts(),
                                            blackBox.filteredModuleAnnos);
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
            blackBox.extModule->getLoc(), arg.getType().cast<FIRRTLType>(),
            hname);
        builder.create<ConnectOp>(blackBox.extModule->getLoc(), arg, hnameExpr);
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

std::unique_ptr<mlir::Pass> circt::firrtl::createGrandCentralTapsPass() {
  return std::make_unique<GrandCentralTapsPass>();
}
