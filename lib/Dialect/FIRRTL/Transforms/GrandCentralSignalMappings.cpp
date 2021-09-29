//===- GrandCentralSignalMappings.cpp ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the GrandCentralSignalMappings pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "gct"

using namespace circt;
using namespace firrtl;

static constexpr const char *signalDriverAnnoClass =
    "sifive.enterprise.grandcentral.SignalDriverAnnotation";

//===----------------------------------------------------------------------===//
// Per-Module Signal Mappings
//===----------------------------------------------------------------------===//

namespace {
enum class MappingDirection {
  /// The local signal forces the remote signal.
  DriveRemote,
  /// The local signal is driven to the value of the remote signal through a
  /// connect.
  ProbeRemote,
};

/// The information necessary to connect a local signal in a module to a remote
/// value (in a different module and/or circuit).
struct SignalMapping {
  /// Whether we force or read from the target.
  MappingDirection dir;
  /// The reference target of the thing we are forcing or probing.
  StringAttr remoteTarget;
  /// The type of the signal being mapped.
  FIRRTLType type;
  /// The block argument or result that interacts with the remote target, either
  /// by forcing it or by reading from it through a connect.
  Value localValue;
  /// The name of the local value, for reuse in the generated signal mappings
  /// module.
  StringAttr localName;
};

/// A helper structure that collects the data necessary to generate the signal
/// mappings module for an existing `FModuleOp` in the IR.
struct ModuleSignalMappings {
  ModuleSignalMappings(FModuleOp module) : module(module) {}
  void run();
  void addTarget(Value value, Annotation anno);
  void emitMappingsModule();
  void instantiateMappingsModule();

  FModuleOp module;
  bool allAnalysesPreserved = false;
  SmallVector<SignalMapping> mappings;
  SmallString<64> mappingsModuleName;
};
} // namespace

// Allow `SignalMapping` to be printed.
template <typename T>
static T &operator<<(T &os, const SignalMapping &mapping) {
  os << "SignalMapping { remote"
     << (mapping.dir == MappingDirection::DriveRemote ? "Sink" : "Source")
     << ": " << mapping.remoteTarget << ", "
     << "localTarget: " << mapping.localName << " }";
  return os;
}

/// Analyze the `module` of this `ModuleSignalMappings` and generate the
/// corresponding auxiliary `FModuleOp` with the necessary cross-module
/// references and `ForceOp`s to probe and drive remote signals. This is
/// dictated by the presence of `SignalDriverAnnotation` on the module and
/// individual operations inside it.
void ModuleSignalMappings::run() {
  // Check whether this module has any `SignalDriverAnnotation`s. These indicate
  // whether the module contains any operations with such annotations and
  // requires processing.
  if (!AnnotationSet::removeAnnotations(module, signalDriverAnnoClass)) {
    LLVM_DEBUG(llvm::dbgs() << "Skipping `" << module.getName()
                            << "` (has no annotations)\n");
    allAnalysesPreserved = true;
    return;
  }
  LLVM_DEBUG(llvm::dbgs() << "Running on module `" << module.getName()
                          << "`\n");

  // Gather the signal driver annotations on the ports of this module.
  LLVM_DEBUG(llvm::dbgs() << "- Gather port annotations\n");
  AnnotationSet::removePortAnnotations(
      module, [&](unsigned i, Annotation anno) {
        if (!anno.isClass(signalDriverAnnoClass))
          return false;
        addTarget(module.getArgument(i), anno);
        return true;
      });

  // Gather the signal driver annotations of the operations within this module.
  LLVM_DEBUG(llvm::dbgs() << "- Gather operation annotations\n");
  module.walk([&](Operation *op) {
    AnnotationSet::removeAnnotations(op, [&](Annotation anno) {
      if (!anno.isClass(signalDriverAnnoClass))
        return false;
      for (auto result : op->getResults())
        addTarget(result, anno);
      return true;
    });
  });

  // Pick a name for the module that implements the signal mappings.
  mappingsModuleName = module.getName();
  mappingsModuleName.append("_signal_mappings");

  // Generate the mappings module.
  emitMappingsModule();

  // Instantiate the mappings module.
  instantiateMappingsModule();
}

/// Mark a `value` inside the `module` as being the target of the
/// `SignalDriverAnnotation` `anno`. This generates the necessary
/// `SignalMapping` information and adds an entry to the `mappings` array, to be
/// later consumed when the mappings module is constructed.
void ModuleSignalMappings::addTarget(Value value, Annotation anno) {
  // We're emitting code for the "local" side of these annotations, which
  // sits in the sub-circuit and interacts with the main circuit on the
  // "remote" side.
  if (anno.getMember<StringAttr>("side").getValue() != "local")
    return;

  SignalMapping mapping;
  mapping.dir = anno.getMember<StringAttr>("dir").getValue() == "source"
                    ? MappingDirection::ProbeRemote
                    : MappingDirection::DriveRemote;
  mapping.remoteTarget = anno.getMember<StringAttr>("peer");
  mapping.localValue = value;
  mapping.type = value.getType().cast<FIRRTLType>();

  // Guess a name for the local value. This is only for readability's sake,
  // giving the pass a hint for picking the names of the generated module ports.
  if (auto blockArg = value.dyn_cast<BlockArgument>()) {
    mapping.localName =
        module.portNames()[blockArg.getArgNumber()].cast<StringAttr>();
  } else if (auto op = value.getDefiningOp()) {
    mapping.localName = op->getAttrOfType<StringAttr>("name");
  }

  LLVM_DEBUG(llvm::dbgs() << "  - " << mapping << "\n");
  mappings.push_back(std::move(mapping));
}

/// Create a separate mappings module that contains cross-module references and
/// `ForceOp`s for each entry in the `mappings` array.
void ModuleSignalMappings::emitMappingsModule() {
  LLVM_DEBUG(llvm::dbgs() << "- Generating `" << mappingsModuleName << "`\n");

  // Determine what ports this module will have, given the signal mappings we
  // are supposed to emit.
  SmallVector<PortInfo> ports;
  for (auto &mapping : mappings) {
    ports.push_back(PortInfo{mapping.localName, mapping.type,
                             mapping.dir == MappingDirection::DriveRemote
                                 ? Direction::In
                                 : Direction::Out,
                             module.getLoc()});
    LLVM_DEBUG(llvm::dbgs() << "  - Adding port " << mapping.localName << "\n");
  }

  // Create the actual module.
  ImplicitLocOpBuilder builder(module.getLoc(), module);
  auto mappingsModule = builder.create<FModuleOp>(
      StringAttr::get(module.getContext(), mappingsModuleName), ports);

  // Generate the connect and force statements inside the module.
  builder.setInsertionPointToStart(mappingsModule.getBodyBlock());
  unsigned portIdx = 0;
  for (auto &mapping : mappings) {
    // TODO: Actually generate a proper XMR here. For now just do some textual
    // replacements. Generating a real IR node (like a proper XMR op) would be
    // much better, but the modules that `EmitSignalMappings` interacts with
    // generally live in a separate circuit. Multiple circuits are not fully
    // supported at the moment.
    auto circuitSplit = mapping.remoteTarget.getValue().split('|').second;
    auto moduleSplit = circuitSplit.split('>');
    SmallString<32> remoteXmrName(moduleSplit.first.split(':').first);
    remoteXmrName.push_back('.');
    for (auto c : moduleSplit.second) {
      if (c == '[' || c == '.')
        remoteXmrName.push_back('_');
      else if (c != ']')
        remoteXmrName.push_back(c);
    }
    if (mapping.dir == MappingDirection::DriveRemote) {
      auto xmr = builder.create<VerbatimWireOp>(mapping.type, remoteXmrName);
      builder.create<ForceOp>(xmr, mappingsModule.getArgument(portIdx++));
    } else {
      auto xmr = builder.create<VerbatimWireOp>(mapping.type, remoteXmrName);
      builder.create<ConnectOp>(mappingsModule.getArgument(portIdx++), xmr);
    }
  }
}

/// Instantiate the generated mappings module inside the `module` we are working
/// on, and generated the necessary connections to local signals.
void ModuleSignalMappings::instantiateMappingsModule() {
  LLVM_DEBUG(llvm::dbgs() << "- Instantiating `" << mappingsModuleName
                          << "`\n");
  // Determine the port types.
  SmallVector<Type> resultTypes;
  for (auto &mapping : mappings)
    resultTypes.push_back(mapping.type);

  // Create the actual module.
  auto builder =
      ImplicitLocOpBuilder::atBlockEnd(module.getLoc(), module.getBodyBlock());
  auto inst = builder.create<InstanceOp>(resultTypes, mappingsModuleName,
                                         "signal_mappings");

  // Generate the connections to and from the instance.
  unsigned portIdx = 0;
  for (auto &mapping : mappings) {
    Value dst = inst.getResult(portIdx++);
    Value src = mapping.localValue;
    if (mapping.dir == MappingDirection::ProbeRemote)
      std::swap(src, dst);
    builder.create<ConnectOp>(dst, src);
  }
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

class GrandCentralSignalMappingsPass
    : public GrandCentralSignalMappingsBase<GrandCentralSignalMappingsPass> {
  void runOnOperation() override;
};

void GrandCentralSignalMappingsPass::runOnOperation() {
  FModuleOp module = getOperation();
  ModuleSignalMappings mapper(module);
  mapper.run();
  if (mapper.allAnalysesPreserved)
    markAllAnalysesPreserved();
}

std::unique_ptr<mlir::Pass>
circt::firrtl::createGrandCentralSignalMappingsPass() {
  return std::make_unique<GrandCentralSignalMappingsPass>();
}
