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
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/InstanceGraph.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/Attributes.h"
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

//===----------------------------------------------------------------------===//
// Static class names
//===----------------------------------------------------------------------===//

static constexpr const char *signalDriverAnnoClass =
    "sifive.enterprise.grandcentral.SignalDriverAnnotation";

//===----------------------------------------------------------------------===//
// Per-Module Signal Mappings
//===----------------------------------------------------------------------===//

namespace {
enum class MappingDirection {
  DriveRemote,
  ProbeRemote,
};

struct SignalMapping {
  MappingDirection dir;
  StringAttr remoteTarget;
  StringAttr localName;
  StringRef localFields;

  /// The type of the signal being mapped.
  FIRRTLType type;
  /// The `localName` and `localFields` resolved to an actual value.
  Value localValue;
};

struct ModuleSignalMappings {
  ModuleSignalMappings(FModuleOp module) : module(module) {}
  void run();
  void addTarget(Value value, Annotation anno);
  void emitMappingsModule();
  void instantiateMappingsModule();

  FModuleOp module;
  bool anyFailed = false;
  bool allAnalysesPreserved = false;
  SmallVector<SignalMapping> mappings;
  SmallString<64> mappingsModuleName;
};
} // namespace

template <typename T>
static T &operator<<(T &os, const SignalMapping &mapping) {
  os << "SignalMapping { remote"
     << (mapping.dir == MappingDirection::DriveRemote ? "Sink" : "Source")
     << ": " << mapping.remoteTarget << ", "
     << "localTarget: " << mapping.localName << " }";
  return os;
}

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

void ModuleSignalMappings::emitMappingsModule() {
  LLVM_DEBUG(llvm::dbgs() << "- Generating `" << mappingsModuleName << "`\n");

  // Determine what ports this module will have, given the signal mappings we
  // are supposed to emit.
  SmallVector<ModulePortInfo> ports;
  for (auto &mapping : mappings) {
    ports.push_back(ModulePortInfo{mapping.localName, mapping.type,
                                   mapping.dir == MappingDirection::DriveRemote
                                       ? Direction::Input
                                       : Direction::Output,
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
      builder.create<ForceOp>(xmr, mappingsModule.getArgument(portIdx));
    } else {
      auto xmr = builder.create<VerbatimWireOp>(mapping.type, remoteXmrName);
      builder.create<ConnectOp>(mappingsModule.getArgument(portIdx), xmr);
    }
    ++portIdx;
  }
}

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
    Value dst = inst.getResult(portIdx);
    Value src = mapping.localValue;
    if (mapping.dir == MappingDirection::ProbeRemote)
      std::swap(src, dst);
    builder.create<ConnectOp>(dst, src);
    ++portIdx;
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
  if (mapper.anyFailed)
    return signalPassFailure();
  if (mapper.allAnalysesPreserved)
    markAllAnalysesPreserved();
}

std::unique_ptr<mlir::Pass>
circt::firrtl::createGrandCentralSignalMappingsPass() {
  return std::make_unique<GrandCentralSignalMappingsPass>();
}
