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
// Traits
//===----------------------------------------------------------------------===//

namespace llvm {
template <>
struct DenseMapInfo<StringAttr> {
  static inline StringAttr getEmptyKey() {
    auto pointer = DenseMapInfo<void *>::getEmptyKey();
    return StringAttr(static_cast<Attribute::ImplType *>(pointer));
  }
  static inline StringAttr getTombstoneKey() {
    auto pointer = DenseMapInfo<void *>::getTombstoneKey();
    return StringAttr(static_cast<Attribute::ImplType *>(pointer));
  }
  static unsigned getHashValue(const StringAttr &val) {
    return mlir::hash_value(val);
  }
  static bool isEqual(const StringAttr &lhs, const StringAttr &rhs) {
    return lhs == rhs;
  }
};
} // end namespace llvm

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
  StringRef localName;
  StringRef localFields;

  /// The type of the signal being mapped.
  FIRRTLType type;
  /// The `localName` and `localFields` resolved to an actual value.
  Value localValue;
};

struct ModuleSignalMappings {
  ModuleSignalMappings(FModuleOp module) : module(module) {}
  void run();
  void addTargets(ArrayAttr targets, MappingDirection dir);
  void emitMappingsModule();
  void instantiateMappingsModule();

  FModuleOp module;
  bool anyFailed = false;
  SmallVector<SignalMapping> mappings;
  SmallString<64> mappingsModuleName;

  /// A lookup table of local operations that carry a name which can be
  /// referenced in an annotation.
  DenseMap<StringRef, Value> localNames;
};
} // namespace

template <typename T>
static T &operator<<(T &os, const SignalMapping &mapping) {
  os << "SignalMapping { remote"
     << (mapping.dir == MappingDirection::DriveRemote ? "Sink" : "Source")
     << ": " << mapping.remoteTarget << ", "
     << "localTarget: \"" << mapping.localName << "\" }";
  return os;
}

void ModuleSignalMappings::run() {
  LLVM_DEBUG(llvm::dbgs() << "Running on module `" << module.getName()
                          << "`\n");

  // Gather the signal driver annotations on this module, and keep track of the
  // target within the module.
  LLVM_DEBUG(llvm::dbgs() << "- Gather annotations\n");
  AnnotationSet::removeAnnotations(module, [&](Annotation anno) {
    if (!anno.isClass(signalDriverAnnoClass))
      return false;
    if (auto sinks = anno.getMember<ArrayAttr>("sinkTargets"))
      addTargets(sinks, MappingDirection::DriveRemote);
    if (auto sources = anno.getMember<ArrayAttr>("sourceTargets"))
      addTargets(sources, MappingDirection::ProbeRemote);
    return true;
  });
  if (anyFailed)
    return;

  // Nothing to do if there are no signal mappings.
  if (mappings.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "- No annotations, nothing left to do\n");
    return;
  }

  // Gather a name table to resolve local mappings.
  module.walk([&](Operation *op) {
    TypeSwitch<Operation *>(op).Case<WireOp, NodeOp, RegOp, RegResetOp>(
        [&](auto op) {
          localNames.insert({op.name(), op});
        });
  });

  // Gather the types and values for the signal mappings. This may incur
  // additional subfield/subfindex operations to be generated.
  ImplicitLocOpBuilder builder(module.getLoc(), module);
  builder.setInsertionPointToEnd(module.getBodyBlock());
  for (auto &mapping : mappings) {
    // Resolve the name locally.
    mapping.localValue = localNames.lookup(mapping.localName);
    if (!mapping.localValue) {
      module.emitError("unknown local target \"")
          << mapping.localName << "\" in SignalDriverAnnotation";
      anyFailed = true;
      return;
    }
    LLVM_DEBUG(llvm::dbgs() << "- Resolved `" << mapping.localName << "` to "
                            << mapping.localValue << "\n");

    // Resolve any subfield/subindex accesses.
    StringRef fields = mapping.localFields;
    while (!fields.empty()) {
      module.emitError(
          "subfield accesses not yet supported in SignalDriverAnnotation");
      anyFailed = true;
      return;
    }

    mapping.type = mapping.localValue.getType().cast<FIRRTLType>();
    // TODO: Resolve the remote value somehow.
  }

  // Pick a name for the module that implements the signal mappings.
  mappingsModuleName = module.getName();
  mappingsModuleName.append("_signal_mappings");

  // Generate the mappings module.
  emitMappingsModule();

  // Instantiate the mappings module.
  instantiateMappingsModule();
}

/// Add the entries of the `sinkTargets` or `sourceTargets` array in a
/// `SignalDriverAnnotation` to the list of mappings that need to be established
/// for this module.
void ModuleSignalMappings::addTargets(ArrayAttr targets, MappingDirection dir) {
  const char *dirString =
      dir == MappingDirection::DriveRemote ? "sink" : "source";
  for (auto target : targets.getAsRange<DictionaryAttr>()) {
    SignalMapping mapping;
    mapping.dir = dir;
    mapping.remoteTarget = target.getAs<StringAttr>("_1");
    if (!mapping.remoteTarget) {
      module.emitError("SignalDriverAnnotation ")
          << dirString << " target " << target << " missing \"_1\" attribute";
      anyFailed = true;
      continue;
    }
    auto localTarget = target.getAs<StringAttr>("_2");
    if (!localTarget) {
      module.emitError("SignalDriverAnnotation ")
          << dirString << " target " << target << " missing \"_2\" attribute";
      anyFailed = true;
      continue;
    }
    mapping.localName = localTarget.getValue();
    LLVM_DEBUG(llvm::dbgs() << "  - " << mapping << "\n");
    mappings.push_back(std::move(mapping));
  }
}

void ModuleSignalMappings::emitMappingsModule() {
  LLVM_DEBUG(llvm::dbgs() << "- Generating `" << mappingsModuleName << "`\n");
  auto *context = module.getContext();

  // Determine what ports this module will have, given the signal mappings we
  // are supposed to emit.
  SmallVector<ModulePortInfo> ports;
  for (auto &mapping : mappings) {
    // TODO: We really need the local operation this is targeting, to get at the
    // type and all the other good stuff.
    ports.push_back(ModulePortInfo{
        StringAttr::get(context, mapping.localName),
        // TODO: This needs to come from the local target:
        mapping.type,
        mapping.dir == MappingDirection::DriveRemote ? Direction::Input
                                                     : Direction::Output,
        module.getLoc()});
    LLVM_DEBUG(llvm::dbgs()
               << "  - Adding port `" << mapping.localName << "`\n");
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
    // replacements.
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
  ImplicitLocOpBuilder builder(module.getLoc(), module);
  builder.setInsertionPointToEnd(module.getBodyBlock());
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
  LLVM_DEBUG(llvm::dbgs() << "Running the GCT Signal Mappings pass\n");
  getOperation().walk([&](FModuleOp module) {
    ModuleSignalMappings mapper(module);
    mapper.run();
    if (mapper.anyFailed) {
      signalPassFailure();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
}

std::unique_ptr<mlir::Pass>
circt::firrtl::createGrandCentralSignalMappingsPass() {
  return std::make_unique<GrandCentralSignalMappingsPass>();
}
