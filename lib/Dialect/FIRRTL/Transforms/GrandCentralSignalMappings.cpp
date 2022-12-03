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
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotationHelper.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/Path.h"

#define DEBUG_TYPE "gct"

namespace json = llvm::json;

using namespace circt;
using namespace firrtl;

//===- Terminology and useful information ---------------------------------===//
//
// For SignalDriver annotations, "local" refers to the subcircuit that
// drives/probes values in the main "remote" circuit.
//
// Driven values ("sinks"), are set via 'force XMR = input_port;' statements.
// Probes ("sources"), are read via 'assign output_port = XMR;' statements.
//
// The end goal is a mappings module that is generated and instantiated in the
// subcircuit which interacts with targeted values in the main circuit.
//
//===- Flow ---------------------------------------------------------------===//
//
// Signal driver annotations are needed when processing both circuits:
//
// First the original annotations are used with the main circuit, in order to
// emit extra wires around forced ports and to emit updated annotations
// pointing to the values across naming or hierarchy changes performed.
//
// Second these updated annotations are used when processing the subcircuit
// which is when the mappings module is created and instantiated.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Per-Module Signal Mappings
//===----------------------------------------------------------------------===//

namespace {
enum class MappingDirection {
  /// The local signal forces the remote signal.
  DriveRemote, // Sink
  /// The local signal is driven to the value of the remote signal through a
  /// connect.
  ProbeRemote, // Source
};

/// The information necessary to connect a local signal in a module to a remote
/// value (in a different module and/or circuit).
struct SignalMapping {
  /// Whether we force or read from the target.
  MappingDirection dir;
  /// The reference target of the thing we are forcing or probing.
  StringAttr remoteTarget;
  /// The type of the signal being mapped.
  FIRRTLBaseType type;
  /// The block argument or result that interacts with the remote target, either
  /// by forcing it or by reading from it through a connect.
  Value localValue;
  /// The name of the local value, for reuse in the generated signal mappings
  /// module.
  StringAttr localName;
  /// Which "side" we're on
  bool isLocal;
  /// NLA, if present
  FlatSymbolRefAttr nlaSym;
};

/// A helper structure that collects the data necessary to generate the signal
/// mappings module for an existing `FModuleOp` in the IR.
struct ModuleSignalMappings {
  ModuleSignalMappings(FModuleOp module) : module(module) {}
  void run();
  void addTarget(Value value, Annotation anno);
  FModuleOp emitMappingsModule();
  void instantiateMappingsModule(FModuleOp mappingsModule);

  FModuleOp module;
  bool allAnalysesPreserved = true;
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

//===----------------------------------------------------------------------===//
// Code related to annotation handling
//===----------------------------------------------------------------------===//

LogicalResult circt::firrtl::applyGCTSignalMappings(const AnnoPathValue &target,
                                                    DictionaryAttr anno,
                                                    ApplyState &state) {
  auto id = state.newID();
  auto *context = state.circuit.getContext();
  auto loc = state.circuit.getLoc();

  // Rework the circuit-level annotation to no longer include the
  // information we are scattering away anyway.
  NamedAttrList fields;
  auto annotationsAttr = tryGetAs<ArrayAttr>(anno, anno, "annotations", loc,
                                             signalDriverAnnoClass);
  auto circuitAttr =
      tryGetAs<StringAttr>(anno, anno, "circuit", loc, signalDriverAnnoClass);
  auto circuitPackageAttr = tryGetAs<StringAttr>(anno, anno, "circuitPackage",
                                                 loc, signalDriverAnnoClass);
  if (!annotationsAttr || !circuitAttr || !circuitPackageAttr)
    return failure();
  fields.append("class", StringAttr::get(context, signalDriverAnnoClass));
  fields.append("id", id);
  fields.append("annotations", annotationsAttr);
  fields.append("circuit", circuitAttr);
  fields.append("circuitPackage", circuitPackageAttr);

  // Handle the source and sink targets.
  auto sourcesAttr = tryGetAs<ArrayAttr>(anno, anno, "sourceTargets", loc,
                                         signalDriverAnnoClass);
  auto sinksAttr = tryGetAs<ArrayAttr>(anno, anno, "sinkTargets", loc,
                                       signalDriverAnnoClass);
  if (!sourcesAttr || !sinksAttr)
    return failure();

  // Add field indicating if we're the subcircuit or not.  Do this by looking at
  // the first source or sink and seeing what its circuit target is.
  DictionaryAttr firstSourceOrSink;
  if (!sourcesAttr.empty())
    firstSourceOrSink = sourcesAttr.begin()->dyn_cast<DictionaryAttr>();
  else if (!sinksAttr.empty())
    firstSourceOrSink = sinksAttr.begin()->dyn_cast<DictionaryAttr>();
  else
    return success();
  if (!firstSourceOrSink)
    return failure();
  auto prefix = tryGetAs<StringAttr>(firstSourceOrSink, anno, "_1", loc,
                                     signalDriverAnnoClass)
                    .getValue();
  bool isSubCircuit = !(prefix.consume_front("~") &&
                        prefix.consume_front(state.circuit.getName()) &&
                        prefix.consume_front("|"));
  fields.append("isSubCircuit", BoolAttr::get(context, isSubCircuit));

  auto dontTouch = [&](StringRef targetStr) {
    NamedAttrList anno;
    anno.append("class", StringAttr::get(context, dontTouchAnnoClass));
    anno.append("target", StringAttr::get(context, targetStr));
    state.addToWorklistFn(DictionaryAttr::get(context, anno));
  };

  StringSet<> modules;
  auto buildSourceOrSinkAnno = [&](Attribute attr,
                                   bool isSource) -> DictionaryAttr {
    auto dict = attr.dyn_cast<DictionaryAttr>();
    if (!dict)
      return {};
    NamedAttrList fields;
    fields.append("class",
                  StringAttr::get(context, signalDriverTargetAnnoClass));
    StringAttr targetAttr = tryGetAs<StringAttr>(
        dict, dict, isSubCircuit ? "_2" : "_1", loc, signalDriverAnnoClass);
    StringAttr peer = tryGetAs<StringAttr>(
        dict, dict, isSubCircuit ? "_1" : "_2", loc, signalDriverAnnoClass);
    if (!targetAttr || !peer)
      return {};

    auto target = tokenizePath(targetAttr);
    if (!target)
      return {};
    target->component.clear();
    target->name = "";
    modules.insert(target->str());

    auto targetId = state.newID();
    fields.append("dir",
                  StringAttr::get(context, isSource ? "source" : "sink"));
    fields.append("id", id);
    fields.append("peer", peer);
    fields.append("side",
                  StringAttr::get(context, isSubCircuit ? "local" : "remote"));
    fields.append("target", targetAttr);
    fields.append("targetId", targetId);

    // TODO: Always applying DontTouchAnnotation to a target is too restrictive.
    // This should only be necessary if this is a sink (which will produce a
    // force) and we are in the circuit where the force will be applied.  Relax
    // this restriction in the future.
    dontTouch(targetAttr.getValue());

    return DictionaryAttr::get(context, fields);
  };

  // Add any sinks or sources to the worklist for later processing.
  for (auto attr : sourcesAttr) {
    auto sourceAnno = buildSourceOrSinkAnno(attr, true);
    if (!sourceAnno)
      return failure();
    state.addToWorklistFn(sourceAnno);
  }
  for (auto attr : sinksAttr) {
    auto sinkAnno = buildSourceOrSinkAnno(attr, false);
    if (!sinkAnno)
      return failure();
    state.addToWorklistFn(sinkAnno);
  }

  AnnotationSet annotations(state.circuit);
  annotations.addAnnotations({DictionaryAttr::get(context, fields)});
  annotations.applyToOperation(state.circuit);

  for (auto &module : modules) {
    NamedAttrList fields;
    fields.append("class",
                  StringAttr::get(context, signalDriverModuleAnnoClass));
    fields.append("id", id);
    fields.append("target", StringAttr::get(context, module.getKey()));
    state.addToWorklistFn(DictionaryAttr::get(context, fields));
  }

  return success();
}

/// Analyze the `module` of this `ModuleSignalMappings` and generate the
/// corresponding auxiliary `FModuleOp` with the necessary cross-module
/// references and `ForceOp`s to probe and drive remote signals. This is
/// dictated by the presence of `SignalDriverAnnotation.module` on the module
/// and individual operations inside it. Anything targeted by a source/sink will
/// show up with a `SignalDriverAnnotation.target`. Global circuit-level
/// information lives in a circuit-level `SignalDriverAnnotation`.
///
/// If the module is the "remote" (main) circuit, gather those mappings for use
/// handling forced ports and creating updated mappings.
void ModuleSignalMappings::run() {
  // Check whether this module has any `SignalDriverAnnotation.module`s. These
  // indicate whether the module contains any operations with such annotations
  // and requires processing.
  if (!AnnotationSet::removeAnnotations(module, signalDriverModuleAnnoClass)) {
    LLVM_DEBUG(llvm::dbgs() << "Skipping `" << module.getName()
                            << "` (has no annotations)\n");
    return;
  }
  LLVM_DEBUG(llvm::dbgs() << "Running on module `" << module.getName()
                          << "`\n");

  // Gather the `SignalDriverAnnotation.target`s on the ports of this module.
  LLVM_DEBUG(llvm::dbgs() << "- Gather port annotations\n");
  AnnotationSet::removePortAnnotations(
      module, [&](unsigned i, Annotation anno) {
        if (!anno.isClass(signalDriverTargetAnnoClass))
          return false;
        addTarget(module.getArgument(i), anno);
        return true;
      });

  // Gather the `SignalDriverAnnotation.target`s of the operations within this
  // module.
  LLVM_DEBUG(llvm::dbgs() << "- Gather operation annotations\n");
  module.walk([&](Operation *op) {
    AnnotationSet::removeAnnotations(op, [&](Annotation anno) {
      if (!anno.isClass(signalDriverTargetAnnoClass))
        return false;
      for (auto result : op->getResults())
        addTarget(result, anno);
      return true;
    });
  });

  auto localMappings =
      llvm::make_filter_range(mappings, [](auto &m) { return m.isLocal; });

  // Remove connections to sources in the subcircuit.  This is done to cleanup
  // invalidations that occur from the Chisel API of Grand Central.
  // Only the remote value "source" should be connected to local source values.
  for (auto mapping : localMappings)
    if (mapping.dir == MappingDirection::ProbeRemote) {
      for (auto &use : llvm::make_early_inc_range(mapping.localValue.getUses()))
        if (auto connect = dyn_cast<FConnectLike>(use.getOwner()))
          if (connect.getDest() == mapping.localValue) {
            connect.erase();
            allAnalysesPreserved = false;
          };
    }

  // If there aren't any local-side (subcircuit-side) mappings, we're done.
  if (localMappings.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "Skipping `" << module.getName()
                            << "` (has no non-zero sources or sinks)\n");
    return;
  }

  // Changes are coming, mark analyses not preserved
  allAnalysesPreserved = false;

  if (!llvm::all_of(mappings, [](auto &m) { return m.isLocal; })) {
    emitWarning(module->getLoc(),
                "both remote and local SignalDriverAnnotation mappings found");
  }

  // Pick a name for the module that implements the signal mappings.
  CircuitNamespace circuitNamespace(module->getParentOfType<CircuitOp>());
  mappingsModuleName =
      circuitNamespace.newName(Twine(module.getName()) + "_signal_mappings");

  // Generate the mappings module.
  auto mappingsModule = emitMappingsModule();

  // Instantiate the mappings module.
  instantiateMappingsModule(mappingsModule);
}

/// Mark a `value` inside the `module` as being the target of the
/// `SignalDriverAnnotation` `anno`. This generates the necessary
/// `SignalMapping` information and adds an entry to the `mappings` array, to be
/// later consumed when the mappings module is constructed.
void ModuleSignalMappings::addTarget(Value value, Annotation anno) {
  // Ignore the target if it is zero width or not a base type.
  auto targetType = value.getType().dyn_cast<FIRRTLBaseType>();
  if (!targetType || !targetType.getBitWidthOrSentinel())
    return;

  SignalMapping mapping;
  mapping.dir = anno.getMember<StringAttr>("dir").getValue() == "source"
                    ? MappingDirection::ProbeRemote
                    : MappingDirection::DriveRemote;
  mapping.remoteTarget = anno.getMember<StringAttr>("peer");
  mapping.localValue = value;
  mapping.type = targetType;
  mapping.isLocal = anno.getMember<StringAttr>("side").getValue() == "local";
  mapping.nlaSym = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal");

  // Guess a name for the local value. This is only for readability's sake,
  // giving the pass a hint for picking the names of the generated module ports.
  if (auto blockArg = value.dyn_cast<BlockArgument>()) {
    mapping.localName = module.getPortNameAttr(blockArg.getArgNumber());
  } else if (auto *op = value.getDefiningOp()) {
    mapping.localName = op->getAttrOfType<StringAttr>("name");
  }

  LLVM_DEBUG(llvm::dbgs() << "  - " << mapping << "\n");
  mappings.push_back(std::move(mapping));
}

/// Create a separate mappings module that contains cross-module references and
/// `ForceOp`s for each entry in the `mappings` array.
FModuleOp ModuleSignalMappings::emitMappingsModule() {
  LLVM_DEBUG(llvm::dbgs() << "- Generating `" << mappingsModuleName << "`\n");

  // Determine what ports this module will have, given the signal mappings we
  // are supposed to emit.
  SmallVector<PortInfo> ports;
  auto localMappings =
      llvm::make_filter_range(mappings, [](auto &m) { return m.isLocal; });
  for (auto &mapping : localMappings) {
    ports.push_back(PortInfo{mapping.localName,
                             mapping.type,
                             mapping.dir == MappingDirection::DriveRemote
                                 ? Direction::In
                                 : Direction::Out,
                             {},
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
  for (auto &mapping : localMappings) {
    // TODO: Actually generate a proper XMR here. For now just do some textual
    // replacements. Generating a real IR node (like a proper XMR op) would be
    // much better, but the modules that `EmitSignalMappings` interacts with
    // generally live in a separate circuit. Multiple circuits are not fully
    // supported at the moment.
    auto tokenized = tokenizePath(mapping.remoteTarget);
    if (!tokenized) {
      emitWarning(module.getLoc(),
                  llvm::formatv("unable to parse remote target '{0}' found in "
                                "SignalDriverAnnotation, skipping",
                                mapping.remoteTarget));
      continue;
    }

    SmallString<32> remoteXmrName;
    if (tokenized->instances.empty()) {
      // If no instance path, just use module directly
      remoteXmrName = tokenized->module;
    } else {
      // First module encountered is our starting point
      auto &[firstMod, _] = tokenized->instances.front();
      remoteXmrName = firstMod;
      for (auto &[mod, inst] : tokenized->instances) {
        remoteXmrName.push_back('.');
        remoteXmrName += inst;
      }
    }
    assert(!remoteXmrName.empty());

    remoteXmrName.push_back('.');
    remoteXmrName += tokenized->name;

    // TODO: Actually index into vectors/bundles?
    // For now, convert to expected name of expanded aggregate
    for (auto &comp : tokenized->component) {
      remoteXmrName.push_back('_');
      remoteXmrName += comp.name;
    }

    // llvm::errs() << "XMR: " << remoteXmrName << "\n\n";
    if (mapping.dir == MappingDirection::DriveRemote) {
      auto xmr = builder.create<VerbatimWireOp>(mapping.type, remoteXmrName);
      builder.create<ForceOp>(xmr, mappingsModule.getArgument(portIdx++));
    } else {
      auto xmr = builder.create<VerbatimWireOp>(mapping.type, remoteXmrName);
      builder.create<ConnectOp>(mappingsModule.getArgument(portIdx++), xmr);
    }
  }
  return mappingsModule;
}

/// Instantiate the generated mappings module inside the `module` we are working
/// on, and generated the necessary connections to local signals.
void ModuleSignalMappings::instantiateMappingsModule(FModuleOp mappingsModule) {
  LLVM_DEBUG(llvm::dbgs() << "- Instantiating `" << mappingsModuleName
                          << "`\n");
  // Create the actual module.
  auto builder =
      ImplicitLocOpBuilder::atBlockEnd(module.getLoc(), module.getBodyBlock());
  auto inst = builder.create<InstanceOp>(mappingsModule, "signal_mappings");

  // Generate the connections to and from the instance.
  unsigned portIdx = 0;
  auto localMappings =
      llvm::make_filter_range(mappings, [](auto &m) { return m.isLocal; });
  for (auto &mapping : localMappings) {
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

/// External modules, for emitting when processing subcircuit
struct ExtModules {
  /// External modules put in the vsrcs field of the JSON.
  SmallVector<FExtModuleOp> vsrc;
  /// External modules put in the "load_jsons" field of the JSON.
  SmallVector<FExtModuleOp> json;
};

/// Information gathered about mappings, used for identifying forced ports and
/// generating updated mappings for the remote side (main circuit).
struct ModuleMappingResult {
  /// Were changes made that invalidate analyses?
  bool allAnalysesPreserved;
  /// Module scanned
  FModuleOp module;
  /// Signal mappings whose remote targets are in this circuit
  SmallVector<SignalMapping> remoteMappings;
};

class GrandCentralSignalMappingsPass
    : public GrandCentralSignalMappingsBase<GrandCentralSignalMappingsPass> {

  /// Generate operation to output list of external modules
  /// Used when processing the sub- (local) circuit.
  void emitSubCircuitJSON(CircuitOp circuit, StringAttr circuitPackage,
                          StringRef outputFilename,
                          const ExtModules &extmodules);

  /// Fixup forced ports and emit updated mappings.
  /// Used when processing the main (remote) circuit.
  FailureOr<bool>
  emitUpdatedMappings(CircuitOp circuit, StringAttr circuitPackage,
                      MutableArrayRef<ModuleMappingResult> result);

  void runOnOperation() override;

public:
  std::string outputFilename;
};

void GrandCentralSignalMappingsPass::runOnOperation() {
  CircuitOp circuit = getOperation();

  StringAttr circuitPackage;
  BoolAttr isSubCircuitAttr;
  bool circuitHasAnno =
      AnnotationSet::removeAnnotations(circuit, [&](Annotation anno) {
        if (!anno.isClass(signalDriverAnnoClass))
          return false;

        isSubCircuitAttr = anno.getMember<BoolAttr>("isSubCircuit");
        circuitPackage = anno.getMember<StringAttr>("circuitPackage");
        return true;
      });

  if (!circuitHasAnno) {
    // Nothing to do here
    markAllAnalysesPreserved();
    return;
  }

  if (!isSubCircuitAttr) {
    emitError(
        circuit->getLoc(),
        "has invalid SignalDriverAnnotation (missing 'isSubCircuit' field')");
    return signalPassFailure();
  }
  bool isSubCircuit = isSubCircuitAttr.getValue();

  if (isSubCircuit && !circuitPackage) {
    emitError(circuit->getLoc())
        << "has invalid SignalDriverAnnotation (subcircuit JSON emission is "
           "enabled, but no circuitPackage was provided)";
    return signalPassFailure();
  }

  SmallVector<FModuleOp> modules;
  ExtModules extmodules;

  for (auto op : circuit.getBody().getOps<FModuleLike>()) {
    if (auto *extModule = dyn_cast<FExtModuleOp>(&op)) {
      AnnotationSet annotations(*extModule);
      if (annotations.hasAnnotation(blackBoxInlineAnnoClass)) {
        extmodules.vsrc.push_back(*extModule);
        continue;
      }
      extmodules.json.push_back(*extModule);
    } else if (auto *module = dyn_cast<FModuleOp>(&op)) {
      modules.push_back(*module);
    }
  }

  // Pre-allocate result vector
  SmallVector<ModuleMappingResult> results;
  results.resize(modules.size());

  auto updateStats = [this](auto &mappings) {
    size_t sources = 0, sinks = 0;
    for (auto &mapping : mappings) {
      if (mapping.dir == MappingDirection::ProbeRemote)
        ++sources;
      else
        ++sinks;
    }
    if (sources)
      numSources += sources;
    if (sinks)
      numSinks += sinks;
    if (sources || sinks)
      ++numModules;
  };

  mlir::parallelFor(
      circuit.getContext(), 0, modules.size(),
      [&modules, &results, &updateStats](size_t index) {
        ModuleSignalMappings mapper(modules[index]);
        mapper.run();

        // Update statistics
        updateStats(mapper.mappings);

        results[index] = {
            mapper.allAnalysesPreserved, modules[index],
            SmallVector<SignalMapping>{llvm::make_filter_range(
                mapper.mappings, [](auto &m) { return !m.isLocal; })}};
      });
  bool allAnalysesPreserved =
      llvm::all_of(results, [](auto &r) { return r.allAnalysesPreserved; });

  // If this is a subcircuit, then emit JSON information necessary to drive
  // SiFive tools.
  // Otherwise handle any forced ports and emit updated mappings.
  if (isSubCircuit) {
    emitSubCircuitJSON(circuit, circuitPackage, outputFilename, extmodules);
  } else {
    auto result = emitUpdatedMappings(circuit, circuitPackage, results);
    if (failed(result))
      return signalPassFailure();
    allAnalysesPreserved &= *result;
  }

  if (allAnalysesPreserved)
    markAllAnalysesPreserved();
}

void GrandCentralSignalMappingsPass::emitSubCircuitJSON(
    CircuitOp circuit, StringAttr circuitPackage, StringRef outputFilename,
    const ExtModules &extmodules) {
  std::string jsonString;
  llvm::raw_string_ostream jsonStream(jsonString);
  json::OStream j(jsonStream, 2);
  j.object([&] {
    j.attributeObject("vendor", [&]() {
      j.attributeObject("vcs", [&]() {
        j.attributeArray("vsrcs", [&]() {
          for (FModuleOp module : circuit.getBody().getOps<FModuleOp>()) {
            SmallVector<char> file(outputFilename.begin(),
                                   outputFilename.end());
            llvm::sys::fs::make_absolute(file);
            llvm::sys::path::append(file, Twine(module.moduleName()) + ".sv");
            j.value(file);
          }
          for (FExtModuleOp ext : extmodules.vsrc) {
            SmallVector<char> file(outputFilename.begin(),
                                   outputFilename.end());
            llvm::sys::fs::make_absolute(file);
            llvm::sys::path::append(file, Twine(ext.moduleName()) + ".sv");
            j.value(file);
          }
        });
      });
      j.attributeObject("verilator", [&]() {
        j.attributeArray("error", [&]() {
          j.value("force statement is not supported in verilator");
        });
      });
    });
    j.attributeArray("remove_vsrcs", []() {});
    j.attributeArray("vsrcs", []() {});
    j.attributeArray("load_jsons", [&]() {
      for (FExtModuleOp extModule : extmodules.json)
        j.value((Twine(extModule.moduleName()) + ".json").str());
    });
  });
  auto b = OpBuilder::atBlockEnd(circuit.getBodyBlock());
  auto jsonOp = b.create<sv::VerbatimOp>(b.getUnknownLoc(), jsonString);
  jsonOp->setAttr(
      "output_file",
      hw::OutputFileAttr::getFromFilename(
          b.getContext(), Twine(circuitPackage) + ".subcircuit.json", true));
}

FailureOr<bool> GrandCentralSignalMappingsPass::emitUpdatedMappings(
    CircuitOp circuit, StringAttr circuitPackage,
    MutableArrayRef<ModuleMappingResult> results) {
  bool allAnalysesPreserved = true;

  // 1) Sanity checking
  // 2) Fixup ports that are driven from subcircuit
  // 3) Generate new remote references
  // 4) Emit updated mappings as SignalDriverAnnotations
  //    for consumption when processing the subcircuit
  //    (as replacements for the original annotations)

  // Helper to instantiate module namespaces as-needed
  DenseMap<Operation *, ModuleNamespace> moduleNamespaces;
  auto getModuleNamespace =
      [&moduleNamespaces](Operation *module) -> ModuleNamespace & {
    return moduleNamespaces.try_emplace(module, module).first->second;
  };

  // (1) Scan mappings for unexpected or unsupported values
  for (auto const &[_, mod, mappings] : results) {
    for (auto &mapping : mappings) {
      // No local-side mappings on main circuit side
      if (mapping.isLocal) {
        emitError(mapping.localValue.getLoc())
            << "local-side SignalDriverAnnotation found in main circuit, "
               "mixing not supported";
        return failure();
      }
    }
  }

  // (2) Find ports that are sinks, insert buffer wires to break the net.
  // Input ports are buffered at instantiation points, outputs in the module.
  auto *instanceGraph = &getAnalysis<InstanceGraph>();
  llvm::SmallVector<unsigned, 32> forcedInputPorts;
  llvm::SmallVector<unsigned, 32> forcedOutputPorts;
  for (auto &[_, mod, mappings] : results) {
    // Walk mappings looking for module ports that are being driven,
    // and gather their indices for fixing up.
    forcedInputPorts.clear();
    forcedOutputPorts.clear();
    for (auto &mapping : mappings) {
      if (mapping.dir == MappingDirection::DriveRemote /* Sink */) {
        if (auto blockArg = mapping.localValue.dyn_cast<BlockArgument>()) {
          auto portIdx = blockArg.getArgNumber();
          // Port may be driven multiple times along different instance paths
          if (mod.getPortDirection(portIdx) == Direction::In) {
            forcedInputPorts.push_back(portIdx);
          } else {
            forcedOutputPorts.push_back(portIdx);
          }
        }
      }
    }
    llvm::sort(forcedInputPorts);
    forcedInputPorts.erase(
        std::unique(forcedInputPorts.begin(), forcedInputPorts.end()),
        forcedInputPorts.end());
    llvm::sort(forcedOutputPorts);
    forcedOutputPorts.erase(
        std::unique(forcedOutputPorts.begin(), forcedOutputPorts.end()),
        forcedOutputPorts.end());

    // Replace uses of each driven port with a wire that's connected to a wire
    // that is connected to the port. This is done to cause an 'assign' to be
    // created, disconnecting the forced port's net from its uses.
    auto breakNet = [modName = mod.moduleName()](
                        OpBuilder &builder, Value port,
                        ModuleNamespace &moduleNamespace, StringRef portName) {
      // Create chain like:
      // port_result <= foo_dataIn_x_buffer
      // foo_dataIn_x_buffer <= foo_dataIn_x
      auto replacementWireName = builder.getStringAttr(
          moduleNamespace.newName(modName + "_" + portName));
      auto bufferWireName = builder.getStringAttr(
          moduleNamespace.newName(replacementWireName.getValue() + "_buffer"));
      auto bufferWire =
          builder.create<WireOp>(builder.getUnknownLoc(), port.getType(),
                                 bufferWireName, NameKindEnum::DroppableName,
                                 builder.getArrayAttr({}), bufferWireName);
      auto replacementWire = builder.create<WireOp>(
          builder.getUnknownLoc(), port.getType(), replacementWireName,
          NameKindEnum::DroppableName, builder.getArrayAttr({}),
          replacementWireName);
      port.replaceAllUsesWith(replacementWire);
      AnnotationSet::addDontTouch(replacementWire);
      AnnotationSet::addDontTouch(bufferWire);
      builder.create<StrictConnectOp>(builder.getUnknownLoc(), port,
                                      bufferWire);
      builder.create<StrictConnectOp>(builder.getUnknownLoc(), bufferWire,
                                      replacementWire);
    };

    // Update the forced ports statistics.
    numForcedInputPorts += forcedInputPorts.size();
    numForcedOutputPorts += forcedOutputPorts.size();

    if (!forcedOutputPorts.empty()) {
      ModuleNamespace &moduleNamespace = getModuleNamespace(mod);
      auto builder = OpBuilder::atBlockBegin(mod.getBodyBlock());

      // Update statistic
      numBufferWirePairsAdded += forcedOutputPorts.size();

      for (auto portIdx : forcedOutputPorts) {
        auto port = mod.getArgument(portIdx);
        breakNet(builder, port, moduleNamespace, mod.getPortName(portIdx));
      }
    }

    if (forcedInputPorts.empty())
      continue;

    for (auto *use : instanceGraph->lookup(mod)->uses()) {
      allAnalysesPreserved = false;

      auto inst = use->getInstance();
      OpBuilder builder(inst.getContext());
      builder.setInsertionPointAfter(inst);
      auto parentModule = inst->getParentOfType<FModuleOp>();
      ModuleNamespace &moduleNamespace = getModuleNamespace(parentModule);

      // Update statistic
      numBufferWirePairsAdded += forcedInputPorts.size();

      for (auto portIdx : forcedInputPorts) {
        auto port = inst->getResult(portIdx);
        breakNet(builder, port, moduleNamespace, mod.getPortName(portIdx));
      }
    }
  }

  // (3) Generate updated remote references

  // For instance path references, determine where to start.
  // This is either the main module or the DUT if any.
  auto dut = circuit.getMainModule();
  for (auto op : circuit.getBody().getOps<FModuleOp>())
    if (AnnotationSet(op).hasAnnotation(dutAnnoClass)) {
      dut = op;
      break;
    }

  // Helpers to emit XMR's for the mappings.
  SmallVector<Attribute> symbols;
  SmallDenseMap<Attribute, size_t> symMap;

  auto getSymIdx = [&](Attribute symbol) {
    auto it = symMap.find(symbol);
    if (it != symMap.end())
      return it->second;

    auto id = symbols.size();
    symbols.push_back(symbol);
    symMap.insert({symbol, id});
    return id;
  };
  auto mkRef = [&](FModuleOp module,
                   const SignalMapping &mapping) -> std::string {
    // Generate placeholder into specified storage, return StringRef for it
    auto mkSymPlaceholder = [&](Attribute symbol, auto &out) -> StringRef {
      ("{{" + Twine(getSymIdx(symbol)) + "}}").toVector(out);
      return out;
    };

    // Construct target using placeholders:
    SmallString<32> circuitStr, moduleStr, nameStr;
    // Storage for <mod,inst> strings along instance path
    SmallVector<std::pair<SmallString<8>, SmallString<8>>, 16> stringStorage;
    TokenAnnoTarget target;
    target.circuit = mkSymPlaceholder(FlatSymbolRefAttr::get(dut), circuitStr);
    target.module = mkSymPlaceholder(FlatSymbolRefAttr::get(module), moduleStr);
    target.component = {};

    if (auto port = mapping.localValue.dyn_cast<BlockArgument>())
      target.name = mkSymPlaceholder(
          ::getInnerRefTo(module, port.getArgNumber(), mapping.localName,
                          getModuleNamespace),
          nameStr);
    else
      target.name =
          mkSymPlaceholder(::getInnerRefTo(mapping.localValue.getDefiningOp(),
                                           "", getModuleNamespace),
                           nameStr);

    // If there's an NLA, add instance path information.
    if (mapping.nlaSym) {
      auto nla =
          cast<HierPathOp>(circuit.lookupSymbol(mapping.nlaSym.getAttr()));
      assert(!nla.getNamepath().empty());

      // Start from root of NLA, or from top/DUT if through it
      bool seenRoot = false;
      bool usesTop = nla.hasModule(dut.moduleNameAttr());
      ArrayRef<Attribute> path = nla.getNamepath().getValue();
      stringStorage.resize(path.drop_back().size());
      for (const auto &attr : llvm::enumerate(path.drop_back())) {
        auto ref = attr.value().cast<hw::InnerRefAttr>();
        if (usesTop && !seenRoot) {
          if (ref.getModule() == dut.moduleNameAttr())
            seenRoot = true;
        }

        if (!usesTop || seenRoot) {
          auto &store = stringStorage[attr.index()];
          auto modStr = mkSymPlaceholder(ref.getModuleRef(), store.first);
          auto instStr = mkSymPlaceholder(
              hw::InnerRefAttr::get(ref.getModule(), ref.getName()),
              store.second);
          target.instances.emplace_back(modStr, instStr);
        }
      };
    }
    return target.str();
  };

  // Generate and sort new mappings for (better) output stability
  SmallVector<std::pair<std::string, StringRef>, 32> sinks, sources;
  auto addUpdatedMappings = [&](MappingDirection filterDir, auto &vec) {
    for (auto const &[_, mod, mappings] : results)
      for (auto &mapping : mappings)
        if (mapping.dir == filterDir)
          vec.emplace_back(mkRef(mod, mapping),
                           mapping.remoteTarget.getValue());
  };
  addUpdatedMappings(MappingDirection::DriveRemote, sinks);
  addUpdatedMappings(MappingDirection::ProbeRemote, sources);
  // Sort on local targets only, remote may contain substitution placeholders
  llvm::sort(sinks, llvm::less_second());
  llvm::sort(sources, llvm::less_second());

  // (4) Emit updated mappings for consumption by subcircuit
  std::string jsonString;
  llvm::raw_string_ostream jsonStream(jsonString);
  json::OStream j(jsonStream, 2);

  j.array([&] {
    j.object([&] {
      j.attribute("class", signalDriverAnnoClass);
      j.attributeArray("sinkTargets", [&]() {
        for (auto &[remote, local] : sinks) {
          j.objectBegin();
          j.attribute("_1", remote);
          j.attribute("_2", local);
          j.objectEnd();
        };
      });
      j.attributeArray("sourceTargets", [&]() {
        for (auto &[remote, local] : sources) {
          j.objectBegin();
          j.attribute("_1", remote);
          j.attribute("_2", local);
          j.objectEnd();
        };
      });
      // Emit these but don't attempt to plumb through their original values
      j.attribute("circuit", "circuit empty :\n  module empty :\n\n    skip\n");
      j.attributeArray("annotations", [&]() {});
      if (circuitPackage)
        j.attribute("circuitPackage", circuitPackage.getValue());
    });
  });
  auto b = OpBuilder::atBlockEnd(circuit.getBodyBlock());
  auto jsonOp = b.create<sv::VerbatimOp>(b.getUnknownLoc(), jsonString,
                                         ValueRange{}, b.getArrayAttr(symbols));
  // TODO: plumb option to specify path/name?
  constexpr char jsonOutFile[] = "sigdrive.json";
  jsonOp->setAttr("output_file", hw::OutputFileAttr::getFromFilename(
                                     b.getContext(), jsonOutFile, true));

  return allAnalysesPreserved;
}

std::unique_ptr<mlir::Pass>
circt::firrtl::createGrandCentralSignalMappingsPass(StringRef outputFilename) {
  auto pass = std::make_unique<GrandCentralSignalMappingsPass>();
  if (!outputFilename.empty())
    pass->outputFilename = outputFilename;
  return pass;
}
