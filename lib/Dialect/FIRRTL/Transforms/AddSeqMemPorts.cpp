//===- AddSeqMemPorts.cpp - Add extra ports to memories ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This file defines the AddSeqMemPorts pass.  This pass will add extra ports
// to memory modules.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Emit/EmitOps.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotationHelper.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/InnerSymbolNamespace.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Support/Namespace.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/Path.h"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_ADDSEQMEMPORTS
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

namespace {
struct AddSeqMemPortsPass
    : public circt::firrtl::impl::AddSeqMemPortsBase<AddSeqMemPortsPass> {
  void runOnOperation() override;
  LogicalResult processAddPortAnno(Location loc, Annotation anno);
  LogicalResult processFileAnno(Location loc, StringRef metadataDir,
                                Annotation anno);
  LogicalResult processAnnos(CircuitOp circuit);
  void createOutputFile(igraph::ModuleOpInterface module);
  InstanceGraphNode *findDUT();
  void processMemModule(FMemModuleOp mem);
  LogicalResult processModule(FModuleOp module, bool isDUT);

  /// Get the cached namespace for a module.
  hw::InnerSymbolNamespace &getModuleNamespace(FModuleLike module) {
    return moduleNamespaces.try_emplace(module, module).first->second;
  }

  /// Obtain an inner reference to an operation, possibly adding an `inner_sym`
  /// to that operation.
  hw::InnerRefAttr getInnerRefTo(Operation *op) {
    return ::getInnerRefTo(op,
                           [&](FModuleLike mod) -> hw::InnerSymbolNamespace & {
                             return getModuleNamespace(mod);
                           });
  }

  /// This represents the collected information of the memories in a module.
  struct MemoryInfo {
    /// The list of extra ports added to this module to support additional ports
    /// on all memories underneath this module.
    std::vector<std::pair<unsigned, PortInfo>> extraPorts;

    /// This is an ordered list of instance paths to all memories underneath
    /// this module. This will record each memory once for every port added,
    /// which is for some reason the format of the metadata file.
    std::vector<SmallVector<Attribute>> instancePaths;
  };

  CircuitNamespace circtNamespace;
  /// This maps a module to information about the memories.
  DenseMap<Operation *, MemoryInfo> memInfoMap;
  DenseMap<Attribute, Operation *> innerRefToInstanceMap;

  InstanceGraph *instanceGraph;

  /// If the metadata output file was specified in an annotation.
  StringAttr outputFile;

  /// This is the list of every port to be added to each sequential memory.
  SmallVector<PortInfo> userPorts;
  /// This is an attribute holding the metadata for extra ports.
  ArrayAttr extraPortsAttr;

  /// Cached module namespaces.
  DenseMap<Operation *, hw::InnerSymbolNamespace> moduleNamespaces;

  bool anythingChanged;
};
} // end anonymous namespace

LogicalResult AddSeqMemPortsPass::processAddPortAnno(Location loc,
                                                     Annotation anno) {
  auto name = anno.getMember<StringAttr>("name");
  if (!name)
    return emitError(
        loc, "AddSeqMemPortAnnotation requires field 'name' of string type");

  auto input = anno.getMember<BoolAttr>("input");
  if (!input)
    return emitError(
        loc, "AddSeqMemPortAnnotation requires field 'input' of boolean type");
  auto direction = input.getValue() ? Direction::In : Direction::Out;

  auto width = anno.getMember<IntegerAttr>("width");
  if (!width)
    return emitError(
        loc, "AddSeqMemPortAnnotation requires field 'width' of integer type");
  auto type = UIntType::get(&getContext(), width.getInt());
  userPorts.push_back({name, type, direction});
  return success();
}

LogicalResult AddSeqMemPortsPass::processFileAnno(Location loc,
                                                  StringRef metadataDir,
                                                  Annotation anno) {
  if (outputFile)
    return emitError(
        loc, "circuit has two AddSeqMemPortsFileAnnotation annotations");

  auto filename = anno.getMember<StringAttr>("filename");
  if (!filename)
    return emitError(loc,
                     "AddSeqMemPortsFileAnnotation requires field 'filename' "
                     "of string type");

  SmallString<128> outputFilePath(metadataDir);
  llvm::sys::path::append(outputFilePath, filename.getValue());
  outputFile = StringAttr::get(&getContext(), outputFilePath);
  return success();
}

LogicalResult AddSeqMemPortsPass::processAnnos(CircuitOp circuit) {
  auto loc = circuit.getLoc();

  // Find the metadata directory.
  auto dirAnno =
      AnnotationSet(circuit).getAnnotation(metadataDirectoryAttrName);
  StringRef metadataDir = "metadata";
  if (dirAnno) {
    auto dir = dirAnno.getMember<StringAttr>("dirname");
    if (!dir)
      return emitError(loc, "MetadataDirAnnotation requires field 'dirname' of "
                            "string type");
    metadataDir = dir.getValue();
  }

  // Remove the annotations we care about.
  bool error = false;
  AnnotationSet::removeAnnotations(circuit, [&](Annotation anno) {
    if (error)
      return false;
    if (anno.isClass(addSeqMemPortAnnoClass)) {
      error = failed(processAddPortAnno(loc, anno));
      return true;
    }
    if (anno.isClass(addSeqMemPortsFileAnnoClass)) {
      error = failed(processFileAnno(loc, metadataDir, anno));
      return true;
    }
    return false;
  });
  return failure(error);
}

InstanceGraphNode *AddSeqMemPortsPass::findDUT() {
  // Find the DUT module.
  for (auto *node : *instanceGraph) {
    if (AnnotationSet::hasAnnotation(node->getModule(), dutAnnoClass))
      return node;
  }
  return instanceGraph->getTopLevelNode();
}

void AddSeqMemPortsPass::processMemModule(FMemModuleOp mem) {
  // We have to add the user ports to every mem module.
  size_t portIndex = mem.getNumPorts();
  auto &memInfo = memInfoMap[mem];
  auto &extraPorts = memInfo.extraPorts;
  for (auto &p : userPorts)
    extraPorts.emplace_back(portIndex, p);
  mem.insertPorts(extraPorts);
  // Attach the extraPorts metadata.
  mem.setExtraPortsAttr(extraPortsAttr);
}

LogicalResult AddSeqMemPortsPass::processModule(FModuleOp module, bool isDUT) {
  auto *context = &getContext();
  // Insert the new port connections at the end of the module.
  auto builder = OpBuilder::atBlockEnd(module.getBodyBlock());
  auto &memInfo = memInfoMap[module];
  auto &extraPorts = memInfo.extraPorts;
  // List of ports added to submodules which must be connected to this module's
  // ports.
  SmallVector<Value> values;

  // The base index to use when adding ports to the current module.
  unsigned firstPortIndex = module.getNumPorts();

  for (auto &op : llvm::make_early_inc_range(*module.getBodyBlock())) {
    if (auto inst = dyn_cast<InstanceOp>(op)) {
      auto submodule = inst.getReferencedModule(*instanceGraph);

      auto subMemInfoIt = memInfoMap.find(submodule);
      // If there are no extra ports, we don't have to do anything.
      if (subMemInfoIt == memInfoMap.end() ||
          subMemInfoIt->second.extraPorts.empty())
        continue;
      auto &subMemInfo = subMemInfoIt->second;
      // Find out how many memory ports we have to add.
      auto &subExtraPorts = subMemInfo.extraPorts;

      // Add the extra ports to the instance operation.
      auto clone = inst.cloneAndInsertPorts(subExtraPorts);
      inst.replaceAllUsesWith(
          clone.getResults().drop_back(subExtraPorts.size()));
      instanceGraph->replaceInstance(inst, clone);
      inst->erase();
      inst = clone;

      // Connect each submodule port up to the parent module ports.
      for (unsigned i = 0, e = subExtraPorts.size(); i < e; ++i) {
        auto &[firstSubIndex, portInfo] = subExtraPorts[i];
        // This is the index of the user port we are adding.
        auto userIndex = i % userPorts.size();
        auto &sramPort = userPorts[userIndex];
        // Construct a port name, e.g. "sram_0_user_inputs".
        auto sramIndex = extraPorts.size() / userPorts.size();
        auto portName =
            StringAttr::get(context, "sram_" + Twine(sramIndex) + "_" +
                                         sramPort.name.getValue());
        auto portDirection = sramPort.direction;
        auto portType = sramPort.type;
        // Record the extra port.
        extraPorts.push_back(
            {firstPortIndex,
             {portName, type_cast<FIRRTLType>(portType), portDirection}});
        // If this is the DUT, then add a DontTouchAnnotation to any added ports
        // to guarantee that it won't be removed.
        if (isDUT)
          extraPorts.back().second.annotations.addDontTouch();
        // Record the instance result for now, so that we can connect it to the
        // parent module port after we actually add the ports.
        values.push_back(inst.getResult(firstSubIndex + i));
      }

      // We don't want to collect the instance paths or attach inner_syms to
      // the instance path if we aren't creating the output file.
      if (outputFile) {
        // We record any instance paths to memories which are rooted at the
        // current module.
        auto &instancePaths = memInfo.instancePaths;
        auto ref = getInnerRefTo(inst);
        innerRefToInstanceMap[ref] = inst;
        // If its a mem module, this is the start of a path to the module.
        if (isa<FMemModuleOp>(submodule))
          instancePaths.push_back({ref});
        // Copy any paths through the submodule to memories, adding the ref to
        // the current instance.
        for (const auto &subPath : subMemInfo.instancePaths) {
          instancePaths.push_back(subPath);
          instancePaths.back().push_back(ref);
        }
      }
    }
  }

  // Add the extra ports to this module.
  module.insertPorts(extraPorts);

  // Connect the submodule ports to the parent module ports.
  for (unsigned i = 0, e = values.size(); i < e; ++i) {
    auto &[firstArg, port] = extraPorts[i];
    Value modulePort = module.getArgument(firstArg + i);
    Value instPort = values[i];
    if (port.direction == Direction::In)
      std::swap(modulePort, instPort);
    builder.create<MatchingConnectOp>(port.loc, modulePort, instPort);
  }
  return success();
}

void AddSeqMemPortsPass::createOutputFile(igraph::ModuleOpInterface module) {
  // Insert the verbatim at the bottom of the circuit.
  auto circuit = getOperation();
  auto builder = OpBuilder::atBlockEnd(circuit.getBodyBlock());

  // Output buffer.
  std::string buffer;
  llvm::raw_string_ostream os(buffer);

  SymbolTable &symTable = getAnalysis<SymbolTable>();
  HierPathCache cache(circuit, symTable);

  // The current parameter to the verbatim op.
  unsigned paramIndex = 0;
  // Parameters to the verbatim op.
  SmallVector<Attribute> params;
  // Small cache to reduce the number of parameters passed to the verbatim.
  DenseMap<Attribute, unsigned> usedParams;

  // Helper to add a symbol to the verbatim, hitting the cache on the way.
  auto addSymbol = [&](Attribute ref) {
    auto it = usedParams.find(ref);
    unsigned index;
    if (it != usedParams.end()) {
      index = it->second;
    } else {
      index = paramIndex;
      usedParams[ref] = paramIndex++;
      params.push_back(ref);
    }
    os << "{{" << index << "}}";
  };

  // The current sram we are processing.
  unsigned sramIndex = 0;
  auto &instancePaths = memInfoMap[module].instancePaths;
  auto dutSymbol = FlatSymbolRefAttr::get(module.getModuleNameAttr());

  auto loc = builder.getUnknownLoc();
  // Put the information in a verbatim operation.
  builder.create<emit::FileOp>(loc, outputFile, [&] {
    for (auto instancePath : instancePaths) {
      // Note: Reverse instancepath to construct the NLA.
      SmallVector<Attribute> path(llvm::reverse(instancePath));
      os << sramIndex++ << " -> ";
      addSymbol(dutSymbol);
      os << ".";

      auto nlaSymbol = cache.getRefFor(builder.getArrayAttr(path));
      addSymbol(nlaSymbol);
      NamedAttrList fields;
      // There is no current client for the distinct attr, but it will be used
      // by OM::path once the metadata is moved to OM, instead of the verbatim.
      auto id = DistinctAttr::create(UnitAttr::get(builder.getContext()));
      fields.append("id", id);
      fields.append("class", builder.getStringAttr("circt.tracker"));
      fields.append("circt.nonlocal", nlaSymbol);
      // Now add the nonlocal annotation to the leaf instance.
      auto *leafInstance = innerRefToInstanceMap[instancePath.front()];

      AnnotationSet annos(leafInstance);
      annos.addAnnotations(builder.getDictionaryAttr(fields));
      annos.applyToOperation(leafInstance);

      os << "\n";
    }
    builder.create<sv::VerbatimOp>(loc, buffer, ValueRange{},
                                   builder.getArrayAttr(params));
  });
  anythingChanged = true;
}

void AddSeqMemPortsPass::runOnOperation() {
  auto *context = &getContext();
  auto circuit = getOperation();
  instanceGraph = &getAnalysis<InstanceGraph>();
  circtNamespace = CircuitNamespace(circuit);
  // Clear the state.
  userPorts.clear();
  memInfoMap.clear();
  outputFile = {};
  anythingChanged = false;

  // Collect the annotations from the circuit.
  if (failed(processAnnos(circuit)))
    return signalPassFailure();

  // SFC adds the ports in the opposite order they are attached, so we reverse
  // the list here to match exactly.
  std::reverse(userPorts.begin(), userPorts.end());

  auto *dutNode = findDUT();

  // Process the extra ports so we can attach it as metadata on to each memory.
  SmallVector<Attribute> extraPorts;
  auto ui32Type = IntegerType::get(context, 32, IntegerType::Unsigned);
  for (auto &userPort : userPorts) {
    SmallVector<NamedAttribute, 3> attrs;
    attrs.emplace_back(StringAttr::get(context, "name"), userPort.name);
    attrs.emplace_back(
        StringAttr::get(context, "direction"),
        StringAttr::get(
            context, userPort.direction == Direction::In ? "input" : "output"));
    attrs.emplace_back(
        StringAttr::get(context, "width"),
        IntegerAttr::get(
            ui32Type,
            type_cast<FIRRTLBaseType>(userPort.type).getBitWidthOrSentinel()));
    extraPorts.push_back(DictionaryAttr::get(context, attrs));
  }
  extraPortsAttr = ArrayAttr::get(context, extraPorts);

  // If there are no user ports, don't do anything.
  if (userPorts.size() > 0) {
    // Update ports statistic.
    numAddedPorts += userPorts.size();

    // Visit the nodes in post-order.
    for (auto *node : llvm::post_order(dutNode)) {
      auto op = node->getModule();
      if (auto module = dyn_cast<FModuleOp>(*op)) {
        if (failed(processModule(module, /*isDUT=*/node == dutNode)))
          return signalPassFailure();
      } else if (auto mem = dyn_cast<FMemModuleOp>(*op)) {
        processMemModule(mem);
      }
    }

    // We handle the DUT differently than the rest of the modules.
    if (auto dut = dyn_cast<FModuleOp>(*dutNode->getModule())) {
      // For each instance of the dut, add the instance ports, but tie the port
      // to 0 instead of wiring them to the parent.
      for (auto *instRec : dutNode->uses()) {
        auto inst = cast<InstanceOp>(*instRec->getInstance());
        auto &dutMemInfo = memInfoMap[dut];
        // Find out how many memory ports we have to add.
        auto &subExtraPorts = dutMemInfo.extraPorts;
        // If there are no extra ports, we don't have to do anything.
        if (subExtraPorts.size() == 0)
          continue;

        // Add the extra ports to the instance operation.
        auto clone = inst.cloneAndInsertPorts(subExtraPorts);
        inst.replaceAllUsesWith(
            clone.getResults().drop_back(subExtraPorts.size()));
        instanceGraph->replaceInstance(inst, clone);
        inst->erase();
        inst = clone;

        // Tie each port to 0.
        OpBuilder builder(context);
        builder.setInsertionPointAfter(inst);
        for (unsigned i = 0, e = subExtraPorts.size(); i < e; ++i) {
          auto &[firstResult, portInfo] = subExtraPorts[i];
          if (portInfo.direction == Direction::Out)
            continue;
          auto value = inst.getResult(firstResult + i);
          auto type = value.getType();
          auto attr = getIntZerosAttr(type);
          auto zero = builder.create<ConstantOp>(portInfo.loc, type, attr);
          builder.create<MatchingConnectOp>(portInfo.loc, value, zero);
        }
      }
    }
  }

  // If there is an output file, create it.
  if (outputFile)
    createOutputFile(dutNode->getModule<igraph::ModuleOpInterface>());

  if (anythingChanged)
    markAnalysesPreserved<InstanceGraph>();
  else
    markAllAnalysesPreserved();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createAddSeqMemPortsPass() {
  return std::make_unique<AddSeqMemPortsPass>();
}
