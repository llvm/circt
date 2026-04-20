//===- HWLowerXMR.cpp - HW Lower Probe to XMR ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/HW/HWSymCache.h"
#include "circt/Dialect/HW/HierPathCache.h"
#include "circt/Dialect/HW/InnerSymbolNamespace.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Support/HWUtils.h"
#include "circt/Support/InstanceGraph.h"
#include "circt/Support/Namespace.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/PostOrderIterator.h"

#define DEBUG_TYPE "hw-lower-xmr"

namespace circt {
namespace hw {
#define GEN_PASS_DEF_HWLOWERXMR
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

using namespace circt;
using namespace hw;

namespace {

/// Represents a node in the XMR path from resolve to send
struct XMRNode {
  InnerRefAttr ref;           // Reference to instance or target
  std::optional<size_t> next; // Index of next node in path, or nullopt if leaf
  ProbeSubOp subOp; // Sub operation for indexing (if this node is a sub)
};

/// XMR lowering pass with cross-module support
class HWLowerXMRPass : public circt::hw::impl::HWLowerXMRBase<HWLowerXMRPass> {
  void runOnOperation() override;

private:
  LogicalResult lowerProbesInModule(HWModuleOp module);
  LogicalResult handleProbeSend(ProbeSendOp send);
  LogicalResult handleProbeResolve(ProbeResolveOp resolve);
  LogicalResult handleInstance(InstanceOp inst);
  LogicalResult handleProbeRWProbe(ProbeRWProbeOp rwprobe);
  LogicalResult handleProbeSub(ProbeSubOp sub);
  LogicalResult handleProbeCast(ProbeCastOp cast);
  LogicalResult handleForceReleaseOp(Operation *op);
  LogicalResult handleProbeDefine(ProbeDefineOp define);

  InnerRefAttr getOrAddInnerSym(Value target);
  InnerRefAttr getInnerRefTo(Operation *op);
  InnerRefAttr getInnerRefTo(Value val);

  // Add an entry recording the path to a RefSend
  size_t addReachingSendsEntry(Value val, InnerRefAttr node,
                               std::optional<size_t> next = std::nullopt);

  // Get the remote RefSend for a value
  std::optional<size_t> getRemoteRefSend(Value val);

  // Build hierpath from XMR path nodes
  LogicalResult buildHierPath(size_t pathIdx, ImplicitLocOpBuilder &builder,
                              FlatSymbolRefAttr &pathSym,
                              StringAttr &suffixSym);

  // Check if probe type is zero-width
  bool isZeroWidth(Type type);

  // Create zero-width constant
  Value createZeroWidthConstant(Type type, ImplicitLocOpBuilder &builder);

  // Mark port for removal
  void setPortToRemove(Operation *op, size_t index, size_t numPorts);

  // Check if type is probe type
  bool isProbeType(Type type);

  // Remove all marked probe ports from modules and instances
  void garbageCollectProbePorts();

  InnerSymbolNamespace moduleNamespace;
  HierPathCache *hierPathCache = nullptr;

  // Dataflow analysis: track which RefSend reaches each RefType value
  llvm::EquivalenceClasses<Value> dataFlowClasses;
  DenseMap<Value, size_t> dataflowAt;

  // Path tracking: list of XMR nodes forming paths
  SmallVector<XMRNode> xmrPaths;

  // Operations to remove
  SmallVector<ProbeSendOp> sendsToRemove;
  SmallVector<ProbeResolveOp> resolvesToRemove;
  SmallVector<ProbeDefineOp> definesToRemove;
  SmallVector<WireOp> wiresToRemove;
  SmallVector<ProbeSubOp> subsToRemove;
  SmallVector<Operation *> opsToRemove;

  // Visited modules for instance graph traversal
  DenseSet<Operation *> visitedModules;

  // Track ports to remove from modules/instances
  DenseMap<Operation *, llvm::BitVector> refPortsToRemoveMap;

  /// Record the internal path to an external module or a memory.
  DenseMap<size_t, SmallString<128>> xmrPathSuffix;
};

} // namespace

void HWLowerXMRPass::runOnOperation() {
  auto topModule = getOperation();

  Namespace ns;
  ns.add(topModule);
  HierPathCache hpc(&ns, OpBuilder::InsertPoint(topModule.getBody(),
                                                topModule.getBody()->begin()));
  hierPathCache = &hpc;

  for (auto &op : *topModule.getBody()) {
    if (auto hwModule = dyn_cast<HWModuleOp>(op)) {
      moduleNamespace = InnerSymbolNamespace(hwModule);
      if (failed(lowerProbesInModule(hwModule)))
        return signalPassFailure();

      // Mark probe ports for removal
      size_t numPorts = hwModule.getNumPorts();
      auto *body = hwModule.getBodyBlock();

      if (body) {
        // Check input ports (block arguments)
        for (size_t i = 0, e = hwModule.getNumInputPorts(); i < e; ++i) {
          if (isProbeType(body->getArgument(i).getType())) {
            setPortToRemove(hwModule, i, numPorts);
          }
        }

        // Check output ports (hw.output operands)
        if (auto *terminator = body->getTerminator()) {
          if (auto outputOp = dyn_cast<hw::OutputOp>(terminator)) {
            size_t numInputs = hwModule.getNumInputPorts();
            for (size_t i = 0, e = outputOp.getNumOperands(); i < e; ++i) {
              if (isProbeType(outputOp.getOperand(i).getType())) {
                setPortToRemove(hwModule, numInputs + i, numPorts);
              }
            }
          }
        }
      }
    }
  }

  // Remove probe ports from modules and instances FIRST
  // This updates hw.output operations which may reference probe values
  garbageCollectProbePorts();

  // Remove probe operations after port removal
  // Removal order is critical to avoid use-after-free:
  // 1. Defines first (they use rwprobes and other probe values)
  // 2. Resolves next (they use subs, wires, sends, and other probe values)
  // 3. Subs (they are used by resolves, can be chained)
  //    Note: Subs are removed in reverse order they were added, so leaf subs
  //    (those used by other subs) are removed before parent subs
  // 4. Wires (they are used by resolves, produce probe values)
  // 5. Sends (they produce probe values)
  // 6. Other ops last
  for (auto define : definesToRemove)
    define->erase();
  for (auto resolve : resolvesToRemove)
    resolve->erase();
  // Remove subs in reverse order to handle chained subs correctly
  for (auto sub : llvm::reverse(subsToRemove))
    sub->erase();
  for (auto wire : wiresToRemove)
    wire->erase();
  for (auto send : sendsToRemove)
    send->erase();
  for (auto *op : opsToRemove)
    op->erase();
}

LogicalResult HWLowerXMRPass::lowerProbesInModule(HWModuleOp module) {
  SmallVector<ProbeSendOp> sends;
  SmallVector<ProbeResolveOp> resolves;
  SmallVector<InstanceOp> instances;
  SmallVector<ProbeRWProbeOp> rwprobes;
  SmallVector<ProbeSubOp> subs;
  SmallVector<ProbeCastOp> casts;
  SmallVector<ProbeDefineOp> defines;
  SmallVector<Operation *> forceAndReleaseOps;

  module.walk([&](Operation *op) {
    if (auto send = dyn_cast<ProbeSendOp>(op))
      sends.push_back(send);
    else if (auto resolve = dyn_cast<ProbeResolveOp>(op))
      resolves.push_back(resolve);
    else if (auto inst = dyn_cast<InstanceOp>(op))
      instances.push_back(inst);
    else if (auto rwprobe = dyn_cast<ProbeRWProbeOp>(op))
      rwprobes.push_back(rwprobe);
    else if (auto sub = dyn_cast<ProbeSubOp>(op))
      subs.push_back(sub);
    else if (auto cast = dyn_cast<ProbeCastOp>(op))
      casts.push_back(cast);
    else if (auto define = dyn_cast<ProbeDefineOp>(op))
      defines.push_back(define);
    else if (isa<ProbeForceOp, ProbeForceInitialOp, ProbeReleaseOp,
                 ProbeReleaseInitialOp>(op))
      forceAndReleaseOps.push_back(op);
    else if (auto wire = dyn_cast<WireOp>(op)) {
      // Handle hw.wire with probe-typed operands
      // Wire forwards the probe reference from operand to result
      auto wireResult = wire.getResult();
      if (isa<ProbeType, RWProbeType>(wireResult.getType())) {
        // Get the wire's input (hw.wire has a single input operand)
        // Due to SameOperandsAndResultType trait, input must also be probe type
        Value wireInput = wire.getInput();

        // Forward the probe through the wire by unifying dataflow classes
        dataFlowClasses.unionSets(
            dataFlowClasses.getOrInsertLeaderValue(wireResult),
            dataFlowClasses.getOrInsertLeaderValue(wireInput));

        // Mark wire for removal - wires of probe type are transparent
        // forwarding
        wiresToRemove.push_back(wire);
      }
    }
  });

  // Handle sends first
  for (auto send : sends)
    if (failed(handleProbeSend(send)))
      return failure();

  // Handle rwprobes
  for (auto rwprobe : rwprobes)
    if (failed(handleProbeRWProbe(rwprobe)))
      return failure();

  // Handle instances for cross-module dataflow
  for (auto inst : instances)
    if (failed(handleInstance(inst)))
      return failure();

  // Handle probe manipulation ops (sub, cast)
  for (auto sub : subs)
    if (failed(handleProbeSub(sub)))
      return failure();

  for (auto cast : casts)
    if (failed(handleProbeCast(cast)))
      return failure();

  // Handle define operations (connects destination probe to source probe)
  for (auto define : defines)
    if (failed(handleProbeDefine(define)))
      return failure();

  // Handle resolve operations
  for (auto resolve : resolves)
    if (failed(handleProbeResolve(resolve)))
      return failure();

  // Handle force/release operations
  for (auto *op : forceAndReleaseOps)
    if (failed(handleForceReleaseOp(op)))
      return failure();

  return success();
}

InnerRefAttr HWLowerXMRPass::getOrAddInnerSym(Value target) {
  return getInnerRefTo(target);
}

size_t HWLowerXMRPass::addReachingSendsEntry(Value val, InnerRefAttr node,
                                             std::optional<size_t> next) {
  size_t idx = xmrPaths.size();
  xmrPaths.push_back(XMRNode{node, next, ProbeSubOp()});
  dataflowAt[dataFlowClasses.getOrInsertLeaderValue(val)] = idx;
  return idx;
}

std::optional<size_t> HWLowerXMRPass::getRemoteRefSend(Value val) {
  auto leader = dataFlowClasses.getOrInsertLeaderValue(val);
  auto it = dataflowAt.find(leader);
  if (it != dataflowAt.end())
    return it->getSecond();
  return std::nullopt;
}

InnerRefAttr HWLowerXMRPass::getInnerRefTo(Operation *op) {
  auto hwModule = op->getParentOfType<HWModuleOp>();
  if (!hwModule)
    return {};

  // Get or add inner symbol to the operation
  StringAttr symName;

  if (auto innerSymOp = dyn_cast<InnerSymbolOpInterface>(op)) {
    // Check if the operation already has an inner symbol
    auto innerSymAttr = innerSymOp.getInnerSymAttr();
    if (innerSymAttr) {
      // Operation already has an inner symbol, use it
      symName = innerSymAttr.getSymName();
    } else {
      // Need to add an inner symbol
      symName =
          StringAttr::get(op->getContext(), moduleNamespace.newName("xmr_sym"));
      innerSymOp.setInnerSymbolAttr(hw::InnerSymAttr::get(symName));
    }
  } else {
    // Operation doesn't support inner symbols - this shouldn't happen
    // for valid XMR targets, but handle it gracefully
    return {};
  }

  return InnerRefAttr::get(hwModule.getSymNameAttr(), symName);
}

InnerRefAttr HWLowerXMRPass::getInnerRefTo(Value val) {
  // Handle block arguments (module ports)
  if (auto arg = dyn_cast<BlockArgument>(val)) {
    auto hwModule = dyn_cast<HWModuleOp>(arg.getParentBlock()->getParentOp());
    if (!hwModule)
      return {};

    size_t portIdx = arg.getArgNumber();

    // For HW modules, we need to get/set port attributes through the PortList
    // interface Get the current port list
    SmallVector<PortInfo> ports = hwModule.getPortList();
    if (portIdx >= ports.size())
      return {};

    auto &port = ports[portIdx];

    // Check if port already has an inner symbol
    StringAttr symName;
    if (auto sym = port.getSym()) {
      symName = sym.getSymName();
    } else {
      // Add a symbol to the port
      symName = StringAttr::get(hwModule.getContext(),
                                moduleNamespace.newName("port_sym"));
      port.setSym(hw::InnerSymAttr::get(symName), hwModule.getContext());

      // Update the port attributes in the module
      SmallVector<Attribute> portAttrs;
      for (auto &p : ports)
        portAttrs.push_back(
            p.attrs ? p.attrs : DictionaryAttr::get(hwModule.getContext()));
      hwModule.setAllPortAttrs(portAttrs);
    }

    return InnerRefAttr::get(hwModule.getSymNameAttr(), symName);
  }

  // Handle operation results
  if (auto *op = val.getDefiningOp())
    return getInnerRefTo(op);

  return {};
}

LogicalResult HWLowerXMRPass::handleProbeSend(ProbeSendOp send) {
  // Get a reference to the actual signal to which the XMR will be generated.
  Value xmrDef = send.getInput();

  // Check for zero-width probes and skip them
  auto probeType = cast<ProbeType>(send.getResult().getType());
  if (isZeroWidth(probeType.getInnerType())) {
    sendsToRemove.push_back(send);
    return success();
  }

  // Common lambda to finalize probe send handling:
  // Gets inner ref, adds to reaching sends, and marks for removal
  auto finalizeWithInnerRef =
      [&](Value target,
          StringRef errorMsg = "could not get inner ref") -> LogicalResult {
    auto innerRef = getOrAddInnerSym(target);
    if (!innerRef)
      return send.emitError(errorMsg);
    addReachingSendsEntry(send.getResult(), innerRef);
    sendsToRemove.push_back(send);
    return success();
  };

  // Case 1: VerbatimExpr - represents internal path into a module
  if (auto verbExpr = xmrDef.getDefiningOp<sv::VerbatimExprOp>()) {
    if (verbExpr.getSymbolsAttr().empty() && verbExpr->hasOneUse()) {
      // This represents the internal path into a module. For
      // generating the correct XMR, no node can be created in this
      // module. Create a null InnerRef and ensure the hierarchical
      // path ends at the parent that instantiates this module.
      auto inRef = InnerRefAttr();
      auto ind = addReachingSendsEntry(send.getResult(), inRef);
      xmrPathSuffix[ind] = verbExpr.getFormatString();
      opsToRemove.push_back(verbExpr);
      sendsToRemove.push_back(send);
      return success();
    }
  }

  // Case 2: Block argument (module port) - add symbol directly to the port
  if (isa<BlockArgument>(xmrDef))
    return finalizeWithInnerRef(xmrDef, "could not get inner ref for port");

  // Get the defining operation
  auto *xmrDefOp = xmrDef.getDefiningOp();
  if (!xmrDefOp)
    return send.emitError("probe send input has no defining op");

  // Case 3: Operation with InnerSymbol that targets a specific result
  // This ensures that operations like InstanceOp and MemOp, which have inner
  // symbols that target the operation itself (not a specific result), still get
  // nodes created to distinguish which result is being referenced.
  if (auto innerSymOp = dyn_cast<hw::InnerSymbolOpInterface>(xmrDefOp)) {
    if (innerSymOp.getTargetResultIndex())
      return finalizeWithInnerRef(xmrDef, "could not get inner ref for value");
  }

  // Case 4: Operation cannot support an inner symbol, or it has multiple
  // results and doesn't target a specific result. Create a wire node to hold
  // the symbol.
  ImplicitLocOpBuilder builder(send.getLoc(), send);
  builder.setInsertionPointAfterValue(xmrDef);

  // Try to create a meaningful name for the wire
  SmallString<32> wireName;
  if (auto nameAttr = xmrDefOp->getAttrOfType<StringAttr>("name")) {
    wireName = (nameAttr.getValue() + "_probe").str();
  } else if (auto symAttr = xmrDefOp->getAttrOfType<StringAttr>("sym_name")) {
    wireName = (symAttr.getValue() + "_probe").str();
  } else {
    wireName = "xmr_probe";
  }

  // Create a wire to hold the value - hw.wire just passes through the value
  // but can hold an inner symbol
  auto wireNameAttr = builder.getStringAttr(wireName);
  auto wireOp = builder.create<hw::WireOp>(xmrDef, wireNameAttr);

  // Replace all uses of the original value with the wire (except in connects
  // where the original value is the destination)
  Value wireValue = wireOp.getResult();
  xmrDef.replaceUsesWithIf(wireValue, [&](OpOperand &operand) {
    // Don't replace the use in the wire we just created
    if (operand.getOwner() == wireOp.getOperation())
      return false;
    // Preserve the wire input (operand 0) for sv.assign
    if (auto assign = dyn_cast<sv::AssignOp>(operand.getOwner()))
      if (operand.getOperandNumber() == 0)
        return false;
    return true;
  });

  // Use the common lambda to finalize
  return finalizeWithInnerRef(wireValue, "could not get inner ref for wire");
}

LogicalResult HWLowerXMRPass::buildHierPath(size_t pathIdx,
                                            ImplicitLocOpBuilder &builder,
                                            FlatSymbolRefAttr &pathSym,
                                            StringAttr &suffixSym) {
  // Build path by following the chain of XMRNodes
  SmallVector<Attribute> pathElements;
  SmallVector<ProbeSubOp> subOps;
  size_t currentIdx = pathIdx;
  size_t lastIdx = pathIdx;

  while (true) {
    auto &node = xmrPaths[currentIdx];

    // Collect sub operations for suffix generation
    if (node.subOp)
      subOps.push_back(node.subOp);

    // Only add non-null inner refs to the path
    if (node.ref)
      pathElements.push_back(node.ref);

    lastIdx = currentIdx;
    if (!node.next.has_value())
      break;
    currentIdx = node.next.value();
  }

  // Build suffix from sub operations and any explicit suffix
  SmallString<128> suffixString;

  // First, add any explicit suffix (e.g., for extern modules)
  auto iter = xmrPathSuffix.find(lastIdx);
  if (iter != xmrPathSuffix.end()) {
    suffixString = iter->getSecond();
  }

  // Then, append suffixes from sub operations (in reverse order since we
  // collected them while traversing from resolve to send)
  for (auto subOp : llvm::reverse(subOps)) {
    auto inputType = subOp.getInput().getType();
    auto probeType = dyn_cast<ProbeType>(inputType);
    auto rwProbeType = dyn_cast<RWProbeType>(inputType);

    Type innerType;
    if (probeType)
      innerType = probeType.getInnerType();
    else if (rwProbeType)
      innerType = rwProbeType.getInnerType();
    else
      return subOp.emitError("sub operation input is not a probe type");

    // Generate suffix based on the inner type
    TypeSwitch<Type>(innerType)
        .Case<hw::ArrayType>([&](auto arrayType) {
          suffixString.append(("[" + Twine(subOp.getIndex()) + "]").str());
        })
        .Case<hw::StructType>([&](auto structType) {
          auto fieldName = structType.getElements()[subOp.getIndex()].name;
          suffixString.append(".");
          suffixString.append(fieldName.str());
        })
        .Case<hw::UnionType>([&](auto unionType) {
          auto fieldName = unionType.getElements()[subOp.getIndex()].name;
          suffixString.append(".");
          suffixString.append(fieldName.str());
        })
        .Default([&](auto) {
          subOp.emitError("unsupported aggregate type for sub operation");
        });
  }

  if (!suffixString.empty()) {
    suffixSym = builder.getStringAttr(suffixString);
  }

  // Create hierarchical path
  auto pathOp = hierPathCache->getOrCreatePath(
      builder.getArrayAttr(pathElements), builder.getLoc());
  pathSym = FlatSymbolRefAttr::get(pathOp.getSymNameAttr());
  return success();
}

LogicalResult HWLowerXMRPass::handleInstance(InstanceOp inst) {
  // Get the module being instantiated
  auto moduleName = inst.getModuleName();
  auto topModule = inst->getParentOfType<mlir::ModuleOp>();

  // Try to find the referenced module - could be HWModuleOp or HWModuleExternOp
  auto refModule = topModule.lookupSymbol<HWModuleOp>(moduleName);
  auto refExtModule = topModule.lookupSymbol<HWModuleExternOp>(moduleName);

  if (!refModule && !refExtModule) {
    // Skip if module not found
    return success();
  }

  // Check each result to see if it's a probe type
  auto numResults = inst.getNumResults();

  // Handle extern modules specially
  if (refExtModule) {
    // Get total number of ports in the extern module
    auto modType = refExtModule.getHWModuleType();
    size_t numPorts = modType.getNumPorts();

    // Count number of input ports to find the offset for output ports
    size_t numInputs = modType.getNumInputs();

    for (size_t i = 0; i < numResults; ++i) {
      auto result = inst.getResult(i);
      auto probeType = dyn_cast<ProbeType>(result.getType());
      auto rwProbeType = dyn_cast<RWProbeType>(result.getType());

      if (!probeType && !rwProbeType)
        continue;

      // Mark instance port for removal (using result index)
      setPortToRemove(inst, i, numResults);

      // For extern module, convert result index to port index
      // Result index i corresponds to output port at index (numInputs + i)
      size_t portIdx = numInputs + i;
      setPortToRemove(refExtModule, portIdx, numPorts);

      // Skip if no uses or zero-width (will be removed anyway)
      if (result.use_empty())
        continue;

      if (probeType && isZeroWidth(probeType.getInnerType()))
        continue;
      if (rwProbeType && isZeroWidth(rwProbeType.getInnerType()))
        continue;

      // For external modules with probe ports, create an inner ref to the
      // instance and use a path suffix with a macro naming convention
      auto instRef = getInnerRefTo(inst.getOperation());
      if (!instRef)
        continue;

      // Add entry with path suffix for external module
      auto ind = addReachingSendsEntry(result, instRef);

      // Generate macro name: ref_<module-name>_<port-name>
      SmallString<128> macroName;
      macroName += "`ref_";
      macroName += refExtModule.getModuleName();
      macroName += "_";
      macroName += refExtModule.getPortName(portIdx);
      xmrPathSuffix[ind] = macroName;
    }
    return success();
  }

  // Handle regular HWModuleOp
  for (size_t i = 0; i < numResults; ++i) {
    auto result = inst.getResult(i);
    auto probeType = dyn_cast<ProbeType>(result.getType());
    auto rwProbeType = dyn_cast<RWProbeType>(result.getType());

    if (!probeType && !rwProbeType)
      continue;

    // Mark this port for removal
    setPortToRemove(inst, i, numResults);

    // Skip if no uses or zero-width (will be removed anyway)
    if (result.use_empty())
      continue;

    if (probeType && isZeroWidth(probeType.getInnerType()))
      continue;
    if (rwProbeType && isZeroWidth(rwProbeType.getInnerType()))
      continue;

    // For hw.instance, all results are module outputs
    // Module outputs are NOT block arguments - they are operands to hw.output
    // Get the hw.output terminator
    auto *terminator = refModule.getBodyBlock()->getTerminator();
    auto outputOp = dyn_cast<hw::OutputOp>(terminator);
    if (!outputOp) {
      return refModule.emitOpError("module body must end with hw.output");
    }

    // Get the corresponding output value
    // Instance result i corresponds to output i
    if (i >= outputOp.getNumOperands()) {
      return inst.emitOpError("instance result ")
             << i << " out of bounds (module has "
             << outputOp.getNumOperands() << " outputs)";
    }

    Value refModuleOutput = outputOp.getOperand(i);

    // For output instance ports, the dataflow is into this module.
    // Get the remote RefSendOp that flows through the module ports.
    auto remotePathIdx = getRemoteRefSend(refModuleOutput);
    if (!remotePathIdx.has_value())
      continue;

    // Get the path to reaching RefSend at the referenced module output.
    // Now append this instance to the path to the reaching RefSend.
    auto instRef = getInnerRefTo(inst.getOperation());
    if (!instRef)
      continue;

    // Create new path node with instance as first element
    addReachingSendsEntry(result, instRef, remotePathIdx);
  }

  // TODO: Handle input probe ports (probe operands to the instance)
  // This would require iterating over inst->getOperands() and checking for
  // probe types, then unifying dataflow classes

  return success();
}

LogicalResult HWLowerXMRPass::handleProbeResolve(ProbeResolveOp resolve) {
  Value ref = resolve.getRef();

  // Check for zero-width type
  if (isZeroWidth(resolve.getType())) {
    ImplicitLocOpBuilder builder(resolve.getLoc(), resolve);
    Value zero = createZeroWidthConstant(resolve.getType(), builder);
    resolve.getResult().replaceAllUsesWith(zero);
    resolvesToRemove.push_back(resolve);
    return success();
  }

  auto pathIdx = getRemoteRefSend(ref);

  if (!pathIdx.has_value())
    return resolve.emitError("could not trace probe to send");

  ImplicitLocOpBuilder builder(resolve.getLoc(), resolve);
  FlatSymbolRefAttr pathSym;
  StringAttr suffixSym;

  if (failed(buildHierPath(pathIdx.value(), builder, pathSym, suffixSym)))
    return failure();

  auto xmrType = sv::InOutType::get(resolve.getType());
  Value xmr = builder.create<sv::XMRRefOp>(xmrType, pathSym, suffixSym);
  Value readValue = builder.create<sv::ReadInOutOp>(xmr);

  resolve.getResult().replaceAllUsesWith(readValue);
  resolvesToRemove.push_back(resolve);
  return success();
}

LogicalResult HWLowerXMRPass::handleProbeRWProbe(ProbeRWProbeOp rwprobe) {
  // RWProbe creates an XMR reference to a local target
  auto innerRef = rwprobe.getTarget();
  addReachingSendsEntry(rwprobe.getResult(), innerRef);
  opsToRemove.push_back(rwprobe);
  return success();
}

LogicalResult HWLowerXMRPass::handleProbeSub(ProbeSubOp sub) {
  // ProbeSubOp extracts a sub-element from an aggregate probe reference.
  // We need to:
  // 1. Get the reaching send from the input
  // 2. Create a new XMR path entry that includes this sub operation
  // 3. The sub operation will be processed during path resolution to generate
  //    the appropriate suffix (e.g., "[index]" for arrays or ".field" for
  //    structs)

  auto inputPathIdx = getRemoteRefSend(sub.getInput());
  if (!inputPathIdx.has_value())
    return sub.emitError("could not trace input probe to send");

  // Add a new path entry that references this sub operation
  // The InnerRefAttr is null because the sub itself doesn't have a symbol,
  // it just modifies the path with an indexing suffix
  auto newPathIdx =
      addReachingSendsEntry(sub.getResult(), InnerRefAttr(), inputPathIdx);

  // Store the sub operation in the path info so we can generate the suffix
  // later
  xmrPaths[newPathIdx].subOp = sub;

  // Mark for removal after lowering (in separate list for correct order)
  subsToRemove.push_back(sub);
  return success();
}

LogicalResult HWLowerXMRPass::handleProbeCast(ProbeCastOp cast) {
  // Cast just forwards the dataflow
  dataFlowClasses.unionSets(
      dataFlowClasses.getOrInsertLeaderValue(cast.getInput()),
      dataFlowClasses.getOrInsertLeaderValue(cast.getResult()));
  opsToRemove.push_back(cast);
  return success();
}

LogicalResult HWLowerXMRPass::handleForceReleaseOp(Operation *op) {
  // Unified handler for ProbeForceOp, ProbeForceInitialOp, ProbeReleaseOp, and
  // ProbeReleaseInitialOp following FIRRTL LowerXMR pattern

  bool isInitial = isa<ProbeForceInitialOp, ProbeReleaseInitialOp>(op);
  bool isForce = isa<ProbeForceOp, ProbeForceInitialOp>(op);

  // Get the destination rwprobe operand (common to all four ops)
  Value dest;
  if (auto force = dyn_cast<ProbeForceOp>(op))
    dest = force.getDest();
  else if (auto force = dyn_cast<ProbeForceInitialOp>(op))
    dest = force.getDest();
  else if (auto release = dyn_cast<ProbeReleaseOp>(op))
    dest = release.getDest();
  else if (auto release = dyn_cast<ProbeReleaseInitialOp>(op))
    dest = release.getDest();
  else
    return op->emitError("unexpected operation type");

  // Get the type for zero-width check (force ops have src, release ops don't)
  Type checkType = dest.getType();
  if (isForce) {
    if (auto force = dyn_cast<ProbeForceOp>(op))
      checkType = force.getSrc().getType();
    else if (auto force = dyn_cast<ProbeForceInitialOp>(op))
      checkType = force.getSrc().getType();
  }

  // Skip zero-width operations
  if (isZeroWidth(checkType)) {
    opsToRemove.push_back(op);
    return success();
  }

  // Get the reaching send for the destination
  auto pathIdx = getRemoteRefSend(dest);
  if (!pathIdx.has_value())
    return op->emitError("could not trace rwprobe to target");

  ImplicitLocOpBuilder builder(op->getLoc(), op);
  FlatSymbolRefAttr pathSym;
  StringAttr suffixSym;

  if (failed(buildHierPath(pathIdx.value(), builder, pathSym, suffixSym)))
    return failure();

  auto rwProbeType = dyn_cast<RWProbeType>(dest.getType());
  if (!rwProbeType)
    return op->emitError("destination must be rwprobe type");

  auto xmrType = sv::InOutType::get(rwProbeType.getInnerType());
  Value xmr = builder.create<sv::XMRRefOp>(xmrType, pathSym, suffixSym);

  // Create the appropriate SV operation
  if (isInitial) {
    // Initial operations (force.initial or release.initial)
    Value predicate;
    if (auto force = dyn_cast<ProbeForceInitialOp>(op))
      predicate = force.getPredicate();
    else if (auto release = dyn_cast<ProbeReleaseInitialOp>(op))
      predicate = release.getPredicate();

    builder.create<sv::InitialOp>([&]() {
      builder.create<sv::IfOp>(predicate, [&]() {
        if (isForce) {
          Value src = cast<ProbeForceInitialOp>(op).getSrc();
          builder.create<sv::ForceOp>(xmr, src);
        } else {
          builder.create<sv::ReleaseOp>(xmr);
        }
      });
    });
  } else {
    // Clocked operations (force or release)
    Value clock, predicate;
    if (auto force = dyn_cast<ProbeForceOp>(op)) {
      clock = force.getClock();
      predicate = force.getPredicate();
    } else if (auto release = dyn_cast<ProbeReleaseOp>(op)) {
      clock = release.getClock();
      predicate = release.getPredicate();
    }

    builder.create<sv::AlwaysOp>(sv::EventControl::AtPosEdge, clock, [&]() {
      builder.create<sv::IfOp>(predicate, [&]() {
        if (isForce) {
          Value src = cast<ProbeForceOp>(op).getSrc();
          builder.create<sv::ForceOp>(xmr, src);
        } else {
          builder.create<sv::ReleaseOp>(xmr);
        }
      });
    });
  }

  opsToRemove.push_back(op);
  return success();
}

LogicalResult HWLowerXMRPass::handleProbeDefine(ProbeDefineOp define) {
  // ProbeDefine connects the destination probe to the source probe.
  // This is done by unifying their dataflow classes so they resolve to the
  // same XMR path.

  Value dest = define.getDest();
  Value src = define.getSrc();

  // Get the reaching send for the source
  auto srcPathIdx = getRemoteRefSend(src);
  if (!srcPathIdx.has_value())
    return define.emitError("could not trace source probe to send");

  // Unify the destination with the source in the dataflow classes
  dataFlowClasses.unionSets(dataFlowClasses.getOrInsertLeaderValue(dest),
                            dataFlowClasses.getOrInsertLeaderValue(src));

  // Record the reaching send for the destination
  dataflowAt[dataFlowClasses.getOrInsertLeaderValue(dest)] = srcPathIdx.value();

  // Mark the define operation for removal (in separate list to ensure correct
  // order)
  definesToRemove.push_back(define);

  return success();
}

bool HWLowerXMRPass::isZeroWidth(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type))
    return intType.getWidth() == 0;
  // TODO: Handle aggregate types
  return false;
}

Value HWLowerXMRPass::createZeroWidthConstant(Type type,
                                              ImplicitLocOpBuilder &builder) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    if (intType.getWidth() == 0)
      return builder.create<hw::ConstantOp>(APInt(0, uint64_t(0)));
  }
  // TODO: Handle aggregate zero-width types
  return Value();
}

bool HWLowerXMRPass::isProbeType(Type type) {
  return isa<ProbeType, RWProbeType>(type);
}

void HWLowerXMRPass::setPortToRemove(Operation *op, size_t index,
                                     size_t numPorts) {
  auto &portsToRemove = refPortsToRemoveMap[op];
  if (portsToRemove.size() == 0)
    portsToRemove.resize(numPorts);
  portsToRemove.set(index);
}

void HWLowerXMRPass::garbageCollectProbePorts() {
  // Build an instance graph for the module
  auto topModule = getOperation();
  igraph::InstanceGraph instanceGraph(topModule);

  // Process each module that has ports to remove
  for (auto &entry : refPortsToRemoveMap) {
    Operation *op = entry.first;
    const llvm::BitVector &portsToRemove = entry.second;

    if (auto hwModule = dyn_cast<HWModuleOp>(op)) {
      // Use circt::removePorts to handle port removal and instance updates
      circt::removePorts(
          hwModule, instanceGraph,
          // Predicate: should this port be removed?
          [&](const hw::PortInfo &portInfo) -> bool {
            // Find the port index in the module type
            auto modType = hwModule.getHWModuleType();
            for (size_t portIdx = 0; portIdx < modType.getNumPorts();
                 ++portIdx) {
              auto port = hwModule.getPort(portIdx);
              if (port.name == portInfo.name && port.type == portInfo.type &&
                  port.dir == portInfo.dir) {
                return portIdx < portsToRemove.size() &&
                       portsToRemove.test(portIdx);
              }
            }
            return false;
          },
          // Callback for removed input block arguments
          [](BlockArgument) -> bool { return true; },
          // Callback for removed output results
          [](Operation *, unsigned) -> bool { return true; });
    } else if (auto hwExtModule = dyn_cast<HWModuleExternOp>(op)) {
      // Use circt::removePorts for extern modules
      circt::removePorts(
          hwExtModule, instanceGraph,
          // Predicate: should this port be removed?
          [&](const hw::PortInfo &portInfo) -> bool {
            // Find the port index in the module type
            auto modType = hwExtModule.getHWModuleType();
            for (size_t portIdx = 0; portIdx < modType.getNumPorts();
                 ++portIdx) {
              auto port = hwExtModule.getPort(portIdx);
              if (port.name == portInfo.name && port.type == portInfo.type &&
                  port.dir == portInfo.dir) {
                return portIdx < portsToRemove.size() &&
                       portsToRemove.test(portIdx);
              }
            }
            return false;
          },
          // Callback for removed output results
          [](Operation *, unsigned) -> bool { return true; });
    }
  }
}
