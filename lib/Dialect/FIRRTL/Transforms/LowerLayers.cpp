//===- LowerLayers.cpp - Lower Layers by Convention -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This pass lowers FIRRTL layers based on their specified convention.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/InnerSymbolNamespace.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Support/Utils.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Mutex.h"

#define DEBUG_TYPE "firrtl-lower-layers"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_LOWERLAYERS
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

namespace {

/// Indicates the kind of reference that was captured.
enum class ConnectKind {
  /// A normal captured value.  This is a read of a value outside the
  /// layerblock.
  NonRef,
  /// A reference.  This is a destination of a ref define.
  Ref
};

struct ConnectInfo {
  Value value;
  ConnectKind kind;
};

/// The delimiters that should be used for a given generated name.  These vary
/// for modules and files, as well as by convention.
enum class Delimiter { BindModule = '_', BindFile = '-', InlineMacro = '$' };

} // namespace

// A mapping of an old InnerRefAttr to the new inner symbol and module name that
// need to be spliced into the old InnerRefAttr.  This is used to fix
// hierarchical path operations after layers are converted to modules.
using InnerRefMap =
    DenseMap<hw::InnerRefAttr, std::pair<hw::InnerSymAttr, StringAttr>>;

//===----------------------------------------------------------------------===//
// Naming Helpers
//===----------------------------------------------------------------------===//

static void appendName(StringRef name, SmallString<32> &output,
                       bool toLower = false,
                       Delimiter delimiter = Delimiter::BindFile) {
  if (name.empty())
    return;
  if (!output.empty())
    output.push_back(static_cast<char>(delimiter));
  output.append(name);
  if (!toLower)
    return;
  auto i = output.size() - name.size();
  output[i] = llvm::toLower(output[i]);
}

static void appendName(SymbolRefAttr name, SmallString<32> &output,
                       bool toLower = false,
                       Delimiter delimiter = Delimiter::BindFile) {
  appendName(name.getRootReference(), output, toLower, delimiter);
  for (auto nested : name.getNestedReferences())
    appendName(nested.getValue(), output, toLower, delimiter);
}

/// For a layer `@A::@B::@C` in module Module,
/// the generated module is called `Module_A_B_C`.
static SmallString<32> moduleNameForLayer(StringRef moduleName,
                                          SymbolRefAttr layerName) {
  SmallString<32> result;
  appendName(moduleName, result, /*toLower=*/false,
             /*delimiter=*/Delimiter::BindModule);
  appendName(layerName, result, /*toLower=*/false,
             /*delimiter=*/Delimiter::BindModule);
  return result;
}

/// For a layerblock `@A::@B::@C`,
/// the generated instance is called `a_b_c`.
static SmallString<32> instanceNameForLayer(SymbolRefAttr layerName) {
  SmallString<32> result;
  appendName(layerName, result, /*toLower=*/true,
             /*delimiter=*/Delimiter::BindModule);
  return result;
}

/// For all layerblocks `@A::@B::@C` in a circuit called Circuit,
/// the output filename is `layers-Circuit-A-B-C.sv`.
static SmallString<32> fileNameForLayer(StringRef circuitName,
                                        SymbolRefAttr layerName) {
  SmallString<32> result;
  result.append("layers");
  appendName(circuitName, result);
  appendName(layerName, result);
  result.append(".sv");
  return result;
}

/// For a layerblock `@A::@B::@C`, the verilog macro is `A_B_C`.
static SmallString<32>
macroNameForLayer(StringRef circuitName,
                  ArrayRef<FlatSymbolRefAttr> layerName) {
  SmallString<32> result("layer_");
  result.append(circuitName);
  for (auto part : layerName)
    appendName(part, result, /*toLower=*/false,
               /*delimiter=*/Delimiter::InlineMacro);
  return result;
}

//===----------------------------------------------------------------------===//
// LowerLayersPass
//===----------------------------------------------------------------------===//

class LowerLayersPass
    : public circt::firrtl::impl::LowerLayersBase<LowerLayersPass> {
  hw::OutputFileAttr getOutputFile(SymbolRefAttr layerName) {
    auto layer = symbolToLayer.lookup(layerName);
    if (!layer)
      return nullptr;
    return layer->getAttrOfType<hw::OutputFileAttr>("output_file");
  }

  hw::OutputFileAttr outputFileForLayer(StringRef circuitName,
                                        SymbolRefAttr layerName) {
    if (auto file = getOutputFile(layerName))
      return hw::OutputFileAttr::getFromDirectoryAndFilename(
          &getContext(), file.getDirectory(),
          fileNameForLayer(circuitName, layerName),
          /*excludeFromFileList=*/true);
    return hw::OutputFileAttr::getFromFilename(
        &getContext(), fileNameForLayer(circuitName, layerName),
        /*excludeFromFileList=*/true);
  }

  /// Safely build a new module with a given namehint.  This handles geting a
  /// lock to modify the top-level circuit.
  FModuleOp buildNewModule(OpBuilder &builder, LayerBlockOp layerBlock,
                           SmallVectorImpl<PortInfo> &ports);

  /// Strip layer colors from the module's interface.
  FailureOr<InnerRefMap> runOnModuleLike(FModuleLike moduleLike);

  /// Extract layerblocks and strip probe colors from all ops under the module.
  LogicalResult runOnModuleBody(FModuleOp moduleOp, InnerRefMap &innerRefMap);

  /// Update the module's port types to remove any explicit layer requirements
  /// from any probe types.
  void removeLayersFromPorts(FModuleLike moduleLike);

  /// Update the value's type to remove any layers from any probe types.
  void removeLayersFromValue(Value value);

  /// Remove any layers from the result of the cast. If the cast becomes a nop,
  /// remove the cast itself from the IR.
  void removeLayersFromRefCast(RefCastOp cast);

  /// Lower an inline layerblock to an ifdef block.
  void lowerInlineLayerBlock(LayerOp layer, LayerBlockOp layerBlock);

  /// Preprocess layers to build macro declarations and cache information about
  /// the layers so that this can be quired later.
  void preprocessLayers(CircuitNamespace &ns, OpBuilder &b, LayerOp layer,
                        StringRef circuitName,
                        SmallVector<FlatSymbolRefAttr> &stack);
  void preprocessLayers(CircuitNamespace &ns, LayerOp layer,
                        StringRef circuitName);

  /// Entry point for the function.
  void runOnOperation() override;

  /// Indicates exclusive access to modify the circuitNamespace and the circuit.
  llvm::sys::SmartMutex<true> *circuitMutex;

  /// A map of layer blocks to module name that should be created for it.
  DenseMap<LayerBlockOp, StringRef> moduleNames;

  /// A map from inline layers to their macro names.
  DenseMap<LayerOp, FlatSymbolRefAttr> macroNames;

  /// A mapping of symbol name to layer operation.
  DenseMap<SymbolRefAttr, LayerOp> symbolToLayer;
};

/// Multi-process safe function to build a module in the circuit and return it.
/// The name provided is only a namehint for the module---a unique name will be
/// generated if there are conflicts with the namehint in the circuit-level
/// namespace.
FModuleOp LowerLayersPass::buildNewModule(OpBuilder &builder,
                                          LayerBlockOp layerBlock,
                                          SmallVectorImpl<PortInfo> &ports) {
  auto location = layerBlock.getLoc();
  auto namehint = moduleNames.lookup(layerBlock);
  llvm::sys::SmartScopedLock<true> instrumentationLock(*circuitMutex);
  FModuleOp newModule = builder.create<FModuleOp>(
      location, builder.getStringAttr(namehint),
      ConventionAttr::get(builder.getContext(), Convention::Internal), ports,
      ArrayAttr{});
  if (auto dir = getOutputFile(layerBlock.getLayerNameAttr())) {
    assert(dir.isDirectory());
    newModule->setAttr("output_file", dir);
  }
  SymbolTable::setSymbolVisibility(newModule, SymbolTable::Visibility::Private);
  return newModule;
}

void LowerLayersPass::removeLayersFromValue(Value value) {
  auto type = dyn_cast<RefType>(value.getType());
  if (!type || !type.getLayer())
    return;
  value.setType(type.removeLayer());
}

void LowerLayersPass::removeLayersFromRefCast(RefCastOp cast) {
  auto result = cast.getResult();
  auto oldType = result.getType();
  if (oldType.getLayer()) {
    auto input = cast.getInput();
    auto srcType = input.getType();
    auto newType = oldType.removeLayer();
    if (newType == srcType) {
      result.replaceAllUsesWith(input);
      cast->erase();
      return;
    }
    result.setType(newType);
  }
}

void LowerLayersPass::removeLayersFromPorts(FModuleLike moduleLike) {
  auto oldTypeAttrs = moduleLike.getPortTypesAttr();
  SmallVector<Attribute> newTypeAttrs;
  newTypeAttrs.reserve(oldTypeAttrs.size());
  bool changed = false;

  for (auto typeAttr : oldTypeAttrs.getAsRange<TypeAttr>()) {
    if (auto refType = dyn_cast<RefType>(typeAttr.getValue())) {
      if (refType.getLayer()) {
        typeAttr = TypeAttr::get(refType.removeLayer());
        changed = true;
      }
    }
    newTypeAttrs.push_back(typeAttr);
  }

  if (!changed)
    return;

  moduleLike.setPortTypesAttr(
      ArrayAttr::get(moduleLike.getContext(), newTypeAttrs));

  if (auto moduleOp = dyn_cast<FModuleOp>(moduleLike.getOperation())) {
    for (auto arg : moduleOp.getBodyBlock()->getArguments())
      removeLayersFromValue(arg);
  }
}

FailureOr<InnerRefMap>
LowerLayersPass::runOnModuleLike(FModuleLike moduleLike) {
  LLVM_DEBUG({
    llvm::dbgs() << "Module: " << moduleLike.getModuleName() << "\n";
    llvm::dbgs() << "  Examining Layer Blocks:\n";
  });

  // Strip away layers from the interface of the module-like op.
  InnerRefMap innerRefMap;
  auto result =
      TypeSwitch<Operation *, LogicalResult>(moduleLike.getOperation())
          .Case<FModuleOp>([&](auto op) {
            op.setLayers({});
            removeLayersFromPorts(op);
            return runOnModuleBody(op, innerRefMap);
          })
          .Case<FExtModuleOp, FIntModuleOp, FMemModuleOp>([&](auto op) {
            op.setLayers({});
            removeLayersFromPorts(op);
            return success();
          })
          .Case<ClassOp, ExtClassOp>([](auto) { return success(); })
          .Default(
              [](auto *op) { return op->emitError("unknown module-like op"); });

  if (failed(result))
    return failure();

  return innerRefMap;
}

void LowerLayersPass::lowerInlineLayerBlock(LayerOp layer,
                                            LayerBlockOp layerBlock) {
  OpBuilder builder(layerBlock);
  auto macroName = macroNames[layer];
  auto ifDef = builder.create<sv::IfDefOp>(layerBlock.getLoc(), macroName);
  ifDef.getBodyRegion().takeBody(layerBlock.getBodyRegion());
  layerBlock.erase();
}

LogicalResult LowerLayersPass::runOnModuleBody(FModuleOp moduleOp,
                                               InnerRefMap &innerRefMap) {
  CircuitOp circuitOp = moduleOp->getParentOfType<CircuitOp>();
  StringRef circuitName = circuitOp.getName();
  hw::InnerSymbolNamespace ns(moduleOp);

  // A map of instance ops to modules that this pass creates.  This is used to
  // check if this was an instance that we created and to do fast module
  // dereferencing (avoiding a symbol table).
  DenseMap<InstanceOp, FModuleOp> createdInstances;

  // Post-order traversal that expands a layer block into its parent. For each
  // layer block found do the following:
  //
  // 1. Create and connect one ref-type output port for each value defined in
  //    this layer block that drives an instance marked lowerToBind and move
  //    this instance outside the layer block.
  // 2. Create one input port for each value captured by this layer block.
  // 3. Create a new module for this layer block and move the (mutated) body of
  //    this layer block to the new module.
  // 4. Instantiate the new module outside the layer block and hook it up.
  // 5. Erase the layer block.
  auto result = moduleOp.walk<mlir::WalkOrder::PostOrder>([&](Operation *op) {
    // Strip layer requirements from any op that might represent a probe.
    if (auto wire = dyn_cast<WireOp>(op)) {
      removeLayersFromValue(wire.getResult());
      return WalkResult::advance();
    }
    if (auto sub = dyn_cast<RefSubOp>(op)) {
      removeLayersFromValue(sub.getResult());
      return WalkResult::advance();
    }
    if (auto instance = dyn_cast<InstanceOp>(op)) {
      instance.setLayers({});
      for (auto result : instance.getResults())
        removeLayersFromValue(result);
      return WalkResult::advance();
    }
    if (auto cast = dyn_cast<RefCastOp>(op)) {
      removeLayersFromRefCast(cast);
      return WalkResult::advance();
    }

    auto layerBlock = dyn_cast<LayerBlockOp>(op);
    if (!layerBlock)
      return WalkResult::advance();

    auto layer = symbolToLayer.lookup(layerBlock.getLayerName());

    if (layer.getConvention() == LayerConvention::Inline) {
      lowerInlineLayerBlock(layer, layerBlock);
      return WalkResult::advance();
    }

    assert(layer.getConvention() == LayerConvention::Bind);
    Block *body = layerBlock.getBody(0);
    OpBuilder builder(moduleOp);

    // Ports that need to be created for the module derived from this layer
    // block.
    SmallVector<PortInfo> ports;

    // Connection that need to be made to the instance of the derived module.
    SmallVector<ConnectInfo> connectValues;

    // Create an input port for an operand that is captured from outside.
    auto createInputPort = [&](Value operand, Location loc) {
      const auto &[portName, _] = getFieldName(FieldRef(operand, 0), true);

      // If the value is a ref, we must resolve the ref inside the parent,
      // passing the input as a value instead of a ref. Inside the layer, we
      // convert (ref.send) the value back into a ref.
      auto type = operand.getType();
      if (auto refType = dyn_cast<RefType>(type))
        type = refType.getType();

      ports.push_back({builder.getStringAttr(portName), type, Direction::In,
                       /*symName=*/{},
                       /*location=*/loc});
      // Update the layer block's body with arguments as we will swap this body
      // into the module when we create it.  If this is a ref type, then add a
      // refsend to convert from the non-ref type input port.
      Value replacement = body->addArgument(type, loc);
      if (isa<RefType>(operand.getType())) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(body);
        replacement = builder.create<RefSendOp>(loc, replacement);
      }
      operand.replaceUsesWithIf(replacement, [&](OpOperand &use) {
        auto *user = use.getOwner();
        if (!layerBlock->isAncestor(user))
          return false;
        if (auto connectLike = dyn_cast<FConnectLike>(user)) {
          if (use.getOperandNumber() == 0)
            return false;
        }
        return true;
      });

      connectValues.push_back({operand, ConnectKind::NonRef});
    };

    // Set the location intelligently.  Use the location of the capture if this
    // is a port created for forwarding from a parent layer block to a nested
    // layer block.  Otherwise, use unknown.
    auto getPortLoc = [&](Value port) -> Location {
      Location loc = UnknownLoc::get(port.getContext());
      if (auto *destOp = port.getDefiningOp())
        if (auto instOp = dyn_cast<InstanceOp>(destOp)) {
          auto modOpIt = createdInstances.find(instOp);
          if (modOpIt != createdInstances.end()) {
            auto portNum = cast<OpResult>(port).getResultNumber();
            loc = modOpIt->getSecond().getPortLocation(portNum);
          }
        }
      return loc;
    };

    // Create an output probe port port and adds a ref.define/ref.send to
    // drive the port if this was not already capturing a ref type.
    auto createOutputPort = [&](Value dest, Value src) {
      auto loc = getPortLoc(dest);
      auto portNum = ports.size();
      const auto &[portName, _] = getFieldName(FieldRef(dest, 0), true);

      RefType refType;
      if (auto oldRef = dyn_cast<RefType>(dest.getType()))
        refType = oldRef;
      else
        refType = RefType::get(
            type_cast<FIRRTLBaseType>(dest.getType()).getPassiveType(),
            /*forceable=*/false);

      ports.push_back({builder.getStringAttr(portName), refType, Direction::Out,
                       /*symName=*/{}, /*location=*/loc});
      Value replacement = body->addArgument(refType, loc);
      if (isa<RefType>(dest.getType())) {
        dest.replaceUsesWithIf(replacement, [&](OpOperand &use) {
          auto *user = use.getOwner();
          if (!layerBlock->isAncestor(user))
            return false;
          if (auto connectLike = dyn_cast<FConnectLike>(user)) {
            if (use.getOperandNumber() == 0)
              return true;
          }
          return false;
        });
        connectValues.push_back({dest, ConnectKind::Ref});
        return;
      }
      connectValues.push_back({dest, ConnectKind::NonRef});
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfterValue(src);
      builder.create<RefDefineOp>(
          loc, body->getArgument(portNum),
          builder.create<RefSendOp>(loc, src)->getResult(0));
    };

    SmallVector<hw::InnerSymAttr> innerSyms;
    SmallVector<RWProbeOp> rwprobes;
    SmallVector<sv::VerbatimOp> verbatims;
    auto layerBlockWalkResult = layerBlock.walk([&](Operation *op) {
      // Record any operations inside the layer block which have inner symbols.
      // Theses may have symbol users which need to be updated.
      if (auto symOp = dyn_cast<hw::InnerSymbolOpInterface>(op))
        if (auto innerSym = symOp.getInnerSymAttr())
          innerSyms.push_back(innerSym);

      // Handle instance ops that were created from nested layer blocks.  These
      // ops need to be moved outside the layer block to avoid nested binds.
      // Nested binds are illegal in the SystemVerilog specification (and
      // checked by FIRRTL verification).
      //
      // For each value defined in this layer block which drives a port of one
      // of these instances, create an output reference type port on the
      // to-be-created module and drive it with the value.  Move the instance
      // outside the layer block.  We will hook it up later once we replace the
      // layer block with an instance.
      if (auto instOp = dyn_cast<InstanceOp>(op)) {
        // Ignore instances which this pass did not create.
        if (!createdInstances.contains(instOp))
          return WalkResult::advance();

        LLVM_DEBUG({
          llvm::dbgs()
              << "      Found instance created from nested layer block:\n"
              << "        module: " << instOp.getModuleName() << "\n"
              << "        instance: " << instOp.getName() << "\n";
        });
        instOp->moveBefore(layerBlock);
        return WalkResult::advance();
      }

      // Handle subfields, subindexes, and subaccesses which are indexing into
      // non-passive values.  If these are kept in the module, then the module
      // must create bi-directional ports.  This doesn't make sense as the
      // FIRRTL spec states that no layerblock may write to values outside it.
      // Fix this for subfield and subindex by moving these ops outside the
      // layerblock.  Try to fix this for subaccess and error if the move can't
      // be made because the index is defined inside the layerblock.  (This case
      // is exceedingly rare given that subaccesses are almost always unexepcted
      // when this pass runs.)
      if (isa<SubfieldOp, SubindexOp>(op)) {
        auto input = op->getOperand(0);
        if (!firrtl::type_cast<FIRRTLBaseType>(input.getType()).isPassive() &&
            !isAncestorOfValueOwner(layerBlock, input))
          op->moveBefore(layerBlock);
        return WalkResult::advance();
      }

      if (auto subOp = dyn_cast<SubaccessOp>(op)) {
        auto input = subOp.getInput();
        if (firrtl::type_cast<FIRRTLBaseType>(input.getType()).isPassive())
          return WalkResult::advance();

        if (!isAncestorOfValueOwner(layerBlock, input) &&
            !isAncestorOfValueOwner(layerBlock, subOp.getIndex())) {
          subOp->moveBefore(layerBlock);
          return WalkResult::advance();
        }
        auto diag = op->emitOpError()
                    << "has a non-passive operand and captures a value defined "
                       "outside its enclosing bind-convention layerblock.  The "
                       "'LowerLayers' pass cannot lower this as it would "
                       "create an output port on the resulting module.";
        diag.attachNote(layerBlock.getLoc())
            << "the layerblock is defined here";
        return WalkResult::interrupt();
      }

      if (auto rwprobe = dyn_cast<RWProbeOp>(op)) {
        rwprobes.push_back(rwprobe);
        return WalkResult::advance();
      }

      if (auto connect = dyn_cast<FConnectLike>(op)) {
        auto src = connect.getSrc();
        auto dst = connect.getDest();
        auto srcInLayerBlock = isAncestorOfValueOwner(layerBlock, src);
        auto dstInLayerBlock = isAncestorOfValueOwner(layerBlock, dst);
        if (!srcInLayerBlock && !dstInLayerBlock) {
          connect->moveBefore(layerBlock);
          return WalkResult::advance();
        }
        // Create an input port.
        if (!srcInLayerBlock) {
          createInputPort(src, op->getLoc());
          return WalkResult::advance();
        }
        // Create an output port.
        if (!dstInLayerBlock) {
          createOutputPort(dst, src);
          if (!isa<RefType>(dst.getType()))
            connect.erase();
          return WalkResult::advance();
        }
        // Source and destination in layer block.  Nothing to do.
        return WalkResult::advance();
      }

      // Pre-emptively de-squiggle connections that we are creating.  This will
      // later be cleaned up by the de-squiggling pass.  However, there is no
      // point in creating deeply squiggled connections if we don't have to.
      //
      // This pattern matches the following structure.  Move the ref.resolve
      // outside the layer block.  The matchingconnect will be moved outside in
      // the next loop iteration:
      //     %0 = ...
      //     %1 = ...
      //     firrtl.layerblock {
      //       %2 = ref.resolve %0
      //       firrtl.matchingconnect %1, %2
      //     }
      if (auto refResolve = dyn_cast<RefResolveOp>(op))
        if (refResolve.getResult().hasOneUse() &&
            refResolve.getRef().getParentBlock() != body)
          if (auto connect = dyn_cast<MatchingConnectOp>(
                  *refResolve.getResult().getUsers().begin()))
            if (connect.getDest().getParentBlock() != body) {
              refResolve->moveBefore(layerBlock);
              return WalkResult::advance();
            }

      // For any other ops, create input ports for any captured operands.
      for (auto operand : op->getOperands()) {
        if (!isAncestorOfValueOwner(layerBlock, operand))
          createInputPort(operand, op->getLoc());
      }

      if (auto verbatim = dyn_cast<sv::VerbatimOp>(op))
        verbatims.push_back(verbatim);

      return WalkResult::advance();
    });

    if (layerBlockWalkResult.wasInterrupted())
      return WalkResult::interrupt();

    // Create the new module.  This grabs a lock to modify the circuit.
    FModuleOp newModule = buildNewModule(builder, layerBlock, ports);
    SymbolTable::setSymbolVisibility(newModule,
                                     SymbolTable::Visibility::Private);
    newModule.getBody().takeBody(layerBlock.getRegion());

    LLVM_DEBUG({
      llvm::dbgs() << "      New Module: " << moduleNames.lookup(layerBlock)
                   << "\n";
      llvm::dbgs() << "        ports:\n";
      for (size_t i = 0, e = ports.size(); i != e; ++i) {
        auto port = ports[i];
        auto value = connectValues[i];
        llvm::dbgs() << "          - name: " << port.getName() << "\n"
                     << "            type: " << port.type << "\n"
                     << "            direction: " << port.direction << "\n"
                     << "            value: " << value.value << "\n"
                     << "            kind: "
                     << (value.kind == ConnectKind::NonRef ? "NonRef" : "Ref")
                     << "\n";
      }
    });

    // Replace the original layer block with an instance.  Hook up the instance.
    // Intentionally create instance with probe ports which do not have an
    // associated layer.  This is illegal IR that will be made legal by the end
    // of the pass.  This is done to avoid having to revisit and rewrite each
    // instance everytime it is moved into a parent layer.
    builder.setInsertionPointAfter(layerBlock);
    auto instanceName = instanceNameForLayer(layerBlock.getLayerName());
    auto instanceOp = builder.create<InstanceOp>(
        layerBlock.getLoc(), /*moduleName=*/newModule,
        /*name=*/
        instanceName, NameKindEnum::DroppableName,
        /*annotations=*/ArrayRef<Attribute>{},
        /*portAnnotations=*/ArrayRef<Attribute>{}, /*lowerToBind=*/true,
        /*innerSym=*/
        (innerSyms.empty() ? hw::InnerSymAttr{}
                           : hw::InnerSymAttr::get(builder.getStringAttr(
                                 ns.newName(instanceName)))));

    auto outputFile =
        outputFileForLayer(circuitName, layerBlock.getLayerName());
    instanceOp->setAttr("output_file", outputFile);

    createdInstances.try_emplace(instanceOp, newModule);

    LLVM_DEBUG(llvm::dbgs() << "        moved inner refs:\n");
    for (hw::InnerSymAttr innerSym : innerSyms) {
      auto oldInnerRef = hw::InnerRefAttr::get(moduleOp.getModuleNameAttr(),
                                               innerSym.getSymName());
      auto splice = std::make_pair(instanceOp.getInnerSymAttr(),
                                   newModule.getModuleNameAttr());
      innerRefMap.insert({oldInnerRef, splice});
      LLVM_DEBUG(llvm::dbgs() << "          - ref: " << oldInnerRef << "\n"
                              << "            splice: " << splice.first << ", "
                              << splice.second << "\n";);
    }

    // Update RWProbe operations.
    for (auto rwprobe : rwprobes) {
      auto targetRef = rwprobe.getTarget();
      auto mapped = innerRefMap.find(targetRef);
      if (mapped == innerRefMap.end()) {
        assert(targetRef.getModule() == moduleOp.getNameAttr());
        auto ist = hw::InnerSymbolTable::get(moduleOp);
        if (failed(ist))
          return WalkResult::interrupt();
        auto target = ist->lookup(targetRef.getName());
        assert(target);
        auto fieldref = getFieldRefForTarget(target);
        rwprobe
            .emitError(
                "rwprobe capture not supported with bind convention layer")
            .attachNote(fieldref.getLoc())
            .append("rwprobe target outside of bind layer");
        return WalkResult::interrupt();
      }

      if (mapped->second.second != newModule.getModuleNameAttr())
        return rwprobe.emitError("rwprobe target refers to different module"),
               WalkResult::interrupt();

      rwprobe.setTargetAttr(
          hw::InnerRefAttr::get(mapped->second.second, targetRef.getName()));
    }

    // Update verbatims that target operations extracted alongside.
    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement(
        [&innerRefMap](hw::InnerRefAttr ref) -> std::optional<Attribute> {
          auto it = innerRefMap.find(ref);
          if (it != innerRefMap.end())
            return hw::InnerRefAttr::get(it->second.second, ref.getName());
          return std::nullopt;
        });
    for (auto verbatim : verbatims)
      replacer.replaceElementsIn(verbatim);

    // Connect instance ports to values.
    assert(ports.size() == connectValues.size() &&
           "the number of instance ports and values to connect to them must be "
           "equal");
    for (unsigned portNum = 0, e = newModule.getNumPorts(); portNum < e;
         ++portNum) {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfterValue(instanceOp.getResult(portNum));
      if (instanceOp.getPortDirection(portNum) == Direction::In) {
        auto src = connectValues[portNum].value;
        if (isa<RefType>(src.getType()))
          src = builder.create<RefResolveOp>(
              newModule.getPortLocationAttr(portNum), src);
        builder.create<MatchingConnectOp>(
            newModule.getPortLocationAttr(portNum),
            instanceOp.getResult(portNum), src);
      } else if (isa<RefType>(instanceOp.getResult(portNum).getType()) &&
                 connectValues[portNum].kind == ConnectKind::Ref)
        builder.create<RefDefineOp>(getPortLoc(connectValues[portNum].value),
                                    connectValues[portNum].value,
                                    instanceOp.getResult(portNum));
      else
        builder.create<MatchingConnectOp>(
            getPortLoc(connectValues[portNum].value),
            connectValues[portNum].value,
            builder.create<RefResolveOp>(newModule.getPortLocationAttr(portNum),
                                         instanceOp.getResult(portNum)));
    }
    layerBlock.erase();

    return WalkResult::advance();
  });
  return success(!result.wasInterrupted());
}

void LowerLayersPass::preprocessLayers(CircuitNamespace &ns, OpBuilder &b,
                                       LayerOp layer, StringRef circuitName,
                                       SmallVector<FlatSymbolRefAttr> &stack) {
  stack.emplace_back(FlatSymbolRefAttr::get(layer.getSymNameAttr()));
  ArrayRef stackRef(stack);
  symbolToLayer.insert(
      {SymbolRefAttr::get(stackRef.front().getAttr(), stackRef.drop_front()),
       layer});
  if (layer.getConvention() == LayerConvention::Inline) {
    auto *ctx = &getContext();
    auto macName = macroNameForLayer(circuitName, stack);
    auto symName = ns.newName(macName);

    auto symNameAttr = StringAttr::get(ctx, symName);
    auto macNameAttr = StringAttr();
    if (macName != symName)
      macNameAttr = StringAttr::get(ctx, macName);

    b.create<sv::MacroDeclOp>(layer->getLoc(), symNameAttr, ArrayAttr(),
                              macNameAttr);
    macroNames[layer] = FlatSymbolRefAttr::get(&getContext(), symNameAttr);
  }
  for (auto child : layer.getOps<LayerOp>())
    preprocessLayers(ns, b, child, circuitName, stack);
  stack.pop_back();
}

void LowerLayersPass::preprocessLayers(CircuitNamespace &ns, LayerOp layer,
                                       StringRef circuitName) {
  OpBuilder b(layer);
  SmallVector<FlatSymbolRefAttr> stack;
  preprocessLayers(ns, b, layer, circuitName, stack);
}

/// Process a circuit to remove all layer blocks in each module and top-level
/// layer definition.
void LowerLayersPass::runOnOperation() {
  LLVM_DEBUG(
      llvm::dbgs() << "==----- Running LowerLayers "
                      "-------------------------------------------------===\n");
  CircuitOp circuitOp = getOperation();
  StringRef circuitName = circuitOp.getName();

  // Initialize members which cannot be initialized automatically.
  llvm::sys::SmartMutex<true> mutex;
  circuitMutex = &mutex;

  CircuitNamespace ns(circuitOp);
  for (auto &op : *circuitOp.getBodyBlock()) {
    // Determine names for all modules that will be created.  Do this serially
    // to avoid non-determinism from creating these in the parallel region.
    if (auto moduleOp = dyn_cast<FModuleOp>(op)) {
      moduleOp->walk([&](LayerBlockOp layerBlockOp) {
        auto name = moduleNameForLayer(moduleOp.getModuleName(),
                                       layerBlockOp.getLayerName());
        moduleNames.insert({layerBlockOp, ns.newName(name)});
      });
      continue;
    }
    // Build verilog macro declarations for each inline layer declarations.
    // Cache layer symbol refs for lookup later.
    if (auto layerOp = dyn_cast<LayerOp>(op)) {
      preprocessLayers(ns, layerOp, circuitName);
      continue;
    }
  }

  auto mergeMaps = [](auto &&a, auto &&b) {
    if (failed(a))
      return std::forward<decltype(a)>(a);
    if (failed(b))
      return std::forward<decltype(b)>(b);

    for (auto bb : *b)
      a->insert(bb);
    return std::forward<decltype(a)>(a);
  };

  // Lower the layer blocks of each module.
  SmallVector<FModuleLike> modules(
      circuitOp.getBodyBlock()->getOps<FModuleLike>());
  auto failureOrInnerRefMap = transformReduce(
      circuitOp.getContext(), modules, FailureOr<InnerRefMap>(InnerRefMap{}),
      mergeMaps, [this](FModuleLike mod) -> FailureOr<InnerRefMap> {
        return runOnModuleLike(mod);
      });
  if (failed(failureOrInnerRefMap))
    return signalPassFailure();
  auto &innerRefMap = *failureOrInnerRefMap;

  // Rewrite any hw::HierPathOps which have namepaths that contain rewritting
  // inner refs.
  //
  // TODO: This unnecessarily computes a new namepath for every hw::HierPathOp
  // even if that namepath is not used.  It would be better to only build the
  // new namepath when a change is needed, e.g., by recording updates to the
  // namepath.
  for (hw::HierPathOp hierPathOp : circuitOp.getOps<hw::HierPathOp>()) {
    SmallVector<Attribute> newNamepath;
    bool modified = false;
    for (auto attr : hierPathOp.getNamepath()) {
      hw::InnerRefAttr innerRef = dyn_cast<hw::InnerRefAttr>(attr);
      if (!innerRef) {
        newNamepath.push_back(attr);
        continue;
      }
      auto it = innerRefMap.find(innerRef);
      if (it == innerRefMap.end()) {
        newNamepath.push_back(attr);
        continue;
      }

      auto &[inst, mod] = it->getSecond();
      newNamepath.push_back(
          hw::InnerRefAttr::get(innerRef.getModule(), inst.getSymName()));
      newNamepath.push_back(hw::InnerRefAttr::get(mod, innerRef.getName()));
      modified = true;
    }
    if (modified)
      hierPathOp.setNamepathAttr(
          ArrayAttr::get(circuitOp.getContext(), newNamepath));
  }

  // Generate the header and footer of each bindings file.  The body will be
  // populated later when binds are exported to Verilog.  This produces text
  // like:
  //
  //     `include "layers-A.sv"
  //     `include "layers-A-B.sv"
  //     `ifndef layers_A_B_C
  //     `define layers_A_B_C
  //     <body>
  //     `endif // layers_A_B_C
  //
  // TODO: This would be better handled without the use of verbatim ops.
  OpBuilder builder(circuitOp);
  SmallVector<std::pair<LayerOp, StringAttr>> layers;
  circuitOp.walk<mlir::WalkOrder::PreOrder>([&](LayerOp layerOp) {
    auto parentOp = layerOp->getParentOfType<LayerOp>();
    while (!layers.empty() && parentOp != layers.back().first)
      layers.pop_back();

    if (layerOp.getConvention() == LayerConvention::Inline) {
      layers.emplace_back(layerOp, nullptr);
      return;
    }

    builder.setInsertionPointToStart(circuitOp.getBodyBlock());

    // Save the "layers-<circuit>-<group>" and "layers_<circuit>_<group>" string
    // as this is reused a bunch.
    SmallString<32> prefixFile("layers-"), prefixMacro("layers_");
    prefixFile.append(circuitName);
    prefixFile.append("-");
    prefixMacro.append(circuitName);
    prefixMacro.append("_");
    for (auto [layer, _] : layers) {
      prefixFile.append(layer.getSymName());
      prefixFile.append("-");
      prefixMacro.append(layer.getSymName());
      prefixMacro.append("_");
    }
    prefixFile.append(layerOp.getSymName());
    prefixMacro.append(layerOp.getSymName());

    SmallString<128> includes;
    for (auto [_, strAttr] : layers) {
      if (!strAttr)
        continue;
      includes.append(strAttr);
      includes.append("\n");
    }

    hw::OutputFileAttr bindFile;
    if (auto outputFile =
            layerOp->getAttrOfType<hw::OutputFileAttr>("output_file")) {
      auto dir = outputFile.getDirectory();
      bindFile = hw::OutputFileAttr::getFromDirectoryAndFilename(
          &getContext(), dir, prefixFile + ".sv",
          /*excludeFromFileList=*/true);
    } else {
      bindFile = hw::OutputFileAttr::getFromFilename(
          &getContext(), prefixFile + ".sv", /*excludeFromFileList=*/true);
    }

    // Write header to a verbatim.
    auto header = builder.create<sv::VerbatimOp>(
        layerOp.getLoc(),
        includes + "`ifndef " + prefixMacro + "\n" + "`define " + prefixMacro);
    header->setAttr("output_file", bindFile);

    // Write footer to a verbatim.
    builder.setInsertionPointToEnd(circuitOp.getBodyBlock());
    auto footer = builder.create<sv::VerbatimOp>(layerOp.getLoc(),
                                                 "`endif // " + prefixMacro);
    footer->setAttr("output_file", bindFile);

    if (!layerOp.getBody().getOps<LayerOp>().empty())
      layers.emplace_back(
          layerOp,
          builder.getStringAttr("`include \"" +
                                bindFile.getFilename().getValue() + "\""));
  });

  // All layers definitions can now be deleted.
  for (auto layerOp :
       llvm::make_early_inc_range(circuitOp.getBodyBlock()->getOps<LayerOp>()))
    layerOp.erase();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createLowerLayersPass() {
  return std::make_unique<LowerLayersPass>();
}
