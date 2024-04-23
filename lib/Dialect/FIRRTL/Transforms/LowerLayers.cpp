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

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/InnerSymbolNamespace.h"
#include "circt/Dialect/SV/SVOps.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/RWMutex.h"

#define DEBUG_TYPE "firrtl-lower-layers"

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static bool isAncestor(Operation *op, Value value) {
  if (auto result = dyn_cast<OpResult>(value))
    return op->isAncestor(result.getOwner());
  auto argument = cast<BlockArgument>(value);
  return op->isAncestor(argument.getOwner()->getParentOp());
}

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
                       bool toLower = false) {
  if (name.empty())
    return;
  if (!output.empty())
    output.append("_");
  output.append(name);
  if (!toLower)
    return;
  auto i = output.size() - name.size();
  output[i] = llvm::toLower(output[i]);
}

static void appendName(SymbolRefAttr name, SmallString<32> &output,
                       bool toLower = false) {
  appendName(name.getRootReference(), output, toLower);
  for (auto nested : name.getNestedReferences())
    appendName(nested.getValue(), output, toLower);
}

/// For a layer `@A::@B::@C` in module Module,
/// the generated module is called `Module_A_B_C`.
static SmallString<32> moduleNameForLayer(StringRef moduleName,
                                          SymbolRefAttr layerName) {
  SmallString<32> result;
  appendName(moduleName, result);
  appendName(layerName, result);
  return result;
}

/// For a layerblock `@A::@B::@C`,
/// the generated instance is called `a_b_c`.
static SmallString<32> instanceNameForLayer(SymbolRefAttr layerName) {
  SmallString<32> result;
  appendName(layerName, result, /*toLower=*/true);
  return result;
}

/// For all layerblocks `@A::@B::@C` in a circuit called Circuit,
/// the output filename is `layers_Circuit_A_B_C.sv`.
static SmallString<32> fileNameForLayer(StringRef circuitName,
                                        SymbolRefAttr layerName) {
  SmallString<32> result;
  result.append("layers");
  appendName(circuitName, result);
  appendName(layerName, result);
  result.append(".sv");
  return result;
}

//===----------------------------------------------------------------------===//
// LowerLayersPass
//===----------------------------------------------------------------------===//

class LowerLayersPass : public LowerLayersBase<LowerLayersPass> {
  /// Safely build a new module with a given namehint.  This handles geting a
  /// lock to modify the top-level circuit.
  FModuleOp buildNewModule(OpBuilder &builder, Location location,
                           Twine namehint, SmallVectorImpl<PortInfo> &ports);

  /// Strip layer colors from the module's interface.
  InnerRefMap runOnModuleLike(FModuleLike moduleLike);

  /// Extract layerblocks and strip probe colors from all ops under the module.
  void runOnModuleBody(FModuleOp moduleOp, InnerRefMap &innerRefMap);

  /// Update the module's port types to remove any explicit layer requirements
  /// from any probe types.
  void removeLayersFromPorts(FModuleLike moduleLike);

  /// Update the value's type to remove any layers from any probe types.
  void removeLayersFromValue(Value value);

  /// Remove any layers from the result of the cast. If the cast becomes a nop,
  /// remove the cast itself from the IR.
  void removeLayersFromRefCast(RefCastOp cast);

  /// Set an output file attribute pointing at the testbench directory if a
  /// testbench directory is known to the pass.
  void maybeSetOutputDir(Operation *op);

  /// Set an output file attribute pointing at the provided filename.  Prepend
  /// the testbench directory if one is known to the pass.
  void setOutputFile(Operation *op, const Twine &filename);

  /// Entry point for the function.
  void runOnOperation() override;

  /// Indicates exclusive access to modify the circuitNamespace and the circuit.
  llvm::sys::SmartMutex<true> *circuitMutex;

  /// A map of layer blocks to module name that should be created for it.
  DenseMap<LayerBlockOp, StringRef> moduleNames;

  /// The design-under-test (DUT) as indicated by the presence of a
  /// "sifive.enterprise.firrtl.MarkDUTAnnotation".  This will be null if no
  /// annotation is present.
  FModuleOp dut;

  /// The directory for verification collateral as indicated by the presence of
  /// a "sifive.enterprise.firrtl.TestBenchDirAnnotation".  Empty if not
  /// specified.
  StringRef testBenchDir;
};

/// Multi-process safe function to build a module in the circuit and return it.
/// The name provided is only a namehint for the module---a unique name will be
/// generated if there are conflicts with the namehint in the circuit-level
/// namespace.
FModuleOp LowerLayersPass::buildNewModule(OpBuilder &builder, Location location,
                                          Twine namehint,
                                          SmallVectorImpl<PortInfo> &ports) {
  llvm::sys::SmartScopedLock<true> instrumentationLock(*circuitMutex);
  FModuleOp newModule = builder.create<FModuleOp>(
      location, builder.getStringAttr(namehint),
      ConventionAttr::get(builder.getContext(), Convention::Internal), ports,
      ArrayAttr{});
  maybeSetOutputDir(newModule);
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

  moduleLike->setAttr(FModuleLike::getPortTypesAttrName(),
                      ArrayAttr::get(moduleLike.getContext(), newTypeAttrs));

  if (auto moduleOp = dyn_cast<FModuleOp>(moduleLike.getOperation())) {
    for (auto arg : moduleOp.getBodyBlock()->getArguments())
      removeLayersFromValue(arg);
  }
}

InnerRefMap LowerLayersPass::runOnModuleLike(FModuleLike moduleLike) {
  LLVM_DEBUG({
    llvm::dbgs() << "Module: " << moduleLike.getModuleName() << "\n";
    llvm::dbgs() << "  Examining Layer Blocks:\n";
  });

  // Strip away layers from the interface of the module-like op.
  InnerRefMap innerRefMap;
  TypeSwitch<Operation *, void>(moduleLike.getOperation())
      .Case<FModuleOp>([&](auto op) {
        op.setLayers({});
        removeLayersFromPorts(op);
        runOnModuleBody(op, innerRefMap);
      })
      .Case<FExtModuleOp, FIntModuleOp, FMemModuleOp>([&](auto op) {
        op.setLayers({});
        removeLayersFromPorts(op);
      })
      .Case<ClassOp, ExtClassOp>([](auto) {})
      .Default([](auto) { assert(0 && "unknown module-like op"); });

  return innerRefMap;
}

void LowerLayersPass::runOnModuleBody(FModuleOp moduleOp,
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
  moduleOp.walk<mlir::WalkOrder::PostOrder>([&](Operation *op) {
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
                       /*sym=*/{},
                       /*loc=*/loc});
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
                       /*sym=*/{}, /*loc=*/loc});
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
    for (auto &op : llvm::make_early_inc_range(*body)) {
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
          continue;

        LLVM_DEBUG({
          llvm::dbgs()
              << "      Found instance created from nested layer block:\n"
              << "        module: " << instOp.getModuleName() << "\n"
              << "        instance: " << instOp.getName() << "\n";
        });
        instOp->moveBefore(layerBlock);
        continue;
      }

      if (auto refSend = dyn_cast<RefSendOp>(op)) {
        auto src = refSend.getBase();
        if (!isAncestor(layerBlock, src))
          createInputPort(src, op.getLoc());
        continue;
      }

      if (auto refCast = dyn_cast<RefCastOp>(op)) {
        if (!isAncestor(layerBlock, refCast))
          createInputPort(refCast.getInput(), op.getLoc());
        continue;
      }

      if (auto connect = dyn_cast<FConnectLike>(op)) {
        auto src = connect.getSrc();
        auto dst = connect.getDest();
        auto srcInLayerBlock = isAncestor(layerBlock, src);
        auto dstInLayerBlock = isAncestor(layerBlock, dst);
        if (!srcInLayerBlock && !dstInLayerBlock) {
          connect->moveBefore(layerBlock);
          continue;
        }
        // Create an input port.
        if (!srcInLayerBlock) {
          createInputPort(src, op.getLoc());
          continue;
        }
        // Create an output port.
        if (!dstInLayerBlock) {
          createOutputPort(dst, src);
          if (!isa<RefType>(dst.getType()))
            connect.erase();
          continue;
        }
        // Source and destination in layer block.  Nothing to do.
        continue;
      }

      // Pre-emptively de-squiggle connections that we are creating.  This will
      // later be cleaned up by the de-squiggling pass.  However, there is no
      // point in creating deeply squiggled connections if we don't have to.
      //
      // This pattern matches the following structure.  Move the ref.resolve
      // outside the layer block.  The strictconnect will be moved outside in
      // the next loop iteration:
      //     %0 = ...
      //     %1 = ...
      //     firrtl.layerblock {
      //       %2 = ref.resolve %0
      //       firrtl.strictconnect %1, %2
      //     }
      if (auto refResolve = dyn_cast<RefResolveOp>(op))
        if (refResolve.getResult().hasOneUse() &&
            refResolve.getRef().getParentBlock() != body)
          if (auto connect = dyn_cast<StrictConnectOp>(
                  *refResolve.getResult().getUsers().begin()))
            if (connect.getDest().getParentBlock() != body) {
              refResolve->moveBefore(layerBlock);
              continue;
            }

      // For any other ops, create input ports for any captured operands.
      for (auto operand : op.getOperands()) {
        if (!isAncestor(layerBlock, operand))
          createInputPort(operand, op.getLoc());
      }
    }

    // Create the new module.  This grabs a lock to modify the circuit.
    FModuleOp newModule = buildNewModule(builder, layerBlock.getLoc(),
                                         moduleNames.lookup(layerBlock), ports);
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

    auto fileName = fileNameForLayer(circuitName, layerBlock.getLayerName());
    setOutputFile(instanceOp, fileName);

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
        builder.create<StrictConnectOp>(newModule.getPortLocationAttr(portNum),
                                        instanceOp.getResult(portNum), src);
      } else if (isa<RefType>(instanceOp.getResult(portNum).getType()) &&
                 connectValues[portNum].kind == ConnectKind::Ref)
        builder.create<RefDefineOp>(getPortLoc(connectValues[portNum].value),
                                    connectValues[portNum].value,
                                    instanceOp.getResult(portNum));
      else
        builder.create<StrictConnectOp>(
            getPortLoc(connectValues[portNum].value),
            connectValues[portNum].value,
            builder.create<RefResolveOp>(newModule.getPortLocationAttr(portNum),
                                         instanceOp.getResult(portNum)));
    }
    layerBlock.erase();

    return WalkResult::advance();
  });
}

/// Set an output file attribute on the provided operation if the pass has a
/// non-empty test bench directory.
void LowerLayersPass::maybeSetOutputDir(Operation *op) {
  if (!testBenchDir.empty() && dut)
    op->setAttr("output_file", hw::OutputFileAttr::getAsDirectory(
                                   op->getContext(), testBenchDir,
                                   /*excludeFromFileList=*/true));
}

void LowerLayersPass::setOutputFile(Operation *op, const Twine &filename) {
  if (!testBenchDir.empty() && dut) {
    op->setAttr("output_file", hw::OutputFileAttr::getFromDirectoryAndFilename(
                                   op->getContext(), testBenchDir, filename,
                                   /*excludeFromFileList=*/true));

    return;
  }
  op->setAttr("output_file", hw::OutputFileAttr::getFromFilename(
                                 op->getContext(), filename,
                                 /*excludeFromFileList=*/true));
}

/// Process a circuit to remove all layer blocks in each module and top-level
/// layer definition.
void LowerLayersPass::runOnOperation() {
  LLVM_DEBUG(
      llvm::dbgs() << "==----- Running LowerLayers "
                      "-------------------------------------------------===\n");
  CircuitOp circuitOp = getOperation();

  // Initialize members which cannot be initialized automatically.
  llvm::sys::SmartMutex<true> mutex;
  circuitMutex = &mutex;

  // Extract information from circuit-level annotations.
  for (auto anno : AnnotationSet(circuitOp)) {
    if (!anno.isClass(testBenchDirAnnoClass))
      continue;
    auto dir = anno.getMember<StringAttr>("dirname");
    assert(dir && "invalid test bench annotation");
    testBenchDir = dir.getValue();
  }

  // Determine names for all modules that will be created.  Do this serially to
  // avoid non-determinism from creating these in the parallel region.  While
  // walking modules, set the DUT if it is so marked.
  CircuitNamespace ns(circuitOp);
  WalkResult result = circuitOp->walk([&](FModuleOp moduleOp) {
    if (failed(extractDUT(moduleOp, dut)))
      return WalkResult::interrupt();
    moduleOp->walk([&](LayerBlockOp layerBlockOp) {
      auto name = moduleNameForLayer(moduleOp.getModuleName(),
                                     layerBlockOp.getLayerName());
      moduleNames.insert({layerBlockOp, ns.newName(name)});
    });
    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return signalPassFailure();

  auto mergeMaps = [](auto &&a, auto &&b) {
    for (auto bb : b)
      a.insert(bb);
    return std::forward<decltype(a)>(a);
  };

  // Lower the layer blocks of each module.
  SmallVector<FModuleLike> modules(
      circuitOp.getBodyBlock()->getOps<FModuleLike>());
  auto innerRefMap =
      transformReduce(circuitOp.getContext(), modules, InnerRefMap{}, mergeMaps,
                      [this](FModuleLike mod) { return runOnModuleLike(mod); });

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
          hw::InnerRefAttr::get(innerRef.getRoot(), inst.getSymName()));
      newNamepath.push_back(hw::InnerRefAttr::get(mod, innerRef.getTarget()));
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
  //     `include "layers_A.sv"
  //     `include "layers_A_B.sv"
  //     `ifndef layers_A_B_C
  //     `define layers_A_B_C
  //     <body>
  //     `endif // layers_A_B_C
  //
  // TODO: This would be better handled without the use of verbatim ops.
  OpBuilder builder(circuitOp);
  SmallVector<std::pair<LayerOp, StringAttr>> layers;
  StringRef circuitName = circuitOp.getName();
  circuitOp.walk<mlir::WalkOrder::PreOrder>([&](LayerOp layerOp) {
    auto parentOp = layerOp->getParentOfType<LayerOp>();
    while (parentOp && parentOp != layers.back().first)
      layers.pop_back();
    builder.setInsertionPointToStart(circuitOp.getBodyBlock());

    // Save the "layers_CIRCUIT_GROUP" string as this is reused a bunch.
    SmallString<32> prefix("layers_");
    prefix.append(circuitName);
    prefix.append("_");
    for (auto [layer, _] : layers) {
      prefix.append(layer.getSymName());
      prefix.append("_");
    }
    prefix.append(layerOp.getSymName());

    SmallString<128> includes;
    for (auto [_, strAttr] : layers) {
      includes.append(strAttr);
      includes.append("\n");
    }

    // Write header to a verbatim.
    setOutputFile(builder.create<sv::VerbatimOp>(
                      layerOp.getLoc(), includes + "`ifndef " + prefix + "\n" +
                                            "`define " + prefix),
                  prefix + ".sv");

    // Write footer to a verbatim.
    builder.setInsertionPointToEnd(circuitOp.getBodyBlock());
    setOutputFile(
        builder.create<sv::VerbatimOp>(layerOp.getLoc(), "`endif // " + prefix),
        prefix + ".sv");

    if (!layerOp.getBody().getOps<LayerOp>().empty())
      layers.push_back(
          {layerOp, builder.getStringAttr("`include \"" + prefix + ".sv\"")});
  });

  // All layers definitions can now be deleted.
  for (auto layerOp :
       llvm::make_early_inc_range(circuitOp.getBodyBlock()->getOps<LayerOp>()))
    layerOp.erase();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createLowerLayersPass() {
  return std::make_unique<LowerLayersPass>();
}
