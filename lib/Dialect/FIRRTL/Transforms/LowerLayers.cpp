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
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/RWMutex.h"

#define DEBUG_TYPE "firrtl-lower-layers"

using namespace circt;
using namespace firrtl;

class LowerLayersPass : public LowerLayersBase<LowerLayersPass> {
  /// Safely build a new module with a given namehint.  This handles geting a
  /// lock to modify the top-level circuit.
  FModuleOp buildNewModule(OpBuilder &builder, Location location,
                           Twine namehint, SmallVectorImpl<PortInfo> &ports);

  /// Function to process each module.
  void runOnModule(FModuleOp moduleOp);

  /// Entry point for the function.
  void runOnOperation() override;

  /// Indicates exclusive access to modify the circuitNamespace and the circuit.
  llvm::sys::SmartMutex<true> *circuitMutex;

  /// A map of layer blocks to module name that should be created for it.
  DenseMap<LayerBlockOp, StringRef> moduleNames;
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
  SymbolTable::setSymbolVisibility(newModule, SymbolTable::Visibility::Private);
  return newModule;
}

/// Process a module to remove any layer blocks it has.
void LowerLayersPass::runOnModule(FModuleOp moduleOp) {
  LLVM_DEBUG({
    llvm::dbgs() << "Module: " << moduleOp.getModuleName() << "\n";
    llvm::dbgs() << "  Examining Layer Blocks:\n";
  });

  CircuitOp circuitOp = moduleOp->getParentOfType<CircuitOp>();
  StringRef circuitName = circuitOp.getName();

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
    auto layerBlock = dyn_cast<LayerBlockOp>(op);
    if (!layerBlock)
      return WalkResult::advance();

    // Compute the expanded layer name.  For layer @A::@B::@C, this is "A_B_C".
    SmallString<32> layerName(layerBlock.getLayerName().getRootReference());
    for (auto ref : layerBlock.getLayerName().getNestedReferences()) {
      layerName.append("_");
      layerName.append(ref.getValue());
    }
    LLVM_DEBUG(llvm::dbgs() << "    - Layer: " << layerName << "\n");

    Block *body = layerBlock.getBody(0);
    OpBuilder builder(moduleOp);

    // Ports that need to be created for the module derived from this layer
    // block.
    SmallVector<PortInfo> ports;

    // Connectsion that need to be made to the instance of the derived module.
    SmallVector<Value> connectValues;

    // Create an input port for an operand that is captured from outside.
    //
    // TODO: If we allow capturing reference types, this will need to be
    // updated.
    auto createInputPort = [&](Value operand, Location loc) {
      auto portNum = ports.size();
      auto operandName = getFieldName(FieldRef(operand, 0), true);

      ports.push_back({builder.getStringAttr("_" + operandName.first),
                       operand.getType(), Direction::In, /*sym=*/{},
                       /*loc=*/loc});
      // Update the layer block's body with arguments as we will swap this body
      // into the module when we create it.
      body->addArgument(operand.getType(), loc);
      operand.replaceUsesWithIf(body->getArgument(portNum),
                                [&](OpOperand &operand) {
                                  return operand.getOwner()->getBlock() == body;
                                });

      connectValues.push_back(operand);
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
            auto portNum = port.cast<OpResult>().getResultNumber();
            loc = modOpIt->getSecond().getPortLocation(portNum);
          }
        }
      return loc;
    };

    // Create an output probe port port and adds the ref.define/ref.send to
    // drive the port.
    auto createOutputPort = [&](Value dest, Value src) {
      auto loc = getPortLoc(dest);
      auto portNum = ports.size();
      auto operandName = getFieldName(FieldRef(dest, 0), true);

      auto refType = RefType::get(
          type_cast<FIRRTLBaseType>(dest.getType()).getPassiveType(),
          /*forceable=*/false);

      ports.push_back({builder.getStringAttr("_" + operandName.first), refType,
                       Direction::Out, /*sym=*/{}, /*loc=*/loc});
      body->addArgument(refType, loc);
      connectValues.push_back(dest);
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfterValue(src);
      builder.create<RefDefineOp>(
          loc, body->getArgument(portNum),
          builder.create<RefSendOp>(loc, src)->getResult(0));
    };

    for (auto &op : llvm::make_early_inc_range(*body)) {
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
        // Ignore any instance which this pass did not create from a nested
        // layer block. Instances which are not marked lowerToBind do not need
        // to be split out.
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

      if (auto connect = dyn_cast<FConnectLike>(op)) {
        auto srcInLayerBlock = connect.getSrc().getParentBlock() == body;
        auto destInLayerBlock = connect.getDest().getParentBlock() == body;
        if (!srcInLayerBlock && !destInLayerBlock) {
          connect->moveBefore(layerBlock);
          continue;
        }
        // Create an input port.
        if (!srcInLayerBlock) {
          createInputPort(connect.getSrc(), op.getLoc());
          continue;
        }
        // Create an output port.
        if (!destInLayerBlock) {
          createOutputPort(connect.getDest(), connect.getSrc());
          connect.erase();
          continue;
        }
        // Source and destination in layer block.  Nothing to do.
        continue;
      }

      // Pattern match the following structure.  Move the ref.resolve outside
      // the layer block.  The strictconnect will be moved outside in the next
      // loop iteration:
      //     %0 = ...
      //     %1 = ...
      //     firrtl.layerblock {
      //       %2 = ref.resolve %0
      //       firrtl.strictconnect %1, %2
      //     }
      if (auto refResolve = dyn_cast<RefResolveOp>(op))
        if (refResolve.getResult().hasOneUse())
          if (auto connect = dyn_cast<StrictConnectOp>(
                  *refResolve.getResult().getUsers().begin()))
            if (connect.getDest().getParentBlock() != body) {
              refResolve->moveBefore(layerBlock);
              continue;
            }

      // For any other ops, create input ports for any captured operands.
      for (auto operand : op.getOperands())
        if (operand.getParentBlock() != body)
          createInputPort(operand, op.getLoc());
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
                     << "            value: " << value << "\n";
      }
    });

    // Replace the original layer block with an instance.  Hook up the instance.
    builder.setInsertionPointAfter(layerBlock);
    auto moduleName = newModule.getModuleName();
    auto instanceOp = builder.create<InstanceOp>(
        layerBlock.getLoc(), /*moduleName=*/newModule,
        /*name=*/
        (Twine((char)tolower(moduleName[0])) + moduleName.drop_front()).str(),
        NameKindEnum::DroppableName,
        /*annotations=*/ArrayRef<Attribute>{},
        /*portAnnotations=*/ArrayRef<Attribute>{}, /*lowerToBind=*/true);
    // TODO: Change this to "layers_" once we switch to FIRRTL 4.0.0+.
    instanceOp->setAttr("output_file",
                        hw::OutputFileAttr::getFromFilename(
                            builder.getContext(),

                            "groups_" + circuitName + "_" + layerName + ".sv",
                            /*excludeFromFileList=*/true));
    createdInstances.try_emplace(instanceOp, newModule);

    // Connect instance ports to values.
    assert(ports.size() == connectValues.size() &&
           "the number of instance ports and values to connect to them must be "
           "equal");
    for (unsigned portNum = 0, e = newModule.getNumPorts(); portNum < e;
         ++portNum) {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfterValue(instanceOp.getResult(portNum));
      if (instanceOp.getPortDirection(portNum) == Direction::In)
        builder.create<StrictConnectOp>(newModule.getPortLocationAttr(portNum),
                                        instanceOp.getResult(portNum),
                                        connectValues[portNum]);
      else
        builder.create<StrictConnectOp>(
            getPortLoc(connectValues[portNum]), connectValues[portNum],
            builder.create<RefResolveOp>(newModule.getPortLocationAttr(portNum),
                                         instanceOp.getResult(portNum)));
    }
    layerBlock.erase();

    return WalkResult::advance();
  });
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

  // Determine names for all modules that will be created.  Do this serially to
  // avoid non-determinism from creating these in the parallel region.
  CircuitNamespace ns(circuitOp);
  circuitOp->walk([&](FModuleOp moduleOp) {
    moduleOp->walk([&](LayerBlockOp layerBlockOp) {
      SmallString<32> layerName(layerBlockOp.getLayerName().getRootReference());
      for (auto ref : layerBlockOp.getLayerName().getNestedReferences()) {
        layerName.append("_");
        layerName.append(ref.getValue());
      }
      moduleNames.insert({layerBlockOp, ns.newName(moduleOp.getModuleName() +
                                                   "_" + layerName)});
    });
  });

  // Early exit if no work to do.
  if (moduleNames.empty())
    return markAllAnalysesPreserved();

  // Lower the layer blocks of each module.
  SmallVector<FModuleOp> modules(circuitOp.getBodyBlock()->getOps<FModuleOp>());
  llvm::parallelForEach(modules,
                        [&](FModuleOp moduleOp) { runOnModule(moduleOp); });

  // Generate the header and footer of each bindings file.  The body will be
  // populated later when binds are exported to Verilog.  This produces text
  // like:
  //
  //     `include "groups_A.sv"
  //     `include "groups_A_B.sv"
  //     `ifndef groups_A_B_C
  //     `define groups_A_B_C
  //     <body>
  //     `endif // groups_A_B_C
  //
  // TODO: Change this comment to "layers_" once we switch to FIRRTL 4.0.0+.
  // TODO: This would be better handled without the use of verbatim ops.
  OpBuilder builder(circuitOp);
  SmallVector<std::pair<LayerOp, StringAttr>> layers;
  StringRef circuitName = circuitOp.getName();
  circuitOp.walk<mlir::WalkOrder::PreOrder>([&](LayerOp layerOp) {
    auto parentOp = layerOp->getParentOfType<LayerOp>();
    while (parentOp && parentOp != layers.back().first)
      layers.pop_back();
    builder.setInsertionPointToStart(circuitOp.getBodyBlock());

    // Save the "groups_CIRCUIT_GROUP" string as this is reused a bunch.
    // TODO: Change this to "layers_" once we switch to FIRRTL 4.0.0+.
    SmallString<32> prefix("groups_");
    prefix.append(circuitName);
    prefix.append("_");
    for (auto [layer, _] : layers) {
      prefix.append(layer.getSymName());
      prefix.append("_");
    }
    prefix.append(layerOp.getSymName());

    auto outputFileAttr = hw::OutputFileAttr::getFromFilename(
        builder.getContext(), prefix + ".sv",
        /*excludeFromFileList=*/true);

    SmallString<128> includes;
    for (auto [_, strAttr] : layers) {
      includes.append(strAttr);
      includes.append("\n");
    }

    // Write header to a verbatim.
    builder
        .create<sv::VerbatimOp>(layerOp.getLoc(), includes + "`ifndef " +
                                                      prefix + "\n" +
                                                      "`define " + prefix)
        ->setAttr("output_file", outputFileAttr);

    // Write footer to a verbatim.
    builder.setInsertionPointToEnd(circuitOp.getBodyBlock());
    builder.create<sv::VerbatimOp>(layerOp.getLoc(), "`endif // " + prefix)
        ->setAttr("output_file", outputFileAttr);

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
