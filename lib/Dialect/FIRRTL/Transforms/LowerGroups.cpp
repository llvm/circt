//===- LowerGroups.cpp - Lower Optiona Groups to Instances ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This pass lowers FIRRTL optional groups to modules and instances.
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

#define DEBUG_TYPE "firrtl-lower-groups"

using namespace circt;
using namespace firrtl;

class LowerGroupsPass : public LowerGroupsBase<LowerGroupsPass> {
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

  /// A map of group definition to module name that should be created for it.
  DenseMap<GroupOp, StringRef> moduleNames;
};

/// Multi-process safe function to build a module in the circuit and return it.
/// The name provided is only a namehint for the module---a unique name will be
/// generated if there are conflicts with the namehint in the circuit-level
/// namespace.
FModuleOp LowerGroupsPass::buildNewModule(OpBuilder &builder, Location location,
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

/// Process amodule to remove any groups it has.
void LowerGroupsPass::runOnModule(FModuleOp moduleOp) {
  LLVM_DEBUG({
    llvm::dbgs() << "Module: " << moduleOp.getModuleName() << "\n";
    llvm::dbgs() << "  Examining Groups:\n";
  });

  CircuitOp circuitOp = moduleOp->getParentOfType<CircuitOp>();
  StringRef circuitName = circuitOp.getName();

  // A map of instance ops to modules that this pass creates.  This is used to
  // check if this was an instance that we created and to do fast module
  // dereferencing (avoiding a symbol table).
  DenseMap<InstanceOp, FModuleOp> createdInstances;

  // Post-order traversal that expands a group into its parent. For each group
  // found do the following:
  //
  // 1. Create and connect one ref-type output port for each value defined in
  //    this group that drives an instance marked lowerToBind and move this
  //    instance outside the group.
  // 2. Create one input port for each value captured by this group.
  // 3. Create a new module for this group and move the (mutated) body of this
  //    group to the new module.
  // 4. Instantiate the new module outside the group and hook it up.
  // 5. Erase the group.
  moduleOp.walk<mlir::WalkOrder::PostOrder>([&](Operation *op) {
    auto group = dyn_cast<GroupOp>(op);
    if (!group)
      return WalkResult::advance();

    // Compute the expanded group name.  For group @A::@B::@C, this is "A_B_C".
    SmallString<32> groupName(group.getGroupName().getRootReference());
    for (auto ref : group.getGroupName().getNestedReferences()) {
      groupName.append("_");
      groupName.append(ref.getValue());
    }
    LLVM_DEBUG(llvm::dbgs() << "    - Group: " << groupName << "\n");

    Block *body = group.getBody(0);
    OpBuilder builder(moduleOp);

    // Ports that need to be created for the module derived from this group.
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
      // Update the group's body with arguments as we will swap this body into
      // the module when we create it.
      body->addArgument(operand.getType(), loc);
      operand.replaceUsesWithIf(body->getArgument(portNum),
                                [&](OpOperand &operand) {
                                  return operand.getOwner()->getBlock() == body;
                                });

      connectValues.push_back(operand);
    };

    // Set the location intelligently.  Use the location of the capture if
    // this is a port created for forwarding from a parent group to a nested
    // group.  Otherwise, use unknown.
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
      // Handle instance ops that were created from nested groups.  These ops
      // need to be moved outside the group to avoid nested binds which are
      // illegal in the SystemVerilog specification (and checked by FIRRTL
      // verification).
      //
      // For each value defined in this group which drives a port of one of
      // these instances, create an output reference type port on the
      // to-be-created module and drive it with the value.  Move the instance
      // outside the group. We will hook it up later once we replace the group
      // with an instance.
      if (auto instOp = dyn_cast<InstanceOp>(op)) {
        // Ignore any instance which this pass did not create from a nested
        // group. Instances which are not marked lowerToBind do not need to be
        // split out.
        if (!createdInstances.contains(instOp))
          continue;
        LLVM_DEBUG({
          llvm::dbgs() << "      Found instance created from nested group:\n"
                       << "        module: " << instOp.getModuleName() << "\n"
                       << "        instance: " << instOp.getName() << "\n";
        });
        instOp->moveBefore(group);
        continue;
      }

      if (auto connect = dyn_cast<FConnectLike>(op)) {
        auto srcInGroup = connect.getSrc().getParentBlock() == body;
        auto destInGroup = connect.getDest().getParentBlock() == body;
        if (!srcInGroup && !destInGroup) {
          connect->moveBefore(group);
          continue;
        }
        // Create an input port.
        if (!srcInGroup) {
          createInputPort(connect.getSrc(), op.getLoc());
          continue;
        }
        // Create an output port.
        if (!destInGroup) {
          createOutputPort(connect.getDest(), connect.getSrc());
          connect.erase();
          continue;
        }
        // Source and destination in group.  Nothing to do.
        continue;
      }

      // Pattern match the following structure.  Move the ref.resolve outside
      // the group.  The strictconnect will be moved outside in the next loop
      // iteration:
      //     %0 = ...
      //     %1 = ...
      //     firrtl.group {
      //       %2 = ref.resolve %0
      //       firrtl.strictconnect %1, %2
      //     }
      if (auto refResolve = dyn_cast<RefResolveOp>(op))
        if (refResolve.getResult().hasOneUse())
          if (auto connect = dyn_cast<StrictConnectOp>(
                  *refResolve.getResult().getUsers().begin()))
            if (connect.getDest().getParentBlock() != body) {
              refResolve->moveBefore(group);
              continue;
            }

      // For any other ops, create input ports for any captured operands.
      for (auto operand : op.getOperands())
        if (operand.getParentBlock() != body)
          createInputPort(operand, op.getLoc());
    }

    // Create the new module.  This grabs a lock to modify the circuit.
    FModuleOp newModule = buildNewModule(builder, group.getLoc(),
                                         moduleNames.lookup(group), ports);
    SymbolTable::setSymbolVisibility(newModule,
                                     SymbolTable::Visibility::Private);
    newModule.getBody().takeBody(group.getRegion());

    LLVM_DEBUG({
      llvm::dbgs() << "      New Module: " << moduleNames.lookup(group) << "\n";
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

    // Replace the original group with an instance.  Hook up the instance.
    builder.setInsertionPointAfter(group);
    auto moduleName = newModule.getModuleName();
    auto instanceOp = builder.create<InstanceOp>(
        group.getLoc(), /*moduleName=*/newModule,
        /*name=*/
        (Twine((char)tolower(moduleName[0])) + moduleName.drop_front()).str(),
        NameKindEnum::DroppableName,
        /*annotations=*/ArrayRef<Attribute>{},
        /*portAnnotations=*/ArrayRef<Attribute>{}, /*lowerToBind=*/true);
    instanceOp->setAttr("output_file",
                        hw::OutputFileAttr::getFromFilename(
                            builder.getContext(),
                            "groups_" + circuitName + "_" + groupName + ".sv",
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
    group.erase();

    return WalkResult::advance();
  });
}

/// Process a circuit to remove all groups in each module and top-level group
/// declarations.
void LowerGroupsPass::runOnOperation() {
  LLVM_DEBUG(
      llvm::dbgs() << "==----- Running LowerGroups "
                      "-------------------------------------------------===\n");
  CircuitOp circuitOp = getOperation();

  // Initialize members which cannot be initialized automatically.
  llvm::sys::SmartMutex<true> mutex;
  circuitMutex = &mutex;

  // Determine names for all modules that will be created.  Do this serially to
  // avoid non-determinism from creating these in the parallel region.
  CircuitNamespace ns(circuitOp);
  circuitOp->walk([&](FModuleOp moduleOp) {
    moduleOp->walk([&](GroupOp groupOp) {
      SmallString<32> groupName(groupOp.getGroupName().getRootReference());
      for (auto ref : groupOp.getGroupName().getNestedReferences()) {
        groupName.append("_");
        groupName.append(ref.getValue());
      }
      moduleNames.insert(
          {groupOp, ns.newName(moduleOp.getModuleName() + "_" + groupName)});
    });
  });

  // Early exit if no work to do.
  if (moduleNames.empty())
    return markAllAnalysesPreserved();

  // Lower the groups of each module.
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
  // TODO: This would be better handled without the use of verbatim ops.
  OpBuilder builder(circuitOp);
  SmallVector<std::pair<GroupDeclOp, StringAttr>> groupDecls;
  StringRef circuitName = circuitOp.getName();
  circuitOp.walk<mlir::WalkOrder::PreOrder>([&](GroupDeclOp groupDeclOp) {
    auto parentOp = groupDeclOp->getParentOfType<GroupDeclOp>();
    while (parentOp && parentOp != groupDecls.back().first)
      groupDecls.pop_back();
    builder.setInsertionPointToStart(circuitOp.getBodyBlock());

    // Save the "groups_CIRCUIT_GROUP" string as this is reused a bunch.
    SmallString<32> prefix("groups_");
    prefix.append(circuitName);
    prefix.append("_");
    for (auto [group, _] : groupDecls) {
      prefix.append(group.getSymName());
      prefix.append("_");
    }
    prefix.append(groupDeclOp.getSymName());

    auto outputFileAttr = hw::OutputFileAttr::getFromFilename(
        builder.getContext(), prefix + ".sv",
        /*excludeFromFileList=*/true);

    SmallString<128> includes;
    for (auto [_, strAttr] : groupDecls) {
      includes.append(strAttr);
      includes.append("\n");
    }

    // Write header to a verbatim.
    builder
        .create<sv::VerbatimOp>(groupDeclOp.getLoc(), includes + "`ifndef " +
                                                          prefix + "\n" +
                                                          "`define " + prefix)
        ->setAttr("output_file", outputFileAttr);

    // Write footer to a verbatim.
    builder.setInsertionPointToEnd(circuitOp.getBodyBlock());
    builder.create<sv::VerbatimOp>(groupDeclOp.getLoc(), "`endif // " + prefix)
        ->setAttr("output_file", outputFileAttr);

    if (!groupDeclOp.getBody().getOps<GroupDeclOp>().empty())
      groupDecls.push_back(
          {groupDeclOp,
           builder.getStringAttr("`include \"" + prefix + ".sv\"")});
  });

  // All group declarations can now be deleted.
  for (auto groupDeclOp : llvm::make_early_inc_range(
           circuitOp.getBodyBlock()->getOps<GroupDeclOp>()))
    groupDeclOp.erase();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createLowerGroupsPass() {
  return std::make_unique<LowerGroupsPass>();
}
