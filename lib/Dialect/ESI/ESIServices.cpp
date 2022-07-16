//===- ESIServices.cpp - Code related to ESI services ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/ESI/ESIServices.h"
#include "circt/Dialect/ESI/ESITypes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/MSFT/MSFTPasses.h"

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

#include <memory>

using namespace circt;
using namespace circt::esi;

static DenseMap<Attribute, ServiceGeneratorDispatcher::ServiceGeneratorFunc>
DefaultTableCreator(MLIRContext *);

ServiceGeneratorDispatcher ServiceGeneratorDispatcher::defaultDispatcher() {
  return ServiceGeneratorDispatcher(DefaultTableCreator, false);
}

LogicalResult ServiceGeneratorDispatcher::generate(ServiceImplementReqOp req) {
  if (!lookupGen.hasValue()) {
    lookupGen = create(req.getContext());
  }

  auto genF = lookupGen->find(req.identifierAttr());
  if (genF == lookupGen->end()) {
    if (failIfNotFound)
      return req.emitOpError("Could not find service generator for attribute '")
             << req.identifierAttr() << "'";
    return success();
  }
  return genF->second(req);
}

LogicalResult instantiateCosimEndpoints(ServiceImplementReqOp req) {
  auto *ctxt = req.getContext();
  OpBuilder b(req);
  Block *portReqs = &req.portReqs().getBlocks().front();
  Value clk = req.getOperand(0);
  Value rst = req.getOperand(1);
  uint64_t epIdCtr = 0;

  auto toStringAttr = [&](ArrayAttr strArr) {
    std::string buff;
    llvm::raw_string_ostream os(buff);
    llvm::interleave(llvm::map_range(strArr.getAsRange<StringAttr>(),
                                     [](StringAttr s) { return s.getValue(); }),
                     os, ".");
    return StringAttr::get(ctxt, os.str());
  };

  unsigned clientReqIdx = 0;
  for (auto toClientReq :
       llvm::make_early_inc_range(req.getOps<RequestToClientConnection>())) {
    auto cosimIn = b.create<NullSourceOp>(
        toClientReq.getLoc(), ChannelPort::get(ctxt, b.getI1Type()));
    auto cosim = b.create<CosimEndpoint>(toClientReq.getLoc(),
                                         toClientReq.receiving().getType(), clk,
                                         rst, cosimIn, ++epIdCtr);
    cosim->setAttr("name", toStringAttr(toClientReq.clientNamePath()));
    req.getResult(clientReqIdx).replaceAllUsesWith(cosim.recv());
    toClientReq.erase();
    ++clientReqIdx;
  }

  BlockAndValueMapping argMap;
  for (unsigned i = 0, e = portReqs->getArguments().size(); i < e; ++i)
    argMap.map(portReqs->getArgument(i), req->getOperand(i + 2));

  for (auto toServerReq :
       llvm::make_early_inc_range(req.getOps<RequestToServerConnection>())) {
    auto cosim = b.create<CosimEndpoint>(
        toServerReq.getLoc(), ChannelPort::get(ctxt, b.getI1Type()), clk, rst,
        argMap.lookup(toServerReq.sending()), ++epIdCtr);

    cosim->setAttr("name", toStringAttr(toServerReq.clientNamePath()));
    toServerReq.erase();
  }

  req.erase();
  return success();
}

static DenseMap<Attribute, ServiceGeneratorDispatcher::ServiceGeneratorFunc>
DefaultTableCreator(MLIRContext *ctxt) {
  DenseMap<Attribute, ServiceGeneratorDispatcher::ServiceGeneratorFunc> lut;
  lut[StringAttr::get(ctxt, "cosim")] = instantiateCosimEndpoints;
  return lut;
}

//===----------------------------------------------------------------------===//
// Wire up services pass.
//===----------------------------------------------------------------------===//

namespace {
/// Implements a pass to connect up ESI services clients to the nearest server
/// instantiation. Wires up the ports and generates a generation request to
/// call a user-specified generator.
struct ESIConnectServicesPass
    : public ESIConnectServicesBase<ESIConnectServicesPass>,
      msft::PassCommon {

  ESIConnectServicesPass(ServiceGeneratorDispatcher gen) : genDispatcher(gen) {}
  ESIConnectServicesPass()
      : genDispatcher(ServiceGeneratorDispatcher::defaultDispatcher()) {}

  void runOnOperation() override;

  /// "Bubble up" the specified requests to all of the instantiations of the
  /// module specified. Create and connect up ports to tunnel the ESI channels
  /// through.
  LogicalResult surfaceReqs(hw::HWMutableModuleLike,
                            ArrayRef<RequestToClientConnection>,
                            ArrayRef<RequestToServerConnection>);

  /// For any service which is "local" (provides the requested service) in a
  /// module, replace it with a ServiceImplementOp. Said op is to be replaced
  /// with an instantiation by a generator.
  LogicalResult replaceInst(ServiceInstanceOp, Block *portReqs);

  /// Figure out which requests are "local" vs need to be surfaced. Call
  /// 'surfaceReqs' and/or 'replaceInst' as appropriate.
  LogicalResult process(hw::HWMutableModuleLike);

private:
  ServiceGeneratorDispatcher genDispatcher;
};
} // anonymous namespace

void ESIConnectServicesPass::runOnOperation() {
  ModuleOp outerMod = getOperation();

  topLevelSyms.addDefinitions(outerMod);
  if (failed(verifyInstances(outerMod))) {
    signalPassFailure();
    return;
  }

  // Get a partially-ordered list of modules based on the instantiation DAG.
  // It's _very_ important that we process modules before their instantiations
  // so that the modules where they're instantiated correctly process the
  // surfaced connections.
  SmallVector<hw::HWModuleLike, 64> sortedMods;
  getAndSortModules(outerMod, sortedMods);

  // Process each module.
  for (auto mod : sortedMods) {
    hw::HWMutableModuleLike mutableMod =
        dyn_cast<hw::HWMutableModuleLike>(*mod);
    if (mutableMod && failed(process(mutableMod))) {
      signalPassFailure();
      return;
    }
  }
}

LogicalResult ESIConnectServicesPass::process(hw::HWMutableModuleLike mod) {
  Block &modBlock = mod->getRegion(0).front();

  // Index the local services and create blocks in which to put the requests.
  DenseMap<SymbolRefAttr, Block *> localImplReqs;
  for (auto instOp : modBlock.getOps<ServiceInstanceOp>())
    localImplReqs[instOp.service_symbolAttr()] = new Block();

  // Find all of the "local" requests.
  mod.walk([&](RequestToClientConnection req) {
    auto service = req.servicePortAttr().getModuleRef();
    auto implOpF = localImplReqs.find(service);
    if (implOpF != localImplReqs.end())
      req->moveBefore(implOpF->second, implOpF->second->end());
  });
  mod.walk([&](RequestToServerConnection req) {
    auto service = req.servicePortAttr().getModuleRef();
    auto implOpF = localImplReqs.find(service);
    if (implOpF != localImplReqs.end())
      req->moveBefore(implOpF->second, implOpF->second->end());
  });

  // Replace each service instance with a generation request. If a service
  // generator is registered, generate the server.
  for (auto instOp :
       llvm::make_early_inc_range(modBlock.getOps<ServiceInstanceOp>())) {
    Block *portReqs = localImplReqs[instOp.service_symbolAttr()];
    if (failed(replaceInst(instOp, portReqs)))
      return failure();
  }

  // Identify the non-local reqs which need to be surfaced from this module.
  SmallVector<RequestToClientConnection, 4> nonLocalToClientReqs;
  SmallVector<RequestToServerConnection, 4> nonLocalToServerReqs;
  mod.walk([&](RequestToClientConnection req) {
    auto service = req.servicePortAttr().getModuleRef();
    auto implOpF = localImplReqs.find(service);
    if (implOpF == localImplReqs.end())
      nonLocalToClientReqs.push_back(req);
  });
  mod.walk([&](RequestToServerConnection req) {
    auto service = req.servicePortAttr().getModuleRef();
    auto implOpF = localImplReqs.find(service);
    if (implOpF == localImplReqs.end())
      nonLocalToServerReqs.push_back(req);
  });

  // Surface all of the requests which cannot be fulfilled locally.
  if (nonLocalToClientReqs.empty() && nonLocalToServerReqs.empty())
    return success();
  return surfaceReqs(mod, nonLocalToClientReqs, nonLocalToServerReqs);
}

LogicalResult ESIConnectServicesPass::replaceInst(ServiceInstanceOp instOp,
                                                  Block *portReqs) {
  assert(portReqs);

  // Compute the result types for the new op -- the instance op's output types
  // + the to_client types.
  SmallVector<Type, 8> resultTypes(instOp.getResultTypes().begin(),
                                   instOp.getResultTypes().end());
  for (auto toClient : portReqs->getOps<RequestToClientConnection>())
    resultTypes.push_back(toClient.receiving().getType());

  // Compute the operands for the new op -- the instance op's operands + the
  // to_server types. Reassign the reqs' operand to the new blocks arguments.
  SmallVector<Value, 8> operands(instOp.getOperands().begin(),
                                 instOp.getOperands().end());
  for (auto toServer : portReqs->getOps<RequestToServerConnection>()) {
    Value sending = toServer.sending();
    operands.push_back(sending);
    toServer.sendingMutable().assign(
        portReqs->addArgument(sending.getType(), toServer.getLoc()));
  }

  // Create the generation request op.
  OpBuilder b(instOp);
  auto implOp = b.create<ServiceImplementReqOp>(
      instOp.getLoc(), resultTypes, instOp.service_symbolAttr(),
      instOp.identifierAttr(), operands);
  implOp->setDialectAttrs(instOp->getDialectAttrs());
  implOp.portReqs().push_back(portReqs);

  // Update the users.
  for (auto [n, o] : llvm::zip(implOp.getResults(), instOp.getResults()))
    o.replaceAllUsesWith(n);
  unsigned instOpNumResults = instOp.getNumResults();
  for (auto e : llvm::enumerate(portReqs->getOps<RequestToClientConnection>()))
    e.value().receiving().replaceAllUsesWith(
        implOp.getResult(e.index() + instOpNumResults));

  // Try to generate the service provider.
  if (failed(genDispatcher.generate(implOp)))
    return instOp.emitOpError("failed to generate server");

  instOp.erase();
  return success();
}

LogicalResult ESIConnectServicesPass::surfaceReqs(
    hw::HWMutableModuleLike mod,
    ArrayRef<RequestToClientConnection> toClientReqs,
    ArrayRef<RequestToServerConnection> toServerReqs) {
  auto ctxt = mod.getContext();
  Block &modBlock = mod->getRegion(0).front();
  Operation *modTerminator = modBlock.getTerminator();

  // Track initial operand/result counts and the new IO.
  unsigned origNumInputs = mod.getNumInputs();
  SmallVector<std::pair<unsigned, hw::PortInfo>> newInputs;
  unsigned origNumOutputs = mod.getNumOutputs();
  SmallVector<std::pair<unsigned, hw::PortInfo>> newOutputs;

  // Assemble a port name from an array.
  auto getPortName = [](ArrayAttr namePath) {
    std::string portName;
    llvm::raw_string_ostream nameOS(portName);
    llvm::interleave(
        namePath.getValue(), nameOS,
        [&](Attribute attr) { nameOS << attr.cast<StringAttr>().getValue(); },
        ".");
    return nameOS.str();
  };

  // Append input ports to new port list and replace uses with new block args
  // which will correspond to said ports.
  unsigned inputCounter = origNumInputs;
  for (auto toClient : toClientReqs) {
    newInputs.push_back(std::make_pair(
        origNumInputs,
        hw::PortInfo{
            StringAttr::get(ctxt, getPortName(toClient.clientNamePath())),
            hw::PortDirection::INPUT, toClient.getType(), inputCounter++}));
    toClient.replaceAllUsesWith(modBlock.addArgument(
        toClient.receiving().getType(), toClient.getLoc()));
  }
  // Append output ports to new port list and redirect toServer inputs to
  // output op.
  unsigned outputCounter = origNumOutputs;
  for (auto toServer : toServerReqs) {
    newOutputs.push_back(std::make_pair(
        origNumOutputs,
        hw::PortInfo{
            StringAttr::get(ctxt, getPortName(toServer.clientNamePath())),
            hw::PortDirection::OUTPUT, toServer.sending().getType(),
            outputCounter++}));
    modTerminator->insertOperands(modTerminator->getNumOperands(),
                                  toServer.sending());
  }

  // Insert new module ESI ports.
  mod.insertPorts(newInputs, newOutputs);

  // Prepend a name to the instance tracking array.
  auto prependNamePart = [&](ArrayAttr namePath, StringRef part) {
    SmallVector<Attribute, 8> newNamePath;
    newNamePath.push_back(StringAttr::get(namePath.getContext(), part));
    newNamePath.append(namePath.begin(), namePath.end());
    return ArrayAttr::get(namePath.getContext(), newNamePath);
  };

  // Update the module instantiations.
  SmallVector<hw::HWInstanceLike, 1> newModuleInstantiations;
  StringAttr argsAttrName = StringAttr::get(ctxt, "argNames");
  StringAttr resultsAttrName = StringAttr::get(ctxt, "resultNames");
  for (auto inst : moduleInstantiations[mod]) {
    OpBuilder b(inst);

    // Assemble lists for the new instance op. Seed it with the existing
    // values.
    SmallVector<Value, 16> newOperands(inst->getOperands().begin(),
                                       inst->getOperands().end());
    SmallVector<Type, 16> newResultTypes(inst->getResultTypes().begin(),
                                         inst->getResultTypes().end());

    // Add new inputs for the new to_client requests and clone the request
    // into the module containing `inst`.
    for (auto [toClient, newPort] : llvm::zip(toClientReqs, newInputs)) {
      auto instToClient = cast<RequestToClientConnection>(b.clone(*toClient));
      instToClient.clientNamePathAttr(
          prependNamePart(instToClient.clientNamePath(), inst.instanceName()));
      newOperands.push_back(instToClient.receiving());
    }

    // Append the results for the to_server requests.
    for (auto newPort : newOutputs)
      newResultTypes.push_back(newPort.second.type);

    // Create a replacement instance of the same operation type.
    SmallVector<NamedAttribute> newAttrs;
    for (auto attr : inst->getAttrs()) {
      if (attr.getName() == argsAttrName)
        newAttrs.push_back(b.getNamedAttr(argsAttrName, mod.getArgNames()));
      else if (attr.getName() == resultsAttrName)
        newAttrs.push_back(
            b.getNamedAttr(resultsAttrName, mod.getResultNames()));
      else
        newAttrs.push_back(attr);
    }
    auto newHWInst = b.insert(
        Operation::create(inst->getLoc(), inst->getName(), newResultTypes,
                          newOperands, b.getDictionaryAttr(newAttrs),
                          inst->getSuccessors(), inst->getRegions()));
    newModuleInstantiations.push_back(newHWInst);

    // Replace all uses of the instance being replaced.
    for (auto [newV, oldV] :
         llvm::zip(newHWInst->getResults(), inst->getResults()))
      oldV.replaceAllUsesWith(newV);

    // Clone the to_server requests and wire them up to the new instance.
    outputCounter = origNumOutputs;
    for (auto [toServer, newPort] : llvm::zip(toServerReqs, newOutputs)) {
      auto instToServer = cast<RequestToServerConnection>(b.clone(*toServer));
      instToServer.clientNamePathAttr(
          prependNamePart(instToServer.clientNamePath(), inst.instanceName()));
      instToServer->setOperand(0, newHWInst->getResult(outputCounter++));
    }
  }

  // Replace the list of instantiations and erase the old ones.
  moduleInstantiations[mod].swap(newModuleInstantiations);
  for (auto oldInst : newModuleInstantiations)
    oldInst->erase();

  // Erase the original requests since they have been cloned into the proper
  // destination modules.
  for (auto toClient : toClientReqs)
    toClient.erase();
  for (auto toServer : toServerReqs)
    toServer.erase();
  return success();
}

namespace circt {
namespace esi {
std::unique_ptr<OperationPass<ModuleOp>> createESIConnectServicesPass() {
  return std::make_unique<ESIConnectServicesPass>();
}
} // namespace esi
} // namespace circt
