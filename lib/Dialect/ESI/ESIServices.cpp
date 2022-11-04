//===- ESIServices.cpp - Code related to ESI services ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/ESI/ESIServices.h"
#include "circt/Dialect/ESI/ESITypes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/MSFT/MSFTPasses.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

#include <memory>

using namespace circt;
using namespace circt::esi;

LogicalResult
ServiceGeneratorDispatcher::generate(ServiceImplementReqOp req,
                                     ServiceDeclOpInterface decl) {
  // Lookup based on 'impl_type' attribute and pass through the generate request
  // if found.
  auto genF = genLookupTable.find(req.getImplTypeAttr().getValue());
  if (genF == genLookupTable.end()) {
    if (failIfNotFound)
      return req.emitOpError("Could not find service generator for attribute '")
             << req.getImplTypeAttr() << "'";
    return success();
  }
  return genF->second(req, decl);
}

/// The generator for the "cosim" impl_type.
static LogicalResult instantiateCosimEndpointOps(ServiceImplementReqOp req,
                                                 ServiceDeclOpInterface) {
  auto *ctxt = req.getContext();
  OpBuilder b(req);
  Value clk = req.getOperand(0);
  Value rst = req.getOperand(1);

  // Determine which EndpointID this generator should start with.
  if (req.getImplOpts()) {
    auto opts = req.getImplOpts()->getValue();
    for (auto nameAttr : opts) {
      return req.emitOpError("did not recognize option name ")
             << nameAttr.getName();
    }
  }

  // Assemble the name to use for an endpoint.
  auto toStringAttr = [&](ArrayAttr strArr) {
    std::string buff;
    llvm::raw_string_ostream os(buff);
    llvm::interleave(strArr.getAsValueRange<StringAttr>(), os, ".");
    return StringAttr::get(ctxt, os.str());
  };

  llvm::DenseMap<RequestToClientConnectionOp, unsigned> toClientResultNum;
  for (auto toClient : req.getOps<RequestToClientConnectionOp>())
    toClientResultNum[toClient] = toClientResultNum.size();

  // Get the request pairs.
  llvm::SmallVector<
      std::pair<RequestToServerConnectionOp, RequestToClientConnectionOp>, 8>
      reqPairs;
  req.gatherPairedReqs(reqPairs);

  // Iterate through them, building a cosim endpoint for each one.
  for (auto [toServer, toClient] : reqPairs) {
    assert((toServer || toClient) &&
           "At least one in all pairs must be non-null");
    Location loc = toServer ? toServer.getLoc() : toClient.getLoc();
    ArrayAttr clientNamePathAttr = toServer ? toServer.getClientNamePathAttr()
                                            : toClient.getClientNamePathAttr();

    Value toServerValue;
    if (toServer)
      toServerValue = toServer.getToServer();
    else
      toServerValue =
          b.create<NullSourceOp>(loc, ChannelType::get(ctxt, b.getI1Type()));

    Type toClientType;
    if (toClient)
      toClientType = toClient.getToClient().getType();
    else
      toClientType = ChannelType::get(ctxt, b.getI1Type());

    auto cosim =
        b.create<CosimEndpointOp>(loc, toClientType, clk, rst, toServerValue,
                                  toStringAttr(clientNamePathAttr));

    if (toClient) {
      unsigned clientReqIdx = toClientResultNum[toClient];
      req.getResult(clientReqIdx).replaceAllUsesWith(cosim.getRecv());
    }
  }

  // Erase the generation request.
  req.erase();
  return success();
}

// Generator for "sv_mem" implementation type. Emits SV ops for an unpacked
// array, hopefully inferred as a memory to the SV compiler.
static LogicalResult
instantiateSystemVerilogMemory(ServiceImplementReqOp req,
                               ServiceDeclOpInterface decl) {
  if (!decl)
    return req.emitOpError(
        "Must specify a service declaration to use 'sv_mem'.");

  ImplicitLocOpBuilder b(req.getLoc(), req);
  BackedgeBuilder bb(b, req.getLoc());

  RandomAccessMemoryDeclOp ramDecl =
      dyn_cast<RandomAccessMemoryDeclOp>(decl.getOperation());
  if (!ramDecl)
    return req.emitOpError("'sv_mem' implementation type can only be used to "
                           "implement RandomAccessMemory declarations");

  if (req.getNumOperands() != 2)
    return req.emitOpError("Implementation requires clk and rst operands");
  auto clk = req.getOperand(0);
  auto rst = req.getOperand(1);
  auto write = b.getStringAttr("write");
  auto read = b.getStringAttr("read");
  auto none = b.create<hw::ConstantOp>(
      APInt(/*numBits*/ 0, /*val*/ 0, /*isSigned*/ false));
  auto i1 = b.getI1Type();
  auto c0 = b.create<hw::ConstantOp>(i1, 0);

  // Assemble a mapping of toClient results to actual consumers.
  DenseMap<Value, Value> outputMap;
  for (auto [bout, reqout] : llvm::zip_longest(
           req.getOps<RequestToClientConnectionOp>(), req.getResults())) {
    assert(bout.has_value());
    assert(reqout.has_value());
    outputMap[*bout] = *reqout;
  }

  // Create the SV memory.
  hw::UnpackedArrayType memType =
      hw::UnpackedArrayType::get(ramDecl.getInnerType(), ramDecl.getDepth());
  auto mem = b.create<sv::RegOp>(memType, req.getServiceSymbolAttr().getAttr())
                 .getResult();

  // Get the request pairs.
  llvm::SmallVector<
      std::pair<RequestToServerConnectionOp, RequestToClientConnectionOp>, 8>
      reqPairs;
  req.gatherPairedReqs(reqPairs);

  // Do everything which doesn't actually write to the memory, store the signals
  // needed for the actual memory writes for later.
  SmallVector<std::tuple<Value, Value, Value>> writeGoAddressData;
  for (auto [toServerReq, toClientReq] : reqPairs) {
    assert(toServerReq && toClientReq); // All of our interfaces are inout.
    assert(toServerReq.getServicePort() == toClientReq.getServicePort());
    auto port = toServerReq.getServicePort().getName();
    WrapValidReadyOp toClientResp;

    if (port == write) {
      // If this pair is doing a write...

      // Construct the response channel.
      auto doneValid = bb.get(i1);
      toClientResp = b.create<WrapValidReadyOp>(none, doneValid);

      // Unwrap the write request and 'explode' the struct.
      auto unwrap = b.create<UnwrapValidReadyOp>(toServerReq.getToServer(),
                                                 toClientResp.getReady());

      Value address = b.create<hw::StructExtractOp>(unwrap.getRawOutput(),
                                                    b.getStringAttr("address"));
      Value data = b.create<hw::StructExtractOp>(unwrap.getRawOutput(),
                                                 b.getStringAttr("data"));

      // Determine if the write should occur this cycle.
      auto go = b.create<comb::AndOp>(unwrap.getValid(), unwrap.getReady());
      go->setAttr("sv.namehint", b.getStringAttr("write_go"));
      // Register the 'go' signal and use it as the done message.
      doneValid.setValue(
          b.create<seq::CompRegOp>(go, clk, rst, c0, "write_done"));
      // Store the necessary data for the 'always' memory writing block.
      writeGoAddressData.push_back(std::make_tuple(go, address, data));

    } else if (port == read) {
      // If it's a read...

      // Construct the response channel.
      auto dataValid = bb.get(i1);
      auto data = bb.get(ramDecl.getInnerType());
      toClientResp = b.create<WrapValidReadyOp>(data, dataValid);

      // Unwrap the requested address and read from that memory location.
      auto addressUnwrap = b.create<UnwrapValidReadyOp>(
          toServerReq.getToServer(), toClientResp.getReady());
      Value memLoc =
          b.create<sv::ArrayIndexInOutOp>(mem, addressUnwrap.getRawOutput());
      auto readData = b.create<sv::ReadInOutOp>(memLoc);

      // Set the data on the response.
      data.setValue(readData);
      dataValid.setValue(addressUnwrap.getValid());
    } else {
      assert(false && "Port should be either 'read' or 'write'");
    }

    outputMap[toClientReq.getToClient()].replaceAllUsesWith(
        toClientResp.getChanOutput());
  }

  // Now construct the memory writes.
  b.create<sv::AlwaysFFOp>(
      sv::EventControl::AtPosEdge, clk, ResetType::SyncReset,
      sv::EventControl::AtPosEdge, rst, [&] {
        for (auto [go, address, data] : writeGoAddressData) {
          Value a = address, d = data; // So the lambda can capture.
          // If we're told to go, do the write.
          b.create<sv::IfOp>(go, [&] {
            Value memLoc = b.create<sv::ArrayIndexInOutOp>(mem, a);
            b.create<sv::PAssignOp>(memLoc, d);
          });
        }
      });

  req.erase();
  return success();
}

static ServiceGeneratorDispatcher
    globalDispatcher({{"cosim", instantiateCosimEndpointOps},
                      {"sv_mem", instantiateSystemVerilogMemory}},
                     false);

ServiceGeneratorDispatcher &ServiceGeneratorDispatcher::globalDispatcher() {
  return ::globalDispatcher;
}

void ServiceGeneratorDispatcher::registerGenerator(StringRef implType,
                                                   ServiceGeneratorFunc gen) {
  genLookupTable[implType] = gen;
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
      : genDispatcher(ServiceGeneratorDispatcher::globalDispatcher()) {}

  void runOnOperation() override;

  /// "Bubble up" the specified requests to all of the instantiations of the
  /// module specified. Create and connect up ports to tunnel the ESI channels
  /// through.
  LogicalResult surfaceReqs(hw::HWMutableModuleLike,
                            ArrayRef<RequestToClientConnectionOp>,
                            ArrayRef<RequestToServerConnectionOp>);

  /// Copy all service metadata up the instance hierarchy. Modify the service
  /// name path while copying.
  void copyMetadata(hw::HWMutableModuleLike);

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
  Block *anyServiceInst = nullptr;
  for (auto instOp : modBlock.getOps<ServiceInstanceOp>()) {
    auto b = new Block();
    localImplReqs[instOp.getServiceSymbolAttr()] = b;
    if (!instOp.getServiceSymbol().has_value())
      anyServiceInst = b;
  }

  // Decompose the 'inout' requests int to 'in' and 'out' requests.
  mod.walk([&](RequestInOutChannelOp reqInOut) {
    ImplicitLocOpBuilder b(reqInOut.getLoc(), reqInOut);
    b.create<RequestToServerConnectionOp>(reqInOut.getServicePortAttr(),
                                          reqInOut.getToServer(),
                                          reqInOut.getClientNamePathAttr());
    auto toClientReq = b.create<RequestToClientConnectionOp>(
        reqInOut.getToClient().getType(), reqInOut.getServicePortAttr(),
        reqInOut.getClientNamePathAttr());
    reqInOut.getToClient().replaceAllUsesWith(toClientReq.getToClient());
    reqInOut.erase();
  });

  // Find all of the "local" requests.
  mod.walk([&](Operation *op) {
    if (auto req = dyn_cast<RequestToClientConnectionOp>(op)) {
      auto service = req.getServicePortAttr().getModuleRef();
      auto implOpF = localImplReqs.find(service);
      if (implOpF != localImplReqs.end())
        req->moveBefore(implOpF->second, implOpF->second->end());
      else if (anyServiceInst)
        req->moveBefore(anyServiceInst, anyServiceInst->end());
    } else if (auto req = dyn_cast<RequestToServerConnectionOp>(op)) {
      auto service = req.getServicePortAttr().getModuleRef();
      auto implOpF = localImplReqs.find(service);
      if (implOpF != localImplReqs.end())
        req->moveBefore(implOpF->second, implOpF->second->end());
      else if (anyServiceInst)
        req->moveBefore(anyServiceInst, anyServiceInst->end());
    }
  });

  // Replace each service instance with a generation request. If a service
  // generator is registered, generate the server.
  for (auto instOp :
       llvm::make_early_inc_range(modBlock.getOps<ServiceInstanceOp>())) {
    Block *portReqs = localImplReqs[instOp.getServiceSymbolAttr()];
    if (failed(replaceInst(instOp, portReqs)))
      return failure();
  }

  // Copy any metadata up the instance hierarchy.
  copyMetadata(mod);

  // Identify the non-local reqs which need to be surfaced from this module.
  SmallVector<RequestToClientConnectionOp, 4> nonLocalToClientReqs;
  SmallVector<RequestToServerConnectionOp, 4> nonLocalToServerReqs;
  mod.walk([&](Operation *op) {
    if (auto req = dyn_cast<RequestToClientConnectionOp>(op)) {
      auto service = req.getServicePortAttr().getModuleRef();
      auto implOpF = localImplReqs.find(service);
      if (implOpF == localImplReqs.end())
        nonLocalToClientReqs.push_back(req);
    } else if (auto req = dyn_cast<RequestToServerConnectionOp>(op)) {
      auto service = req.getServicePortAttr().getModuleRef();
      auto implOpF = localImplReqs.find(service);
      if (implOpF == localImplReqs.end())
        nonLocalToServerReqs.push_back(req);
    }
  });

  // Surface all of the requests which cannot be fulfilled locally.
  if (nonLocalToClientReqs.empty() && nonLocalToServerReqs.empty())
    return success();
  return surfaceReqs(mod, nonLocalToClientReqs, nonLocalToServerReqs);
}

void ESIConnectServicesPass::copyMetadata(hw::HWMutableModuleLike mod) {
  SmallVector<ServiceHierarchyMetadataOp, 8> metadataOps;
  mod.walk([&](ServiceHierarchyMetadataOp op) { metadataOps.push_back(op); });

  for (auto inst : moduleInstantiations[mod]) {
    OpBuilder b(inst);
    auto instName = b.getStringAttr(inst.instanceName());
    for (auto metadata : metadataOps) {
      SmallVector<Attribute, 4> path;
      path.push_back(hw::InnerRefAttr::get(
          cast<hw::HWModuleLike>(mod.getOperation()).moduleNameAttr(),
          instName));
      for (auto attr : metadata.getServerNamePathAttr())
        path.push_back(attr);

      auto metadataCopy = cast<ServiceHierarchyMetadataOp>(b.clone(*metadata));
      metadataCopy.setServerNamePathAttr(b.getArrayAttr(path));
    }
  }
}

/// Create an op which contains metadata about the soon-to-be implemented
/// service. To be used by later passes which require these data (e.g.
/// automated software API creation).
static void emitServiceMetadata(ServiceImplementReqOp implReqOp) {
  ImplicitLocOpBuilder b(implReqOp.getLoc(), implReqOp);

  llvm::SmallVector<
      std::pair<RequestToServerConnectionOp, RequestToClientConnectionOp>, 8>
      reqPairs;
  implReqOp.gatherPairedReqs(reqPairs);

  SmallVector<Attribute, 8> clients;
  for (auto [toServer, toClient] : reqPairs) {
    SmallVector<NamedAttribute, 4> clientAttrs;
    Attribute servicePort, clientNamePath;
    if (toServer) {
      clientNamePath = toServer.getClientNamePathAttr();
      servicePort = toServer.getServicePortAttr();
      clientAttrs.push_back(b.getNamedAttr(
          "to_server_type", TypeAttr::get(toServer.getToServer().getType())));
    }
    if (toClient) {
      clientNamePath = toClient.getClientNamePathAttr();
      servicePort = toClient.getServicePortAttr();
      clientAttrs.push_back(b.getNamedAttr(
          "to_client_type", TypeAttr::get(toClient.getToClient().getType())));
    }

    clientAttrs.push_back(b.getNamedAttr("port", servicePort));
    clientAttrs.push_back(b.getNamedAttr("client_name", clientNamePath));

    clients.push_back(b.getDictionaryAttr(clientAttrs));
  }

  auto clientsAttr = b.getArrayAttr(clients);
  auto nameAttr = b.getArrayAttr(ArrayRef<Attribute>{});
  b.create<ServiceHierarchyMetadataOp>(
      implReqOp.getServiceSymbolAttr(), nameAttr, implReqOp.getImplTypeAttr(),
      implReqOp.getImplOptsAttr(), clientsAttr);
}

LogicalResult ESIConnectServicesPass::replaceInst(ServiceInstanceOp instOp,
                                                  Block *portReqs) {
  assert(portReqs);
  auto declSym = instOp.getServiceSymbolAttr();
  ServiceDeclOpInterface decl;
  if (declSym) {
    decl = dyn_cast_or_null<ServiceDeclOpInterface>(
        topLevelSyms.getDefinition(declSym));
    if (!decl)
      return instOp.emitOpError("Could not find service declaration ")
             << declSym;
  }

  // Compute the result types for the new op -- the instance op's output types
  // + the to_client types.
  SmallVector<Type, 8> resultTypes(instOp.getResultTypes().begin(),
                                   instOp.getResultTypes().end());
  for (auto toClient : portReqs->getOps<RequestToClientConnectionOp>())
    resultTypes.push_back(toClient.getToClient().getType());

  // Create the generation request op.
  OpBuilder b(instOp);
  auto implOp = b.create<ServiceImplementReqOp>(
      instOp.getLoc(), resultTypes, instOp.getServiceSymbolAttr(),
      instOp.getImplTypeAttr(), instOp.getImplOptsAttr(), instOp.getOperands());
  implOp->setDialectAttrs(instOp->getDialectAttrs());
  implOp.getPortReqs().push_back(portReqs);

  // Update the users.
  for (auto [n, o] : llvm::zip(implOp.getResults(), instOp.getResults()))
    o.replaceAllUsesWith(n);
  unsigned instOpNumResults = instOp.getNumResults();
  for (auto e :
       llvm::enumerate(portReqs->getOps<RequestToClientConnectionOp>()))
    e.value().getToClient().replaceAllUsesWith(
        implOp.getResult(e.index() + instOpNumResults));

  emitServiceMetadata(implOp);

  // Try to generate the service provider.
  if (failed(genDispatcher.generate(implOp, decl)))
    return instOp.emitOpError("failed to generate server");

  instOp.erase();
  return success();
}

LogicalResult ESIConnectServicesPass::surfaceReqs(
    hw::HWMutableModuleLike mod,
    ArrayRef<RequestToClientConnectionOp> toClientReqs,
    ArrayRef<RequestToServerConnectionOp> toServerReqs) {
  auto ctxt = mod.getContext();

  // Track initial operand/result counts and the new IO.
  unsigned origNumInputs = mod.getNumInputs();
  SmallVector<std::pair<unsigned, hw::PortInfo>> newInputs;
  unsigned origNumOutputs = mod.getNumOutputs();
  SmallVector<std::pair<mlir::StringAttr, Value>> newOutputs;

  // Assemble a port name from an array.
  auto getPortName = [&](ArrayAttr namePath) {
    std::string portName;
    llvm::raw_string_ostream nameOS(portName);
    llvm::interleave(
        namePath.getValue(), nameOS,
        [&](Attribute attr) { nameOS << attr.cast<StringAttr>().getValue(); },
        ".");
    return StringAttr::get(ctxt, nameOS.str());
  };

  // Insert new module input ESI ports.
  for (auto toClient : toClientReqs) {
    newInputs.push_back(std::make_pair(
        origNumInputs, hw::PortInfo{getPortName(toClient.getClientNamePath()),
                                    hw::PortDirection::INPUT,
                                    toClient.getType(), origNumInputs}));
  }
  mod.insertPorts(newInputs, {});
  Block *body = &mod->getRegion(0).front();

  // Replace uses with new block args which will correspond to said ports.
  // Note: no zip or enumerate here because we need mutable access to
  // toClientReqs.
  int i = 0;
  for (auto toClient : toClientReqs) {
    toClient.replaceAllUsesWith(body->getArguments()[origNumInputs + i]);
    ++i;
  }

  // Append output ports to new port list and redirect toServer inputs to
  // output op.
  unsigned outputCounter = origNumOutputs;
  for (auto toServer : toServerReqs)
    newOutputs.push_back(
        {getPortName(toServer.getClientNamePath()), toServer.getToServer()});

  mod.appendOutputs(newOutputs);

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
      auto instToClient = cast<RequestToClientConnectionOp>(b.clone(*toClient));
      instToClient.setClientNamePathAttr(prependNamePart(
          instToClient.getClientNamePath(), inst.instanceName()));
      newOperands.push_back(instToClient.getToClient());
    }

    // Append the results for the to_server requests.
    for (auto newPort : newOutputs)
      newResultTypes.push_back(newPort.second.getType());

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
      auto instToServer = cast<RequestToServerConnectionOp>(b.clone(*toServer));
      instToServer.setClientNamePathAttr(prependNamePart(
          instToServer.getClientNamePath(), inst.instanceName()));
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

std::unique_ptr<OperationPass<ModuleOp>>
circt::esi::createESIConnectServicesPass() {
  return std::make_unique<ESIConnectServicesPass>();
}
