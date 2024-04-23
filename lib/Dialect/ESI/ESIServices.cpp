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
#include <utility>

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

  // Since we always need a record of generation, create it here then pass it to
  // the generator for possible modification.
  OpBuilder b(req);
  auto implRecord = b.create<ServiceImplRecordOp>(
      req.getLoc(), req.getAppID(), req.getServiceSymbolAttr(),
      req.getStdServiceAttr(), req.getImplTypeAttr(), b.getDictionaryAttr({}));
  implRecord.getReqDetails().emplaceBlock();

  return genF->second(req, decl, implRecord);
}

/// The generator for the "cosim" impl_type.
static LogicalResult
instantiateCosimEndpointOps(ServiceImplementReqOp implReq,
                            ServiceDeclOpInterface,
                            ServiceImplRecordOp implRecord) {
  auto *ctxt = implReq.getContext();
  OpBuilder b(implReq);
  Value clk = implReq.getOperand(0);
  Value rst = implReq.getOperand(1);

  if (implReq.getImplOpts()) {
    auto opts = implReq.getImplOpts()->getValue();
    for (auto nameAttr : opts) {
      return implReq.emitOpError("did not recognize option name ")
             << nameAttr.getName();
    }
  }

  Block &connImplBlock = implRecord.getReqDetails().front();
  OpBuilder implRecords = OpBuilder::atBlockEnd(&connImplBlock);

  // Assemble the name to use for an endpoint.
  auto toStringAttr = [&](ArrayAttr strArr, StringAttr channelName) {
    std::string buff;
    llvm::raw_string_ostream os(buff);
    llvm::interleave(
        strArr.getAsRange<AppIDAttr>(), os,
        [&](AppIDAttr appid) {
          os << appid.getName().getValue();
          if (appid.getIndex())
            os << "[" << appid.getIndex() << "]";
        },
        ".");
    os << "." << channelName.getValue();
    return StringAttr::get(ctxt, os.str());
  };

  llvm::DenseMap<ServiceImplementConnReqOp, unsigned> toClientResultNum;
  for (auto req : implReq.getOps<ServiceImplementConnReqOp>())
    toClientResultNum[req] = toClientResultNum.size();

  // Iterate through the requests, building a cosim endpoint for each channel in
  // the bundle.
  // TODO: The cosim op should probably be able to take a bundle type and get
  // lowered to the SV primitive later on. The SV primitive will also need some
  // work to suit this new world order, so let's put this off.
  for (auto req : implReq.getOps<ServiceImplementConnReqOp>()) {
    Location loc = req->getLoc();
    ChannelBundleType bundleType = req.getToClient().getType();
    SmallVector<NamedAttribute, 8> channelAssignments;

    SmallVector<Value, 8> toServerValues;
    for (BundledChannel ch : bundleType.getChannels()) {
      if (ch.direction == ChannelDirection::to) {
        auto cosim = b.create<CosimFromHostEndpointOp>(
            loc, ch.type, clk, rst,
            toStringAttr(req.getRelativeAppIDPathAttr(), ch.name));
        toServerValues.push_back(cosim.getFromHost());
        channelAssignments.push_back(
            b.getNamedAttr(ch.name, cosim.getIdAttr()));
      }
    }

    auto pack =
        b.create<PackBundleOp>(implReq.getLoc(), bundleType, toServerValues);
    implReq.getResult(toClientResultNum[req])
        .replaceAllUsesWith(pack.getBundle());

    size_t chanIdx = 0;
    for (BundledChannel ch : bundleType.getChannels()) {
      if (ch.direction == ChannelDirection::from) {
        auto cosim = b.create<CosimToHostEndpointOp>(
            loc, clk, rst, pack.getFromChannels()[chanIdx++],
            toStringAttr(req.getRelativeAppIDPathAttr(), ch.name));
        channelAssignments.push_back(
            b.getNamedAttr(ch.name, cosim.getIdAttr()));
      }
    }

    implRecords.create<ServiceImplClientRecordOp>(
        req.getLoc(), req.getRelativeAppIDPathAttr(), req.getServicePortAttr(),
        TypeAttr::get(bundleType),
        b.getDictionaryAttr(b.getNamedAttr(
            "channel_assignments", b.getDictionaryAttr(channelAssignments))));
  }

  // Erase the generation request.
  implReq.erase();
  return success();
}

// Generator for "sv_mem" implementation type. Emits SV ops for an unpacked
// array, hopefully inferred as a memory to the SV compiler.
static LogicalResult
instantiateSystemVerilogMemory(ServiceImplementReqOp implReq,
                               ServiceDeclOpInterface decl,
                               ServiceImplRecordOp) {
  if (!decl)
    return implReq.emitOpError(
        "Must specify a service declaration to use 'sv_mem'.");

  ImplicitLocOpBuilder b(implReq.getLoc(), implReq);
  BackedgeBuilder bb(b, implReq.getLoc());

  RandomAccessMemoryDeclOp ramDecl =
      dyn_cast<RandomAccessMemoryDeclOp>(decl.getOperation());
  if (!ramDecl)
    return implReq.emitOpError(
        "'sv_mem' implementation type can only be used to "
        "implement RandomAccessMemory declarations");

  if (implReq.getNumOperands() != 2)
    return implReq.emitOpError("Implementation requires clk and rst operands");
  auto clk = implReq.getOperand(0);
  auto rst = implReq.getOperand(1);
  auto write = b.getStringAttr("write");
  auto read = b.getStringAttr("read");
  auto none = b.create<hw::ConstantOp>(
      APInt(/*numBits*/ 0, /*val*/ 0, /*isSigned*/ false));
  auto i1 = b.getI1Type();
  auto c0 = b.create<hw::ConstantOp>(i1, 0);

  // List of reqs which have a result.
  SmallVector<ServiceImplementConnReqOp, 8> toClientReqs(
      llvm::make_filter_range(
          implReq.getOps<ServiceImplementConnReqOp>(),
          [](auto req) { return req.getToClient() != nullptr; }));

  // Assemble a mapping of toClient results to actual consumers.
  DenseMap<Value, Value> outputMap;
  for (auto [bout, reqout] :
       llvm::zip_longest(toClientReqs, implReq.getResults())) {
    assert(bout.has_value());
    assert(reqout.has_value());
    Value toClient = bout->getToClient();
    outputMap[toClient] = *reqout;
  }

  // Create the SV memory.
  hw::UnpackedArrayType memType =
      hw::UnpackedArrayType::get(ramDecl.getInnerType(), ramDecl.getDepth());
  auto mem =
      b.create<sv::RegOp>(memType, implReq.getServiceSymbolAttr().getAttr())
          .getResult();

  // Do everything which doesn't actually write to the memory, store the signals
  // needed for the actual memory writes for later.
  SmallVector<std::tuple<Value, Value, Value>> writeGoAddressData;
  for (auto req : implReq.getOps<ServiceImplementConnReqOp>()) {
    auto port = req.getServicePort().getTarget();
    Value toClientResp;

    if (port == write) {
      // If this pair is doing a write...

      // Construct the response channel.
      auto doneValid = bb.get(i1);
      auto ackChannel = b.create<WrapValidReadyOp>(none, doneValid);

      auto pack =
          b.create<PackBundleOp>(implReq.getLoc(), req.getToClient().getType(),
                                 ackChannel.getChanOutput());
      Value toServer =
          pack.getFromChannels()[RandomAccessMemoryDeclOp::ReqDirChannelIdx];
      toClientResp = pack.getBundle();

      // Unwrap the write request and 'explode' the struct.
      auto unwrap =
          b.create<UnwrapValidReadyOp>(toServer, ackChannel.getReady());

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
      auto dataChannel = b.create<WrapValidReadyOp>(data, dataValid);

      auto pack =
          b.create<PackBundleOp>(implReq.getLoc(), req.getToClient().getType(),
                                 dataChannel.getChanOutput());
      Value toServer =
          pack.getFromChannels()[RandomAccessMemoryDeclOp::RespDirChannelIdx];
      toClientResp = pack.getBundle();

      // Unwrap the requested address and read from that memory location.
      auto addressUnwrap =
          b.create<UnwrapValidReadyOp>(toServer, dataChannel.getReady());
      Value memLoc =
          b.create<sv::ArrayIndexInOutOp>(mem, addressUnwrap.getRawOutput());
      auto readData = b.create<sv::ReadInOutOp>(memLoc);

      // Set the data on the response.
      data.setValue(readData);
      dataValid.setValue(addressUnwrap.getValid());
    } else {
      assert(false && "Port should be either 'read' or 'write'");
    }

    outputMap[req.getToClient()].replaceAllUsesWith(toClientResp);
  }

  // Now construct the memory writes.
  auto hwClk = b.create<seq::FromClockOp>(clk);
  b.create<sv::AlwaysFFOp>(
      sv::EventControl::AtPosEdge, hwClk, ResetType::SyncReset,
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

  implReq.erase();
  return success();
}

static ServiceGeneratorDispatcher globalDispatcher(
    DenseMap<StringRef, ServiceGeneratorDispatcher::ServiceGeneratorFunc>{
        {"cosim", instantiateCosimEndpointOps},
        {"sv_mem", instantiateSystemVerilogMemory}},
    false);

ServiceGeneratorDispatcher &ServiceGeneratorDispatcher::globalDispatcher() {
  return ::globalDispatcher;
}

void ServiceGeneratorDispatcher::registerGenerator(StringRef implType,
                                                   ServiceGeneratorFunc gen) {
  genLookupTable[implType] = std::move(gen);
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

  ESIConnectServicesPass(const ServiceGeneratorDispatcher &gen)
      : genDispatcher(gen) {}
  ESIConnectServicesPass()
      : genDispatcher(ServiceGeneratorDispatcher::globalDispatcher()) {}

  void runOnOperation() override;

  /// Convert connection requests to service implement connection requests,
  /// which have a relative appid path instead of just an appid. Leave being a
  /// record for the manifest of the original request.
  void convertReq(RequestConnectionOp);

  /// "Bubble up" the specified requests to all of the instantiations of the
  /// module specified. Create and connect up ports to tunnel the ESI channels
  /// through.
  LogicalResult surfaceReqs(hw::HWMutableModuleLike,
                            ArrayRef<ServiceImplementConnReqOp>);

  /// For any service which is "local" (provides the requested service) in a
  /// module, replace it with a ServiceImplementOp. Said op is to be replaced
  /// with an instantiation by a generator.
  LogicalResult replaceInst(ServiceInstanceOp, Block *portReqs);

  /// Figure out which requests are "local" vs need to be surfaced. Call
  /// 'surfaceReqs' and/or 'replaceInst' as appropriate.
  LogicalResult process(hw::HWModuleLike);

  /// If the servicePort is referring to a std service, return the name of it.
  StringAttr getStdService(FlatSymbolRefAttr serviceSym);

private:
  ServiceGeneratorDispatcher genDispatcher;
};
} // anonymous namespace

void ESIConnectServicesPass::runOnOperation() {
  ModuleOp outerMod = getOperation();
  topLevelSyms.addDefinitions(outerMod);

  outerMod.walk([&](RequestConnectionOp req) { convertReq(req); });

  // Get a partially-ordered list of modules based on the instantiation DAG.
  // It's _very_ important that we process modules before their instantiations
  // so that the modules where they're instantiated correctly process the
  // surfaced connections.
  SmallVector<hw::HWModuleLike, 64> sortedMods;
  getAndSortModules(outerMod, sortedMods);

  // Process each module.
  for (auto mod : sortedMods) {
    hw::HWModuleLike mutableMod = dyn_cast<hw::HWModuleLike>(*mod);
    if (mutableMod && failed(process(mutableMod))) {
      signalPassFailure();
      return;
    }
  }
}

// Get the std service name, if any.
StringAttr ESIConnectServicesPass::getStdService(FlatSymbolRefAttr svcSym) {
  if (!svcSym)
    return {};
  Operation *svcDecl = topLevelSyms.getDefinition(svcSym);
  if (!isa<CustomServiceDeclOp>(svcDecl))
    return svcDecl->getName().getIdentifier();
  return {};
}

void ESIConnectServicesPass::convertReq(RequestConnectionOp req) {
  OpBuilder b(req);
  auto newReq = b.create<ServiceImplementConnReqOp>(
      req.getLoc(), req.getToClient().getType(), req.getServicePortAttr(),
      ArrayAttr::get(&getContext(), {req.getAppIDAttr()}));
  newReq->setDialectAttrs(req->getDialectAttrs());
  req.getToClient().replaceAllUsesWith(newReq.getToClient());

  // Emit a record of the original request.
  b.create<ServiceRequestRecordOp>(
      req.getLoc(), req.getAppID(), req.getServicePortAttr(),
      getStdService(req.getServicePortAttr().getRootRef()),
      req.getToClient().getType());
  req.erase();
}

LogicalResult ESIConnectServicesPass::process(hw::HWModuleLike mod) {
  // If 'mod' doesn't have a body, assume it's an external module.
  if (mod->getNumRegions() == 0 || mod->getRegion(0).empty())
    return success();

  Block &modBlock = mod->getRegion(0).front();

  // The non-local reqs which need to be surfaced from this module.
  SmallVector<ServiceImplementConnReqOp, 4> nonLocalReqs;
  // Index the local services and create blocks in which to put the requests.
  DenseMap<SymbolRefAttr, Block *> localImplReqs;
  Block *anyServiceInst = nullptr;
  for (auto instOp : modBlock.getOps<ServiceInstanceOp>()) {
    auto *b = new Block();
    localImplReqs[instOp.getServiceSymbolAttr()] = b;
    if (!instOp.getServiceSymbolAttr())
      anyServiceInst = b;
  }

  // Sort the various requests by destination.
  mod.walk([&](ServiceImplementConnReqOp req) {
    auto service = req.getServicePort().getRootRef();
    auto implOpF = localImplReqs.find(service);
    if (implOpF != localImplReqs.end())
      req->moveBefore(implOpF->second, implOpF->second->end());
    else if (anyServiceInst)
      req->moveBefore(anyServiceInst, anyServiceInst->end());
    else
      nonLocalReqs.push_back(req);
  });

  // Replace each service instance with a generation request. If a service
  // generator is registered, generate the server.
  for (auto instOp :
       llvm::make_early_inc_range(modBlock.getOps<ServiceInstanceOp>())) {
    Block *portReqs = localImplReqs[instOp.getServiceSymbolAttr()];
    if (failed(replaceInst(instOp, portReqs)))
      return failure();
  }

  // Surface all of the requests which cannot be fulfilled locally.
  if (nonLocalReqs.empty())
    return success();

  if (auto mutableMod = dyn_cast<hw::HWMutableModuleLike>(mod.getOperation()))
    return surfaceReqs(mutableMod, nonLocalReqs);
  return mod.emitOpError(
      "Cannot surface requests through module without mutable ports");
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
  for (auto req : portReqs->getOps<ServiceImplementConnReqOp>())
    resultTypes.push_back(req.getBundleType());

  // Create the generation request op.
  OpBuilder b(instOp);
  auto implOp = b.create<ServiceImplementReqOp>(
      instOp.getLoc(), resultTypes, instOp.getAppIDAttr(),
      instOp.getServiceSymbolAttr(), instOp.getImplTypeAttr(),
      getStdService(declSym), instOp.getImplOptsAttr(), instOp.getOperands());
  implOp->setDialectAttrs(instOp->getDialectAttrs());
  implOp.getPortReqs().push_back(portReqs);

  // Update the users.
  for (auto [n, o] : llvm::zip(implOp.getResults(), instOp.getResults()))
    o.replaceAllUsesWith(n);
  unsigned instOpNumResults = instOp.getNumResults();
  for (auto [idx, req] :
       llvm::enumerate(portReqs->getOps<ServiceImplementConnReqOp>())) {
    req.getToClient().replaceAllUsesWith(
        implOp.getResult(idx + instOpNumResults));
  }

  // Try to generate the service provider.
  if (failed(genDispatcher.generate(implOp, decl)))
    return instOp.emitOpError("failed to generate server");

  instOp.erase();
  return success();
}

LogicalResult
ESIConnectServicesPass::surfaceReqs(hw::HWMutableModuleLike mod,
                                    ArrayRef<ServiceImplementConnReqOp> reqs) {
  auto *ctxt = mod.getContext();
  Block *body = &mod->getRegion(0).front();

  // Track initial operand/result counts and the new IO.
  unsigned origNumInputs = mod.getNumInputPorts();
  SmallVector<std::pair<unsigned, hw::PortInfo>> newInputs;

  // Assemble a port name from an array.
  auto getPortName = [&](ArrayAttr namePath) {
    std::string portName;
    llvm::raw_string_ostream nameOS(portName);
    llvm::interleave(
        namePath.getAsRange<AppIDAttr>(), nameOS,
        [&](AppIDAttr appid) {
          nameOS << appid.getName().getValue();
          if (appid.getIndex())
            nameOS << "_" << appid.getIndex();
        },
        ".");
    return StringAttr::get(ctxt, nameOS.str());
  };

  for (auto req : reqs)
    if (req->getParentWithTrait<OpTrait::IsIsolatedFromAbove>() != mod)
      return req.emitOpError(
          "Cannot surface requests through isolated from above ops");

  // Insert new module input ESI ports.
  for (auto req : reqs) {
    newInputs.push_back(std::make_pair(
        origNumInputs,
        hw::PortInfo{{getPortName(req.getRelativeAppIDPathAttr()),
                      req.getBundleType(), hw::ModulePort::Direction::Input},
                     origNumInputs,
                     {},
                     req->getLoc()}));

    // Replace uses with new block args which will correspond to said ports.
    Value replValue = body->addArgument(req.getBundleType(), req->getLoc());
    req.getToClient().replaceAllUsesWith(replValue);
  }
  mod.insertPorts(newInputs, {});

  // Prepend a name to the instance tracking array.
  auto prependNamePart = [&](ArrayAttr appIDPath, AppIDAttr appID) {
    SmallVector<Attribute, 8> newAppIDPath;
    newAppIDPath.push_back(appID);
    newAppIDPath.append(appIDPath.begin(), appIDPath.end());
    return ArrayAttr::get(appIDPath.getContext(), newAppIDPath);
  };

  // Update the module instantiations.
  SmallVector<igraph::InstanceOpInterface, 1> newModuleInstantiations;
  for (auto inst : moduleInstantiations[mod]) {
    OpBuilder b(inst);

    // Add new inputs for the new bundles being requested.
    SmallVector<Value, 16> newOperands;
    for (auto req : reqs) {
      // If the instance has an AppID, prepend it.
      ArrayAttr appIDPath = req.getRelativeAppIDPathAttr();
      if (auto instAppID = dyn_cast_or_null<AppIDAttr>(
              inst->getDiscardableAttr(AppIDAttr::AppIDAttrName)))
        appIDPath = prependNamePart(appIDPath, instAppID);

      // Clone the request.
      auto clone = b.create<ServiceImplementConnReqOp>(
          req.getLoc(), req.getToClient().getType(), req.getServicePortAttr(),
          appIDPath);
      clone->setDialectAttrs(req->getDialectAttrs());
      newOperands.push_back(clone.getToClient());
    }
    inst->insertOperands(inst->getNumOperands(), newOperands);
    // Set the names, if we know how.
    if (auto hwInst = dyn_cast<hw::InstanceOp>(*inst))
      hwInst.setArgNamesAttr(b.getArrayAttr(mod.getInputNames()));
  }

  // Erase the original requests since they have been cloned into the proper
  // destination modules.
  for (auto req : reqs)
    req.erase();
  return success();
}

std::unique_ptr<OperationPass<ModuleOp>>
circt::esi::createESIConnectServicesPass() {
  return std::make_unique<ESIConnectServicesPass>();
}
