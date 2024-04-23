//===- IbisTunneling.cpp - Implementation of tunneling --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "circt/Dialect/Ibis/IbisDialect.h"
#include "circt/Dialect/Ibis/IbisOps.h"
#include "circt/Dialect/Ibis/IbisPasses.h"
#include "circt/Dialect/Ibis/IbisTypes.h"

#include "circt/Support/InstanceGraph.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;
using namespace circt::ibis;
using namespace circt::igraph;

namespace {

#define GEN_PASS_DEF_IBISTUNNELING
#include "circt/Dialect/Ibis/IbisPasses.h.inc"

// The PortInfo struct is used to keep track of the get_port ops that
// specify which ports needs to be tunneled through the hierarchy.
struct PortInfo {
  // Name used for portrefs of this get_port in the instance hierarchy.
  mlir::StringAttr portName;

  // Source get_port op.
  GetPortOp getPortOp;

  PortRefType getType() {
    return cast<PortRefType>(getPortOp.getPort().getType());
  }
  Type getInnerType() { return getType().getPortType(); }
  Direction getRequestedDirection() { return getType().getDirection(); }
};

struct Tunneler {
public:
  Tunneler(const IbisTunnelingOptions &options, PathOp op,
           ConversionPatternRewriter &rewriter, InstanceGraph &ig);

  // A mapping between requested port names from a ScopeRef and the actual
  // portref SSA values that are used to replace the get_port ops.
  // MapVector to ensure determinism.
  using PortRefMapping = llvm::MapVector<PortInfo *, Value>;

  // Launch the tunneling process.
  LogicalResult go();

private:
  // Dispatches tunneling in the current container and returns a value of the
  // target scoperef inside the current container.
  LogicalResult tunnelDispatch(InstanceGraphNode *currentContainer,
                               llvm::ArrayRef<PathStepAttr> path,
                               PortRefMapping &mapping);

  // "Port forwarding" check - ibis.get_port specifies the intended
  // direction which a port is accessed by from within the hierarchy.
  // If the intended direction is not the same as the actual port
  // direction, we need to insert a wire to flip the direction of the
  // mapped port.
  Value portForwardIfNeeded(PortOpInterface actualPort, PortInfo &portInfo);

  // Tunnels up relative to the current container. This will write to the
  // target input port of the current container from any parent
  // (instantiating) containers, and return the value of the target scoperef
  // inside the current container.
  LogicalResult tunnelUp(InstanceGraphNode *currentContainer,
                         llvm::ArrayRef<PathStepAttr> path,
                         PortRefMapping &portMapping);

  // Tunnels down relative to the current container, and returns the value of
  // the target scoperef inside the current container.
  LogicalResult tunnelDown(InstanceGraphNode *currentContainer,
                           FlatSymbolRefAttr tunnelInto,
                           llvm::ArrayRef<PathStepAttr> path,
                           PortRefMapping &portMapping);

  // Generates names for the port refs to be created.
  void genPortNames(llvm::SmallVectorImpl<PortInfo> &portInfos);

  PathOp op;
  ConversionPatternRewriter &rewriter;
  InstanceGraph &ig;
  const IbisTunnelingOptions &options;
  mlir::StringAttr pathName;
  llvm::SmallVector<PathStepAttr> path;

  // MapVector to ensure determinism.
  llvm::SmallVector<PortInfo> portInfos;

  // "Target" refers to the last step in the path which is the scoperef that
  // all port requests are tunneling towards.
  PathStepAttr target;
  FlatSymbolRefAttr targetName;
};

Tunneler::Tunneler(const IbisTunnelingOptions &options, PathOp op,
                   ConversionPatternRewriter &rewriter, InstanceGraph &ig)
    : op(op), rewriter(rewriter), ig(ig), options(options) {
  llvm::copy(op.getPathAsRange(), std::back_inserter(path));
  assert(!path.empty() &&
         "empty paths should never occur - illegal for ibis.path ops");
  target = path.back();
  targetName = target.getChild();
}

void Tunneler::genPortNames(llvm::SmallVectorImpl<PortInfo> &portInfos) {
  std::string pathName;
  llvm::raw_string_ostream ss(pathName);
  llvm::interleave(
      op.getPathAsRange(), ss,
      [&](PathStepAttr step) {
        if (step.getDirection() == PathDirection::Parent)
          ss << "p"; // use 'p' instead of 'parent' to avoid long SSA names.
        else
          ss << step.getChild().getValue();
      },
      "_");

  for (PortInfo &pi : portInfos) {
    // Suffix the ports by the intended usage (read/write). This also de-aliases
    // cases where one both reads and writes from the same input port.
    std::string suffix = pi.getRequestedDirection() == Direction::Input
                             ? options.writeSuffix
                             : options.readSuffix;
    pi.portName = rewriter.getStringAttr(pathName + "_" +
                                         pi.portName.getValue() + suffix);
  }
}

LogicalResult Tunneler::go() {
  // Gather the required port accesses of the ScopeRef.
  for (auto *user : op.getResult().getUsers()) {
    auto getPortOp = dyn_cast<GetPortOp>(user);
    if (!getPortOp)
      return user->emitOpError() << "unknown user of a PathOp result - "
                                    "tunneling only supports ibis.get_port";
    portInfos.push_back(
        PortInfo{getPortOp.getPortSymbolAttr().getAttr(), getPortOp});
  }
  genPortNames(portInfos);

  InstanceGraphNode *currentContainer =
      ig.lookup(cast<ModuleOpInterface>(op.getOperation()->getParentOp()));

  PortRefMapping mapping;
  if (failed(tunnelDispatch(currentContainer, path, mapping)))
    return failure();

  // Replace the get_port ops with the target value.
  for (PortInfo &pi : portInfos) {
    auto *it = mapping.find(&pi);
    assert(it != mapping.end() &&
           "expected to find a portref mapping for all get_port ops");
    rewriter.replaceOp(pi.getPortOp, it->second);
  }

  // And finally erase the path.
  rewriter.eraseOp(op);
  return success();
}

// NOLINTNEXTLINE(misc-no-recursion)
LogicalResult Tunneler::tunnelDispatch(InstanceGraphNode *currentContainer,
                                       llvm::ArrayRef<PathStepAttr> path,
                                       PortRefMapping &mapping) {
  PathStepAttr currentStep = path.front();
  PortRefMapping targetValue;
  path = path.drop_front();
  if (currentStep.getDirection() == PathDirection::Parent) {
    LogicalResult upRes = tunnelUp(currentContainer, path, mapping);
    if (failed(upRes))
      return failure();
  } else {
    FlatSymbolRefAttr tunnelInto = currentStep.getChild();
    LogicalResult downRes =
        tunnelDown(currentContainer, tunnelInto, path, mapping);
    if (failed(downRes))
      return failure();
  }
  return success();
}

Value Tunneler::portForwardIfNeeded(PortOpInterface actualPort,
                                    PortInfo &portInfo) {
  Direction actualDir =
      cast<PortRefType>(actualPort.getPort().getType()).getDirection();
  Direction requestedDir = portInfo.getRequestedDirection();

  // Match - just return the port itself.
  if (actualDir == requestedDir)
    return actualPort.getPort();

  // Mismatch...
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(actualPort);

  // If the requested direction was an input, this means that someone tried
  // to write to an output port. We need to insert an ibis.wire.input that
  // provides a writeable input port, and assign the wire output to the
  // output port.
  if (requestedDir == Direction::Input) {
    auto wireOp = rewriter.create<InputWireOp>(
        op.getLoc(),
        rewriter.getStringAttr(actualPort.getPortName().strref() + ".wr"),
        portInfo.getInnerType());

    rewriter.create<PortWriteOp>(op.getLoc(), actualPort.getPort(),
                                 wireOp.getOutput());
    return wireOp.getPort();
  }

  // If the requested direction was an output, this means that someone tried
  // to read from an input port. We need to insert an ibis.wire.output that
  // provides a readable output port, and read the input port as the value
  // of the wire.
  Value inputValue =
      rewriter.create<PortReadOp>(op.getLoc(), actualPort.getPort());
  auto wireOp = rewriter.create<OutputWireOp>(
      op.getLoc(),
      hw::InnerSymAttr::get(
          rewriter.getStringAttr(actualPort.getPortName().strref() + ".rd")),
      inputValue);
  return wireOp.getPort();
}

// Lookup an instance in the parent op. If the parent op is a symbol table, will
// use that - else, scan the ibis.container.instance operations in the parent.
static FailureOr<ContainerInstanceOp> locateInstanceIn(Operation *parentOp,
                                                       FlatSymbolRefAttr name) {
  if (parentOp->hasTrait<OpTrait::SymbolTable>()) {
    auto *tunnelInstanceOp = SymbolTable::lookupSymbolIn(parentOp, name);
    if (!tunnelInstanceOp)
      return failure();
    return cast<ContainerInstanceOp>(tunnelInstanceOp);
  }

  // Default: scan the container instances.
  for (auto instanceOp : parentOp->getRegion(0).getOps<ContainerInstanceOp>()) {
    if (instanceOp.getInnerSym().getSymName() == name.getValue())
      return instanceOp;
  }

  return failure();
}

// NOLINTNEXTLINE(misc-no-recursion)
LogicalResult Tunneler::tunnelDown(InstanceGraphNode *currentContainer,
                                   FlatSymbolRefAttr tunnelInto,
                                   llvm::ArrayRef<PathStepAttr> path,
                                   PortRefMapping &portMapping) {
  // Locate the instance that we're tunneling into
  Operation *parentOp = currentContainer->getModule().getOperation();
  auto parentSymbolOp = dyn_cast<hw::InnerSymbolOpInterface>(parentOp);
  assert(parentSymbolOp && "expected current container to be a symbol op");
  FailureOr<ContainerInstanceOp> locateRes =
      locateInstanceIn(parentOp, tunnelInto);
  if (failed(locateRes))
    return op->emitOpError()
           << "expected an instance named " << tunnelInto << " in @"
           << parentSymbolOp.getInnerSymAttr().getSymName().getValue()
           << " but found none";
  ContainerInstanceOp tunnelInstance = *locateRes;

  if (path.empty()) {
    // Tunneling ended with a 'child' step - create get_ports of all of the
    // requested ports.
    rewriter.setInsertionPointAfter(tunnelInstance);
    for (PortInfo &pi : portInfos) {
      auto targetGetPortOp =
          rewriter.create<GetPortOp>(op.getLoc(), pi.getType(), tunnelInstance,
                                     pi.getPortOp.getPortSymbol());
      portMapping[&pi] = targetGetPortOp.getResult();
    }
    return success();
  }

  // We're not in the target, but tunneling into a child instance.
  // Create output ports in the child instance for the requested ports.
  auto *tunnelScopeNode =
      ig.lookup(tunnelInstance.getTargetNameAttr().getTarget());
  auto tunnelScope = tunnelScopeNode->getModule<ScopeOpInterface>();

  rewriter.setInsertionPointToEnd(tunnelScope.getBodyBlock());
  llvm::DenseMap<StringAttr, OutputPortOp> outputPortOps;
  for (PortInfo &pi : portInfos) {
    outputPortOps[pi.portName] = rewriter.create<OutputPortOp>(
        op.getLoc(), pi.portName, circt::hw::InnerSymAttr::get(pi.portName),
        pi.getType());
  }

  // Recurse into the tunnel instance container.
  PortRefMapping childMapping;
  if (failed(tunnelDispatch(tunnelScopeNode, path, childMapping)))
    return failure();

  for (auto [pi, res] : childMapping) {
    PortInfo &portInfo = *pi;

    // Write the target value to the output port.
    rewriter.setInsertionPointToEnd(tunnelScope.getBodyBlock());
    rewriter.create<PortWriteOp>(op.getLoc(), outputPortOps[portInfo.portName],
                                 res);

    // Back in the current container, read the new output port of the child
    // instance and assign it to the port mapping.
    rewriter.setInsertionPointAfter(tunnelInstance);
    auto getPortOp = rewriter.create<GetPortOp>(
        op.getLoc(), tunnelInstance, portInfo.portName, portInfo.getType(),
        Direction::Output);
    portMapping[pi] =
        rewriter.create<PortReadOp>(op.getLoc(), getPortOp).getResult();
  }

  return success();
}

// NOLINTNEXTLINE(misc-no-recursion)
LogicalResult Tunneler::tunnelUp(InstanceGraphNode *currentContainer,
                                 llvm::ArrayRef<PathStepAttr> path,
                                 PortRefMapping &portMapping) {
  auto scopeOp = currentContainer->getModule<ScopeOpInterface>();
  if (currentContainer->noUses())
    return op->emitOpError()
           << "cannot tunnel up from " << scopeOp.getScopeName()
           << " because it has no uses";

  for (auto *use : currentContainer->uses()) {
    InstanceGraphNode *parentScopeNode =
        ig.lookup(use->getParent()->getModule());
    auto parentScope = parentScopeNode->getModule<ScopeOpInterface>();
    PortRefMapping targetPortMapping;
    if (path.empty()) {
      // Tunneling ended with a 'parent' step - all of the requested ports
      // should be available right here in the parent scope.
      for (PortInfo &pi : portInfos) {
        StringRef targetPortName = pi.getPortOp.getPortSymbol();
        PortOpInterface portLikeOp = parentScope.lookupPort(targetPortName);
        if (!portLikeOp)
          return op->emitOpError()
                 << "expected a port named " << targetPortName << " in "
                 << parentScope.getScopeName() << " but found none";

        // "Port forwarding" check - see comment in portForwardIfNeeded.
        targetPortMapping[&pi] = portForwardIfNeeded(portLikeOp, pi);
      }
    } else {
      // recurse into the parents, which will define the target value that
      // we can write to the input port of the current container instance.
      if (failed(tunnelDispatch(parentScopeNode, path, targetPortMapping)))
        return failure();
    }

    auto instance = use->getInstance<ContainerInstanceOp>();
    rewriter.setInsertionPointAfter(instance);
    for (PortInfo &pi : portInfos) {
      auto getPortOp = rewriter.create<GetPortOp>(
          op.getLoc(), instance, pi.portName, pi.getType(), Direction::Input);
      rewriter.create<PortWriteOp>(op.getLoc(), getPortOp,
                                   targetPortMapping[&pi]);
    }
  }

  // Create input ports for the requested portrefs.
  rewriter.setInsertionPointToEnd(scopeOp.getBodyBlock());
  for (PortInfo &pi : portInfos) {
    auto inputPort = rewriter.create<InputPortOp>(
        op.getLoc(), pi.portName, hw::InnerSymAttr::get(pi.portName),
        pi.getType());
    // Read the input port of the current container to forward the portref.

    portMapping[&pi] =
        rewriter.create<PortReadOp>(op.getLoc(), inputPort.getResult())
            .getResult();
  }

  return success();
}

class TunnelingConversionPattern : public OpConversionPattern<PathOp> {
public:
  TunnelingConversionPattern(MLIRContext *context, InstanceGraph &ig,
                             IbisTunnelingOptions options)
      : OpConversionPattern<PathOp>(context), ig(ig),
        options(std::move(options)) {}

  LogicalResult
  matchAndRewrite(PathOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return Tunneler(options, op, rewriter, ig).go();
  }

protected:
  InstanceGraph &ig;
  IbisTunnelingOptions options;
};

struct TunnelingPass : public impl::IbisTunnelingBase<TunnelingPass> {
  using IbisTunnelingBase<TunnelingPass>::IbisTunnelingBase;
  void runOnOperation() override;
};

} // anonymous namespace

void TunnelingPass::runOnOperation() {
  auto &ig = getAnalysis<InstanceGraph>();
  auto *ctx = &getContext();
  ConversionTarget target(*ctx);
  target.addIllegalOp<PathOp>();
  target.addLegalOp<InputPortOp, OutputPortOp, PortReadOp, PortWriteOp,
                    GetPortOp, InputWireOp, OutputWireOp>();

  RewritePatternSet patterns(ctx);
  patterns.add<TunnelingConversionPattern>(
      ctx, ig, IbisTunnelingOptions{readSuffix, writeSuffix});

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass>
circt::ibis::createTunnelingPass(const IbisTunnelingOptions &options) {
  return std::make_unique<TunnelingPass>(options);
}
