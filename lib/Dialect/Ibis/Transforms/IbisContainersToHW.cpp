//===- IbisContainersToHW.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Ibis/IbisDialect.h"
#include "circt/Dialect/Ibis/IbisOps.h"
#include "circt/Dialect/Ibis/IbisPasses.h"
#include "circt/Dialect/Ibis/IbisTypes.h"

#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace ibis;

static llvm::StringRef getScopeRefModuleName(Type type) {
  return type.cast<ScopeRefType>().getScopeRef().getAttr().strref();
}

namespace {

// Analysis result for generating the port interface of a container + a bit of
// port op caching.
struct ContainerPortInfo {
  std::unique_ptr<hw::ModulePortInfo> hwPorts;

  // A mapping between the port name and the port op within the container.
  llvm::DenseMap<StringAttr, InputPortOp> opInputs;

  // A mapping between the port name and the port op within the container.
  llvm::DenseMap<StringAttr, OutputPortOp> opOutputs;

  ContainerPortInfo() = default;
  ContainerPortInfo(ContainerOp container) {
    SmallVector<hw::PortInfo, 4> inputs, outputs;

    // Copies all attributes from a port, except for the port symbol and type.
    auto copyPortAttrs = [](auto port) {
      llvm::DenseSet<StringAttr> elidedAttrs;
      elidedAttrs.insert(port.getInnerSymAttrName());
      elidedAttrs.insert(port.getTypeAttrName());
      llvm::SmallVector<NamedAttribute> attrs;
      for (NamedAttribute namedAttr : port->getAttrs()) {
        if (elidedAttrs.contains(namedAttr.getName()))
          continue;
        attrs.push_back(namedAttr);
      }
      return DictionaryAttr::get(port.getContext(), attrs);
    };

    // Gather in and output port ops.
    for (auto input : container.getBodyBlock()->getOps<InputPortOp>()) {
      opInputs[input.getInnerSym().getSymName()] = input;

      hw::PortInfo portInfo;
      portInfo.name = input.getInnerSym().getSymName();
      portInfo.type = cast<PortOpInterface>(input.getOperation()).getPortType();
      portInfo.dir = hw::ModulePort::Direction::Input;
      portInfo.attrs = copyPortAttrs(input);
      inputs.push_back(portInfo);
    }

    for (auto output : container.getBodyBlock()->getOps<OutputPortOp>()) {
      opOutputs[output.getInnerSym().getSymName()] = output;

      hw::PortInfo portInfo;
      portInfo.name = output.getInnerSym().getSymName();
      portInfo.type =
          cast<PortOpInterface>(output.getOperation()).getPortType();
      portInfo.dir = hw::ModulePort::Direction::Output;
      portInfo.attrs = copyPortAttrs(output);
      outputs.push_back(portInfo);
    }
    hwPorts = std::make_unique<hw::ModulePortInfo>(inputs, outputs);
  }
};

using ContainerPortInfoMap = llvm::DenseMap<StringAttr, ContainerPortInfo>;

struct ContainerOpConversionPattern : public OpConversionPattern<ContainerOp> {
  ContainerOpConversionPattern(MLIRContext *ctx,
                               ContainerPortInfoMap &portOrder)
      : OpConversionPattern<ContainerOp>(ctx), portOrder(portOrder) {}

  LogicalResult
  matchAndRewrite(ContainerOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);

    const ContainerPortInfo &cpi = portOrder.at(op.getSymNameAttr());
    auto hwMod = rewriter.create<hw::HWModuleOp>(
        op.getLoc(), op.getSymNameAttr(), *cpi.hwPorts);

    hw::OutputOp outputOp =
        cast<hw::OutputOp>(hwMod.getBodyBlock()->getTerminator());

    // Replace all of the reads of the inputs to use the input block arguments.
    for (auto [idx, input] : llvm::enumerate(cpi.hwPorts->getInputs())) {
      Value barg = hwMod.getBodyBlock()->getArgument(idx);
      InputPortOp inputPort = cpi.opInputs.at(input.name);
      // Replace all reads of the input port with the input block argument.
      for (auto *user : inputPort.getOperation()->getUsers()) {
        auto reader = dyn_cast<PortReadOp>(user);
        if (!reader)
          return rewriter.notifyMatchFailure(
              user, "expected only ibis.port.read ops of the input port");

        rewriter.replaceOp(reader, barg);
      }

      rewriter.eraseOp(inputPort);
    }

    // Adjust the hw.output op to use ibis.port.write values
    llvm::SmallVector<Value> outputValues;
    for (auto [idx, output] : llvm::enumerate(cpi.hwPorts->getOutputs())) {
      auto outputPort = cpi.opOutputs.at(output.name);
      // Locate the write to the output op.
      auto users = outputPort->getUsers();
      size_t nUsers = std::distance(users.begin(), users.end());
      if (nUsers != 1)
        return outputPort->emitOpError()
               << "expected exactly one ibis.port.write op of the output "
                  "port: "
               << output.name.str() << " found: " << nUsers;
      auto writer = cast<PortWriteOp>(*users.begin());
      outputValues.push_back(writer.getValue());
      rewriter.eraseOp(outputPort);
      rewriter.eraseOp(writer);
    }

    rewriter.mergeBlocks(&op.getBodyRegion().front(), hwMod.getBodyBlock());

    // Rewrite the hw.output op.
    rewriter.eraseOp(outputOp);
    rewriter.setInsertionPointToEnd(hwMod.getBodyBlock());
    outputOp = rewriter.create<hw::OutputOp>(op.getLoc(), outputValues);
    rewriter.eraseOp(op);
    return success();
  }

  ContainerPortInfoMap &portOrder;
};

struct ThisOpConversionPattern : public OpConversionPattern<ThisOp> {
  ThisOpConversionPattern(MLIRContext *ctx)
      : OpConversionPattern<ThisOp>(ctx) {}

  LogicalResult
  matchAndRewrite(ThisOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: remove this op from the dialect - not needed anymore.
    rewriter.eraseOp(op);
    return success();
  }
};

struct ContainerInstanceOpConversionPattern
    : public OpConversionPattern<ContainerInstanceOp> {

  ContainerInstanceOpConversionPattern(MLIRContext *ctx,
                                       ContainerPortInfoMap &portOrder)
      : OpConversionPattern<ContainerInstanceOp>(ctx), portOrder(portOrder) {}

  LogicalResult
  matchAndRewrite(ContainerInstanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    llvm::SmallVector<Value> operands;

    // Gather the get_port ops that target the instance
    llvm::DenseMap<StringAttr, PortReadOp> outputReadsToReplace;
    llvm::DenseMap<StringAttr, PortWriteOp> inputWritesToUse;
    llvm::SmallVector<Operation *> getPortsToErase;
    for (auto *user : op->getUsers()) {
      auto getPort = dyn_cast<GetPortOp>(user);
      if (!getPort)
        return rewriter.notifyMatchFailure(
            user, "expected only ibis.get_port op usage of the instance");

      for (auto *user : getPort->getUsers()) {
        auto res =
            llvm::TypeSwitch<Operation *, LogicalResult>(user)
                .Case<PortReadOp>([&](auto read) {
                  auto [it, inserted] = outputReadsToReplace.insert(
                      {getPort.getPortSymbolAttr().getAttr(), read});
                  if (!inserted)
                    return rewriter.notifyMatchFailure(
                        read, "expected only one ibis.port.read op of the "
                              "output port");
                  return success();
                })
                .Case<PortWriteOp>([&](auto write) {
                  auto [it, inserted] = inputWritesToUse.insert(
                      {getPort.getPortSymbolAttr().getAttr(), write});
                  if (!inserted)
                    return rewriter.notifyMatchFailure(
                        write,
                        "expected only one ibis.port.write op of the input "
                        "port");
                  return success();
                })
                .Default([&](auto op) {
                  return rewriter.notifyMatchFailure(
                      op, "expected only ibis.port.read or ibis.port.write ops "
                          "of the "
                          "instance");
                });
        if (failed(res))
          return failure();
      }
      getPortsToErase.push_back(getPort);
    }

    // Grab the operands in the order of the hw.module ports.
    const ContainerPortInfo &cpi = portOrder.at(rewriter.getStringAttr(
        getScopeRefModuleName(op.getResult().getType())));
    size_t nInputPorts = std::distance(cpi.hwPorts->getInputs().begin(),
                                       cpi.hwPorts->getInputs().end());
    if (nInputPorts != inputWritesToUse.size()) {
      std::string errMsg;
      llvm::raw_string_ostream ers(errMsg);
      ers << "Error when lowering instance ";
      op.print(ers, mlir::OpPrintingFlags().printGenericOpForm());

      ers << "\nexpected exactly one ibis.port.write op of each input port. "
             "Mising port assignments were:\n";
      for (auto input : cpi.hwPorts->getInputs()) {
        if (inputWritesToUse.find(input.name) == inputWritesToUse.end())
          ers << "\t" << input.name << "\n";
      }
      return rewriter.notifyMatchFailure(op, errMsg);
    }
    for (auto input : cpi.hwPorts->getInputs()) {
      auto writeOp = inputWritesToUse.at(input.name);
      operands.push_back(writeOp.getValue());
      rewriter.eraseOp(writeOp);
    }

    // Determine the result types.
    llvm::SmallVector<Type> retTypes;
    for (auto output : cpi.hwPorts->getOutputs())
      retTypes.push_back(output.type);

    // Gather arg and res names
    // TODO: @mortbopet - this should be part of ModulePortInfo
    llvm::SmallVector<Attribute> argNames, resNames;
    llvm::transform(cpi.hwPorts->getInputs(), std::back_inserter(argNames),
                    [](auto port) { return port.name; });
    llvm::transform(cpi.hwPorts->getOutputs(), std::back_inserter(resNames),
                    [](auto port) { return port.name; });

    // Create the hw.instance op.
    StringRef moduleName = getScopeRefModuleName(op.getType());
    auto hwInst = rewriter.create<hw::InstanceOp>(
        op.getLoc(), retTypes, op.getInnerSym().getSymName(), moduleName,
        operands, rewriter.getArrayAttr(argNames),
        rewriter.getArrayAttr(resNames),
        /*parameters*/ rewriter.getArrayAttr({}), /*innerSym*/ nullptr);

    // Replace the reads of the output ports with the hw.instance results.
    for (auto [output, value] :
         llvm::zip(cpi.hwPorts->getOutputs(), hwInst.getResults())) {
      auto outputReadIt = outputReadsToReplace.find(output.name);
      if (outputReadIt == outputReadsToReplace.end())
        continue;
      rewriter.replaceAllUsesWith(outputReadIt->second.getResult(), value);
      rewriter.eraseOp(outputReadIt->second);
    }

    // Erase the get_port ops.
    for (auto *getPort : getPortsToErase)
      rewriter.eraseOp(getPort);

    // And finally erase the instance op.
    rewriter.eraseOp(op);
    return success();
  }

  ContainerPortInfoMap &portOrder;
}; // namespace

struct ContainersToHWPass : public IbisContainersToHWBase<ContainersToHWPass> {
  void runOnOperation() override;
};
} // anonymous namespace

void ContainersToHWPass::runOnOperation() {
  auto *ctx = &getContext();

  // Generate module signatures.
  ContainerPortInfoMap portOrder;
  for (auto container : getOperation().getOps<ContainerOp>())
    portOrder.try_emplace(container.getSymNameAttr(),
                          ContainerPortInfo(container));

  ConversionTarget target(*ctx);
  target.addIllegalOp<ContainerOp, ContainerInstanceOp, ThisOp>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

  // Parts of the conversion patterns will update operations in place, which in
  // turn requires the updated operations to be legalizeable. These in-place ops
  // also include ibis ops that eventually will get replaced once all of the
  // patterns apply.
  target.addLegalDialect<IbisDialect>();

  RewritePatternSet patterns(ctx);
  patterns
      .add<ContainerOpConversionPattern, ContainerInstanceOpConversionPattern>(
          ctx, portOrder);
  patterns.add<ThisOpConversionPattern>(ctx);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> circt::ibis::createContainersToHWPass() {
  return std::make_unique<ContainersToHWPass>();
}
