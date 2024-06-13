//===- LowerDPI.cpp - Lower to DPI to Sim dialects ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LowerDPI pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/MapVector.h"

using namespace mlir;
using namespace llvm;
using namespace circt;
using namespace circt::firrtl;

struct LowerDPIPass : public LowerDPIBase<LowerDPIPass> {
  void runOnOperation() override;
};

void LowerDPIPass::runOnOperation() {
  auto circuitOp = getOperation();

  CircuitNamespace nameSpace(circuitOp);
  MapVector<StringAttr, SmallVector<DPICallIntrinsicOp>> funcNameToCallSites;
  {
    // A helper struct to collect DPI calls in the circuit.
    struct DpiCallCollections {
      FModuleOp module;
      SmallVector<DPICallIntrinsicOp> dpiOps;
    };

    SmallVector<DpiCallCollections, 0> collections;
    collections.reserve(64);

    for (auto module : circuitOp.getOps<FModuleOp>())
      collections.push_back(DpiCallCollections{module, {}});

    parallelForEach(&getContext(), collections, [](auto &result) {
      result.module.walk(
          [&](DPICallIntrinsicOp dpi) { result.dpiOps.push_back(dpi); });
    });

    for (auto &collection : collections)
      for (auto dpi : collection.dpiOps)
        funcNameToCallSites[dpi.getFunctionNameAttr()].push_back(dpi);
  }

  for (auto [name, calls] : funcNameToCallSites) {
    auto firstDPICallop = calls.front();
    // Construct DPI func op.
    auto inputTypes = firstDPICallop.getInputs().getTypes();
    auto outputTypes = firstDPICallop.getResultTypes();
    SmallVector<hw::ModulePort> ports;
    ImplicitLocOpBuilder builder(firstDPICallop.getLoc(),
                                 circuitOp.getOperation());
    ports.reserve(inputTypes.size() + outputTypes.size());

    // Add input arguments.
    for (auto [idx, inType] : llvm::enumerate(inputTypes)) {
      hw::ModulePort port;
      port.dir = hw::ModulePort::Direction::Input;
      port.name = builder.getStringAttr(Twine("in_") + Twine(idx));
      port.type = lowerType(inType);
      ports.push_back(port);
    }

    // Add output arguments.
    for (auto [idx, outType] : llvm::enumerate(outputTypes)) {
      hw::ModulePort port;
      port.dir = hw::ModulePort::Direction::Output;
      port.name = builder.getStringAttr(Twine("out_") + Twine(idx));
      port.type = lowerType(outType);
      ports.push_back(port);
    }

    auto modType = hw::ModuleType::get(&getContext(), ports);
    auto funcSymbol =
        nameSpace.newName(firstDPICallop.getFunctionNameAttr().getValue());
    builder.setInsertionPointToStart(circuitOp.getBodyBlock());
    auto sim = builder.create<sim::DPIFuncOp>(
        funcSymbol, modType, ArrayAttr(), ArrayAttr(),
        firstDPICallop.getFunctionNameAttr());
    sim.setPrivate();

    auto lowerCall = [&builder, funcSymbol](DPICallIntrinsicOp dpiOp) {
      auto getLowered = [&](Value value) -> Value {
        // Insert an unrealized conversion to cast FIRRTL type to HW type.
        if (!value)
          return value;
        auto type = lowerType(value.getType());
        return builder.create<mlir::UnrealizedConversionCastOp>(type, value)
            ->getResult(0);
      };
      builder.setInsertionPoint(dpiOp);
      auto clock = getLowered(dpiOp.getClock());
      auto enable = getLowered(dpiOp.getEnable());
      SmallVector<Value, 4> inputs;
      inputs.reserve(dpiOp.getInputs().size());
      for (auto input : dpiOp.getInputs())
        inputs.push_back(getLowered(input));

      SmallVector<Type> outputTypes;
      if (dpiOp.getResult())
        outputTypes.push_back(lowerType(dpiOp.getResult().getType()));

      auto call = builder.create<sim::DPICallOp>(outputTypes, funcSymbol, clock,
                                                 enable, inputs);
      if (!call.getResults().empty()) {
        // Insert unrealized conversion cast HW type to FIRRTL type.
        auto result = builder
                          .create<mlir::UnrealizedConversionCastOp>(
                              dpiOp.getResult().getType(), call.getResult(0))
                          ->getResult(0);
        dpiOp.getResult().replaceAllUsesWith(result);
      }
      dpiOp.erase();
    };

    lowerCall(firstDPICallop);
    for (auto dpiOp : llvm::ArrayRef(calls).drop_front()) {
      // Check that all DPI declaration match.
      // TODO: This should be implemented as a verifier once function is added
      //       to FIRRTL.
      if (dpiOp.getInputs().getTypes() != inputTypes) {
        auto diag = firstDPICallop.emitOpError()
                    << "DPI function " << firstDPICallop.getFunctionNameAttr()
                    << " input types don't match ";
        diag.attachNote(dpiOp.getLoc()) << " mismatched caller is here";
        return signalPassFailure();
      }
      if (dpiOp.getResultTypes() != outputTypes) {
        auto diag = firstDPICallop.emitOpError()
                    << "DPI function " << firstDPICallop.getFunctionNameAttr()
                    << " output types don't match";
        diag.attachNote(dpiOp.getLoc()) << " mismatched caller is here";
        return signalPassFailure();
      }
      lowerCall(dpiOp);
    }
  }
}

std::unique_ptr<mlir::Pass> circt::firrtl::createLowerDPIPass() {
  return std::make_unique<LowerDPIPass>();
}
