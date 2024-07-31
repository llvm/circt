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

#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/MapVector.h"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_LOWERDPI
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace mlir;
using namespace llvm;
using namespace circt;
using namespace circt::firrtl;

namespace {
struct LowerDPIPass : public circt::firrtl::impl::LowerDPIBase<LowerDPIPass> {
  void runOnOperation() override;
};

// A helper struct to lower DPI intrinsics in the circuit.
struct LowerDPI {
  LowerDPI(CircuitOp circuitOp) : circuitOp(circuitOp), nameSpace(circuitOp) {}
  // Tte main logic.
  LogicalResult run();
  bool changed() const { return !funcNameToCallSites.empty(); }

private:
  // Walk all modules and peel `funcNameToCallSites`.
  void collectIntrinsics();

  // Lower intrinsics recorded in `funcNameToCallSites`.
  LogicalResult lower();

  sim::DPIFuncOp getOrCreateDPIFuncDecl(DPICallIntrinsicOp op);
  LogicalResult lowerDPIIntrinsic(DPICallIntrinsicOp op);

  MapVector<StringAttr, SmallVector<DPICallIntrinsicOp>> funcNameToCallSites;

  // A map stores DPI func op for its function name and type.
  llvm::DenseMap<std::pair<StringAttr, Type>, sim::DPIFuncOp>
      functionSignatureToDPIFuncOp;

  firrtl::CircuitOp circuitOp;
  CircuitNamespace nameSpace;
};
} // namespace

void LowerDPI::collectIntrinsics() {
  // A helper struct to collect DPI calls in the circuit.
  struct DpiCallCollections {
    FModuleOp module;
    SmallVector<DPICallIntrinsicOp> dpiOps;
  };

  SmallVector<DpiCallCollections, 0> collections;
  collections.reserve(64);

  for (auto module : circuitOp.getOps<FModuleOp>())
    collections.push_back(DpiCallCollections{module, {}});

  parallelForEach(circuitOp.getContext(), collections, [](auto &result) {
    result.module.walk(
        [&](DPICallIntrinsicOp dpi) { result.dpiOps.push_back(dpi); });
  });

  for (auto &collection : collections)
    for (auto dpi : collection.dpiOps)
      funcNameToCallSites[dpi.getFunctionNameAttr()].push_back(dpi);
}

LogicalResult LowerDPI::lower() {
  for (auto [name, calls] : funcNameToCallSites) {
    auto firstDPICallop = calls.front();
    // Construct DPI func op.
    auto firstDPIDecl = getOrCreateDPIFuncDecl(firstDPICallop);

    auto inputTypes = firstDPICallop.getInputs().getTypes();
    auto outputTypes = firstDPICallop.getResultTypes();

    ImplicitLocOpBuilder builder(firstDPICallop.getLoc(),
                                 circuitOp.getOperation());
    auto lowerCall = [&](DPICallIntrinsicOp dpiOp) {
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

      auto call = builder.create<sim::DPICallOp>(
          outputTypes, firstDPIDecl.getSymNameAttr(), clock, enable, inputs);
      if (!call.getResults().empty()) {
        // Insert unrealized conversion cast HW type to FIRRTL type.
        auto result = builder
                          .create<mlir::UnrealizedConversionCastOp>(
                              dpiOp.getResult().getType(), call.getResult(0))
                          ->getResult(0);
        dpiOp.getResult().replaceAllUsesWith(result);
      }
      return success();
    };

    if (failed(lowerCall(firstDPICallop)))
      return failure();

    for (auto dpiOp : llvm::ArrayRef(calls).drop_front()) {
      // Check that all DPI declaration match.
      // TODO: This should be implemented as a verifier once function is added
      //       to FIRRTL.
      if (dpiOp.getInputs().getTypes() != inputTypes) {
        auto diag = firstDPICallop.emitOpError()
                    << "DPI function " << firstDPICallop.getFunctionNameAttr()
                    << " input types don't match ";
        diag.attachNote(dpiOp.getLoc()) << " mismatched caller is here";
        return failure();
      }

      if (dpiOp.getResultTypes() != outputTypes) {
        auto diag = firstDPICallop.emitOpError()
                    << "DPI function " << firstDPICallop.getFunctionNameAttr()
                    << " output types don't match";
        diag.attachNote(dpiOp.getLoc()) << " mismatched caller is here";
        return failure();
      }

      if (failed(lowerCall(dpiOp)))
        return failure();
    }

    for (auto callOp : calls)
      callOp.erase();
  }

  return success();
}

sim::DPIFuncOp LowerDPI::getOrCreateDPIFuncDecl(DPICallIntrinsicOp op) {
  ImplicitLocOpBuilder builder(op.getLoc(), circuitOp.getOperation());
  builder.setInsertionPointToStart(circuitOp.getBodyBlock());
  auto inputTypes = op.getInputs().getTypes();
  auto outputTypes = op.getResultTypes();
  ArrayAttr inputNames = op.getInputNamesAttr();
  StringAttr outputName = op.getOutputNameAttr();
  assert(outputTypes.size() <= 1);

  SmallVector<hw::ModulePort> ports;
  ports.reserve(inputTypes.size() + outputTypes.size());

  // Add input arguments.
  for (auto [idx, inType] : llvm::enumerate(inputTypes)) {
    hw::ModulePort port;
    port.dir = hw::ModulePort::Direction::Input;
    port.name = inputNames ? cast<StringAttr>(inputNames[idx])
                           : builder.getStringAttr(Twine("in_") + Twine(idx));
    port.type = lowerType(inType);
    ports.push_back(port);
  }

  // Add output arguments.
  for (auto [idx, outType] : llvm::enumerate(outputTypes)) {
    hw::ModulePort port;
    port.dir = hw::ModulePort::Direction::Output;
    port.name = outputName ? outputName
                           : builder.getStringAttr(Twine("out_") + Twine(idx));
    port.type = lowerType(outType);
    ports.push_back(port);
  }

  auto modType = hw::ModuleType::get(builder.getContext(), ports);
  auto it =
      functionSignatureToDPIFuncOp.find({op.getFunctionNameAttr(), modType});
  if (it != functionSignatureToDPIFuncOp.end())
    return it->second;

  auto funcSymbol = nameSpace.newName(op.getFunctionNameAttr().getValue());
  auto funcOp = builder.create<sim::DPIFuncOp>(
      funcSymbol, modType, ArrayAttr(), ArrayAttr(), op.getFunctionNameAttr());
  // External function must have a private linkage.
  funcOp.setPrivate();
  functionSignatureToDPIFuncOp[{op.getFunctionNameAttr(), modType}] = funcOp;
  return funcOp;
}

LogicalResult LowerDPI::run() {
  collectIntrinsics();
  return lower();
}

void LowerDPIPass::runOnOperation() {
  auto circuitOp = getOperation();
  LowerDPI lowerDPI(circuitOp);
  if (failed(lowerDPI.run()))
    return signalPassFailure();
  if (!lowerDPI.changed())
    return markAllAnalysesPreserved();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createLowerDPIPass() {
  return std::make_unique<LowerDPIPass>();
}
