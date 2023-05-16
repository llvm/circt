//===- BuildOM.cpp - Build OM IR from modules and properties ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the BuildOM pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/OM/OMOps.h"

#define DEBUG_TYPE "build-om"

using namespace circt;
using namespace firrtl;

namespace {
class BuildOMPass : public BuildOMBase<BuildOMPass> {
  void runOnOperation() override {
    auto circuits = getOperation().getOps<CircuitOp>();
    if (circuits.empty())
      return;
    auto circuitOp = *circuits.begin();

    // First register all const ports
    for (auto moduleOp : circuitOp.getOps<FModuleOp>())
      registerConstPorts(moduleOp);

    auto builder = mlir::OpBuilder::atBlockEnd(topModuleBody());

    for (auto moduleOp : circuitOp.getOps<FModuleOp>())
      buildOM(moduleOp, builder);
  }

  /// Build up `moduleConstPortIndices` so that instance const ports can be
  /// looked up when building om objects for a module's om class
  void registerConstPorts(FModuleOp moduleOp) {
    auto &constPortIndices = moduleConstPortIndices[moduleOp.getNameAttr()];
    constPortIndices.resize(moduleOp.getNumPorts());
    for (size_t portIndex = 0, numPorts = moduleOp.getNumPorts();
         portIndex < numPorts; ++portIndex) {
      if (AnnotationSet(moduleOp.getAnnotationsAttrForPort(portIndex))
              .hasAnnotation(constPropertyAnnoClass)) {
        constPortIndices.set(portIndex);
      }
    }
  }

  /// Build the om class for the module
  void buildOM(FModuleOp moduleOp, OpBuilder &builder) {
    auto &constPortIndices = moduleConstPortIndices[moduleOp.getNameAttr()];
    SmallVector<Attribute> parameterNames;
    DenseMap<Value, Value> firrtlToOmValues;
    SmallVector<std::pair<StringAttr, Value>> fields;

    auto block = std::make_unique<Block>();
    builder.setInsertionPointToStart(block.get());

    for (size_t portIndex : constPortIndices.set_bits()) {
      auto portName = moduleOp.getPortNameAttr(portIndex);
      if (moduleOp.getPortDirection(portIndex) == Direction::In) {
        parameterNames.push_back(portName);
        auto parameter =
            block->addArgument(moduleOp.getPortType(portIndex),
                               moduleOp.getPortLocation(portIndex));
        firrtlToOmValues[moduleOp.getArgument(portIndex)] = parameter;
      } else {
        fields.push_back({portName, moduleOp.getArgument(portIndex)});
      }
    }

    auto getOMValue = [&](Value firrtlValue, StringRef name) -> Value {
      for (auto *user :
           llvm::reverse(SmallVector<Operation *>(firrtlValue.getUsers()))) {
        if (auto connect = dyn_cast<FConnectLike>(user);
            connect && connect.getDest() == firrtlValue) {
          auto src = connect.getSrc();
          if (auto it = firrtlToOmValues.find(src);
              it != firrtlToOmValues.end()) {
            return it->second;
          }

          if (auto constantOp = dyn_cast<ConstantOp>(src.getDefiningOp())) {
            auto omConstant =
                builder
                    .create<om::ConstantOp>(constantOp->getLoc(),
                                            constantOp.getType(),
                                            constantOp.getValueAttr())
                    .getResult();
            firrtlToOmValues[src] = omConstant;
            return omConstant;
          }
          mlir::emitError(firrtlValue.getLoc())
              << "Could not determine om value for " << name;
          signalPassFailure();
          return {};
        }
      }
      mlir::emitError(firrtlValue.getLoc())
          << "Could not determine om value for " << name;
      signalPassFailure();
      return {};
    };

    for (auto instanceOp : moduleOp.getOps<InstanceOp>()) {
      SmallVector<Value> objectParameters;
      auto constPortIndices =
          moduleConstPortIndices.find(instanceOp.getModuleNameAttr().getAttr());
      if (constPortIndices == moduleConstPortIndices.end())
        continue;

      for (size_t portIndex : constPortIndices->second.set_bits()) {
        if (instanceOp.getPortDirection(portIndex) == Direction::In) {
          if (auto value = getOMValue(instanceOp->getResult(portIndex),
                                      instanceOp.getPortNameStr(portIndex)))
            objectParameters.push_back(value);
          else
            return;
        }
      }

      auto object = builder
                        .create<om::ObjectOp>(
                            instanceOp->getLoc(),
                            om::ClassType::get(&getContext(),
                                               instanceOp.getModuleNameAttr()),
                            instanceOp.getModuleName(), objectParameters)
                        .getResult();

      for (size_t portIndex : constPortIndices->second.set_bits()) {
        if (instanceOp.getPortDirection(portIndex) == Direction::Out) {
          auto field = builder
                           .create<om::ObjectFieldOp>(
                               instanceOp->getLoc(),
                               instanceOp->getResultTypes()[portIndex], object,
                               builder.getArrayAttr(FlatSymbolRefAttr::get(
                                   instanceOp.getPortName(portIndex))))
                           .getResult();
          firrtlToOmValues[instanceOp->getResult(portIndex)] = field;
        }
      }
    }

    for (auto [fieldName, portValue] : fields) {
      if (auto value = getOMValue(portValue, fieldName.getValue()))
        builder.create<om::ClassFieldOp>(portValue.getLoc(), fieldName, value);
      else
        return;
    }

    builder.setInsertionPointToEnd(topModuleBody());
    auto classOp = builder.create<om::ClassOp>(
        moduleOp->getLoc(), moduleOp.getModuleNameAttr(),
        builder.getArrayAttr(parameterNames));
    classOp.getBody().push_back(block.release());
  }

  Block *topModuleBody() { return getOperation().getBody(); }

  DenseMap<StringAttr, llvm::BitVector> moduleConstPortIndices;
};
} // namespace

std::unique_ptr<mlir::Pass> circt::firrtl::createBuildOMPass() {
  return std::make_unique<BuildOMPass>();
}
