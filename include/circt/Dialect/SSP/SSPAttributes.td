//===- SSPAttributes.td - SSP attribute definitions --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the SSP (static scheduling problem) dialect attributes.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Dialect attributes
//===----------------------------------------------------------------------===//

def DependenceAttr : AttrDef<SSPDialect, "Dependence"> {
  let summary = "Internal representation of dependence edges.";
  let description = [{
    An attribute to uniformly model def-use and auxiliary
    [dependences](https://circt.llvm.org/docs/Scheduling/#components) as well as
    to attach [properties](https://circt.llvm.org/docs/Scheduling/#properties)
    to them. This attribute is an implementation detail of the `ssp.OperationOp`
    and as such is supposed to be hidden by the custom parser/printer.
  }];

  let parameters = (ins "unsigned":$operandIdx,
                        OptionalParameter<"::mlir::FlatSymbolRefAttr">:$sourceRef,
                        OptionalParameter<"::mlir::ArrayAttr">:$properties);

  let mnemonic = "dependence";
  let assemblyFormat = [{
    `<` $operandIdx (`,` $sourceRef^)? (`:` $properties^)? `>`
  }];
}

def DependenceArrayAttr
  : TypedArrayAttrBase<DependenceAttr, "dependence array attribute">;

//===----------------------------------------------------------------------===//
// Property definitions for in-tree problems
//===----------------------------------------------------------------------===//

include "PropertyBase.td"

// Problem
def LinkedOperatorTypeProp : OperationProperty<SSPDialect,
  "LinkedOperatorType", "::mlir::SymbolRefAttr", "::circt::scheduling::Problem"> {
  let mnemonic = "opr";
  let unwrapValue = [{ getValue().getLeafReference() }];
  let wrapValue = [{ ::mlir::FlatSymbolRefAttr::get(ctx, value.getAttr()) }];
}
def LinkedResourceTypesProp : OperationProperty<SSPDialect,
  "LinkedResourceTypes", "::mlir::ArrayAttr", "::circt::scheduling::Problem"> {
  let mnemonic = "rsrcs";
  let extraClassDeclaration = [{
    static SmallVector<::circt::scheduling::Problem::ResourceType>
    arrayAttrToResourceTypes(::mlir::ArrayAttr attr) {
      SmallVector<::circt::scheduling::Problem::ResourceType> result;
      for (auto a : attr.getAsRange<::mlir::SymbolRefAttr>())
        result.push_back(::circt::scheduling::Problem::ResourceType(a.getLeafReference()));
      return result;
    }

    static ::mlir::ArrayAttr
    resourceTypesToArrayAttr(::mlir::MLIRContext *ctx,
                              ::llvm::ArrayRef<::circt::scheduling::Problem::ResourceType> rsrcs) {
      SmallVector<::mlir::Attribute> attrs;
      for (auto r : rsrcs) {
        auto value = r.getAttr().getValue();
        auto strAttr = mlir::StringAttr::get(ctx, value);
        attrs.push_back(::mlir::FlatSymbolRefAttr::get(strAttr));
      }
      return ::mlir::ArrayAttr::get(ctx, attrs);
    }

    void setInProblem(::circt::scheduling::Problem &prob, ::mlir::Operation *op) {
      SmallVector<::circt::scheduling::Problem::ResourceType> result;
      for (auto a : getValue().getAsRange<::mlir::SymbolRefAttr>()) {
        auto attr = a.getLeafReference();
        result.push_back(::circt::scheduling::Problem::ResourceType(attr));
      }
      prob.setLinkedResourceTypes(op, result);
    }

    static ::mlir::Attribute getFromProblem(::circt::scheduling::Problem &prob,
                                            ::mlir::Operation *op,
                                            ::mlir::MLIRContext *context) {
      if (auto optValue = prob.getLinkedResourceTypes(op)) {
        const SmallVector<::circt::scheduling::Problem::ResourceType> &vec = *optValue;
        return LinkedResourceTypesAttr::get(context, resourceTypesToArrayAttr(context, vec));
      }
      return {};
    }
  }];
  let unwrapValue = [{ arrayAttrToResourceTypes(getValue()) }];
  let wrapValue = [{ resourceTypesToArrayAttr(context, value) }];
}
def StartTimeProp : OperationProperty<SSPDialect,
  "StartTime", "unsigned", "::circt::scheduling::Problem"> {
  let mnemonic = "t";
}
def LatencyProp : OperatorTypeProperty<SSPDialect,
  "Latency", "unsigned", "::circt::scheduling::Problem"> {
  let mnemonic = "latency";
}

// CyclicProblem
def DistanceProp : DependenceProperty<SSPDialect,
  "Distance", "unsigned", "::circt::scheduling::CyclicProblem"> {
  let mnemonic = "dist";
}
def InitiationIntervalProp : InstanceProperty<SSPDialect,
  "InitiationInterval", "unsigned", "::circt::scheduling::CyclicProblem"> {
  let mnemonic = "II";
}

// ChainingProblem
let
  unwrapValue = [{ (float) getValue().getValueAsDouble() }],
  wrapValue = [{ ::mlir::FloatAttr::get(::mlir::Float32Type::get(ctx), value) }]
in {
  def StartTimeInCycleProp : OperationProperty<SSPDialect,
    "StartTimeInCycle", "::mlir::FloatAttr", "::circt::scheduling::ChainingProblem"> {
    let mnemonic = "z";
  }
  def IncomingDelayProp : OperatorTypeProperty<SSPDialect,
    "IncomingDelay", "::mlir::FloatAttr", "::circt::scheduling::ChainingProblem"> {
    let mnemonic = "incDelay";
  }
  def OutgoingDelayProp : OperatorTypeProperty<SSPDialect,
    "OutgoingDelay", "::mlir::FloatAttr", "::circt::scheduling::ChainingProblem"> {
    let mnemonic = "outDelay";
  }
}

// SharedOperatorsProblem
def LimitProp : ResourceTypeProperty<SSPDialect,
  "Limit", "unsigned", "::circt::scheduling::SharedOperatorsProblem"> {
  let mnemonic = "limit";
}
