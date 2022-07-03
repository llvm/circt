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

def DependenceAttr : AttrDef<SSPDialect, "Dependence"> {
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
    `<` $operandIdx (`,` $sourceRef^)? (`,` $properties^)? `>`
  }];
}

def DependenceArrayAttr
  : TypedArrayAttrBase<DependenceAttr, "dependence array attribute">;

//===----------------------------------------------------------------------===//
// Property definitions for in-tree problems
//===----------------------------------------------------------------------===//

include "PropertyBase.td"

// Problem
def LinkedOperatorType : OperationProperty<SSPDialect,
  "LinkedOperatorType", "::mlir::FlatSymbolRefAttr", "::circt::scheduling::Problem"> {
  let extraClassDeclaration = [{
    void setInProblem(}] # problemClassName # [{ &prob, ::mlir::Operation *op) {
      prob.set}] # propertyName # [{(op, getValue().getLeafReference());
    }
    static ::mlir::Attribute getFromProblem(}] # problemClassName # [{ &prob, ::mlir::Operation *op, ::mlir::MLIRContext *ctx) {
      if (auto value = prob.get}] # propertyName # [{(op))
        return }] # cppClassName # [{::get(ctx, ::mlir::FlatSymbolRefAttr::get(ctx, *value));
      return {};
    }
  }];
}
def StartTime : OperationProperty<SSPDialect,
  "StartTime", "unsigned", "::circt::scheduling::Problem">;
def Latency : OperatorTypeProperty<SSPDialect,
  "Latency", "unsigned", "::circt::scheduling::Problem">;

// CyclicProblem
def Distance : DependenceProperty<SSPDialect,
  "Distance", "unsigned", "::circt::scheduling::CyclicProblem">;
def InitiationInterval : InstanceProperty<SSPDialect,
  "InitiationInterval", "unsigned", "::circt::scheduling::CyclicProblem">;

// SharedOperatorsProblem
def Limit : OperatorTypeProperty<SSPDialect,
  "Limit", "unsigned", "::circt::scheduling::SharedOperatorsProblem">;
