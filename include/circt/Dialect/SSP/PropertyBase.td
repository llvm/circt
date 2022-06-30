//===- PropertyBase.td - Base classes for SSP properties ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines base classes for attributes representing properties in the
// SSP (static scheduling problem) dialect.
//
//===----------------------------------------------------------------------===//

#ifndef PROPERTYBASE_TD
#define PROPERTYBASE_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

class PropertyBase<Dialect dialect,
                   string name, string type,
                   string problem,
                   list<Trait> traits = []>
  : AttrDef<dialect, name, traits> {
  
  string propertyName = name;
  string propertyType = type;
  string problemClassName = problem;

  let description = [{
    Models the "}] # propertyName # [{" property in }] # problemClassName # [{.
  }];

  let parameters = (ins propertyType:$value);
  let mnemonic = propertyName;
  let assemblyFormat = [{ `<` $value `>` }];
}

class OperationProperty<Dialect dialect, string name, string type, string problem,
                        list<Trait> traits = []>
    : PropertyBase<dialect, name, type, problem, traits> {
  let extraClassDeclaration = [{
    void setInProblem(}] # problemClassName # [{ &prob, ::mlir::Operation *op) {
      prob.set}] # propertyName # [{(op, getValue());
    }
    static ::mlir::Attribute getFromProblem(}] # problemClassName # [{ &prob, ::mlir::Operation *op, ::mlir::MLIRContext *ctx) {
      if (auto value = prob.get}] # propertyName # [{(op))
        return }] # cppClassName # [{::get(ctx, *value);
      return {};
    }
  }];
}

class OperatorTypeProperty<Dialect dialect, string name, string type, string problem,
                           list<Trait> traits = []>
    : PropertyBase<dialect, name, type, problem, traits> {
  let extraClassDeclaration = [{
    void setInProblem(}] # problemClassName # [{ &prob, ::circt::scheduling::Problem::OperatorType opr) {
      prob.set}] # propertyName # [{(opr, getValue());
    }
    static ::mlir::Attribute getFromProblem(}] # problemClassName # [{ &prob, ::circt::scheduling::Problem::OperatorType opr, ::mlir::MLIRContext *ctx) {
      if (auto value = prob.get}] # propertyName # [{(opr))
        return }] # cppClassName # [{::get(ctx, *value);
      return {};
    }
  }];
}

class DependenceProperty<Dialect dialect, string name, string type, string problem,
                         list<Trait> traits = []>
    : PropertyBase<dialect, name, type, problem, traits> {
  let extraClassDeclaration = [{
    void setInProblem(}] # problemClassName # [{ &prob, ::circt::scheduling::Problem::Dependence dep) {
      prob.set}] # propertyName # [{(dep, getValue());
    }
    static ::mlir::Attribute getFromProblem(}] # problemClassName # [{ &prob, ::circt::scheduling::Problem::Dependence dep, ::mlir::MLIRContext *ctx) {
      if (auto value = prob.get}] # propertyName # [{(dep))
        return }] # cppClassName # [{::get(ctx, *value);
      return {};
    }
  }];
}

class InstanceProperty<Dialect dialect, string name, string type, string problem,
                       list<Trait> traits = []>
    : PropertyBase<dialect, name, type, problem, traits> {
  let extraClassDeclaration = [{
    void setInProblem(}] # problemClassName # [{ &prob) {
      prob.set}] # propertyName # [{(getValue());
    }
    static ::mlir::Attribute getFromProblem(}] # problemClassName # [{ &prob, ::mlir::MLIRContext *ctx) {
      if (auto value = prob.get}] # propertyName # [{())
        return }] # cppClassName # [{::get(ctx, *value);
      return {};
    }
  }];
}

#endif // PROPERTYBASE_TD
