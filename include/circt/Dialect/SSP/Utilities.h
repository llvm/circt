//===- Utilities.h - SSP <-> circt::scheduling infra conversion -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides utilities for the conversion between SSP IR and the
// extensible problem model in the scheduling infrastructure.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SSP_SSPUTILITIES_H
#define CIRCT_DIALECT_SSP_SSPUTILITIES_H

#include "circt/Dialect/SSP/SSPAttributes.h"
#include "circt/Dialect/SSP/SSPOps.h"
#include "circt/Scheduling/Problems.h"

#include "llvm/ADT/TypeSwitch.h"

namespace circt {
namespace ssp {

using OperatorType = scheduling::Problem::OperatorType;
using Dependence = scheduling::Problem::Dependence;

template <typename ProblemT>
void loadOperationProperties(ProblemT &, Operation *, ArrayAttr) {}
template <typename ProblemT, typename OperationPropertyT,
          typename... OperationPropertyTs>
void loadOperationProperties(ProblemT &prob, Operation *op,
                             ArrayAttr propsAttr) {
  if (!propsAttr)
    return;
  for (auto prop : propsAttr) {
    TypeSwitch<Attribute>(prop)
        .Case<OperationPropertyT, OperationPropertyTs...>(
            [&](auto p) { p.setInProblem(prob, op); });
  }
}

template <typename ProblemT>
void loadOperatorTypeProperties(ProblemT &, OperatorType, ArrayAttr) {}
template <typename ProblemT, typename OperatorTypePropertyT,
          typename... OperatorTypePropertyTs>
void loadOperatorTypeProperties(ProblemT &prob, OperatorType opr,
                                ArrayAttr propsAttr) {
  if (!propsAttr)
    return;
  for (auto prop : propsAttr) {
    TypeSwitch<Attribute>(prop)
        .Case<OperatorTypePropertyT, OperatorTypePropertyTs...>(
            [&](auto p) { p.setInProblem(prob, opr); });
  }
}

template <typename ProblemT>
void loadDependenceProperties(ProblemT &, Dependence, ArrayAttr) {}
template <typename ProblemT, typename DependencePropertyT,
          typename... DependencePropertyTs>
void loadDependenceProperties(ProblemT &prob, Dependence dep,
                              ArrayAttr propsAttr) {
  if (!propsAttr)
    return;
  for (auto prop : propsAttr) {
    TypeSwitch<Attribute>(prop)
        .Case<DependencePropertyT, DependencePropertyTs...>(
            [&](auto p) { p.setInProblem(prob, dep); });
  }
}

template <typename ProblemT>
void loadInstanceProperties(ProblemT &, ArrayAttr) {}
template <typename ProblemT, typename InstancePropertyT,
          typename... InstancePropertyTs>
void loadInstanceProperties(ProblemT &prob, ArrayAttr propsAttr) {
  if (!propsAttr)
    return;
  for (auto prop : propsAttr) {
    TypeSwitch<Attribute>(prop).Case<InstancePropertyT, InstancePropertyTs...>(
        [&](auto p) { p.setInProblem(prob); });
  }
}

template <typename ProblemT, typename... OperationPropertyTs,
          typename... OperatorTypePropertyTs, typename... DependencePropertyTs,
          typename... InstancePropertyTs>
ProblemT loadProblem(InstanceOp instOp,
                     std::tuple<OperationPropertyTs...> opProps,
                     std::tuple<OperatorTypePropertyTs...> oprProps,
                     std::tuple<DependencePropertyTs...> depProps,
                     std::tuple<InstancePropertyTs...> instProps) {
  auto prob = ProblemT::get(instOp);

  loadInstanceProperties<ProblemT, InstancePropertyTs...>(
      prob, instOp.getPropertiesAttr());

  instOp.walk([&](OperatorTypeOp oprOp) {
    OperatorType opr = oprOp.getNameAttr();
    prob.insertOperatorType(opr);
    loadOperatorTypeProperties<ProblemT, OperatorTypePropertyTs...>(
        prob, opr, oprOp.getPropertiesAttr());
  });

  instOp.walk([&](OperationOp opOp) {
    prob.insertOperation(opOp);
    loadOperationProperties<ProblemT, OperationPropertyTs...>(
        prob, opOp, opOp.getPropertiesAttr());

    for (auto depAttr : opOp.getDependences().getAsRange<DependenceAttr>()) {
      Dependence dep;
      if (FlatSymbolRefAttr sourceRef = depAttr.getSourceRef()) {
        Operation *sourceOp = SymbolTable::lookupSymbolIn(instOp, sourceRef);
        assert(sourceOp);
        dep = Dependence(sourceOp, opOp);
      } else if (IntegerAttr operandIdx = depAttr.getOperandIdx())
        dep = Dependence(&opOp->getOpOperand(operandIdx.getInt()));
      else
        llvm_unreachable("Malformed dependence attribute");

      // Make sure the dependence (and its endpoints) are registered.
      LogicalResult res = prob.insertDependence(dep);
      assert(succeeded(res));
      (void)res;

      loadDependenceProperties<ProblemT, DependencePropertyTs...>(
          prob, dep, depAttr.getProperties());
    }
  });

  return prob;
}

} // namespace ssp
} // namespace circt

#endif // CIRCT_DIALECT_SSP_SSPUTILITIES_H
