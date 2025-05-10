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
#include "circt/Support/ValueMapper.h"

#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"

#include <functional>

namespace circt {
namespace ssp {

using OperatorType = scheduling::Problem::OperatorType;
using ResourceType = scheduling::Problem::ResourceType;
using Dependence = scheduling::Problem::Dependence;

//===----------------------------------------------------------------------===//
// ssp.InstanceOp -> circt::scheduling::Problem (or subclasses)
//===----------------------------------------------------------------------===//

template <typename ProblemT>
void loadOperationProperties(ProblemT &, Operation *, ArrayAttr) {}
template <typename ProblemT, typename OperationPropertyT,
          typename... OperationPropertyTs>
void loadOperationProperties(ProblemT &prob, Operation *op, ArrayAttr props) {
  if (!props)
    return;
  for (auto prop : props) {
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
                                ArrayAttr props) {
  if (!props)
    return;
  for (auto prop : props) {
    TypeSwitch<Attribute>(prop)
        .Case<OperatorTypePropertyT, OperatorTypePropertyTs...>(
            [&](auto p) { p.setInProblem(prob, opr); });
  }
}

template <typename ProblemT>
void loadResourceTypeProperties(ProblemT &, ResourceType, ArrayAttr) {}
template <typename ProblemT, typename ResourceTypePropertyT,
          typename... ResourceTypePropertyTs>
void loadResourceTypeProperties(ProblemT &prob, ResourceType rsrc,
                                ArrayAttr props) {
  if (!props)
    return;
  for (auto prop : props) {
    TypeSwitch<Attribute>(prop)
        .Case<ResourceTypePropertyT, ResourceTypePropertyTs...>(
            [&](auto p) { p.setInProblem(prob, rsrc); });
  }
}

template <typename ProblemT>
void loadDependenceProperties(ProblemT &, Dependence, ArrayAttr) {}
template <typename ProblemT, typename DependencePropertyT,
          typename... DependencePropertyTs>
void loadDependenceProperties(ProblemT &prob, Dependence dep, ArrayAttr props) {
  if (!props)
    return;
  for (auto prop : props) {
    TypeSwitch<Attribute>(prop)
        .Case<DependencePropertyT, DependencePropertyTs...>(
            [&](auto p) { p.setInProblem(prob, dep); });
  }
}

template <typename ProblemT>
void loadInstanceProperties(ProblemT &, ArrayAttr) {}
template <typename ProblemT, typename InstancePropertyT,
          typename... InstancePropertyTs>
void loadInstanceProperties(ProblemT &prob, ArrayAttr props) {
  if (!props)
    return;
  for (auto prop : props) {
    TypeSwitch<Attribute>(prop).Case<InstancePropertyT, InstancePropertyTs...>(
        [&](auto p) { p.setInProblem(prob); });
  }
}

/// Load the operator type represented by \p oprOp into \p prob under a unique
/// name informed by \p oprIds, and attempt to set its properties from the
/// given attribute classes. The registered name is returned. The template
/// instantiation fails if properties are incompatible with \p ProblemT.
template <typename ProblemT, typename... OperatorTypePropertyTs>
typename ProblemT::OperatorType loadOperatorType(
    ProblemT &prob, OperatorTypeOp oprOp,
    SmallDenseMap<typename ProblemT::OperatorType, unsigned> &oprIds) {
  OperatorType opr = oprOp.getNameAttr();
  unsigned &id = oprIds[opr];
  if (id > 0)
    opr = StringAttr::get(oprOp.getContext(),
                          opr.getValue() + Twine('_') + Twine(id));
  ++id;
  assert(!prob.hasOperatorType(opr));
  prob.insertOperatorType(opr);
  loadOperatorTypeProperties<ProblemT, OperatorTypePropertyTs...>(
      prob, opr, oprOp.getSspPropertiesAttr());
  return opr;
}

/// Load the resource type represented by \p rsrcOp into \p prob under a unique
/// name informed by \p rsrcIds, and attempt to set its properties from the
/// given attribute classes. The registered name is returned. The template
/// instantiation fails if properties are incompatible with \p ProblemT.
template <typename ProblemT, typename... ResourceTypePropertyTs>
typename ProblemT::ResourceType loadResourceType(
    ProblemT &prob, ResourceTypeOp rsrcOp,
    SmallDenseMap<typename ProblemT::ResourceType, unsigned> &rsrcIds) {
  ResourceType rsrc = rsrcOp.getNameAttr();
  unsigned &id = rsrcIds[rsrc];
  if (id > 0)
    rsrc = StringAttr::get(rsrcOp.getContext(),
                           rsrc.getValue() + Twine('_') + Twine(id));
  ++id;
  assert(!prob.hasResourceType(rsrc));
  prob.insertResourceType(rsrc);
  loadResourceTypeProperties<ProblemT, ResourceTypePropertyTs...>(
      prob, rsrc, rsrcOp.getSspPropertiesAttr());
  return rsrc;
}

/// Construct an instance of \p ProblemT from \p instOp, and attempt to set
/// properties from the given attribute classes. The attribute tuples are used
/// solely for grouping/inferring the template parameter packs. The tuple
/// elements may therefore be unitialized objects. The template instantiation
/// fails if properties are incompatible with \p ProblemT.
///
/// Operations may link to operator types in other libraries, but the origin of
/// an operator type will not be preserved in the problem instance. As this
/// could lead to conflicts, operator types will be automatically renamed in the
/// returned instance.
///
/// Example: To load an instance of the `circt::scheduling::CyclicProblem` with
/// all its input and solution properties, call this as follows:
///
/// ```
/// loadProblem<CyclicProblem>(instOp,
///   std::make_tuple(LinkedOperatorTypeAttr(), StartTimeAttr()),
///   std::make_tuple(LatencyAttr()),
///   std::make_tuple(DistanceAttr()),
///   std::make_tuple(InitiationIntervalAttr()));
/// ```
template <typename ProblemT, typename... OperationPropertyTs,
          typename... OperatorTypePropertyTs,
          typename... ResourceTypePropertyTs, typename... DependencePropertyTs,
          typename... InstancePropertyTs>
ProblemT loadProblem(InstanceOp instOp,
                     std::tuple<OperationPropertyTs...> opProps,
                     std::tuple<OperatorTypePropertyTs...> oprProps,
                     std::tuple<ResourceTypePropertyTs...> rsrcProps,
                     std::tuple<DependencePropertyTs...> depProps,
                     std::tuple<InstancePropertyTs...> instProps) {
  ProblemT prob(instOp);

  loadInstanceProperties<ProblemT, InstancePropertyTs...>(
      prob, instOp.getSspPropertiesAttr());
  if (auto instName = instOp.getSymNameAttr())
    prob.setInstanceName(instName);

  // Use IDs to disambiguate operator types with the same name defined in
  // different libraries.
  SmallDenseMap<OperatorType, unsigned> operatorTypeIds;
  // Map `OperatorTypeOp`s to their (possibly uniqued) name in the problem
  // instance.
  SmallDenseMap<Operation *, OperatorType> operatorTypes;

  // Register all operator types in the instance's library.
  auto libraryOp = instOp.getOperatorLibrary();
  libraryOp.walk([&](OperatorTypeOp oprOp) {
    operatorTypes[oprOp] =
        loadOperatorType<ProblemT, OperatorTypePropertyTs...>(prob, oprOp,
                                                              operatorTypeIds);
  });
  if (auto libName = libraryOp.getSymNameAttr())
    prob.setLibraryName(libName);

  // Use IDs to disambiguate resource types with the same name defined in
  // different resource libraries.
  SmallDenseMap<ResourceType, unsigned> resourceTypeIds;
  // Map `ResourceTypeOp`s to their (possibly uniqued) name in the problem
  // instance.
  SmallDenseMap<Operation *, ResourceType> resourceTypes;

  // Register all resource types in the instance's resource library.
  auto rsrcLibraryOp = instOp.getResourceLibrary();
  rsrcLibraryOp.walk([&](ResourceTypeOp rsrcOp) {
    resourceTypes[rsrcOp] =
        loadResourceType<ProblemT, ResourceTypePropertyTs...>(prob, rsrcOp,
                                                              resourceTypeIds);
  });

  if (auto rsrcLibName = rsrcLibraryOp.getSymNameAttr())
    prob.setRsrcLibraryName(rsrcLibName);

  // Register all operations first, in order to retain their original order.
  auto graphOp = instOp.getDependenceGraph();
  graphOp.walk([&](OperationOp opOp) {
    prob.insertOperation(opOp);
    loadOperationProperties<ProblemT, OperationPropertyTs...>(
        prob, opOp, opOp.getSspPropertiesAttr());
    if (auto opName = opOp.getSymNameAttr())
      prob.setOperationName(opOp, opName);

    // Nothing else to check if no linked operator type is set for `opOp`,
    // because the operation doesn't carry a `LinkedOperatorTypeAttr`, or that
    // class is not part of the `OperationPropertyTs` to load.
    if (!prob.getLinkedOperatorType(opOp).has_value())
      return;

    // Otherwise, inspect the corresponding attribute to make sure the operator
    // type is available.
    SymbolRefAttr oprRef = opOp.getLinkedOperatorTypeAttr().getValue();

    Operation *oprOp;
    // 1) Look in the instance's library.
    oprOp = SymbolTable::lookupSymbolIn(libraryOp, oprRef);
    // 2) Try to resolve a nested reference to the instance's library.
    if (!oprOp)
      oprOp = SymbolTable::lookupSymbolIn(instOp, oprRef);
    // 3) Look outside of the instance.
    if (!oprOp)
      oprOp =
          SymbolTable::lookupNearestSymbolFrom(instOp->getParentOp(), oprRef);

    assert(oprOp && isa<OperatorTypeOp>(oprOp)); // checked by verifier

    // Load the operator type from `oprOp` if needed.
    auto &opr = operatorTypes[oprOp];
    if (!opr.getAttr())
      opr = loadOperatorType<ProblemT, OperatorTypePropertyTs...>(
          prob, cast<OperatorTypeOp>(oprOp), operatorTypeIds);

    // Update `opOp`'s property (may be a no-op if `opr` wasn't renamed).
    prob.setLinkedOperatorType(opOp, opr);

    // Nothing else to check if no linked resource type is set for `opOp`,
    // because the operation doesn't carry a `LinkedResourceTypeAttr`, or that
    // class is not part of the `OperationPropertyTs` to load.
    if (!prob.getLinkedResourceTypes(opOp).has_value())
      return;

    // Otherwise, inspect the corresponding attribute to make sure the resource
    // type is available.
    SmallVector<ResourceType> loadedRsrcs;
    for (auto attr : opOp.getLinkedResourceTypesAttr().getValue()) {
      SymbolRefAttr rsrcRef = dyn_cast<SymbolRefAttr>(attr);
      assert(rsrcRef &&
             "expected SymbolRefAttr inside LinkedResourceTypesAttr");

      Operation *rsrcOp;
      // 1) Look in the instance's resource library.
      rsrcOp = SymbolTable::lookupSymbolIn(rsrcLibraryOp, rsrcRef);
      // 2) Try to resolve a nested reference to the instance's resource
      // library.
      if (!rsrcOp)
        rsrcOp = SymbolTable::lookupSymbolIn(instOp, rsrcRef);
      // 3) Look outside of the instance.
      if (!rsrcOp)
        rsrcOp = SymbolTable::lookupNearestSymbolFrom(instOp->getParentOp(),
                                                      rsrcRef);

      assert(rsrcOp && isa<ResourceTypeOp>(rsrcOp)); // checked by verifier

      // Load the resource type from `rsrcOp` if needed.
      auto &rsrc = resourceTypes[rsrcOp];
      if (!rsrc.getAttr())
        rsrc = loadResourceType<ProblemT, ResourceTypePropertyTs...>(
            prob, cast<ResourceTypeOp>(rsrcOp), resourceTypeIds);

      loadedRsrcs.push_back(rsrc);
    }

    // Update `opOp`'s property (may be a no-op if `rsrc` wasn't renamed).
    prob.setLinkedResourceTypes(opOp, loadedRsrcs);
  });

  // Then walk them again, and load auxiliary dependences as well as any
  // dependence properties.
  graphOp.walk([&](OperationOp opOp) {
    ArrayAttr depsAttr = opOp.getDependencesAttr();
    if (!depsAttr)
      return;

    for (auto depAttr : depsAttr.getAsRange<DependenceAttr>()) {
      Dependence dep;
      if (FlatSymbolRefAttr sourceRef = depAttr.getSourceRef()) {
        Operation *sourceOp = SymbolTable::lookupSymbolIn(graphOp, sourceRef);
        assert(sourceOp);
        dep = Dependence(sourceOp, opOp);
        LogicalResult res = prob.insertDependence(dep);
        assert(succeeded(res));
        (void)res;
      } else
        dep = Dependence(&opOp->getOpOperand(depAttr.getOperandIdx()));

      loadDependenceProperties<ProblemT, DependencePropertyTs...>(
          prob, dep, depAttr.getProperties());
    }
  });

  return prob;
}

//===----------------------------------------------------------------------===//
// circt::scheduling::Problem (or subclasses) -> ssp.InstanceOp
//===----------------------------------------------------------------------===//

template <typename ProblemT, typename... OperationPropertyTs>
ArrayAttr saveOperationProperties(ProblemT &prob, Operation *op,
                                  ImplicitLocOpBuilder &b) {
  SmallVector<Attribute> props;
  Attribute prop;
  // Fold expression: Expands to a `getFromProblem` and a conditional
  // `push_back` call for each of the `OperationPropertyTs`.
  ((prop = OperationPropertyTs::getFromProblem(prob, op, b.getContext()),
    prop ? props.push_back(prop) : (void)prop),
   ...);
  return props.empty() ? ArrayAttr() : b.getArrayAttr(props);
}

template <typename ProblemT, typename... OperatorTypePropertyTs>
ArrayAttr saveOperatorTypeProperties(ProblemT &prob, OperatorType opr,
                                     ImplicitLocOpBuilder &b) {
  SmallVector<Attribute> props;
  Attribute prop;
  // Fold expression: Expands to a `getFromProblem` and a conditional
  // `push_back` call for each of the `OperatorTypePropertyTs`.
  ((prop = OperatorTypePropertyTs::getFromProblem(prob, opr, b.getContext()),
    prop ? props.push_back(prop) : (void)prop),
   ...);
  return props.empty() ? ArrayAttr() : b.getArrayAttr(props);
}

template <typename ProblemT, typename... ResourceTypePropertyTs>
ArrayAttr saveResourceTypeProperties(ProblemT &prob, ResourceType rsrc,
                                     ImplicitLocOpBuilder &b) {
  SmallVector<Attribute> props;
  Attribute prop;
  // Fold expression: Expands to a `getFromProblem` and a conditional
  // `push_back` call for each of the `ResourceTypePropertyTs`.
  ((prop = ResourceTypePropertyTs::getFromProblem(prob, rsrc, b.getContext()),
    prop ? props.push_back(prop) : (void)prop),
   ...);
  return props.empty() ? ArrayAttr() : b.getArrayAttr(props);
}

template <typename ProblemT, typename... DependencePropertyTs>
ArrayAttr saveDependenceProperties(ProblemT &prob, Dependence dep,
                                   ImplicitLocOpBuilder &b) {
  SmallVector<Attribute> props;
  Attribute prop;
  // Fold expression: Expands to a `getFromProblem` and a conditional
  // `push_back` call for each of the `DependencePropertyTs`.
  ((prop = DependencePropertyTs::getFromProblem(prob, dep, b.getContext()),
    prop ? props.push_back(prop) : (void)prop),
   ...);
  return props.empty() ? ArrayAttr() : b.getArrayAttr(props);
}

template <typename ProblemT, typename... InstancePropertyTs>
ArrayAttr saveInstanceProperties(ProblemT &prob, ImplicitLocOpBuilder &b) {
  SmallVector<Attribute> props;
  Attribute prop;
  // Fold expression: Expands to a `getFromProblem` and a conditional
  // `push_back` call for each of the `InstancePropertyTs`.
  ((prop = InstancePropertyTs::getFromProblem(prob, b.getContext()),
    prop ? props.push_back(prop) : (void)prop),
   ...);
  return props.empty() ? ArrayAttr() : b.getArrayAttr(props);
}

/// Construct an `InstanceOp` from a given \p ProblemT instance, and
/// create/attach attributes of the given classes for the corresponding
/// properties on the scheduling problem. The returned `InstanceOp` uses the
/// given \p instanceName and \p problemName. `OperationOp`s are created
/// unnamed, unless they represent the source operation in an auxiliary
/// dependence, or the \p operationNameFn callback returns a non-null
/// `StringAttr` with the desired name. The attribute tuples are used
/// solely for grouping/inferring the template parameter packs. The tuple
/// elements may therefore be unitialized objects. The template instantiation
/// fails if properties are incompatible with \p ProblemT.
///
/// Example: To save an instance of the `circt::scheduling::CyclicProblem` with
/// all its input and solution properties, and reyling on default operation
/// names, call this as follows:
///
/// ```
/// saveProblem<CyclicProblem>(prob,
///   std::make_tuple(LinkedOperatorTypeAttr(), StartTimeAttr()),
///   std::make_tuple(LatencyAttr()),
///   std::make_tuple(DistanceAttr()),
///   std::make_tuple(InitiationIntervalAttr()),
///   builder);
/// ```
template <typename ProblemT, typename... OperationPropertyTs,
          typename... OperatorTypePropertyTs,
          typename... ResourceTypePropertyTs, typename... DependencePropertyTs,
          typename... InstancePropertyTs>
InstanceOp
saveProblem(ProblemT &prob, std::tuple<OperationPropertyTs...> opProps,
            std::tuple<OperatorTypePropertyTs...> oprProps,
            std::tuple<ResourceTypePropertyTs...> rsrcProps,
            std::tuple<DependencePropertyTs...> depProps,
            std::tuple<InstancePropertyTs...> instProps, OpBuilder &builder) {
  ImplicitLocOpBuilder b(builder.getUnknownLoc(), builder);

  // Set up instance.
  auto instOp = b.create<InstanceOp>(
      builder.getStringAttr(ProblemT::name),
      saveInstanceProperties<ProblemT, InstancePropertyTs...>(prob, b));
  if (auto instName = prob.getInstanceName())
    instOp.setSymNameAttr(instName);

  // Emit operator types.
  b.setInsertionPointToEnd(instOp.getBodyBlock());
  auto libraryOp = b.create<OperatorLibraryOp>();
  if (auto libName = prob.getLibraryName())
    libraryOp.setSymNameAttr(libName);
  b.setInsertionPointToStart(libraryOp.getBodyBlock());

  for (auto opr : prob.getOperatorTypes())
    b.create<OperatorTypeOp>(
        opr.getAttr(),
        saveOperatorTypeProperties<ProblemT, OperatorTypePropertyTs...>(
            prob, opr, b));

  // Emit resource types.
  b.setInsertionPointToEnd(instOp.getBodyBlock());
  auto rsrcLibraryOp = b.create<ResourceLibraryOp>();
  if (auto rsrcLibName = prob.getRsrcLibraryName())
    rsrcLibraryOp.setSymNameAttr(rsrcLibName);
  b.setInsertionPointToStart(rsrcLibraryOp.getBodyBlock());

  for (auto rsrc : prob.getResourceTypes())
    b.create<ResourceTypeOp>(
        rsrc.getAttr(),
        saveResourceTypeProperties<ProblemT, ResourceTypePropertyTs...>(
            prob, rsrc, b));

  // Determine which operations act as source ops for auxiliary dependences, and
  // therefore need a name. Also, honor names provided by the client.
  DenseMap<Operation *, StringAttr> opNames;
  for (auto *op : prob.getOperations()) {
    if (auto opName = prob.getOperationName(op))
      opNames[op] = opName;

    for (auto &dep : prob.getDependences(op)) {
      Operation *src = dep.getSource();
      if (!dep.isAuxiliary() || opNames.count(src))
        continue;
      if (auto srcOpName = prob.getOperationName(src)) {
        opNames[src] = srcOpName;
        continue;
      }
      opNames[src] = b.getStringAttr(Twine("Op") + Twine(opNames.size()));
    }
  }

  // Construct operations and model their dependences.
  b.setInsertionPointToEnd(instOp.getBodyBlock());
  auto graphOp = b.create<DependenceGraphOp>();
  b.setInsertionPointToStart(graphOp.getBodyBlock());

  BackedgeBuilder backedgeBuilder(b, b.getLoc());
  ValueMapper v(&backedgeBuilder);
  for (auto *op : prob.getOperations()) {
    // Construct the `dependences attribute`. It contains `DependenceAttr` for
    // def-use deps _with_ properties, and all aux deps.
    ArrayAttr dependences;
    SmallVector<Attribute> depAttrs;
    unsigned auxOperandIdx = op->getNumOperands();
    for (auto &dep : prob.getDependences(op)) {
      ArrayAttr depProps =
          saveDependenceProperties<ProblemT, DependencePropertyTs...>(prob, dep,
                                                                      b);
      if (dep.isDefUse() && depProps) {
        auto depAttr = b.getAttr<DependenceAttr>(*dep.getDestinationIndex(),
                                                 FlatSymbolRefAttr(), depProps);
        depAttrs.push_back(depAttr);
        continue;
      }

      if (!dep.isAuxiliary())
        continue;

      auto sourceOpName = opNames.lookup(dep.getSource());
      assert(sourceOpName);
      auto sourceRef = b.getAttr<FlatSymbolRefAttr>(sourceOpName);
      auto depAttr =
          b.getAttr<DependenceAttr>(auxOperandIdx, sourceRef, depProps);
      depAttrs.push_back(depAttr);
      ++auxOperandIdx;
    }
    if (!depAttrs.empty())
      dependences = b.getArrayAttr(depAttrs);

    // Delegate to helper to construct the `properties` attribute.
    ArrayAttr properties =
        saveOperationProperties<ProblemT, OperationPropertyTs...>(prob, op, b);

    // Finally, create the `OperationOp` and inform the value mapper.
    // NB: sym_name, dependences and properties are optional attributes, so
    // passing potentially unitialized String/ArrayAttrs is intentional here.
    auto opOp =
        b.create<OperationOp>(op->getNumResults(), v.get(op->getOperands()),
                              opNames.lookup(op), dependences, properties);
    v.set(op->getResults(), opOp->getResults());
  }

  return instOp;
}

/// Dummy struct to query a problem's default properties (i.e. all input and
/// solution properties). Specializations shall provide the following
/// definitions:
///
/// ```
/// static constexpr auto operationProperties = std::make_tuple(...);
/// static constexpr auto operatorTypeProperties = std::make_tuple(...);
/// static constexpr auto dependenceProperties = std::make_tuple(...);
/// static constexpr auto instanceProperties = std::make_tuple(...);
/// ```
template <typename ProblemT>
struct Default {};

/// Construct an instance of \p ProblemT from \p instOp, and attempt to set all
/// of the problem class' properties.
///
/// Relies on the specialization of template `circt::ssp::Default` for \p
/// ProblemT.
template <typename ProblemT>
ProblemT loadProblem(InstanceOp instOp) {
  return loadProblem<ProblemT>(instOp, Default<ProblemT>::operationProperties,
                               Default<ProblemT>::operatorTypeProperties,
                               Default<ProblemT>::resourceTypeProperties,
                               Default<ProblemT>::dependenceProperties,
                               Default<ProblemT>::instanceProperties);
}

/// Construct an `InstanceOp` from a given \p ProblemT instance, and
/// create/attach attributes for all of the problem class' properties.
///
/// Relies on the specialization of template `circt::ssp::Default` for \p
/// ProblemT.
template <typename ProblemT>
InstanceOp saveProblem(ProblemT &prob, OpBuilder &builder) {
  return saveProblem<ProblemT>(prob, Default<ProblemT>::operationProperties,
                               Default<ProblemT>::operatorTypeProperties,
                               Default<ProblemT>::resourceTypeProperties,
                               Default<ProblemT>::dependenceProperties,
                               Default<ProblemT>::instanceProperties, builder);
}

//===----------------------------------------------------------------------===//
// Default property tuples for the built-in problems
//===----------------------------------------------------------------------===//

template <>
struct Default<scheduling::Problem> {
  static constexpr auto operationProperties = std::make_tuple(
      LinkedOperatorTypeAttr(), LinkedResourceTypesAttr(), StartTimeAttr());
  static constexpr auto operatorTypeProperties = std::make_tuple(LatencyAttr());
  static constexpr auto resourceTypeProperties = std::make_tuple();
  static constexpr auto dependenceProperties = std::make_tuple();
  static constexpr auto instanceProperties = std::make_tuple();
};

template <>
struct Default<scheduling::CyclicProblem> {
  static constexpr auto operationProperties =
      Default<scheduling::Problem>::operationProperties;
  static constexpr auto operatorTypeProperties =
      Default<scheduling::Problem>::operatorTypeProperties;
  static constexpr auto resourceTypeProperties =
      Default<scheduling::Problem>::resourceTypeProperties;
  static constexpr auto dependenceProperties =
      std::tuple_cat(Default<scheduling::Problem>::dependenceProperties,
                     std::make_tuple(DistanceAttr()));
  static constexpr auto instanceProperties =
      std::tuple_cat(Default<scheduling::Problem>::instanceProperties,
                     std::make_tuple(InitiationIntervalAttr()));
};

template <>
struct Default<scheduling::ChainingProblem> {
  static constexpr auto operationProperties =
      std::tuple_cat(Default<scheduling::Problem>::operationProperties,
                     std::make_tuple(StartTimeInCycleAttr()));
  static constexpr auto operatorTypeProperties =
      std::tuple_cat(Default<scheduling::Problem>::operatorTypeProperties,
                     std::make_tuple(IncomingDelayAttr(), OutgoingDelayAttr()));
  static constexpr auto resourceTypeProperties =
      Default<scheduling::Problem>::resourceTypeProperties;
  static constexpr auto dependenceProperties =
      Default<scheduling::Problem>::dependenceProperties;
  static constexpr auto instanceProperties =
      Default<scheduling::Problem>::instanceProperties;
};

template <>
struct Default<scheduling::SharedOperatorsProblem> {
  static constexpr auto operationProperties =
      Default<scheduling::Problem>::operationProperties;
  static constexpr auto operatorTypeProperties =
      Default<scheduling::Problem>::operatorTypeProperties;
  static constexpr auto resourceTypeProperties = std::make_tuple(LimitAttr());
  static constexpr auto dependenceProperties =
      Default<scheduling::Problem>::dependenceProperties;
  static constexpr auto instanceProperties =
      Default<scheduling::Problem>::instanceProperties;
};

template <>
struct Default<scheduling::ModuloProblem> {
  static constexpr auto operationProperties =
      Default<scheduling::Problem>::operationProperties;
  static constexpr auto operatorTypeProperties =
      Default<scheduling::SharedOperatorsProblem>::operatorTypeProperties;
  static constexpr auto resourceTypeProperties =
      Default<scheduling::SharedOperatorsProblem>::resourceTypeProperties;
  static constexpr auto dependenceProperties =
      Default<scheduling::CyclicProblem>::dependenceProperties;
  static constexpr auto instanceProperties =
      Default<scheduling::CyclicProblem>::instanceProperties;
};

template <>
struct Default<scheduling::ChainingCyclicProblem> {
  static constexpr auto operationProperties =
      Default<scheduling::ChainingProblem>::operationProperties;
  static constexpr auto operatorTypeProperties =
      Default<scheduling::ChainingProblem>::operatorTypeProperties;
  static constexpr auto resourceTypeProperties =
      Default<scheduling::ChainingProblem>::resourceTypeProperties;
  static constexpr auto dependenceProperties =
      Default<scheduling::CyclicProblem>::dependenceProperties;
  static constexpr auto instanceProperties =
      Default<scheduling::CyclicProblem>::instanceProperties;
};

} // namespace ssp
} // namespace circt

#endif // CIRCT_DIALECT_SSP_SSPUTILITIES_H
