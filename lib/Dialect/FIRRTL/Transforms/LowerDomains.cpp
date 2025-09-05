//===- LowerDomains.cpp - Erase all FIRRTL Domain Information -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This lower all FIRRTL domain information into properties.  This is part of
// the compilation of FIRRTL domains where they are inferred and checked (See:
// the InferDomains pass) and then lowered (this pass).  This pass has the
// additional effect of removing all domain information from the circuit.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOpInterfaces.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Support/Debug.h"
#include "circt/Support/InstanceGraphInterface.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-lower-domains"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_LOWERDOMAINS
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

namespace {
class Constants {

public:
  Constants(MLIRContext *context) : context(context) {}

  ArrayRef<PortInfo> getDomainPorts();

private:
  MLIRContext *context;
  SmallVector<PortInfo, 1> domainPorts;
};

ArrayRef<PortInfo> Constants::getDomainPorts() {
  if (domainPorts.empty())
    domainPorts.append(
        {{/*name=*/
          StringAttr::get(context, "associations"),
          /*type=*/
          ListType::get(context, cast<PropertyType>(PathType::get(context))),
          /*dir=*/Direction::In}});
  return domainPorts;
}

class LowerModule {

public:
  LowerModule(FModuleLike &op, const DenseMap<Attribute, ClassOp> &classes)
      : op(op), eraseVector(BitVector(op.getNumPorts())), classes(classes) {}

  // Lower the associated module.  Remove domain ports and remove all domain
  // information.
  [[nodiscard]] LogicalResult lowerModule();

  // Lower all instances of the associated module.  This relies on state built
  // up during `lowerModule`.
  void lowerInstances(InstanceGraph &);

private:
  FModuleLike &op;
  BitVector eraseVector;
  const DenseMap<Attribute, ClassOp> &classes;
};

struct Associations {
  ObjectOp object;
  SmallVector<DistinctAttr> associations = SmallVector<DistinctAttr>();
};

LogicalResult LowerModule::lowerModule() {
  auto *body =
      llvm::TypeSwitch<FModuleLike, Block *>(op)
          .Case<FModuleOp>([](FModuleOp op) { return op.getBodyBlock(); })
          .Default([](auto op) { return nullptr; });
  if (!body) {
    op.emitOpError() << "is unsupported in LowerDomains";
    return failure();
  }
  auto builder = OpBuilder::atBlockBegin(body);
  DenseMap<int, Associations> objects;
  SmallVector<Attribute> newPortAnnotations;
  for (auto [index, port] : llvm::enumerate(op.getPorts())) {
    // Mark domain type ports for removal.
    auto domain = dyn_cast<FlatSymbolRefAttr>(port.domains);
    if (domain) {
      objects[index] = {ObjectOp::create(
          builder, port.loc, classes.lookup(domain.getAttr()), port.name)};
      eraseVector.set(index);
      newPortAnnotations.push_back(port.annotations.getArrayAttr());
      continue;
    }

    // If this port has domain associations, then add port annotation trackers.
    // These will be hooked up to the Object's associations later.
    ArrayAttr domainAttr = cast<ArrayAttr>(port.domains);
    if (!domainAttr || domainAttr.empty())
      continue;

    SmallVector<Annotation> newAnnotations;
    for (auto attr : domainAttr) {
      auto index = cast<IntegerAttr>(attr).getUInt();
      auto id = DistinctAttr::create(UnitAttr::get(op.getContext()));
      newAnnotations.push_back(Annotation(DictionaryAttr::getWithSorted(
          op.getContext(),
          {{"class", builder.getStringAttr("circt.tracker")}, {"id", id}})));
      objects[index].associations.push_back(id);
    }
    if (!newAnnotations.empty())
      port.annotations.addAnnotations(newAnnotations);
    newPortAnnotations.push_back(port.annotations.getArrayAttr());
  }
  op.setPortAnnotationsAttr(builder.getArrayAttr(newPortAnnotations));

  // Hook up all assocaitions to the created objects.
  for (size_t i = 0, e = op.getNumPorts(); i != e; ++i) {
    auto itr = objects.find(i);
    if (itr == objects.end())
      continue;
    auto assoc = itr->second;
    builder.setInsertionPointAfter(assoc.object);
    auto sub = ObjectSubfieldOp::create(builder, builder.getUnknownLoc(),
                                        assoc.object, 0);
    SmallVector<Value> paths;
    for (auto id : assoc.associations) {
      paths.push_back(PathOp::create(
          builder, builder.getUnknownLoc(),
          TargetKindAttr::get(op.getContext(), TargetKind::MemberReference),
          id));
    }
    auto list = ListCreateOp::create(
        builder, builder.getUnknownLoc(),
        ListType::get(op.getContext(),
                      cast<PropertyType>(PathType::get(op.getContext()))),
        paths);
    PropAssignOp::create(builder, builder.getUnknownLoc(), sub, list);
  }

  // Erase domain port users, domain ports, and clear the DomainInfo attr.
  if (auto moduleOp = dyn_cast<FModuleOp>(*op)) {
    for (auto idx = eraseVector.find_first(); idx >= 0;
         idx = eraseVector.find_next(idx)) {
      auto arg = moduleOp.getArgument(idx);
      for (auto *user : llvm::make_early_inc_range(arg.getUsers())) {
        if (auto castOp = dyn_cast<UnsafeDomainCastOp>(user)) {
          castOp.getResult().replaceAllUsesWith(castOp.getInput());
          castOp.erase();
          continue;
        }
        user->emitOpError() << "cannot be lowered by LowerDomains";
        return failure();
      }
    }
  }
  op.erasePorts(eraseVector);
  op.setDomainInfoAttr(ArrayAttr::get(op.getContext(), {}));

  return success();
}

void LowerModule::lowerInstances(InstanceGraph &instanceGraph) {
  auto *node = instanceGraph.lookup(cast<igraph::ModuleOpInterface>(*op));
  for (auto *use : llvm::make_early_inc_range(*node)) {
    auto instanceOp = cast<InstanceOp>(*use->getInstance());
    ImplicitLocOpBuilder builder(instanceOp.getLoc(), instanceOp);
    auto newInstanceOp = instanceOp.erasePorts(builder, eraseVector);
    instanceGraph.replaceInstance(instanceOp, newInstanceOp);
    instanceOp.erase();
  }
}

class LowerCircuit {

public:
  LowerCircuit(CircuitOp circuit, InstanceGraph &instanceGraph)
      : circuit(circuit), constants(circuit.getContext()),
        instanceGraph(instanceGraph) {}

  [[nodiscard]] LogicalResult lowerDomain(DomainOp);
  [[nodiscard]] LogicalResult lowerCircuit();

private:
  CircuitOp circuit;
  Constants constants;
  InstanceGraph &instanceGraph;
  DenseMap<Attribute, ClassOp> classes;
};

LogicalResult LowerCircuit::lowerDomain(DomainOp op) {
  ImplicitLocOpBuilder builder(op.getLoc(), op);
  auto name = op.getNameAttr();
  classes.insert(
      {name, ClassOp::create(builder, name, constants.getDomainPorts())});

  op.erase();
  return success();
}

LogicalResult LowerCircuit::lowerCircuit() {
  LLVM_DEBUG(llvm::dbgs() << "Processing domains:\n");
  for (auto domain : llvm::make_early_inc_range(circuit.getOps<DomainOp>())) {
    LLVM_DEBUG(llvm::dbgs() << "  - " << domain.getName() << "\n");
    if (failed(lowerDomain(domain)))
      return failure();
  }

  LLVM_DEBUG(llvm::dbgs() << "Processing modules:\n");
  DenseSet<InstanceGraphNode *> visited;
  for (auto *root : instanceGraph) {
    for (auto *node : llvm::post_order_ext(root, visited)) {
      auto moduleOp = dyn_cast<FModuleLike>(node->getModule<Operation *>());
      if (!moduleOp)
        continue;
      LLVM_DEBUG(llvm::dbgs() << "  - " << moduleOp.getName() << "\n");
      LowerModule lowerModule(moduleOp, classes);
      if (failed(lowerModule.lowerModule()))
        return failure();
      lowerModule.lowerInstances(instanceGraph);
    }
  }

  return success();
}
} // namespace

class LowerDomainsPass : public impl::LowerDomainsBase<LowerDomainsPass> {
  void runOnOperation() override;
};

void LowerDomainsPass::runOnOperation() {
  LLVM_DEBUG(debugPassHeader(this) << "\n";);

  LowerCircuit lowerCircuit(getOperation(), getAnalysis<InstanceGraph>());
  if (failed(lowerCircuit.lowerCircuit()))
    return signalPassFailure();

  LLVM_DEBUG(debugFooter() << "\n";);
}
