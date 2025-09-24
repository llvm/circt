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
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
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
/// Track information about a domain.
struct DomainInfo {
  /// The instantiated class inside a module which contains domain information.
  ObjectOp op;

  /// The index of the created input port for this domain.  This port does not
  /// include association information and only contains the information that the
  /// user must provide.
  size_t inputPort;

  /// The index of the created output port for this domain, which includes
  /// association information.
  size_t outputPort;

  /// The associations to other ports for this domain.  This is in terms of
  /// distinct attributes that have already been established in the annotations
  /// for the associated ports.
  SmallVector<DistinctAttr> associations = SmallVector<DistinctAttr>();
};

class LowerModule {

public:
  LowerModule(FModuleLike &op,
              const DenseMap<Attribute, std::pair<ClassOp, ClassOp>> &classes)
      : op(op), eraseVector(BitVector(op.getNumPorts())), classes(classes) {}

  // Lower the associated module.  Remove domain ports and remove all domain
  // information.
  LogicalResult lowerModule();

  // Lower all instances of the associated module.  This relies on state built
  // up during `lowerModule`.
  void lowerInstances(InstanceGraph &);

private:
  FModuleLike &op;
  BitVector eraseVector;
  const DenseMap<Attribute, std::pair<ClassOp, ClassOp>> &classes;
};

LogicalResult LowerModule::lowerModule() {
  // Much of the lowering is conditioned on whether or not his module has a
  // body.  If it has a body, then we need to instantiate an object for each
  // domain port and hook up all the domain ports to annotations added to each
  // associated port.
  Block *body = nullptr;
  std::optional<OpBuilder> builder;
  if (auto moduleOp = dyn_cast<FModuleOp>(*op)) {
    body = moduleOp.getBodyBlock();
    builder = OpBuilder::atBlockBegin(body);
  }

  auto *context = op.getContext();

  // Information about a domain.  This is built up during the first iteration
  // over the ports.  This needs to preserve insertion order.
  llvm::MapVector<int, DomainInfo> domainInfo;

  // Track different "views" of the port indices.  These two variables track the
  // index of the ports after domains are deleted and after class type ports
  // have been inserted.  This is necessary due to the pass needing to track
  // information about where to insert a port (the postDeletionIndex) or which
  // port to hook up something to (the postInsertionIndex).  These are not
  // updated if there is no body.
  unsigned postDeletionIndex = 0;
  unsigned postInsertionIndex = 0;

  // The ports that should be inserted, _after deletion_.  These are indexed
  // with the postDeletionIndex.
  SmallVector<std::pair<unsigned, PortInfo>> newPorts;

  // The new port annotations.  These will be set after all deletions and
  // insertions.
  SmallVector<Attribute> portAnnotations;

  // Iterate over the ports, staging domain ports for removal and recording the
  // associations of non-domain ports.
  for (auto [index, port] : llvm::enumerate(op.getPorts())) {
    // Mark domain type ports for removal.  Add information to `domainInfo`.
    auto domain = dyn_cast<FlatSymbolRefAttr>(port.domains);
    if (domain) {
      eraseVector.set(index);

      // If this module does _not_ have a body, then we don't add ports or need
      // to populate `domainInfo`.
      if (!body)
        continue;

      // Instantiate a domain object with association information.
      auto port = cast<PortInfo>(op.getPorts()[index]);
      auto name = cast<FlatSymbolRefAttr>(port.domains);
      auto [classIn, classOut] = classes.lookup(name.getAttr());
      auto object = ObjectOp::create(
          *builder, port.loc, classOut,
          StringAttr::get(context, Twine(port.name) + "_object"));
      domainInfo[index] = {object, postInsertionIndex, postInsertionIndex + 1};

      // Erase users of the domain in the module body.
      auto arg = body->getArgument(index);
      for (auto *user : llvm::make_early_inc_range(arg.getUsers())) {
        if (auto castOp = dyn_cast<UnsafeDomainCastOp>(user)) {
          castOp.getResult().replaceAllUsesWith(castOp.getInput());
          castOp.erase();
          continue;
        }
        user->emitOpError()
            << "has an unimplemented lowering in the LowerDomains pass.";
        return failure();
      }

      // Add input and output property ports that encode the property inputs
      // (which the user must provide for the domain) and the outputs that
      // encode this information and the associations.
      newPorts.append(
          {{postDeletionIndex,
            PortInfo(port.name, classIn.getInstanceType(), Direction::In)},
           {postDeletionIndex,
            PortInfo(builder->getStringAttr(Twine(port.name) + "_out"),
                     object.getType(), Direction::Out)}});
      portAnnotations.append(
          {builder->getArrayAttr({}), builder->getArrayAttr({})});

      // Don't increment the postDeletionIndex since we deleted one port.
      // Increment the postInsertionIndex by 2 since we added two ports.
      postInsertionIndex += 2;
      continue;
    }

    // If this operation has no body, then there is no need to continue.
    if (!body)
      continue;

    // This port will not be deleted, so increment both indices.
    ++postDeletionIndex;
    ++postInsertionIndex;

    // If this port has domain associations, then add port annotation trackers.
    // These will be hooked up to the Object's associations later.
    ArrayAttr domainAttr = cast<ArrayAttr>(port.domains);
    if (!domainAttr || domainAttr.empty()) {
      portAnnotations.push_back(port.annotations.getArrayAttr());
      continue;
    }

    SmallVector<Annotation> newAnnotations;
    for (auto attr : domainAttr) {
      auto domainIndex = cast<IntegerAttr>(attr).getUInt();
      auto id = DistinctAttr::create(builder->getUnitAttr());
      newAnnotations.push_back(Annotation(DictionaryAttr::getWithSorted(
          builder->getContext(),
          {{"class", builder->getStringAttr("circt.tracker")}, {"id", id}})));
      domainInfo[domainIndex].associations.push_back(id);
    }
    if (!newAnnotations.empty())
      port.annotations.addAnnotations(newAnnotations);
    portAnnotations.push_back(port.annotations.getArrayAttr());
  }

  // Erease domain ports and clear domain association information.
  op.erasePorts(eraseVector);
  op.setDomainInfoAttr(ArrayAttr::get(context, {}));

  // Insert new property ports and hook these up to the object that was
  // instantiated earlier.
  if (body) {
    op.insertPorts(newPorts);
    for (auto const &[_, info] : domainInfo) {
      auto [object, inputPort, outputPort, associations] = info;
      builder->setInsertionPointAfter(object);
      // Assign input domain info.
      auto subDomainInfoIn = ObjectSubfieldOp::create(
          *builder, builder->getUnknownLoc(), object, 0);
      PropAssignOp::create(*builder, builder->getUnknownLoc(), subDomainInfoIn,
                           body->getArgument(inputPort));
      auto subAssociations = ObjectSubfieldOp::create(
          *builder, builder->getUnknownLoc(), object, 2);
      // Assign associations.
      SmallVector<Value> paths;
      for (auto id : associations) {
        paths.push_back(PathOp::create(
            *builder, builder->getUnknownLoc(),
            TargetKindAttr::get(context, TargetKind::MemberReference), id));
      }
      auto list = ListCreateOp::create(
          *builder, builder->getUnknownLoc(),
          ListType::get(context, cast<PropertyType>(PathType::get(context))),
          paths);
      PropAssignOp::create(*builder, builder->getUnknownLoc(), subAssociations,
                           list);
      // Connect the object to the output port.
      PropAssignOp::create(*builder, builder->getUnknownLoc(),
                           body->getArgument(outputPort), object);
    }
  }

  // Set new port annotations now that all deletions and insertions are done.
  op.setPortAnnotationsAttr(ArrayAttr::get(context, portAnnotations));

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
      : circuit(circuit), instanceGraph(instanceGraph) {}

  LogicalResult lowerDomain(DomainOp);
  LogicalResult lowerCircuit();

private:
  CircuitOp circuit;
  InstanceGraph &instanceGraph;
  DenseMap<Attribute, std::pair<ClassOp, ClassOp>> classes;
};

LogicalResult LowerCircuit::lowerDomain(DomainOp op) {
  ImplicitLocOpBuilder builder(op.getLoc(), op);
  auto *context = op.getContext();
  auto name = op.getNameAttr();
  // TODO: Update this once DomainOps have properties.
  auto classIn = ClassOp::create(builder, name, {});
  auto classInType = classIn.getInstanceType();
  auto pathListType =
      ListType::get(context, cast<PropertyType>(PathType::get(context)));
  auto classOut =
      ClassOp::create(builder, StringAttr::get(context, Twine(name) + "_out"),
                      {{/*name=*/StringAttr::get(context, "domainInfo_in"),
                        /*type=*/classInType,
                        /*dir=*/Direction::In},
                       {/*name=*/StringAttr::get(context, "domainInfo_out"),
                        /*type=*/classInType,
                        /*dir=*/Direction::Out},
                       {/*name=*/StringAttr::get(context, "associations_in"),
                        /*type=*/pathListType,
                        /*dir=*/Direction::In},
                       {/*name=*/StringAttr::get(context, "associations_out"),
                        /*type=*/pathListType,
                        /*dir=*/Direction::Out}});
  builder.setInsertionPointToStart(classOut.getBodyBlock());
  PropAssignOp::create(builder, classOut.getArgument(1),
                       classOut.getArgument(0));
  PropAssignOp::create(builder, classOut.getArgument(3),
                       classOut.getArgument(2));
  classes.insert({name, {classIn, classOut}});
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
  return instanceGraph.walkPostOrder([&](InstanceGraphNode &node) {
    auto moduleOp = dyn_cast<FModuleLike>(node.getModule<Operation *>());
    if (!moduleOp)
      return success();
    LLVM_DEBUG(llvm::dbgs() << "  - " << moduleOp.getName() << "\n");
    LowerModule lowerModule(moduleOp, classes);
    if (failed(lowerModule.lowerModule()))
      return failure();
    lowerModule.lowerInstances(instanceGraph);
    return success();
  });

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
