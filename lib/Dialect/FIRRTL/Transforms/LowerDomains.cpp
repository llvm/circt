//===- LowerDomains.cpp - Lower domain information to properties ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers all FIRRTL domain information into classes, objects, and
// properties.  This is part of the compilation of FIRRTL domains where they are
// inferred and checked (See: the InferDomains pass) and then lowered (this
// pass).  After this pass runs, all domain information has been removed from
// its original representation.
//
// Each domain is lowered into two classes: (1) a class that has the exact same
// input/output properties as its corresponding domain and (2) a class that is
// used to track the associations of the domain.  Every input domain port is
// lowered to an input of type (1) and an output of type (2).  Every output
// domain port is lowered to an output of type (2).
//
// Intuitively, (1) is the information that a user must specify about a domain
// and (2) is the associations for that domain.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Support/Debug.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_LOWERDOMAINS
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

class LowerDomainsPass : public impl::LowerDomainsBase<LowerDomainsPass> {
  void runOnOperation() override;
};

#define DEBUG_TYPE                                                             \
  impl::LowerDomainsBase<LowerDomainsPass>::getArgumentName().data()

namespace {
/// Track information about the lowering of a domain port.
struct DomainInfo {
  /// An instance of an object which will be used to track an instance of the
  /// domain-lowered class (which is the identity of the domain) and all its
  /// associations.
  ObjectOp op;

  /// The index of the input port that will be hooked up to a field of the
  /// ObjectOp.  This port is an instance of the domain-lowered class.
  unsigned inputPort;

  /// The index of the output port that the ObjectOp will be connected to.  This
  /// port communicates back to the user information about the associations.
  unsigned outputPort;

  /// A vector of associations that will be hooked up to the associations of
  /// this ObjectOp.
  SmallVector<DistinctAttr> associations{};
};

/// Store of the two classes created from a domain, an input class (which is
/// one-to-one with the domain) and an output class (which tracks the input
/// class and any associations).
struct Classes {

  /// The domain-lowered class.
  ClassOp input;

  /// A class tracking an instance of the input class and a list of
  /// associations.
  ClassOp output;
};

class LowerModule {

public:
  LowerModule(FModuleLike &op, const DenseMap<Attribute, Classes> &classes)
      : op(op), eraseVector(op.getNumPorts()), domainToClasses(classes) {}

  // Lower the associated module.  Replace domain ports with input/ouput class
  // ports.
  LogicalResult lowerModule();

  // Lower all instances of the associated module.  This relies on state built
  // up during `lowerModule` and must be run _afterwards_.
  LogicalResult lowerInstances(InstanceGraph &);

private:
  // The module this class is lowering
  FModuleLike &op;

  // Ports that should be erased
  BitVector eraseVector;

  // The ports that should be inserted, _after deletion_ by application of
  // `eraseVector`.
  SmallVector<std::pair<unsigned, PortInfo>> newPorts;

  // A mapping of old result to new result
  SmallVector<std::pair<unsigned, unsigned>> resultMap;

  // Mapping of domain name to the lowered input and output class
  const DenseMap<Attribute, Classes> &domainToClasses;
};

LogicalResult LowerModule::lowerModule() {
  // Only lower modules or external modules!
  //
  // Much of the lowering is conditioned on whether or not this module has a
  // body.  If it has a body, then we need to instantiate an object for each
  // domain port and hook up all the domain ports to annotations added to each
  // associated port.
  Block *body = nullptr;
  if (auto moduleOp = dyn_cast<FModuleOp>(*op)) {
    body = moduleOp.getBodyBlock();
  } else if (!isa<FExtModuleOp>(op)) {
    return success();
  }

  auto *context = op.getContext();

  // Information about a domain.  This is built up during the first iteration
  // over the ports.  This needs to preserve insertion order.
  llvm::MapVector<unsigned, DomainInfo> indexToDomain;

  // The new port annotations.  These will be set after all deletions and
  // insertions.
  SmallVector<Attribute> portAnnotations;

  // Iterate over the ports, staging domain ports for removal and recording the
  // associations of non-domain ports.  After this, domain ports will be deleted
  // and then class ports will be inserted.  This loop therefore needs to track
  // three indices:
  //   1. i tracks the original port index.
  //   2. iDel tracks the port index after deletion.
  //   3. iIns tracks the port index after insertion.
  auto ports = op.getPorts();
  for (unsigned i = 0, iDel = 0, iIns = 0, e = op.getNumPorts(); i != e; ++i) {
    auto port = cast<PortInfo>(ports[i]);
    // Mark domain type ports for removal.  Add information to `domainInfo`.
    auto domain = dyn_cast<FlatSymbolRefAttr>(port.domains);
    if (domain) {
      eraseVector.set(i);

      // Instantiate a domain object with association information.
      auto [classIn, classOut] = domainToClasses.at(domain.getAttr());

      if (body) {
        auto builder = OpBuilder::atBlockBegin(body);
        auto object = ObjectOp::create(
            builder, port.loc, classOut,
            StringAttr::get(context, Twine(port.name) + "_object"));
        indexToDomain[i] = {object, iIns, iIns + 1};

        // Erase users of the domain in the module body.
        auto arg = body->getArgument(i);
        for (auto *user : llvm::make_early_inc_range(arg.getUsers())) {
          if (auto castOp = dyn_cast<UnsafeDomainCastOp>(user)) {
            castOp.getResult().replaceAllUsesWith(castOp.getInput());
            castOp.erase();
            continue;
          }
          return user->emitOpError()
                 << "has an unimplemented lowering in the LowerDomains pass.";
        }
      }

      // Add input and output property ports that encode the property inputs
      // (which the user must provide for the domain) and the outputs that
      // encode this information and the associations.
      newPorts.append(
          {{iDel,
            PortInfo(port.name, classIn.getInstanceType(), Direction::In)},
           {iDel, PortInfo(StringAttr::get(context, Twine(port.name) + "_out"),
                           classOut.getInstanceType(), Direction::Out)}});
      portAnnotations.append(
          {ArrayAttr::get(context, {}), ArrayAttr::get(context, {})});

      // Don't increment the iDel since we deleted one port.
      // Increment the iIns by 2 since we added two ports.
      iIns += 2;
      continue;
    }

    // Record the mapping of the old port to the new port.  This can be used
    // later to update instances.  This port will not be deleted, so
    // post-increment both indices.
    resultMap.emplace_back(iDel++, iIns++);

    // If this port has domain associations, then we need to add port annotation
    // trackers.  These will be hooked up to the Object's associations later.
    // However, if there is no domain information, then annotations do not need
    // to be modified.  Early continue first, adding trackers otherwise.
    ArrayAttr domainAttr = cast<ArrayAttr>(port.domains);
    if (!domainAttr || domainAttr.empty()) {
      portAnnotations.push_back(port.annotations.getArrayAttr());
      continue;
    }

    SmallVector<Annotation> newAnnotations;
    for (auto indexAttr : domainAttr.getAsRange<IntegerAttr>()) {
      auto domainIndex = indexAttr.getUInt();
      auto id = DistinctAttr::create(UnitAttr::get(context));
      newAnnotations.push_back(Annotation(DictionaryAttr::getWithSorted(
          context,
          {{"class", StringAttr::get(context, "circt.tracker")}, {"id", id}})));
      indexToDomain[domainIndex].associations.push_back(id);
    }
    if (!newAnnotations.empty())
      port.annotations.addAnnotations(newAnnotations);
    portAnnotations.push_back(port.annotations.getArrayAttr());
  }

  // Erase domain ports and clear domain association information.
  op.erasePorts(eraseVector);
  op.setDomainInfoAttr(ArrayAttr::get(context, {}));

  // Insert new property ports and hook these up to the object that was
  // instantiated earlier.
  op.insertPorts(newPorts);
  if (body) {
    for (auto const &[_, info] : indexToDomain) {
      auto [object, inputPort, outputPort, associations] = info;
      OpBuilder builder(object);
      builder.setInsertionPointAfter(object);
      // Assign input domain info.
      auto subDomainInfoIn =
          ObjectSubfieldOp::create(builder, builder.getUnknownLoc(), object, 0);
      PropAssignOp::create(builder, builder.getUnknownLoc(), subDomainInfoIn,
                           body->getArgument(inputPort));
      auto subAssociations =
          ObjectSubfieldOp::create(builder, builder.getUnknownLoc(), object, 2);
      // Assign associations.
      SmallVector<Value> paths;
      for (auto id : associations) {
        paths.push_back(PathOp::create(
            builder, builder.getUnknownLoc(),
            TargetKindAttr::get(context, TargetKind::MemberReference), id));
      }
      auto list = ListCreateOp::create(
          builder, builder.getUnknownLoc(),
          ListType::get(context, cast<PropertyType>(PathType::get(context))),
          paths);
      PropAssignOp::create(builder, builder.getUnknownLoc(), subAssociations,
                           list);
      // Connect the object to the output port.
      PropAssignOp::create(builder, builder.getUnknownLoc(),
                           body->getArgument(outputPort), object);
    }
  }

  // Set new port annotations.
  op.setPortAnnotationsAttr(ArrayAttr::get(context, portAnnotations));

  LLVM_DEBUG({
    llvm::dbgs() << "    portMap:\n";
    for (auto [oldIndex, newIndex] : resultMap)
      llvm::dbgs() << "      - " << oldIndex << ": " << newIndex << "\n";
  });

  return success();
}

LogicalResult LowerModule::lowerInstances(InstanceGraph &instanceGraph) {
  auto *node = instanceGraph.lookup(cast<igraph::ModuleOpInterface>(*op));
  for (auto *use : llvm::make_early_inc_range(node->uses())) {
    auto instanceOp = cast<InstanceOp>(*use->getInstance());
    LLVM_DEBUG(llvm::dbgs()
               << "      - " << instanceOp.getInstanceName() << "\n");

    for (auto bit : eraseVector.set_bits()) {
      auto result = instanceOp.getResult(bit);
      for (auto *user : llvm::make_early_inc_range(result.getUsers())) {
        if (auto castOp = dyn_cast<UnsafeDomainCastOp>(user)) {
          castOp.getResult().replaceAllUsesWith(castOp.getInput());
          castOp.erase();
          continue;
        }
        return user->emitOpError()
               << "has an unimplemented lowering in the LowerDomains pass.";
      }
    }

    ImplicitLocOpBuilder builder(instanceOp.getLoc(), instanceOp);
    auto erased = instanceOp.erasePorts(builder, eraseVector);
    auto inserted = erased.cloneAndInsertPorts(newPorts);
    for (auto [oldIndex, newIndex] : resultMap) {
      auto oldPort = erased.getResult(oldIndex);
      auto newPort = inserted.getResult(newIndex);
      oldPort.replaceAllUsesWith(newPort);
    }
    instanceGraph.replaceInstance(instanceOp, inserted);

    instanceOp.erase();
    erased.erase();
  }

  return success();
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
  DenseMap<Attribute, Classes> classes;
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
    LLVM_DEBUG(llvm::dbgs() << "  - module: " << moduleOp.getName() << "\n");
    LowerModule lowerModule(moduleOp, classes);
    if (failed(lowerModule.lowerModule()))
      return failure();
    LLVM_DEBUG(llvm::dbgs() << "    instances:\n");
    return lowerModule.lowerInstances(instanceGraph);
  });

  return success();
}
} // namespace

void LowerDomainsPass::runOnOperation() {
  CIRCT_DEBUG_SCOPED_PASS_LOGGER(this);

  LowerCircuit lowerCircuit(getOperation(), getAnalysis<InstanceGraph>());
  if (failed(lowerCircuit.lowerCircuit()))
    return signalPassFailure();
}
