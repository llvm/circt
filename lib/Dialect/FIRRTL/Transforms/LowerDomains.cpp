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
// This pass needs to run after InferDomains and before LowerClasses.  This pass
// assumes that all domain information is available.  It is not written in such
// a way that partial domain information can be lowered incrementally, e.g.,
// interleaving InferDomains and LowerDomains with passes that incrementally add
// domain information will not work.  This is because LowerDomains is closer to
// a conversion than a pass.  It is expected that this is part of the FIRRTL to
// HW pass pipeline.
//
// There are a number of limitations in this pass presently, much of which are
// coupled to the representation of domains.  Currently, domain information on
// ports marks a port as being either a domain or having domain association
// information, but not both.  This precludes having aggregates that contain
// domain types.  (Or: this pass assumes that a pass like LowerOpenAggs has run
// to do this splitting.)  There are no requirements that LowerTypes has run,
// assuming this post-LowerOpenAggs representation.
//
// As the representation of domains changes to allow for associations on fields
// and domain types to be part of aggregates, this pass will require updates.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Support/Debug.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Mutex.h"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_LOWERDOMAINS
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

class LowerDomainsPass : public impl::LowerDomainsBase<LowerDomainsPass> {
  using Base::Base;
  void runOnOperation() override;
};

#define DEBUG_TYPE                                                             \
  impl::LowerDomainsBase<LowerDomainsPass>::getArgumentName().data()

namespace {
/// Minimally track information about an association of a port to a domain.
struct AssociationInfo {
  /// The DistinctAttr (annotation) that is used to identify the port.
  DistinctAttr distinctAttr;

  /// The port's location.  This is used to generate exact information about
  /// certain property ops created later.
  Location loc;
};

/// Track information about the lowering of a domain port.
struct DomainInfo {
  /// An instance of an object which will be used to track an instance of the
  /// domain-lowered class (which is the identity of the domain) and all its
  /// associations.
  ObjectOp op;

  /// The index of the optional input port that will be hooked up to a field of
  /// the ObjectOp.  This port is an instance of the domain-lowered class.  If
  /// this is created due to an output domain port, then this is nullopt.
  std::optional<unsigned> inputPort;

  /// The index of the output port that the ObjectOp will be connected to.  This
  /// port communicates back to the user information about the associations.
  unsigned outputPort;

  /// A vector of minimal association info that will be hooked up to the
  /// associations of this ObjectOp.
  SmallVector<AssociationInfo> associations{};
};

/// Struct of the two classes created from a domain, an input class (which is
/// one-to-one with the domain) and an output class (which tracks the input
/// class and any associations).
struct Classes {
  /// The domain-lowered class.
  ClassOp input;

  /// A class tracking an instance of the input class and a list of
  /// associations.
  ClassOp output;
};

/// Thread safe, lazy pool of constant attributes
class Constants {

public:
  Constants(MLIRContext *context) : context(context) {}

  // Return an empty ArrayAttr.
  ArrayAttr getEmptyArrayAttr() {
    if (!emptyArrayAttr) {
      llvm::sys::SmartScopedLock<true> lock(mutex);
      emptyArrayAttr = ArrayAttr::get(context, {});
    }
    return emptyArrayAttr;
  }

private:
  /// Construct all the field info attributes.
  void constructFieldNameAttrs() {
    if (domainInfoIn)
      return;
    llvm::sys::SmartScopedLock<true> lock(mutex);
    domainInfoIn = StringAttr::get(context, "domainInfo_in");
    domainInfoOut = StringAttr::get(context, "domainInfo_out");
    associationsIn = StringAttr::get(context, "associations_in");
    associationsOut = StringAttr::get(context, "associations_out");
  }

public:
  StringAttr getDomainInfoIn() {
    constructFieldNameAttrs();
    assert(domainInfoIn);
    return domainInfoIn;
  }

  StringAttr getDomainInfoOut() {
    constructFieldNameAttrs();
    assert(domainInfoOut);
    return domainInfoOut;
  }

  StringAttr getAssociationsIn() {
    constructFieldNameAttrs();
    assert(associationsIn);
    return associationsIn;
  }

  StringAttr getAssociationsOut() {
    constructFieldNameAttrs();
    assert(associationsOut);
    return associationsOut;
  }

private:
  /// Mutex indicating that, when held, allows for mutation.
  llvm::sys::SmartMutex<true> mutex;

  /// An MLIR context necessary for creating new attributes.
  MLIRContext *context;

  /// Lazily constructed attributes
  ArrayAttr emptyArrayAttr;
  StringAttr domainInfoIn;
  StringAttr domainInfoOut;
  StringAttr associationsIn;
  StringAttr associationsOut;
};

class LowerModule {

public:
  LowerModule(FModuleLike &op, const DenseMap<Attribute, Classes> &classes,
              Constants &constants)
      : op(op), eraseVector(op.getNumPorts()), domainToClasses(classes),
        constants(constants) {}

  // Lower the associated module.  Replace domain ports with input/ouput class
  // ports.
  LogicalResult lowerModule();

  // Lower all instances of the associated module.  This relies on state built
  // up during `lowerModule` and must be run _afterwards_.
  LogicalResult lowerInstances(InstanceGraph &);

private:
  /// Erase all users of domain type ports.
  LogicalResult eraseDomainUsers(Value value) {
    for (auto *user : llvm::make_early_inc_range(value.getUsers())) {
      if (auto castOp = dyn_cast<UnsafeDomainCastOp>(user)) {
        castOp.getResult().replaceAllUsesWith(castOp.getInput());
        castOp.erase();
        continue;
      }
      return user->emitOpError()
             << "has an unimplemented lowering in the LowerDomains pass.";
    }
    return success();
  }

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

  // Lazy constant pool
  Constants &constants;
};

LogicalResult LowerModule::lowerModule() {
  // Much of the lowering is conditioned on whether or not this module has a
  // body.  If it has a body, then we need to instantiate an object for each
  // domain port and hook up all the domain ports to annotations added to each
  // associated port.  Skip modules which don't have domains.
  auto shouldProcess =
      TypeSwitch<Operation *, std::optional<Block *>>(op)
          .Case<FModuleOp>([](auto op) { return op.getBodyBlock(); })
          .Case<FExtModuleOp>([](auto) { return nullptr; })
          // Skip all other modules.
          .Default([](auto) { return std::nullopt; });
  if (!shouldProcess)
    return success();
  Block *body = *shouldProcess;

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
        auto builder = ImplicitLocOpBuilder::atBlockEnd(port.loc, body);
        auto object = ObjectOp::create(
            builder, classOut,
            StringAttr::get(context, Twine(port.name) + "_object"));
        if (port.direction == Direction::In)
          indexToDomain[i] = {object, iIns, iIns + 1};
        else
          indexToDomain[i] = {object, std::nullopt, iIns};

        // Erase users of the domain in the module body.
        if (failed(eraseDomainUsers(body->getArgument(i))))
          return failure();
      }

      // Add input and output property ports that encode the property inputs
      // (which the user must provide for the domain) and the outputs that
      // encode this information and the associations.
      if (port.direction == Direction::In) {
        newPorts.push_back({iDel, PortInfo(port.name, classIn.getInstanceType(),
                                           Direction::In)});
        portAnnotations.push_back(constants.getEmptyArrayAttr());
        ++iIns;
      }
      newPorts.push_back(
          {iDel, PortInfo(StringAttr::get(context, Twine(port.name) + "_out"),
                          classOut.getInstanceType(), Direction::Out)});
      portAnnotations.push_back(constants.getEmptyArrayAttr());
      ++iIns;

      // Don't increment the iDel since we deleted one port.
      continue;
    }

    // Record the mapping of the old port to the new port.  This can be used
    // later to update instances.  This port will not be deleted, so
    // post-increment both indices.
    resultMap.emplace_back(iDel++, iIns++);

    // If this port has domain associations, then we need to add port annotation
    // trackers.  These will be hooked up to the Object's associations later.
    // However, if there is no domain information, then annotations do not need
    // to be modified.  Early continue first, adding trackers otherwise.  Only
    // create one tracker for all associations.
    ArrayAttr domainAttr = cast<ArrayAttr>(port.domains);
    if (!domainAttr || domainAttr.empty()) {
      portAnnotations.push_back(port.annotations.getArrayAttr());
      continue;
    }

    SmallVector<Annotation> newAnnotations;
    DistinctAttr id;
    for (auto indexAttr : domainAttr.getAsRange<IntegerAttr>()) {
      if (!id) {
        id = DistinctAttr::create(UnitAttr::get(context));
        newAnnotations.push_back(Annotation(DictionaryAttr::getWithSorted(
            context, {{"class", StringAttr::get(context, "circt.tracker")},
                      {"id", id}})));
      }
      indexToDomain[indexAttr.getUInt()].associations.push_back({id, port.loc});
    }
    if (!newAnnotations.empty())
      port.annotations.addAnnotations(newAnnotations);
    portAnnotations.push_back(port.annotations.getArrayAttr());
  }

  // Erase domain ports and clear domain association information.
  op.erasePorts(eraseVector);
  op.setDomainInfoAttr(constants.getEmptyArrayAttr());

  // Insert new property ports and hook these up to the object that was
  // instantiated earlier.
  op.insertPorts(newPorts);

  if (body) {
    for (auto const &[_, info] : indexToDomain) {
      auto [object, inputPort, outputPort, associations] = info;
      OpBuilder builder(object);
      builder.setInsertionPointAfter(object);
      // Assign input domain info if needed.
      //
      // TODO: Change this to hook up to its connection once domain connects are
      // available.
      if (inputPort) {
        auto subDomainInfoIn =
            ObjectSubfieldOp::create(builder, object.getLoc(), object, 0);
        PropAssignOp::create(builder, object.getLoc(), subDomainInfoIn,
                             body->getArgument(*inputPort));
      }
      auto subAssociations =
          ObjectSubfieldOp::create(builder, object.getLoc(), object, 2);
      // Assign associations.
      SmallVector<Value> paths;
      for (auto [id, loc] : associations) {
        paths.push_back(PathOp::create(
            builder, loc, TargetKindAttr::get(context, TargetKind::Reference),
            id));
      }
      auto list = ListCreateOp::create(
          builder, object.getLoc(),
          ListType::get(context, cast<PropertyType>(PathType::get(context))),
          paths);
      PropAssignOp::create(builder, object.getLoc(), subAssociations, list);
      // Connect the object to the output port.
      PropAssignOp::create(builder, object.getLoc(),
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
    auto instanceOp = dyn_cast<InstanceOp>(*use->getInstance());
    if (!instanceOp) {
      use->getInstance().emitOpError()
          << "has an unimplemented lowering in LowerDomains";
      return failure();
    }
    LLVM_DEBUG(llvm::dbgs()
               << "      - " << instanceOp.getInstanceName() << "\n");

    for (auto bit : eraseVector.set_bits())
      if (failed(eraseDomainUsers(instanceOp.getResult(bit))))
        return failure();

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
  LowerCircuit(CircuitOp circuit, InstanceGraph &instanceGraph,
               llvm::Statistic &numDomains)
      : circuit(circuit), instanceGraph(instanceGraph),
        constants(circuit.getContext()), numDomains(numDomains) {}

  LogicalResult lowerDomain(DomainOp);
  LogicalResult lowerCircuit();

private:
  CircuitOp circuit;
  InstanceGraph &instanceGraph;
  Constants constants;
  llvm::Statistic &numDomains;
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
                      {{/*name=*/constants.getDomainInfoIn(),
                        /*type=*/classInType,
                        /*dir=*/Direction::In},
                       {/*name=*/constants.getDomainInfoOut(),
                        /*type=*/classInType,
                        /*dir=*/Direction::Out},
                       {/*name=*/constants.getAssociationsIn(),
                        /*type=*/pathListType,
                        /*dir=*/Direction::In},
                       {/*name=*/constants.getAssociationsOut(),
                        /*type=*/pathListType,
                        /*dir=*/Direction::Out}});
  builder.setInsertionPointToStart(classOut.getBodyBlock());
  PropAssignOp::create(builder, classOut.getArgument(1),
                       classOut.getArgument(0));
  PropAssignOp::create(builder, classOut.getArgument(3),
                       classOut.getArgument(2));
  classes.insert({name, {classIn, classOut}});
  op.erase();
  ++numDomains;
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
    LowerModule lowerModule(moduleOp, classes, constants);
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

  LowerCircuit lowerCircuit(getOperation(), getAnalysis<InstanceGraph>(),
                            numDomains);
  if (failed(lowerCircuit.lowerCircuit()))
    return signalPassFailure();

  markAllAnalysesPreserved();
}
