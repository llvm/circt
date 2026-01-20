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
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Support/Debug.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Threading.h"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_LOWERDOMAINS
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;
using mlir::UnrealizedConversionCastOp;

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

  /// A conversion cast that is used to temporarily replace the port while it is
  /// being deleted.  If the port has no uses, then this will be empty.  Note:
  /// this is used as a true temporary and may be overwritten.  During
  /// `lowerModules()` this is used to store a module port temporary.  During
  /// `lowerInstances()`, this stores an instance port temporary.
  UnrealizedConversionCastOp temp;

  /// A vector of minimal association info that will be hooked up to the
  /// associations of this ObjectOp.
  SmallVector<AssociationInfo> associations{};

  /// Return a DomainInfo for an input domain port.  This will have both an
  /// input port at the current index and an output port at the next index.
  /// Other members are default-initialized and will be set later.
  static DomainInfo input(unsigned portIndex) {
    return DomainInfo({{}, portIndex, portIndex + 1, {}, {}});
  }

  /// Return a DomainInfo for an output domain port.  This creates only an
  /// output port at the current index.  Other members are default-initialized
  /// and will be set later.
  static DomainInfo output(unsigned portIndex) {
    return DomainInfo({{}, std::nullopt, portIndex, {}, {}});
  }
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

  /// Lazily constructed empty array attribute.
  struct EmptyArray {
    llvm::once_flag flag;
    ArrayAttr attr;
  };

  /// Lazy constructed attributes necessary for building an output class.
  struct ClassOut {
    llvm::once_flag flag;
    StringAttr domainInfoIn;
    StringAttr domainInfoOut;
    StringAttr associationsIn;
    StringAttr associationsOut;
  };

public:
  Constants(MLIRContext *context) : context(context) {}

  /// Return an empty ArrayAttr.
  ArrayAttr getEmptyArrayAttr() {
    llvm::call_once(emptyArray.flag,
                    [&] { emptyArray.attr = ArrayAttr::get(context, {}); });
    return emptyArray.attr;
  }

private:
  /// Construct all the field info attributes.
  void initClassOut() {
    llvm::call_once(classOut.flag, [&] {
      classOut.domainInfoIn = StringAttr::get(context, "domainInfo_in");
      classOut.domainInfoOut = StringAttr::get(context, "domainInfo_out");
      classOut.associationsIn = StringAttr::get(context, "associations_in");
      classOut.associationsOut = StringAttr::get(context, "associations_out");
    });
  }

public:
  /// Return a "domainInfo_in" attr.
  StringAttr getDomainInfoIn() {
    initClassOut();
    return classOut.domainInfoIn;
  }

  /// Return a "domainInfo_out" attr.
  StringAttr getDomainInfoOut() {
    initClassOut();
    return classOut.domainInfoOut;
  }

  /// Return an "associations_in" attr.
  StringAttr getAssociationsIn() {
    initClassOut();
    return classOut.associationsIn;
  }

  /// Return an "associations_out" attr.
  StringAttr getAssociationsOut() {
    initClassOut();
    return classOut.associationsOut;
  }

private:
  /// An MLIR context necessary for creating new attributes.
  MLIRContext *context;

  /// Lazily constructed attributes
  EmptyArray emptyArray;
  ClassOut classOut;
};

/// Replace the value with a temporary 0-1 unrealized conversion.  Return the
/// single conversion operation.  This is used to "stub out" the users of a
/// module or instance port while it is being converted from a domain port to a
/// class port.  This function lets us erase the port.  This function is
/// intended to be used with `splice` to replace the 0-1 conversion with a 1-1
/// conversion once the new port, of the new type, is available.
static UnrealizedConversionCastOp stubOut(Value value) {
  if (!value.hasNUsesOrMore(1))
    return {};

  OpBuilder builder(value.getContext());
  builder.setInsertionPointAfterValue(value);
  auto temp = UnrealizedConversionCastOp::create(
      builder, builder.getUnknownLoc(), {value.getType()}, {});
  value.replaceAllUsesWith(temp.getResult(0));
  return temp;
}

/// Replace a temporary 0-1 conversion cast with a 1-1 conversion cast.  Erase
/// the old conversion.  Return the new conversion.  This function is used to
/// hook up a module or instance port to an operation user of the old type.
/// After all ports have been spliced, the splice-operation-splice can be
/// replaced with a new operation that handles the new port types.
///
/// Assumptions:
///
///   1. `temp` is either null or a 0-1 conversion cast.
///   2. If `temp` is non-null, then `value` is non-null.
static UnrealizedConversionCastOp splice(UnrealizedConversionCastOp temp,
                                         Value value) {
  if (!temp)
    return {};

  // This must be a 0-1 unrealized conversion.  Anything else is unexpected.
  assert(temp && temp.getNumResults() == 1 && temp.getNumOperands() == 0);

  // Value must be non-null.
  assert(value);

  auto oldValue = temp.getResult(0);

  OpBuilder builder(temp);
  auto splice = UnrealizedConversionCastOp::create(
      builder, builder.getUnknownLoc(), {oldValue.getType()}, {value});
  oldValue.replaceAllUsesWith(splice.getResult(0));
  temp.erase();
  return splice;
}

/// Class that is used to lower a module that may contain domain ports.  This is
/// intended to be used by calling `lowerModule()` to lower the module.  This
/// builds up state in the class which is then used when calling
/// `lowerInstances()` to update all instantiations of this module.  If
/// multi-threading, care needs to be taken to only call `lowerInstances()`
/// when instantiating modules are not _also_ being updated.
class LowerModule {

public:
  LowerModule(FModuleLike op, const DenseMap<Attribute, Classes> &classes,
              Constants &constants, InstanceGraph &instanceGraph)
      : op(op), eraseVector(op.getNumPorts()), domainToClasses(classes),
        constants(constants), instanceGraph(instanceGraph) {}

  /// Lower the associated module.  Replace domain ports with input/output class
  /// ports.
  LogicalResult lowerModule();

  /// Lower all instances of the associated module.  This relies on state built
  /// up during `lowerModule` and must be run _afterwards_.
  LogicalResult lowerInstances();

private:
  /// The module this class is lowering
  FModuleLike op;

  /// Ports that should be erased
  BitVector eraseVector;

  /// The ports that should be inserted, _after deletion_ by application of
  /// `eraseVector`.
  SmallVector<std::pair<unsigned, PortInfo>> newPorts;

  /// A mapping of old result to new result
  SmallVector<std::pair<unsigned, unsigned>> resultMap;

  /// Mapping of domain name to the lowered input and output class
  const DenseMap<Attribute, Classes> &domainToClasses;

  /// Lazy constant pool
  Constants &constants;

  /// Reference to an instance graph.  This _will_ be mutated.
  ///
  /// TODO: The mutation of this is _not_ thread safe.  This needs to be fixed
  /// if this pass is parallelized.
  InstanceGraph &instanceGraph;

  // Information about a domain.  This is built up during the first iteration
  // over the ports.  This needs to preserve insertion order.
  llvm::MapVector<unsigned, DomainInfo> indexToDomain;
};

LogicalResult LowerModule::lowerModule() {
  // TOOD: Is there an early exit condition here?  It is not as simple as
  // checking for domain ports as the module may have no domain ports, but have
  // been modified by an earlier `lowerInstances()` call.

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
  OpBuilder::InsertPoint insertPoint;
  if (body)
    insertPoint = {body, body->begin()};
  auto ports = op.getPorts();
  for (unsigned i = 0, iDel = 0, iIns = 0, e = op.getNumPorts(); i != e; ++i) {
    auto port = cast<PortInfo>(ports[i]);

    // Mark domain type ports for removal.  Add information to `domainInfo`.
    if (auto domain = dyn_cast_or_null<FlatSymbolRefAttr>(port.domains)) {
      eraseVector.set(i);

      // Instantiate a domain object with association information.
      auto [classIn, classOut] = domainToClasses.at(domain.getAttr());

      indexToDomain[i] = port.direction == Direction::In
                             ? DomainInfo::input(iIns)
                             : DomainInfo::output(iIns);

      if (body) {
        // Insert objects in-order at the top of the module's body.  These
        // cannot be inserted at the end as they may have users.
        ImplicitLocOpBuilder builder(port.loc, context);
        builder.restoreInsertionPoint(insertPoint);

        // Create the object, add information about it to domain info.
        auto object = ObjectOp::create(
            builder, classOut,
            StringAttr::get(context, Twine(port.name) + "_object"));
        instanceGraph.lookup(op)->addInstance(object,
                                              instanceGraph.lookup(classOut));
        indexToDomain[i].op = object;
        indexToDomain[i].temp = stubOut(body->getArgument(i));

        // Save the insertion point for the next go-around.  This allows the
        // objects to be inserted in port order as opposed to reverse port
        // order.
        insertPoint = builder.saveInsertionPoint();
      }

      // Add input and output property ports that encode the property inputs
      // (which the user must provide for the domain) and the outputs that
      // encode this information and the associations.  Keep iIns up to date
      // based on the number of ports added.
      if (port.direction == Direction::In) {
        newPorts.push_back({iDel, PortInfo(port.name, classIn.getInstanceType(),
                                           Direction::In)});
        portAnnotations.push_back(constants.getEmptyArrayAttr());
        ++iIns;
      }
      newPorts.push_back(
          {iDel, PortInfo(StringAttr::get(context, Twine(port.name) + "_out"),
                          classOut.getInstanceType(), Direction::Out)});
      ++iIns;

      // Update annotations.
      portAnnotations.push_back(constants.getEmptyArrayAttr());

      // Don't increment the iDel since we deleted one port.
      continue;
    }

    // This is a non-domain port.  It will NOT be deleted.  Increment both
    // indices.
    ++iDel;
    ++iIns;

    // If this port has domain associations and is not zero width, then we need
    // to add port annotation trackers.  These will be hooked up to the Object's
    // associations later.  However, if there is no domain information or the
    // port is zero width, then annotations do not need to be modified.  Early
    // continue first, adding trackers otherwise.  Only create one tracker for
    // all associations.
    ArrayAttr domainAttr = cast_or_null<ArrayAttr>(port.domains);
    if (!domainAttr || domainAttr.empty() ||
        hasZeroBitWidth(type_cast<FIRRTLType>(port.type))) {
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
      auto [object, inputPort, outputPort, temp, associations] = info;
      OpBuilder builder(object);
      builder.setInsertionPointAfter(object);
      // Hook up domain ports.  If this was an input domain port, then drive the
      // object's "domain in" port with the new input class port.  Splice
      // references to the old port with the new port---users (e.g., domain
      // defines) will be updated later.  If an output, then splice to the
      // object's "domain in" port.  If this participates in domain defines,
      // this will be hoked up later.
      auto subDomainInfoIn =
          ObjectSubfieldOp::create(builder, object.getLoc(), object, 0);
      if (inputPort) {
        PropAssignOp::create(builder, object.getLoc(), subDomainInfoIn,
                             body->getArgument(*inputPort));
        splice(temp, body->getArgument(*inputPort));
      } else {
        splice(temp, subDomainInfoIn);
      }

      // Hook up the "association in" port.
      auto subAssociations =
          ObjectSubfieldOp::create(builder, object.getLoc(), object, 2);
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

    // Remove all domain users.  Delay deleting conversions until the module
    // body is walked as it is possible that conversions have multiple users.
    //
    // This has the effect of removing all the conversions that we created.
    // Note: this relies on the fact that we don't create conversions if there
    // are no users.  (See early exits in `stubOut` and `splice`.)
    //
    // Note: this cannot visit conversions directly as we don't have guarantees
    // that there won't be other conversions flying around.  E.g., LowerDPI
    // leaves conversions that are cleaned up by LowerToHW.  (This is likely
    // wrong, but it doesn't cost us anything to do it this way.)
    DenseSet<Operation *> conversionsToErase;
    DenseSet<Operation *> operationsToErase;
    auto walkResult = op.walk([&](Operation *walkOp) {
      // This is an operation that we have previously determined can be deleted
      // when examining an earlier operation.  Delete it now as it is only safe
      // to do so when visiting it.
      if (operationsToErase.contains(walkOp)) {
        walkOp->erase();
        return WalkResult::advance();
      }

      // Handle UnsafeDomainCastOp.
      if (auto castOp = dyn_cast<UnsafeDomainCastOp>(walkOp)) {
        for (auto value : castOp.getDomains()) {
          auto *conversion = value.getDefiningOp();
          assert(isa<UnrealizedConversionCastOp>(conversion));
          conversionsToErase.insert(conversion);
        }

        castOp.getResult().replaceAllUsesWith(castOp.getInput());
        castOp.erase();
        return WalkResult::advance();
      }

      // Track anonymous domains for later traversal and erasure.
      if (auto anonDomain = dyn_cast<DomainCreateAnonOp>(walkOp)) {
        conversionsToErase.insert(anonDomain);
        return WalkResult::advance();
      }

      // If we see a WireOp of a domain type, then we want to erase it.  To do
      // this, find what is driving it and what it is driving and then replace
      // that triplet of operations with a single domain define inserted before
      // the latest define.  If the wire is undriven or if the wire drives
      // nothing, then everything will be deleted.
      //
      // Before:
      //
      //     %a = firrtl.wire : !firrtl.domain // <- operation being visited
      //     firrtl.domain.define %a, %src
      //     firrtl.domain.define %dst, %a
      //
      // After:
      //     %a = firrtl.wire : !firrtl.domain // <- to-be-deleted after walk
      //     firrtl.domain.define %a, %src     // <- to-be-deleted when visited
      //     firrtl.domain.define %dst, %src   // <- added
      //     firrtl.domain.define %dst, %a     // <- to-be-deleted when visited
      if (auto wireOp = dyn_cast<WireOp>(walkOp)) {
        if (type_isa<DomainType>(wireOp.getResult().getType())) {
          Value src, dst;
          DomainDefineOp lastDefineOp;
          for (auto *user : llvm::make_early_inc_range(wireOp->getUsers())) {
            if (src && dst)
              break;
            auto domainDefineOp = dyn_cast<DomainDefineOp>(user);
            if (!domainDefineOp) {
              auto diag = wireOp.emitOpError()
                          << "cannot be lowered by `LowerDomains` because it "
                             "has a user that is not a domain define op";
              diag.attachNote(user->getLoc()) << "is one such user";
              return WalkResult::interrupt();
            }
            if (!lastDefineOp || lastDefineOp->isBeforeInBlock(domainDefineOp))
              lastDefineOp = domainDefineOp;
            if (wireOp == domainDefineOp.getSrc().getDefiningOp())
              dst = domainDefineOp.getDest();
            else
              src = domainDefineOp.getSrc();
            operationsToErase.insert(domainDefineOp);
          }
          conversionsToErase.insert(wireOp);
          // If this wire is dead or undriven, then there's nothing to do.
          if (!src || !dst)
            return WalkResult::advance();
          // Insert a domain define that removes the need for the wire.  This is
          // inserted just before the latest domain define involving the wire.
          // This is done to prevent unnecessary permutations of the IR.
          OpBuilder builder(lastDefineOp);
          DomainDefineOp::create(builder, builder.getUnknownLoc(), dst, src);
        }
        return WalkResult::advance();
      }

      // Handle DomainDefineOp.  Skip all other operations.
      auto defineOp = dyn_cast<DomainDefineOp>(walkOp);
      if (!defineOp)
        return WalkResult::advance();

      // There are only two possibilities for kinds of `DomainDefineOp`s that we
      // can see a this point: the destination is always a conversion cast and
      // the source is _either_ (1) a conversion cast if the source is a module
      // or instance port or (2) an anonymous domain op.  This relies on the
      // earlier "canonicalization" that erased `WireOp`s to leave only
      // `DomainDefineOp`s.
      auto *src = defineOp.getSrc().getDefiningOp();
      auto dest = dyn_cast<UnrealizedConversionCastOp>(
          defineOp.getDest().getDefiningOp());
      if (!src || !dest)
        return WalkResult::advance();

      conversionsToErase.insert(src);
      conversionsToErase.insert(dest);

      if (auto srcCast = dyn_cast<UnrealizedConversionCastOp>(src)) {
        assert(srcCast.getNumOperands() == 1 && srcCast.getNumResults() == 1);
        OpBuilder builder(defineOp);
        PropAssignOp::create(builder, defineOp.getLoc(), dest.getOperand(0),
                             srcCast.getOperand(0));
      } else if (!isa<DomainCreateAnonOp>(src)) {
        auto diag = defineOp.emitOpError()
                    << "has a source which cannot be lowered by 'LowerDomains'";
        diag.attachNote(src->getLoc()) << "unsupported source is here";
        return WalkResult::interrupt();
      }

      defineOp->erase();
      return WalkResult::advance();
    });

    if (walkResult.wasInterrupted())
      return failure();

    // Erase all the conversions.
    for (auto *op : conversionsToErase)
      op->erase();
  }

  // Set new port annotations.
  op.setPortAnnotationsAttr(ArrayAttr::get(context, portAnnotations));

  return success();
}

LogicalResult LowerModule::lowerInstances() {
  // Early exit if there is no work to do.
  if (eraseVector.none() && newPorts.empty())
    return success();

  // TODO: There is nothing to do unless this instance is a module or external
  // module.  This mirros code in the `lowerModule` member function.  Figure out
  // a way to clean this up, possible by making `LowerModule` a true noop if
  // this is not one of these kinds of modulelikes.
  if (!isa<FModuleOp, FExtModuleOp>(op))
    return success();

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

    for (auto i : eraseVector.set_bits())
      indexToDomain[i].temp = stubOut(instanceOp.getResult(i));

    auto erased = instanceOp.cloneWithErasedPortsAndReplaceUses(eraseVector);
    auto inserted = erased.cloneWithInsertedPortsAndReplaceUses(newPorts);
    instanceGraph.replaceInstance(instanceOp, inserted);

    for (auto &[i, info] : indexToDomain) {
      Value splicedValue;
      if (info.inputPort) {
        // Handle input port.  Just hook it up.
        splicedValue = inserted.getResult(*info.inputPort);
      } else {
        // Handle output port.  Splice in the output field that contains the
        // domain object.  This requires creating an object subfield.
        OpBuilder builder(inserted);
        builder.setInsertionPointAfter(inserted);
        splicedValue = ObjectSubfieldOp::create(
            builder, inserted.getLoc(), inserted.getResult(info.outputPort), 1);
      }

      splice(info.temp, splicedValue);
    }

    instanceOp.erase();
    erased.erase();
  }

  return success();
}

/// Class used to lwoer a circuit that contains domains.  This hides any state
/// that may need to be cleared across invocations of this pass to keep the
/// actual pass code cleaner.
class LowerCircuit {

public:
  LowerCircuit(CircuitOp circuit, InstanceGraph &instanceGraph,
               llvm::Statistic &numDomains)
      : circuit(circuit), instanceGraph(instanceGraph),
        constants(circuit.getContext()), numDomains(numDomains) {}

  /// Lower the circuit, removing all domains.
  LogicalResult lowerCircuit();

private:
  /// Lower one domain.
  LogicalResult lowerDomain(DomainOp);

  /// The circuit this class is lowering.
  CircuitOp circuit;

  /// A reference to an instance graph.  This will be mutated.
  InstanceGraph &instanceGraph;

  /// Internal store of lazily constructed constants.
  Constants constants;

  /// Mutable reference to the number of domains this class will lower.
  llvm::Statistic &numDomains;

  /// Store of the mapping from a domain name to the classes that it has been
  /// lowered into.
  DenseMap<Attribute, Classes> classes;
};

LogicalResult LowerCircuit::lowerDomain(DomainOp op) {
  ImplicitLocOpBuilder builder(op.getLoc(), op);
  auto *context = op.getContext();
  auto name = op.getNameAttr();
  SmallVector<PortInfo> classInPorts;
  for (auto field : op.getFields().getAsRange<DomainFieldAttr>())
    classInPorts.append({{/*name=*/builder.getStringAttr(
                              Twine(field.getName().getValue()) + "_in"),
                          /*type=*/field.getType(), /*dir=*/Direction::In},
                         {/*name=*/builder.getStringAttr(
                              Twine(field.getName().getValue()) + "_out"),
                          /*type=*/field.getType(), /*dir=*/Direction::Out}});
  auto classIn = ClassOp::create(builder, name, classInPorts);
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

  auto connectPairWise = [&builder](ClassOp &classOp) {
    builder.setInsertionPointToStart(classOp.getBodyBlock());
    for (size_t i = 0, e = classOp.getNumPorts(); i != e; i += 2)
      PropAssignOp::create(builder, classOp.getArgument(i + 1),
                           classOp.getArgument(i));
  };
  connectPairWise(classIn);
  connectPairWise(classOut);

  classes.insert({name, {classIn, classOut}});
  instanceGraph.addModule(classIn);
  instanceGraph.addModule(classOut);
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
    LowerModule lowerModule(moduleOp, classes, constants, instanceGraph);
    if (failed(lowerModule.lowerModule()))
      return failure();
    LLVM_DEBUG(llvm::dbgs() << "    instances:\n");
    return lowerModule.lowerInstances();
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

  markAnalysesPreserved<InstanceGraph>();
}
