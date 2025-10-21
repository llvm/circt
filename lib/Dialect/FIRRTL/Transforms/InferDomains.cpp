//===- InferDomains.cpp - Infer and Check FIRRTL Domains ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements FIRRTL domain inference and checking with canonical
// domain representation. Domain sequences are canonicalized by sorting and
// removing duplicates, making domain order irrelevant and allowing duplicate
// domains to be treated as equivalent. The result of this pass is either a
// correctly domain-inferred circuit or pass failure if the circuit contains
// illegal domain crossings.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/Debug.h"
#include "circt/Support/Namespace.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-infer-domains"
#undef NDEBUG

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_INFERDOMAINS
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

using InstanceIterator = InstanceGraphNode::UseIterator;
using InstanceRange = llvm::iterator_range<InstanceIterator>;
using PortInsertions = SmallVector<std::pair<unsigned, PortInfo>>;

//====--------------------------------------------------------------------------
// Helpers for working with module or instance domain info.
//====--------------------------------------------------------------------------

/// From a domain info attribute, get the domain-type of a domain value at
/// index i.
static StringAttr getDomainPortTypeName(ArrayAttr info, size_t i) {
  if (info.empty())
    return nullptr;
  auto ref = cast<FlatSymbolRefAttr>(info[i]);
  return ref.getAttr();
}

/// From a domain info attribute, get the row of associated domains for a
/// hardware value at index i.
static auto getPortDomainAssociation(ArrayAttr info, size_t i) {
  if (info.empty())
    return info.getAsRange<IntegerAttr>();
  return cast<ArrayAttr>(info[i]).getAsRange<IntegerAttr>();
}

/// Return true if the value is a port on the module.
static bool isPort(FModuleOp module, BlockArgument arg) {
  return arg.getOwner()->getParentOp() == module;
}

/// Return true if the value is a port on the module.
static bool isPort(FModuleOp module, Value value) {
  auto arg = dyn_cast<BlockArgument>(value);
  if (!arg)
    return false;
  return isPort(module, arg);
}

//====--------------------------------------------------------------------------
// Circuit-wide state.
//====--------------------------------------------------------------------------

/// Each declared domain in the circuit is assigned an index, based on the order
/// in which it appears. Domain associations for hardware values are represented
/// as a list of domains, sorted by the index of the domain type.
using DomainTypeID = size_t;

/// Information about the domains in the circuit. Able to map domains to their
/// type ID, which in this pass is the canonical way to reference the type
/// of a domain.
namespace {
struct CircuitDomainInfo {
  CircuitDomainInfo(CircuitOp circuit) { processCircuit(circuit); }

  ArrayRef<DomainOp> getDomains() const { return domainTable; }
  size_t getNumDomains() const { return domainTable.size(); }
  DomainOp getDomain(DomainTypeID id) const { return domainTable[id]; }

  DomainTypeID getDomainTypeID(DomainOp op) const {
    return typeIDTable.at(op.getNameAttr());
  }

  DomainTypeID getDomainTypeID(StringAttr name) const {
    return typeIDTable.at(name);
  }

  DomainTypeID getDomainTypeID(FlatSymbolRefAttr ref) const {
    return getDomainTypeID(ref.getAttr());
  }

  DomainTypeID getDomainTypeID(ArrayAttr info, size_t i) const {
    auto name = getDomainPortTypeName(info, i);
    return getDomainTypeID(name);
  }

  DomainTypeID getDomainTypeID(Value value) const {
    assert(isa<DomainType>(value.getType()));
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      auto *block = arg.getOwner();
      auto *owner = block->getParentOp();
      auto module = cast<FModuleOp>(owner);
      auto info = module.getDomainInfoAttr();
      auto i = arg.getArgNumber();
      return getDomainTypeID(info, i);
    }

    auto result = dyn_cast<OpResult>(value);
    auto *owner = result.getOwner();
    auto instance = cast<InstanceOp>(owner);
    auto info = instance.getDomainInfoAttr();
    auto i = result.getResultNumber();
    return getDomainTypeID(info, i);
  }

private:
  void processDomain(DomainOp op) {
    auto index = domainTable.size();
    auto name = op.getNameAttr();
    domainTable.push_back(op);
    typeIDTable.insert({name, index});
  }

  void processCircuit(CircuitOp circuit) {
    for (auto decl : circuit.getOps<DomainOp>())
      processDomain(decl);
  }

  /// A map from domain type ID to op.
  SmallVector<DomainOp> domainTable;

  /// A map from domain name to type ID.
  DenseMap<StringAttr, DomainTypeID> typeIDTable;
};

/// Information about the changes made to the interface of a module, which can
/// be replayed onto an instance.
struct ModuleUpdateInfo {
  /// The updated domain information for a module.
  ArrayAttr portDomainInfo;
  /// The domain ports which have been inserted into a module.
  PortInsertions portInsertions;
};

struct GlobalState {
  GlobalState(CircuitOp circuit) : circuitInfo(circuit) {}

  CircuitDomainInfo circuitInfo;
  DenseMap<StringAttr, ModuleUpdateInfo> moduleUpdateTable;
};

} // namespace

//====--------------------------------------------------------------------------
// Terms: Syntax for unifying domain and domain-rows.
//====--------------------------------------------------------------------------

namespace {

/// The different sorts of terms in the unification engine.
enum class TermKind {
  Variable,
  Value,
  Row,
};

/// A term in the unification engine.
struct Term {
  constexpr Term(TermKind kind) : kind(kind) {}
  TermKind kind;
};

/// Helper to define a term kind.
template <TermKind K>
struct TermBase : Term {
  static bool classof(const Term *term) { return term->kind == K; }
  TermBase() : Term(K) {}
};

/// An unknown value.
struct VariableTerm : public TermBase<TermKind::Variable> {
  VariableTerm() : leader(nullptr) {}
  VariableTerm(Term *leader) : leader(leader) {}
  Term *leader;
};

/// A concrete value defined in the IR.
struct ValueTerm : public TermBase<TermKind::Value> {
  ValueTerm(Value value) : value(value) {}
  Value getValue() const { return value; }
  Value value;
};

/// A row of domains.
struct RowTerm : public TermBase<TermKind::Row> {
  RowTerm(ArrayRef<Term *> elements) : elements(elements) {}
  ArrayRef<Term *> elements;
};

/// A helper for assigning low numeric IDs to variables for user-facing output.
struct VariableIDTable {
  size_t get(VariableTerm *term) {
    auto [it, inserted] = table.insert({term, table.size() + 1});
    return it->second;
  }

  DenseMap<VariableTerm *, size_t> table;
};

#ifndef NDEBUG

raw_ostream &dump(llvm::raw_ostream &out, const Term *term);

// NOLINTNEXTLINE(misc-no-recursion)
raw_ostream &dump(raw_ostream &out, const VariableTerm *term) {
  return out << "var@" << (void *)term << "{leader=" << term->leader << "}";
}

// NOLINTNEXTLINE(misc-no-recursion)
raw_ostream &dump(raw_ostream &out, const ValueTerm *term) {
  return out << "val@" << term << "{" << term->value << "}";
}

// NOLINTNEXTLINE(misc-no-recursion)
raw_ostream &dump(raw_ostream &out, const RowTerm *term) {
  out << "row@" << term << "{";
  bool first = true;
  for (auto *element : term->elements) {
    if (!first)
      out << ", ";
    dump(out, element);
    first = false;
  }
  out << "}";
  return out;
}

// NOLINTNEXTLINE(misc-no-recursion)
raw_ostream &dump(raw_ostream &out, const Term *term) {
  if (!term)
    return out << "null";
  if (auto *var = dyn_cast<VariableTerm>(term))
    return dump(out, var);
  if (auto *val = dyn_cast<ValueTerm>(term))
    return dump(out, val);
  if (auto *row = dyn_cast<RowTerm>(term))
    return dump(out, row);
  llvm_unreachable("unknown term");
}
#endif // DEBUG

// NOLINTNEXTLINE(misc-no-recursion)
Term *find(Term *x) {
  if (!x)
    return nullptr;

  if (auto *var = dyn_cast<VariableTerm>(x)) {
    if (var->leader == nullptr)
      return var;

    auto *leader = find(var->leader);
    if (leader != var->leader)
      var->leader = leader;
    return leader;
  }

  return x;
}

LogicalResult unify(Term *lhs, Term *rhs);

LogicalResult unify(VariableTerm *x, Term *y) {
  x->leader = y;
  return success();
}

LogicalResult unify(ValueTerm *xv, Term *y) {
  if (auto *yv = dyn_cast<VariableTerm>(y)) {
    yv->leader = xv;
    return success();
  }
  if (auto *yv = dyn_cast<ValueTerm>(y)) {
    return success(xv == yv);
  }
  return failure();
}

// NOLINTNEXTLINE(misc-no-recursion)
LogicalResult unify(RowTerm *lhsRow, Term *rhs) {
  if (auto *rhsVar = dyn_cast<VariableTerm>(rhs)) {
    rhsVar->leader = lhsRow;
    return success();
  }
  if (auto *rhsRow = dyn_cast<RowTerm>(rhs)) {
    assert(lhsRow->elements.size() == rhsRow->elements.size());
    for (auto [x, y] : llvm::zip(lhsRow->elements, rhsRow->elements)) {
      if (failed(unify(x, y)))
        return failure();
    }
    return success();
  }

  return failure();
}

// NOLINTNEXTLINE(misc-no-recursion)
LogicalResult unify(Term *lhs, Term *rhs) {
  LLVM_DEBUG(auto &out = llvm::errs(); out << "unify x="; dump(out, lhs);
             out << " y="; dump(out, rhs); out << "\n";);
  if (!lhs || !rhs)
    return success();
  lhs = find(lhs);
  rhs = find(rhs);
  if (lhs == rhs)
    return success();
  if (auto *lhsVar = dyn_cast<VariableTerm>(lhs))
    return unify(lhsVar, rhs);
  if (auto *lhsVal = dyn_cast<ValueTerm>(lhs))
    return unify(lhsVal, rhs);
  if (auto *lhsRow = dyn_cast<RowTerm>(lhs))
    return unify(lhsRow, rhs);
  return failure();
}

void solve(Term *lhs, Term *rhs) {
  auto result = unify(lhs, rhs);
  (void)result;
  assert(result.succeeded());
}

} // namespace

//====--------------------------------------------------------------------------
// CheckModuleDomains
//====--------------------------------------------------------------------------

/// Check that a module has complete domain information.
static LogicalResult checkModuleDomains(GlobalState &globals,
                                        FModuleLike module) {
  auto numDomains = globals.circuitInfo.getNumDomains();
  auto domainInfo = module.getDomainInfoAttr();
  DenseMap<unsigned, unsigned> typeIDTable;
  for (size_t i = 0, e = module.getNumPorts(); i < e; ++i) {
    auto type = module.getPortType(i);

    if (isa<DomainType>(type)) {
      auto typeID = globals.circuitInfo.getDomainTypeID(domainInfo, i);
      typeIDTable[i] = typeID;
      continue;
    }

    if (auto baseType = type_dyn_cast<FIRRTLBaseType>(type)) {
      SmallVector<IntegerAttr> associations(numDomains);
      auto domains = getPortDomainAssociation(domainInfo, i);
      for (auto index : domains) {
        auto typeID = typeIDTable[index.getUInt()];
        auto &entry = associations[typeID];
        if (entry && entry != index) {
          auto domainName = globals.circuitInfo.getDomain(typeID).getNameAttr();
          auto portName = module.getPortNameAttr(i);
          auto diag = emitError(module.getPortLocation(i))
                      << "ambiguous " << domainName << " association for port "
                      << portName;

          auto d1Loc = module.getPortLocation(entry.getUInt());
          auto d1Name = module.getPortNameAttr(entry.getUInt());
          diag.attachNote(d1Loc)
              << "associated with " << domainName << " port " << d1Name;

          auto d2Loc = module.getPortLocation(index.getUInt());
          auto d2Name = module.getPortNameAttr(index.getUInt());
          diag.attachNote(d2Loc)
              << "associated with " << domainName << " port " << d2Name;
        }
        entry = index;
      }

      for (size_t typeID = 0; typeID < numDomains; ++typeID) {
        auto association = associations[typeID];
        if (!association) {
          auto domainName = globals.circuitInfo.getDomain(typeID).getNameAttr();
          auto portName = module.getPortNameAttr(i);
          return emitError(module.getPortLocation(i))
                 << "missing " << domainName << " association for port "
                 << portName;
        }
      }
    }
  }

  return success();
}

//====--------------------------------------------------------------------------
// InferModuleDomains: Primary workhorse for inferring domains on modules.
//====--------------------------------------------------------------------------

namespace {
class InferModuleDomains {
public:
  /// Run infer-domains on a module.
  static LogicalResult run(GlobalState &, FModuleOp);

private:
  /// Initialize module-level state.
  InferModuleDomains(GlobalState &);

  /// Execute on the given module.
  LogicalResult operator()(FModuleOp);

  /// Record the domain associations of hardware ports, and record the
  /// underlying value of output domain ports.
  LogicalResult processPorts(FModuleOp);

  /// Record the domain associations of hardware, and record the underlying
  /// value of domains, defined within the body of the module.
  LogicalResult processBody(FModuleOp);

  /// Record the domain associations of any operands or results, updating the op
  /// if necessary.
  LogicalResult processOp(Operation *);
  LogicalResult processOp(InstanceOp);
  LogicalResult processOp(InstanceChoiceOp);
  LogicalResult processOp(UnsafeDomainCastOp);
  LogicalResult processOp(DomainDefineOp);

  /// Apply the port changes of a module onto an instance-like op.
  template <typename T>
  T updateInstancePorts(T op, const ModuleUpdateInfo &update);

  /// Record the domain associations of the ports of an instance-like op.
  template <typename T>
  LogicalResult processInstancePorts(T op);

  LogicalResult updateModule(FModuleOp);

  /// Build a table of exported domains: a map from domains defined internally,
  /// to their set of aliasing output ports.
  void initializeExportTable(FModuleOp);

  /// After generalizing the module, all domains should be solved. Reflect the
  /// solved domain associations into the port domain info attribute.
  LogicalResult updatePortDomainAssociations(FModuleOp);

  /// After updating the port domain associations, walk the body of the module
  /// to fix up any child instance modules.
  LogicalResult updateDomainAssociationsInBody(FModuleOp);
  LogicalResult updateOpDomainAssociations(Operation *);

  template <typename T>
  LogicalResult updateInstanceDomainAssociations(T op);

  /// Copy the domain associations from the module domain info attribute into a
  /// small vector.
  SmallVector<Attribute> copyPortDomainAssociations(ArrayAttr, size_t);

  /// Add domain ports for any uninferred domains associated to hardware.
  /// Returns the inserted ports, which will be used later to generalize the
  /// instances of this module.
  void generalizeModule(FModuleOp);

  /// Unify the associated domain rows of two terms.
  LogicalResult unifyAssociations(Operation *, Value, Value);

  /// If the domain value is an alias, returns the domain it aliases.
  Value getUnderlyingDomain(Value);

  /// Record a mapping from domain in the IR to its corresponding term.
  void setTermForDomain(Value, Term *);

  /// Get the corresponding term for a domain in the IR.
  Term *getTermForDomain(Value);

  /// Get the corresponding term for a domain in the IR, or null if unset.
  Term *getOptTermForDomain(Value) const;

  /// Record a mapping from a hardware value in the IR to a term which
  /// represents the row of domains it is associated with.
  void setDomainAssociation(Value, Term *);

  /// Get the associated domain row, forced to be at least a row.
  RowTerm *getDomainAssociationAsRow(Value);

  /// For a hardware value, get the term which represents the row of associated
  /// domains. If no mapping has been defined, allocate a variable to stand for
  /// the row of domains.
  Term *getDomainAssociation(Value);

  /// For a hardware value, get the term which represents the row of associated
  /// domains. If no mapping has been defined, returns nullptr.
  Term *getOptDomainAssociation(Value) const;

  /// Allocate a row, where each domain is a variable.
  RowTerm *allocateRow();

  /// Allocate a row.
  RowTerm *allocateRow(ArrayRef<Term *>);

  /// Allocate a term.
  template <typename T, typename... Args>
  T *allocate(Args &&...);

  /// Allocate an array of terms. If any terms were left null, automatically
  /// replace them with a new variable.
  ArrayRef<Term *> allocateArray(ArrayRef<Term *>);

  /// Print a term in a user-friendly way.
  void render(Diagnostic &, Term *) const;
  void render(Diagnostic &, VariableIDTable &, Term *) const;

  template <typename T>
  void emitPortDomainCrossingError(T, size_t, DomainTypeID, Term *,
                                   Term *) const;

  /// Emit an error when we fail to infer the concrete domain to drive to a
  /// domain port.
  template <typename T>
  void emitDomainPortInferenceError(T, size_t) const;

  /// Information about the domains in a circuit.
  GlobalState &globals;

  /// Term allocator.
  llvm::BumpPtrAllocator allocator;

  /// Map from domains in the IR to their underlying term.
  DenseMap<Value, Term *> termTable;

  /// A map from hardware values to their associated row of domains, as a term.
  DenseMap<Value, Term *> associationTable;

  /// A map from local domain definition to its aliasing output ports.
  DenseMap<Value, TinyPtrVector<BlockArgument>> exportTable;
};
} // namespace

LogicalResult InferModuleDomains::run(GlobalState &globals, FModuleOp module) {
  return InferModuleDomains(globals)(module);
}

InferModuleDomains::InferModuleDomains(GlobalState &globals)
    : globals(globals) {}

LogicalResult InferModuleDomains::operator()(FModuleOp module) {
  LLVM_DEBUG(
      llvm::errs() << "================================================\n";
      llvm::errs() << "infer module domains: " << module.getModuleName()
                   << "\n";
      llvm::errs() << "================================================\n";);

  if (failed(processPorts(module)))
    return failure();

  if (failed(processBody(module)))
    return failure();

  LLVM_DEBUG(for (auto association : associationTable) {
    llvm::errs() << "association:\n";
    llvm::errs() << "  " << association.first << "\n";
    llvm::errs() << "  " << association.second << "\n";
  });

  return updateModule(module);
}

LogicalResult InferModuleDomains::processPorts(FModuleOp module) {
  auto domainInfo = module.getDomainInfoAttr();
  auto numPorts = module.getNumPorts();

  // Process module ports - domain ports define explicit domains.
  DenseMap<unsigned, unsigned> domainTypeIDTable;
  for (size_t i = 0; i < numPorts; ++i) {
    BlockArgument port = module.getArgument(i);

    // This is a domain port.
    if (isa<DomainType>(port.getType())) {
      auto typeID = globals.circuitInfo.getDomainTypeID(domainInfo, i);
      domainTypeIDTable[i] = typeID;
      if (module.getPortDirection(i) == Direction::In) {
        setTermForDomain(port, allocate<ValueTerm>(port));
      }
      continue;
    }

    // This is a port, which may have explicit domain information.
    auto portDomains = getPortDomainAssociation(domainInfo, i);
    if (portDomains.empty())
      continue;

    SmallVector<Term *> elements(globals.circuitInfo.getNumDomains());
    for (auto domainPortIndexAttr : portDomains) {
      auto domainPortIndex = domainPortIndexAttr.getUInt();
      auto domainTypeID = domainTypeIDTable[domainPortIndex];
      auto domainValue = module.getArgument(domainPortIndex);
      auto *term = getTermForDomain(domainValue);
      auto &slot = elements[domainTypeID];
      if (failed(unify(slot, term))) {
        emitPortDomainCrossingError(module, i, domainTypeID, slot, term);
        return failure();
      }
      elements[domainTypeID] = term;
    }
    auto *row = allocateRow(elements);
    setDomainAssociation(port, row);
  }

  return success();
}

LogicalResult InferModuleDomains::processBody(FModuleOp module) {
  LogicalResult result = success();
  module.getBody().walk([&](Operation *op) -> WalkResult {
    if (failed(processOp(op))) {
      result = failure();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result;
}

LogicalResult InferModuleDomains::processOp(Operation *op) {
  LLVM_DEBUG(llvm::errs() << "process op: " << *op << "\n");

  if (auto instance = dyn_cast<InstanceOp>(op))
    return processOp(instance);
  if (auto instance = dyn_cast<InstanceChoiceOp>(op))
    return processOp(instance);
  if (auto cast = dyn_cast<UnsafeDomainCastOp>(op))
    return processOp(cast);
  if (auto def = dyn_cast<DomainDefineOp>(op))
    return processOp(def);

  // For all other operations (including connections), propagate domains from
  // operands to results. This is a conservative approach - all operands and
  // results share the same domain associations.
  Value lhs;
  for (auto rhs : op->getOperands()) {
    if (!isa<FIRRTLBaseType>(rhs.getType()))
      continue;
    if (auto *op = rhs.getDefiningOp();
        op && op->hasTrait<OpTrait::ConstantLike>())
      continue;
    if (failed(unifyAssociations(op, lhs, rhs)))
      return failure();
    lhs = rhs;
  }
  for (auto rhs : op->getResults()) {
    if (!isa<FIRRTLBaseType>(rhs.getType()))
      continue;
    if (auto *op = rhs.getDefiningOp();
        op && op->hasTrait<OpTrait::ConstantLike>())
      continue;
    if (failed(unifyAssociations(op, lhs, rhs)))
      return failure();
    lhs = rhs;
  }
  return success();
}

LogicalResult InferModuleDomains::processOp(InstanceOp op) {
  auto module = op.getReferencedModuleNameAttr();
  auto lookup = globals.moduleUpdateTable.find(module);
  if (lookup != globals.moduleUpdateTable.end())
    op = updateInstancePorts(op, lookup->second);
  return processInstancePorts(op);
}

LogicalResult InferModuleDomains::processOp(InstanceChoiceOp op) {
  auto module = op.getDefaultTargetAttr().getAttr();
  auto lookup = globals.moduleUpdateTable.find(module);
  if (lookup != globals.moduleUpdateTable.end())
    op = updateInstancePorts(op, lookup->second);
  return processInstancePorts(op);
}

LogicalResult InferModuleDomains::processOp(UnsafeDomainCastOp op) {
  auto domains = op.getDomains();
  if (domains.empty())
    return unifyAssociations(op, op.getInput(), op.getResult());

  auto input = op.getInput();
  RowTerm *inputRow = getDomainAssociationAsRow(input);
  SmallVector<Term *> elements(inputRow->elements);
  for (auto domain : op.getDomains()) {
    auto typeID = globals.circuitInfo.getDomainTypeID(domain);
    elements[typeID] = getTermForDomain(domain);
  }

  auto *row = allocateRow(elements);
  setDomainAssociation(op.getResult(), row);
  return success();
}

LogicalResult InferModuleDomains::processOp(DomainDefineOp op) {
  auto src = op.getSrc();
  auto dst = op.getDest();
  auto *srcTerm = getTermForDomain(src);
  auto *dstTerm = getTermForDomain(dst);
  if (failed(unify(dstTerm, srcTerm))) {
    VariableIDTable idTable;
    auto diag = op->emitOpError("failed to propagate source to destination");
    auto &note1 = diag.attachNote();
    note1 << "destination has underlying value: ";
    render(note1, idTable, dstTerm);

    auto &note2 = diag.attachNote(src.getLoc());
    note2 << "source has underlying value: ";
    render(note2, idTable, srcTerm);
  }
  return unify(dstTerm, srcTerm);
}

template <typename T>
T InferModuleDomains::updateInstancePorts(T op,
                                          const ModuleUpdateInfo &update) {
  auto clone = op.cloneWithInsertedPortsAndReplaceUses(update.portInsertions);
  clone.setDomainInfoAttr(update.portDomainInfo);
  op->erase();
  return clone;
}

template <typename T>
LogicalResult InferModuleDomains::processInstancePorts(T op) {
  auto circuitInfo = globals.circuitInfo;
  auto numDomainTypes = circuitInfo.getNumDomains();
  DenseMap<unsigned, unsigned> domainPortTypeIDTable;
  auto domainInfo = op.getDomainInfoAttr();
  for (size_t i = 0, e = op->getNumResults(); i < e; ++i) {
    Value port = op.getResult(i);

    LLVM_DEBUG(llvm::errs() << "handling instance port: " << port << "\n");

    if (isa<DomainType>(port.getType())) {
      auto typeID = circuitInfo.getDomainTypeID(domainInfo, i);
      domainPortTypeIDTable[i] = typeID;
      if (op.getPortDirection(i) == Direction::Out) {
        setTermForDomain(port, allocate<ValueTerm>(port));
      }
      continue;
    }

    if (!isa<FIRRTLBaseType>(port.getType()))
      continue;

    // This is a port, which may have explicit domain information. Associate the
    // port with a row of domains, where each element is derived from the domain
    // associations recorded in the domain info attribute of the instance.
    SmallVector<Term *> elements(numDomainTypes);
    auto associations = getPortDomainAssociation(domainInfo, i);
    for (auto domainPortIndexAttr : associations) {
      auto domainPortIndex = domainPortIndexAttr.getUInt();
      auto typeID = domainPortTypeIDTable[domainPortIndex];
      auto *term = getTermForDomain(op.getResult(domainPortIndex));
      elements[typeID] = term;
    }

    // Confirm that we have complete domain information for the port. We can be
    // missing information if, for example, this was an instance of an
    // extmodule.
    for (size_t domainTypeID = 0; domainTypeID < numDomainTypes;
         ++domainTypeID) {
      if (elements[domainTypeID])
        continue;
      auto domainDecl = circuitInfo.getDomain(domainTypeID);
      auto domainName = domainDecl.getNameAttr();
      auto portName = op.getPortNameAttr(i);
      op->emitOpError() << "missing " << domainName << " association for port "
                        << portName;
      return failure();
    }

    setDomainAssociation(port, allocateRow(elements));
  }

  return success();
}

LogicalResult InferModuleDomains::updateModule(FModuleOp op) {
  initializeExportTable(op);

  generalizeModule(op);
  if (failed(updatePortDomainAssociations(op)))
    return failure();

  if (failed(updateDomainAssociationsInBody(op)))
    return failure();

  return success();
}

void InferModuleDomains::initializeExportTable(FModuleOp module) {
  size_t numPorts = module.getNumPorts();
  for (size_t i = 0; i < numPorts; ++i) {
    auto port = module.getArgument(i);
    auto type = port.getType();
    if (!isa<DomainType>(type))
      continue;
    auto value = getUnderlyingDomain(port);
    if (value)
      exportTable[value].push_back(port);
  }
}

LogicalResult
InferModuleDomains::updatePortDomainAssociations(FModuleOp module) {
  // At this point, all domain variables mentioned in ports have been
  // solved by generalizing the module (adding input domain ports). Now, we have
  // to form the new port domain information for the module by examining the
  // the associated domains of each port.
  auto *context = module.getContext();
  auto numDomains = globals.circuitInfo.getNumDomains();
  auto builder = OpBuilder::atBlockEnd(module.getBodyBlock());
  auto oldModuleDomainInfo = module.getDomainInfoAttr();
  auto numPorts = module.getNumPorts();
  SmallVector<Attribute> newModuleDomainInfo(numPorts);

  for (size_t i = 0; i < numPorts; ++i) {
    auto port = module.getArgument(i);
    auto type = port.getType();

    // If the port is an output domain, we may need to drive the output with
    // a value. If we don't know what value to drive to the port, error.
    if (isa<DomainType>(type)) {
      if (module.getPortDirection(i) == Direction::Out) {
        bool driven = false;
        for (auto *user : port.getUsers()) {
          if (auto connect = dyn_cast<FConnectLike>(user)) {
            if (connect.getDest() == port) {
              driven = true;
              break;
            }
          }
        }

        // Get the underlying value of the output port.
        auto *term = getTermForDomain(port);
        term = find(term);
        auto *val = dyn_cast<ValueTerm>(term);
        if (!val) {
          emitDomainPortInferenceError(module, i);
          return failure();
        }

        // If the output port is not driven, drive it.
        if (!driven) {
          auto loc = port.getLoc();
          auto value = val->value;
          DomainDefineOp::create(builder, loc, port, value);
        }
      }

      newModuleDomainInfo[i] = oldModuleDomainInfo[i];
      continue;
    }

    if (isa<FIRRTLBaseType>(type)) {
      auto associations = copyPortDomainAssociations(oldModuleDomainInfo, i);
      auto *row = getDomainAssociationAsRow(port);
      for (size_t domainTypeID = 0; domainTypeID < numDomains; ++domainTypeID) {
        if (associations[domainTypeID])
          continue;

        auto domain = cast<ValueTerm>(find(row->elements[domainTypeID]))->value;
        auto &exports = exportTable[domain];
        if (exports.empty()) {
          auto portName = module.getPortNameAttr(i);
          auto portLoc = module.getPortLocation(i);
          auto domainDecl = globals.circuitInfo.getDomain(domainTypeID);
          auto domainName = domainDecl.getNameAttr();
          auto diag = emitError(portLoc)
                      << "private " << domainName << " association for port "
                      << portName;
          diag.attachNote(domain.getLoc()) << "associated domain: " << domain;
          return failure();
        }

        if (exports.size() > 1) {
          auto portName = module.getPortNameAttr(i);
          auto portLoc = module.getPortLocation(i);
          auto domainDecl = globals.circuitInfo.getDomain(domainTypeID);
          auto domainName = domainDecl.getNameAttr();
          auto diag = emitError(portLoc)
                      << "ambiguous " << domainName << " association for port "
                      << portName;
          for (auto arg : exports) {
            auto name = module.getPortNameAttr(arg.getArgNumber());
            auto loc = module.getPortLocation(arg.getArgNumber());
            diag.attachNote(loc) << "candidate association " << name;
          }
          return failure();
        }

        auto argument = cast<BlockArgument>(exports[0]);
        auto domainPortIndex = argument.getArgNumber();
        associations[domainTypeID] = IntegerAttr::get(
            IntegerType::get(context, 32, IntegerType::Unsigned),
            domainPortIndex);
      }

      newModuleDomainInfo[i] = ArrayAttr::get(context, associations);
      continue;
    }

    newModuleDomainInfo[i] = oldModuleDomainInfo[i];
  }

  auto newModuleDomainInfoAttr =
      ArrayAttr::get(module.getContext(), newModuleDomainInfo);
  module.setDomainInfoAttr(newModuleDomainInfoAttr);

  // record the domain info for replaying on instances.
  auto &update = globals.moduleUpdateTable[module.getNameAttr()];
  update.portDomainInfo = newModuleDomainInfoAttr;

  return success();
}

SmallVector<Attribute>
InferModuleDomains::copyPortDomainAssociations(ArrayAttr moduleDomainInfo,
                                               size_t portIndex) {
  SmallVector<Attribute> result(globals.circuitInfo.getNumDomains());
  auto oldAssociations = getPortDomainAssociation(moduleDomainInfo, portIndex);
  for (auto domainPortIndexAttr : oldAssociations) {
    auto domainPortIndex = domainPortIndexAttr.getUInt();
    auto domainTypeID =
        globals.circuitInfo.getDomainTypeID(moduleDomainInfo, domainPortIndex);
    result[domainTypeID] = domainPortIndexAttr;
  };
  return result;
}

void InferModuleDomains::generalizeModule(FModuleOp module) {
  PortInsertions insertions;
  // If the port is hardware, we have to check the associated row of
  // domains. If any associated domain is a variable, we solve the variable
  // by generalizing the module with an additional input domain port. If any
  // associated domain is defined internally to the module, we have to add
  // an output domain port, to allow the domain to escape.
  DenseMap<VariableTerm *, unsigned> pendingSolutions;
  llvm::MapVector<Value, unsigned> pendingExports;

  size_t inserted = 0;
  auto numPorts = module.getNumPorts();
  for (size_t i = 0; i < numPorts; ++i) {
    auto port = module.getArgument(i);
    auto type = port.getType();

    if (!isa<FIRRTLBaseType>(type))
      continue;

    auto *row = getDomainAssociationAsRow(port);
    for (auto [typeID, term] : llvm::enumerate(row->elements)) {
      auto *domain = find(term);

      if (auto *val = dyn_cast<ValueTerm>(domain)) {
        auto value = val->value;
        // If the domain value is defined inside the module body, we must output
        // export the domain, so it may appear in the signature of the
        // module.
        if (isPort(module, value))
          continue;

        // The domain is defined internally. If there value is already exported,
        // or will be exported, we are done.
        if (exportTable.contains(value) || pendingExports.contains(value))
          continue;

        // We must insert a new output domain port.
        auto domainDecl = globals.circuitInfo.getDomain(typeID);
        auto domainName = domainDecl.getNameAttr();

        auto portInsertionPoint = i;
        auto portName = domainName;
        auto portType = DomainType::get(module.getContext());
        auto portDirection = Direction::Out;
        auto portSym = StringAttr();
        auto portLoc = port.getLoc();
        auto portAnnos = std::nullopt;
        auto portDomainInfo = FlatSymbolRefAttr::get(domainName);
        PortInfo portInfo(portName, portType, portDirection, portSym, portLoc,
                          portAnnos, portDomainInfo);
        insertions.push_back({portInsertionPoint, portInfo});

        // Record the pending export.
        auto exportedPortIndex = inserted + portInsertionPoint;
        pendingExports[val->value] = exportedPortIndex;
        ++inserted;
      }

      if (auto *var = dyn_cast<VariableTerm>(domain)) {
        if (pendingSolutions.contains(var))
          continue;

        // insert a new input domain port for the variable.
        auto domainDecl = globals.circuitInfo.getDomain(typeID);
        auto domainName = domainDecl.getNameAttr();

        auto portInsertionPoint = i;
        auto portName = domainName;
        auto portType = DomainType::get(module.getContext());
        auto portDirection = Direction::In;
        auto portSym = StringAttr();
        auto portLoc = port.getLoc();
        auto portAnnos = std::nullopt;
        auto portDomainInfo = FlatSymbolRefAttr::get(domainName);
        PortInfo portInfo(portName, portType, portDirection, portSym, portLoc,
                          portAnnos, portDomainInfo);
        insertions.push_back({portInsertionPoint, portInfo});

        // Record the pending solution.
        auto solutionPortIndex = inserted + portInsertionPoint;
        pendingSolutions[var] = solutionPortIndex;
        ++inserted;
      }
    }
  }

  // Put the domain ports in place.
  module.insertPorts(insertions);

  // Solve the variables and record them as "self-exporting".
  for (auto [var, portIndex] : pendingSolutions) {
    auto port = module.getArgument(portIndex);
    auto *solution = allocate<ValueTerm>(port);
    solve(var, solution);
    // The port is an export of itself.
    exportTable[port].push_back(port);
  }

  // Drive the pending exports.
  auto builder = OpBuilder::atBlockEnd(module.getBodyBlock());
  for (auto [value, portIndex] : pendingExports) {
    auto port = module.getArgument(portIndex);
    DomainDefineOp::create(builder, port.getLoc(), port, value);
    exportTable[value].push_back(port);
    setTermForDomain(port, allocate<ValueTerm>(value));
  }

  // Record the insertions, so we can replay them on instances later.
  auto &update = globals.moduleUpdateTable[module.getNameAttr()];
  update.portInsertions = std::move(insertions);
}

LogicalResult
InferModuleDomains::updateDomainAssociationsInBody(FModuleOp module) {
  auto result = success();
  module.getBodyBlock()->walk([&](Operation *op) -> WalkResult {
    if (failed(updateOpDomainAssociations(op))) {
      result = failure();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result;
}

LogicalResult InferModuleDomains::updateOpDomainAssociations(Operation *op) {
  if (auto instance = dyn_cast<InstanceOp>(op))
    return updateInstanceDomainAssociations(instance);
  if (auto instance = dyn_cast<InstanceChoiceOp>(op))
    return updateInstanceDomainAssociations(instance);
  return success();
}

template <typename T>
LogicalResult InferModuleDomains::updateInstanceDomainAssociations(T op) {
  auto *context = op.getContext();
  OpBuilder builder(context);
  builder.setInsertionPointAfter(op);
  auto numPorts = op->getNumResults();
  for (size_t i = 0; i < numPorts; ++i) {
    auto port = op.getResult(i);
    auto type = port.getType();
    auto direction = op.getPortDirection(i);
    if (isa<DomainType>(type)) {
      if (direction == Direction::In) {
        bool driven = false;
        for (auto *user : port.getUsers()) {
          if (auto connect = dyn_cast<FConnectLike>(user)) {
            if (connect.getDest() == port) {
              driven = true;
              break;
            }
          }
        }
        if (!driven) {
          auto *term = getTermForDomain(port);
          term = find(term);
          if (auto *val = dyn_cast<ValueTerm>(term)) {
            auto loc = port.getLoc();
            auto value = val->value;
            DomainDefineOp::create(builder, loc, port, value);
          } else {
            emitDomainPortInferenceError(op, i);
            return failure();
          }
        }
      }
    }
  }
  return success();
}

LogicalResult InferModuleDomains::unifyAssociations(Operation *op, Value lhs,
                                                    Value rhs) {
  LLVM_DEBUG(llvm::errs() << "  unify associations of:\n";
             llvm::errs() << "    lhs=" << lhs << "\n";
             llvm::errs() << "    rhs=" << rhs << "\n";);

  if (!lhs || !rhs)
    return success();

  if (lhs == rhs)
    return success();

  auto *lhsTerm = getOptDomainAssociation(lhs);
  auto *rhsTerm = getOptDomainAssociation(rhs);

  if (lhsTerm) {
    if (rhsTerm) {
      if (failed(unify(lhsTerm, rhsTerm))) {
        auto diag = op->emitOpError("illegal domain crossing in operation");
        auto &note1 = diag.attachNote(lhs.getLoc());

        note1 << "1st operand has domains: ";
        VariableIDTable idTable;
        render(note1, idTable, lhsTerm);

        auto &note2 = diag.attachNote(rhs.getLoc());
        note2 << "2nd operand has domains: ";
        render(note2, idTable, rhsTerm);

        return failure();
      }
    }
    setDomainAssociation(rhs, lhsTerm);
    return success();
  }

  if (rhsTerm) {
    setDomainAssociation(lhs, rhsTerm);
    return success();
  }

  auto *var = allocate<VariableTerm>();
  setDomainAssociation(lhs, var);
  setDomainAssociation(rhs, var);
  return success();
}

Value InferModuleDomains::getUnderlyingDomain(Value value) {
  assert(isa<DomainType>(value.getType()));
  auto *term = getOptTermForDomain(value);
  if (auto *val = llvm::dyn_cast_if_present<ValueTerm>(term))
    return val->value;
  return nullptr;
}

Term *InferModuleDomains::getTermForDomain(Value value) {
  assert(isa<DomainType>(value.getType()));
  if (auto *term = getOptTermForDomain(value))
    return term;
  auto *term = allocate<VariableTerm>();
  setTermForDomain(value, term);
  return term;
}

Term *InferModuleDomains::getOptTermForDomain(Value value) const {
  assert(isa<DomainType>(value.getType()));
  auto it = termTable.find(value);
  if (it == termTable.end())
    return nullptr;
  return find(it->second);
}

void InferModuleDomains::setTermForDomain(Value value, Term *term) {
  assert(isa<DomainType>(value.getType()));
  assert(term);
  assert(!termTable.contains(value));
  termTable.insert({value, term});
}

RowTerm *InferModuleDomains::getDomainAssociationAsRow(Value value) {
  assert(isa<FIRRTLBaseType>(value.getType()));
  auto *term = getOptDomainAssociation(value);

  // If the term is unknown, allocate a fresh row and set the association.
  if (!term) {
    auto *row = allocateRow();
    setDomainAssociation(value, row);
    return row;
  }

  // If the term is already a row, return it.
  if (auto *row = dyn_cast<RowTerm>(term))
    return row;

  // Otherwise, unify the term with a fresh row of domains.
  if (auto *var = dyn_cast<VariableTerm>(term)) {
    auto *row = allocateRow();
    solve(var, row);
    return row;
  }

  assert(false && "unhandled term type");
  return nullptr;
}

Term *InferModuleDomains::getDomainAssociation(Value value) {
  auto *term = getOptDomainAssociation(value);
  if (term)
    return term;
  term = allocate<VariableTerm>();
  setDomainAssociation(value, term);
  return term;
}

Term *InferModuleDomains::getOptDomainAssociation(Value value) const {
  assert(isa<FIRRTLBaseType>(value.getType()));
  auto it = associationTable.find(value);
  if (it == associationTable.end())
    return nullptr;
  return find(it->second);
}

void InferModuleDomains::setDomainAssociation(Value value, Term *term) {
  assert(isa<FIRRTLBaseType>(value.getType()));
  assert(term);
  term = find(term);
  associationTable.insert({value, term});
  LLVM_DEBUG(llvm::errs() << "  set domain association: " << value;
             llvm::errs() << " -> " << term << "\n";);
}

RowTerm *InferModuleDomains::allocateRow() {
  SmallVector<Term *> elements;
  elements.resize(globals.circuitInfo.getNumDomains());
  return allocateRow(elements);
}

RowTerm *InferModuleDomains::allocateRow(ArrayRef<Term *> elements) {
  auto ds = allocateArray(elements);
  return allocate<RowTerm>(ds);
}

template <typename T, typename... Args>
T *InferModuleDomains::allocate(Args &&...args) {
  static_assert(std::is_base_of_v<Term, T>, "T must be a term");
  return new (allocator) T(std::forward<Args>(args)...);
}

ArrayRef<Term *> InferModuleDomains::allocateArray(ArrayRef<Term *> elements) {
  auto size = elements.size();
  if (size == 0)
    return {};

  auto *result = allocator.Allocate<Term *>(size);
  llvm::uninitialized_copy(elements, result);
  for (size_t i = 0; i < size; ++i)
    if (!result[i])
      result[i] = allocate<VariableTerm>();

  return ArrayRef(result, elements.size());
}

void InferModuleDomains::render(Diagnostic &out, Term *term) const {
  VariableIDTable idTable;
  render(out, idTable, term);
}

// NOLINTNEXTLINE(misc-no-recursion)
void InferModuleDomains::render(Diagnostic &out, VariableIDTable &idTable,
                                Term *term) const {
  term = find(term);
  if (auto *var = dyn_cast<VariableTerm>(term)) {
    out << "?" << idTable.get(var);
    return;
  }
  if (auto *val = dyn_cast<ValueTerm>(term)) {
    auto value = val->value;
    auto [name, rooted] = getFieldName(FieldRef(value, 0), false);
    out << name;
    return;
  }
  if (auto *row = dyn_cast<RowTerm>(term)) {
    bool first = true;
    out << "[";
    for (size_t i = 0, e = globals.circuitInfo.getNumDomains(); i < e; ++i) {
      auto domainOp = globals.circuitInfo.getDomain(i);
      if (!first) {
        out << ", ";
        first = false;
      }
      out << domainOp.getName() << ": ";
      render(out, idTable, row->elements[i]);
    }
    out << "]";
    return;
  }
}

template <typename T>
void InferModuleDomains::emitPortDomainCrossingError(T op, size_t i,
                                                     size_t domainTypeID,
                                                     Term *term1,
                                                     Term *term2) const {
  VariableIDTable idTable;

  auto portName = op.getPortNameAttr(i);
  auto portLoc = op.getPortLocation(i);
  auto domainDecl = globals.circuitInfo.getDomain(domainTypeID);
  auto domainName = domainDecl.getNameAttr();

  auto diag = emitError(portLoc);
  diag << "illegal " << domainName << " crossing in port " << portName;

  auto &note1 = diag.attachNote();
  note1 << "1st instance: ";
  render(note1, idTable, term1);

  auto &note2 = diag.attachNote();
  note2 << "2nd instance: ";
  render(note2, idTable, term2);
}

template <typename T>
void InferModuleDomains::emitDomainPortInferenceError(T op, size_t i) const {
  auto name = op.getPortNameAttr(i);
  auto diag = emitError(op->getLoc());
  auto info = op.getDomainInfo();
  diag << "unable to infer value for domain port " << name;
  for (size_t j = 0, e = op.getNumPorts(); j < e; ++j) {
    if (auto assocs = dyn_cast<ArrayAttr>(info[j])) {
      for (auto assoc : assocs) {
        if (i == cast<IntegerAttr>(assoc).getValue()) {
          auto name = op.getPortNameAttr(j);
          auto loc = op.getPortLocation(j);
          diag.attachNote(loc) << "associated with hardware port " << name;
          break;
        }
      }
    }
  }
}

static LogicalResult inferModuleDomains(GlobalState &globals,
                                        FModuleOp module) {
  return InferModuleDomains::run(globals, module);
}

//===---------------------------------------------------------------------------
// InferDomainsPass: Top-level pass implementation.
//===---------------------------------------------------------------------------

static LogicalResult runOnModuleLike(bool inferPublic, GlobalState &globals,
                                     Operation *op) {
  if (auto module = dyn_cast<FModuleOp>(op)) {
    if (module.isPublic() && !inferPublic)
      return checkModuleDomains(globals, module);
    return inferModuleDomains(globals, module);
  }

  if (auto extModule = dyn_cast<FExtModuleOp>(op)) {
    return checkModuleDomains(globals, extModule);
  }

  return success();
}

namespace {
struct InferDomainsPass
    : public circt::firrtl::impl::InferDomainsBase<InferDomainsPass> {
  using InferDomainsBase::InferDomainsBase;
  void runOnOperation() override;
};
} // namespace

void InferDomainsPass::runOnOperation() {
  LLVM_DEBUG(debugPassHeader(this) << "\n");
  auto circuit = getOperation();
  auto &instanceGraph = getAnalysis<InstanceGraph>();
  GlobalState globals(circuit);
  DenseSet<InstanceGraphNode *> visited;
  for (auto *root : instanceGraph) {
    for (auto *node : llvm::post_order_ext(root, visited)) {
      if (failed(runOnModuleLike(inferPublic, globals, node->getModule()))) {
        signalPassFailure();
        return;
      }
    }
  }
  LLVM_DEBUG(debugFooter() << "\n");
}
