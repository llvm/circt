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
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/Debug.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TrailingObjects.h"

#define DEBUG_TYPE "firrtl-infer-domains"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_INFERDOMAINS
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;
using llvm::TrailingObjects;

namespace {

using InstanceIterator = InstanceGraphNode::UseIterator;
using InstanceRange = llvm::iterator_range<InstanceIterator>;
using PortInsertions = SmallVector<std::pair<unsigned, PortInfo>>;

/// From a domain info attribute, get the domain-type of a domain value at
/// index i.
StringAttr getDomainPortTypeName(ArrayAttr info, size_t i) {
  if (info.empty())
    return nullptr;
  auto ref = cast<FlatSymbolRefAttr>(info[i]);
  return ref.getAttr();
}

/// From a domain info attribute, get the row of associated domains for a
/// hardware value at index i.
ArrayAttr getPortDomainAssociation(ArrayAttr info, size_t i) {
  if (info.empty())
    return info;
  return cast<ArrayAttr>(info[i]);
}

/// Each declared domain in the circuit is assigned an index, based on the order
/// in which it appears. Domain associations for hardware values are represented
/// as a list of domains, sorted by the index of the domain type.
using DomainIndex = size_t;

/// Information about the domains in the circuit. Able to map domains to their
/// domain-index, which in this pass is the canonical way to reference the type
/// of a domain.
struct CircuitDomainInfo {
  static CircuitDomainInfo get(CircuitOp circuit) {
    CircuitDomainInfo info;
    info.processCircuit(circuit);
    return info;
  }

  ArrayRef<DomainOp> getDomains() const { return domainTable; }
  size_t getNumDomains() const { return domainTable.size(); }
  DomainOp getDomain(DomainIndex id) const { return domainTable[id]; }

  DomainIndex getDomainIndex(DomainOp op) const {
    return indexTable.at(op.getNameAttr());
  }
  DomainIndex getDomainIndex(StringAttr name) const {
    return indexTable.at(name);
  }
  DomainIndex getDomainIndex(FlatSymbolRefAttr ref) const {
    return getDomainIndex(ref.getAttr());
  }
  DomainIndex getDomainIndex(ArrayAttr info, size_t i) const {
    auto name = getDomainPortTypeName(info, i);
    return getDomainIndex(name);
  }
  DomainIndex getDomainIndex(Value value) const {
    assert(isa<DomainType>(value.getType()));
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      auto *block = arg.getOwner();
      auto *owner = block->getParentOp();
      auto module = cast<FModuleOp>(owner);
      auto info = module.getDomainInfoAttr();
      auto i = arg.getArgNumber();
      return getDomainIndex(info, i);
    }

    auto result = dyn_cast<OpResult>(value);
    auto *owner = result.getOwner();
    auto instance = cast<InstanceOp>(owner);
    auto info = instance.getDomainInfoAttr();
    auto i = result.getResultNumber();
    return getDomainIndex(info, i);
  }

  void clear() {
    domainTable.clear();
    indexTable.clear();
  }

  void processCircuit(CircuitOp circuit) {
    clear();
    for (auto decl : circuit.getOps<DomainOp>())
      processDomain(decl);

    for (auto [i, domain] : llvm::enumerate(domainTable))
      llvm::errs() << "domain " << i << " = " << domain << "\n";
  }

  void processDomain(DomainOp op) {
    auto index = domainTable.size();
    auto name = op.getNameAttr();
    domainTable.push_back(op);
    indexTable.insert({name, index});
  }

  SmallVector<DomainOp> domainTable;
  DenseMap<StringAttr, DomainIndex> indexTable;
};

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

template <typename T>
T &operator<<(T &out, const Term &term);

template <typename T>
T &operator<<(T &out, const VariableTerm &term) {
  return out << "var@" << (void *)&term << "{leader=" << term.leader << "}";
}

template <typename T>
T &operator<<(T &out, const ValueTerm &term) {
  return out << "value@" << (void *)&term << "{" << term.value << "}";
}

template <typename T>
T &operator<<(T &out, const RowTerm &term) {
  out << "row@" << (void *)&term << "{";
  bool first = true;
  for (auto *element : term.elements) {
    if (!first)
      out << ", ";
    out << element;
    first = false;
  }
  out << "}";
  return out;
}

template <typename T>
T &operator<<(T &out, const Term &term) {
  if (auto *var = dyn_cast<VariableTerm>(&term))
    return out << *var;
  if (auto *val = dyn_cast<ValueTerm>(&term))
    return out << *val;
  if (auto *row = dyn_cast<RowTerm>(&term))
    return out << *row;
  assert(0);
  llvm_unreachable("unknown term");
  return out;
}

template <typename T>
T &operator<<(T &out, const Term *term) {
  if (!term)
    return out << "null";
  return out << *term;
}

Term *find(Term *x) {
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

LogicalResult unify(RowTerm *lhsRow, Term *rhs) {
  if (auto rhsVar = dyn_cast<VariableTerm>(rhs)) {
    rhsVar->leader = lhsRow;
    return success();
  }
  if (auto rhsRow = dyn_cast<RowTerm>(rhs)) {
    assert(lhsRow->elements.size() == rhsRow->elements.size());
    for (auto [x, y] : llvm::zip(lhsRow->elements, rhsRow->elements)) {
      if (failed(unify(x, y)))
        return failure();
    }
    return success();
  }

  return failure();
}

LogicalResult unify(Term *lhs, Term *rhs) {
  llvm::errs() << "unify x=" << *lhs << " y=" << *rhs << "\n";
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

class InferModuleDomains {
public:
  /// Run infer-domains on a module.
  static LogicalResult run(const CircuitDomainInfo &, FModuleOp);

private:
  /// Initialize module-level state.
  InferModuleDomains(const CircuitDomainInfo &);

  /// Execute on the given module.
  LogicalResult operator()(FModuleOp);

  /// Record the domain associations of hardware ports, and record the
  /// underlying value of output domain ports.
  LogicalResult processPorts(FModuleOp);

  /// Record the domain associations of hardware, and record the underlying
  /// value of domains, defined within the body of the module.
  LogicalResult processBody(FModuleOp);

  /// Record the domain associations of any operands or results.
  LogicalResult processOp(Operation *);
  LogicalResult processOp(InstanceOp);
  LogicalResult processOp(UnsafeDomainCastOp);

  LogicalResult updateModule(FModuleOp);

  /// After generalizing the module, all domains should be solved. Reflect the
  /// solved domain associations into the port domain info attribute.
  LogicalResult updatePortDomainAssociations(FModuleOp);

  /// Add domain ports for any uninferred domains associated to hardware.
  /// Returns the inserted ports, which will be used later to generalize the
  /// instances of this module.
  PortInsertions generalizeModule(FModuleOp);

  void generalizeInstance(InstanceOp, const PortInsertions &);

  /// Unify the associated domain rows of two terms.
  LogicalResult unifyAssociations(Value, Value);

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

  Term *getDomainAssociation(Value);

  /// Get the term which represents the row of domains associated with a
  /// hardware value in the design.
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

  /// Information about the domains in a circuit.
  const CircuitDomainInfo &circuitInfo;

  /// Term allocator.
  llvm::BumpPtrAllocator allocator;

  /// Map from domains in the IR to their underlying term.
  DenseMap<Value, Term *> termTable;

  /// A map from hardware values to their associated row of domains, as a term.
  DenseMap<Value, Term *> associationTable;

  /// A boolean tracking if a non-fatal error occurred, or not.
  bool ok = true;
};

LogicalResult InferModuleDomains::run(const CircuitDomainInfo &circuitInfo,
                                      FModuleOp module) {
  return InferModuleDomains(circuitInfo)(module);
}

InferModuleDomains::InferModuleDomains(const CircuitDomainInfo &circuitInfo)
    : circuitInfo(circuitInfo) {}

LogicalResult InferModuleDomains::operator()(FModuleOp module) {
  if (failed(processPorts(module)))
    return failure();

  if (failed(processBody(module)))
    return failure();

  for (auto association : associationTable) {
    llvm::errs() << "association:\n";
    llvm::errs() << "  " << association.first << "\n";
    llvm::errs() << "  " << association.second << "\n";
  }

  if (failed(updateModule(module)))
    return failure();

  return llvm::success(ok);
}

LogicalResult InferModuleDomains::processPorts(FModuleOp module) {
  auto portDomainInfo = module.getDomainInfoAttr();
  auto numPorts = module.getNumPorts();

  // Process module ports - domain ports define explicit domains.
  DenseMap<unsigned, unsigned> domainIndexTable;
  for (size_t i = 0; i < numPorts; ++i) {
    Value port = module.getArgument(i);

    // This is a domain port.
    if (isa<DomainType>(port.getType())) {
      auto index = circuitInfo.getDomainIndex(portDomainInfo, i);
      domainIndexTable[i] = index;
      if (module.getPortDirection(i) == Direction::In) {
        setTermForDomain(port, allocate<ValueTerm>(port));
      }
      continue;
    }

    // This is a port, which may have explicit domain information.
    auto portDomains = getPortDomainAssociation(portDomainInfo, i);
    if (portDomains.empty())
      continue;

    SmallVector<Term *> elements(circuitInfo.getNumDomains());
    for (auto domainPortIndexAttr : portDomains.getAsRange<IntegerAttr>()) {
      auto domainPortIndex = domainPortIndexAttr.getUInt();
      auto domainIndex = domainIndexTable[domainPortIndex];
      auto *term = getTermForDomain(module.getArgument(domainPortIndex));
      elements[domainIndex] = term;
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
  llvm::errs() << "process op: " << *op << "\n";

  if (auto instance = dyn_cast<InstanceOp>(op))
    return processOp(instance);
  if (auto cast = dyn_cast<UnsafeDomainCastOp>(op))
    return processOp(cast);

  // For all operations (including connections), propagate domains from operands
  // to results This is a conservative approach - all operands and results share
  // the same domain associations.
  Value lhs;
  for (auto rhs : op->getOperands()) {
    if (failed(unifyAssociations(lhs, rhs)))
      return failure();
    lhs = rhs;
  }
  for (auto rhs : op->getResults()) {
    if (failed(unifyAssociations(lhs, rhs)))
      return failure();
    lhs = rhs;
  }
  return success();
}

LogicalResult InferModuleDomains::processOp(InstanceOp op) {
  DenseMap<unsigned, unsigned> portDomainIndexTable;
  auto domainInfo = op.getDomainInfoAttr();
  for (size_t i = 0, e = op->getNumResults(); i < e; ++i) {
    Value port = op.getResult(i);

    // This is a domain port.
    if (isa<DomainType>(port.getType())) {
      auto index = circuitInfo.getDomainIndex(domainInfo, i);
      portDomainIndexTable[i] = index;
      if (op.getPortDirection(i) == Direction::Out) {
        setTermForDomain(port, allocate<ValueTerm>(port));
      } else {
        setTermForDomain(port, allocate<VariableTerm>());
      }
      continue;
    }

    // This is a port, which may have explicit domain information.
    SmallVector<Term *> associations(circuitInfo.getNumDomains());
    auto domains = cast<ArrayAttr>(domainInfo).getAsRange<IntegerAttr>();
    for (auto domainPortIndexAttr : domains) {
      auto domainPortIndex = domainPortIndexAttr.getUInt();
      auto domainIndex = portDomainIndexTable[domainPortIndex];
      auto *term = getTermForDomain(op.getResult(domainPortIndex));
      associations[domainIndex] = term;
    }

    // Since we are processing bottom-up, we must have complete domain info
    // for each port on the instance.
    for (auto *domain : associations)
      assert(domain && "must have complete domain information.");

    setDomainAssociation(port, allocateRow(associations));
  }

  return success();
}

LogicalResult InferModuleDomains::processOp(UnsafeDomainCastOp op) {
  auto domains = op.getDomains();
  if (domains.empty())
    return unifyAssociations(op.getInput(), op.getResult());

  auto input = op.getInput();
  RowTerm *inputRow = getDomainAssociationAsRow(input);
  SmallVector<Term *> elements(inputRow->elements);
  for (auto domain : op.getDomains()) {
    auto index = circuitInfo.getDomainIndex(domain);
    elements[index] = getTermForDomain(domain);
  }

  auto *row = allocateRow(elements);
  setDomainAssociation(op.getResult(), row);
  return success();
}

LogicalResult InferModuleDomains::updateModule(FModuleOp op) {
  auto insertions = generalizeModule(op);

  if (failed(updatePortDomainAssociations(op)))
    return failure();

  return success();
}

LogicalResult
InferModuleDomains::updatePortDomainAssociations(FModuleOp module) {
  // At this point, all domain variables mentioned in ports have been
  // solved by generalizing the module (adding input domain ports). Now, we have
  // to form the new port domain information for the module by examining the
  // solutions to the associated domains of each port.
  auto *context = module.getContext();
  auto builder = OpBuilder::atBlockBegin(module.getBodyBlock());
  auto oldDomainInfo = module.getDomainInfoAttr();
  auto numPorts = module.getNumPorts();
  SmallVector<Attribute> domainInfo(numPorts);

  for (size_t i = 0; i < numPorts; ++i) {
    auto port = module.getArgument(i);
    auto type = port.getType();

    // By default, copy the old domain info over.
    domainInfo[i] = oldDomainInfo[i];

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

        if (!driven) {
          auto *term = getTermForDomain(port);
          term = find(term);
          if (auto *val = dyn_cast<ValueTerm>(term)) {
            auto loc = port.getLoc();
            auto type = getDomainPortTypeName(oldDomainInfo, i);
            auto value = val->value;
            DomainDefineOp::create(builder, loc, port, value, type);
          } else {
            return module.emitError() << "unable to infer output domain value";
          }
        }
      }
      continue;
    }

    if (isa<FIRRTLBaseType>(type)) {
      SmallVector<Attribute> associations(circuitInfo.getNumDomains());
      auto *row = getDomainAssociationAsRow(port);
      for (auto [typeID, term] : llvm::enumerate(row->elements)) {
        auto *domain = find(term);
        auto *val = dyn_cast<ValueTerm>(domain);
        if (!val)
          return module.emitError() << "unable to infer domain for port";
        auto arg = cast<BlockArgument>(val->value);
        auto idx = arg.getArgNumber();
        associations[typeID] = IntegerAttr::get(
            IntegerType::get(context, 32, IntegerType::Unsigned), idx);
      }
      domainInfo[i] = ArrayAttr::get(context, associations);
      continue;
    }
  }

  auto domainInfoAttr = ArrayAttr::get(module.getContext(), domainInfo);
  module.setDomainInfoAttr(domainInfoAttr);
  return success();
}

PortInsertions InferModuleDomains::generalizeModule(FModuleOp module) {
  // If the port is hardware, we have to check the associated row of
  // domains. If any associated domain is a variable, we solve the variable
  // by generalizing the module with an additional input domain port. If any
  // associated domain is defined internally to the module, we have to add
  // an output domain port, to allow the domain to escape.
  SmallVector<std::pair<unsigned, PortInfo>> insertions;
  DenseMap<VariableTerm *, unsigned> solutions;
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
      auto *var = dyn_cast<VariableTerm>(domain);

      if (!var)
        continue;

      if (solutions.contains(var))
        continue;

      // insert a new port for the variable.
      auto domainDecl = circuitInfo.getDomain(typeID);
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
      solutions[var] = solutionPortIndex;
      ++inserted;
    }
  }

  // Put the ports in place.
  module.insertPorts(insertions);

  llvm::errs() << "generalization complete\n";
  llvm::errs() << module << "\n";

  // Solve the variables.
  for (auto [var, portIndex] : solutions) {
    auto *solution = allocate<ValueTerm>(module.getArgument(portIndex));
    solve(var, solution);
  }

  return insertions;
}

LogicalResult InferModuleDomains::unifyAssociations(Value lhs, Value rhs) {
  llvm::errs() << "  unify associations of:\n";
  llvm::errs() << "    lhs=" << lhs << "\n";
  llvm::errs() << "    rhs=" << rhs << "\n";

  if (!lhs || !rhs)
    return success();

  if (lhs == rhs)
    return success();

  auto *lhsTerm = getOptDomainAssociation(lhs);
  auto *rhsTerm = getOptDomainAssociation(rhs);

  if (lhsTerm) {
    if (rhsTerm) {
      return unify(lhsTerm, rhsTerm);
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

Term *InferModuleDomains::getTermForDomain(Value value) {
  assert(isa<DomainType>(value.getType()));
  if (auto *term = getOptTermForDomain(value))
    return term;
  auto *term = allocate<VariableTerm>();
  setDomainAssociation(value, term);
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
  auto *row = dyn_cast<RowTerm>(term);
  if (row)
    return row;

  // Otherwise, unify the term with a fresh row of domains.
  row = allocateRow();
  auto result = unify(row, term);
  assert(result.succeeded());
  return row;
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
  llvm::errs() << "  set domain association: " << value << " -> " << term
               << "\n";
}

RowTerm *InferModuleDomains::allocateRow() {
  SmallVector<Term *> elements;
  elements.resize(circuitInfo.getNumDomains());
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

  auto result = allocator.Allocate<Term *>(size);
  llvm::uninitialized_copy(elements, result);
  for (size_t i = 0; i < size; ++i)
    if (!result[i])
      result[i] = allocate<VariableTerm>();

  return ArrayRef(result, elements.size());
}

////////////////////////////////////////////////////////////////////////////////

/// Domain inference and checking pass implementation.
/// Uses canonical domain representation to allow domain order independence
/// and duplicate domain handling.
class InferDomainsPass
    : public circt::firrtl::impl::InferDomainsBase<InferDomainsPass> {

public:
  InferDomainsPass() = default;

  /// Copy the pass by allocating fresh state.
  InferDomainsPass(const InferDomainsPass &) : InferDomainsPass() {}

  void runOnOperation() override;

private:
  /// Process a module and infer domains.
  LogicalResult processModule(const CircuitDomainInfo &, FModuleOp,
                              InstanceRange);
};

} // namespace

LogicalResult
InferDomainsPass::processModule(const CircuitDomainInfo &circuitInfo,
                                FModuleOp module, InstanceRange instances) {
  LLVM_DEBUG(llvm::dbgs() << "Processing module: " << module.getName() << "\n");
  return InferModuleDomains::run(circuitInfo, module);

  // Insert domain ports if needed.
  // TODO:

  // // Update domain information for all non-domain ports
  // SmallVector<Attribute> newDomainInfo;
  // bool anyUpdated = false;

  // for (auto [index, port] : llvm::enumerate(module.getPorts())) {
  //   // Skip domain ports - they don't need domain information
  //   if (isa<DomainType>(port.type)) {
  //     newDomainInfo.push_back(port.domains
  //                                 ? port.domains
  //                                 : ArrayAttr::get(module.getContext(),
  //                                 {}));
  //     continue;
  //   }

  //   // Get the inferred domains for this port
  //   Value portValue = module.getArgument(index);
  //   const auto &domains = domainUF.getDomains(portValue);

  //   // Convert domain values to domain indices
  //   SmallVector<Attribute> domainIndices;
  //   for (auto *term : domains) {
  //     auto *value = dyn_cast<ValueTerm>(find(term));
  //     if (!value)
  //       continue;
  //     // TODO

  //     auto ir = value->value;
  //     if (auto blockArg = dyn_cast<BlockArgument>(ir)) {
  //       // This is a reference to a domain port
  //       domainIndices.push_back(IntegerAttr::get(
  //           IntegerType::get(module.getContext(), 32,
  //           IntegerType::Unsigned), blockArg.getArgNumber()));
  //     }
  //   }

  //   ArrayAttr newDomains = ArrayAttr::get(module.getContext(),
  //   domainIndices);

  //   // Check if this is different from the existing domain information
  //   if (!port.domains || port.domains != newDomains) {
  //     anyUpdated = true;
  //   }

  //   newDomainInfo.push_back(newDomains);
}

// Update the module's domain information if anything changed
// if (anyUpdated) {
//   module->setAttr("domainInfo",
//                   ArrayAttr::get(module.getContext(), newDomainInfo));
// }

//   LLVM_DEBUG({
//     for (auto [index, port] : llvm::enumerate(module.getPorts())) {
//       llvm::dbgs() << "  - port: " << port.getName() << "\n"
//                    << "    domains:\n";
//       auto domains = domainUF.getDomains(module.getArgument(index));
//       if (domains.empty()) {
//         llvm::dbgs() << "      - inferred\n";
//         continue;
//       }
//       for (auto term : domains) {
//         auto leader = find(term);
//         Value domain;
//         if (auto value = dyn_cast<ValueTerm>(leader))
//           domain = value->value;
//         if (auto port = dyn_cast<BlockArgument>(domain)) {
//           llvm::dbgs() << "      - " <<
//           module.getPortName(port.getArgNumber())
//                        << "\n";
//           continue;
//         }
//       }
//     }
//   });

//   return success();
// }

void InferDomainsPass::runOnOperation() {
  LLVM_DEBUG(debugPassHeader(this) << "\n");
  auto circuit = getOperation();
  auto &instanceGraph = getAnalysis<InstanceGraph>();

  auto circuitInfo = CircuitDomainInfo::get(circuit);

  // Process each module in the circuit.
  DenseSet<InstanceGraphNode *> visited;
  for (auto *root : instanceGraph) {
    for (auto *node : llvm::post_order_ext(root, visited)) {
      if (auto module = dyn_cast<FModuleOp>(node->getModule()))
        if (failed(processModule(circuitInfo, module, node->uses()))) {
          signalPassFailure();
          return;
        }
    }
  }

  LLVM_DEBUG(debugFooter() << "\n");
}
