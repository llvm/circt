//===- InferDomains.cpp - Infer and Check FIRRTL Domains ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// InferDomains implements FIRRTL domain inference and checking. This pass is a
// bottom-up transform acting on modules. For each module, we ensure there are
// no domain crossings, and we make explicit the domain associations of ports.
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

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_INFERDOMAINS
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

namespace {

//====--------------------------------------------------------------------------
// Helpers.
//====--------------------------------------------------------------------------

using PortInsertions = SmallVector<std::pair<unsigned, PortInfo>>;

template <typename T>
bool shouldInfer(T op, InferDomainsMode mode) {
  return op.isPublic() ? shouldInferPublicModules(mode)
                       : shouldInferPrivateModules(mode);
}

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
auto getPortDomainAssociation(ArrayAttr info, size_t i) {
  if (info.empty())
    return info.getAsRange<IntegerAttr>();
  return cast<ArrayAttr>(info[i]).getAsRange<IntegerAttr>();
}

/// Return true if the value is a port on the module.
bool isPort(FModuleOp module, BlockArgument arg) {
  return arg.getOwner()->getParentOp() == module;
}

/// Return true if the value is a port on the module.
bool isPort(FModuleOp module, Value value) {
  auto arg = dyn_cast<BlockArgument>(value);
  if (!arg)
    return false;
  return isPort(module, arg);
}

/// Returns true if the value is driven by a connect op.
bool isDriven(Value port) {
  for (auto *user : port.getUsers())
    if (auto connect = dyn_cast<FConnectLike>(user))
      if (connect.getDest() == port)
        return true;
  return false;
}

//====--------------------------------------------------------------------------
// Global State.
//====--------------------------------------------------------------------------

/// Each domain type declared in the circuit is assigned a type-id, based on the
/// order of declaration. Domain associations for hardware values are
/// represented as a list, or row, of domains. The domains in a row are ordered
/// according to their type's id.
using DomainTypeID = size_t;

/// Information about the domains in the circuit. Able to map domains to their
/// type ID, which in this pass is the canonical way to reference the type
/// of a domain, as well as provide fast access to domain ops
class DomainInfo {
public:
  DomainInfo(CircuitOp circuit) { processCircuit(circuit); }

  ArrayRef<DomainOp> getDomains() const { return domainTable; }
  size_t getNumDomains() const { return domainTable.size(); }
  DomainOp getDomain(DomainTypeID id) const { return domainTable[id]; }

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

using ModuleUpdateTable = DenseMap<StringAttr, ModuleUpdateInfo>;

/// Apply the port changes of a module onto an instance-like op.
template <typename T>
T fixInstancePorts(T op, const ModuleUpdateInfo &update) {
  auto clone = op.cloneWithInsertedPortsAndReplaceUses(update.portInsertions);
  clone.setDomainInfoAttr(update.portDomainInfo);
  op->erase();
  return clone;
}

//====--------------------------------------------------------------------------
// Terms: Syntax for unifying domain and domain-rows.
//====--------------------------------------------------------------------------

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
  Term *leader = nullptr;
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

/// A helper for assigning low numeric IDs to variables for user-facing output.
class VariableIDTable {
public:
  size_t get(VariableTerm *term) {
    auto [it, inserted] = table.insert({term, table.size() + 1});
    return it->second;
  }

private:
  DenseMap<VariableTerm *, size_t> table;
};

// NOLINTNEXTLINE(misc-no-recursion)
void render(const DomainInfo &info, Diagnostic &out, VariableIDTable &idTable,
            Term *term) {
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
    for (size_t i = 0, e = info.getNumDomains(); i < e; ++i) {
      auto domainOp = info.getDomain(i);
      if (!first) {
        out << ", ";
        first = false;
      }
      out << domainOp.getName() << ": ";
      render(info, out, idTable, row->elements[i]);
    }
    out << "]";
    return;
  }
}

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
  llvm::interleaveComma(term->elements, out,
                        [&](auto element) { dump(out, element); });
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

LogicalResult unify(Term *lhs, Term *rhs);

LogicalResult unify(VariableTerm *x, Term *y) {
  assert(!x->leader);
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

class TermAllocator {
public:
  /// Allocate a row of fresh domain variables.
  RowTerm *allocRow(size_t size) {
    SmallVector<Term *> elements;
    elements.resize(size);
    return allocRow(elements);
  }

  /// Allocate a row of terms.
  RowTerm *allocRow(ArrayRef<Term *> elements) {
    auto ds = allocArray(elements);
    return alloc<RowTerm>(ds);
  }

  /// Allocate a fresh variable.
  VariableTerm *allocVar() { return alloc<VariableTerm>(); }

  /// Allocate a concrete domain.
  ValueTerm *allocVal(Value value) { return alloc<ValueTerm>(value); }

private:
  template <typename T, typename... Args>
  T *alloc(Args &&...args) {
    static_assert(std::is_base_of_v<Term, T>, "T must be a term");
    return new (allocator) T(std::forward<Args>(args)...);
  }

  ArrayRef<Term *> allocArray(ArrayRef<Term *> elements) {
    auto size = elements.size();
    if (size == 0)
      return {};

    auto *result = allocator.Allocate<Term *>(size);
    llvm::uninitialized_copy(elements, result);
    for (size_t i = 0; i < size; ++i)
      if (!result[i])
        result[i] = alloc<VariableTerm>();

    return ArrayRef(result, elements.size());
  }

  llvm::BumpPtrAllocator allocator;
};

//====--------------------------------------------------------------------------
// DomainTable: A mapping from IR to terms.
//====--------------------------------------------------------------------------

/// Tracks domain infomation for IR values.
class DomainTable {
public:
  /// If the domain value is an alias, returns the domain it aliases.
  Value getOptUnderlyingDomain(Value value) const {
    assert(isa<DomainType>(value.getType()));
    auto *term = getOptTermForDomain(value);
    if (auto *val = llvm::dyn_cast_if_present<ValueTerm>(term))
      return val->value;
    return nullptr;
  }

  /// Get the corresponding term for a domain in the IR, or null if unset.
  Term *getOptTermForDomain(Value value) const {
    assert(isa<DomainType>(value.getType()));
    auto it = termTable.find(value);
    if (it == termTable.end())
      return nullptr;
    return find(it->second);
  }

  /// Get the corresponding term for a domain in the IR.
  Term *getTermForDomain(Value value) const {
    auto *term = getOptTermForDomain(value);
    assert(term);
    return term;
  }

  /// Record a mapping from domain in the IR to its corresponding term.
  void setTermForDomain(Value value, Term *term) {
    assert(isa<DomainType>(value.getType()));
    assert(term);
    assert(!termTable.contains(value));
    termTable.insert({value, term});
  }

  /// For a hardware value, get the term which represents the row of associated
  /// domains. If no mapping has been defined, returns nullptr.
  Term *getOptDomainAssociation(Value value) const {
    assert(isa<FIRRTLBaseType>(value.getType()));
    auto it = associationTable.find(value);
    if (it == associationTable.end())
      return nullptr;
    return find(it->second);
  }

  /// For a hardware value, get the term which represents the row of associated
  /// domains.
  Term *getDomainAssociation(Value value) const {
    auto *term = getOptDomainAssociation(value);
    assert(term);
    return term;
  }

  /// Record a mapping from a hardware value in the IR to a term which
  /// represents the row of domains it is associated with.
  void setDomainAssociation(Value value, Term *term) {
    assert(isa<FIRRTLBaseType>(value.getType()));
    assert(term);
    term = find(term);
    associationTable.insert({value, term});
  }

private:
  /// Map from domains in the IR to their underlying term.
  DenseMap<Value, Term *> termTable;

  /// A map from hardware values to their associated row of domains, as a term.
  DenseMap<Value, Term *> associationTable;
};

//====--------------------------------------------------------------------------
// Module processing: solve for the domain associations of hardware.
//====--------------------------------------------------------------------------

/// Get the corresponding term for a domain in the IR. If we don't know what the
/// term is, then map the domain in the IR to a variable term.
Term *getTermForDomain(TermAllocator &allocator, DomainTable &table,
                       Value value) {
  assert(isa<DomainType>(value.getType()));
  if (auto *term = table.getOptTermForDomain(value))
    return term;
  auto *term = allocator.allocVar();
  table.setTermForDomain(value, term);
  return term;
}

/// Get the row of domains that a hardware value in the IR is associated with.
/// The returned term is forced to be at least a row.
RowTerm *getDomainAssociationAsRow(const DomainInfo &info,
                                   TermAllocator &allocator, DomainTable &table,
                                   Value value) {
  assert(isa<FIRRTLBaseType>(value.getType()));
  auto *term = table.getOptDomainAssociation(value);

  // If the term is unknown, allocate a fresh row and set the association.
  if (!term) {
    auto *row = allocator.allocRow(info.getNumDomains());
    table.setDomainAssociation(value, row);
    return row;
  }

  // If the term is already a row, return it.
  if (auto *row = dyn_cast<RowTerm>(term))
    return row;

  // Otherwise, unify the term with a fresh row of domains.
  if (auto *var = dyn_cast<VariableTerm>(term)) {
    auto *row = allocator.allocRow(info.getNumDomains());
    solve(var, row);
    return row;
  }

  assert(false && "unhandled term type");
  return nullptr;
}

template <typename T>
void emitPortDomainCrossingError(const DomainInfo &info, T op, size_t i,
                                 size_t domainTypeID, Term *term1,
                                 Term *term2) {
  VariableIDTable idTable;

  auto portName = op.getPortNameAttr(i);
  auto portLoc = op.getPortLocation(i);
  auto domainDecl = info.getDomain(domainTypeID);
  auto domainName = domainDecl.getNameAttr();

  auto diag = emitError(portLoc);
  diag << "illegal " << domainName << " crossing in port " << portName;

  auto &note1 = diag.attachNote();
  note1 << "1st instance: ";
  render(info, note1, idTable, term1);

  auto &note2 = diag.attachNote();
  note2 << "2nd instance: ";
  render(info, note2, idTable, term2);
}

/// Emit an error when we fail to infer the concrete domain to drive to a
/// domain port.
template <typename T>
void emitDomainPortInferenceError(T op, size_t i) {
  auto name = op.getPortNameAttr(i);
  auto diag = emitError(op->getLoc());
  auto info = op.getDomainInfo();
  diag << "unable to infer value for undriven domain port " << name;
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

template <typename T>
void emitAmbiguousPortDomainAssociation(T op, size_t i) {}

template <typename T>
void emitMissingPortDomainAssociationError(const DomainInfo &info, T op,
                                           size_t typeID, size_t i) {
  auto domainName = info.getDomain(typeID).getNameAttr();
  auto portName = op.getPortNameAttr(i);
  emitError(op.getPortLocation(i))
      << "missing " << domainName << " association for port " << portName;
}

/// Unify the associated domain rows of two terms.
LogicalResult unifyAssociations(const DomainInfo &info,
                                TermAllocator &allocator, DomainTable &table,
                                Operation *op, Value lhs, Value rhs) {
  LLVM_DEBUG(llvm::errs() << "  unify associations of:\n";
             llvm::errs() << "    lhs=" << lhs << "\n";
             llvm::errs() << "    rhs=" << rhs << "\n";);

  if (!lhs || !rhs)
    return success();

  if (lhs == rhs)
    return success();

  auto *lhsTerm = table.getOptDomainAssociation(lhs);
  auto *rhsTerm = table.getOptDomainAssociation(rhs);

  if (lhsTerm) {
    if (rhsTerm) {
      if (failed(unify(lhsTerm, rhsTerm))) {
        auto diag = op->emitOpError("illegal domain crossing in operation");
        auto &note1 = diag.attachNote(lhs.getLoc());

        note1 << "1st operand has domains: ";
        VariableIDTable idTable;
        render(info, note1, idTable, lhsTerm);

        auto &note2 = diag.attachNote(rhs.getLoc());
        note2 << "2nd operand has domains: ";
        render(info, note2, idTable, rhsTerm);

        return failure();
      }
    }
    table.setDomainAssociation(rhs, lhsTerm);
    return success();
  }

  if (rhsTerm) {
    table.setDomainAssociation(lhs, rhsTerm);
    return success();
  }

  auto *var = allocator.allocVar();
  table.setDomainAssociation(lhs, var);
  table.setDomainAssociation(rhs, var);
  return success();
}

LogicalResult processModulePorts(const DomainInfo &info,
                                 TermAllocator &allocator, DomainTable &table,
                                 FModuleOp module) {
  auto domainInfo = module.getDomainInfoAttr();
  auto numPorts = module.getNumPorts();
  DenseMap<unsigned, unsigned> domainTypeIDTable;
  for (size_t i = 0; i < numPorts; ++i) {
    BlockArgument port = module.getArgument(i);

    if (isa<DomainType>(port.getType())) {
      auto typeID = info.getDomainTypeID(domainInfo, i);
      domainTypeIDTable[i] = typeID;
      if (module.getPortDirection(i) == Direction::In) {
        table.setTermForDomain(port, allocator.allocVal(port));
      }
      continue;
    }

    auto portDomains = getPortDomainAssociation(domainInfo, i);
    if (portDomains.empty())
      continue;

    SmallVector<Term *> elements(info.getNumDomains());
    for (auto domainPortIndexAttr : portDomains) {
      auto domainPortIndex = domainPortIndexAttr.getUInt();
      auto domainTypeID = domainTypeIDTable[domainPortIndex];
      auto domainValue = module.getArgument(domainPortIndex);
      auto *term = getTermForDomain(allocator, table, domainValue);
      auto &slot = elements[domainTypeID];
      if (failed(unify(slot, term))) {
        emitPortDomainCrossingError(info, module, i, domainTypeID, slot, term);
        return failure();
      }
      elements[domainTypeID] = term;
    }
    auto *row = allocator.allocRow(elements);
    table.setDomainAssociation(port, row);
  }

  return success();
}

template <typename T>
LogicalResult processInstancePorts(const DomainInfo &info,
                                   TermAllocator &allocator, DomainTable &table,
                                   T op) {
                                    llvm::errs() << "ins=" << op << "\n";
  auto numDomainTypes = info.getNumDomains();
  DenseMap<unsigned, unsigned> domainPortTypeIDTable;
  auto domainInfo = op.getDomainInfoAttr();
  for (size_t i = 0, e = op->getNumResults(); i < e; ++i) {
    Value port = op.getResult(i);

    if (isa<DomainType>(port.getType())) {
      auto typeID = info.getDomainTypeID(domainInfo, i);
      domainPortTypeIDTable[i] = typeID;
      if (op.getPortDirection(i) == Direction::Out) {
        table.setTermForDomain(port, allocator.allocVal(port));
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
      auto *term =
          getTermForDomain(allocator, table, op.getResult(domainPortIndex));
      elements[typeID] = term;
    }

    // Confirm that we have complete domain information for the port. We can be
    // missing information if, for example, this was an instance of an
    // extmodule.
    for (size_t domainTypeID = 0; domainTypeID < numDomainTypes;
         ++domainTypeID) {
      if (elements[domainTypeID])
        continue;
      auto domainDecl = info.getDomain(domainTypeID);
      auto domainName = domainDecl.getNameAttr();
      auto portName = op.getPortNameAttr(i);
      op->emitOpError() << "missing " << domainName << " association for port "
                        << portName;
      return failure();
    }

    table.setDomainAssociation(port, allocator.allocRow(elements));
  }

  return success();
}

LogicalResult processOp(const DomainInfo &info, TermAllocator &allocator,
                        DomainTable &table,
                        const ModuleUpdateTable &updateTable, InstanceOp op) {
  auto module = op.getReferencedModuleNameAttr();
  auto lookup = updateTable.find(module);
  if (lookup != updateTable.end())
    op = fixInstancePorts(op, lookup->second);
  return processInstancePorts(info, allocator, table, op);
}

LogicalResult processOp(const DomainInfo &info, TermAllocator &allocator,
                        DomainTable &table,
                        const ModuleUpdateTable &updateTable,
                        InstanceChoiceOp op) {
  auto module = op.getDefaultTargetAttr().getAttr();
  auto lookup = updateTable.find(module);
  if (lookup != updateTable.end())
    op = fixInstancePorts(op, lookup->second);
  return processInstancePorts(info, allocator, table, op);
}

LogicalResult processOp(const DomainInfo &info, TermAllocator &allocator,
                        DomainTable &table, UnsafeDomainCastOp op) {
  auto domains = op.getDomains();
  if (domains.empty())
    return unifyAssociations(info, allocator, table, op, op.getInput(),
                             op.getResult());

  auto input = op.getInput();
  RowTerm *inputRow = getDomainAssociationAsRow(info, allocator, table, input);
  SmallVector<Term *> elements(inputRow->elements);
  for (auto domain : op.getDomains()) {
    auto typeID = info.getDomainTypeID(domain);
    elements[typeID] = getTermForDomain(allocator, table, domain);
  }

  auto *row = allocator.allocRow(elements);
  table.setDomainAssociation(op.getResult(), row);
  return success();
}

LogicalResult processOp(const DomainInfo &info, TermAllocator &allocator,
                        DomainTable &table, DomainDefineOp op) {
  auto src = op.getSrc();
  auto dst = op.getDest();
  auto *srcTerm = getTermForDomain(allocator, table, src);
  auto *dstTerm = getTermForDomain(allocator, table, dst);
  if (failed(unify(dstTerm, srcTerm))) {
    VariableIDTable idTable;
    auto diag = op->emitOpError("failed to propagate source to destination");
    auto &note1 = diag.attachNote();
    note1 << "destination has underlying value: ";
    render(info, note1, idTable, dstTerm);

    auto &note2 = diag.attachNote(src.getLoc());
    note2 << "source has underlying value: ";
    render(info, note2, idTable, srcTerm);
  }
  return success();
}

LogicalResult processOp(const DomainInfo &info, TermAllocator &allocator,
                        DomainTable &table,
                        const ModuleUpdateTable &updateTable, Operation *op) {
  LLVM_DEBUG(llvm::errs() << "process op: " << *op << "\n");
  if (auto instance = dyn_cast<InstanceOp>(op))
    return processOp(info, allocator, table, updateTable, instance);
  if (auto instance = dyn_cast<InstanceChoiceOp>(op))
    return processOp(info, allocator, table, updateTable, instance);
  if (auto cast = dyn_cast<UnsafeDomainCastOp>(op))
    return processOp(info, allocator, table, cast);
  if (auto def = dyn_cast<DomainDefineOp>(op))
    return processOp(info, allocator, table, def);

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
    if (failed(unifyAssociations(info, allocator, table, op, lhs, rhs)))
      return failure();
    lhs = rhs;
  }
  for (auto rhs : op->getResults()) {
    if (!isa<FIRRTLBaseType>(rhs.getType()))
      continue;
    if (auto *op = rhs.getDefiningOp();
        op && op->hasTrait<OpTrait::ConstantLike>())
      continue;
    if (failed(unifyAssociations(info, allocator, table, op, lhs, rhs)))
      return failure();
    lhs = rhs;
  }
  return success();
}

LogicalResult processModuleBody(const DomainInfo &info,
                                TermAllocator &allocator, DomainTable &table,
                                const ModuleUpdateTable &updateTable,
                                FModuleOp module) {
  LogicalResult result = success();
  module.getBody().walk([&](Operation *op) -> WalkResult {
    if (failed(processOp(info, allocator, table, updateTable, op))) {
      result = failure();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result;
}

/// Populate the domain table by processing the module. If the module has any
/// domain crossing errors, return failure.
LogicalResult processModule(const DomainInfo &info, TermAllocator &allocator,
                            DomainTable &table,
                            const ModuleUpdateTable &updateTable,
                            FModuleOp module) {
  if (failed(processModulePorts(info, allocator, table, module)))
    return failure();

  return processModuleBody(info, allocator, table, updateTable, module);
}

} // namespace

//====--------------------------------------------------------------------------
// Module updating: write the computed domains back to the IR.
//====--------------------------------------------------------------------------

namespace {

using ExportTable = DenseMap<Value, TinyPtrVector<BlockArgument>>;

/// Build a table of exported domains: a map from domains defined internally,
/// to their set of aliasing output ports.
ExportTable initializeExportTable(const DomainTable &table, FModuleOp module) {
  ExportTable exports;
  size_t numPorts = module.getNumPorts();
  for (size_t i = 0; i < numPorts; ++i) {
    auto port = module.getArgument(i);
    auto type = port.getType();
    if (!isa<DomainType>(type))
      continue;
    auto value = table.getOptUnderlyingDomain(port);
    if (value)
      exports[value].push_back(port);
  }

  return exports;
}

/// Copy the domain associations from the module domain info attribute into a
/// small vector.
SmallVector<Attribute> copyPortDomainAssociations(const DomainInfo &info,
                                                  ArrayAttr moduleDomainInfo,
                                                  size_t portIndex) {
  SmallVector<Attribute> result(info.getNumDomains());
  auto oldAssociations = getPortDomainAssociation(moduleDomainInfo, portIndex);
  for (auto domainPortIndexAttr : oldAssociations) {
    auto domainPortIndex = domainPortIndexAttr.getUInt();
    auto domainTypeID = info.getDomainTypeID(moduleDomainInfo, domainPortIndex);
    result[domainTypeID] = domainPortIndexAttr;
  };
  return result;
}

/// Add domain ports for any uninferred domains associated to hardware.
/// Returns the inserted ports, which will be used later to generalize the
/// instances of this module.
///
/// If the port is hardware, we have to check the associated row of
/// domains. If any associated domain is a variable, we solve the variable
/// by generalizing the module with an additional input domain port. If any
/// associated domain is defined internally to the module, we have to add
/// an output domain port, to allow the domain to escape.
void createModuleDomainPorts(const DomainInfo &info, TermAllocator &allocator,
                             DomainTable &table, ExportTable &exportTable,
                             PortInsertions &insertions, FModuleOp module) {
  DenseMap<VariableTerm *, unsigned> pendingSolutions;
  llvm::MapVector<Value, unsigned> pendingExports;
  size_t inserted = 0;
  auto numPorts = module.getNumPorts();
  for (size_t i = 0; i < numPorts; ++i) {
    auto port = module.getArgument(i);
    auto type = port.getType();

    if (!isa<FIRRTLBaseType>(type))
      continue;

    auto *row = getDomainAssociationAsRow(info, allocator, table, port);
    for (auto [typeID, term] : llvm::enumerate(row->elements)) {
      auto *domain = find(term);

      if (auto *val = dyn_cast<ValueTerm>(domain)) {
        auto value = val->value;
        // If the domain value is defined inside the module body, we must output
        // export the domain, so it may appear in the signature of the
        // module.
        if (isPort(module, value))
          continue;

        // The domain is defined internally. If the value is already exported,
        // or will be exported, we are done.
        if (exportTable.contains(value) || pendingExports.contains(value))
          continue;

        // We must insert a new output domain port.
        auto domainDecl = info.getDomain(typeID);
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
        auto domainDecl = info.getDomain(typeID);
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
    auto *solution = allocator.allocVal(port);
    solve(var, solution);
    exportTable[port].push_back(port);
  }

  // Drive the pending exports.
  auto builder = OpBuilder::atBlockEnd(module.getBodyBlock());
  for (auto [value, portIndex] : pendingExports) {
    auto port = module.getArgument(portIndex);
    DomainDefineOp::create(builder, port.getLoc(), port, value);
    exportTable[value].push_back(port);
    table.setTermForDomain(port, allocator.allocVal(value));
  }
}

/// After generalizing the module, all domains should be solved. Reflect the
/// solved domain associations into the port domain info attribute.
LogicalResult updateModuleDomainInfo(const DomainInfo &info,
                                     const DomainTable &table,
                                     const ExportTable &exportTable,
                                     ArrayAttr &result, FModuleOp module) {
  // At this point, all domain variables mentioned in ports have been
  // solved by generalizing the module (adding input domain ports). Now, we have
  // to form the new port domain information for the module by examining the
  // the associated domains of each port.
  auto *context = module.getContext();
  auto numDomains = info.getNumDomains();
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
      // If the output port is not driven, drive it.
      if (module.getPortDirection(i) == Direction::Out && !isDriven(port)) {
        // Get the underlying value of the output port.
        auto *term = table.getOptTermForDomain(port);
        auto *val = llvm::dyn_cast_if_present<ValueTerm>(term);
        if (!val) {
          emitDomainPortInferenceError(module, i);
          return failure();
        }

        auto loc = port.getLoc();
        auto value = val->value;
        DomainDefineOp::create(builder, loc, port, value);
      }

      newModuleDomainInfo[i] = oldModuleDomainInfo[i];
      continue;
    }

    if (isa<FIRRTLBaseType>(type)) {
      auto associations =
          copyPortDomainAssociations(info, oldModuleDomainInfo, i);
      auto *row = cast<RowTerm>(table.getDomainAssociation(port));
      for (size_t domainTypeID = 0; domainTypeID < numDomains; ++domainTypeID) {
        if (associations[domainTypeID])
          continue;

        auto domain = cast<ValueTerm>(find(row->elements[domainTypeID]))->value;
        auto &exports = exportTable.at(domain);
        if (exports.empty()) {
          auto portName = module.getPortNameAttr(i);
          auto portLoc = module.getPortLocation(i);
          auto domainDecl = info.getDomain(domainTypeID);
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
          auto domainDecl = info.getDomain(domainTypeID);
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

    newModuleDomainInfo[i] = ArrayAttr::get(context, {});
  }

  result = ArrayAttr::get(module.getContext(), newModuleDomainInfo);
  module.setDomainInfoAttr(result);
  return success();
}

/// Update the ports of the module and record the change in the module update
/// table.
LogicalResult updateModulePorts(const DomainInfo &info,
                                TermAllocator &allocator, DomainTable &table,
                                ModuleUpdateTable &updateTable, FModuleOp op) {
  // The export table tracks how domains are exported by the ports of the
  // module. Initialize the export table by scanning the current ports.
  auto exportTable = initializeExportTable(table, op);

  // Now, create any necessary domain ports.
  PortInsertions portInsertions;
  createModuleDomainPorts(info, allocator, table, exportTable, portInsertions,
                          op);

  // Update the domain info for the module's ports.
  ArrayAttr portDomainInfo;
  if (failed(
          updateModuleDomainInfo(info, table, exportTable, portDomainInfo, op)))
    return failure();

  // Record the updated interface change in the update table.
  auto &entry = updateTable[op.getModuleNameAttr()];
  entry.portDomainInfo = portDomainInfo;
  entry.portInsertions = portInsertions;
  return success();
}

template <typename T>
LogicalResult updateInstance(const DomainTable &table, T op) {
  auto *context = op.getContext();
  OpBuilder builder(context);
  builder.setInsertionPointAfter(op);
  auto numPorts = op->getNumResults();
  for (size_t i = 0; i < numPorts; ++i) {
    auto port = op.getResult(i);
    auto type = port.getType();
    auto direction = op.getPortDirection(i);
    // If the port is an input domain, we may need to drive the input with
    // a value. If we don't know what value to drive to the port, error.
    if (isa<DomainType>(type) && direction == Direction::In &&
        !isDriven(port)) {
      auto *term = table.getOptTermForDomain(port);
      auto *val = llvm::dyn_cast_if_present<ValueTerm>(term);
      if (!val) {
        emitDomainPortInferenceError(op, i);
        return failure();
      }

      auto loc = port.getLoc();
      auto value = val->value;
      DomainDefineOp::create(builder, loc, port, value);
    }
  }
  return success();
}

LogicalResult updateOp(const DomainTable &table, Operation *op) {
  if (auto instance = dyn_cast<InstanceOp>(op))
    return updateInstance(table, instance);
  if (auto instance = dyn_cast<InstanceChoiceOp>(op))
    return updateInstance(table, instance);
  return success();
}

/// After updating the port domain associations, walk the body of the module
/// to fix up any child instance modules.
LogicalResult updateModuleBody(const DomainTable &table, FModuleOp module) {
  auto result = success();
  module.getBodyBlock()->walk([&](Operation *op) -> WalkResult {
    if (failed(updateOp(table, op))) {
      result = failure();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result;
}

/// Write the domain associations recorded in the domain table back to the IR.
LogicalResult updateModule(const DomainInfo &info, TermAllocator &allocator,
                           DomainTable &table, ModuleUpdateTable &updateTable,
                           FModuleOp op) {
  if (failed(updateModulePorts(info, allocator, table, updateTable, op)))
    return failure();

  if (failed(updateModuleBody(table, op)))
    return failure();

  return success();
}

//====--------------------------------------------------------------------------
// Domain Inference: solve domains and check for correctness,then update the
// IR to reflect the solved domains.
//====--------------------------------------------------------------------------

/// Solve for domains and then write the domain associations back to the IR.
LogicalResult inferModule(const DomainInfo &info,
                          ModuleUpdateTable &updateTable, FModuleOp module) {
  TermAllocator allocator;
  DomainTable table;
  if (failed(processModule(info, allocator, table, updateTable, module)))
    return failure();

  return updateModule(info, allocator, table, updateTable, module);
}

//====--------------------------------------------------------------------------
// Domain Inference+Checking: Check that the interface of the module is fully
// annotated, before proceeding to run domain inference on the body of the
// module.
//====--------------------------------------------------------------------------

/// Check that a module's hardware ports have complete domain associations.
LogicalResult checkModulePorts(const DomainInfo &info, FModuleLike module) {
  auto numDomains = info.getNumDomains();
  auto domainInfo = module.getDomainInfoAttr();
  DenseMap<unsigned, unsigned> typeIDTable;
  for (size_t i = 0, e = module.getNumPorts(); i < e; ++i) {
    auto type = module.getPortType(i);

    if (isa<DomainType>(type)) {
      auto typeID = info.getDomainTypeID(domainInfo, i);
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
          auto domainName = info.getDomain(typeID).getNameAttr();
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
          emitMissingPortDomainAssociationError(info, module, typeID, i);
          return failure();
        }
      }
    }
  }

  return success();
}

/// Check that output domain ports are driven.
LogicalResult checkModuleDomainPortDrivers(const DomainInfo &info,
                                           FModuleOp module) {
  for (size_t i = 0, e = module.getNumPorts(); i < e; ++i) {
    auto port = module.getArgument(i);
    auto type = port.getType();
    if (!isa<DomainType>(type) ||
        module.getPortDirection(i) != Direction::Out || isDriven(port))
      continue;

    auto name = module.getPortNameAttr(i);
    emitError(module.getPortLocation(i)) << "undriven domain port " << name;
    return failure();
  }

  return success();
}

/// Check that the input domain ports are driven.
template <typename T>
LogicalResult checkInstanceDomainPortDrivers(T op) {
  for (size_t i = 0, e = op.getNumResults(); i < e; ++i) {
    auto port = op.getResult(i);
    auto type = port.getType();
    if (!isa<DomainType>(type) || op.getPortDirection(i) != Direction::In ||
        isDriven(port))
      continue;

    auto name = op.getPortNameAttr(i);
    emitError(op.getPortLocation(i)) << "undriven domain port " << name;
    return failure();
  }

  return success();
}

LogicalResult checkOp(Operation *op) {
  if (auto inst = dyn_cast<InstanceOp>(op))
    return checkInstanceDomainPortDrivers(inst);
  if (auto inst = dyn_cast<InstanceChoiceOp>(op))
    return checkInstanceDomainPortDrivers(inst);
  return success();
}

/// Check that instances under this module have driven domain input ports.
LogicalResult checkModuleBody(FModuleOp module) {
  LogicalResult result = success();
  module.getBody().walk([&](Operation *op) -> WalkResult {
    if (failed(checkOp(op))) {
      result = failure();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result;
}

/// Check that a module's ports are fully annotated, before performing domain
/// inference on the module.
LogicalResult checkModule(const DomainInfo &info, FModuleOp module) {
  if (failed(checkModulePorts(info, module)))
    return failure();

  if (failed(checkModuleDomainPortDrivers(info, module)))
    return failure();

  if (failed(checkModuleBody(module)))
    return failure();

  TermAllocator allocator;
  DomainTable table;
  ModuleUpdateTable updateTable;
  return processModule(info, allocator, table, updateTable, module);
}

/// Check that an extmodule's ports are fully annotated.
LogicalResult checkModule(const DomainInfo &info, FExtModuleOp module) {
  return checkModulePorts(info, module);
}

//====--------------------------------------------------------------------------
// Hybrid Mode: Check the interface, then infer.
//====--------------------------------------------------------------------------

/// Check that a module's ports are fully annotated, before performing domain
/// inference on the module. We use this when private module interfaces are
/// inferred but public module interfaces are checked.
LogicalResult checkAndInferModule(const DomainInfo &info,
                                  ModuleUpdateTable &updateTable,
                                  FModuleOp module) {
  if (failed(checkModulePorts(info, module)))
    return failure();

  return inferModule(info, updateTable, module);
}

//===---------------------------------------------------------------------------
// InferDomainsPass: Top-level pass implementation.
//===---------------------------------------------------------------------------

LogicalResult runOnModuleLike(InferDomainsMode mode, const DomainInfo &info,
                              ModuleUpdateTable &updateTable, Operation *op) {
  
                                llvm::errs() << *op << "\n";
                                if (auto module = dyn_cast<FModuleOp>(op)) {
    if (mode == InferDomainsMode::Check)
      return checkModule(info, module);

    if (mode == InferDomainsMode::InferAll || module.isPrivate())
      return inferModule(info, updateTable, module);

    return checkAndInferModule(info, updateTable, module);
  }

  if (auto extModule = dyn_cast<FExtModuleOp>(op))
    return checkModule(info, extModule);

  return success();
}

struct InferDomainsPass
    : public circt::firrtl::impl::InferDomainsBase<InferDomainsPass> {
  using InferDomainsBase::InferDomainsBase;
  void runOnOperation() override {
    CIRCT_DEBUG_SCOPED_PASS_LOGGER(this);
    auto circuit = getOperation();
    auto &instanceGraph = getAnalysis<InstanceGraph>();
    DomainInfo info(circuit);
    DenseSet<InstanceGraphNode *> visited;
    ModuleUpdateTable updateTable;
    auto result = instanceGraph.walkPostOrder([&](auto &node) {
      return runOnModuleLike(mode, info, updateTable, node.getModule());
    });
    if (failed(result))
      signalPassFailure();
  }
};

} // namespace
