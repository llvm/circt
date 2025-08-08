//===- InferDomains.cpp - Infer and Check FIRRTL Domains ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// InferDomains implements FIRRTL domain inference and checking. This pass is a
// bottom-up transform acting on modules. For each moduleOp, we ensure there are
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

/// Return true if the value is a port on the moduleOp.
bool isPort(BlockArgument arg) {
  return isa<FModuleOp>(arg.getOwner()->getParentOp());
}

/// Return true if the value is a port on the moduleOp.
bool isPort(Value value) {
  auto arg = dyn_cast<BlockArgument>(value);
  if (!arg)
    return false;
  return isPort(arg);
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
struct DomainTypeID {
  size_t index;
};

/// Information about the domains in the circuit. Able to map domains to their
/// type ID, which in this pass is the canonical way to reference the type
/// of a domain, as well as provide fast access to domain ops
class DomainInfo {
public:
  DomainInfo(CircuitOp circuit) { processCircuit(circuit); }

  ArrayRef<DomainOp> getDomains() const { return domainTable; }
  size_t getNumDomains() const { return domainTable.size(); }
  DomainOp getDomain(DomainTypeID id) const { return domainTable[id.index]; }
  // DomainOp getDomain(DomainType type) const {
  //   return getDomain(getDomainTypeID(type));
  // }
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
      auto moduleOp = cast<FModuleOp>(owner);
      auto info = moduleOp.getDomainInfoAttr();
      auto i = arg.getArgNumber();
      return getDomainTypeID(info, i);
    }

    auto result = dyn_cast<OpResult>(value);
    auto *owner = result.getOwner();

    auto info = TypeSwitch<Operation *, ArrayAttr>(owner)
                    .Case<InstanceOp, InstanceChoiceOp>(
                        [&](auto inst) { return inst.getDomainInfoAttr(); })
                    .Default([&](auto inst) { return nullptr; });
    assert(info && "unable to obtain domain information from op");

    auto i = result.getResultNumber();
    return getDomainTypeID(info, i);
  }

private:
  void processDomain(DomainOp op) {
    auto index = domainTable.size();
    auto name = op.getNameAttr();
    domainTable.push_back(op);
    typeIDTable.insert({name, {index}});
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

/// Information about the changes made to the interface of a moduleOp, which can
/// be replayed onto an instance.
struct ModuleUpdateInfo {
  /// The updated domain information for a moduleOp.
  ArrayAttr portDomainInfo;
  /// The domain ports which have been inserted into a moduleOp.
  PortInsertions portInsertions;
};

using ModuleUpdateTable = DenseMap<StringAttr, ModuleUpdateInfo>;

/// Apply the port changes of a moduleOp onto an instance-like op.
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
      auto domainOp = info.getDomain(DomainTypeID{i});
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
    for (auto [x, y] : llvm::zip_equal(lhsRow->elements, rhsRow->elements))
      if (failed(unify(x, y)))
        return failure();
    return success();
  }
  return failure();
}

// NOLINTNEXTLINE(misc-no-recursion)
LogicalResult unify(Term *lhs, Term *rhs) {
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
  [[maybe_unused]] auto result = unify(lhs, rhs);
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

void processDomainDefinition(TermAllocator &allocator, DomainTable &table,
                             Value domain) {
  assert(isa<DomainType>(domain.getType()));
  auto *newTerm = allocator.allocVal(domain);
  auto *oldTerm = table.getOptTermForDomain(domain);
  if (!oldTerm) {
    table.setTermForDomain(domain, newTerm);
    return;
  }

  [[maybe_unused]] auto result = unify(oldTerm, newTerm);
  assert(result.succeeded());
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

void noteLocation(mlir::InFlightDiagnostic &diag, Operation *op) {
  auto &note = diag.attachNote(op->getLoc());
  if (auto mod = dyn_cast<FModuleOp>(op)) {
    note << "in module " << mod.getModuleNameAttr();
    return;
  }
  if (auto mod = dyn_cast<FExtModuleOp>(op)) {
    note << "in extmodule " << mod.getModuleNameAttr();
    return;
  }
  if (auto inst = dyn_cast<InstanceOp>(op)) {
    note << "in instance " << inst.getInstanceNameAttr();
    return;
  }
  if (auto inst = dyn_cast<InstanceChoiceOp>(op)) {
    note << "in instance_choice " << inst.getNameAttr();
    return;
  }

  note << "here";
}

template <typename T>
void emitPortDomainCrossingError(const DomainInfo &info, T op, size_t i,
                                 DomainTypeID domainTypeID, Term *term1,
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

  noteLocation(diag, op);
}

template <typename T>
void emitDuplicatePortDomainError(const DomainInfo &info, T op, size_t i,
                                  DomainTypeID domainTypeID,
                                  IntegerAttr domainPortIndexAttr1,
                                  IntegerAttr domainPortIndexAttr2) {
  VariableIDTable idTable;

  auto portName = op.getPortNameAttr(i);
  auto portLoc = op.getPortLocation(i);
  auto domainDecl = info.getDomain(domainTypeID);
  auto domainName = domainDecl.getNameAttr();
  auto domainPortIndex1 = domainPortIndexAttr1.getUInt();
  auto domainPortIndex2 = domainPortIndexAttr2.getUInt();
  auto domainPortName1 = op.getPortNameAttr(domainPortIndex1);
  auto domainPortName2 = op.getPortNameAttr(domainPortIndex2);
  auto domainPortLoc1 = op.getPortLocation(domainPortIndex1);
  auto domainPortLoc2 = op.getPortLocation(domainPortIndex2);

  auto diag = emitError(portLoc);
  diag << "duplicate " << domainName << " association for port " << portName;
  auto &note1 = diag.attachNote(domainPortLoc1);
  note1 << "associated with " << domainName << " port " << domainPortName1;
  auto &note2 = diag.attachNote(domainPortLoc2);
  note2 << "associated with " << domainName << " port " << domainPortName2;
  noteLocation(diag, op);
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
  noteLocation(diag, op);
}

template <typename T>
void emitAmbiguousPortDomainAssociation(
    const DomainInfo &info, T op,
    const llvm::TinyPtrVector<mlir::BlockArgument> &exports,
    DomainTypeID typeID, size_t i) {
  auto portName = op.getPortNameAttr(i);
  auto portLoc = op.getPortLocation(i);
  auto domainDecl = info.getDomain(typeID);
  auto domainName = domainDecl.getNameAttr();
  auto diag = emitError(portLoc) << "ambiguous " << domainName
                                 << " association for port " << portName;
  for (auto arg : exports) {
    auto name = op.getPortNameAttr(arg.getArgNumber());
    auto loc = op.getPortLocation(arg.getArgNumber());
    diag.attachNote(loc) << "candidate association " << name;
  }
  noteLocation(diag, op);
}

template <typename T>
void emitMissingPortDomainAssociationError(const DomainInfo &info, T op,
                                           DomainTypeID typeID, size_t i) {
  auto domainName = info.getDomain(typeID).getNameAttr();
  auto portName = op.getPortNameAttr(i);
  auto diag = emitError(op.getPortLocation(i))
              << "missing " << domainName << " association for port "
              << portName;
  noteLocation(diag, op);
}

/// Unify the associated domain rows of two terms.
LogicalResult unifyAssociations(const DomainInfo &info,
                                TermAllocator &allocator, DomainTable &table,
                                Operation *op, Value lhs, Value rhs) {
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
                                 FModuleOp moduleOp) {
  auto numDomains = info.getNumDomains();
  auto domainInfo = moduleOp.getDomainInfoAttr();
  auto numPorts = moduleOp.getNumPorts();

  DenseMap<unsigned, DomainTypeID> domainTypeIDTable;
  for (size_t i = 0; i < numPorts; ++i) {
    BlockArgument port = moduleOp.getArgument(i);
    if (!isa<DomainType>(port.getType()))
      continue;

    if (moduleOp.getPortDirection(i) == Direction::In)
      processDomainDefinition(allocator, table, port);

    domainTypeIDTable[i] = info.getDomainTypeID(domainInfo, i);
  }

  for (size_t i = 0; i < numPorts; ++i) {
    BlockArgument port = moduleOp.getArgument(i);
    auto type = type_dyn_cast<FIRRTLBaseType>(port.getType());
    if (!type)
      continue;

    SmallVector<IntegerAttr> associations(numDomains);
    for (auto domainPortIndex : getPortDomainAssociation(domainInfo, i)) {
      auto domainTypeID = domainTypeIDTable.at(domainPortIndex.getUInt());
      auto prevDomainPortIndex = associations[domainTypeID.index];
      if (prevDomainPortIndex) {
        emitDuplicatePortDomainError(info, moduleOp, i, domainTypeID,
                                     prevDomainPortIndex, domainPortIndex);
        return failure();
      }
      associations[domainTypeID.index] = domainPortIndex;
    }

    SmallVector<Term *> elements(numDomains);
    for (size_t domainTypeIndex = 0; domainTypeIndex < numDomains;
         ++domainTypeIndex) {
      auto domainPortIndex = associations[domainTypeIndex];
      if (!domainPortIndex)
        continue;
      auto domainPortValue = moduleOp.getArgument(domainPortIndex.getUInt());
      elements[domainTypeIndex] =
          getTermForDomain(allocator, table, domainPortValue);
    }

    auto *domainAssociations = allocator.allocRow(elements);
    table.setDomainAssociation(port, domainAssociations);
  }

  return success();
}

template <typename T>
LogicalResult processInstancePorts(const DomainInfo &info,
                                   TermAllocator &allocator, DomainTable &table,
                                   T op) {
  auto numDomains = info.getNumDomains();
  auto domainInfo = op.getDomainInfoAttr();
  auto numPorts = op.getNumPorts();

  DenseMap<unsigned, DomainTypeID> domainTypeIDTable;
  for (size_t i = 0; i < numPorts; ++i) {
    Value port = op.getResult(i);
    if (!isa<DomainType>(port.getType()))
      continue;

    if (op.getPortDirection(i) == Direction::Out)
      processDomainDefinition(allocator, table, port);

    domainTypeIDTable[i] = info.getDomainTypeID(domainInfo, i);
  }

  for (size_t i = 0; i < numPorts; ++i) {
    Value port = op.getResult(i);
    auto type = type_dyn_cast<FIRRTLBaseType>(port.getType());
    if (!type)
      continue;

    SmallVector<IntegerAttr> associations(numDomains);
    for (auto domainPortIndex : getPortDomainAssociation(domainInfo, i)) {
      auto domainTypeID = domainTypeIDTable.at(domainPortIndex.getUInt());
      auto prevDomainPortIndex = associations[domainTypeID.index];
      if (prevDomainPortIndex) {
        emitDuplicatePortDomainError(info, op, i, domainTypeID,
                                     prevDomainPortIndex, domainPortIndex);
        return failure();
      }
      associations[domainTypeID.index] = domainPortIndex;
    }

    SmallVector<Term *> elements(numDomains);
    for (size_t domainTypeIndex = 0; domainTypeIndex < numDomains;
         ++domainTypeIndex) {
      auto domainPortIndex = associations[domainTypeIndex];
      if (!domainPortIndex)
        continue;
      auto domainPortValue = op.getResult(domainPortIndex.getUInt());
      elements[domainTypeIndex] =
          getTermForDomain(allocator, table, domainPortValue);
    }

    auto *domainAssociations = allocator.allocRow(elements);
    table.setDomainAssociation(port, domainAssociations);
  }

  return success();
}

LogicalResult processOp(const DomainInfo &info, TermAllocator &allocator,
                        DomainTable &table,
                        const ModuleUpdateTable &updateTable, InstanceOp op) {
  auto moduleOp = op.getReferencedModuleNameAttr();
  auto lookup = updateTable.find(moduleOp);
  if (lookup != updateTable.end())
    op = fixInstancePorts(op, lookup->second);
  return processInstancePorts(info, allocator, table, op);
}

LogicalResult processOp(const DomainInfo &info, TermAllocator &allocator,
                        DomainTable &table,
                        const ModuleUpdateTable &updateTable,
                        InstanceChoiceOp op) {
  auto moduleOp = op.getDefaultTargetAttr().getAttr();
  auto lookup = updateTable.find(moduleOp);
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
    elements[typeID.index] = getTermForDomain(allocator, table, domain);
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
  if (succeeded(unify(dstTerm, srcTerm)))
    return success();

  VariableIDTable idTable;
  auto diag = op->emitOpError("failed to propagate source to destination");
  auto &note1 = diag.attachNote();
  note1 << "destination has underlying value: ";
  render(info, note1, idTable, dstTerm);

  auto &note2 = diag.attachNote(src.getLoc());
  note2 << "source has underlying value: ";
  render(info, note2, idTable, srcTerm);
  return failure();
}

LogicalResult processOp(const DomainInfo &info, TermAllocator &allocator,
                        DomainTable &table,
                        const ModuleUpdateTable &updateTable, Operation *op) {
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
                                FModuleOp moduleOp) {
  auto result = moduleOp.getBody().walk([&](Operation *op) -> WalkResult {
    return processOp(info, allocator, table, updateTable, op);
  });
  return failure(result.wasInterrupted());
}

/// Populate the domain table by processing the moduleOp. If the moduleOp has any
/// domain crossing errors, return failure.
LogicalResult processModule(const DomainInfo &info, TermAllocator &allocator,
                            DomainTable &table,
                            const ModuleUpdateTable &updateTable,
                            FModuleOp moduleOp) {
  if (failed(processModulePorts(info, allocator, table, moduleOp)))
    return failure();

  return processModuleBody(info, allocator, table, updateTable, moduleOp);
}

//===---------------------------------------------------------------------------
// ExportTable
//===---------------------------------------------------------------------------

/// A map from domain IR values defined internal to the moduleOp, to ports that
/// alias that domain. These ports make the domain useable as associations of
/// ports, and we say these are exporting ports.
using ExportTable = DenseMap<Value, TinyPtrVector<BlockArgument>>;

/// Build a table of exported domains: a map from domains defined internally,
/// to their set of aliasing output ports.
ExportTable initializeExportTable(const DomainTable &table, FModuleOp moduleOp) {
  ExportTable exports;
  size_t numPorts = moduleOp.getNumPorts();
  for (size_t i = 0; i < numPorts; ++i) {
    auto port = moduleOp.getArgument(i);
    auto type = port.getType();
    if (!isa<DomainType>(type))
      continue;
    auto value = table.getOptUnderlyingDomain(port);
    if (value)
      exports[value].push_back(port);
  }

  return exports;
}

//====--------------------------------------------------------------------------
// Updating: write domains back to the IR.
//====--------------------------------------------------------------------------

/// A map from unsolved variables to a port index, where that port has not yet
/// been created. Eventually we will have an input domain at the port index,
/// which will be the solution to the recorded variable.
using PendingSolutions = DenseMap<VariableTerm *, unsigned>;

/// A map from local domains to an aliasing port index, where that port has not
/// yet been created. Eventually we will be exporting the domain value at the
/// port index.
using PendingExports = llvm::MapVector<Value, unsigned>;

struct PendingUpdates {
  PortInsertions insertions;
  PendingSolutions solutions;
  PendingExports exports;
};

/// If `var` is not solved, solve it by recording a pending input port at
/// the indicated insertion point.
void ensureSolved(const DomainInfo &info, Namespace &ns, DomainTypeID typeID,
                  size_t ip, LocationAttr loc, VariableTerm *var,
                  PendingUpdates &pending) {
  if (pending.solutions.contains(var))
    return;

  auto *context = loc.getContext();
  auto domainDecl = info.getDomain(typeID);
  auto domainName = domainDecl.getNameAttr();

  auto portName = StringAttr::get(context, ns.newName(domainName.getValue()));
  auto portType = DomainType::get(loc.getContext());
  auto portDirection = Direction::In;
  auto portSym = StringAttr();
  auto portLoc = loc;
  auto portAnnos = std::nullopt;
  auto portDomainInfo = FlatSymbolRefAttr::get(domainName);
  PortInfo portInfo(portName, portType, portDirection, portSym, portLoc,
                    portAnnos, portDomainInfo);

  pending.solutions[var] = pending.insertions.size() + ip;
  pending.insertions.push_back({ip, portInfo});
}

/// Ensure that the domain value is available in the signature of the moduleOp,
/// so that subsequent hardware ports may be associated with this domain.
// If the domain is defined internally in the moduleOp, ensure it is aliased by an
/// output port.
void ensureExported(const DomainInfo &info, Namespace &ns,
                    const ExportTable &exports, DomainTypeID typeID, size_t ip,
                    LocationAttr loc, ValueTerm *val, PendingUpdates &pending) {
  auto value = val->value;
  assert(isa<DomainType>(value.getType()));
  if (isPort(value) || exports.contains(value) ||
      pending.exports.contains(value))
    return;

  auto *context = loc.getContext();

  auto domainDecl = info.getDomain(typeID);
  auto domainName = domainDecl.getNameAttr();

  auto portName = StringAttr::get(context, ns.newName(domainName.getValue()));
  auto portType = DomainType::get(loc.getContext());
  auto portDirection = Direction::Out;
  auto portSym = StringAttr();
  auto portLoc = value.getLoc();
  auto portAnnos = std::nullopt;
  auto portDomainInfo = FlatSymbolRefAttr::get(domainName);
  PortInfo portInfo(portName, portType, portDirection, portSym, portLoc,
                    portAnnos, portDomainInfo);
  pending.exports[value] = pending.insertions.size() + ip;
  pending.insertions.push_back({ip, portInfo});
}

void getUpdatesForDomainAssociationOfPort(const DomainInfo &info, Namespace &ns,
                                          PendingUpdates &pending,
                                          DomainTypeID typeID, size_t ip,
                                          LocationAttr loc, Term *term,
                                          const ExportTable &exports) {
  if (auto *var = dyn_cast<VariableTerm>(term)) {
    ensureSolved(info, ns, typeID, ip, loc, var, pending);
    return;
  }
  if (auto *val = dyn_cast<ValueTerm>(term)) {
    ensureExported(info, ns, exports, typeID, ip, loc, val, pending);
    return;
  }
  llvm_unreachable("invalid domain association");
}

void getUpdatesForDomainAssociationOfPort(const DomainInfo &info, Namespace &ns,
                                          const ExportTable &exports, size_t ip,
                                          LocationAttr loc, RowTerm *row,
                                          PendingUpdates &pending) {
  for (auto [index, term] : llvm::enumerate(row->elements))
    getUpdatesForDomainAssociationOfPort(info, ns, pending, DomainTypeID{index},
                                         ip, loc, find(term), exports);
}

void getUpdatesForModulePorts(const DomainInfo &info, TermAllocator &allocator,
                              const ExportTable &exports, DomainTable &table,
                              Namespace &ns, FModuleOp moduleOp,
                              PendingUpdates &pending) {
  for (size_t i = 0, e = moduleOp.getNumPorts(); i < e; ++i) {
    auto port = moduleOp.getArgument(i);
    auto type = port.getType();
    if (!isa<FIRRTLBaseType>(type))
      continue;
    getUpdatesForDomainAssociationOfPort(
        info, ns, exports, i, moduleOp.getPortLocation(i),
        getDomainAssociationAsRow(info, allocator, table, port), pending);
  }
}

void getUpdatesForModule(const DomainInfo &info, TermAllocator &allocator,
                         const ExportTable &exports, DomainTable &table,
                         FModuleOp mod, PendingUpdates &pending) {
  Namespace ns;
  auto names = mod.getPortNamesAttr();
  for (auto name : names.getAsRange<StringAttr>())
    ns.add(name);
  getUpdatesForModulePorts(info, allocator, exports, table, ns, mod, pending);
}

void applyUpdatesToModule(const DomainInfo &info, TermAllocator &allocator,
                          ExportTable &exports, DomainTable &table,
                          FModuleOp moduleOp, const PendingUpdates &pending) {
  // Put the domain ports in place.
  moduleOp.insertPorts(pending.insertions);

  // Solve any variables and record them as "self-exporting".
  for (auto [var, portIndex] : pending.solutions) {
    auto portValue = moduleOp.getArgument(portIndex);
    auto *solution = allocator.allocVal(portValue);
    solve(var, solution);
    exports[portValue].push_back(portValue);
  }

  // Drive the output ports, and record the export.
  auto builder = OpBuilder::atBlockEnd(moduleOp.getBodyBlock());
  for (auto [domainValue, portIndex] : pending.exports) {
    auto portValue = moduleOp.getArgument(portIndex);
    builder.setInsertionPointAfterValue(domainValue);
    DomainDefineOp::create(builder, portValue.getLoc(), portValue, domainValue);

    exports[domainValue].push_back(portValue);
    table.setTermForDomain(portValue, allocator.allocVal(domainValue));
  }
}

/// Copy the domain associations from the moduleOp domain info attribute into a
/// small vector.
SmallVector<Attribute> copyPortDomainAssociations(const DomainInfo &info,
                                                  ArrayAttr moduleDomainInfo,
                                                  size_t portIndex) {
  SmallVector<Attribute> result(info.getNumDomains());
  auto oldAssociations = getPortDomainAssociation(moduleDomainInfo, portIndex);
  for (auto domainPortIndexAttr : oldAssociations) {
    auto domainPortIndex = domainPortIndexAttr.getUInt();
    auto domainTypeID = info.getDomainTypeID(moduleDomainInfo, domainPortIndex);
    result[domainTypeID.index] = domainPortIndexAttr;
  };
  return result;
}

// If the port is an output domain, we may need to drive the output with
// a value. If we don't know what value to drive to the port, error.
LogicalResult driveModuleOutputDomainPorts(const DomainInfo &info,
                                           const DomainTable &table,
                                           FModuleOp moduleOp) {
  auto builder = OpBuilder::atBlockEnd(moduleOp.getBodyBlock());
  for (size_t i = 0, e = moduleOp.getNumPorts(); i < e; ++i) {
    auto port = moduleOp.getArgument(i);
    auto type = port.getType();

    if (!isa<DomainType>(type) || moduleOp.getPortDirection(i) == Direction::In ||
        isDriven(port))
      continue;

    auto *term = table.getOptTermForDomain(port);
    auto *val = llvm::dyn_cast_if_present<ValueTerm>(term);
    if (!val) {
      emitDomainPortInferenceError(moduleOp, i);
      return failure();
    }

    auto loc = port.getLoc();
    auto value = val->value;
    DomainDefineOp::create(builder, loc, port, value);
  }

  return success();
}

/// After generalizing the moduleOp, all domains should be solved. Reflect the
/// solved domain associations into the port domain info attribute.
LogicalResult updateModuleDomainInfo(const DomainInfo &info,
                                     const DomainTable &table,
                                     const ExportTable &exportTable,
                                     ArrayAttr &result, FModuleOp moduleOp) {
  // At this point, all domain variables mentioned in ports have been
  // solved by generalizing the moduleOp (adding input domain ports). Now, we have
  // to form the new port domain information for the moduleOp by examining the
  // the associated domains of each port.
  auto *context = moduleOp.getContext();
  auto numDomains = info.getNumDomains();
  auto oldModuleDomainInfo = moduleOp.getDomainInfoAttr();
  auto numPorts = moduleOp.getNumPorts();
  SmallVector<Attribute> newModuleDomainInfo(numPorts);

  for (size_t i = 0; i < numPorts; ++i) {
    auto port = moduleOp.getArgument(i);
    auto type = port.getType();

    if (isa<DomainType>(type)) {
      newModuleDomainInfo[i] = oldModuleDomainInfo[i];
      continue;
    }

    if (isa<FIRRTLBaseType>(type)) {
      auto associations =
          copyPortDomainAssociations(info, oldModuleDomainInfo, i);
      auto *row = cast<RowTerm>(table.getDomainAssociation(port));
      for (size_t domainIndex = 0; domainIndex < numDomains; ++domainIndex) {
        auto domainTypeID = DomainTypeID{domainIndex};
        if (associations[domainIndex])
          continue;

        auto domain = cast<ValueTerm>(find(row->elements[domainIndex]))->value;
        auto &exports = exportTable.at(domain);
        if (exports.empty()) {
          auto portName = moduleOp.getPortNameAttr(i);
          auto portLoc = moduleOp.getPortLocation(i);
          auto domainDecl = info.getDomain(domainTypeID);
          auto domainName = domainDecl.getNameAttr();
          auto diag = emitError(portLoc)
                      << "private " << domainName << " association for port "
                      << portName;
          diag.attachNote(domain.getLoc()) << "associated domain: " << domain;
          noteLocation(diag, moduleOp);
          return failure();
        }

        if (exports.size() > 1) {
          emitAmbiguousPortDomainAssociation(info, moduleOp, exports,
                                             domainTypeID, i);
          return failure();
        }

        auto argument = cast<BlockArgument>(exports[0]);
        auto domainPortIndex = argument.getArgNumber();
        associations[domainTypeID.index] = IntegerAttr::get(
            IntegerType::get(context, 32, IntegerType::Unsigned),
            domainPortIndex);
      }

      newModuleDomainInfo[i] = ArrayAttr::get(context, associations);
      continue;
    }

    newModuleDomainInfo[i] = ArrayAttr::get(context, {});
  }

  result = ArrayAttr::get(moduleOp.getContext(), newModuleDomainInfo);
  moduleOp.setDomainInfoAttr(result);
  return success();
}

template <typename T>
LogicalResult updateInstance(const DomainInfo &info, TermAllocator &allocator,
                             DomainTable &table, T op) {
  OpBuilder builder(op.getContext());
  builder.setInsertionPointAfter(op);
  auto numPorts = op->getNumResults();
  for (size_t i = 0; i < numPorts; ++i) {
    auto port = op.getResult(i);
    auto type = port.getType();
    auto direction = op.getPortDirection(i);

    // If the port is an input domain, we may need to drive the input with
    // a value. If we don't know what value to drive to the port, drive an
    // anonymous domain.
    if (isa<DomainType>(type) && direction == Direction::In &&
        !isDriven(port)) {
      auto loc = port.getLoc();
      auto *term = getTermForDomain(allocator, table, port);
      if (auto *var = dyn_cast<VariableTerm>(term)) {
        auto name = getDomainPortTypeName(op.getDomainInfo(), i);
        auto anon = DomainCreateAnonOp::create(builder, loc, name);
        solve(var, allocator.allocVal(anon));
        DomainDefineOp::create(builder, loc, port, anon);
        continue;
      }
      if (auto *val = dyn_cast<ValueTerm>(term)) {
        auto value = val->value;
        DomainDefineOp::create(builder, loc, port, value);
        continue;
      }
      llvm_unreachable("unhandled domain term type");
    }
  }

  return success();
}

LogicalResult updateOp(const DomainInfo &info, TermAllocator &allocator,
                       DomainTable &table, Operation *op) {
  if (auto instance = dyn_cast<InstanceOp>(op))
    return updateInstance(info, allocator, table, instance);
  if (auto instance = dyn_cast<InstanceChoiceOp>(op))
    return updateInstance(info, allocator, table, instance);
  return success();
}

/// After updating the port domain associations, walk the body of the moduleOp
/// to fix up any child instance modules.
LogicalResult updateModuleBody(const DomainInfo &info, TermAllocator &allocator,
                               DomainTable &table, FModuleOp moduleOp) {
  auto result = moduleOp.getBodyBlock()->walk([&](Operation *op) -> WalkResult {
    return updateOp(info, allocator, table, op);
  });
  return failure(result.wasInterrupted());
}

/// Write the domain associations recorded in the domain table back to the IR.
LogicalResult updateModule(const DomainInfo &info, TermAllocator &allocator,
                           DomainTable &table, ModuleUpdateTable &updates,
                           FModuleOp op) {
  auto exports = initializeExportTable(table, op);
  PendingUpdates pending;
  getUpdatesForModule(info, allocator, exports, table, op, pending);
  applyUpdatesToModule(info, allocator, exports, table, op, pending);

  // Update the domain info for the moduleOp's ports.
  ArrayAttr portDomainInfo;
  if (failed(updateModuleDomainInfo(info, table, exports, portDomainInfo, op)))
    return failure();

  // Drive output domain ports.
  if (failed(driveModuleOutputDomainPorts(info, table, op)))
    return failure();

  // Record the updated interface change in the update table.
  auto &entry = updates[op.getModuleNameAttr()];
  entry.portDomainInfo = portDomainInfo;
  entry.portInsertions = std::move(pending.insertions);

  return updateModuleBody(info, allocator, table, op);
}

//===---------------------------------------------------------------------------
// Checking: Check that a moduleOp has complete domain information.
//===---------------------------------------------------------------------------

/// Check that a moduleOp's hardware ports have complete domain associations.
LogicalResult checkModulePorts(const DomainInfo &info, FModuleLike moduleOp) {
  auto numDomains = info.getNumDomains();
  auto domainInfo = moduleOp.getDomainInfoAttr();
  auto numPorts = moduleOp.getNumPorts();

  DenseMap<unsigned, DomainTypeID> domainTypeIDTable;
  for (size_t i = 0; i < numPorts; ++i) {
    if (isa<DomainType>(moduleOp.getPortType(i)))
      domainTypeIDTable[i] = info.getDomainTypeID(domainInfo, i);
  }

  for (size_t i = 0; i < numPorts; ++i) {
    auto type = type_dyn_cast<FIRRTLBaseType>(moduleOp.getPortType(i));
    if (!type)
      continue;

    // Record the domain associations of this port.
    SmallVector<IntegerAttr> associations(numDomains);
    for (auto domainPortIndex : getPortDomainAssociation(domainInfo, i)) {
      auto domainTypeID = domainTypeIDTable.at(domainPortIndex.getUInt());
      auto prevDomainPortIndex = associations[domainTypeID.index];
      if (prevDomainPortIndex) {
        emitDuplicatePortDomainError(info, moduleOp, i, domainTypeID,
                                     prevDomainPortIndex, domainPortIndex);
        return failure();
      }
      associations[domainTypeID.index] = domainPortIndex;
    }

    // Check the associations for completeness.
    for (size_t domainIndex = 0; domainIndex < numDomains; ++domainIndex) {
      auto typeID = DomainTypeID{domainIndex};
      if (!associations[domainIndex]) {
        emitMissingPortDomainAssociationError(info, moduleOp, typeID, i);
        return failure();
      }
    }
  }

  return success();
}

/// Check that output domain ports are driven.
LogicalResult checkModuleDomainPortDrivers(const DomainInfo &info,
                                           FModuleOp moduleOp) {
  for (size_t i = 0, e = moduleOp.getNumPorts(); i < e; ++i) {
    auto port = moduleOp.getArgument(i);
    auto type = port.getType();
    if (!isa<DomainType>(type) ||
        moduleOp.getPortDirection(i) != Direction::Out || isDriven(port))
      continue;

    auto name = moduleOp.getPortNameAttr(i);
    auto diag = emitError(moduleOp.getPortLocation(i))
                << "undriven domain port " << name;
    noteLocation(diag, moduleOp);
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
    auto diag = emitError(op.getPortLocation(i))
                << "undriven domain port " << name;
    noteLocation(diag, op);
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

/// Check that instances under this moduleOp have driven domain input ports.
LogicalResult checkModuleBody(FModuleOp moduleOp) {
  auto result = moduleOp.getBody().walk(
      [&](Operation *op) -> WalkResult { return checkOp(op); });
  return failure(result.wasInterrupted());
}

//===---------------------------------------------------------------------------
// InferDomainsPass: Top-level pass implementation.
//===---------------------------------------------------------------------------

/// Solve for domains and then write the domain associations back to the IR.
LogicalResult inferModule(const DomainInfo &info, ModuleUpdateTable &updates,
                          FModuleOp moduleOp) {
  TermAllocator allocator;
  DomainTable table;
  if (failed(processModule(info, allocator, table, updates, moduleOp)))
    return failure();

  return updateModule(info, allocator, table, updates, moduleOp);
}

/// Check that a moduleOp's ports are fully annotated, before performing domain
/// inference on the moduleOp.
LogicalResult checkModule(const DomainInfo &info, FModuleOp moduleOp) {
  if (failed(checkModulePorts(info, moduleOp)))
    return failure();

  if (failed(checkModuleDomainPortDrivers(info, moduleOp)))
    return failure();

  if (failed(checkModuleBody(moduleOp)))
    return failure();

  TermAllocator allocator;
  DomainTable table;
  ModuleUpdateTable updateTable;
  return processModule(info, allocator, table, updateTable, moduleOp);
}

/// Check that an extmodule's ports are fully annotated.
LogicalResult checkModule(const DomainInfo &info, FExtModuleOp moduleOp) {
  return checkModulePorts(info, moduleOp);
}

/// Check that a moduleOp's ports are fully annotated, before performing domain
/// inference on the moduleOp. We use this when private moduleOp interfaces are
/// inferred but public moduleOp interfaces are checked.
LogicalResult checkAndInferModule(const DomainInfo &info,
                                  ModuleUpdateTable &updateTable,
                                  FModuleOp moduleOp) {
  if (failed(checkModulePorts(info, moduleOp)))
    return failure();

  TermAllocator allocator;
  DomainTable table;
  if (failed(processModule(info, allocator, table, updateTable, moduleOp)))
    return failure();

  if (failed(driveModuleOutputDomainPorts(info, table, moduleOp)))
    return failure();

  return updateModuleBody(info, allocator, table, moduleOp);
}

LogicalResult runOnModuleLike(InferDomainsMode mode, const DomainInfo &info,
                              ModuleUpdateTable &updateTable, Operation *op) {
  if (auto moduleOp = dyn_cast<FModuleOp>(op)) {
    if (mode == InferDomainsMode::Check)
      return checkModule(info, moduleOp);

    if (mode == InferDomainsMode::InferAll || moduleOp.isPrivate())
      return inferModule(info, updateTable, moduleOp);

    return checkAndInferModule(info, updateTable, moduleOp);
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
    DenseSet<Operation *> errored;
    instanceGraph.walkPostOrder([&](auto &node) {
      auto moduleOp = node.getModule();
      for (auto *inst : node) {
        if (errored.contains(inst->getTarget()->getModule())) {
          errored.insert(moduleOp);
          return;
        }
      }
      if (failed(runOnModuleLike(mode, info, updateTable, node.getModule())))
        errored.insert(moduleOp);
    });
    if (errored.size())
      signalPassFailure();
  }
};

} // namespace
