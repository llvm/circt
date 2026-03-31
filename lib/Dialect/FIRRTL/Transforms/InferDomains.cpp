//===- InferDomains.cpp - Infer and Check FIRRTL Domains ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// InferDomains implements FIRRTL domain inference and checking.  This pass is
// a bottom-up transform acting on modules.  For each moduleOp, we ensure there
// are no domain crossings, and we make explicit the domain associations of
// ports.
//
// This pass does not require that ExpandWhens has run, but it should have run.
// If ExpandWhens has not been run, then duplicate connections will influence
// domain inference and this can result in errors.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/Debug.h"
#include "circt/Support/Namespace.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TinyPtrVector.h"

#define DEBUG_TYPE "firrtl-infer-domains"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_INFERDOMAINS
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

using llvm::concat;
using mlir::AsmState;
using mlir::ReverseIterator;

namespace {
struct VariableTerm;
} // namespace

//====--------------------------------------------------------------------------
// Helpers.
//====--------------------------------------------------------------------------

using DomainValue = mlir::TypedValue<DomainType>;

using PortInsertions = SmallVector<std::pair<unsigned, PortInfo>>;

/// From a domain info attribute, get the row of associated domains for a
/// hardware value at index i.
static auto getPortDomainAssociation(ArrayAttr info, size_t i) {
  if (info.empty())
    return info.getAsRange<IntegerAttr>();
  return cast<ArrayAttr>(info[i]).getAsRange<IntegerAttr>();
}

/// Return true if the value is a port on the module.
static bool isPort(BlockArgument arg) {
  return isa<FModuleOp>(arg.getOwner()->getParentOp());
}

/// Return true if the value is a port on the module.
static bool isPort(Value value) {
  auto arg = dyn_cast<BlockArgument>(value);
  if (!arg)
    return false;
  return isPort(arg);
}

/// Returns true if the value is driven by a connect op.
static bool isDriven(DomainValue port) {
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
namespace {
struct DomainTypeID {
  size_t index;
};
} // namespace

/// Information about the domains in the circuit. Able to map domains to their
/// type ID, which in this pass is the canonical way to reference the type
/// of a domain, as well as provide fast access to domain ops.
namespace {
class DomainInfo {
public:
  DomainInfo(CircuitOp circuit) { processCircuit(circuit); }

  ArrayRef<DomainOp> getDomains() const { return domainTable; }
  size_t getNumDomains() const { return domainTable.size(); }
  DomainOp getDomain(DomainTypeID id) const { return domainTable[id.index]; }

  DomainTypeID getDomainTypeID(Type type) const { return typeIDTable.at(type); }

  DomainTypeID getDomainTypeID(FModuleLike module, size_t i) const {
    return getDomainTypeID(module.getPortType(i));
  }

  DomainTypeID getDomainTypeID(FInstanceLike op, size_t i) const {
    return getDomainTypeID(op->getResult(i).getType());
  }

  DomainTypeID getDomainTypeID(DomainValue value) const {
    return getDomainTypeID(value.getType());
  }

private:
  void processDomain(DomainOp op) {
    auto index = domainTable.size();
    auto domainType = DomainType::getFromDomainOp(op);
    domainTable.push_back(op);
    typeIDTable.insert({domainType, {index}});
  }

  void processCircuit(CircuitOp circuit) {
    for (auto decl : circuit.getOps<DomainOp>())
      processDomain(decl);
  }

  /// A map from domain type ID to op.
  SmallVector<DomainOp> domainTable;

  /// A map from domain type to type ID.
  DenseMap<Type, DomainTypeID> typeIDTable;
};
} // namespace

namespace {
/// A helper for assigning low numeric IDs to variables for user-facing output.
class VariableIDTable {
public:
  size_t get(VariableTerm *term) {
    return table.insert({term, table.size() + 1}).first->second;
  }

private:
  DenseMap<VariableTerm *, size_t> table;
};
} // namespace

namespace {
struct PassGlobals {
  PassGlobals(CircuitOp circuit) : circuit(circuit), domainInfo(circuit) {}

  void dirty() { asmState = nullptr; }

  AsmState &getAsmState() {
    if (!asmState) {
      asmState = std::make_unique<AsmState>(
          circuit, mlir::OpPrintingFlags().assumeVerified());
    }
    return *asmState;
  }

  CircuitOp circuit;
  const DomainInfo domainInfo;
  VariableIDTable variableIDTable;
  std::unique_ptr<AsmState> asmState;
};
} // namespace

//====--------------------------------------------------------------------------
// Terms: Syntax for unifying domain and domain-rows.
//====--------------------------------------------------------------------------

/// The different sorts of terms in the unification engine.
namespace {
enum class TermKind {
  Variable,
  Value,
  Row,
};
} // namespace

/// A term in the unification engine.
namespace {
struct Term {
  constexpr Term(TermKind kind) : kind(kind) {}
  TermKind kind;
};
} // namespace

/// Helper to define a term kind.
namespace {
template <TermKind K>
struct TermBase : Term {
  static bool classof(const Term *term) { return term->kind == K; }
  TermBase() : Term(K) {}
};
} // namespace

/// An unknown value.
namespace {
struct VariableTerm : public TermBase<TermKind::Variable> {
  VariableTerm() : leader(nullptr) {}
  VariableTerm(Term *leader) : leader(leader) {}
  Term *leader;
};
} // namespace

/// A concrete value defined in the IR.
namespace {
struct ValueTerm : public TermBase<TermKind::Value> {
  ValueTerm(DomainValue value) : value(value) {}
  DomainValue value;
};
} // namespace

/// A row of domains.
namespace {
struct RowTerm : public TermBase<TermKind::Row> {
  RowTerm(ArrayRef<Term *> elements) : elements(elements) {}
  ArrayRef<Term *> elements;
};
} // namespace

// NOLINTNEXTLINE(misc-no-recursion)
static Term *find(Term *x) {
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

//====--------------------------------------------------------------------------
// Term Printing.
//====--------------------------------------------------------------------------

template <typename T>
static void render(PassGlobals &globals, Operation *op, T &out) {
  op->print(out, globals.getAsmState());
}

template <typename T>
static void render(PassGlobals &globals, Value value, T &out) {
  if (!value) {
    out << "null";
    return;
  }

  auto [name, _] = getFieldName(value);
  if (name.empty()) {
    llvm::raw_string_ostream os(name);
    value.printAsOperand(os, globals.getAsmState());
  }
  out << name;

  if (auto type = dyn_cast<DomainType>(value.getType()))
    out << " : " << type.getName().getValue();
}

template <typename T>
// NOLINTNEXTLINE(misc-no-recursion)
static void render(PassGlobals &globals, Term *term, T &out) {
  auto &idTable = globals.variableIDTable;
  auto &info = globals.domainInfo;
  if (!term) {
    out << "null";
    return;
  }
  term = find(term);
  if (auto *var = dyn_cast<VariableTerm>(term)) {
    out << "?" << idTable.get(var);
    return;
  }
  if (auto *val = dyn_cast<ValueTerm>(term)) {
    auto value = val->value;
    render(globals, value, out);
    return;
  }
  if (auto *row = dyn_cast<RowTerm>(term)) {
    out << "[";
    llvm::interleaveComma(
        llvm::seq(size_t(0), info.getNumDomains()), out,
        [&](auto i) { render(globals, row->elements[i], out); });
    out << "]";
    return;
  }
  out << "unknown";
}

namespace {
template <typename T>
struct Render {
  PassGlobals &globals;
  T subject;
};
} // namespace

template <typename T>
static Render<T> render(PassGlobals &globals, T &&subject) {
  return Render<T>{globals, std::forward<T>(subject)};
}

template <typename T>
static llvm::raw_ostream &operator<<(llvm::raw_ostream &out, Render<T> r) {
  render(r.globals, r.subject, out);
  return out;
}

//====--------------------------------------------------------------------------
// Term Unification.
//====--------------------------------------------------------------------------

static LogicalResult unify(PassGlobals &globals, Term *lhs, Term *rhs);

static LogicalResult unify(VariableTerm *x, Term *y) {
  assert(!x->leader);
  x->leader = y;
  return success();
}

static LogicalResult unify(ValueTerm *xv, Term *y) {
  if (auto *yv = dyn_cast<VariableTerm>(y)) {
    yv->leader = xv;
    return success();
  }

  if (auto *yv = dyn_cast<ValueTerm>(y))
    return success(xv == yv);

  return failure();
}

// NOLINTNEXTLINE(misc-no-recursion)
static LogicalResult unify(PassGlobals &globals, RowTerm *lhsRow, Term *rhs) {
  if (auto *rhsVar = dyn_cast<VariableTerm>(rhs)) {
    rhsVar->leader = lhsRow;
    return success();
  }
  if (auto *rhsRow = dyn_cast<RowTerm>(rhs)) {
    for (auto [x, y] : llvm::zip_equal(lhsRow->elements, rhsRow->elements))
      if (failed(unify(globals, x, y)))
        return failure();
    return success();
  }
  return failure();
}

// NOLINTNEXTLINE(misc-no-recursion)
static LogicalResult unify(PassGlobals &globals, Term *lhs, Term *rhs) {
  if (!lhs || !rhs)
    return success();
  lhs = find(lhs);
  rhs = find(rhs);
  if (lhs == rhs)
    return success();

  LLVM_DEBUG(llvm::dbgs().indent(6) << "unify " << render(globals, lhs) << " = "
                                    << render(globals, rhs) << "\n");

  if (auto *lhsVar = dyn_cast<VariableTerm>(lhs))
    return unify(lhsVar, rhs);
  if (auto *lhsVal = dyn_cast<ValueTerm>(lhs))
    return unify(lhsVal, rhs);
  if (auto *lhsRow = dyn_cast<RowTerm>(lhs))
    return unify(globals, lhsRow, rhs);
  return failure();
}

static void solve(PassGlobals &globals, Term *lhs, Term *rhs) {
  [[maybe_unused]] auto result = unify(globals, lhs, rhs);
  assert(result.succeeded());
}

//====--------------------------------------------------------------------------
// Term Allocation.
//====--------------------------------------------------------------------------

namespace {
class TermAllocator {
public:
  /// Allocate a row of fresh domain variables.
  [[nodiscard]] RowTerm *allocRow(size_t size) {
    SmallVector<Term *> elements;
    elements.resize(size);
    return allocRow(elements);
  }

  /// Allocate a row of terms.
  [[nodiscard]] RowTerm *allocRow(ArrayRef<Term *> elements) {
    auto ds = allocArray(elements);
    return alloc<RowTerm>(ds);
  }

  /// Allocate a fresh variable.
  [[nodiscard]] VariableTerm *allocVar() { return alloc<VariableTerm>(); }

  /// Allocate a concrete domain.
  [[nodiscard]] ValueTerm *allocVal(DomainValue value) {
    return alloc<ValueTerm>(value);
  }

private:
  template <typename T, typename... Args>
  [[nodiscard]] T *alloc(Args &&...args) {
    static_assert(std::is_base_of_v<Term, T>, "T must be a term");
    return new (allocator) T(std::forward<Args>(args)...);
  }

  [[nodiscard]] ArrayRef<Term *> allocArray(ArrayRef<Term *> elements) {
    auto size = elements.size();
    if (size == 0)
      return {};

    auto *result = allocator.Allocate<Term *>(size);
    llvm::uninitialized_copy(elements, result);
    for (size_t i = 0; i < size; ++i)
      if (!result[i])
        result[i] = alloc<VariableTerm>();

    return ArrayRef(result, size);
  }

  llvm::BumpPtrAllocator allocator;
};
} // namespace

//====--------------------------------------------------------------------------
// DomainTable: A mapping from IR to terms.
//====--------------------------------------------------------------------------

namespace {
/// Tracks domain infomation for IR values.
class DomainTable {
public:
  /// If the domain value is an alias, returns the domain it aliases.
  DomainValue getOptUnderlyingDomain(DomainValue value) const {
    auto *term = getOptTermForDomain(value);
    if (auto *val = llvm::dyn_cast_if_present<ValueTerm>(term))
      return val->value;
    return nullptr;
  }

  /// Get the corresponding term for a domain in the IR, or null if unset.
  Term *getOptTermForDomain(DomainValue value) const {
    assert(isa<DomainType>(value.getType()));
    auto it = termTable.find(value);
    if (it == termTable.end())
      return nullptr;
    return find(it->second);
  }

  /// Get the corresponding term for a domain in the IR.
  Term *getTermForDomain(DomainValue value) const {
    auto *term = getOptTermForDomain(value);
    assert(term);
    return term;
  }

  /// Record a mapping from domain in the IR to its corresponding term.
  void setTermForDomain(PassGlobals &globals, DomainValue value, Term *term) {
    assert(term);
    assert(!termTable.contains(value));
    termTable.insert({value, term});
    LLVM_DEBUG({
      llvm::dbgs().indent(6) << "set " << render(globals, value)
                             << " := " << render(globals, term) << "\n";
    });
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
  void setDomainAssociation(PassGlobals &globals, Value value, Term *term) {
    assert(isa<FIRRTLBaseType>(value.getType()));
    assert(term);
    term = find(term);
    associationTable.insert({value, term});
    LLVM_DEBUG({
      llvm::dbgs().indent(6) << "set domains(" << render(globals, value)
                             << ") := " << render(globals, term) << "\n";
    });
  }

private:
  /// Map from domains in the IR to their underlying term.
  DenseMap<Value, Term *> termTable;

  /// A map from hardware values to their associated row of domains, as a term.
  DenseMap<Value, Term *> associationTable;
};
} // namespace

//====--------------------------------------------------------------------------
// ModulUpdateInfo: rewrites to apply to instance ops.
//====--------------------------------------------------------------------------

/// Information about the changes made to the interface of a moduleOp, which can
/// be replayed onto an instance.
namespace {
struct ModuleUpdateInfo {
  /// The updated domain information for a moduleOp.
  ArrayAttr portDomainInfo;
  /// The domain ports which have been inserted into a moduleOp.
  PortInsertions portInsertions;
};
} // namespace

using ModuleUpdateTable = DenseMap<StringAttr, ModuleUpdateInfo>;

/// Apply the port changes of a moduleOp onto an instance-like op.
static FInstanceLike fixInstancePorts(PassGlobals &globals, FInstanceLike op,
                                      const ModuleUpdateInfo &update) {
  auto clone = op.cloneWithInsertedPortsAndReplaceUses(update.portInsertions);
  clone.setDomainInfoAttr(update.portDomainInfo);
  op->erase();
  globals.dirty();
  LLVM_DEBUG(llvm::dbgs().indent(6)
             << "fixup " << render(globals, clone) << "\n");
  return clone;
}

//====--------------------------------------------------------------------------
// Module processing: solve for the domain associations of hardware.
//====--------------------------------------------------------------------------

/// Get the corresponding term for a domain in the IR. If we don't know what the
/// term is, then map the domain in the IR to a variable term.
static Term *getTermForDomain(PassGlobals &globals, TermAllocator &allocator,
                              DomainTable &table, DomainValue value) {
  assert(isa<DomainType>(value.getType()));
  if (auto *term = table.getOptTermForDomain(value))
    return term;
  auto *term = allocator.allocVar();
  table.setTermForDomain(globals, value, term);
  return term;
}

static void processDomainDefinition(PassGlobals &globals,
                                    TermAllocator &allocator,
                                    DomainTable &table, DomainValue domain) {
  assert(isa<DomainType>(domain.getType()));
  auto *newTerm = allocator.allocVal(domain);
  auto *oldTerm = table.getOptTermForDomain(domain);
  if (!oldTerm) {
    table.setTermForDomain(globals, domain, newTerm);
    return;
  }

  [[maybe_unused]] auto result = unify(globals, oldTerm, newTerm);
  assert(result.succeeded());
}

/// Get the row of domains that a hardware value in the IR is associated with.
/// The returned term is forced to be at least a row.
static RowTerm *getDomainAssociationAsRow(PassGlobals &globals,
                                          TermAllocator &allocator,
                                          DomainTable &table, Value value) {
  assert(isa<FIRRTLBaseType>(value.getType()));
  auto *term = table.getOptDomainAssociation(value);

  // If the term is unknown, allocate a fresh row and set the association.
  if (!term) {
    auto *row = allocator.allocRow(globals.domainInfo.getNumDomains());
    table.setDomainAssociation(globals, value, row);
    return row;
  }

  // If the term is already a row, return it.
  if (auto *row = dyn_cast<RowTerm>(term))
    return row;

  // Otherwise, unify the term with a fresh row of domains.
  if (auto *var = dyn_cast<VariableTerm>(term)) {
    auto *row = allocator.allocRow(globals.domainInfo.getNumDomains());
    solve(globals, var, row);
    return row;
  }

  assert(false && "unhandled term type");
  return nullptr;
}

static void noteLocation(mlir::InFlightDiagnostic &diag, Operation *op) {
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
static void emitDuplicatePortDomainError(PassGlobals &globals, T op, size_t i,
                                         DomainTypeID domainTypeID,
                                         IntegerAttr domainPortIndexAttr1,
                                         IntegerAttr domainPortIndexAttr2) {
  auto portName = op.getPortNameAttr(i);
  auto portLoc = op.getPortLocation(i);
  auto domainDecl = globals.domainInfo.getDomain(domainTypeID);
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
static void emitDomainPortInferenceError(T op, size_t i) {
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
static void emitAmbiguousPortDomainAssociation(
    PassGlobals &globals, T op, const llvm::TinyPtrVector<DomainValue> &exports,
    DomainTypeID typeID, size_t i) {
  auto portName = op.getPortNameAttr(i);
  auto portLoc = op.getPortLocation(i);
  auto domainDecl = globals.domainInfo.getDomain(typeID);
  auto domainName = domainDecl.getNameAttr();
  auto diag = emitError(portLoc) << "ambiguous " << domainName
                                 << " association for port " << portName;
  for (auto e : exports) {
    auto arg = cast<BlockArgument>(e);
    auto name = op.getPortNameAttr(arg.getArgNumber());
    auto loc = op.getPortLocation(arg.getArgNumber());
    diag.attachNote(loc) << "candidate association " << name;
  }
  noteLocation(diag, op);
}

template <typename T>
static void emitMissingPortDomainAssociationError(PassGlobals &globals, T op,
                                                  DomainTypeID typeID,
                                                  size_t i) {
  auto domainName = globals.domainInfo.getDomain(typeID).getNameAttr();
  auto portName = op.getPortNameAttr(i);
  auto diag = emitError(op.getPortLocation(i))
              << "missing " << domainName << " association for port "
              << portName;
  noteLocation(diag, op);
}

/// Unify the associated domain rows of two terms.
static LogicalResult unifyAssociations(PassGlobals &globals,
                                       TermAllocator &allocator,
                                       DomainTable &table, Operation *op,
                                       Value lhs, Value rhs) {
  if (!lhs || !rhs)
    return success();

  if (lhs == rhs)
    return success();

  LLVM_DEBUG({
    llvm::dbgs().indent(6) << "unify domains(" << render(globals, lhs)
                           << ") = domains(" << render(globals, rhs) << ")\n";
  });

  auto *lhsTerm = table.getOptDomainAssociation(lhs);
  auto *rhsTerm = table.getOptDomainAssociation(rhs);

  if (lhsTerm) {
    if (rhsTerm) {
      if (failed(unify(globals, lhsTerm, rhsTerm))) {
        auto diag = op->emitOpError("illegal domain crossing in operation");
        auto &note1 = diag.attachNote(lhs.getLoc());
        note1 << "1st operand has domains: ";
        render(globals, lhsTerm, note1);
        auto &note2 = diag.attachNote(rhs.getLoc());
        note2 << "2nd operand has domains: ";
        render(globals, rhsTerm, note2);
        return failure();
      }
      return success();
    }
    table.setDomainAssociation(globals, rhs, lhsTerm);
    return success();
  }

  if (rhsTerm) {
    table.setDomainAssociation(globals, lhs, rhsTerm);
    return success();
  }

  auto *var = allocator.allocVar();
  table.setDomainAssociation(globals, lhs, var);
  table.setDomainAssociation(globals, rhs, var);
  return success();
}

static LogicalResult processModulePorts(PassGlobals &globals,
                                        TermAllocator &allocator,
                                        DomainTable &table,
                                        FModuleOp moduleOp) {
  auto numDomains = globals.domainInfo.getNumDomains();
  auto domainInfo = moduleOp.getDomainInfoAttr();
  auto numPorts = moduleOp.getNumPorts();

  DenseMap<unsigned, DomainTypeID> domainTypeIDTable;
  for (size_t i = 0; i < numPorts; ++i) {
    auto port = dyn_cast<DomainValue>(moduleOp.getArgument(i));
    if (!port)
      continue;

    LLVM_DEBUG(llvm::dbgs().indent(4)
               << "process port " << render(globals, port) << "\n");

    if (moduleOp.getPortDirection(i) == Direction::In)
      processDomainDefinition(globals, allocator, table, port);

    domainTypeIDTable[i] = globals.domainInfo.getDomainTypeID(moduleOp, i);
  }

  for (size_t i = 0; i < numPorts; ++i) {
    BlockArgument port = moduleOp.getArgument(i);
    auto type = type_dyn_cast<FIRRTLBaseType>(port.getType());
    if (!type)
      continue;

    LLVM_DEBUG(llvm::dbgs().indent(4)
               << "process port " << render(globals, port) << "\n");

    SmallVector<IntegerAttr> associations(numDomains);
    for (auto domainPortIndex : getPortDomainAssociation(domainInfo, i)) {
      auto domainTypeID = domainTypeIDTable.at(domainPortIndex.getUInt());
      auto prevDomainPortIndex = associations[domainTypeID.index];
      if (prevDomainPortIndex) {
        emitDuplicatePortDomainError(globals, moduleOp, i, domainTypeID,
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
      auto domainPortValue =
          cast<DomainValue>(moduleOp.getArgument(domainPortIndex.getUInt()));
      elements[domainTypeIndex] =
          getTermForDomain(globals, allocator, table, domainPortValue);
    }

    auto *domainAssociations = allocator.allocRow(elements);
    table.setDomainAssociation(globals, port, domainAssociations);
  }

  return success();
}

template <typename T>
static LogicalResult processInstancePorts(PassGlobals &globals,
                                          TermAllocator &allocator,
                                          DomainTable &table, T op) {
  auto numDomains = globals.domainInfo.getNumDomains();
  auto domainInfo = op.getDomainInfoAttr();
  auto numPorts = op.getNumPorts();

  DenseMap<unsigned, DomainTypeID> domainTypeIDTable;
  for (size_t i = 0; i < numPorts; ++i) {
    auto port = dyn_cast<DomainValue>(op->getResult(i));
    if (!port)
      continue;

    if (op.getPortDirection(i) == Direction::Out)
      processDomainDefinition(globals, allocator, table, port);

    domainTypeIDTable[i] = globals.domainInfo.getDomainTypeID(op, i);
  }

  for (size_t i = 0; i < numPorts; ++i) {
    Value port = op->getResult(i);
    auto type = type_dyn_cast<FIRRTLBaseType>(port.getType());
    if (!type)
      continue;

    SmallVector<IntegerAttr> associations(numDomains);
    for (auto domainPortIndex : getPortDomainAssociation(domainInfo, i)) {
      auto domainTypeID = domainTypeIDTable.at(domainPortIndex.getUInt());
      auto prevDomainPortIndex = associations[domainTypeID.index];
      if (prevDomainPortIndex) {
        emitDuplicatePortDomainError(globals, op, i, domainTypeID,
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
      auto domainPortValue =
          cast<DomainValue>(op->getResult(domainPortIndex.getUInt()));
      elements[domainTypeIndex] =
          getTermForDomain(globals, allocator, table, domainPortValue);
    }

    auto *domainAssociations = allocator.allocRow(elements);
    table.setDomainAssociation(globals, port, domainAssociations);
  }

  return success();
}

static LogicalResult processOp(PassGlobals &globals, TermAllocator &allocator,
                               DomainTable &table,
                               const ModuleUpdateTable &updateTable,
                               FInstanceLike op) {
  auto moduleName =
      cast<StringAttr>(cast<ArrayAttr>(op.getReferencedModuleNamesAttr())[0]);
  auto lookup = updateTable.find(moduleName);
  if (lookup != updateTable.end())
    op = fixInstancePorts(globals, op, lookup->second);
  return processInstancePorts(globals, allocator, table, op);
}

static LogicalResult processOp(PassGlobals &globals, TermAllocator &allocator,
                               DomainTable &table, UnsafeDomainCastOp op) {
  auto domains = op.getDomains();
  if (domains.empty())
    return unifyAssociations(globals, allocator, table, op, op.getInput(),
                             op.getResult());

  auto input = op.getInput();
  RowTerm *inputRow =
      getDomainAssociationAsRow(globals, allocator, table, input);
  SmallVector<Term *> elements(inputRow->elements);
  for (auto value : op.getDomains()) {
    auto domain = cast<DomainValue>(value);
    auto typeID = globals.domainInfo.getDomainTypeID(domain);
    elements[typeID.index] =
        getTermForDomain(globals, allocator, table, domain);
  }

  auto *row = allocator.allocRow(elements);
  table.setDomainAssociation(globals, op.getResult(), row);
  return success();
}

static LogicalResult processOp(PassGlobals &globals, TermAllocator &allocator,
                               DomainTable &table, DomainDefineOp op) {
  auto src = op.getSrc();
  auto dst = op.getDest();

  auto *srcTerm = getTermForDomain(globals, allocator, table, src);
  auto *dstTerm = getTermForDomain(globals, allocator, table, dst);
  if (succeeded(unify(globals, dstTerm, srcTerm)))
    return success();

  auto diag =
      op->emitOpError()
      << "defines a domain value that was inferred to be a different domain '";
  render(globals, dstTerm, diag);
  diag << "'";

  return failure();
}

static LogicalResult processOp(PassGlobals &globals, TermAllocator &allocator,
                               DomainTable &table, WireOp op) {
  // If the wire has explicit domain operands, seed the domain table with them
  // as constraints. When this op is visited, connections have not yet been
  // processed (wire declarations precede their uses), so the existing row
  // contains only fresh variables that unify unconditionally. Any conflict
  // between an explicit wire domain and a connection's inferred domain is
  // caught later by the connection's own processOp.
  if (op.getDomains().empty())
    return success();

  // Build a row with the explicitly-specified domain slots filled in and set
  // it as the association for this wire result.
  SmallVector<Term *> elements(globals.domainInfo.getNumDomains());
  for (auto domain : op.getDomains()) {
    auto domainValue = cast<DomainValue>(domain);
    auto typeID = globals.domainInfo.getDomainTypeID(domainValue);
    elements[typeID.index] =
        getTermForDomain(globals, allocator, table, domainValue);
  }
  table.setDomainAssociation(globals, op.getResult(),
                             allocator.allocRow(elements));

  return success();
}

static LogicalResult processOp(PassGlobals &globals, TermAllocator &allocator,
                               DomainTable &table,
                               const ModuleUpdateTable &updateTable,
                               Operation *op) {
  LLVM_DEBUG(llvm::dbgs().indent(4)
             << "process " << render(globals, op) << "\n");
  if (auto instance = dyn_cast<FInstanceLike>(op))
    return processOp(globals, allocator, table, updateTable, instance);
  if (auto wireOp = dyn_cast<WireOp>(op))
    return processOp(globals, allocator, table, wireOp);
  if (auto cast = dyn_cast<UnsafeDomainCastOp>(op))
    return processOp(globals, allocator, table, cast);
  if (auto def = dyn_cast<DomainDefineOp>(op))
    return processOp(globals, allocator, table, def);
  if (auto create = dyn_cast<DomainCreateOp>(op)) {
    processDomainDefinition(globals, allocator, table, create);
    return success();
  }
  if (auto createAnon = dyn_cast<DomainCreateAnonOp>(op)) {
    processDomainDefinition(globals, allocator, table, createAnon);
    return success();
  }

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
    if (failed(unifyAssociations(globals, allocator, table, op, lhs, rhs)))
      return failure();
    lhs = rhs;
  }
  for (auto rhs : op->getResults()) {
    if (!isa<FIRRTLBaseType>(rhs.getType()))
      continue;
    if (auto *op = rhs.getDefiningOp();
        op && op->hasTrait<OpTrait::ConstantLike>())
      continue;
    if (failed(unifyAssociations(globals, allocator, table, op, lhs, rhs)))
      return failure();
    lhs = rhs;
  }
  return success();
}

static LogicalResult processModuleBody(PassGlobals &globals,
                                       TermAllocator &allocator,
                                       DomainTable &table,
                                       const ModuleUpdateTable &updateTable,
                                       FModuleOp moduleOp) {
  return failure(moduleOp.getBody()
                     .walk([&](Operation *op) -> WalkResult {
                       return processOp(globals, allocator, table, updateTable,
                                        op);
                     })
                     .wasInterrupted());
}

/// Populate the domain table by processing the moduleOp. If the moduleOp has
/// any domain crossing errors, return failure.
static LogicalResult processModule(PassGlobals &globals,
                                   TermAllocator &allocator, DomainTable &table,
                                   const ModuleUpdateTable &updateTable,
                                   FModuleOp moduleOp) {
  LLVM_DEBUG(llvm::dbgs().indent(2) << "processing:\n");
  if (failed(processModulePorts(globals, allocator, table, moduleOp)))
    return failure();
  if (failed(
          processModuleBody(globals, allocator, table, updateTable, moduleOp)))
    return failure();
  return success();
}

//===---------------------------------------------------------------------------
// ExportTable
//===---------------------------------------------------------------------------

/// A map from domain IR values defined internal to the moduleOp, to ports that
/// alias that domain. These ports make the domain useable as associations of
/// ports, and we say these are exporting ports.
using ExportTable = DenseMap<DomainValue, TinyPtrVector<DomainValue>>;

/// Build a table of exported domains: a map from domains defined internally,
/// to their set of aliasing output ports.
static ExportTable initializeExportTable(PassGlobals &globals,
                                         const DomainTable &table,
                                         FModuleOp moduleOp) {
  ExportTable exports;
  size_t numPorts = moduleOp.getNumPorts();
  for (size_t i = 0; i < numPorts; ++i) {
    auto port = dyn_cast<DomainValue>(moduleOp.getArgument(i));
    if (!port)
      continue;
    auto value = table.getOptUnderlyingDomain(port);
    if (value)
      exports[value].push_back(port);
  }

  LLVM_DEBUG({
    llvm::dbgs().indent(2) << "domain exports:\n";
    for (auto entry : exports) {
      llvm::dbgs().indent(4) << render(globals, entry.first) << " exported as ";
      llvm::interleaveComma(entry.second, llvm::dbgs(), [&](auto e) {
        llvm::dbgs() << render(globals, e);
      });
      llvm::dbgs() << "\n";
    }
  });

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
using PendingExports = llvm::MapVector<DomainValue, unsigned>;

namespace {
struct PendingUpdates {
  PortInsertions insertions;
  PendingSolutions solutions;
  PendingExports exports;
};
} // namespace

/// If `var` is not solved, solve it by recording a pending input port at
/// the indicated insertion point.
static void ensureSolved(PassGlobals &globals, Namespace &ns,
                         DomainTypeID typeID, size_t ip, LocationAttr loc,
                         VariableTerm *var, PendingUpdates &pending) {
  if (pending.solutions.contains(var))
    return;

  auto *context = loc.getContext();
  auto domainDecl = globals.domainInfo.getDomain(typeID);
  auto domainName = domainDecl.getNameAttr();

  auto portName = StringAttr::get(context, ns.newName(domainName.getValue()));
  auto portType = DomainType::getFromDomainOp(domainDecl);
  auto portDirection = Direction::In;
  auto portSym = StringAttr();
  auto portLoc = loc;
  auto portAnnos = std::nullopt;
  // Domain type ports have no associations (domain info is in the type).
  auto portDomainInfo = ArrayAttr::get(context, {});
  PortInfo portInfo(portName, portType, portDirection, portSym, portLoc,
                    portAnnos, portDomainInfo);

  pending.solutions[var] = pending.insertions.size() + ip;
  pending.insertions.push_back({ip, portInfo});
}

/// Ensure that the domain value is available in the signature of the moduleOp,
/// so that subsequent hardware ports may be associated with this domain.
// If the domain is defined internally in the moduleOp, ensure it is aliased by
// an
/// output port.
static void ensureExported(PassGlobals &globals, Namespace &ns,
                           const ExportTable &exports, DomainTypeID typeID,
                           size_t ip, LocationAttr loc, ValueTerm *val,
                           PendingUpdates &pending) {
  auto value = val->value;
  assert(isa<DomainType>(value.getType()));
  if (isPort(value) || exports.contains(value) ||
      pending.exports.contains(value))
    return;

  auto *context = loc.getContext();

  auto domainDecl = globals.domainInfo.getDomain(typeID);
  auto domainName = domainDecl.getNameAttr();

  auto portName = StringAttr::get(context, ns.newName(domainName.getValue()));
  auto portType = DomainType::getFromDomainOp(domainDecl);
  auto portDirection = Direction::Out;
  auto portSym = StringAttr();
  auto portLoc = value.getLoc();
  auto portAnnos = std::nullopt;
  // Domain type ports have no associations (domain info is in the type).
  auto portDomainInfo = ArrayAttr::get(context, {});
  PortInfo portInfo(portName, portType, portDirection, portSym, portLoc,
                    portAnnos, portDomainInfo);
  pending.exports[value] = pending.insertions.size() + ip;
  pending.insertions.push_back({ip, portInfo});
}

static void getUpdatesForDomainAssociationOfPort(PassGlobals &globals,
                                                 Namespace &ns,
                                                 PendingUpdates &pending,
                                                 DomainTypeID typeID, size_t ip,
                                                 LocationAttr loc, Term *term,
                                                 const ExportTable &exports) {
  if (auto *var = dyn_cast<VariableTerm>(term)) {
    ensureSolved(globals, ns, typeID, ip, loc, var, pending);
    return;
  }
  if (auto *val = dyn_cast<ValueTerm>(term)) {
    ensureExported(globals, ns, exports, typeID, ip, loc, val, pending);
    return;
  }
  llvm_unreachable("invalid domain association");
}

static void getUpdatesForDomainAssociationOfPort(
    PassGlobals &globals, Namespace &ns, const ExportTable &exports, size_t ip,
    LocationAttr loc, RowTerm *row, PendingUpdates &pending) {
  for (auto [index, term] : llvm::enumerate(row->elements))
    getUpdatesForDomainAssociationOfPort(globals, ns, pending,
                                         DomainTypeID{index}, ip, loc,
                                         find(term), exports);
}

static void getUpdatesForModulePorts(PassGlobals &globals,
                                     TermAllocator &allocator,
                                     const ExportTable &exports,
                                     DomainTable &table, Namespace &ns,
                                     FModuleOp moduleOp,
                                     PendingUpdates &pending) {
  for (size_t i = 0, e = moduleOp.getNumPorts(); i < e; ++i) {
    auto port = moduleOp.getArgument(i);
    auto type = port.getType();
    if (!isa<FIRRTLBaseType>(type))
      continue;

    getUpdatesForDomainAssociationOfPort(
        globals, ns, exports, i, moduleOp.getPortLocation(i),
        getDomainAssociationAsRow(globals, allocator, table, port), pending);
  }
}

static void getUpdatesForModule(PassGlobals &globals, TermAllocator &allocator,
                                const ExportTable &exports, DomainTable &table,
                                FModuleOp mod, PendingUpdates &pending) {
  Namespace ns;
  auto names = mod.getPortNamesAttr();
  for (auto name : names.getAsRange<StringAttr>())
    ns.add(name);
  getUpdatesForModulePorts(globals, allocator, exports, table, ns, mod,
                           pending);
}

static void applyUpdatesToModule(PassGlobals &globals, TermAllocator &allocator,
                                 ExportTable &exports, DomainTable &table,
                                 FModuleOp moduleOp,
                                 const PendingUpdates &pending) {
  LLVM_DEBUG(llvm::dbgs().indent(2) << "applying updates:\n");
  // Put the domain ports in place.
  moduleOp.insertPorts(pending.insertions);
  globals.dirty();

  // Solve any variables and record them as "self-exporting".
  for (auto [var, portIndex] : pending.solutions) {
    auto portValue = cast<DomainValue>(moduleOp.getArgument(portIndex));
    auto *solution = allocator.allocVal(portValue);
    LLVM_DEBUG(llvm::dbgs().indent(4)
               << "new-input " << render(globals, portValue) << "\n");
    solve(globals, var, solution);
    exports[portValue].push_back(portValue);
  }

  // Drive the output ports, and record the export.
  auto builder = OpBuilder::atBlockEnd(moduleOp.getBodyBlock());
  for (auto [domainValue, portIndex] : pending.exports) {
    auto portValue = cast<DomainValue>(moduleOp.getArgument(portIndex));
    builder.setInsertionPointAfterValue(domainValue);
    DomainDefineOp::create(builder, portValue.getLoc(), portValue, domainValue);
    LLVM_DEBUG(llvm::dbgs().indent(4)
               << "new-output " << render(globals, portValue)
               << " := " << render(globals, domainValue) << "\n");
    exports[domainValue].push_back(portValue);
    table.setTermForDomain(globals, portValue, allocator.allocVal(domainValue));
  }
}

/// Copy the domain associations from the moduleOp domain info attribute into a
/// small vector.
static SmallVector<Attribute>
copyPortDomainAssociations(PassGlobals &globals, FModuleLike moduleOp,
                           ArrayAttr moduleDomainInfo, size_t portIndex) {
  SmallVector<Attribute> result(globals.domainInfo.getNumDomains());
  auto oldAssociations = getPortDomainAssociation(moduleDomainInfo, portIndex);
  for (auto domainPortIndexAttr : oldAssociations) {
    auto domainPortIndex = domainPortIndexAttr.getUInt();
    auto domainTypeID =
        globals.domainInfo.getDomainTypeID(moduleOp, domainPortIndex);
    result[domainTypeID.index] = domainPortIndexAttr;
  };
  return result;
}

// If the port is an output domain, we may need to drive the output with
// a value. If we don't know what value to drive to the port, error.
static LogicalResult driveModuleOutputDomainPorts(PassGlobals &globals,
                                                  const DomainTable &table,
                                                  FModuleOp moduleOp) {
  auto builder = OpBuilder::atBlockEnd(moduleOp.getBodyBlock());
  for (size_t i = 0, e = moduleOp.getNumPorts(); i < e; ++i) {
    auto port = dyn_cast<DomainValue>(moduleOp.getArgument(i));
    if (!port || moduleOp.getPortDirection(i) == Direction::In ||
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
    LLVM_DEBUG(llvm::dbgs().indent(4)
               << "connect " << render(globals, port)
               << " := " << render(globals, value) << "\n");
    DomainDefineOp::create(builder, loc, port, value);
  }

  return success();
}

/// After generalizing the moduleOp, all domains should be solved. Reflect the
/// solved domain associations into the port domain info attribute.
static LogicalResult updateModuleDomainInfo(PassGlobals &globals,
                                            const DomainTable &table,
                                            const ExportTable &exportTable,
                                            ArrayAttr &result,
                                            FModuleOp moduleOp) {
  // At this point, all domain variables mentioned in ports have been
  // solved by generalizing the moduleOp (adding input domain ports). Now, we
  // have to form the new port domain information for the moduleOp by examining
  // the the associated domains of each port.
  auto *context = moduleOp.getContext();
  auto numDomains = globals.domainInfo.getNumDomains();
  auto oldModuleDomainInfo = moduleOp.getDomainInfoAttr();
  auto numPorts = moduleOp.getNumPorts();
  SmallVector<Attribute> newModuleDomainInfo(numPorts);

  for (size_t i = 0; i < numPorts; ++i) {
    auto port = moduleOp.getArgument(i);
    auto type = port.getType();

    if (isa<DomainType>(type)) {
      // Domain type ports have no associations (domain info is in the type).
      newModuleDomainInfo[i] = ArrayAttr::get(context, {});
      continue;
    }

    if (isa<FIRRTLBaseType>(type)) {
      auto associations =
          copyPortDomainAssociations(globals, moduleOp, oldModuleDomainInfo, i);
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
          auto domainDecl = globals.domainInfo.getDomain(domainTypeID);
          auto domainName = domainDecl.getNameAttr();
          auto diag = emitError(portLoc)
                      << "private " << domainName << " association for port "
                      << portName;
          diag.attachNote(domain.getLoc()) << "associated domain: " << domain;
          noteLocation(diag, moduleOp);
          return failure();
        }

        if (exports.size() > 1) {
          emitAmbiguousPortDomainAssociation(globals, moduleOp, exports,
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

static LogicalResult updateInstance(PassGlobals &globals,
                                    TermAllocator &allocator,
                                    DomainTable &table, FInstanceLike op,
                                    OpBuilder &builder) {
  LLVM_DEBUG(llvm::dbgs().indent(4)
             << "update " << render(globals, op) << "\n");
  auto numPorts = op->getNumResults();
  for (size_t i = 0; i < numPorts; ++i) {
    auto port = dyn_cast<DomainValue>(op->getResult(i));
    auto direction = op.getPortDirection(i);

    // If the port is an input domain, we may need to drive the input with
    // a value. If we don't know what value to drive to the port, drive an
    // anonymous domain.
    if (port && direction == Direction::In && !isDriven(port)) {
      auto loc = port.getLoc();
      auto *term = getTermForDomain(globals, allocator, table, port);
      if (auto *var = dyn_cast<VariableTerm>(term)) {
        auto domainType = cast<DomainType>(op->getResult(i).getType());
        auto domainTypeID = globals.domainInfo.getDomainTypeID(domainType);
        auto domainDecl = globals.domainInfo.getDomain(domainTypeID);
        auto name = domainDecl.getNameAttr();
        DomainValue anon;
        {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointAfter(op);
          auto op = DomainCreateAnonOp::create(builder, loc, domainType, name);
          globals.dirty();
          LLVM_DEBUG(llvm::dbgs().indent(6)
                     << "create " << render(globals, op.getOperation())
                     << "\n");
          anon = op;
        }
        solve(globals, var, allocator.allocVal(anon));
        LLVM_DEBUG(llvm::dbgs().indent(6)
                   << "connect " << render(globals, port)
                   << " := " << render(globals, anon) << "\n");
        // Create domain.define at the end of the block to avoid use-before-def.
        DomainDefineOp::create(builder, loc, port, anon);
        continue;
      }
      if (auto *val = dyn_cast<ValueTerm>(term)) {
        auto value = val->value;
        LLVM_DEBUG(llvm::dbgs().indent(6)
                   << "connect " << render(globals, port)
                   << " := " << render(globals, value) << "\n");
        // Create domain.define at the end of the block to avoid use-before-def.
        DomainDefineOp::create(builder, loc, port, value);
        continue;
      }
      llvm_unreachable("unhandled domain term type");
    }
  }

  return success();
}

/// Update a wire operation with inferred domain associations.
static LogicalResult updateWire(PassGlobals &globals, TermAllocator &allocator,
                                DomainTable &table, WireOp wireOp) {
  auto result = wireOp.getResult();
  if (!isa<FIRRTLBaseType>(result.getType()))
    return success();

  // Get the inferred domain associations for this wire.
  auto *term = table.getOptDomainAssociation(result);
  if (!term)
    return success();

  auto *row = dyn_cast<RowTerm>(find(term));
  if (!row)
    return success();

  // Collect the domain values to add as operands.
  SmallVector<Value> domainOperands;
  for (auto *element : llvm::map_range(row->elements, find))
    if (auto *val = dyn_cast_or_null<ValueTerm>(element))
      domainOperands.push_back(val->value);

  // Update the wire's domain operands in place. $domains is the only operand
  // group on WireOp, so setOperands replaces exactly the domain list.
  if (!domainOperands.empty() && wireOp.getDomains().empty())
    wireOp->setOperands(domainOperands);

  return success();
}

/// After updating the port domain associations, walk the body of the moduleOp
/// to fix up any child instance modules and update wires with inferred domains.
static LogicalResult updateModuleBody(PassGlobals &globals,
                                      TermAllocator &allocator,
                                      DomainTable &table, FModuleOp moduleOp) {
  // Set insertion point to end of block so all domain.define operations are
  // created there, avoiding use-before-def issues.
  OpBuilder builder(moduleOp.getContext());
  builder.setInsertionPointToEnd(moduleOp.getBodyBlock());

  // Update instances
  auto instanceResult =
      moduleOp.getBodyBlock()->walk([&](FInstanceLike op) -> WalkResult {
        return updateInstance(globals, allocator, table, op, builder);
      });
  if (instanceResult.wasInterrupted())
    return failure();

  // Update wires with inferred domain associations
  auto wireResult = moduleOp.getBodyBlock()->walk([&](WireOp op) -> WalkResult {
    return updateWire(globals, allocator, table, op);
  });
  return failure(wireResult.wasInterrupted());
}

/// Write the domain associations recorded in the domain table back to the IR.
static LogicalResult updateModule(PassGlobals &globals,
                                  TermAllocator &allocator, DomainTable &table,
                                  ModuleUpdateTable &updates, FModuleOp op) {
  auto exports = initializeExportTable(globals, table, op);
  PendingUpdates pending;
  getUpdatesForModule(globals, allocator, exports, table, op, pending);
  applyUpdatesToModule(globals, allocator, exports, table, op, pending);

  // Update the domain info for the moduleOp's ports.
  ArrayAttr portDomainInfo;
  if (failed(
          updateModuleDomainInfo(globals, table, exports, portDomainInfo, op)))
    return failure();

  // Drive output domain ports.
  if (failed(driveModuleOutputDomainPorts(globals, table, op)))
    return failure();

  // Record the updated interface change in the update table.
  auto &entry = updates[op.getModuleNameAttr()];
  entry.portDomainInfo = portDomainInfo;
  entry.portInsertions = std::move(pending.insertions);

  if (failed(updateModuleBody(globals, allocator, table, op)))
    return failure();

  LLVM_DEBUG({
    llvm::dbgs().indent(2) << "port summary:\n";
    for (auto port : op.getBodyBlock()->getArguments()) {
      llvm::dbgs().indent(4) << render(globals, port);
      auto info =
          cast<ArrayAttr>(op.getDomainInfoAttrForPort(port.getArgNumber()));
      if (info.size()) {
        llvm::dbgs() << " domains [";
        llvm::interleaveComma(
            info.getAsRange<IntegerAttr>(), llvm::dbgs(), [&](auto i) {
              llvm::dbgs() << render(globals, op.getArgument(i.getUInt()));
            });
        llvm::dbgs() << "]";
      }
      llvm::dbgs() << "\n";
    }
  });

  return success();
}

//===---------------------------------------------------------------------------
// Checking: Check that a moduleOp has complete domain information.
//===---------------------------------------------------------------------------

/// Check that a module's hardware ports have complete domain associations.
static LogicalResult checkModulePorts(PassGlobals &globals,
                                      FModuleLike moduleOp) {
  auto numDomains = globals.domainInfo.getNumDomains();
  auto domainInfo = moduleOp.getDomainInfoAttr();
  auto numPorts = moduleOp.getNumPorts();

  DenseMap<unsigned, DomainTypeID> domainTypeIDTable;
  for (size_t i = 0; i < numPorts; ++i) {
    if (isa<DomainType>(moduleOp.getPortType(i)))
      domainTypeIDTable[i] = globals.domainInfo.getDomainTypeID(moduleOp, i);
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
        emitDuplicatePortDomainError(globals, moduleOp, i, domainTypeID,
                                     prevDomainPortIndex, domainPortIndex);
        return failure();
      }
      associations[domainTypeID.index] = domainPortIndex;
    }

    // Check the associations for completeness.
    for (size_t domainIndex = 0; domainIndex < numDomains; ++domainIndex) {
      auto typeID = DomainTypeID{domainIndex};
      if (!associations[domainIndex]) {
        emitMissingPortDomainAssociationError(globals, moduleOp, typeID, i);
        return failure();
      }
    }
  }

  return success();
}

/// Check that output domain ports are driven.
static LogicalResult checkModuleDomainPortDrivers(PassGlobals &globals,
                                                  FModuleOp moduleOp) {
  for (size_t i = 0, e = moduleOp.getNumPorts(); i < e; ++i) {
    auto port = dyn_cast<DomainValue>(moduleOp.getArgument(i));
    if (!port || moduleOp.getPortDirection(i) != Direction::Out ||
        isDriven(port))
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
static LogicalResult checkInstanceDomainPortDrivers(FInstanceLike op) {
  for (size_t i = 0, e = op->getNumResults(); i < e; ++i) {
    auto port = dyn_cast<DomainValue>(op->getResult(i));

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

/// Check that instances under this module have driven domain input ports.
static LogicalResult checkModuleBody(FModuleOp moduleOp) {
  auto result = moduleOp.getBody().walk([](FInstanceLike op) -> WalkResult {
    return checkInstanceDomainPortDrivers(op);
  });
  return failure(result.wasInterrupted());
}

//===---------------------------------------------------------------------------
// Domain Stripping.
//===---------------------------------------------------------------------------

static LogicalResult stripModule(FModuleLike op) {
  WalkResult result = op->walk<mlir::WalkOrder::PostOrder, ReverseIterator>(
      [=](Operation *op) -> WalkResult {
        return TypeSwitch<Operation *, WalkResult>(op)
            .Case<FModuleLike>([](FModuleLike op) {
              auto n = op.getNumPorts();
              BitVector erasures(n);
              for (size_t i = 0; i < n; ++i)
                if (isa<DomainType>(op.getPortType(i)))
                  erasures.set(i);
              op.erasePorts(erasures);
              return WalkResult::advance();
            })
            .Case<DomainDefineOp, DomainCreateAnonOp, DomainCreateOp>(
                [](Operation *op) {
                  op->erase();
                  return WalkResult::advance();
                })
            .Case<DomainSubfieldOp>([](DomainSubfieldOp op) {
              if (!op->use_empty()) {
                OpBuilder builder(op);
                op.replaceAllUsesWith(
                    UnknownValueOp::create(builder, op.getLoc(), op.getType())
                        .getResult());
              }
              op.erase();
              return WalkResult::advance();
            })
            .Case<UnsafeDomainCastOp>([](UnsafeDomainCastOp op) {
              op.replaceAllUsesWith(op.getInput());
              op.erase();
              return WalkResult::advance();
            })
            .Case<WireOp>([](WireOp op) {
              // Erase wires of DomainType
              if (isa<DomainType>(op.getType(0))) {
                op->erase();
                return WalkResult::advance();
              }
              // Erase domain operands from regular wires
              if (!op.getDomains().empty()) {
                op->eraseOperands(0, op.getNumOperands());
              }
              return WalkResult::advance();
            })
            .Case<FInstanceLike>([](auto op) {
              auto n = op.getNumPorts();
              BitVector erasures(n);
              for (size_t i = 0; i < n; ++i)
                if (isa<DomainType>(op->getResult(i).getType()))
                  erasures.set(i);
              op.cloneWithErasedPortsAndReplaceUses(erasures);
              op.erase();
              return WalkResult::advance();
            })
            .Default([](Operation *op) {
              for (auto type :
                   concat<Type>(op->getOperandTypes(), op->getResultTypes())) {
                if (isa<DomainType>(type)) {
                  op->emitOpError("cannot be stripped");
                  return WalkResult::interrupt();
                }
              }
              return WalkResult::advance();
            });
      });
  return failure(result.wasInterrupted());
}

static LogicalResult stripCircuit(MLIRContext *context, CircuitOp circuit) {
  llvm::SmallVector<FModuleLike> modules;
  for (Operation &op : make_early_inc_range(*circuit.getBodyBlock())) {
    TypeSwitch<Operation *, void>(&op)
        .Case<FModuleLike>([&](FModuleLike op) { modules.push_back(op); })
        .Case<DomainOp>([](DomainOp op) { op.erase(); });
  }
  return failableParallelForEach(context, modules, stripModule);
}

//===---------------------------------------------------------------------------
// InferDomainsPass: Top-level pass implementation.
//===---------------------------------------------------------------------------

/// Solve for domains and then write the domain associations back to the IR.
static LogicalResult inferModule(PassGlobals &globals,
                                 ModuleUpdateTable &updates,
                                 FModuleOp moduleOp) {
  LLVM_DEBUG(llvm::dbgs() << "infer: " << moduleOp.getModuleName() << "\n");

  TermAllocator allocator;
  DomainTable table;

  if (failed(processModule(globals, allocator, table, updates, moduleOp)))
    return failure();

  return updateModule(globals, allocator, table, updates, moduleOp);
}

/// Check that a module's ports are fully annotated, before performing domain
/// inference on the module.
static LogicalResult checkModule(PassGlobals &globals, FModuleOp moduleOp) {
  LLVM_DEBUG(llvm::dbgs() << "check: " << moduleOp.getModuleName() << "\n");

  if (failed(checkModulePorts(globals, moduleOp)))
    return failure();

  if (failed(checkModuleDomainPortDrivers(globals, moduleOp)))
    return failure();

  if (failed(checkModuleBody(moduleOp)))
    return failure();

  TermAllocator allocator;
  DomainTable table;
  ModuleUpdateTable updateTable;
  return processModule(globals, allocator, table, updateTable, moduleOp);
}

/// Check that an extmodule's ports are fully annotated.
static LogicalResult checkModule(PassGlobals &globals, FExtModuleOp moduleOp) {
  LLVM_DEBUG(llvm::dbgs() << "check: " << moduleOp.getModuleName() << "\n");
  return checkModulePorts(globals, moduleOp);
}

/// Check that a module's ports are fully annotated, before performing domain
/// inference on the module. We use this when private module interfaces are
/// inferred but public module interfaces are checked.
static LogicalResult checkAndInferModule(PassGlobals &globals,
                                         ModuleUpdateTable &updateTable,
                                         FModuleOp moduleOp) {
  LLVM_DEBUG(llvm::dbgs() << "check/infer: " << moduleOp.getModuleName()
                          << "\n");

  if (failed(checkModulePorts(globals, moduleOp)))
    return failure();

  TermAllocator allocator;
  DomainTable table;
  if (failed(processModule(globals, allocator, table, updateTable, moduleOp)))
    return failure();

  if (failed(driveModuleOutputDomainPorts(globals, table, moduleOp)))
    return failure();

  return updateModuleBody(globals, allocator, table, moduleOp);
}

static LogicalResult runOnModuleLike(InferDomainsMode mode,
                                     PassGlobals &globals,
                                     ModuleUpdateTable &updateTable,
                                     Operation *op) {
  assert(mode != InferDomainsMode::Strip);

  if (auto moduleOp = dyn_cast<FModuleOp>(op)) {
    if (mode == InferDomainsMode::Check)
      return checkModule(globals, moduleOp);

    if (mode == InferDomainsMode::InferAll || moduleOp.isPrivate())
      return inferModule(globals, updateTable, moduleOp);

    return checkAndInferModule(globals, updateTable, moduleOp);
  }

  if (auto extModule = dyn_cast<FExtModuleOp>(op))
    return checkModule(globals, extModule);

  return success();
}

namespace {
struct InferDomainsPass
    : public circt::firrtl::impl::InferDomainsBase<InferDomainsPass> {
  using Base::Base;
  void runOnOperation() override {
    CIRCT_DEBUG_SCOPED_PASS_LOGGER(this);
    auto circuit = getOperation();

    if (mode == InferDomainsMode::Strip) {
      if (failed(stripCircuit(&getContext(), circuit)))
        signalPassFailure();
      return;
    }

    auto &instanceGraph = getAnalysis<InstanceGraph>();
    PassGlobals globals(circuit);
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
      if (failed(runOnModuleLike(mode, globals, updateTable, node.getModule())))
        errored.insert(moduleOp);
    });
    if (errored.size())
      signalPassFailure();
  }
};
} // namespace
