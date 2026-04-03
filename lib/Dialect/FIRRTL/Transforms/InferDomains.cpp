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

namespace {
struct PassGlobals {
  PassGlobals(CircuitOp circuit) : circuit(circuit) { processCircuit(circuit); }

  ArrayRef<DomainOp> getDomains() const { return domainTable; }
  size_t getNumDomains() const { return domainTable.size(); }
  DomainOp getDomain(DomainTypeID id) const { return domainTable[id.index]; }
  DomainTypeID getDomainTypeID(Type type) { return typeIDTable[type]; }

  void dirty() { asmState = nullptr; }
  AsmState &getAsmState() {
    if (!asmState) {
      asmState = std::make_unique<AsmState>(
          circuit, mlir::OpPrintingFlags().assumeVerified());
    }
    return *asmState;
  }

  size_t getVariableID(VariableTerm *term) {
    return variableIDTable.insert({term, variableIDTable.size() + 1})
        .first->second;
  }

  DenseMap<StringAttr, ModuleUpdateInfo> &getModuleUpdateTable() {
    return moduleUpdateTable;
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

  CircuitOp circuit;
  SmallVector<DomainOp> domainTable;
  DenseMap<Type, DomainTypeID> typeIDTable;
  DenseMap<VariableTerm *, size_t> variableIDTable;
  std::unique_ptr<AsmState> asmState;
  DenseMap<StringAttr, ModuleUpdateInfo> moduleUpdateTable;
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

//====--------------------------------------------------------------------------
// Module processing: solve for the domain associations of hardware.
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

/// A map from domain IR values defined internal to the moduleOp, to ports that
/// alias that domain. These ports make the domain useable as associations of
/// ports, and we say these are exporting ports.
using ExportTable = DenseMap<DomainValue, TinyPtrVector<DomainValue>>;

namespace {
class PassState {
public:
  explicit PassState(PassGlobals &globals) : globals(globals) {}

  ArrayRef<DomainOp> getDomains() { return globals.getDomains(); }
  size_t getNumDomains() { return globals.getNumDomains(); }
  DomainOp getDomain(DomainTypeID id) { return globals.getDomain(id); }
  DomainTypeID getDomainTypeID(Type type) {
    return globals.getDomainTypeID(type);
  }
  DomainTypeID getDomainTypeID(FModuleLike module, size_t i) {
    return globals.getDomainTypeID(module.getPortType(i));
  }
  DomainTypeID getDomainTypeID(FInstanceLike op, size_t i) const {
    return globals.getDomainTypeID(op->getResult(i).getType());
  }
  DomainTypeID getDomainTypeID(DomainValue value) const {
    return globals.getDomainTypeID(value.getType());
  }
  auto &getModuleUpdateTable() { return globals.getModuleUpdateTable(); }

  mlir::AsmState &getAsmState() { return globals.getAsmState(); }
  void dirty() { globals.dirty(); }

  template <typename T>
  void render(Operation *op, T &out);
  template <typename T>
  void render(Value value, T &out);
  template <typename T>
  void render(Term *term, T &out);
  template <typename T>
  struct Render;
  template <typename T>
  Render<T> render(T &&subject);

  Term *find(Term *x);
  LogicalResult unify(Term *lhs, Term *rhs);
  LogicalResult unify(VariableTerm *x, Term *y);
  LogicalResult unify(ValueTerm *xv, Term *y);
  LogicalResult unify(RowTerm *lhsRow, Term *rhs);
  void solve(Term *lhs, Term *rhs);

  [[nodiscard]] RowTerm *allocRow(size_t size);
  [[nodiscard]] RowTerm *allocRow(ArrayRef<Term *> elements);
  [[nodiscard]] VariableTerm *allocVar();
  [[nodiscard]] ValueTerm *allocVal(DomainValue value);
  template <typename T, typename... Args>
  T *alloc(Args &&...args);
  ArrayRef<Term *> allocArray(ArrayRef<Term *> elements);

  DomainValue getOptUnderlyingDomain(DomainValue value);
  Term *getOptTermForDomain(DomainValue value);
  Term *getTermForDomain(DomainValue value);
  void setTermForDomain(DomainValue value, Term *term);

  Term *getOptDomainAssociation(Value value);
  Term *getDomainAssociation(Value value);
  void setDomainAssociation(Value value, Term *term);

  void processDomainDefinition(DomainValue domain);
  RowTerm *getDomainAssociationAsRow(Value value);

  void noteLocation(mlir::InFlightDiagnostic &diag, Operation *op);
  template <typename T>
  void emitDuplicatePortDomainError(T op, size_t i, DomainTypeID domainTypeID,
                                    IntegerAttr domainPortIndexAttr1,
                                    IntegerAttr domainPortIndexAttr2);
  template <typename T>
  void emitDomainPortInferenceError(T op, size_t i);
  template <typename T>
  void emitAmbiguousPortDomainAssociation(
      T op, const llvm::TinyPtrVector<DomainValue> &exports,
      DomainTypeID typeID, size_t i);
  template <typename T>
  void emitMissingPortDomainAssociationError(T op, DomainTypeID typeID,
                                             size_t i);

  LogicalResult unifyAssociations(Operation *op, Value lhs, Value rhs);

  LogicalResult processModulePorts(FModuleOp moduleOp);
  template <typename T>
  LogicalResult processInstancePorts(T op);
  FInstanceLike fixInstancePorts(FInstanceLike op,
                                 const ModuleUpdateInfo &update);
  LogicalResult processOp(FInstanceLike op);
  LogicalResult processOp(UnsafeDomainCastOp op);
  LogicalResult processOp(DomainDefineOp op);
  LogicalResult processOp(WireOp op);
  LogicalResult processOp(Operation *op);
  LogicalResult processModuleBody(FModuleOp moduleOp);
  LogicalResult processModule(FModuleOp moduleOp);

  ExportTable initializeExportTable(FModuleOp moduleOp);
  void ensureSolved(Namespace &ns, DomainTypeID typeID, size_t ip,
                    LocationAttr loc, VariableTerm *var,
                    PendingUpdates &pending);
  void ensureExported(Namespace &ns, const ExportTable &exports,
                      DomainTypeID typeID, size_t ip, LocationAttr loc,
                      ValueTerm *val, PendingUpdates &pending);
  void getUpdatesForDomainAssociationOfPort(Namespace &ns,
                                            PendingUpdates &pending,
                                            DomainTypeID typeID, size_t ip,
                                            LocationAttr loc, Term *term,
                                            const ExportTable &exports);
  void getUpdatesForDomainAssociationOfPort(Namespace &ns,
                                            const ExportTable &exports,
                                            size_t ip, LocationAttr loc,
                                            RowTerm *row,
                                            PendingUpdates &pending);
  void getUpdatesForModulePorts(FModuleOp moduleOp, const ExportTable &exports,
                                Namespace &ns, PendingUpdates &pending);
  void getUpdatesForModule(FModuleOp moduleOp, const ExportTable &exports,
                           PendingUpdates &pending);
  void applyUpdatesToModule(FModuleOp moduleOp, ExportTable &exports,
                            const PendingUpdates &pending);
  SmallVector<Attribute> copyPortDomainAssociations(FModuleOp moduleOp,
                                                    ArrayAttr moduleDomainInfo,
                                                    size_t portIndex);
  LogicalResult driveModuleOutputDomainPorts(FModuleOp moduleOp);
  LogicalResult updateModuleDomainInfo(FModuleOp moduleOp,
                                       const ExportTable &exportTable,
                                       ArrayAttr &result);
  LogicalResult updateInstance(FInstanceLike op, OpBuilder &builder);
  LogicalResult updateWire(WireOp wireOp, OpBuilder &builder);
  LogicalResult updateModuleBody(FModuleOp moduleOp);
  LogicalResult updateModule(FModuleOp moduleOp);

  LogicalResult checkModulePorts(FModuleLike moduleOp);
  LogicalResult checkModuleDomainPortDrivers(FModuleOp moduleOp);
  LogicalResult checkInstanceDomainPortDrivers(FInstanceLike op);
  LogicalResult checkModuleBody(FModuleOp moduleOp);

  LogicalResult inferModule(FModuleOp moduleOp);
  LogicalResult checkModule(FModuleOp moduleOp);
  LogicalResult checkModule(FExtModuleOp extModuleOp);
  LogicalResult checkAndInferModule(FModuleOp moduleOp);

private:
  PassGlobals &globals;
  DenseMap<Value, Term *> termTable;
  DenseMap<Value, Term *> associationTable;
  llvm::BumpPtrAllocator allocator;
};
} // namespace

template <typename T>
void PassState::render(Operation *op, T &out) {
  op->print(out, getAsmState());
}

template <typename T>
void PassState::render(Value value, T &out) {
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
void PassState::render(Term *term, T &out) {
  if (!term) {
    out << "null";
    return;
  }
  term = find(term);
  if (auto *var = dyn_cast<VariableTerm>(term)) {
    out << "?" << globals.getVariableID(var);
    return;
  }
  if (auto *val = dyn_cast<ValueTerm>(term)) {
    auto value = val->value;
    render(value, out);
    return;
  }
  if (auto *row = dyn_cast<RowTerm>(term)) {
    out << "[";
    llvm::interleaveComma(llvm::seq(size_t(0), getNumDomains()), out,
                          [&](auto i) { render(row->elements[i], out); });
    out << "]";
    return;
  }
  out << "unknown";
}

template <typename T>
struct PassState::Render {
  PassState *state;
  T subject;
};

template <typename T>
PassState::Render<T> PassState::render(T &&subject) {
  return Render<T>{this, std::forward<T>(subject)};
}

template <typename T>
static llvm::raw_ostream &operator<<(llvm::raw_ostream &out,
                                     PassState::Render<T> r) {
  r.state->render(r.subject, out);
  return out;
}

// NOLINTNEXTLINE(misc-no-recursion)
Term *PassState::find(Term *x) {
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

LogicalResult PassState::unify(VariableTerm *x, Term *y) {
  assert(!x->leader);
  x->leader = y;
  return success();
}

LogicalResult PassState::unify(ValueTerm *xv, Term *y) {
  if (auto *yv = dyn_cast<VariableTerm>(y)) {
    yv->leader = xv;
    return success();
  }

  if (auto *yv = dyn_cast<ValueTerm>(y))
    return success(xv == yv);

  return failure();
}

// NOLINTNEXTLINE(misc-no-recursion)
LogicalResult PassState::unify(RowTerm *lhsRow, Term *rhs) {
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
LogicalResult PassState::unify(Term *lhs, Term *rhs) {
  if (!lhs || !rhs)
    return success();
  lhs = find(lhs);
  rhs = find(rhs);
  if (lhs == rhs)
    return success();

  LLVM_DEBUG(llvm::dbgs().indent(6)
             << "unify " << render(lhs) << " = " << render(rhs) << "\n");

  if (auto *lhsVar = dyn_cast<VariableTerm>(lhs))
    return unify(lhsVar, rhs);
  if (auto *lhsVal = dyn_cast<ValueTerm>(lhs))
    return unify(lhsVal, rhs);
  if (auto *lhsRow = dyn_cast<RowTerm>(lhs))
    return unify(lhsRow, rhs);
  return failure();
}

void PassState::solve(Term *lhs, Term *rhs) {
  [[maybe_unused]] auto result = unify(lhs, rhs);
  assert(result.succeeded());
}

RowTerm *PassState::allocRow(size_t size) {
  SmallVector<Term *> elements;
  elements.resize(size);
  return allocRow(elements);
}

RowTerm *PassState::allocRow(ArrayRef<Term *> elements) {
  auto ds = allocArray(elements);
  return alloc<RowTerm>(ds);
}

VariableTerm *PassState::allocVar() { return alloc<VariableTerm>(); }

ValueTerm *PassState::allocVal(DomainValue value) {
  return alloc<ValueTerm>(value);
}

template <typename T, typename... Args>
T *PassState::alloc(Args &&...args) {
  static_assert(std::is_base_of_v<Term, T>, "T must be a term");
  return new (allocator) T(std::forward<Args>(args)...);
}

ArrayRef<Term *> PassState::allocArray(ArrayRef<Term *> elements) {
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

DomainValue PassState::getOptUnderlyingDomain(DomainValue value) {
  auto *term = getOptTermForDomain(value);
  if (auto *val = llvm::dyn_cast_if_present<ValueTerm>(term))
    return val->value;
  return nullptr;
}

Term *PassState::getOptTermForDomain(DomainValue value) {
  assert(isa<DomainType>(value.getType()));
  auto it = termTable.find(value);
  if (it == termTable.end())
    return nullptr;
  return find(it->second);
}

Term *PassState::getTermForDomain(DomainValue value) {
  assert(isa<DomainType>(value.getType()));
  if (auto *term = getOptTermForDomain(value))
    return term;
  auto *term = allocVar();
  setTermForDomain(value, term);
  return term;
}

void PassState::setTermForDomain(DomainValue value, Term *term) {
  assert(term);
  assert(!termTable.contains(value));
  termTable.insert({value, term});
  LLVM_DEBUG(llvm::dbgs().indent(6)
             << "set " << render(value) << " := " << render(term) << "\n");
}

Term *PassState::getOptDomainAssociation(Value value) {
  assert(isa<FIRRTLBaseType>(value.getType()));
  auto it = associationTable.find(value);
  if (it == associationTable.end())
    return nullptr;
  return find(it->second);
}

Term *PassState::getDomainAssociation(Value value) {
  auto *term = getOptDomainAssociation(value);
  assert(term);
  return term;
}

void PassState::setDomainAssociation(Value value, Term *term) {
  assert(isa<FIRRTLBaseType>(value.getType()));
  assert(term);
  term = find(term);
  associationTable.insert({value, term});
  LLVM_DEBUG({
    llvm::dbgs().indent(6) << "set domains(" << render(value)
                           << ") := " << render(term) << "\n";
  });
}

void PassState::processDomainDefinition(DomainValue domain) {
  assert(isa<DomainType>(domain.getType()));
  auto *newTerm = allocVal(domain);
  auto *oldTerm = getOptTermForDomain(domain);
  if (!oldTerm) {
    setTermForDomain(domain, newTerm);
    return;
  }

  [[maybe_unused]] auto result = unify(oldTerm, newTerm);
  assert(result.succeeded());
}

RowTerm *PassState::getDomainAssociationAsRow(Value value) {
  assert(isa<FIRRTLBaseType>(value.getType()));
  auto *term = getOptDomainAssociation(value);

  // If the term is unknown, allocate a fresh row and set the association.
  if (!term) {
    auto *row = allocRow(getNumDomains());
    setDomainAssociation(value, row);
    return row;
  }

  // If the term is already a row, return it.
  if (auto *row = dyn_cast<RowTerm>(term))
    return row;

  // Otherwise, unify the term with a fresh row of domains.
  if (auto *var = dyn_cast<VariableTerm>(term)) {
    auto *row = allocRow(getNumDomains());
    solve(var, row);
    return row;
  }

  assert(false && "unhandled term type");
  return nullptr;
}

void PassState::noteLocation(mlir::InFlightDiagnostic &diag, Operation *op) {
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
void PassState::emitDuplicatePortDomainError(T op, size_t i,
                                             DomainTypeID domainTypeID,
                                             IntegerAttr domainPortIndexAttr1,
                                             IntegerAttr domainPortIndexAttr2) {
  auto portName = op.getPortNameAttr(i);
  auto portLoc = op.getPortLocation(i);
  auto domainDecl = getDomain(domainTypeID);
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
void PassState::emitDomainPortInferenceError(T op, size_t i) {
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
void PassState::emitAmbiguousPortDomainAssociation(
    T op, const llvm::TinyPtrVector<DomainValue> &exports, DomainTypeID typeID,
    size_t i) {
  auto portName = op.getPortNameAttr(i);
  auto portLoc = op.getPortLocation(i);
  auto domainDecl = getDomain(typeID);
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
void PassState::emitMissingPortDomainAssociationError(T op, DomainTypeID typeID,
                                                      size_t i) {
  auto domainName = getDomain(typeID).getNameAttr();
  auto portName = op.getPortNameAttr(i);
  auto diag = emitError(op.getPortLocation(i))
              << "missing " << domainName << " association for port "
              << portName;
  noteLocation(diag, op);
}

LogicalResult PassState::unifyAssociations(Operation *op, Value lhs,
                                           Value rhs) {
  if (!lhs || !rhs)
    return success();

  if (lhs == rhs)
    return success();

  LLVM_DEBUG({
    llvm::dbgs().indent(6) << "unify domains(" << render(lhs) << ") = domains("
                           << render(rhs) << ")\n";
  });

  auto *lhsTerm = getOptDomainAssociation(lhs);
  auto *rhsTerm = getOptDomainAssociation(rhs);

  if (lhsTerm) {
    if (rhsTerm) {
      if (failed(unify(lhsTerm, rhsTerm))) {
        auto diag = op->emitOpError("illegal domain crossing in operation");
        auto &note1 = diag.attachNote(lhs.getLoc());
        note1 << "1st operand has domains: ";
        render(lhsTerm, note1);
        auto &note2 = diag.attachNote(rhs.getLoc());
        note2 << "2nd operand has domains: ";
        render(rhsTerm, note2);
        return failure();
      }
      return success();
    }
    setDomainAssociation(rhs, lhsTerm);
    return success();
  }

  if (rhsTerm) {
    setDomainAssociation(lhs, rhsTerm);
    return success();
  }

  auto *var = allocVar();
  setDomainAssociation(lhs, var);
  setDomainAssociation(rhs, var);
  return success();
}

LogicalResult PassState::processModulePorts(FModuleOp moduleOp) {
  auto numDomains = getNumDomains();
  auto domainInfo = moduleOp.getDomainInfoAttr();
  auto numPorts = moduleOp.getNumPorts();

  DenseMap<unsigned, DomainTypeID> domainTypeIDTable;
  for (size_t i = 0; i < numPorts; ++i) {
    auto port = dyn_cast<DomainValue>(moduleOp.getArgument(i));
    if (!port)
      continue;

    LLVM_DEBUG(llvm::dbgs().indent(4)
               << "process port " << render(port) << "\n");

    if (moduleOp.getPortDirection(i) == Direction::In)
      processDomainDefinition(port);

    domainTypeIDTable[i] = getDomainTypeID(moduleOp, i);
  }

  for (size_t i = 0; i < numPorts; ++i) {
    BlockArgument port = moduleOp.getArgument(i);
    auto type = type_dyn_cast<FIRRTLBaseType>(port.getType());
    if (!type)
      continue;

    LLVM_DEBUG(llvm::dbgs().indent(4)
               << "process port " << render(port) << "\n");

    SmallVector<IntegerAttr> associations(numDomains);
    for (auto domainPortIndex : getPortDomainAssociation(domainInfo, i)) {
      auto domainTypeID = domainTypeIDTable.at(domainPortIndex.getUInt());
      auto prevDomainPortIndex = associations[domainTypeID.index];
      if (prevDomainPortIndex) {
        emitDuplicatePortDomainError(moduleOp, i, domainTypeID,
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
      elements[domainTypeIndex] = getTermForDomain(domainPortValue);
    }

    auto *domainAssociations = allocRow(elements);
    setDomainAssociation(port, domainAssociations);
  }

  return success();
}

template <typename T>
LogicalResult PassState::processInstancePorts(T op) {
  auto numDomains = getNumDomains();
  auto domainInfo = op.getDomainInfoAttr();
  auto numPorts = op.getNumPorts();

  DenseMap<unsigned, DomainTypeID> domainTypeIDTable;
  for (size_t i = 0; i < numPorts; ++i) {
    auto port = dyn_cast<DomainValue>(op->getResult(i));
    if (!port)
      continue;

    if (op.getPortDirection(i) == Direction::Out)
      processDomainDefinition(port);

    domainTypeIDTable[i] = getDomainTypeID(op, i);
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
        emitDuplicatePortDomainError(op, i, domainTypeID, prevDomainPortIndex,
                                     domainPortIndex);
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
      elements[domainTypeIndex] = getTermForDomain(domainPortValue);
    }

    auto *domainAssociations = allocRow(elements);
    setDomainAssociation(port, domainAssociations);
  }

  return success();
}

FInstanceLike PassState::fixInstancePorts(FInstanceLike op,
                                          const ModuleUpdateInfo &update) {
  auto clone = op.cloneWithInsertedPortsAndReplaceUses(update.portInsertions);
  clone.setDomainInfoAttr(update.portDomainInfo);
  op->erase();
  dirty();
  LLVM_DEBUG(llvm::dbgs().indent(6) << "fixup " << render(clone) << "\n");
  return clone;
}

LogicalResult PassState::processOp(FInstanceLike op) {
  auto moduleName =
      cast<StringAttr>(cast<ArrayAttr>(op.getReferencedModuleNamesAttr())[0]);
  auto updateTable = getModuleUpdateTable();
  auto lookup = updateTable.find(moduleName);
  if (lookup != updateTable.end())
    op = fixInstancePorts(op, lookup->second);
  return processInstancePorts(op);
}

LogicalResult PassState::processOp(UnsafeDomainCastOp op) {
  auto domains = op.getDomains();
  if (domains.empty())
    return unifyAssociations(op, op.getInput(), op.getResult());

  auto input = op.getInput();
  RowTerm *inputRow = getDomainAssociationAsRow(input);
  SmallVector<Term *> elements(inputRow->elements);
  for (auto value : op.getDomains()) {
    auto domain = cast<DomainValue>(value);
    auto typeID = getDomainTypeID(domain);
    elements[typeID.index] = getTermForDomain(domain);
  }

  auto *row = allocRow(elements);
  setDomainAssociation(op.getResult(), row);
  return success();
}

LogicalResult PassState::processOp(DomainDefineOp op) {
  auto src = op.getSrc();
  auto dst = op.getDest();

  auto *srcTerm = getTermForDomain(src);
  auto *dstTerm = getTermForDomain(dst);
  if (succeeded(unify(dstTerm, srcTerm)))
    return success();

  auto diag =
      op->emitOpError()
      << "defines a domain value that was inferred to be a different domain '";
  render(dstTerm, diag);
  diag << "'";

  return failure();
}

LogicalResult PassState::processOp(WireOp op) {
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
  SmallVector<Term *> elements(getNumDomains());
  for (auto domain : op.getDomains()) {
    auto domainValue = cast<DomainValue>(domain);
    auto typeID = getDomainTypeID(domainValue);
    elements[typeID.index] = getTermForDomain(domainValue);
  }
  setDomainAssociation(op.getResult(), allocRow(elements));

  return success();
}

LogicalResult PassState::processOp(Operation *op) {
  LLVM_DEBUG(llvm::dbgs().indent(4) << "process " << render(op) << "\n");
  if (auto instance = dyn_cast<FInstanceLike>(op))
    return processOp(instance);
  if (auto wireOp = dyn_cast<WireOp>(op))
    return processOp(wireOp);
  if (auto cast = dyn_cast<UnsafeDomainCastOp>(op))
    return processOp(cast);
  if (auto def = dyn_cast<DomainDefineOp>(op))
    return processOp(def);
  if (auto create = dyn_cast<DomainCreateOp>(op)) {
    processDomainDefinition(create);
    return success();
  }
  if (auto createAnon = dyn_cast<DomainCreateAnonOp>(op)) {
    processDomainDefinition(createAnon);
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

LogicalResult PassState::processModuleBody(FModuleOp moduleOp) {
  return failure(
      moduleOp.getBody()
          .walk([&](Operation *op) -> WalkResult { return processOp(op); })
          .wasInterrupted());
}

LogicalResult PassState::processModule(FModuleOp moduleOp) {
  LLVM_DEBUG(llvm::dbgs().indent(2) << "processing:\n");
  if (failed(processModulePorts(moduleOp)))
    return failure();
  if (failed(processModuleBody(moduleOp)))
    return failure();
  return success();
}

ExportTable PassState::initializeExportTable(FModuleOp moduleOp) {
  ExportTable exports;
  size_t numPorts = moduleOp.getNumPorts();
  for (size_t i = 0; i < numPorts; ++i) {
    auto port = dyn_cast<DomainValue>(moduleOp.getArgument(i));
    if (!port)
      continue;
    auto value = getOptUnderlyingDomain(port);
    if (value)
      exports[value].push_back(port);
  }

  LLVM_DEBUG({
    llvm::dbgs().indent(2) << "domain exports:\n";
    for (auto entry : exports) {
      llvm::dbgs().indent(4) << render(entry.first) << " exported as ";
      llvm::interleaveComma(entry.second, llvm::dbgs(),
                            [&](auto e) { llvm::dbgs() << render(e); });
      llvm::dbgs() << "\n";
    }
  });

  return exports;
}

void PassState::ensureSolved(Namespace &ns, DomainTypeID typeID, size_t ip,
                             LocationAttr loc, VariableTerm *var,
                             PendingUpdates &pending) {
  if (pending.solutions.contains(var))
    return;

  auto *context = loc.getContext();
  auto domainDecl = getDomain(typeID);
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

void PassState::ensureExported(Namespace &ns, const ExportTable &exports,
                               DomainTypeID typeID, size_t ip, LocationAttr loc,
                               ValueTerm *val, PendingUpdates &pending) {
  auto value = val->value;
  assert(isa<DomainType>(value.getType()));
  if (isPort(value) || exports.contains(value) ||
      pending.exports.contains(value))
    return;

  auto *context = loc.getContext();

  auto domainDecl = getDomain(typeID);
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

void PassState::getUpdatesForDomainAssociationOfPort(
    Namespace &ns, PendingUpdates &pending, DomainTypeID typeID, size_t ip,
    LocationAttr loc, Term *term, const ExportTable &exports) {
  if (auto *var = dyn_cast<VariableTerm>(term)) {
    ensureSolved(ns, typeID, ip, loc, var, pending);
    return;
  }
  if (auto *val = dyn_cast<ValueTerm>(term)) {
    ensureExported(ns, exports, typeID, ip, loc, val, pending);
    return;
  }
  llvm_unreachable("invalid domain association");
}

void PassState::getUpdatesForDomainAssociationOfPort(
    Namespace &ns, const ExportTable &exports, size_t ip, LocationAttr loc,
    RowTerm *row, PendingUpdates &pending) {
  for (auto [index, term] : llvm::enumerate(row->elements))
    getUpdatesForDomainAssociationOfPort(ns, pending, DomainTypeID{index}, ip,
                                         loc, find(term), exports);
}

void PassState::getUpdatesForModulePorts(FModuleOp moduleOp,
                                         const ExportTable &exports,
                                         Namespace &ns,
                                         PendingUpdates &pending) {
  for (size_t i = 0, e = moduleOp.getNumPorts(); i < e; ++i) {
    auto port = moduleOp.getArgument(i);
    auto type = port.getType();
    if (!isa<FIRRTLBaseType>(type))
      continue;

    getUpdatesForDomainAssociationOfPort(
        ns, exports, i, moduleOp.getPortLocation(i),
        getDomainAssociationAsRow(port), pending);
  }
}

void PassState::getUpdatesForModule(FModuleOp moduleOp,
                                    const ExportTable &exports,
                                    PendingUpdates &pending) {
  Namespace ns;
  auto names = moduleOp.getPortNamesAttr();
  for (auto name : names.getAsRange<StringAttr>())
    ns.add(name);
  getUpdatesForModulePorts(moduleOp, exports, ns, pending);
}

void PassState::applyUpdatesToModule(FModuleOp moduleOp, ExportTable &exports,
                                     const PendingUpdates &pending) {
  LLVM_DEBUG(llvm::dbgs().indent(2) << "applying updates:\n");
  // Put the domain ports in place.
  moduleOp.insertPorts(pending.insertions);
  dirty();

  // Solve any variables and record them as "self-exporting".
  for (auto [var, portIndex] : pending.solutions) {
    auto portValue = cast<DomainValue>(moduleOp.getArgument(portIndex));
    auto *solution = allocVal(portValue);
    LLVM_DEBUG(llvm::dbgs().indent(4)
               << "new-input " << render(portValue) << "\n");
    solve(var, solution);
    exports[portValue].push_back(portValue);
  }

  // Drive the output ports, and record the export.
  auto builder = OpBuilder::atBlockEnd(moduleOp.getBodyBlock());
  for (auto [domainValue, portIndex] : pending.exports) {
    auto portValue = cast<DomainValue>(moduleOp.getArgument(portIndex));
    builder.setInsertionPointAfterValue(domainValue);
    DomainDefineOp::create(builder, portValue.getLoc(), portValue, domainValue);
    LLVM_DEBUG(llvm::dbgs().indent(4) << "new-output " << render(portValue)
                                      << " := " << render(domainValue) << "\n");
    exports[domainValue].push_back(portValue);
    setTermForDomain(portValue, allocVal(domainValue));
  }
}

SmallVector<Attribute> PassState::copyPortDomainAssociations(
    FModuleOp moduleOp, ArrayAttr moduleDomainInfo, size_t portIndex) {
  SmallVector<Attribute> result(getNumDomains());
  auto oldAssociations = getPortDomainAssociation(moduleDomainInfo, portIndex);
  for (auto domainPortIndexAttr : oldAssociations) {
    auto domainPortIndex = domainPortIndexAttr.getUInt();
    auto domainTypeID = getDomainTypeID(moduleOp, domainPortIndex);
    result[domainTypeID.index] = domainPortIndexAttr;
  };
  return result;
}

LogicalResult PassState::driveModuleOutputDomainPorts(FModuleOp moduleOp) {
  auto builder = OpBuilder::atBlockEnd(moduleOp.getBodyBlock());
  for (size_t i = 0, e = moduleOp.getNumPorts(); i < e; ++i) {
    auto port = dyn_cast<DomainValue>(moduleOp.getArgument(i));
    if (!port || moduleOp.getPortDirection(i) == Direction::In ||
        isDriven(port))
      continue;

    auto *term = getOptTermForDomain(port);
    auto *val = llvm::dyn_cast_if_present<ValueTerm>(term);
    if (!val) {
      emitDomainPortInferenceError(moduleOp, i);
      return failure();
    }

    auto loc = port.getLoc();
    auto value = val->value;
    LLVM_DEBUG(llvm::dbgs().indent(4) << "connect " << render(port)
                                      << " := " << render(value) << "\n");
    DomainDefineOp::create(builder, loc, port, value);
  }

  return success();
}

LogicalResult PassState::updateModuleDomainInfo(FModuleOp moduleOp,
                                                const ExportTable &exportTable,
                                                ArrayAttr &result) {
  // At this point, all domain variables mentioned in ports have been
  // solved by generalizing the moduleOp (adding input domain ports). Now, we
  // have to form the new port domain information for the moduleOp by examining
  // the the associated domains of each port.
  auto *context = moduleOp.getContext();
  auto numDomains = getNumDomains();
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
          copyPortDomainAssociations(moduleOp, oldModuleDomainInfo, i);
      auto *row = cast<RowTerm>(getDomainAssociation(port));
      for (size_t domainIndex = 0; domainIndex < numDomains; ++domainIndex) {
        auto domainTypeID = DomainTypeID{domainIndex};
        if (associations[domainIndex])
          continue;

        auto domain = cast<ValueTerm>(find(row->elements[domainIndex]))->value;
        auto &exports = exportTable.at(domain);
        if (exports.empty()) {
          auto portName = moduleOp.getPortNameAttr(i);
          auto portLoc = moduleOp.getPortLocation(i);
          auto domainDecl = getDomain(domainTypeID);
          auto domainName = domainDecl.getNameAttr();
          auto diag = emitError(portLoc)
                      << "private " << domainName << " association for port "
                      << portName;
          diag.attachNote(domain.getLoc()) << "associated domain: " << domain;
          noteLocation(diag, moduleOp);
          return failure();
        }

        if (exports.size() > 1) {
          emitAmbiguousPortDomainAssociation(moduleOp, exports, domainTypeID,
                                             i);
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

LogicalResult PassState::updateInstance(FInstanceLike op, OpBuilder &builder) {
  LLVM_DEBUG(llvm::dbgs().indent(4) << "update " << render(op) << "\n");
  auto numPorts = op->getNumResults();
  for (size_t i = 0; i < numPorts; ++i) {
    auto port = dyn_cast<DomainValue>(op->getResult(i));
    auto direction = op.getPortDirection(i);

    // If the port is an input domain, we may need to drive the input with
    // a value. If we don't know what value to drive to the port, drive an
    // anonymous domain.
    if (port && direction == Direction::In && !isDriven(port)) {
      auto loc = port.getLoc();
      auto *term = getTermForDomain(port);
      if (auto *var = dyn_cast<VariableTerm>(term)) {
        auto domainType = cast<DomainType>(op->getResult(i).getType());
        auto domainTypeID = getDomainTypeID(domainType);
        auto domainDecl = getDomain(domainTypeID);
        auto name = domainDecl.getNameAttr();
        DomainValue anon;
        {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointAfter(op);
          auto op = DomainCreateAnonOp::create(builder, loc, domainType, name);
          dirty();
          LLVM_DEBUG(llvm::dbgs().indent(6)
                     << "create " << render(op.getOperation()) << "\n");
          anon = op;
        }
        solve(var, allocVal(anon));
        LLVM_DEBUG(llvm::dbgs().indent(6) << "connect " << render(port)
                                          << " := " << render(anon) << "\n");
        // Create domain.define at the end of the block to avoid use-before-def.
        DomainDefineOp::create(builder, loc, port, anon);
        continue;
      }
      if (auto *val = dyn_cast<ValueTerm>(term)) {
        auto value = val->value;
        LLVM_DEBUG(llvm::dbgs().indent(6) << "connect " << render(port)
                                          << " := " << render(value) << "\n");
        // Create domain.define at the end of the block to avoid use-before-def.
        DomainDefineOp::create(builder, loc, port, value);
        continue;
      }
      llvm_unreachable("unhandled domain term type");
    }
  }

  return success();
}

LogicalResult PassState::updateWire(WireOp wireOp, OpBuilder &builder) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(wireOp);

  auto result = wireOp.getResult();
  if (!isa<FIRRTLBaseType>(result.getType()))
    return success();

  auto *row = getDomainAssociationAsRow(wireOp.getResult());

  SmallVector<Value> domainOperands;
  for (auto [i, element] : llvm::enumerate(
           llvm::map_range(row->elements, [&](auto e) { return find(e); }))) {
    if (auto *val = dyn_cast<ValueTerm>(element)) {
      domainOperands.push_back(val->value);
      continue;
    }
    if (auto *var = dyn_cast<VariableTerm>(element)) {
      auto domainDecl = getDomain(DomainTypeID{i});
      auto domainType = DomainType::getFromDomainOp(domainDecl);
      auto domainName = domainDecl.getNameAttr();
      auto anonDomain = DomainCreateAnonOp::create(builder, wireOp.getLoc(),
                                                   domainType, domainName);
      domainOperands.push_back(anonDomain);
      auto *val = allocVal(anonDomain);
      solve(var, val);
      continue;
    }
    assert(0 && "unhandled domain type");
  }
  wireOp.getDomainsMutable().assign(domainOperands);
  return success();
}

LogicalResult PassState::updateModuleBody(FModuleOp moduleOp) {
  // Set insertion point to end of block so all domain.define operations are
  // created there, avoiding use-before-def issues.
  OpBuilder builder(moduleOp.getContext());
  builder.setInsertionPointToEnd(moduleOp.getBodyBlock());
  auto result = moduleOp.getBodyBlock()->walk([&](Operation *op) -> WalkResult {
    if (auto wire = dyn_cast<WireOp>(op))
      return updateWire(wire, builder);
    if (auto instance = dyn_cast<FInstanceLike>(op))
      return updateInstance(instance, builder);
    return success();
  });
  return failure(result.wasInterrupted());
}

LogicalResult PassState::updateModule(FModuleOp moduleOp) {
  auto exports = initializeExportTable(moduleOp);
  PendingUpdates pending;
  getUpdatesForModule(moduleOp, exports, pending);
  applyUpdatesToModule(moduleOp, exports, pending);

  ArrayAttr portDomainInfo;
  if (failed(updateModuleDomainInfo(moduleOp, exports, portDomainInfo)))
    return failure();

  if (failed(driveModuleOutputDomainPorts(moduleOp)))
    return failure();

  // Record the updated interface change in the update
  auto &entry = getModuleUpdateTable()[moduleOp.getModuleNameAttr()];
  entry.portDomainInfo = portDomainInfo;
  entry.portInsertions = std::move(pending.insertions);

  if (failed(updateModuleBody(moduleOp)))
    return failure();

  LLVM_DEBUG({
    llvm::dbgs().indent(2) << "port summary:\n";
    for (auto port : moduleOp.getBodyBlock()->getArguments()) {
      llvm::dbgs().indent(4) << render(port);
      auto info = cast<ArrayAttr>(
          moduleOp.getDomainInfoAttrForPort(port.getArgNumber()));
      if (info.size()) {
        llvm::dbgs() << " domains [";
        llvm::interleaveComma(
            info.getAsRange<IntegerAttr>(), llvm::dbgs(), [&](auto i) {
              llvm::dbgs() << render(moduleOp.getArgument(i.getUInt()));
            });
        llvm::dbgs() << "]";
      }
      llvm::dbgs() << "\n";
    }
  });

  return success();
}

LogicalResult PassState::checkModulePorts(FModuleLike moduleOp) {
  auto numDomains = getNumDomains();
  auto domainInfo = moduleOp.getDomainInfoAttr();
  auto numPorts = moduleOp.getNumPorts();

  DenseMap<unsigned, DomainTypeID> domainTypeIDTable;
  for (size_t i = 0; i < numPorts; ++i) {
    if (isa<DomainType>(moduleOp.getPortType(i)))
      domainTypeIDTable[i] = getDomainTypeID(moduleOp, i);
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
        emitDuplicatePortDomainError(moduleOp, i, domainTypeID,
                                     prevDomainPortIndex, domainPortIndex);
        return failure();
      }
      associations[domainTypeID.index] = domainPortIndex;
    }

    // Check the associations for completeness.
    for (size_t domainIndex = 0; domainIndex < numDomains; ++domainIndex) {
      auto typeID = DomainTypeID{domainIndex};
      if (!associations[domainIndex]) {
        emitMissingPortDomainAssociationError(moduleOp, typeID, i);
        return failure();
      }
    }
  }

  return success();
}

LogicalResult PassState::checkModuleDomainPortDrivers(FModuleOp moduleOp) {
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

LogicalResult PassState::checkInstanceDomainPortDrivers(FInstanceLike op) {
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

LogicalResult PassState::checkModuleBody(FModuleOp moduleOp) {
  auto result = moduleOp.getBody().walk([&](FInstanceLike op) -> WalkResult {
    return checkInstanceDomainPortDrivers(op);
  });
  return failure(result.wasInterrupted());
}

LogicalResult PassState::inferModule(FModuleOp moduleOp) {
  LLVM_DEBUG(llvm::dbgs() << "infer: " << moduleOp.getModuleName() << "\n");
  if (failed(processModule(moduleOp)))
    return failure();

  return updateModule(moduleOp);
}

LogicalResult PassState::checkModule(FModuleOp moduleOp) {
  LLVM_DEBUG(llvm::dbgs() << "check: " << moduleOp.getModuleName() << "\n");
  if (failed(checkModulePorts(moduleOp)))
    return failure();

  if (failed(checkModuleDomainPortDrivers(moduleOp)))
    return failure();

  if (failed(checkModuleBody(moduleOp)))
    return failure();

  return processModule(moduleOp);
}

LogicalResult PassState::checkModule(FExtModuleOp extModuleOp) {
  LLVM_DEBUG(llvm::dbgs() << "check: " << extModuleOp.getModuleName() << "\n");
  return checkModulePorts(extModuleOp);
}

LogicalResult PassState::checkAndInferModule(FModuleOp moduleOp) {
  LLVM_DEBUG(llvm::dbgs() << "check/infer: " << moduleOp.getModuleName()
                          << "\n");

  if (failed(checkModulePorts(moduleOp)))
    return failure();

  if (failed(processModule(moduleOp)))
    return failure();

  if (failed(driveModuleOutputDomainPorts(moduleOp)))
    return failure();

  return updateModuleBody(moduleOp);
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

static LogicalResult runOnModuleLike(InferDomainsMode mode,
                                     PassGlobals &globals, Operation *op) {
  assert(mode != InferDomainsMode::Strip);
  PassState state(globals);
  if (auto moduleOp = dyn_cast<FModuleOp>(op)) {
    if (mode == InferDomainsMode::Check)
      return state.checkModule(moduleOp);

    if (mode == InferDomainsMode::InferAll || moduleOp.isPrivate())
      return state.inferModule(moduleOp);

    return state.checkAndInferModule(moduleOp);
  }

  if (auto extModuleOp = dyn_cast<FExtModuleOp>(op))
    return state.checkModule(extModuleOp);

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
    DenseSet<Operation *> errored;
    instanceGraph.walkPostOrder([&](auto &node) {
      auto moduleOp = node.getModule();
      for (auto *inst : node) {
        if (errored.contains(inst->getTarget()->getModule())) {
          errored.insert(moduleOp);
          return;
        }
      }
      if (failed(runOnModuleLike(mode, globals, node.getModule())))
        errored.insert(moduleOp);
    });
    if (errored.size())
      signalPassFailure();
  }
};
} // namespace
