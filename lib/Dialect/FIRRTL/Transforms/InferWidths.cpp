//===- InferWidths.cpp - Infer width of types -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the InferWidths pass.
//
//===----------------------------------------------------------------------===//

#include "./PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "infer-widths"

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Constraint Expressions
//===----------------------------------------------------------------------===//

namespace {
struct Expr;
} // namespace

/// Allow rvalue refs to `Expr` and subclasses to be printed to streams.
template <typename T, typename std::enable_if<std::is_base_of<Expr, T>::value,
                                              int>::type = 0>
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const T &e) {
  e.print(os);
  return os;
}

// Allow expression subclasses to be hashed.
namespace mlir {
template <typename T, typename std::enable_if<std::is_base_of<Expr, T>::value,
                                              int>::type = 0>
inline llvm::hash_code hash_value(const T &e) {
  return e.hash_value();
}
} // namespace mlir

namespace {
#define EXPR_NAMES(x) Root##x, Var##x, Known##x, Add##x, Pow##x, Max##x, Min##x
#define EXPR_KINDS EXPR_NAMES()
#define EXPR_CLASSES EXPR_NAMES(Expr)

/// An expression on the right-hand side of a constraint.
struct Expr {
  enum class Kind { EXPR_KINDS };
  llvm::Optional<int32_t> solution = {};
  Kind kind;

  /// Print a human-readable representation of this expr.
  void print(llvm::raw_ostream &os) const;

  // Iterators over the child expressions.
  typedef Expr *const *iterator;
  iterator begin() const;
  iterator end() const;

protected:
  Expr(Kind kind) : kind(kind) {}
  llvm::hash_code hash_value() const { return llvm::hash_value(kind); }
};

/// Helper class to CRTP-derive common functions.
template <class DerivedT, Expr::Kind DerivedKind>
struct ExprBase : public Expr {
  ExprBase() : Expr(DerivedKind) {}
  static bool classof(const Expr *e) { return e->kind == DerivedKind; }
  bool operator==(const Expr &other) const {
    if (auto otherSame = dyn_cast<DerivedT>(other))
      return *static_cast<DerivedT *>(this) == otherSame;
    return false;
  }
  iterator begin() const { return nullptr; }
  iterator end() const { return nullptr; }
};

/// A free variable.
struct RootExpr : public ExprBase<RootExpr, Expr::Kind::Root> {
  RootExpr(std::vector<Expr *> &exprs) : exprs(exprs) {}
  void print(llvm::raw_ostream &os) const { os << "root"; }
  iterator begin() const { return &exprs[0]; }
  iterator end() const { return &exprs[0] + exprs.size(); }
  std::vector<Expr *> &exprs;
};

/// A free variable.
struct VarExpr : public ExprBase<VarExpr, Expr::Kind::Var> {
  void print(llvm::raw_ostream &os) const {
    // Hash the `this` pointer into something somewhat human readable. Since
    // this is just for debug dumping, we wrap around at 4096 variables.
    os << "var" << ((size_t)this / llvm::PowerOf2Ceil(sizeof(*this)) & 0xFFF);
  }
  iterator begin() const { return &constraint; }
  iterator end() const { return &constraint + 1; }

  /// The constraint expression this variable is supposed to be greater than or
  /// equal to. This is not part of the variable's hash and equality property.
  Expr *constraint = nullptr;
};

/// A known constant value.
struct KnownExpr : public ExprBase<KnownExpr, Expr::Kind::Known> {
  KnownExpr(int32_t value) : ExprBase() { solution = value; }
  void print(llvm::raw_ostream &os) const { os << solution.getValue(); }
  bool operator==(const KnownExpr &other) const {
    return solution.getValue() == other.solution.getValue();
  }
  llvm::hash_code hash_value() const {
    return llvm::hash_combine(Expr::hash_value(), solution.getValue());
  }
};

/// A unary expression. Contains the actual data. Concrete subclasses are merely
/// there for show and ease of use.
struct UnaryExpr : public Expr {
  bool operator==(const UnaryExpr &other) const {
    return kind == other.kind && arg == other.arg;
  }
  llvm::hash_code hash_value() const {
    return llvm::hash_combine(Expr::hash_value(), arg);
  }
  iterator begin() const { return &arg; }
  iterator end() const { return &arg + 1; }

  /// The child expression.
  Expr *const arg;

protected:
  UnaryExpr(Kind kind, Expr *arg) : Expr(kind), arg(arg) {}
};

/// Helper class to CRTP-derive common functions.
template <class DerivedT, Expr::Kind DerivedKind>
struct UnaryExprBase : public UnaryExpr {
  template <typename... Args>
  UnaryExprBase(Args &&... args)
      : UnaryExpr(DerivedKind, std::forward<Args>(args)...) {}
  static bool classof(const Expr *e) { return e->kind == DerivedKind; }
};

/// A power of two.
struct PowExpr : public UnaryExprBase<PowExpr, Expr::Kind::Pow> {
  using UnaryExprBase::UnaryExprBase;
  void print(llvm::raw_ostream &os) const { os << "2^" << arg; }
};

/// A binary expression. Contains the actual data. Concrete subclasses are
/// merely there for show and ease of use.
struct BinaryExpr : public Expr {
  bool operator==(const BinaryExpr &other) const {
    return kind == other.kind && lhs() == other.lhs() && rhs() == other.rhs();
  }
  llvm::hash_code hash_value() const {
    return llvm::hash_combine(Expr::hash_value(), *args);
  }
  Expr *lhs() const { return args[0]; }
  Expr *rhs() const { return args[1]; }
  iterator begin() const { return args; }
  iterator end() const { return args + 2; }

  /// The child expressions.
  Expr *const args[2];

protected:
  BinaryExpr(Kind kind, Expr *lhs, Expr *rhs) : Expr(kind), args{lhs, rhs} {}
};

/// Helper class to CRTP-derive common functions.
template <class DerivedT, Expr::Kind DerivedKind>
struct BinaryExprBase : public BinaryExpr {
  template <typename... Args>
  BinaryExprBase(Args &&... args)
      : BinaryExpr(DerivedKind, std::forward<Args>(args)...) {}
  static bool classof(const Expr *e) { return e->kind == DerivedKind; }
};

/// An addition.
struct AddExpr : public BinaryExprBase<AddExpr, Expr::Kind::Add> {
  using BinaryExprBase::BinaryExprBase;
  void print(llvm::raw_ostream &os) const {
    os << "(" << *lhs() << " + " << *rhs() << ")";
  }
};

/// The maximum of two expressions.
struct MaxExpr : public BinaryExprBase<MaxExpr, Expr::Kind::Max> {
  using BinaryExprBase::BinaryExprBase;
  void print(llvm::raw_ostream &os) const {
    os << "max(" << *lhs() << ", " << *rhs() << ")";
  }
};

/// The minimum of two expressions.
struct MinExpr : public BinaryExprBase<MinExpr, Expr::Kind::Min> {
  using BinaryExprBase::BinaryExprBase;
  void print(llvm::raw_ostream &os) const {
    os << "min(" << *lhs() << ", " << *rhs() << ")";
  }
};

void Expr::print(llvm::raw_ostream &os) const {
  TypeSwitch<const Expr *>(this).Case<EXPR_CLASSES>(
      [&](auto *e) { e->print(os); });
}

Expr::iterator Expr::begin() const {
  return TypeSwitch<const Expr *, Expr::iterator>(this).Case<EXPR_CLASSES>(
      [&](auto *e) { return e->begin(); });
}

Expr::iterator Expr::end() const {
  return TypeSwitch<const Expr *, Expr::iterator>(this).Case<EXPR_CLASSES>(
      [&](auto *e) { return e->end(); });
}

} // namespace

//===----------------------------------------------------------------------===//
// GraphTraits on constraint expressions
//===----------------------------------------------------------------------===//

namespace llvm {
template <>
struct GraphTraits<Expr *> {
  using ChildIteratorType = Expr::iterator;
  using NodeRef = Expr *;

  static NodeRef getEntryNode(NodeRef node) { return node; }
  static inline ChildIteratorType child_begin(NodeRef node) {
    return node->begin();
  }
  static inline ChildIteratorType child_end(NodeRef node) {
    return node->end();
  }
};
} // namespace llvm

//===----------------------------------------------------------------------===//
// Fast bump allocator with optional interning
//===----------------------------------------------------------------------===//

namespace {

/// An allocation slot in the `InternedAllocator`.
template <typename T>
struct InternedSlot {
  T *ptr;
  InternedSlot(T *ptr) : ptr(ptr) {}
};

/// A simple bump allocator that ensures only ever one copy per object exists.
/// The allocated objects must not have a destructor.
template <typename T, typename std::enable_if_t<
                          std::is_trivially_destructible<T>::value, int> = 0>
class InternedAllocator {
  using Slot = InternedSlot<T>;
  llvm::DenseSet<Slot> interned;
  llvm::BumpPtrAllocator &allocator;

public:
  InternedAllocator(llvm::BumpPtrAllocator &allocator) : allocator(allocator) {}

  /// Allocate a new object if it does not yet exist, or return a pointer to the
  /// existing one. `R` is the type of the object to be allocated. `R` must be
  /// derived from or be the type `T`.
  template <typename R = T, typename... Args>
  std::pair<R *, bool> alloc(Args &&... args) {
    auto stack_value = R(std::forward<Args>(args)...);
    auto stack_slot = Slot(&stack_value);
    auto it = interned.find(stack_slot);
    if (it != interned.end())
      return std::make_pair(static_cast<R *>(it->ptr), false);
    auto heap_value = new (allocator) R(std::move(stack_value));
    interned.insert(Slot(heap_value));
    return std::make_pair(heap_value, true);
  }
};

/// A simple bump allocator. The allocated objects must not have a destructor.
/// This allocator is mainly there for symmetry with the `InternedAllocator`.
class VarAllocator {
  llvm::BumpPtrAllocator &allocator;
  using T = VarExpr;

public:
  VarAllocator(llvm::BumpPtrAllocator &allocator) : allocator(allocator) {}

  /// Allocate a new object. `R` is the type of the object to be allocated. `R`
  /// must be derived from or be the type `T`.
  template <typename R = T, typename... Args>
  R *alloc(Args &&... args) {
    return new (allocator) R(std::forward<Args>(args)...);
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Constraint Solver
//===----------------------------------------------------------------------===//

namespace {

/// A simple solver for width constraints.
class ConstraintSolver {
public:
  ConstraintSolver() = default;

  VarExpr *var() {
    auto v = vars.alloc();
    exprs.push_back(v);
    return v;
  }
  KnownExpr *known(int32_t value) { return alloc<KnownExpr>(knowns, value); }
  PowExpr *pow(Expr *arg) { return alloc<PowExpr>(uns, arg); }
  AddExpr *add(Expr *lhs, Expr *rhs) { return alloc<AddExpr>(bins, lhs, rhs); }
  MaxExpr *max(Expr *lhs, Expr *rhs) { return alloc<MaxExpr>(bins, lhs, rhs); }
  MinExpr *min(Expr *lhs, Expr *rhs) { return alloc<MinExpr>(bins, lhs, rhs); }

  /// Add a constraint `lhs >= rhs`. Multiple constraints on the same variable
  /// are coalesced into a `max(a, b)` expr.
  Expr *addGeqConstraint(VarExpr *lhs, Expr *rhs) {
    lhs->constraint = lhs->constraint ? max(lhs->constraint, rhs) : rhs;
    return lhs->constraint;
  }

  void dumpConstraints(llvm::raw_ostream &os);
  void solve();

private:
  // Allocator for constraint expressions.
  llvm::BumpPtrAllocator allocator;
  VarAllocator vars = {allocator};
  InternedAllocator<KnownExpr> knowns = {allocator};
  InternedAllocator<UnaryExpr> uns = {allocator};
  InternedAllocator<BinaryExpr> bins = {allocator};

  /// A list of expressions in the order they were created.
  std::vector<Expr *> exprs;
  RootExpr root = {exprs};

  /// Add an allocated expression to the list above.
  template <typename R, typename T, typename... Args>
  R *alloc(InternedAllocator<T> &allocator, Args &&... args) {
    auto it = allocator.template alloc<R>(std::forward<Args>(args)...);
    if (it.second)
      exprs.push_back(it.first);
    return it.first;
  }

  // Forbid copyign or moving the solver, which would invalidate the refs to
  // allocator held by the allocators.
  ConstraintSolver(ConstraintSolver &&) = delete;
  ConstraintSolver(const ConstraintSolver &) = delete;
  ConstraintSolver &operator=(ConstraintSolver &&) = delete;
  ConstraintSolver &operator=(const ConstraintSolver &) = delete;
};

} // namespace

/// Print all constraints in the solver to an output stream.
void ConstraintSolver::dumpConstraints(llvm::raw_ostream &os) {
  for (auto *e : exprs) {
    if (auto *v = dyn_cast<VarExpr>(e)) {
      if (v->constraint)
        os << "- " << *v << " >= " << *v->constraint << "\n";
      else
        os << "- " << *v << " unconstrained\n";
    }
  }
}

// Helper function to compute unary expressions if the operand has a solution.
static void solveUnary(UnaryExpr *expr, std::function<int32_t(int32_t)> f) {
  if (expr->arg->solution.hasValue())
    expr->solution = f(expr->arg->solution.getValue());
}

// Helper function to compute binary expressions if both operands have a
// solution.
static void solveBinary(BinaryExpr *expr,
                        std::function<int32_t(int32_t, int32_t)> f) {
  if (expr->lhs()->solution.hasValue() && expr->rhs()->solution.hasValue())
    expr->solution =
        f(expr->lhs()->solution.getValue(), expr->rhs()->solution.getValue());
}

/// Solve the constraint problem. This is a very simple implementation that
/// does not fully solve the problem if there are weird dependency cycles
/// present.
void ConstraintSolver::solve() {
  // Iterate over the expressions in depth-first order and start substituting in
  // solutions.
  for (auto *expr : llvm::post_order(static_cast<Expr *>(&root))) {
    if (expr->solution.hasValue())
      continue;
    TypeSwitch<Expr *>(expr)
        .Case<VarExpr>([&](auto *var) {
          if (var->constraint)
            var->solution = var->constraint->solution;
        })
        .Case<PowExpr>([&](auto *expr) {
          solveUnary(expr, [](int32_t arg) {
            assert(arg < 32);
            return 1 << arg;
          });
        })
        .Case<AddExpr>([&](auto *expr) {
          solveBinary(expr, [](int32_t lhs, int32_t rhs) { return lhs + rhs; });
        })
        .Case<MaxExpr>([&](auto *expr) {
          solveBinary(expr, [](int32_t lhs, int32_t rhs) {
            return std::max(lhs, rhs);
          });
        })
        .Case<MinExpr>([&](auto *expr) {
          solveBinary(expr, [](int32_t lhs, int32_t rhs) {
            return std::min(lhs, rhs);
          });
        });
    LLVM_DEBUG({
      if (expr->solution.hasValue())
        llvm::dbgs() << "Setting " << *expr << " = "
                     << expr->solution.getValue() << "\n";
      else
        llvm::dbgs() << "Leaving " << *expr << " unsolved\n";
    });
  }
}

//===----------------------------------------------------------------------===//
// Inference Constraint Problem Mapping
//===----------------------------------------------------------------------===//

namespace {

/// A helper class which maps the types and operations in a design to a set of
/// variables and constraints to be solved later.
class InferenceMapping {
public:
  InferenceMapping(ConstraintSolver &solver) : solver(solver) {}

  LogicalResult map(CircuitOp op);
  LogicalResult map(FModuleOp op);
  LogicalResult mapOperation(Operation *op);

  Expr *declareVars(Value value);
  Expr *declareVars(Type type);
  Expr *declareVars(FIRRTLType type);

  void constrainTypes(Expr *larger, Expr *smaller);

  Expr *getExpr(Value value);
  Expr *getExprOrNull(Value value);
  void setExpr(Value value, Expr *expr);

private:
  /// The constraint solver into which we emit variables and constraints.
  ConstraintSolver &solver;

  /// The constraint exprs for each result type of an operation.
  // TODO: This should actually not map to `Expr *` directly, but rather a
  // view class that can represent aggregate exprs for bundles/arrays as well.
  DenseMap<Value, Expr *> opExprs;
};

} // namespace

LogicalResult InferenceMapping::map(CircuitOp op) {
  for (auto &op : *op.getBody()) {
    if (auto module = dyn_cast<FModuleOp>(&op))
      if (failed(map(module)))
        return failure();
  }
  return success();
}

LogicalResult InferenceMapping::map(FModuleOp module) {
  // Ensure we have constraint variables for the module ports.
  for (auto arg : module.getArguments())
    declareVars(arg);

  // Go through operations, creating type variables for results, and generating
  // constraints.
  auto result = module.getBody().walk(
      [&](Operation *op) { return WalkResult(mapOperation(op)); });

  return failure(result.wasInterrupted());
}

LogicalResult InferenceMapping::mapOperation(Operation *op) {
  bool mappingFailed = false;
  TypeSwitch<Operation *>(op)
      .Case<ConstantOp>([&](auto op) {
        // If the constant has a known width, use that. Otherwise pick the
        // smallest number of bits necessary to represent the constant.
        Expr *e;
        if (auto width = op.getType().getWidth())
          e = solver.known(*width);
        else {
          auto v = op.value();
          auto w = v.getBitWidth() - (v.isNegative() ? v.countLeadingOnes()
                                                     : v.countLeadingZeros());
          if (v.isSigned())
            w += 1;
          e = solver.known(std::max(w, 1u));
        }
        setExpr(op.getResult(), e);
      })
      .Case<WireOp, InvalidValueOp>(
          [&](auto op) { declareVars(op.getResult()); })

      // Arithmetic and Logical Binary Primitives
      .Case<AddPrimOp, SubPrimOp>([&](auto op) {
        auto lhs = getExpr(op.lhs());
        auto rhs = getExpr(op.rhs());
        auto e = solver.add(solver.max(lhs, rhs), solver.known(1));
        setExpr(op.getResult(), e);
      })
      .Case<MulPrimOp>([&](auto op) {
        auto lhs = getExpr(op.lhs());
        auto rhs = getExpr(op.rhs());
        auto e = solver.add(lhs, rhs);
        setExpr(op.getResult(), e);
      })
      .Case<DivPrimOp>([&](auto op) {
        auto lhs = getExpr(op.lhs());
        Expr *e;
        if (op.getType().isSigned()) {
          e = solver.add(lhs, solver.known(1));
        } else {
          e = lhs;
        }
        setExpr(op.getResult(), e);
      })
      .Case<RemPrimOp>([&](auto op) {
        auto lhs = getExpr(op.lhs());
        auto rhs = getExpr(op.rhs());
        auto e = solver.min(lhs, rhs);
        setExpr(op.getResult(), e);
      })
      .Case<AndPrimOp, OrPrimOp, XorPrimOp>([&](auto op) {
        auto lhs = getExpr(op.lhs());
        auto rhs = getExpr(op.rhs());
        auto e = solver.max(lhs, rhs);
        setExpr(op.getResult(), e);
      })

      // Misc Binary Primitives
      .Case<CatPrimOp>([&](auto op) {
        auto lhs = getExpr(op.lhs());
        auto rhs = getExpr(op.rhs());
        auto e = solver.add(lhs, rhs);
        setExpr(op.getResult(), e);
      })
      .Case<DShlPrimOp>([&](auto op) {
        auto lhs = getExpr(op.lhs());
        auto rhs = getExpr(op.rhs());
        auto e = solver.add(lhs, solver.add(solver.pow(rhs), solver.known(-1)));
        setExpr(op.getResult(), e);
      })
      .Case<DShlwPrimOp, DShrPrimOp>([&](auto op) {
        auto e = getExpr(op.lhs());
        setExpr(op.getResult(), e);
      })

      // Unary operators
      .Case<NegPrimOp>([&](auto op) {
        auto input = getExpr(op.input());
        auto e = solver.add(input, solver.known(1));
        setExpr(op.getResult(), e);
      })
      .Case<CvtPrimOp>([&](auto op) {
        auto input = getExpr(op.input());
        auto e = op.input().getType().template cast<IntType>().isSigned()
                     ? input
                     : solver.add(input, solver.known(1));
        setExpr(op.getResult(), e);
      })

      // Miscellaneous
      .Case<BitsPrimOp>([&](auto op) {
        setExpr(op.getResult(), solver.known(op.hi() - op.lo() + 1));
      })
      .Case<HeadPrimOp>(
          [&](auto op) { setExpr(op.getResult(), solver.known(op.amount())); })
      .Case<TailPrimOp>([&](auto op) {
        auto input = getExpr(op.input());
        auto e = solver.add(input, solver.known(-op.amount()));
        setExpr(op.getResult(), e);
      })
      .Case<PadPrimOp>([&](auto op) {
        auto input = getExpr(op.input());
        auto e = solver.max(input, solver.known(op.amount()));
        setExpr(op.getResult(), e);
      })
      .Case<ShlPrimOp>([&](auto op) {
        auto input = getExpr(op.input());
        auto e = solver.add(input, solver.known(op.amount()));
        setExpr(op.getResult(), e);
      })
      .Case<ShrPrimOp>([&](auto op) {
        auto input = getExpr(op.input());
        auto e = solver.max(solver.add(input, solver.known(-op.amount())),
                            solver.known(1));
        setExpr(op.getResult(), e);
      })

      // Handle operations whose output width matches the input width.
      .Case<NotPrimOp, AsSIntPrimOp, AsUIntPrimOp, AsPassivePrimOp,
            AsNonPassivePrimOp>(
          [&](auto op) { setExpr(op.getResult(), getExpr(op.input())); })

      // Handle operations with a single result type that always has a
      // well-known width.
      .Case<LEQPrimOp, LTPrimOp, GEQPrimOp, GTPrimOp, EQPrimOp, NEQPrimOp,
            AsClockPrimOp, AsAsyncResetPrimOp, AndRPrimOp, OrRPrimOp,
            XorRPrimOp>([&](auto op) {
        auto width = op.getType().getBitWidthOrSentinel();
        assert(width > 0 && "width should have been checked by verifier");
        setExpr(op.getResult(), solver.known(width));
      })

      .Case<MuxPrimOp>([&](auto op) {
        auto sel = getExpr(op.sel());
        constrainTypes(sel, solver.known(1));
        auto high = getExpr(op.high());
        auto low = getExpr(op.low());
        auto e = solver.max(high, low);
        setExpr(op.getResult(), e);
      })

      // Handle the various connect statements that imply a type constraint.
      .Case<ConnectOp>([&](auto op) {
        auto dest = getExpr(op.dest());
        auto src = getExpr(op.src());
        constrainTypes(dest, src);
      })

      // Handle the no-ops that don't interact with width inference.
      .Case<PrintFOp, SkipOp, StopOp, WhenOp, AssertOp, AssumeOp, CoverOp>(
          [&](auto) {})

      .Default([&](auto op) {
        op->emitOpError("not supported in width inference");
        mappingFailed = true;
      });
  // TODO: Handle PartialConnect
  // TODO: Handle DefRegister
  // TODO: Handle Attach
  // TODO: Handle Conditionally
  return failure(mappingFailed);
}

/// Declare free variables for the type of a value, and associate the resulting
/// set of variables with that value.
Expr *InferenceMapping::declareVars(Value value) {
  Expr *e = declareVars(value.getType());
  setExpr(value, e);
  return e;
}

/// Declare free variables for a type.
Expr *InferenceMapping::declareVars(Type type) {
  if (auto ftype = type.dyn_cast<FIRRTLType>())
    return declareVars(ftype);
  // TODO: Once we support compound types, non-FIRRTL types will just map to
  // an empty list of expressions in the solver. At that point we'll have
  // something proper to return here.
  llvm_unreachable("non-FIRRTL ops not supported");
}

/// Declare free variables for a FIRRTL type.
Expr *InferenceMapping::declareVars(FIRRTLType type) {
  // TODO: Support aggregate and compound types as well.
  auto width = type.getBitWidthOrSentinel();
  if (width >= 0) {
    return solver.known(width);
  } else if (width == -1) {
    return solver.var();
  } else if (auto inner = type.dyn_cast<FlipType>()) {
    return declareVars(inner.getElementType());
  } else {
    // TODO: Once we support compound types, non-FIRRTL types will just map to
    // an empty list of expressions in the solver. At that point we'll have
    // something proper to return here.
    llvm_unreachable("non-FIRRTL ops not supported");
  }
}

/// Establishes constraints to ensure the sizes in the `larger` type are greater
/// than or equal to the sizes in the `smaller` type.
void InferenceMapping::constrainTypes(Expr *larger, Expr *smaller) {
  // Mimic the Scala implementation here by simply doing nothing if the larger
  // expr is not a free variable. Apparently there are many cases where
  // useless constraints can be added, e.g. on multiple well-known values. As
  // long as we don't want to do type checking itself here, but only width
  // inference, we should be fine ignoring expr we cannot constraint anyway.
  if (auto largerVar = dyn_cast<VarExpr>(larger)) {
    auto c = solver.addGeqConstraint(largerVar, smaller);
    LLVM_DEBUG(llvm::dbgs()
               << "Constrained " << *largerVar << " >= " << *c << "\n");
  }
}

/// Get the constraint expression for a value.
Expr *InferenceMapping::getExpr(Value value) {
  auto expr = getExprOrNull(value);
  assert(expr && "constraint expr should have been constructed for value");
  return expr;
}

/// Get the constraint expression for a value, or null if no expression exists
/// for the value.
Expr *InferenceMapping::getExprOrNull(Value value) {
  auto it = opExprs.find(value);
  return it != opExprs.end() ? it->second : nullptr;
}

/// Associate a constraint expression with a value.
void InferenceMapping::setExpr(Value value, Expr *expr) {
  LLVM_DEBUG(llvm::dbgs() << "Expr " << *expr << " for " << value << "\n");
  opExprs.insert(std::make_pair(value, expr));
}

//===----------------------------------------------------------------------===//
// Inference Result Application
//===----------------------------------------------------------------------===//

namespace {
/// A helper class which maps the types and operations in a design to a set
/// of variables and constraints to be solved later.
class InferenceTypeUpdate {
public:
  InferenceTypeUpdate(InferenceMapping &mapping) : mapping(mapping) {}

  void update(CircuitOp op);
  bool updateOperation(Operation *op);
  bool updateValue(Value value);
  FIRRTLType updateType(FIRRTLType type, uint32_t solution);

private:
  InferenceMapping &mapping;
};

} // namespace

/// Update the types throughout a circuit.
void InferenceTypeUpdate::update(CircuitOp op) {
  op.walk([&](Operation *op) { updateOperation(op); });
}

/// Update the result types of an operation.
bool InferenceTypeUpdate::updateOperation(Operation *op) {
  bool anyChanged = false;
  for (Value v : op->getResults()) {
    anyChanged |= updateValue(v);
  }

  // If this is a connect operation, width inference might have inferred a RHS
  // that is wider than the LHS, in which case an additional BitsPrimOp is
  // necessary to truncate the value.
  if (auto con = dyn_cast<ConnectOp>(op)) {
    auto lhs = con.dest().getType().cast<FIRRTLType>();
    auto rhs = con.src().getType().cast<FIRRTLType>();
    auto lhsWidth = lhs.getBitWidthOrSentinel();
    auto rhsWidth = rhs.getBitWidthOrSentinel();
    if (lhsWidth > 0 && rhsWidth > 0 && lhsWidth < rhsWidth) {
      OpBuilder builder(op);
      auto trunc = builder.createOrFold<BitsPrimOp>(con.getLoc(), con.src(),
                                                    lhsWidth - 1, 0);
      if (rhs.isa<SIntType>())
        trunc = builder.createOrFold<AsSIntPrimOp>(con.getLoc(), lhs, trunc);
      LLVM_DEBUG(llvm::dbgs()
                 << "Truncating RHS to " << lhs << " in " << con << "\n");
      con->replaceUsesOfWith(con.src(), trunc);
    }
  }

  // If this is a module, update its ports.
  if (auto module = dyn_cast<FModuleOp>(op)) {
    // Update the block argument types.
    bool argsChanged = false;
    std::vector<Type> argTypes;
    argTypes.reserve(module.getArguments().size());
    for (auto arg : module.getArguments()) {
      argsChanged |= updateValue(arg);
      argTypes.push_back(arg.getType());
    }

    // Update the module function type if needed.
    if (argsChanged) {
      auto type =
          FunctionType::get(op->getContext(), argTypes, /*resultTypes*/ {});
      module->setAttr(FModuleOp::getTypeAttrName(), TypeAttr::get(type));
      anyChanged = true;
    }
  }
  return anyChanged;
}

/// Resize a `uint`, `sint`, or `analog` type to a specific width.
static FIRRTLType resizeType(FIRRTLType type, uint32_t newWidth) {
  auto *context = type.getContext();
  return TypeSwitch<FIRRTLType, FIRRTLType>(type)
      .Case<UIntType>(
          [&](auto type) { return UIntType::get(context, newWidth); })
      .Case<SIntType>(
          [&](auto type) { return SIntType::get(context, newWidth); })
      .Case<AnalogType>(
          [&](auto type) { return AnalogType::get(context, newWidth); })
      .Default([&](auto type) { return type; });
}

/// Update the type of a value.
bool InferenceTypeUpdate::updateValue(Value value) {
  // Check if the value has a type which we can update.
  auto type = value.getType().dyn_cast<FIRRTLType>();
  if (!type)
    return false;

  // Get the inferred width.
  Expr *expr = mapping.getExprOrNull(value);
  if (!expr || !expr->solution.hasValue())
    return false;
  int32_t solution = expr->solution.getValue();
  if (solution < 0)
    return false;

  // Update the type.
  auto newType = updateType(type, solution);
  LLVM_DEBUG(llvm::dbgs() << "Update " << value << " to " << newType << "\n");
  value.setType(newType);

  // If this is a ConstantOp, adjust the width of the underlying APInt. Unsized
  // constants have APInts which are *at least* wide enough to hold the value,
  // but may be larger. This can trip up the verifier.
  if (auto op = value.getDefiningOp<ConstantOp>()) {
    auto k = op.value();
    if (k.getBitWidth() > unsigned(solution))
      k = k.trunc(solution);
    op->setAttr("value", IntegerAttr::get(op.getContext(), k));
  }

  return newType != type;
}

/// Update a type.
FIRRTLType InferenceTypeUpdate::updateType(FIRRTLType type, uint32_t solution) {
  // TODO: This should actually take an aggregate bunch of constraint
  // expressions such that aggregate types can pick them apart appropriately.
  if (auto flip = type.dyn_cast<FlipType>()) {
    return FlipType::get(updateType(flip.getElementType(), solution));
  } else if (type.getBitWidthOrSentinel() == -1) {
    return resizeType(type, solution);
  } else {
    return type;
  }
}

//===----------------------------------------------------------------------===//
// Hashing
//===----------------------------------------------------------------------===//

// Hash slots in the interned allocator as if they were the pointed-to value
// itself.
namespace llvm {
template <typename T>
struct DenseMapInfo<InternedSlot<T>> {
  using Slot = InternedSlot<T>;
  static Slot getEmptyKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return Slot(static_cast<T *>(pointer));
  }
  static Slot getTombstoneKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return Slot(static_cast<T *>(pointer));
  }
  static unsigned getHashValue(Slot val) { return mlir::hash_value(*val.ptr); }
  static bool isEqual(Slot LHS, Slot RHS) {
    auto empty = getEmptyKey().ptr;
    auto tombstone = getTombstoneKey().ptr;
    if (LHS.ptr == empty || RHS.ptr == empty || LHS.ptr == tombstone ||
        RHS.ptr == tombstone)
      return LHS.ptr == RHS.ptr;
    return *LHS.ptr == *RHS.ptr;
  }
};
} // namespace llvm

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
class InferWidthsPass : public InferWidthsBase<InferWidthsPass> {
  void runOnOperation() override;
};
} // namespace

void InferWidthsPass::runOnOperation() {
  // Collect variables and constraints
  ConstraintSolver solver;
  InferenceMapping mapping(solver);
  if (failed(mapping.map(getOperation()))) {
    signalPassFailure();
    return;
  }
  LLVM_DEBUG({
    llvm::dbgs() << "Constraints:\n";
    solver.dumpConstraints(llvm::dbgs());
  });

  // Solve the constraints.
  solver.solve();

  // Update the types with the inferred widths.
  InferenceTypeUpdate(mapping).update(getOperation());
}

std::unique_ptr<mlir::Pass> circt::firrtl::createInferWidthsPass() {
  return std::make_unique<InferWidthsPass>();
}
