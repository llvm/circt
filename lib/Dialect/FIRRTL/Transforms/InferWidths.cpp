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

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/Debug.h"
#include "circt/Support/FieldRef.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "infer-widths"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_INFERWIDTHS
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using mlir::InferTypeOpInterface;
using mlir::WalkOrder;

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static void diagnoseUninferredType(InFlightDiagnostic &diag, Type t,
                                   Twine str) {
  auto basetype = type_dyn_cast<FIRRTLBaseType>(t);
  if (!basetype)
    return;
  if (!basetype.hasUninferredWidth())
    return;

  if (basetype.isGround())
    diag.attachNote() << "Field: \"" << str << "\"";
  else if (auto vecType = type_dyn_cast<FVectorType>(basetype))
    diagnoseUninferredType(diag, vecType.getElementType(), str + "[]");
  else if (auto bundleType = type_dyn_cast<BundleType>(basetype))
    for (auto &elem : bundleType.getElements())
      diagnoseUninferredType(diag, elem.type, str + "." + elem.name.getValue());
}

/// Calculate the "InferWidths-fieldID" equivalent for the given fieldID + type.
static uint64_t convertFieldIDToOurVersion(uint64_t fieldID, FIRRTLType type) {
  uint64_t convertedFieldID = 0;

  auto curFID = fieldID;
  Type curFType = type;
  while (curFID != 0) {
    auto [child, subID] =
        hw::FieldIdImpl::getSubTypeByFieldID(curFType, curFID);
    if (isa<FVectorType>(curFType))
      convertedFieldID++; // Vector fieldID is 1.
    else
      convertedFieldID += curFID - subID; // Add consumed portion.
    curFID = subID;
    curFType = child;
  }

  return convertedFieldID;
}

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
#define EXPR_NAMES(x)                                                          \
  Var##x, Derived##x, Id##x, Known##x, Add##x, Pow##x, Max##x, Min##x
#define EXPR_KINDS EXPR_NAMES()
#define EXPR_CLASSES EXPR_NAMES(Expr)

/// An expression on the right-hand side of a constraint.
struct Expr {
  enum class Kind : uint8_t { EXPR_KINDS };

  /// Print a human-readable representation of this expr.
  void print(llvm::raw_ostream &os) const;

  std::optional<int32_t> getSolution() const {
    if (hasSolution)
      return solution;
    return std::nullopt;
  }

  void setSolution(int32_t solution) {
    hasSolution = true;
    this->solution = solution;
  }

  Kind getKind() const { return kind; }

protected:
  Expr(Kind kind) : kind(kind) {}
  llvm::hash_code hash_value() const { return llvm::hash_value(kind); }

private:
  int32_t solution;
  Kind kind;
  bool hasSolution = false;
};

/// Helper class to CRTP-derive common functions.
template <class DerivedT, Expr::Kind DerivedKind>
struct ExprBase : public Expr {
  ExprBase() : Expr(DerivedKind) {}
  static bool classof(const Expr *e) { return e->getKind() == DerivedKind; }
  bool operator==(const Expr &other) const {
    if (auto otherSame = dyn_cast<DerivedT>(other))
      return *static_cast<DerivedT *>(this) == otherSame;
    return false;
  }
};

/// A free variable.
struct VarExpr : public ExprBase<VarExpr, Expr::Kind::Var> {
  void print(llvm::raw_ostream &os) const {
    // Hash the `this` pointer into something somewhat human readable. Since
    // this is just for debug dumping, we wrap around at 65536 variables.
    os << "var" << ((size_t)this / llvm::PowerOf2Ceil(sizeof(*this)) & 0xFFFF);
  }

  /// The constraint expression this variable is supposed to be greater than or
  /// equal to. This is not part of the variable's hash and equality property.
  Expr *constraint = nullptr;

  /// The upper bound this variable is supposed to be smaller than or equal to.
  Expr *upperBound = nullptr;
  std::optional<int32_t> upperBoundSolution;
};

/// A derived width.
///
/// These are generated for `InvalidValueOp`s which want to derived their width
/// from connect operations that they are on the right hand side of.
struct DerivedExpr : public ExprBase<DerivedExpr, Expr::Kind::Derived> {
  void print(llvm::raw_ostream &os) const {
    // Hash the `this` pointer into something somewhat human readable.
    os << "derive"
       << ((size_t)this / llvm::PowerOf2Ceil(sizeof(*this)) & 0xFFF);
  }

  /// The expression this derived width is equivalent to.
  Expr *assigned = nullptr;
};

/// An identity expression.
///
/// This expression evaluates to its inner expression. It is used in a very
/// specific case of constraints on variables, in order to be able to track
/// where the constraint was imposed. Constraints on variables are represented
/// as `var >= <expr>`. When the first constraint `a` is imposed, it is stored
/// as the constraint expression (`var >= a`). When the second constraint `b` is
/// imposed, a *new* max expression is allocated (`var >= max(a, b)`).
/// Expressions are annotated with a location when they are created, which in
/// this case are connect ops. Since imposing the first constraint does not
/// create any new expression, the location information of that connect would be
/// lost. With an identity expression, imposing the first constraint becomes
/// `var >= identity(a)`, which is a *new* expression and properly tracks the
/// location info.
struct IdExpr : public ExprBase<IdExpr, Expr::Kind::Id> {
  IdExpr(Expr *arg) : arg(arg) { assert(arg); }
  void print(llvm::raw_ostream &os) const { os << "*" << *arg; }
  bool operator==(const IdExpr &other) const {
    return getKind() == other.getKind() && arg == other.arg;
  }
  llvm::hash_code hash_value() const {
    return llvm::hash_combine(Expr::hash_value(), arg);
  }

  /// The inner expression.
  Expr *const arg;
};

/// A known constant value.
struct KnownExpr : public ExprBase<KnownExpr, Expr::Kind::Known> {
  KnownExpr(int32_t value) : ExprBase() { setSolution(value); }
  void print(llvm::raw_ostream &os) const { os << *getSolution(); }
  bool operator==(const KnownExpr &other) const {
    return *getSolution() == *other.getSolution();
  }
  llvm::hash_code hash_value() const {
    return llvm::hash_combine(Expr::hash_value(), *getSolution());
  }
  int32_t getValue() const { return *getSolution(); }
};

/// A unary expression. Contains the actual data. Concrete subclasses are merely
/// there for show and ease of use.
struct UnaryExpr : public Expr {
  bool operator==(const UnaryExpr &other) const {
    return getKind() == other.getKind() && arg == other.arg;
  }
  llvm::hash_code hash_value() const {
    return llvm::hash_combine(Expr::hash_value(), arg);
  }

  /// The child expression.
  Expr *const arg;

protected:
  UnaryExpr(Kind kind, Expr *arg) : Expr(kind), arg(arg) { assert(arg); }
};

/// Helper class to CRTP-derive common functions.
template <class DerivedT, Expr::Kind DerivedKind>
struct UnaryExprBase : public UnaryExpr {
  template <typename... Args>
  UnaryExprBase(Args &&...args)
      : UnaryExpr(DerivedKind, std::forward<Args>(args)...) {}
  static bool classof(const Expr *e) { return e->getKind() == DerivedKind; }
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
    return getKind() == other.getKind() && lhs() == other.lhs() &&
           rhs() == other.rhs();
  }
  llvm::hash_code hash_value() const {
    return llvm::hash_combine(Expr::hash_value(), *args);
  }
  Expr *lhs() const { return args[0]; }
  Expr *rhs() const { return args[1]; }

  /// The child expressions.
  Expr *const args[2];

protected:
  BinaryExpr(Kind kind, Expr *lhs, Expr *rhs) : Expr(kind), args{lhs, rhs} {
    assert(lhs);
    assert(rhs);
  }
};

/// Helper class to CRTP-derive common functions.
template <class DerivedT, Expr::Kind DerivedKind>
struct BinaryExprBase : public BinaryExpr {
  template <typename... Args>
  BinaryExprBase(Args &&...args)
      : BinaryExpr(DerivedKind, std::forward<Args>(args)...) {}
  static bool classof(const Expr *e) { return e->getKind() == DerivedKind; }
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

} // namespace

//===----------------------------------------------------------------------===//
// Fast bump allocator with optional interning
//===----------------------------------------------------------------------===//

namespace {

// Hash slots in the interned allocator as if they were the pointed-to value
// itself.
template <typename T>
struct InternedSlotInfo : DenseMapInfo<T *> {
  static T *getEmptyKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return static_cast<T *>(pointer);
  }
  static T *getTombstoneKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return static_cast<T *>(pointer);
  }
  static unsigned getHashValue(const T *val) { return mlir::hash_value(*val); }
  static bool isEqual(const T *lhs, const T *rhs) {
    auto empty = getEmptyKey();
    auto tombstone = getTombstoneKey();
    if (lhs == empty || rhs == empty || lhs == tombstone || rhs == tombstone)
      return lhs == rhs;
    return *lhs == *rhs;
  }
};

/// A simple bump allocator that ensures only ever one copy per object exists.
/// The allocated objects must not have a destructor.
template <typename T, typename std::enable_if_t<
                          std::is_trivially_destructible<T>::value, int> = 0>
class InternedAllocator {
  llvm::DenseSet<T *, InternedSlotInfo<T>> interned;
  llvm::BumpPtrAllocator &allocator;

public:
  InternedAllocator(llvm::BumpPtrAllocator &allocator) : allocator(allocator) {}

  /// Allocate a new object if it does not yet exist, or return a pointer to the
  /// existing one. `R` is the type of the object to be allocated. `R` must be
  /// derived from or be the type `T`.
  template <typename R = T, typename... Args>
  std::pair<R *, bool> alloc(Args &&...args) {
    auto stackValue = R(std::forward<Args>(args)...);
    auto *stackSlot = &stackValue;
    auto it = interned.find(stackSlot);
    if (it != interned.end())
      return std::make_pair(static_cast<R *>(*it), false);
    auto heapValue = new (allocator) R(std::move(stackValue));
    interned.insert(heapValue);
    return std::make_pair(heapValue, true);
  }
};

/// A simple bump allocator. The allocated objects must not have a destructor.
/// This allocator is mainly there for symmetry with the `InternedAllocator`.
template <typename T, typename std::enable_if_t<
                          std::is_trivially_destructible<T>::value, int> = 0>
class Allocator {
  llvm::BumpPtrAllocator &allocator;

public:
  Allocator(llvm::BumpPtrAllocator &allocator) : allocator(allocator) {}

  /// Allocate a new object. `R` is the type of the object to be allocated. `R`
  /// must be derived from or be the type `T`.
  template <typename R = T, typename... Args>
  R *alloc(Args &&...args) {
    return new (allocator) R(std::forward<Args>(args)...);
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Constraint Solver
//===----------------------------------------------------------------------===//

namespace {
/// A canonicalized linear inequality that maps a constraint on var `x` to the
/// linear inequality `x >= max(a*x+b, c) + (failed ? ∞ : 0)`.
///
/// The inequality separately tracks recursive (a, b) and non-recursive (c)
/// constraints on `x`. This allows it to properly identify the combination of
/// the two constraints `x >= x-1` and `x >= 4` to be satisfiable as
/// `x >= max(x-1, 4)`. If it only tracked inequality as `x >= a*x+b`, the
/// combination of these two constraints would be `x >= x+4` (due to max(-1,4) =
/// 4), which would be unsatisfiable.
///
/// The `failed` flag acts as an additional `∞` term that renders the inequality
/// unsatisfiable. It is used as a tombstone value in case an operation renders
/// the equality unsatisfiable (e.g. `x >= 2**x` would be represented as the
/// inequality `x >= ∞`).
///
/// Inequalities represented in this form can easily be checked for
/// unsatisfiability in the presence of recursion by inspecting the coefficients
/// a and b. The `sat` function performs this action.
struct LinIneq {
  // x >= max(a*x+b, c) + (failed ? ∞ : 0)
  int32_t recScale = 0;   // a
  int32_t recBias = 0;    // b
  int32_t nonrecBias = 0; // c
  bool failed = false;

  /// Create a new unsatisfiable inequality `x >= ∞`.
  static LinIneq unsat() { return LinIneq(true); }

  /// Create a new inequality `x >= (failed ? ∞ : 0)`.
  explicit LinIneq(bool failed = false) : failed(failed) {}

  /// Create a new inequality `x >= bias`.
  explicit LinIneq(int32_t bias) : nonrecBias(bias) {}

  /// Create a new inequality `x >= scale*x+bias`.
  explicit LinIneq(int32_t scale, int32_t bias) {
    if (scale != 0) {
      recScale = scale;
      recBias = bias;
    } else {
      nonrecBias = bias;
    }
  }

  /// Create a new inequality `x >= max(recScale*x+recBias, nonrecBias) +
  /// (failed ? ∞ : 0)`.
  explicit LinIneq(int32_t recScale, int32_t recBias, int32_t nonrecBias,
                   bool failed = false)
      : failed(failed) {
    if (recScale != 0) {
      this->recScale = recScale;
      this->recBias = recBias;
      this->nonrecBias = nonrecBias;
    } else {
      this->nonrecBias = std::max(recBias, nonrecBias);
    }
  }

  /// Combine two inequalities by taking the maxima of corresponding
  /// coefficients.
  ///
  /// This essentially combines `x >= max(a1*x+b1, c1)` and `x >= max(a2*x+b2,
  /// c2)` into a new `x >= max(max(a1,a2)*x+max(b1,b2), max(c1,c2))`. This is
  /// a pessimistic upper bound, since e.g. `x >= 2x-10` and `x >= x-5` may both
  /// hold, but the resulting `x >= 2x-5` may pessimistically not hold.
  static LinIneq max(const LinIneq &lhs, const LinIneq &rhs) {
    return LinIneq(std::max(lhs.recScale, rhs.recScale),
                   std::max(lhs.recBias, rhs.recBias),
                   std::max(lhs.nonrecBias, rhs.nonrecBias),
                   lhs.failed || rhs.failed);
  }

  /// Combine two inequalities by summing up the two right hand sides.
  ///
  /// This is a tricky one, since the addition of the two max terms will lead to
  /// a maximum over four possible terms (similar to a binomial expansion). In
  /// order to shoehorn this back into a two-term maximum, we have to pick the
  /// recursive term that will grow the fastest.
  ///
  /// As an example for this problem, consider the following addition:
  ///
  ///   x >= max(a1*x+b1, c1) + max(a2*x+b2, c2)
  ///
  /// We would like to expand and rearrange this again into a maximum:
  ///
  ///   x >= max(a1*x+b1 + max(a2*x+b2, c2), c1 + max(a2*x+b2, c2))
  ///   x >= max(max(a1*x+b1 + a2*x+b2, a1*x+b1 + c2),
  ///            max(c1 + a2*x+b2, c1 + c2))
  ///   x >= max((a1+a2)*x+(b1+b2), a1*x+(b1+c2), a2*x+(b2+c1), c1+c2)
  ///
  /// Since we are combining two two-term maxima, there are four possible ways
  /// how the terms can combine, leading to the above four-term maximum. An easy
  /// upper bound of the form we want would be the following:
  ///
  ///   x >= max(max(a1+a2, a1, a2)*x + max(b1+b2, b1+c2, b2+c1), c1+c2)
  ///
  /// However, this is a very pessimistic upper-bound that will declare very
  /// common patterns in the IR as unbreakable cycles, despite them being very
  /// much breakable. For example:
  ///
  ///   x >= max(x, 42) + max(0, -3)  <-- breakable recursion
  ///   x >= max(max(1+0, 1, 0)*x + max(42+0, -3, 42), 42-2)
  ///   x >= max(x + 42, 39)          <-- unbreakable recursion!
  ///
  /// A better approach is to take the expanded four-term maximum, retain the
  /// non-recursive term (c1+c2), and estimate which one of the recursive terms
  /// (first three) will become dominant as we choose greater values for x.
  /// Since x never is inferred to be negative, the recursive term in the
  /// maximum with the highest scaling factor for x will end up dominating as
  /// x tends to ∞:
  ///
  ///   x >= max({
  ///     (a1+a2)*x+(b1+b2) if a1+a2 >= max(a1+a2, a1, a2) and a1>0 and a2>0,
  ///     a1*x+(b1+c2)      if    a1 >= max(a1+a2, a1, a2) and a1>0,
  ///     a2*x+(b2+c1)      if    a2 >= max(a1+a2, a1, a2) and a2>0,
  ///     0                 otherwise
  ///   }, c1+c2)
  ///
  /// In case multiple cases apply, the highest bias of the recursive term is
  /// picked. With this, the above problematic example triggers the second case
  /// and becomes:
  ///
  ///   x >= max(1*x+(0-3), 42-3) = max(x-3, 39)
  ///
  /// Of which the first case is chosen, as it has the lower bias value.
  static LinIneq add(const LinIneq &lhs, const LinIneq &rhs) {
    // Determine the maximum scaling factor among the three possible recursive
    // terms.
    auto enable1 = lhs.recScale > 0 && rhs.recScale > 0;
    auto enable2 = lhs.recScale > 0;
    auto enable3 = rhs.recScale > 0;
    auto scale1 = lhs.recScale + rhs.recScale; // (a1+a2)
    auto scale2 = lhs.recScale;                // a1
    auto scale3 = rhs.recScale;                // a2
    auto bias1 = lhs.recBias + rhs.recBias;    // (b1+b2)
    auto bias2 = lhs.recBias + rhs.nonrecBias; // (b1+c2)
    auto bias3 = rhs.recBias + lhs.nonrecBias; // (b2+c1)
    auto maxScale = std::max(scale1, std::max(scale2, scale3));

    // Among those terms that have a maximum scaling factor, determine the
    // largest bias value.
    std::optional<int32_t> maxBias;
    if (enable1 && scale1 == maxScale)
      maxBias = bias1;
    if (enable2 && scale2 == maxScale && (!maxBias || bias2 > *maxBias))
      maxBias = bias2;
    if (enable3 && scale3 == maxScale && (!maxBias || bias3 > *maxBias))
      maxBias = bias3;

    // Pick from the recursive terms the one with maximum scaling factor and
    // minimum bias value.
    auto nonrecBias = lhs.nonrecBias + rhs.nonrecBias; // c1+c2
    auto failed = lhs.failed || rhs.failed;
    if (enable1 && scale1 == maxScale && bias1 == *maxBias)
      return LinIneq(scale1, bias1, nonrecBias, failed);
    if (enable2 && scale2 == maxScale && bias2 == *maxBias)
      return LinIneq(scale2, bias2, nonrecBias, failed);
    if (enable3 && scale3 == maxScale && bias3 == *maxBias)
      return LinIneq(scale3, bias3, nonrecBias, failed);
    return LinIneq(0, 0, nonrecBias, failed);
  }

  /// Check if the inequality is satisfiable.
  ///
  /// The inequality becomes unsatisfiable if the RHS is ∞, or a>1, or a==1 and
  /// b <= 0. Otherwise there exists as solution for `x` that satisfies the
  /// inequality.
  bool sat() const {
    if (failed)
      return false;
    if (recScale > 1)
      return false;
    if (recScale == 1 && recBias > 0)
      return false;
    return true;
  }

  /// Dump the inequality in human-readable form.
  void print(llvm::raw_ostream &os) const {
    bool any = false;
    bool both = (recScale != 0 || recBias != 0) && nonrecBias != 0;
    os << "x >= ";
    if (both)
      os << "max(";
    if (recScale != 0) {
      any = true;
      if (recScale != 1)
        os << recScale << "*";
      os << "x";
    }
    if (recBias != 0) {
      if (any) {
        if (recBias < 0)
          os << " - " << -recBias;
        else
          os << " + " << recBias;
      } else {
        any = true;
        os << recBias;
      }
    }
    if (both)
      os << ", ";
    if (nonrecBias != 0) {
      any = true;
      os << nonrecBias;
    }
    if (both)
      os << ")";
    if (failed) {
      if (any)
        os << " + ";
      os << "∞";
    }
    if (!any)
      os << "0";
  }
};

/// A simple solver for width constraints.
class ConstraintSolver {
public:
  ConstraintSolver() = default;

  VarExpr *var() {
    auto *v = vars.alloc();
    varExprs.push_back(v);
    if (currentInfo)
      info[v].insert(currentInfo);
    if (currentLoc)
      locs[v].insert(*currentLoc);
    return v;
  }
  DerivedExpr *derived() {
    auto *d = derivs.alloc();
    derivedExprs.push_back(d);
    return d;
  }
  KnownExpr *known(int32_t value) { return alloc<KnownExpr>(knowns, value); }
  IdExpr *id(Expr *arg) { return alloc<IdExpr>(ids, arg); }
  PowExpr *pow(Expr *arg) { return alloc<PowExpr>(uns, arg); }
  AddExpr *add(Expr *lhs, Expr *rhs) { return alloc<AddExpr>(bins, lhs, rhs); }
  MaxExpr *max(Expr *lhs, Expr *rhs) { return alloc<MaxExpr>(bins, lhs, rhs); }
  MinExpr *min(Expr *lhs, Expr *rhs) { return alloc<MinExpr>(bins, lhs, rhs); }

  /// Add a constraint `lhs >= rhs`. Multiple constraints on the same variable
  /// are coalesced into a `max(a, b)` expr.
  Expr *addGeqConstraint(VarExpr *lhs, Expr *rhs) {
    if (lhs->constraint)
      lhs->constraint = max(lhs->constraint, rhs);
    else
      lhs->constraint = id(rhs);
    return lhs->constraint;
  }

  /// Add a constraint `lhs <= rhs`. Multiple constraints on the same variable
  /// are coalesced into a `min(a, b)` expr.
  Expr *addLeqConstraint(VarExpr *lhs, Expr *rhs) {
    if (lhs->upperBound)
      lhs->upperBound = min(lhs->upperBound, rhs);
    else
      lhs->upperBound = id(rhs);
    return lhs->upperBound;
  }

  void dumpConstraints(llvm::raw_ostream &os);
  LogicalResult solve();

  using ContextInfo = DenseMap<Expr *, llvm::SmallSetVector<FieldRef, 1>>;
  const ContextInfo &getContextInfo() const { return info; }
  void setCurrentContextInfo(FieldRef fieldRef) { currentInfo = fieldRef; }
  void setCurrentLocation(std::optional<Location> loc) { currentLoc = loc; }

private:
  // Allocator for constraint expressions.
  llvm::BumpPtrAllocator allocator;
  Allocator<VarExpr> vars = {allocator};
  Allocator<DerivedExpr> derivs = {allocator};
  InternedAllocator<KnownExpr> knowns = {allocator};
  InternedAllocator<IdExpr> ids = {allocator};
  InternedAllocator<UnaryExpr> uns = {allocator};
  InternedAllocator<BinaryExpr> bins = {allocator};

  /// A list of expressions in the order they were created.
  std::vector<VarExpr *> varExprs;
  std::vector<DerivedExpr *> derivedExprs;

  /// Add an allocated expression to the list above.
  template <typename R, typename T, typename... Args>
  R *alloc(InternedAllocator<T> &allocator, Args &&...args) {
    auto [expr, inserted] =
        allocator.template alloc<R>(std::forward<Args>(args)...);
    if (currentInfo)
      info[expr].insert(currentInfo);
    if (currentLoc)
      locs[expr].insert(*currentLoc);
    return expr;
  }

  /// Contextual information for each expression, indicating which values in the
  /// IR lead to this expression.
  ContextInfo info;
  FieldRef currentInfo = {};
  DenseMap<Expr *, llvm::SmallSetVector<Location, 1>> locs;
  std::optional<Location> currentLoc;

  // Forbid copyign or moving the solver, which would invalidate the refs to
  // allocator held by the allocators.
  ConstraintSolver(ConstraintSolver &&) = delete;
  ConstraintSolver(const ConstraintSolver &) = delete;
  ConstraintSolver &operator=(ConstraintSolver &&) = delete;
  ConstraintSolver &operator=(const ConstraintSolver &) = delete;

  void emitUninferredWidthError(VarExpr *var);

  LinIneq checkCycles(VarExpr *var, Expr *expr,
                      SmallPtrSetImpl<Expr *> &seenVars,
                      InFlightDiagnostic *reportInto = nullptr,
                      unsigned indent = 1);
};

} // namespace

/// Print all constraints in the solver to an output stream.
void ConstraintSolver::dumpConstraints(llvm::raw_ostream &os) {
  for (auto *v : varExprs) {
    if (v->constraint)
      os << "- " << *v << " >= " << *v->constraint << "\n";
    else
      os << "- " << *v << " unconstrained\n";
  }
}

#ifndef NDEBUG
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const LinIneq &l) {
  l.print(os);
  return os;
}
#endif

/// Compute the canonicalized linear inequality expression starting at `expr`,
/// for the `var` as the left hand side `x` of the inequality. `seenVars` is
/// used as a recursion breaker. Occurrences of `var` itself within the
/// expression are mapped to the `a` coefficient in the inequality. Any other
/// variables are substituted and, in the presence of a recursion in a variable
/// other than `var`, treated as zero. `info` is a mapping from constraint
/// expressions to values and operations that produced the expression, and is
/// used during error reporting. If `reportInto` is present, the function will
/// additionally attach unsatisfiable inequalities as notes to the diagnostic as
/// it encounters them.
LinIneq ConstraintSolver::checkCycles(VarExpr *var, Expr *expr,
                                      SmallPtrSetImpl<Expr *> &seenVars,
                                      InFlightDiagnostic *reportInto,
                                      unsigned indent) {
  auto ineq =
      TypeSwitch<Expr *, LinIneq>(expr)
          .Case<KnownExpr>(
              [&](auto *expr) { return LinIneq(expr->getValue()); })
          .Case<VarExpr>([&](auto *expr) {
            if (expr == var)
              return LinIneq(1, 0); // x >= 1*x + 0
            if (!seenVars.insert(expr).second)
              // Count recursions in other variables as 0. This is sane
              // since the cycle is either breakable, in which case the
              // recursion does not modify the resulting value of the
              // variable, or it is not breakable and will be caught by
              // this very function once it is called on that variable.
              return LinIneq(0);
            if (!expr->constraint)
              // Count unconstrained variables as `x >= 0`.
              return LinIneq(0);
            auto l = checkCycles(var, expr->constraint, seenVars, reportInto,
                                 indent + 1);
            seenVars.erase(expr);
            return l;
          })
          .Case<IdExpr>([&](auto *expr) {
            return checkCycles(var, expr->arg, seenVars, reportInto,
                               indent + 1);
          })
          .Case<PowExpr>([&](auto *expr) {
            // If we can evaluate `2**arg` to a sensible constant, do
            // so. This is the case if a == 0 and c < 31 such that 2**c is
            // representable.
            auto arg =
                checkCycles(var, expr->arg, seenVars, reportInto, indent + 1);
            if (arg.recScale != 0 || arg.nonrecBias < 0 || arg.nonrecBias >= 31)
              return LinIneq::unsat();
            return LinIneq(1 << arg.nonrecBias); // x >= 2**arg
          })
          .Case<AddExpr>([&](auto *expr) {
            return LinIneq::add(
                checkCycles(var, expr->lhs(), seenVars, reportInto, indent + 1),
                checkCycles(var, expr->rhs(), seenVars, reportInto,
                            indent + 1));
          })
          .Case<MaxExpr, MinExpr>([&](auto *expr) {
            // Combine the inequalities of the LHS and RHS into a single overly
            // pessimistic inequality. We treat `MinExpr` the same as `MaxExpr`,
            // since `max(a,b)` is an upper bound to `min(a,b)`.
            return LinIneq::max(
                checkCycles(var, expr->lhs(), seenVars, reportInto, indent + 1),
                checkCycles(var, expr->rhs(), seenVars, reportInto,
                            indent + 1));
          })
          .Default([](auto) { return LinIneq::unsat(); });

  // If we were passed an in-flight diagnostic and the current inequality is
  // unsatisfiable, attach notes to the diagnostic indicating the values or
  // operations that contributed to this part of the constraint expression.
  if (reportInto && !ineq.sat()) {
    auto report = [&](Location loc) {
      auto &note = reportInto->attachNote(loc);
      note << "constrained width W >= ";
      if (ineq.recScale == -1)
        note << "-";
      if (ineq.recScale != 1)
        note << ineq.recScale;
      note << "W";
      if (ineq.recBias < 0)
        note << "-" << -ineq.recBias;
      if (ineq.recBias > 0)
        note << "+" << ineq.recBias;
      note << " here:";
    };
    auto it = locs.find(expr);
    if (it != locs.end())
      for (auto loc : it->second)
        report(loc);
  }
  if (!reportInto)
    LLVM_DEBUG(llvm::dbgs().indent(indent * 2)
               << "- Visited " << *expr << ": " << ineq << "\n");

  return ineq;
}

using ExprSolution = std::pair<std::optional<int32_t>, bool>;

static ExprSolution
computeUnary(ExprSolution arg, llvm::function_ref<int32_t(int32_t)> operation) {
  if (arg.first)
    arg.first = operation(*arg.first);
  return arg;
}

static ExprSolution
computeBinary(ExprSolution lhs, ExprSolution rhs,
              llvm::function_ref<int32_t(int32_t, int32_t)> operation) {
  auto result = ExprSolution{std::nullopt, lhs.second || rhs.second};
  if (lhs.first && rhs.first)
    result.first = operation(*lhs.first, *rhs.first);
  else if (lhs.first)
    result.first = lhs.first;
  else if (rhs.first)
    result.first = rhs.first;
  return result;
}

namespace {
struct Frame {
  Frame(Expr *expr, unsigned indent) : expr(expr), indent(indent) {}
  Expr *expr;
  // Indent is only used for debug logs.
  unsigned indent;
};
} // namespace

/// Compute the value of a constraint `expr`. `seenVars` is used as a recursion
/// breaker. Recursive variables are treated as zero. Returns the computed value
/// and a boolean indicating whether a recursion was detected. This may be used
/// to memoize the result of expressions in case they were not involved in a
/// cycle (which may alter their value from the perspective of a variable).
static ExprSolution solveExpr(Expr *expr, SmallPtrSetImpl<Expr *> &seenVars,
                              std::vector<Frame> &worklist) {
  worklist.clear();
  worklist.emplace_back(expr, 1);
  llvm::DenseMap<Expr *, ExprSolution> solvedExprs;

  while (!worklist.empty()) {
    auto &frame = worklist.back();
    auto indent = frame.indent;
    auto setSolution = [&](ExprSolution solution) {
      // Memoize the result.
      if (solution.first && !solution.second)
        frame.expr->setSolution(*solution.first);
      solvedExprs[frame.expr] = solution;

      // Produce some useful debug prints.
      LLVM_DEBUG({
        if (!isa<KnownExpr>(frame.expr)) {
          if (solution.first)
            llvm::dbgs().indent(indent * 2)
                << "= Solved " << *frame.expr << " = " << *solution.first;
          else
            llvm::dbgs().indent(indent * 2) << "= Skipped " << *frame.expr;
          llvm::dbgs() << " (" << (solution.second ? "cycle broken" : "unique")
                       << ")\n";
        }
      });

      worklist.pop_back();
    };

    // See if we have a memoized result we can return.
    if (frame.expr->getSolution()) {
      LLVM_DEBUG({
        if (!isa<KnownExpr>(frame.expr))
          llvm::dbgs().indent(indent * 2) << "- Cached " << *frame.expr << " = "
                                          << *frame.expr->getSolution() << "\n";
      });
      setSolution(ExprSolution{*frame.expr->getSolution(), false});
      continue;
    }

    // Otherwise compute the value of the expression.
    LLVM_DEBUG({
      if (!isa<KnownExpr>(frame.expr))
        llvm::dbgs().indent(indent * 2) << "- Solving " << *frame.expr << "\n";
    });

    TypeSwitch<Expr *>(frame.expr)
        .Case<KnownExpr>([&](auto *expr) {
          setSolution(ExprSolution{expr->getValue(), false});
        })
        .Case<VarExpr>([&](auto *expr) {
          if (solvedExprs.contains(expr->constraint)) {
            auto solution = solvedExprs[expr->constraint];
            // If we've solved the upper bound already, store the solution.
            // This will be explicitly solved for later if not computed as
            // part of the solving that resolved this constraint.
            // This should only happen if somehow the constraint is
            // solved before visiting this expression, so that our upperBound
            // was not added to the worklist such that it was handled first.
            if (expr->upperBound && solvedExprs.contains(expr->upperBound))
              expr->upperBoundSolution = solvedExprs[expr->upperBound].first;
            seenVars.erase(expr);
            // Constrain variables >= 0.
            if (solution.first && *solution.first < 0)
              solution.first = 0;
            return setSolution(solution);
          }

          // Unconstrained variables produce no solution.
          if (!expr->constraint)
            return setSolution(ExprSolution{std::nullopt, false});
          // Return no solution for recursions in the variables. This is sane
          // and will cause the expression to be ignored when computing the
          // parent, e.g. `a >= max(a, 1)` will become just `a >= 1`.
          if (!seenVars.insert(expr).second)
            return setSolution(ExprSolution{std::nullopt, true});

          worklist.emplace_back(expr->constraint, indent + 1);
          if (expr->upperBound)
            worklist.emplace_back(expr->upperBound, indent + 1);
        })
        .Case<IdExpr>([&](auto *expr) {
          if (solvedExprs.contains(expr->arg))
            return setSolution(solvedExprs[expr->arg]);
          worklist.emplace_back(expr->arg, indent + 1);
        })
        .Case<PowExpr>([&](auto *expr) {
          if (solvedExprs.contains(expr->arg))
            return setSolution(computeUnary(
                solvedExprs[expr->arg], [](int32_t arg) { return 1 << arg; }));

          worklist.emplace_back(expr->arg, indent + 1);
        })
        .Case<AddExpr>([&](auto *expr) {
          if (solvedExprs.contains(expr->lhs()) &&
              solvedExprs.contains(expr->rhs()))
            return setSolution(computeBinary(
                solvedExprs[expr->lhs()], solvedExprs[expr->rhs()],
                [](int32_t lhs, int32_t rhs) { return lhs + rhs; }));

          worklist.emplace_back(expr->lhs(), indent + 1);
          worklist.emplace_back(expr->rhs(), indent + 1);
        })
        .Case<MaxExpr>([&](auto *expr) {
          if (solvedExprs.contains(expr->lhs()) &&
              solvedExprs.contains(expr->rhs()))
            return setSolution(computeBinary(
                solvedExprs[expr->lhs()], solvedExprs[expr->rhs()],
                [](int32_t lhs, int32_t rhs) { return std::max(lhs, rhs); }));

          worklist.emplace_back(expr->lhs(), indent + 1);
          worklist.emplace_back(expr->rhs(), indent + 1);
        })
        .Case<MinExpr>([&](auto *expr) {
          if (solvedExprs.contains(expr->lhs()) &&
              solvedExprs.contains(expr->rhs()))
            return setSolution(computeBinary(
                solvedExprs[expr->lhs()], solvedExprs[expr->rhs()],
                [](int32_t lhs, int32_t rhs) { return std::min(lhs, rhs); }));

          worklist.emplace_back(expr->lhs(), indent + 1);
          worklist.emplace_back(expr->rhs(), indent + 1);
        })
        .Default([&](auto) {
          setSolution(ExprSolution{std::nullopt, false});
        });
  }

  return solvedExprs[expr];
}

/// Solve the constraint problem. This is a very simple implementation that
/// does not fully solve the problem if there are weird dependency cycles
/// present.
LogicalResult ConstraintSolver::solve() {
  LLVM_DEBUG({
    llvm::dbgs() << "\n";
    debugHeader("Constraints") << "\n\n";
    dumpConstraints(llvm::dbgs());
  });

  // Ensure that there are no adverse cycles around.
  LLVM_DEBUG({
    llvm::dbgs() << "\n";
    debugHeader("Checking for unbreakable loops") << "\n\n";
  });
  SmallPtrSet<Expr *, 16> seenVars;
  bool anyFailed = false;

  for (auto *var : varExprs) {
    if (!var->constraint)
      continue;
    LLVM_DEBUG(llvm::dbgs()
               << "- Checking " << *var << " >= " << *var->constraint << "\n");

    // Canonicalize the variable's constraint expression into a form that allows
    // us to easily determine if any recursion leads to an unsatisfiable
    // constraint. The `seenVars` set acts as a recursion breaker.
    seenVars.insert(var);
    auto ineq = checkCycles(var, var->constraint, seenVars);
    seenVars.clear();

    // If the constraint is satisfiable, we're done.
    // TODO: It's possible that this result is already sufficient to arrive at a
    // solution for the constraint, and the second pass further down is not
    // necessary. This would require more proper handling of `MinExpr` in the
    // cycle checking code.
    if (ineq.sat()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  = Breakable since " << ineq << " satisfiable\n");
      continue;
    }

    // If we arrive here, the constraint is not satisfiable at all. To provide
    // some guidance to the user, we call the cycle checking code again, but
    // this time with an in-flight diagnostic to attach notes indicating
    // unsatisfiable paths in the cycle.
    LLVM_DEBUG(llvm::dbgs()
               << "  = UNBREAKABLE since " << ineq << " unsatisfiable\n");
    anyFailed = true;
    for (auto fieldRef : info.find(var)->second) {
      // Depending on whether this value stems from an operation or not, create
      // an appropriate diagnostic identifying the value.
      auto *op = fieldRef.getDefiningOp();
      auto diag = op ? op->emitOpError()
                     : mlir::emitError(fieldRef.getValue().getLoc())
                           << "value ";
      diag << "is constrained to be wider than itself";

      // Re-run the cycle checking, but this time reporting into the diagnostic.
      seenVars.insert(var);
      checkCycles(var, var->constraint, seenVars, &diag);
      seenVars.clear();
    }
  }

  // If there were cycles, return now to avoid complaining to the user about
  // dependent widths not being inferred.
  if (anyFailed)
    return failure();

  // Iterate over the constraint variables and solve each.
  LLVM_DEBUG({
    llvm::dbgs() << "\n";
    debugHeader("Solving constraints") << "\n\n";
  });
  std::vector<Frame> worklist;
  for (auto *var : varExprs) {
    // Complain about unconstrained variables.
    if (!var->constraint) {
      LLVM_DEBUG(llvm::dbgs() << "- Unconstrained " << *var << "\n");
      emitUninferredWidthError(var);
      anyFailed = true;
      continue;
    }

    // Compute the value for the variable.
    LLVM_DEBUG(llvm::dbgs()
               << "- Solving " << *var << " >= " << *var->constraint << "\n");
    seenVars.insert(var);
    auto solution = solveExpr(var->constraint, seenVars, worklist);
    // Compute the upperBound if there is one and haven't already.
    if (var->upperBound && !var->upperBoundSolution)
      var->upperBoundSolution =
          solveExpr(var->upperBound, seenVars, worklist).first;
    seenVars.clear();

    // Constrain variables >= 0.
    if (solution.first) {
      if (*solution.first < 0)
        solution.first = 0;
      var->setSolution(*solution.first);
    }

    // In case the width could not be inferred, complain to the user. This might
    // be the case if the width depends on an unconstrained variable.
    if (!solution.first) {
      LLVM_DEBUG(llvm::dbgs() << "  - UNSOLVED " << *var << "\n");
      emitUninferredWidthError(var);
      anyFailed = true;
      continue;
    }
    LLVM_DEBUG(llvm::dbgs()
               << "  = Solved " << *var << " = " << solution.first << " ("
               << (solution.second ? "cycle broken" : "unique") << ")\n");

    // Check if the solution we have found violates an upper bound.
    if (var->upperBoundSolution && var->upperBoundSolution < *solution.first) {
      LLVM_DEBUG(llvm::dbgs() << "  ! Unsatisfiable " << *var
                              << " <= " << var->upperBoundSolution << "\n");
      emitUninferredWidthError(var);
      anyFailed = true;
    }
  }

  // Copy over derived widths.
  for (auto *derived : derivedExprs) {
    auto *assigned = derived->assigned;
    if (!assigned || !assigned->getSolution()) {
      LLVM_DEBUG(llvm::dbgs() << "- Unused " << *derived << " set to 0\n");
      derived->setSolution(0);
    } else {
      LLVM_DEBUG(llvm::dbgs() << "- Deriving " << *derived << " = "
                              << assigned->getSolution() << "\n");
      derived->setSolution(*assigned->getSolution());
    }
  }

  return failure(anyFailed);
}

// Emits the diagnostic to inform the user about an uninferred width in the
// design. Returns true if an error was reported, false otherwise.
void ConstraintSolver::emitUninferredWidthError(VarExpr *var) {
  FieldRef fieldRef = info.find(var)->second.back();
  Value value = fieldRef.getValue();

  auto diag = mlir::emitError(value.getLoc(), "uninferred width:");

  // Try to hint the user at what kind of node this is.
  if (isa<BlockArgument>(value)) {
    diag << " port";
  } else if (auto *op = value.getDefiningOp()) {
    TypeSwitch<Operation *>(op)
        .Case<WireOp>([&](auto) { diag << " wire"; })
        .Case<RegOp, RegResetOp>([&](auto) { diag << " reg"; })
        .Case<NodeOp>([&](auto) { diag << " node"; })
        .Default([&](auto) { diag << " value"; });
  } else {
    diag << " value";
  }

  // Actually print what the user can refer to.
  auto [fieldName, rootKnown] = getFieldName(fieldRef);
  if (!fieldName.empty()) {
    if (!rootKnown)
      diag << " field";
    diag << " \"" << fieldName << "\"";
  }

  if (!var->constraint) {
    diag << " is unconstrained";
  } else if (var->getSolution() && var->upperBoundSolution &&
             var->getSolution() > var->upperBoundSolution) {
    diag << " cannot satisfy all width requirements";
    LLVM_DEBUG(llvm::dbgs() << *var->constraint << "\n");
    LLVM_DEBUG(llvm::dbgs() << *var->upperBound << "\n");
    auto loc = locs.find(var->constraint)->second.back();
    diag.attachNote(loc) << "width is constrained to be at least "
                         << *var->getSolution() << " here:";
    loc = locs.find(var->upperBound)->second.back();
    diag.attachNote(loc) << "width is constrained to be at most "
                         << *var->upperBoundSolution << " here:";
  } else {
    diag << " width cannot be determined";
    LLVM_DEBUG(llvm::dbgs() << *var->constraint << "\n");
    auto loc = locs.find(var->constraint)->second.back();
    diag.attachNote(loc) << "width is constrained by an uninferred width here:";
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
  InferenceMapping(ConstraintSolver &solver, SymbolTable &symtbl,
                   hw::InnerSymbolTableCollection &istc)
      : solver(solver), symtbl(symtbl), irn{symtbl, istc} {}

  LogicalResult map(CircuitOp op);
  bool allWidthsKnown(Operation *op);
  LogicalResult mapOperation(Operation *op);

  /// Declare all the variables in the value. If the value is a ground type,
  /// there is a single variable declared.  If the value is an aggregate type,
  /// it sets up variables for each unknown width.
  void declareVars(Value value, bool isDerived = false);

  /// Assign the constraint expressions of the fields in the `result` argument
  /// as the max of expressions in the `rhs` and `lhs` arguments. Both fields
  /// must be the same type.
  void maximumOfTypes(Value result, Value rhs, Value lhs);

  /// Constrain the value "larger" to be greater than or equal to "smaller".
  /// These may be aggregate values. This is used for regular connects.
  void constrainTypes(Value larger, Value smaller, bool equal = false);

  /// Constrain the expression "larger" to be greater than or equals to
  /// the expression "smaller".
  void constrainTypes(Expr *larger, Expr *smaller,
                      bool imposeUpperBounds = false, bool equal = false);

  /// Assign the constraint expressions of the fields in the `src` argument as
  /// the expressions for the `dst` argument. Both fields must be of the given
  /// `type`.
  void unifyTypes(FieldRef lhs, FieldRef rhs, FIRRTLType type);

  /// Get the expr associated with the value.  The value must be a non-aggregate
  /// type
  Expr *getExpr(Value value) const;

  /// Get the expr associated with a specific field in a value.
  Expr *getExpr(FieldRef fieldRef) const;

  /// Get the expr associated with a specific field in a value. If value is
  /// NULL, then this returns NULL.
  Expr *getExprOrNull(FieldRef fieldRef) const;

  /// Set the expr associated with the value. The value must be a non-aggregate
  /// type.
  void setExpr(Value value, Expr *expr);

  /// Set the expr associated with a specific field in a value.
  void setExpr(FieldRef fieldRef, Expr *expr);

  /// Return whether a module was skipped due to being fully inferred already.
  bool isModuleSkipped(FModuleOp module) const {
    return skippedModules.count(module);
  }

  /// Return whether all modules in the mapping were fully inferred.
  bool areAllModulesSkipped() const { return allModulesSkipped; }

private:
  /// The constraint solver into which we emit variables and constraints.
  ConstraintSolver &solver;

  /// The constraint exprs for each result type of an operation.
  DenseMap<FieldRef, Expr *> opExprs;

  /// The fully inferred modules that were skipped entirely.
  SmallPtrSet<Operation *, 16> skippedModules;
  bool allModulesSkipped = true;

  /// Cache of module symbols
  SymbolTable &symtbl;

  /// Full design inner symbol information.
  hw::InnerRefNamespace irn;
};

} // namespace

/// Check if a type contains any FIRRTL type with uninferred widths.
static bool hasUninferredWidth(Type type) {
  return TypeSwitch<Type, bool>(type)
      .Case<FIRRTLBaseType>([](auto base) { return base.hasUninferredWidth(); })
      .Case<RefType>(
          [](auto ref) { return ref.getType().hasUninferredWidth(); })
      .Default([](auto) { return false; });
}

LogicalResult InferenceMapping::map(CircuitOp op) {
  LLVM_DEBUG(llvm::dbgs()
             << "\n===----- Mapping ops to constraint exprs -----===\n\n");

  // Ensure we have constraint variables established for all module ports.
  for (auto module : op.getOps<FModuleOp>())
    for (auto arg : module.getArguments()) {
      solver.setCurrentContextInfo(FieldRef(arg, 0));
      declareVars(arg);
    }

  for (auto module : op.getOps<FModuleOp>()) {
    // Check if the module contains *any* uninferred widths. This allows us to
    // do an early skip if the module is already fully inferred.
    bool anyUninferred = false;
    for (auto arg : module.getArguments()) {
      anyUninferred |= hasUninferredWidth(arg.getType());
      if (anyUninferred)
        break;
    }
    module.walk([&](Operation *op) {
      for (auto type : op->getResultTypes())
        anyUninferred |= hasUninferredWidth(type);
      if (anyUninferred)
        return WalkResult::interrupt();
      return WalkResult::advance();
    });

    if (!anyUninferred) {
      LLVM_DEBUG(llvm::dbgs() << "Skipping fully-inferred module '"
                              << module.getName() << "'\n");
      skippedModules.insert(module);
      continue;
    }

    allModulesSkipped = false;

    // Go through operations in the module, creating type variables for results,
    // and generating constraints.
    auto result = module.getBodyBlock()->walk(
        [&](Operation *op) { return WalkResult(mapOperation(op)); });
    if (result.wasInterrupted())
      return failure();
  }

  return success();
}

bool InferenceMapping::allWidthsKnown(Operation *op) {
  /// Ignore property assignments, no widths to infer.
  if (isa<PropAssignOp>(op))
    return true;

  // If this is a mux, and the select signal is uninferred, we need to set an
  // upperbound limit on it.
  if (isa<MuxPrimOp, Mux4CellIntrinsicOp, Mux2CellIntrinsicOp>(op))
    if (hasUninferredWidth(op->getOperand(0).getType()))
      return false;

  //  We need to propagate through connects.
  if (isa<FConnectLike, AttachOp>(op))
    return false;

  // Check if we know the width of every result of this operation.
  return llvm::all_of(op->getResults(), [&](auto result) {
    // Only consider FIRRTL types for width constraints. Ignore any foreign
    // types as they don't participate in the width inference process.
    if (auto type = type_dyn_cast<FIRRTLType>(result.getType()))
      if (hasUninferredWidth(type))
        return false;
    return true;
  });
}

LogicalResult InferenceMapping::mapOperation(Operation *op) {
  if (allWidthsKnown(op))
    return success();

  // Actually generate the necessary constraint expressions.
  bool mappingFailed = false;
  solver.setCurrentContextInfo(
      op->getNumResults() > 0 ? FieldRef(op->getResults()[0], 0) : FieldRef());
  solver.setCurrentLocation(op->getLoc());
  TypeSwitch<Operation *>(op)
      .Case<ConstantOp>([&](auto op) {
        // If the constant has a known width, use that. Otherwise pick the
        // smallest number of bits necessary to represent the constant.
        auto v = op.getValue();
        auto w = v.getBitWidth() - (v.isNegative() ? v.countLeadingOnes()
                                                   : v.countLeadingZeros());
        if (v.isSigned())
          w += 1;
        setExpr(op.getResult(), solver.known(std::max(w, 1u)));
      })
      .Case<SpecialConstantOp>([&](auto op) {
        // Nothing required.
      })
      .Case<InvalidValueOp>([&](auto op) {
        // We must duplicate the invalid value for each use, since each use can
        // be inferred to a different width.
        declareVars(op.getResult(), /*isDerived=*/true);
      })
      .Case<WireOp, RegOp>([&](auto op) { declareVars(op.getResult()); })
      .Case<RegResetOp>([&](auto op) {
        // The original Scala code also constrains the reset signal to be at
        // least 1 bit wide. We don't do this here since the MLIR FIRRTL
        // dialect enforces the reset signal to be an async reset or a
        // `uint<1>`.
        declareVars(op.getResult());
        // Contrain the register to be greater than or equal to the reset
        // signal.
        constrainTypes(op.getResult(), op.getResetValue());
      })
      .Case<NodeOp>([&](auto op) {
        // Nodes have the same type as their input.
        unifyTypes(FieldRef(op.getResult(), 0), FieldRef(op.getInput(), 0),
                   op.getResult().getType());
      })

      // Aggregate Values
      .Case<SubfieldOp>([&](auto op) {
        BundleType bundleType = op.getInput().getType();
        auto fieldID = bundleType.getFieldID(op.getFieldIndex());
        unifyTypes(FieldRef(op.getResult(), 0),
                   FieldRef(op.getInput(), fieldID), op.getType());
      })
      .Case<SubindexOp, SubaccessOp>([&](auto op) {
        // All vec fields unify to the same thing. Always use the first element
        // of the vector, which has a field ID of 1.
        unifyTypes(FieldRef(op.getResult(), 0), FieldRef(op.getInput(), 1),
                   op.getType());
      })
      .Case<SubtagOp>([&](auto op) {
        FEnumType enumType = op.getInput().getType();
        auto fieldID = enumType.getFieldID(op.getFieldIndex());
        unifyTypes(FieldRef(op.getResult(), 0),
                   FieldRef(op.getInput(), fieldID), op.getType());
      })

      .Case<RefSubOp>([&](RefSubOp op) {
        uint64_t fieldID = TypeSwitch<FIRRTLBaseType, uint64_t>(
                               op.getInput().getType().getType())
                               .Case<FVectorType>([](auto _) { return 1; })
                               .Case<BundleType>([&](auto type) {
                                 return type.getFieldID(op.getIndex());
                               });
        unifyTypes(FieldRef(op.getResult(), 0),
                   FieldRef(op.getInput(), fieldID), op.getType());
      })

      // Arithmetic and Logical Binary Primitives
      .Case<AddPrimOp, SubPrimOp>([&](auto op) {
        auto lhs = getExpr(op.getLhs());
        auto rhs = getExpr(op.getRhs());
        auto e = solver.add(solver.max(lhs, rhs), solver.known(1));
        setExpr(op.getResult(), e);
      })
      .Case<MulPrimOp>([&](auto op) {
        auto lhs = getExpr(op.getLhs());
        auto rhs = getExpr(op.getRhs());
        auto e = solver.add(lhs, rhs);
        setExpr(op.getResult(), e);
      })
      .Case<DivPrimOp>([&](auto op) {
        auto lhs = getExpr(op.getLhs());
        Expr *e;
        if (op.getType().base().isSigned()) {
          e = solver.add(lhs, solver.known(1));
        } else {
          e = lhs;
        }
        setExpr(op.getResult(), e);
      })
      .Case<RemPrimOp>([&](auto op) {
        auto lhs = getExpr(op.getLhs());
        auto rhs = getExpr(op.getRhs());
        auto e = solver.min(lhs, rhs);
        setExpr(op.getResult(), e);
      })
      .Case<AndPrimOp, OrPrimOp, XorPrimOp>([&](auto op) {
        auto lhs = getExpr(op.getLhs());
        auto rhs = getExpr(op.getRhs());
        auto e = solver.max(lhs, rhs);
        setExpr(op.getResult(), e);
      })

      .Case<CatPrimOp>([&](auto op) {
        if (op.getInputs().empty()) {
          setExpr(op.getResult(), solver.known(0));
          return;
        }
        auto result = getExpr(op.getInputs().front());
        for (auto operand : op.getInputs().drop_front()) {
          auto operandExpr = getExpr(operand);
          result = solver.add(result, operandExpr);
        }
        setExpr(op.getResult(), result);
      })
      // Misc Binary Primitives
      .Case<DShlPrimOp>([&](auto op) {
        auto lhs = getExpr(op.getLhs());
        auto rhs = getExpr(op.getRhs());
        auto e = solver.add(lhs, solver.add(solver.pow(rhs), solver.known(-1)));
        setExpr(op.getResult(), e);
      })
      .Case<DShlwPrimOp, DShrPrimOp>([&](auto op) {
        auto e = getExpr(op.getLhs());
        setExpr(op.getResult(), e);
      })

      // Unary operators
      .Case<NegPrimOp>([&](auto op) {
        auto input = getExpr(op.getInput());
        auto e = solver.add(input, solver.known(1));
        setExpr(op.getResult(), e);
      })
      .Case<CvtPrimOp>([&](auto op) {
        auto input = getExpr(op.getInput());
        auto e = op.getInput().getType().base().isSigned()
                     ? input
                     : solver.add(input, solver.known(1));
        setExpr(op.getResult(), e);
      })

      // Miscellaneous
      .Case<BitsPrimOp>([&](auto op) {
        setExpr(op.getResult(), solver.known(op.getHi() - op.getLo() + 1));
      })
      .Case<HeadPrimOp>([&](auto op) {
        setExpr(op.getResult(), solver.known(op.getAmount()));
      })
      .Case<TailPrimOp>([&](auto op) {
        auto input = getExpr(op.getInput());
        auto e = solver.add(input, solver.known(-op.getAmount()));
        setExpr(op.getResult(), e);
      })
      .Case<PadPrimOp>([&](auto op) {
        auto input = getExpr(op.getInput());
        auto e = solver.max(input, solver.known(op.getAmount()));
        setExpr(op.getResult(), e);
      })
      .Case<ShlPrimOp>([&](auto op) {
        auto input = getExpr(op.getInput());
        auto e = solver.add(input, solver.known(op.getAmount()));
        setExpr(op.getResult(), e);
      })
      .Case<ShrPrimOp>([&](auto op) {
        auto input = getExpr(op.getInput());
        // UInt saturates at 0 bits, SInt at 1 bit
        auto minWidth = op.getInput().getType().base().isUnsigned() ? 0 : 1;
        auto e = solver.max(solver.add(input, solver.known(-op.getAmount())),
                            solver.known(minWidth));
        setExpr(op.getResult(), e);
      })

      // Handle operations whose output width matches the input width.
      .Case<NotPrimOp, AsSIntPrimOp, AsUIntPrimOp, ConstCastOp>(
          [&](auto op) { setExpr(op.getResult(), getExpr(op.getInput())); })
      .Case<mlir::UnrealizedConversionCastOp>(
          [&](auto op) { setExpr(op.getResult(0), getExpr(op.getOperand(0))); })

      // Handle operations with a single result type that always has a
      // well-known width.
      .Case<LEQPrimOp, LTPrimOp, GEQPrimOp, GTPrimOp, EQPrimOp, NEQPrimOp,
            AsClockPrimOp, AsAsyncResetPrimOp, AndRPrimOp, OrRPrimOp,
            XorRPrimOp>([&](auto op) {
        auto width = op.getType().getBitWidthOrSentinel();
        assert(width > 0 && "width should have been checked by verifier");
        setExpr(op.getResult(), solver.known(width));
      })
      .Case<MuxPrimOp, Mux2CellIntrinsicOp>([&](auto op) {
        auto *sel = getExpr(op.getSel());
        constrainTypes(solver.known(1), sel, /*imposeUpperBounds=*/true);
        maximumOfTypes(op.getResult(), op.getHigh(), op.getLow());
      })
      .Case<Mux4CellIntrinsicOp>([&](Mux4CellIntrinsicOp op) {
        auto *sel = getExpr(op.getSel());
        constrainTypes(solver.known(2), sel, /*imposeUpperBounds=*/true);
        maximumOfTypes(op.getResult(), op.getV3(), op.getV2());
        maximumOfTypes(op.getResult(), op.getResult(), op.getV1());
        maximumOfTypes(op.getResult(), op.getResult(), op.getV0());
      })

      .Case<ConnectOp, MatchingConnectOp>(
          [&](auto op) { constrainTypes(op.getDest(), op.getSrc()); })
      .Case<RefDefineOp>([&](auto op) {
        // Dest >= Src, but also check Src <= Dest for correctness
        // (but don't solve to make this true, don't back-propagate)
        constrainTypes(op.getDest(), op.getSrc(), true);
      })
      .Case<AttachOp>([&](auto op) {
        // Attach connects multiple analog signals together. All signals must
        // have the same bit width. Signals without bit width inherit from the
        // other signals.
        if (op.getAttached().empty())
          return;
        auto prev = op.getAttached()[0];
        for (auto operand : op.getAttached().drop_front()) {
          auto e1 = getExpr(prev);
          auto e2 = getExpr(operand);
          constrainTypes(e1, e2, /*imposeUpperBounds=*/true);
          constrainTypes(e2, e1, /*imposeUpperBounds=*/true);
          prev = operand;
        }
      })

      // Handle the no-ops that don't interact with width inference.
      .Case<PrintFOp, FFlushOp, SkipOp, StopOp, WhenOp, AssertOp, AssumeOp,
            UnclockedAssumeIntrinsicOp, CoverOp>([&](auto) {})

      // Handle instances of other modules.
      .Case<InstanceOp>([&](auto op) {
        auto refdModule = op.getReferencedOperation(symtbl);
        auto module = dyn_cast<FModuleOp>(&*refdModule);
        if (!module) {
          auto diag = mlir::emitError(op.getLoc());
          diag << "extern module `" << op.getModuleName()
               << "` has ports of uninferred width";

          auto fml = cast<FModuleLike>(&*refdModule);
          auto ports = fml.getPorts();
          for (auto &port : ports) {
            auto baseType = getBaseType(port.type);
            if (baseType && baseType.hasUninferredWidth()) {
              diag.attachNote(op.getLoc()) << "Port: " << port.name;
              if (!baseType.isGround())
                diagnoseUninferredType(diag, baseType, port.name.getValue());
            }
          }

          diag.attachNote(op.getLoc())
              << "Only non-extern FIRRTL modules may contain unspecified "
                 "widths to be inferred automatically.";
          diag.attachNote(refdModule->getLoc())
              << "Module `" << op.getModuleName() << "` defined here:";
          mappingFailed = true;
          return;
        }
        // Simply look up the free variables created for the instantiated
        // module's ports, and use them for instance port wires. This way,
        // constraints imposed onto the ports of the instance will transparently
        // apply to the ports of the instantiated module.
        for (auto [result, arg] :
             llvm::zip(op->getResults(), module.getArguments()))
          unifyTypes({result, 0}, {arg, 0},
                     type_cast<FIRRTLType>(result.getType()));
      })

      // Handle memories.
      .Case<MemOp>([&](MemOp op) {
        // Create constraint variables for all ports.
        unsigned nonDebugPort = 0;
        for (const auto &result : llvm::enumerate(op.getResults())) {
          declareVars(result.value());
          if (!type_isa<RefType>(result.value().getType()))
            nonDebugPort = result.index();
        }

        // A helper function that returns the indeces of the "data", "rdata",
        // and "wdata" fields in the bundle corresponding to a memory port.
        auto dataFieldIndices = [](MemOp::PortKind kind) -> ArrayRef<unsigned> {
          static const unsigned indices[] = {3, 5};
          static const unsigned debug[] = {0};
          switch (kind) {
          case MemOp::PortKind::Read:
          case MemOp::PortKind::Write:
            return ArrayRef<unsigned>(indices, 1); // {3}
          case MemOp::PortKind::ReadWrite:
            return ArrayRef<unsigned>(indices); // {3, 5}
          case MemOp::PortKind::Debug:
            return ArrayRef<unsigned>(debug);
          }
          llvm_unreachable("Imposible PortKind");
        };

        // This creates independent variables for every data port. Yet, what we
        // actually want is for all data ports to share the same variable. To do
        // this, we find the first data port declared, and use that port's vars
        // for all the other ports.
        unsigned firstFieldIndex =
            dataFieldIndices(op.getPortKind(nonDebugPort))[0];
        FieldRef firstData(
            op.getResult(nonDebugPort),
            type_cast<BundleType>(op.getPortType(nonDebugPort).getPassiveType())
                .getFieldID(firstFieldIndex));
        LLVM_DEBUG(llvm::dbgs() << "Adjusting memory port variables:\n");

        // Reuse data port variables.
        auto dataType = op.getDataType();
        for (unsigned i = 0, e = op.getResults().size(); i < e; ++i) {
          auto result = op.getResult(i);
          if (type_isa<RefType>(result.getType())) {
            // Debug ports are firrtl.ref<vector<data-type, depth>>
            // Use FieldRef of 1, to indicate the first vector element must be
            // of the dataType.
            unifyTypes(firstData, FieldRef(result, 1), dataType);
            continue;
          }

          auto portType =
              type_cast<BundleType>(op.getPortType(i).getPassiveType());
          for (auto fieldIndex : dataFieldIndices(op.getPortKind(i)))
            unifyTypes(FieldRef(result, portType.getFieldID(fieldIndex)),
                       firstData, dataType);
        }
      })

      .Case<RefSendOp>([&](auto op) {
        declareVars(op.getResult());
        constrainTypes(op.getResult(), op.getBase(), true);
      })
      .Case<RefResolveOp>([&](auto op) {
        declareVars(op.getResult());
        constrainTypes(op.getResult(), op.getRef(), true);
      })
      .Case<RefCastOp>([&](auto op) {
        declareVars(op.getResult());
        constrainTypes(op.getResult(), op.getInput(), true);
      })
      .Case<RWProbeOp>([&](auto op) {
        auto ist = irn.lookup(op.getTarget());
        if (!ist) {
          op->emitError("target of rwprobe could not be resolved");
          mappingFailed = true;
          return;
        }
        auto ref = getFieldRefForTarget(ist);
        if (!ref) {
          op->emitError("target of rwprobe resolved to unsupported target");
          mappingFailed = true;
          return;
        }
        auto newFID = convertFieldIDToOurVersion(
            ref.getFieldID(), type_cast<FIRRTLType>(ref.getValue().getType()));
        unifyTypes(FieldRef(op.getResult(), 0),
                   FieldRef(ref.getValue(), newFID), op.getType());
      })
      .Case<mlir::UnrealizedConversionCastOp>([&](auto op) {
        for (Value result : op.getResults()) {
          auto ty = result.getType();
          if (type_isa<FIRRTLType>(ty))
            declareVars(result);
        }
      })
      .Default([&](auto op) {
        op->emitOpError("not supported in width inference");
        mappingFailed = true;
      });

  // Forceable declarations should have the ref constrained to data result.
  if (auto fop = dyn_cast<Forceable>(op); fop && fop.isForceable())
    unifyTypes(FieldRef(fop.getDataRef(), 0), FieldRef(fop.getDataRaw(), 0),
               fop.getDataType());

  return failure(mappingFailed);
}

/// Declare free variables for the type of a value, and associate the resulting
/// set of variables with that value.
void InferenceMapping::declareVars(Value value, bool isDerived) {
  // Declare a variable for every unknown width in the type. If this is a Bundle
  // type or a FVector type, we will have to potentially create many variables.
  unsigned fieldID = 0;
  std::function<void(FIRRTLBaseType)> declare = [&](FIRRTLBaseType type) {
    auto width = type.getBitWidthOrSentinel();
    if (width >= 0) {
      fieldID++;
    } else if (width == -1) {
      // Unknown width integers create a variable.
      FieldRef field(value, fieldID);
      solver.setCurrentContextInfo(field);
      if (isDerived)
        setExpr(field, solver.derived());
      else
        setExpr(field, solver.var());
      fieldID++;
    } else if (auto bundleType = type_dyn_cast<BundleType>(type)) {
      // Bundle types recursively declare all bundle elements.
      fieldID++;
      for (auto &element : bundleType)
        declare(element.type);
    } else if (auto vecType = type_dyn_cast<FVectorType>(type)) {
      fieldID++;
      auto save = fieldID;
      declare(vecType.getElementType());
      // Skip past the rest of the elements
      fieldID = save + vecType.getMaxFieldID();
    } else if (auto enumType = type_dyn_cast<FEnumType>(type)) {
      fieldID++;
      for (auto &element : enumType.getElements())
        declare(element.type);
    } else {
      llvm_unreachable("Unknown type inside a bundle!");
    }
  };
  if (auto type = getBaseType(value.getType()))
    declare(type);
}

/// Assign the constraint expressions of the fields in the `result` argument as
/// the max of expressions in the `rhs` and `lhs` arguments. Both fields must be
/// the same type.
void InferenceMapping::maximumOfTypes(Value result, Value rhs, Value lhs) {
  // Recurse to every leaf element and set larger >= smaller.
  auto fieldID = 0;
  std::function<void(FIRRTLBaseType)> maximize = [&](FIRRTLBaseType type) {
    if (auto bundleType = type_dyn_cast<BundleType>(type)) {
      fieldID++;
      for (auto &element : bundleType.getElements())
        maximize(element.type);
    } else if (auto vecType = type_dyn_cast<FVectorType>(type)) {
      fieldID++;
      auto save = fieldID;
      // Skip 0 length vectors.
      if (vecType.getNumElements() > 0)
        maximize(vecType.getElementType());
      fieldID = save + vecType.getMaxFieldID();
    } else if (auto enumType = type_dyn_cast<FEnumType>(type)) {
      fieldID++;
      for (auto &element : enumType.getElements())
        maximize(element.type);
    } else if (type.isGround()) {
      auto *e = solver.max(getExpr(FieldRef(rhs, fieldID)),
                           getExpr(FieldRef(lhs, fieldID)));
      setExpr(FieldRef(result, fieldID), e);
      fieldID++;
    } else {
      llvm_unreachable("Unknown type inside a bundle!");
    }
  };
  if (auto type = getBaseType(result.getType()))
    maximize(type);
}

/// Establishes constraints to ensure the sizes in the `larger` type are greater
/// than or equal to the sizes in the `smaller` type. Types have to be
/// compatible in the sense that they may only differ in the presence or absence
/// of bit widths.
///
/// This function is used to apply regular connects.
/// Set `equal` for constraining larger <= smaller for correctness but not
/// solving.
void InferenceMapping::constrainTypes(Value larger, Value smaller, bool equal) {
  // Recurse to every leaf element and set larger >= smaller. Ignore foreign
  // types as these do not participate in width inference.

  auto fieldID = 0;
  std::function<void(FIRRTLBaseType, Value, Value)> constrain =
      [&](FIRRTLBaseType type, Value larger, Value smaller) {
        if (auto bundleType = type_dyn_cast<BundleType>(type)) {
          fieldID++;
          for (auto &element : bundleType.getElements()) {
            if (element.isFlip)
              constrain(element.type, smaller, larger);
            else
              constrain(element.type, larger, smaller);
          }
        } else if (auto vecType = type_dyn_cast<FVectorType>(type)) {
          fieldID++;
          auto save = fieldID;
          // Skip 0 length vectors.
          if (vecType.getNumElements() > 0) {
            constrain(vecType.getElementType(), larger, smaller);
          }
          fieldID = save + vecType.getMaxFieldID();
        } else if (auto enumType = type_dyn_cast<FEnumType>(type)) {
          fieldID++;
          for (auto &element : enumType.getElements())
            constrain(element.type, larger, smaller);
        } else if (type.isGround()) {
          // Leaf element, look up their expressions, and create the constraint.
          constrainTypes(getExpr(FieldRef(larger, fieldID)),
                         getExpr(FieldRef(smaller, fieldID)), false, equal);
          fieldID++;
        } else {
          llvm_unreachable("Unknown type inside a bundle!");
        }
      };

  if (auto type = getBaseType(larger.getType()))
    constrain(type, larger, smaller);
}

/// Establishes constraints to ensure the sizes in the `larger` type are greater
/// than or equal to the sizes in the `smaller` type.
void InferenceMapping::constrainTypes(Expr *larger, Expr *smaller,
                                      bool imposeUpperBounds, bool equal) {
  assert(larger && "Larger expression should be specified");
  assert(smaller && "Smaller expression should be specified");

  // If one of the sides is `DerivedExpr`, simply assign the other side as the
  // derived width. This allows `InvalidValueOp`s to properly infer their width
  // from the connects they are used in, but also be inferred to something
  // useful on their own.
  if (auto *largerDerived = dyn_cast<DerivedExpr>(larger)) {
    largerDerived->assigned = smaller;
    LLVM_DEBUG(llvm::dbgs() << "Deriving " << *largerDerived << " from "
                            << *smaller << "\n");
    return;
  }
  if (auto *smallerDerived = dyn_cast<DerivedExpr>(smaller)) {
    smallerDerived->assigned = larger;
    LLVM_DEBUG(llvm::dbgs() << "Deriving " << *smallerDerived << " from "
                            << *larger << "\n");
    return;
  }

  // If the larger expr is a free variable, create a `expr >= x` constraint for
  // it that we can try to satisfy with the smallest width.
  if (auto *largerVar = dyn_cast<VarExpr>(larger)) {
    [[maybe_unused]] auto *c = solver.addGeqConstraint(largerVar, smaller);
    LLVM_DEBUG(llvm::dbgs()
               << "Constrained " << *largerVar << " >= " << *c << "\n");
    // If we're constraining larger == smaller, add the LEQ contraint as well.
    // Solve for GEQ but check that LEQ is true.
    // Used for matchingconnect, some reference operations, and anywhere the
    // widths should be inferred strictly in one direction but are required to
    // also be equal for correctness.
    if (equal) {
      [[maybe_unused]] auto *leq = solver.addLeqConstraint(largerVar, smaller);
      LLVM_DEBUG(llvm::dbgs()
                 << "Constrained " << *largerVar << " <= " << *leq << "\n");
    }
    return;
  }

  // If the smaller expr is a free variable but the larger one is not, create a
  // `expr <= k` upper bound that we can verify once all lower bounds have been
  // satisfied. Since we are always picking the smallest width to satisfy all
  // `>=` constraints, any `<=` constraints have no effect on the solution
  // besides indicating that a width is unsatisfiable.
  if (auto *smallerVar = dyn_cast<VarExpr>(smaller)) {
    if (imposeUpperBounds || equal) {
      [[maybe_unused]] auto *c = solver.addLeqConstraint(smallerVar, larger);
      LLVM_DEBUG(llvm::dbgs()
                 << "Constrained " << *smallerVar << " <= " << *c << "\n");
    }
  }
}

/// Assign the constraint expressions of the fields in the `src` argument as the
/// expressions for the `dst` argument. Both fields must be of the given `type`.
void InferenceMapping::unifyTypes(FieldRef lhs, FieldRef rhs, FIRRTLType type) {
  // Fast path for `unifyTypes(x, x, _)`.
  if (lhs == rhs)
    return;

  // Co-iterate the two field refs, recurring into every leaf element and set
  // them equal.
  auto fieldID = 0;
  std::function<void(FIRRTLBaseType)> unify = [&](FIRRTLBaseType type) {
    if (type.isGround()) {
      // Leaf element, unify the fields!
      FieldRef lhsFieldRef(lhs.getValue(), lhs.getFieldID() + fieldID);
      FieldRef rhsFieldRef(rhs.getValue(), rhs.getFieldID() + fieldID);
      LLVM_DEBUG(llvm::dbgs()
                 << "Unify " << getFieldName(lhsFieldRef).first << " = "
                 << getFieldName(rhsFieldRef).first << "\n");
      // Abandon variables becoming unconstrainable by the unification.
      if (auto *var = dyn_cast_or_null<VarExpr>(getExprOrNull(lhsFieldRef)))
        solver.addGeqConstraint(var, solver.known(0));
      setExpr(lhsFieldRef, getExpr(rhsFieldRef));
      fieldID++;
    } else if (auto bundleType = type_dyn_cast<BundleType>(type)) {
      fieldID++;
      for (auto &element : bundleType) {
        unify(element.type);
      }
    } else if (auto vecType = type_dyn_cast<FVectorType>(type)) {
      fieldID++;
      auto save = fieldID;
      // Skip 0 length vectors.
      if (vecType.getNumElements() > 0) {
        unify(vecType.getElementType());
      }
      fieldID = save + vecType.getMaxFieldID();
    } else if (auto enumType = type_dyn_cast<FEnumType>(type)) {
      fieldID++;
      for (auto &element : enumType.getElements())
        unify(element.type);
    } else {
      llvm_unreachable("Unknown type inside a bundle!");
    }
  };
  if (auto ftype = getBaseType(type))
    unify(ftype);
}

/// Get the constraint expression for a value.
Expr *InferenceMapping::getExpr(Value value) const {
  assert(type_cast<FIRRTLType>(getBaseType(value.getType())).isGround());
  // A field ID of 0 indicates the entire value.
  return getExpr(FieldRef(value, 0));
}

/// Get the constraint expression for a value.
Expr *InferenceMapping::getExpr(FieldRef fieldRef) const {
  auto *expr = getExprOrNull(fieldRef);
  assert(expr && "constraint expr should have been constructed for value");
  return expr;
}

Expr *InferenceMapping::getExprOrNull(FieldRef fieldRef) const {
  auto it = opExprs.find(fieldRef);
  if (it != opExprs.end())
    return it->second;
  // If we don't have an expression for this fieldRef, it should have a
  // constant width.
  auto baseType = getBaseType(fieldRef.getValue().getType());
  auto type =
      hw::FieldIdImpl::getFinalTypeByFieldID(baseType, fieldRef.getFieldID());
  auto width = cast<FIRRTLBaseType>(type).getBitWidthOrSentinel();
  if (width < 0)
    return nullptr;
  return solver.known(width);
}

/// Associate a constraint expression with a value.
void InferenceMapping::setExpr(Value value, Expr *expr) {
  assert(type_cast<FIRRTLType>(getBaseType(value.getType())).isGround());
  // A field ID of 0 indicates the entire value.
  setExpr(FieldRef(value, 0), expr);
}

/// Associate a constraint expression with a value.
void InferenceMapping::setExpr(FieldRef fieldRef, Expr *expr) {
  LLVM_DEBUG({
    llvm::dbgs() << "Expr " << *expr << " for " << fieldRef.getValue();
    if (fieldRef.getFieldID())
      llvm::dbgs() << " '" << getFieldName(fieldRef).first << "'";
    auto fieldName = getFieldName(fieldRef);
    if (fieldName.second)
      llvm::dbgs() << " (\"" << fieldName.first << "\")";
    llvm::dbgs() << "\n";
  });
  opExprs[fieldRef] = expr;
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

  LogicalResult update(CircuitOp op);
  FailureOr<bool> updateOperation(Operation *op);
  FailureOr<bool> updateValue(Value value);
  FIRRTLBaseType updateType(FieldRef fieldRef, FIRRTLBaseType type);

private:
  const InferenceMapping &mapping;
};

} // namespace

/// Update the types throughout a circuit.
LogicalResult InferenceTypeUpdate::update(CircuitOp op) {
  LLVM_DEBUG({
    llvm::dbgs() << "\n";
    debugHeader("Update types") << "\n\n";
  });
  return mlir::failableParallelForEach(
      op.getContext(), op.getOps<FModuleOp>(), [&](FModuleOp op) {
        // Skip this module if it had no widths to be
        // inferred at all.
        if (mapping.isModuleSkipped(op))
          return success();
        auto isFailed = op.walk<WalkOrder::PreOrder>([&](Operation *op) {
                            if (failed(updateOperation(op)))
                              return WalkResult::interrupt();
                            return WalkResult::advance();
                          }).wasInterrupted();
        return failure(isFailed);
      });
}

/// Update the result types of an operation.
FailureOr<bool> InferenceTypeUpdate::updateOperation(Operation *op) {
  bool anyChanged = false;

  for (Value v : op->getResults()) {
    auto result = updateValue(v);
    if (failed(result))
      return result;
    anyChanged |= *result;
  }

  // If this is a connect operation, width inference might have inferred a RHS
  // that is wider than the LHS, in which case an additional BitsPrimOp is
  // necessary to truncate the value.
  if (auto con = dyn_cast<ConnectOp>(op)) {
    auto lhs = con.getDest();
    auto rhs = con.getSrc();
    auto lhsType = type_dyn_cast<FIRRTLBaseType>(lhs.getType());
    auto rhsType = type_dyn_cast<FIRRTLBaseType>(rhs.getType());

    // Nothing to do if not base types.
    if (!lhsType || !rhsType)
      return anyChanged;

    auto lhsWidth = lhsType.getBitWidthOrSentinel();
    auto rhsWidth = rhsType.getBitWidthOrSentinel();
    if (lhsWidth >= 0 && rhsWidth >= 0 && lhsWidth < rhsWidth) {
      OpBuilder builder(op);
      auto trunc = builder.createOrFold<TailPrimOp>(con.getLoc(), con.getSrc(),
                                                    rhsWidth - lhsWidth);
      if (type_isa<SIntType>(rhsType))
        trunc =
            builder.createOrFold<AsSIntPrimOp>(con.getLoc(), lhsType, trunc);

      LLVM_DEBUG(llvm::dbgs()
                 << "Truncating RHS to " << lhsType << " in " << con << "\n");
      con->replaceUsesOfWith(con.getSrc(), trunc);
    }
    return anyChanged;
  }

  // If this is a module, update its ports.
  if (auto module = dyn_cast<FModuleOp>(op)) {
    // Update the block argument types.
    bool argsChanged = false;
    SmallVector<Attribute> argTypes;
    argTypes.reserve(module.getNumPorts());
    for (auto arg : module.getArguments()) {
      auto result = updateValue(arg);
      if (failed(result))
        return result;
      argsChanged |= *result;
      argTypes.push_back(TypeAttr::get(arg.getType()));
    }

    // Update the module function type if needed.
    if (argsChanged) {
      module.setPortTypesAttr(ArrayAttr::get(module.getContext(), argTypes));
      anyChanged = true;
    }
  }
  return anyChanged;
}

/// Resize a `uint`, `sint`, or `analog` type to a specific width.
static FIRRTLBaseType resizeType(FIRRTLBaseType type, uint32_t newWidth) {
  auto *context = type.getContext();
  return FIRRTLTypeSwitch<FIRRTLBaseType, FIRRTLBaseType>(type)
      .Case<UIntType>([&](auto type) {
        return UIntType::get(context, newWidth, type.isConst());
      })
      .Case<SIntType>([&](auto type) {
        return SIntType::get(context, newWidth, type.isConst());
      })
      .Case<AnalogType>([&](auto type) {
        return AnalogType::get(context, newWidth, type.isConst());
      })
      .Default([&](auto type) { return type; });
}

/// Update the type of a value.
FailureOr<bool> InferenceTypeUpdate::updateValue(Value value) {
  // Check if the value has a type which we can update.
  auto type = type_dyn_cast<FIRRTLType>(value.getType());
  if (!type)
    return false;

  // Fast path for types that have fully inferred widths.
  if (!hasUninferredWidth(type))
    return false;

  // If this is an operation that does not generate any free variables that
  // are determined during width inference, simply update the value type based
  // on the operation arguments.
  if (auto op = dyn_cast_or_null<InferTypeOpInterface>(value.getDefiningOp())) {
    SmallVector<Type, 2> types;
    auto res =
        op.inferReturnTypes(op->getContext(), op->getLoc(), op->getOperands(),
                            op->getAttrDictionary(), op->getPropertiesStorage(),
                            op->getRegions(), types);
    if (failed(res))
      return failure();

    assert(types.size() == op->getNumResults());
    for (auto [result, type] : llvm::zip(op->getResults(), types)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Inferring " << result << " as " << type << "\n");
      result.setType(type);
    }
    return true;
  }

  // Recreate the type, substituting the solved widths.
  auto *context = type.getContext();
  unsigned fieldID = 0;
  std::function<FIRRTLBaseType(FIRRTLBaseType)> updateBase =
      [&](FIRRTLBaseType type) -> FIRRTLBaseType {
    auto width = type.getBitWidthOrSentinel();
    if (width >= 0) {
      // Known width integers return themselves.
      fieldID++;
      return type;
    }
    if (width == -1) {
      // Unknown width integers return the solved type.
      auto newType = updateType(FieldRef(value, fieldID), type);
      fieldID++;
      return newType;
    }
    if (auto bundleType = type_dyn_cast<BundleType>(type)) {
      // Bundle types recursively update all bundle elements.
      fieldID++;
      llvm::SmallVector<BundleType::BundleElement, 3> elements;
      for (auto &element : bundleType) {
        auto updatedBase = updateBase(element.type);
        if (!updatedBase)
          return {};
        elements.emplace_back(element.name, element.isFlip, updatedBase);
      }
      return BundleType::get(context, elements, bundleType.isConst());
    }
    if (auto vecType = type_dyn_cast<FVectorType>(type)) {
      fieldID++;
      auto save = fieldID;
      // TODO: this should recurse into the element type of 0 length vectors and
      // set any unknown width to 0.
      if (vecType.getNumElements() > 0) {
        auto updatedBase = updateBase(vecType.getElementType());
        if (!updatedBase)
          return {};
        auto newType = FVectorType::get(updatedBase, vecType.getNumElements(),
                                        vecType.isConst());
        fieldID = save + vecType.getMaxFieldID();
        return newType;
      }
      // If this is a 0 length vector return the original type.
      return type;
    }
    if (auto enumType = type_dyn_cast<FEnumType>(type)) {
      fieldID++;
      llvm::SmallVector<FEnumType::EnumElement> elements;
      for (auto &element : enumType.getElements()) {
        auto updatedBase = updateBase(element.type);
        if (!updatedBase)
          return {};
        elements.emplace_back(element.name, updatedBase);
      }
      return FEnumType::get(context, elements, enumType.isConst());
    }
    llvm_unreachable("Unknown type inside a bundle!");
  };

  // Update the type.
  auto newType = mapBaseTypeNullable(type, updateBase);
  if (!newType)
    return failure();
  LLVM_DEBUG(llvm::dbgs() << "Update " << value << " to " << newType << "\n");
  value.setType(newType);

  // If this is a ConstantOp, adjust the width of the underlying APInt.
  // Unsized constants have APInts which are *at least* wide enough to hold
  // the value, but may be larger. This can trip up the verifier.
  if (auto op = value.getDefiningOp<ConstantOp>()) {
    auto k = op.getValue();
    auto bitwidth = op.getType().getBitWidthOrSentinel();
    if (k.getBitWidth() > unsigned(bitwidth))
      k = k.trunc(bitwidth);
    op->setAttr("value", IntegerAttr::get(op.getContext(), k));
  }

  return newType != type;
}

/// Update a type.
FIRRTLBaseType InferenceTypeUpdate::updateType(FieldRef fieldRef,
                                               FIRRTLBaseType type) {
  assert(type.isGround() && "Can only pass in ground types.");
  auto value = fieldRef.getValue();
  // Get the inferred width.
  Expr *expr = mapping.getExprOrNull(fieldRef);
  if (!expr || !expr->getSolution()) {
    // It should not be possible to arrive at an uninferred width at this point.
    // In case the constraints are not resolvable, checks before the calls to
    // `updateType` must have already caught the issues and aborted the pass
    // early. Might turn this into an assert later.
    mlir::emitError(value.getLoc(), "width should have been inferred");
    return {};
  }
  int32_t solution = *expr->getSolution();
  assert(solution >= 0); // The solver infers variables to be 0 or greater.
  return resizeType(type, solution);
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
class InferWidthsPass
    : public circt::firrtl::impl::InferWidthsBase<InferWidthsPass> {
  void runOnOperation() override;
};
} // namespace

void InferWidthsPass::runOnOperation() {
  // Collect variables and constraints
  ConstraintSolver solver;
  InferenceMapping mapping(solver, getAnalysis<SymbolTable>(),
                           getAnalysis<hw::InnerSymbolTableCollection>());
  if (failed(mapping.map(getOperation())))
    return signalPassFailure();

  // fast path if no inferrable widths are around
  if (mapping.areAllModulesSkipped())
    return markAllAnalysesPreserved();

  // Solve the constraints.
  if (failed(solver.solve()))
    return signalPassFailure();

  // Update the types with the inferred widths.
  if (failed(InferenceTypeUpdate(mapping).update(getOperation())))
    return signalPassFailure();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createInferWidthsPass() {
  return std::make_unique<InferWidthsPass>();
}
