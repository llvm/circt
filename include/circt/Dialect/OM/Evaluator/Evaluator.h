//===- Evaluator.h - Object Model dialect evaluator -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Object Model dialect declaration.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_OM_EVALUATOR_EVALUATOR_H
#define CIRCT_DIALECT_OM_EVALUATOR_EVALUATOR_H

#include "circt/Dialect/OM/OMOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Debug.h"

#include <utility>

namespace circt {
namespace om {

namespace evaluator {
class EvaluatorValue;

/// A value of an object in memory. It is either a composite Object, or a
/// primitive Attribute. Further refinement is expected.
using EvaluatorValuePtr = std::shared_ptr<EvaluatorValue>;

/// The fields of a composite Object, currently represented as a map. Further
/// refinement is expected.
using ObjectFields = SmallDenseMap<StringAttr, EvaluatorValuePtr>;

/// Base class for evaluator runtime values.
/// Enables the shared_from_this functionality so Evaluator Value pointers can
/// be passed through the CAPI and unwrapped back into C++ smart pointers with
/// the appropriate reference count.
class EvaluatorValue : public std::enable_shared_from_this<EvaluatorValue> {
public:
  // Implement LLVM RTTI.
  enum class Kind { Attr, Object, List, BasePath, Path };
  EvaluatorValue(MLIRContext *ctx, Kind kind, Location loc)
      : kind(kind), ctx(ctx), loc(loc) {}
  Kind getKind() const { return kind; }
  MLIRContext *getContext() const { return ctx; }

  // Return true the value is fully evaluated.
  // Unknown values are considered fully evaluated.
  bool isFullyEvaluated() const { return fullyEvaluated; }
  void markFullyEvaluated() {
    assert(!fullyEvaluated && "should not mark twice");
    fullyEvaluated = true;
  }

  /// Return true if the value is unknown (has unknown in its fan-in).
  /// A value is unknown if it depends on an UnknownValueOp or if any of its
  /// inputs are unknown. Unknown values propagate through operations.
  bool isUnknown() const { return unknown; }

  /// Mark this value as unknown.
  /// This also marks the value as fully evaluated if it isn't already, since
  /// unknown values are considered fully evaluated. This maintains the
  /// invariant that unknown implies fullyEvaluated.
  void markUnknown() {
    unknown = true;
    if (!fullyEvaluated)
      markFullyEvaluated();
  }

  /// Return the associated MLIR context.
  MLIRContext *getContext() { return ctx; }

  // Return a MLIR type which the value represents.
  Type getType() const;

  // Return the Location associated with the Value.
  Location getLoc() const { return loc; }
  // Set the Location associated with the Value.
  void setLoc(Location l) { loc = l; }
  // Set the Location, if it is unknown.
  void setLocIfUnknown(Location l) {
    if (isa<UnknownLoc>(loc))
      loc = l;
  }

private:
  const Kind kind;
  MLIRContext *ctx;
  Location loc;
  bool fullyEvaluated = false;
  bool unknown = false;
};

/// Values which can be directly representable by MLIR attributes.
class AttributeValue : public EvaluatorValue {
public:
  Attribute getAttr() const { return attr; }
  template <typename AttrTy>
  AttrTy getAs() const {
    return dyn_cast<AttrTy>(attr);
  }
  static bool classof(const EvaluatorValue *e) {
    return e->getKind() == Kind::Attr;
  }

  // Set Attribute for partially evaluated case.
  LogicalResult setAttr(Attribute attr);

  Type getType() const { return type; }

  // Factory methods that create AttributeValue objects
  static std::shared_ptr<EvaluatorValue> get(Attribute attr,
                                             LocationAttr loc = {});
  static std::shared_ptr<EvaluatorValue> get(Type type, LocationAttr loc = {});

private:
  // Make AttributeValue constructible only by the factory methods
  struct PrivateTag {};

  // Constructor that requires a PrivateTag
  AttributeValue(PrivateTag, Attribute attr, Location loc)
      : EvaluatorValue(attr.getContext(), Kind::Attr, loc), attr(attr),
        type(cast<TypedAttr>(attr).getType()) {
    markFullyEvaluated();
  }

  // Constructor for partially evaluated AttributeValue
  AttributeValue(PrivateTag, Type type, Location loc)
      : EvaluatorValue(type.getContext(), Kind::Attr, loc), type(type) {}

  Attribute attr = {};
  Type type;

  // Friend declaration for the factory methods
  friend std::shared_ptr<EvaluatorValue> get(Attribute attr, LocationAttr loc);
  friend std::shared_ptr<EvaluatorValue> get(Type type, LocationAttr loc);
};

/// A List which contains variadic length of elements with the same type.
class ListValue : public EvaluatorValue {
public:
  ListValue(om::ListType type, SmallVector<EvaluatorValuePtr> elements,
            Location loc)
      : EvaluatorValue(type.getContext(), Kind::List, loc), type(type),
        elements(std::move(elements)) {
    markFullyEvaluated();
  }

  void setElements(SmallVector<EvaluatorValuePtr> newElements) {
    elements = std::move(newElements);
    markFullyEvaluated();
  }

  // Partially evaluated value.
  ListValue(om::ListType type, Location loc)
      : EvaluatorValue(type.getContext(), Kind::List, loc), type(type) {}

  const auto &getElements() const { return elements; }

  /// Return the type of the value, which is a ListType.
  om::ListType getListType() const { return type; }

  /// Implement LLVM RTTI.
  static bool classof(const EvaluatorValue *e) {
    return e->getKind() == Kind::List;
  }

private:
  om::ListType type;
  SmallVector<EvaluatorValuePtr> elements;
};

/// A composite Object, which has a type and fields.
class ObjectValue : public EvaluatorValue {
public:
  ObjectValue(om::ClassLike cls, ObjectFields fields, Location loc)
      : EvaluatorValue(cls.getContext(), Kind::Object, loc), cls(cls),
        fields(std::move(fields)) {
    markFullyEvaluated();
  }

  // Partially evaluated value.
  ObjectValue(om::ClassLike cls, Location loc)
      : EvaluatorValue(cls.getContext(), Kind::Object, loc), cls(cls) {}

  om::ClassLike getClassOp() const { return cls; }
  const auto &getFields() const { return fields; }

  void setFields(llvm::SmallDenseMap<StringAttr, EvaluatorValuePtr> newFields) {
    fields = std::move(newFields);
    markFullyEvaluated();
  }

  /// Return the type of the value, which is a ClassType.
  om::ClassType getObjectType() const {
    auto clsNonConst = const_cast<om::ClassLike &>(cls);
    return ClassType::get(clsNonConst.getContext(),
                          FlatSymbolRefAttr::get(clsNonConst.getSymNameAttr()));
  }

  Type getType() const { return getObjectType(); }

  /// Implement LLVM RTTI.
  static bool classof(const EvaluatorValue *e) {
    return e->getKind() == Kind::Object;
  }

  /// Get a field of the Object by name.
  FailureOr<EvaluatorValuePtr> getField(StringAttr field);
  FailureOr<EvaluatorValuePtr> getField(StringRef field) {
    return getField(StringAttr::get(getContext(), field));
  }

  /// Get all the field names of the Object.
  ArrayAttr getFieldNames();

private:
  om::ClassLike cls;
  llvm::SmallDenseMap<StringAttr, EvaluatorValuePtr> fields;
};

/// A Basepath value.
class BasePathValue : public EvaluatorValue {
public:
  BasePathValue(MLIRContext *context);

  /// Create a path value representing a basepath.
  BasePathValue(om::PathAttr path, Location loc);

  om::PathAttr getPath() const;

  /// Set the basepath which this path is relative to.
  void setBasepath(const BasePathValue &basepath);

  /// Implement LLVM RTTI.
  static bool classof(const EvaluatorValue *e) {
    return e->getKind() == Kind::BasePath;
  }

private:
  om::PathAttr path;
};

/// A Path value.
class PathValue : public EvaluatorValue {
public:
  /// Create a path value representing a regular path.
  PathValue(om::TargetKindAttr targetKind, om::PathAttr path, StringAttr module,
            StringAttr ref, StringAttr field, Location loc);

  static PathValue getEmptyPath(Location loc);

  om::TargetKindAttr getTargetKind() const { return targetKind; }

  om::PathAttr getPath() const { return path; }

  StringAttr getModule() const { return module; }

  StringAttr getRef() const { return ref; }

  StringAttr getField() const { return field; }

  StringAttr getAsString() const;

  void setBasepath(const BasePathValue &basepath);

  /// Implement LLVM RTTI.
  static bool classof(const EvaluatorValue *e) {
    return e->getKind() == Kind::Path;
  }

private:
  om::TargetKindAttr targetKind;
  om::PathAttr path;
  StringAttr module;
  StringAttr ref;
  StringAttr field;
};

} // namespace evaluator

using Object = evaluator::ObjectValue;
using EvaluatorValuePtr = evaluator::EvaluatorValuePtr;

SmallVector<EvaluatorValuePtr>
getEvaluatorValuesFromAttributes(MLIRContext *context,
                                 ArrayRef<Attribute> attributes);

/// An Evaluator, which is constructed with an IR module and can instantiate
/// Objects. Further refinement is expected.
class Evaluator {
public:
  /// Construct an Evaluator with an IR module.
  Evaluator(ModuleOp mod);

  /// Instantiate an Object with its class name and actual parameters.
  FailureOr<evaluator::EvaluatorValuePtr>
  instantiate(StringAttr className, ArrayRef<EvaluatorValuePtr> actualParams);

  /// Get the Module this Evaluator is built from.
  mlir::ModuleOp getModule();

  FailureOr<evaluator::EvaluatorValuePtr>
  getPartiallyEvaluatedValue(Type type, Location loc);

private:
  FailureOr<evaluator::EvaluatorValuePtr>
  instantiateImpl(StringAttr className,
                  ArrayRef<EvaluatorValuePtr> actualParams);

  FailureOr<EvaluatorValuePtr> getOrCreateValue(Value value, Location loc);

  /// Evaluate a Value in a Class body according to the small expression grammar
  /// described in the rationale document.
  FailureOr<EvaluatorValuePtr> evaluateValue(Value value, Location loc);

  /// Evaluate a class body with actual parameters.
  FailureOr<EvaluatorValuePtr>
  evaluateObjectInstance(StringAttr className,
                         ArrayRef<EvaluatorValuePtr> actualParams,
                         Location loc);

  /// Evaluator dispatch functions for the small expression grammar.
  FailureOr<EvaluatorValuePtr> evaluateOp(ConstantOp op, Location loc);
  FailureOr<EvaluatorValuePtr> evaluateOp(ElaboratedObjectOp op, Location loc);
  FailureOr<EvaluatorValuePtr> evaluateOp(ListCreateOp op, Location loc);
  FailureOr<EvaluatorValuePtr> evaluateOp(ListConcatOp op, Location loc);
  FailureOr<EvaluatorValuePtr> evaluateOp(FrozenBasePathCreateOp op,
                                          Location loc);
  FailureOr<EvaluatorValuePtr> evaluateOp(FrozenPathCreateOp op, Location loc);
  FailureOr<EvaluatorValuePtr> evaluateOp(FrozenEmptyPathOp op, Location loc);
  FailureOr<EvaluatorValuePtr> evaluateOp(UnknownValueOp op, Location loc);

  FailureOr<evaluator::EvaluatorValuePtr> createUnknownValue(Type type,
                                                             Location loc);

  /// The symbol table for the IR module the Evaluator was constructed with.
  /// Used to look up class definitions.
  SymbolTable symbolTable;

  /// Evaluator value storage for the current instantiation.
  DenseMap<Value, std::shared_ptr<evaluator::EvaluatorValue>> objects;

#ifndef NDEBUG
  /// Current nesting depth for debug output indentation.
  unsigned debugNesting = 0;

  /// RAII helper to increment/decrement debugNesting.
  struct DebugNesting {
    unsigned &depth;
    DebugNesting(unsigned &depth) : depth(depth) { ++depth; }
    ~DebugNesting() { --depth; }
  };

  raw_ostream &dbgs(unsigned extra = 0) {
    return llvm::dbgs().indent(debugNesting * 2 + extra * 2);
  }

  llvm::indent indent(unsigned extra = 0) {
    return llvm::indent(debugNesting, 2) + extra;
  }
#endif
};

/// Helper to enable printing objects in Diagnostics.
static inline mlir::Diagnostic &
operator<<(mlir::Diagnostic &diag,
           const evaluator::EvaluatorValue &evaluatorValue) {
  if (auto *attr = llvm::dyn_cast<evaluator::AttributeValue>(&evaluatorValue))
    diag << attr->getAttr();
  else if (auto *object =
               llvm::dyn_cast<evaluator::ObjectValue>(&evaluatorValue))
    diag << "Object(" << object->getType() << ")";
  else if (auto *list = llvm::dyn_cast<evaluator::ListValue>(&evaluatorValue))
    diag << "List(" << list->getType() << ")";
  else if (llvm::isa<evaluator::BasePathValue>(&evaluatorValue))
    diag << "BasePath()";
  else if (llvm::isa<evaluator::PathValue>(&evaluatorValue))
    diag << "Path()";
  else
    assert(false && "unhandled evaluator value");

  // Add unknown marker if the value is unknown
  if (evaluatorValue.isUnknown())
    diag << " [unknown]";
  return diag;
}

/// Helper to enable printing objects in Diagnostics.
static inline mlir::Diagnostic &
operator<<(mlir::Diagnostic &diag, const EvaluatorValuePtr &evaluatorValue) {
  return diag << *evaluatorValue.get();
}

#ifndef NDEBUG
/// Helper to enable printing objects to raw_ostream (e.g., llvm::dbgs()).
/// Delegates to the Diagnostic overload via an intermediate string.
static inline llvm::raw_ostream &
operator<<(llvm::raw_ostream &os,
           const evaluator::EvaluatorValue &evaluatorValue) {
  std::string buf;
  llvm::raw_string_ostream ss(buf);
  mlir::Diagnostic diag(UnknownLoc::get(evaluatorValue.getContext()),
                        mlir::DiagnosticSeverity::Note);
  diag << evaluatorValue;
  ss << diag;
  return os << ss.str();
}

static inline llvm::raw_ostream &
operator<<(llvm::raw_ostream &os, const EvaluatorValuePtr &evaluatorValue) {
  if (evaluatorValue)
    return os << *evaluatorValue.get();
  return os << "<null>";
}
#endif // NDEBUG

} // namespace om
} // namespace circt

#endif // CIRCT_DIALECT_OM_EVALUATOR_EVALUATOR_H
