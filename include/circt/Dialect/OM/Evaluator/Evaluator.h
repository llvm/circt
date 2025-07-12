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
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"

#include <queue>
#include <utility>

namespace circt {
namespace om {

namespace evaluator {
struct EvaluatorValue;

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
struct EvaluatorValue : std::enable_shared_from_this<EvaluatorValue> {
  // Implement LLVM RTTI.
  enum class Kind { Attr, Object, List, Reference, BasePath, Path };
  EvaluatorValue(MLIRContext *ctx, Kind kind, Location loc)
      : kind(kind), ctx(ctx), loc(loc) {}
  Kind getKind() const { return kind; }
  MLIRContext *getContext() const { return ctx; }

  // Return true the value is fully evaluated.
  bool isFullyEvaluated() const { return fullyEvaluated; }
  void markFullyEvaluated() {
    assert(!fullyEvaluated && "should not mark twice");
    fullyEvaluated = true;
  }

  /// Return the associated MLIR context.
  MLIRContext *getContext() { return ctx; }

  // Return a MLIR type which the value represents.
  Type getType() const;

  // Finalize the evaluator value. Strip intermidiate reference values.
  LogicalResult finalize();

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
  bool finalized = false;
};

/// Values which can be used as pointers to different values.
/// ReferenceValue is replaced with its element and erased at the end of
/// evaluation.
struct ReferenceValue : EvaluatorValue {
  ReferenceValue(Type type, Location loc)
      : EvaluatorValue(type.getContext(), Kind::Reference, loc), value(nullptr),
        type(type) {}

  // Implement LLVM RTTI.
  static bool classof(const EvaluatorValue *e) {
    return e->getKind() == Kind::Reference;
  }

  Type getValueType() const { return type; }
  EvaluatorValuePtr getValue() const { return value; }
  void setValue(EvaluatorValuePtr newValue) {
    value = std::move(newValue);
    markFullyEvaluated();
  }

  // Finalize the value.
  LogicalResult finalizeImpl();

  // Return the first non-reference value that is reachable from the reference.
  FailureOr<EvaluatorValuePtr> getStrippedValue() const {
    llvm::SmallPtrSet<ReferenceValue *, 4> visited;
    auto currentValue = value;
    while (auto *v = dyn_cast<ReferenceValue>(currentValue.get())) {
      // Detect a cycle.
      if (!visited.insert(v).second)
        return failure();
      currentValue = v->getValue();
    }
    return success(currentValue);
  }

private:
  EvaluatorValuePtr value;
  Type type;
};

/// Values which can be directly representable by MLIR attributes.
struct AttributeValue : EvaluatorValue {
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

  // Finalize the value.
  LogicalResult finalizeImpl();

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

// This perform finalization to `value`.
static inline LogicalResult finalizeEvaluatorValue(EvaluatorValuePtr &value) {
  if (failed(value->finalize()))
    return failure();
  if (auto *ref = llvm::dyn_cast<ReferenceValue>(value.get())) {
    auto v = ref->getStrippedValue();
    if (failed(v))
      return v;
    value = v.value();
  }
  return success();
}

/// A List which contains variadic length of elements with the same type.
struct ListValue : EvaluatorValue {
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

  // Finalize the value.
  LogicalResult finalizeImpl();

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
struct ObjectValue : EvaluatorValue {
  ObjectValue(om::ClassOp cls, ObjectFields fields, Location loc)
      : EvaluatorValue(cls.getContext(), Kind::Object, loc), cls(cls),
        fields(std::move(fields)) {
    markFullyEvaluated();
  }

  // Partially evaluated value.
  ObjectValue(om::ClassOp cls, Location loc)
      : EvaluatorValue(cls.getContext(), Kind::Object, loc), cls(cls) {}

  om::ClassOp getClassOp() const { return cls; }
  const auto &getFields() const { return fields; }

  void setFields(llvm::SmallDenseMap<StringAttr, EvaluatorValuePtr> newFields) {
    fields = std::move(newFields);
    markFullyEvaluated();
  }

  /// Return the type of the value, which is a ClassType.
  om::ClassType getObjectType() const {
    auto clsConst = const_cast<ClassOp &>(cls);
    return ClassType::get(clsConst.getContext(),
                          FlatSymbolRefAttr::get(clsConst.getNameAttr()));
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

  // Finalize the evaluator value.
  LogicalResult finalizeImpl();

private:
  om::ClassOp cls;
  llvm::SmallDenseMap<StringAttr, EvaluatorValuePtr> fields;
};

/// A Basepath value.
struct BasePathValue : EvaluatorValue {
  BasePathValue(MLIRContext *context);

  /// Create a path value representing a basepath.
  BasePathValue(om::PathAttr path, Location loc);

  om::PathAttr getPath() const;

  /// Set the basepath which this path is relative to.
  void setBasepath(const BasePathValue &basepath);

  /// Finalize the evaluator value.
  LogicalResult finalizeImpl() { return success(); }

  /// Implement LLVM RTTI.
  static bool classof(const EvaluatorValue *e) {
    return e->getKind() == Kind::BasePath;
  }

private:
  om::PathAttr path;
};

/// A Path value.
struct PathValue : EvaluatorValue {
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

  // Finalize the evaluator value.
  LogicalResult finalizeImpl() { return success(); }

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
struct Evaluator {
  /// Construct an Evaluator with an IR module.
  Evaluator(ModuleOp mod);

  /// Instantiate an Object with its class name and actual parameters.
  FailureOr<evaluator::EvaluatorValuePtr>
  instantiate(StringAttr className, ArrayRef<EvaluatorValuePtr> actualParams);

  /// Get the Module this Evaluator is built from.
  mlir::ModuleOp getModule();

  FailureOr<evaluator::EvaluatorValuePtr>
  getPartiallyEvaluatedValue(Type type, Location loc);

  using ActualParameters =
      SmallVectorImpl<std::shared_ptr<evaluator::EvaluatorValue>> *;

  using ObjectKey = std::pair<Value, ActualParameters>;

private:
  bool isFullyEvaluated(Value value, ActualParameters key) {
    return isFullyEvaluated({value, key});
  }

  bool isFullyEvaluated(ObjectKey key) {
    auto val = objects.lookup(key);
    return val && val->isFullyEvaluated();
  }

  FailureOr<EvaluatorValuePtr>
  getOrCreateValue(Value value, ActualParameters actualParams, Location loc);
  FailureOr<EvaluatorValuePtr>
  allocateObjectInstance(StringAttr clasName, ActualParameters actualParams);

  /// Evaluate a Value in a Class body according to the small expression grammar
  /// described in the rationale document. The actual parameters are the values
  /// supplied at the current instantiation of the Class being evaluated.
  FailureOr<EvaluatorValuePtr>
  evaluateValue(Value value, ActualParameters actualParams, Location loc);

  /// Evaluator dispatch functions for the small expression grammar.
  FailureOr<EvaluatorValuePtr> evaluateParameter(BlockArgument formalParam,
                                                 ActualParameters actualParams,
                                                 Location loc);

  FailureOr<EvaluatorValuePtr>
  evaluateConstant(ConstantOp op, ActualParameters actualParams, Location loc);

  FailureOr<EvaluatorValuePtr>
  evaluateIntegerBinaryArithmetic(IntegerBinaryArithmeticOp op,
                                  ActualParameters actualParams, Location loc);

  /// Instantiate an Object with its class name and actual parameters.
  FailureOr<EvaluatorValuePtr>
  evaluateObjectInstance(StringAttr className, ActualParameters actualParams,
                         Location loc, ObjectKey instanceKey = {});
  FailureOr<EvaluatorValuePtr>
  evaluateObjectInstance(ObjectOp op, ActualParameters actualParams);
  FailureOr<EvaluatorValuePtr>
  evaluateObjectField(ObjectFieldOp op, ActualParameters actualParams,
                      Location loc);
  FailureOr<EvaluatorValuePtr> evaluateListCreate(ListCreateOp op,
                                                  ActualParameters actualParams,
                                                  Location loc);
  FailureOr<EvaluatorValuePtr> evaluateListConcat(ListConcatOp op,
                                                  ActualParameters actualParams,
                                                  Location loc);
  FailureOr<evaluator::EvaluatorValuePtr>
  evaluateBasePathCreate(FrozenBasePathCreateOp op,
                         ActualParameters actualParams, Location loc);
  FailureOr<evaluator::EvaluatorValuePtr>
  evaluatePathCreate(FrozenPathCreateOp op, ActualParameters actualParams,
                     Location loc);
  FailureOr<evaluator::EvaluatorValuePtr>
  evaluateEmptyPath(FrozenEmptyPathOp op, ActualParameters actualParams,
                    Location loc);

  FailureOr<ActualParameters>
  createParametersFromOperands(ValueRange range, ActualParameters actualParams,
                               Location loc);

  /// The symbol table for the IR module the Evaluator was constructed with.
  /// Used to look up class definitions.
  SymbolTable symbolTable;

  /// This uniquely stores vectors that represent parameters.
  SmallVector<
      std::unique_ptr<SmallVector<std::shared_ptr<evaluator::EvaluatorValue>>>>
      actualParametersBuffers;

  /// A worklist that tracks values which needs to be fully evaluated.
  std::queue<ObjectKey> worklist;

  /// Evaluator value storage. Return an evaluator value for the given
  /// instantiation context (a pair of Value and parameters).
  DenseMap<ObjectKey, std::shared_ptr<evaluator::EvaluatorValue>> objects;
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
  return diag;
}

/// Helper to enable printing objects in Diagnostics.
static inline mlir::Diagnostic &
operator<<(mlir::Diagnostic &diag, const EvaluatorValuePtr &evaluatorValue) {
  return diag << *evaluatorValue.get();
}

} // namespace om
} // namespace circt

#endif // CIRCT_DIALECT_OM_EVALUATOR_EVALUATOR_H
