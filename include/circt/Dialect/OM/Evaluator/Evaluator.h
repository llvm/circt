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
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LogicalResult.h"

#include <deque>

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
/// Enables the shared_from_this functionality so Object pointers can be passed
/// through the CAPI and unwrapped back into C++ smart pointers with the
/// appropriate reference count.
struct EvaluatorValue : std::enable_shared_from_this<EvaluatorValue> {
  // Implement LLVM RTTI.
  enum class Kind { Attr, Object, List, Tuple, Map, Reference };
  EvaluatorValue(MLIRContext *ctx, Kind kind) : kind(kind) {}
  Kind getKind() const { return kind; }
  MLIRContext *getContext() const { return ctx; }
  bool isFullyEvaluated() const { return fullyEvaluated; }
  void markFullyEvaluated() { fullyEvaluated = true; }
  Type getType() const;

private:
  const Kind kind;
  MLIRContext *ctx;
  bool fullyEvaluated = false;
};

struct ReferenceValue : EvaluatorValue {
  // Implement LLVM RTTI.
  ReferenceValue(MLIRContext *ctx, EvaluatorValuePtr value)
      : EvaluatorValue(ctx, Kind::Reference), value(value) {}

  ReferenceValue(MLIRContext *ctx)
      : EvaluatorValue(ctx, Kind::Reference), value(nullptr) {}

  static bool classof(const EvaluatorValue *e) {
    return e->getKind() == Kind::Reference;
  }

  void setValue(EvaluatorValuePtr newValue) {
    value = newValue;
    // if (newValue->isFullyEvaluated())
    markFullyEvaluated();
  }

  Type getValueType() const { return value->getType(); }

  EvaluatorValuePtr getValue() const { return value; }
  EvaluatorValuePtr getStripValue() const {
    if (auto *v = dyn_cast<ReferenceValue>(value.get()))
      return v->getStripValue();
    return value;
  }

private:
  EvaluatorValuePtr value;
};

/// Values which can be directly representable by MLIR attributes.
struct AttributeValue : EvaluatorValue {
  AttributeValue(Attribute attr)
      : EvaluatorValue(attr.getContext(), Kind::Attr), attr(attr) {
    markFullyEvaluated();
  }

  // Partially evaluated value.
  AttributeValue(MLIRContext *ctx) : EvaluatorValue(ctx, Kind::Attr) {}
  Attribute getAttr() const { return attr; }
  template <typename AttrTy>
  AttrTy getAs() const {
    return dyn_cast<AttrTy>(attr);
  }
  static bool classof(const EvaluatorValue *e) {
    return e->getKind() == Kind::Attr;
  }
  void setAttr(Attribute newAttr) {
    attr = newAttr;
    markFullyEvaluated();
  }

  Type getType() const { return attr.cast<TypedAttr>().getType(); }

private:
  Attribute attr = {};
};

/// A List which contains variadic length of elements with the same type.
struct ListValue : EvaluatorValue {
  ListValue(om::ListType type, SmallVector<EvaluatorValuePtr> elements)
      : EvaluatorValue(type.getContext(), Kind::List), type(type),
        elements(std::move(elements)) {
    update();
  }

  void setElements(SmallVector<EvaluatorValuePtr> newElements) {
    elements = std::move(newElements);
    update();
  }

  void update() {
    if (llvm::all_of(elements,
                     [](const auto &ptr) { return ptr->isFullyEvaluated(); }))
      markFullyEvaluated();
  }

  // Partially evaluated value.
  ListValue(om::ListType type)
      : EvaluatorValue(type.getContext(), Kind::List), type(type) {}

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

/// A Map value.
struct MapValue : EvaluatorValue {
  MapValue(om::MapType type, DenseMap<Attribute, EvaluatorValuePtr> elements)
      : EvaluatorValue(type.getContext(), Kind::Map), type(type),
        elements(std::move(elements)) {
    update();
  }

  void update() {
    for (auto [key, value] : elements)
      if (!value->isFullyEvaluated())
        return;
    markFullyEvaluated();
  }

  void setElements(DenseMap<Attribute, EvaluatorValuePtr> newElements) {
    elements = std::move(newElements);
    update();
  }

  // Partially evaluated value.
  MapValue(om::MapType type)
      : EvaluatorValue(type.getContext(), Kind::Map), type(type) {}
  const auto &getElements() const { return elements; }

  /// Return the type of the value, which is a MapType.
  om::MapType getMapType() const { return type; }

  /// Return an array of keys in the ascending order.
  ArrayAttr getKeys();

  /// Implement LLVM RTTI.
  static bool classof(const EvaluatorValue *e) {
    return e->getKind() == Kind::Map;
  }

private:
  om::MapType type;
  DenseMap<Attribute, EvaluatorValuePtr> elements;
};

/// A composite Object, which has a type and fields.
struct ObjectValue : EvaluatorValue {
  ObjectValue(om::ClassOp cls, ObjectFields fields)
      : EvaluatorValue(cls.getContext(), Kind::Object), cls(cls),
        fields(std::move(fields)) {
    update();
  }

  // Partially evaluated value.
  ObjectValue(om::ClassOp cls)
      : EvaluatorValue(cls.getContext(), Kind::Object), cls(cls) {}

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

  void update() {
    for (auto [key, value] : fields)
      if (!value->isFullyEvaluated())
        return;
    markFullyEvaluated();
  }

  /// Implement LLVM RTTI.
  static bool classof(const EvaluatorValue *e) {
    return e->getKind() == Kind::Object;
  }

  /// Get a field of the Object by name.
  FailureOr<EvaluatorValuePtr> getField(StringAttr field);

  /// Get all the field names of the Object.
  ArrayAttr getFieldNames();

private:
  om::ClassOp cls;
  llvm::SmallDenseMap<StringAttr, EvaluatorValuePtr> fields;
};

/// Tuple values.
struct TupleValue : EvaluatorValue {
  using TupleElements = llvm::SmallVector<EvaluatorValuePtr>;
  TupleValue(TupleType type, TupleElements tupleElements)
      : EvaluatorValue(type.getContext(), Kind::Tuple), type(type),
        elements(std::move(tupleElements)) {
    update();
  }

  // Partially evaluated value.
  TupleValue(TupleType type)
      : EvaluatorValue(type.getContext(), Kind::Tuple), type(type) {}

  void update() {
    for (auto v : elements) {
      if (!v->isFullyEvaluated())
        return;
    }
    markFullyEvaluated();
  }
  void setElements(TupleElements newElements) {
    elements = std::move(newElements);
    update();
  }

  /// Implement LLVM RTTI.
  static bool classof(const EvaluatorValue *e) {
    return e->getKind() == Kind::Tuple;
  }

  /// Return the type of the value, which is a TupleType.
  TupleType getTupleType() const { return type; }

  const TupleElements &getElements() const { return elements; }

private:
  TupleType type;
  TupleElements elements;
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

  FailureOr<evaluator::EvaluatorValuePtr> getPartiallyEvaluatedValue(Type type);

  using ActualParameters =
      SmallVectorImpl<std::shared_ptr<evaluator::EvaluatorValue>> *;

  using Key = std::pair<Value, ActualParameters>;

private:
  bool isFullyEvaluated(Key key) {
    auto val = objects.lookup(key);
    return val && val->isFullyEvaluated();
  }

  EvaluatorValuePtr lookupEvaluatorValue(Key key) {
    return objects.lookup(key);
  }
  FailureOr<EvaluatorValuePtr> allocateValue(Value value,
                                             ActualParameters actualParams);
  FailureOr<EvaluatorValuePtr>
  allocateObjectInstance(StringAttr clasName, ActualParameters actualParams);

  /// Evaluate a Value in a Class body according to the small expression grammar
  /// described in the rationale document. The actual parameters are the values
  /// supplied at the current instantiation of the Class being evaluated.
  FailureOr<EvaluatorValuePtr> evaluateValue(Value value,
                                             ActualParameters actualParams);

  /// Evaluator dispatch functions for the small expression grammar.
  FailureOr<EvaluatorValuePtr> evaluateParameter(BlockArgument formalParam,
                                                 ActualParameters actualParams);

  FailureOr<EvaluatorValuePtr> evaluateConstant(ConstantOp op,
                                                ActualParameters actualParams);
  /// Instantiate an Object with its class name and actual parameters.
  FailureOr<EvaluatorValuePtr>
  evaluateObjectInstance(StringAttr className, ActualParameters actualParams,
                         Key caller = {});
  FailureOr<EvaluatorValuePtr>
  evaluateObjectInstance(ObjectOp op, ActualParameters actualParams);
  FailureOr<EvaluatorValuePtr>
  evaluateObjectField(ObjectFieldOp op, ActualParameters actualParams);
  FailureOr<EvaluatorValuePtr>
  evaluateListCreate(ListCreateOp op, ActualParameters actualParams);
  FailureOr<EvaluatorValuePtr>
  evaluateTupleCreate(TupleCreateOp op, ActualParameters actualParams);
  FailureOr<EvaluatorValuePtr> evaluateTupleGet(TupleGetOp op,
                                                ActualParameters actualParams);
  FailureOr<evaluator::EvaluatorValuePtr>
  evaluateMapCreate(MapCreateOp op, ActualParameters actualParams);

  /// The symbol table for the IR module the Evaluator was constructed with.
  /// Used to look up class definitions.
  SymbolTable symbolTable;

  SmallVector<
      std::unique_ptr<SmallVector<std::shared_ptr<evaluator::EvaluatorValue>>>>
      actualParametersBuffers;

  // A worklist that needs to be fully evaluated.
  std::deque<Key> worklist;

  /// Object storage. Currently used for memoizing calls to
  /// evaluateObjectInstance. Further refinement is expected.
  DenseMap<Key, std::shared_ptr<evaluator::EvaluatorValue>> objects;
  DenseMap<Key, ActualParameters> objectCaller;
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
  else if (auto *map = llvm::dyn_cast<evaluator::MapValue>(&evaluatorValue))
    diag << "Map(" << map->getType() << ")";
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

// struct AlwaysLikeOpInfo : public
// llvm::DenseMapInfo<circt::om::Evaluator::Key> {
//   llvm::hash_code hash_value() const {
//     return llvm::hash_combine(Expr::hash_value(), *solution);
//   }
//
//   static inline circt::om::Evaluator::Key getEmpty() {
//     return circt::om::Evaluator::Key{mlir::Value::getEmptyKey(), nullptr};
//   }
//
//   static inline circt::om::Evaluator::Key getTombstoneKey() {
//     return circt::om::Evaluator::Key{
//         mlir::Value::getTombstoneKey(),
//         static_cast<SmallVectorImpl<std::shared_ptr<evaluator::EvaluatorValue>>
//                         *>(~0LL)};
//   }
//
//   static unsigned getHashValue(const circt::om::Evaluator::Key *opC) {
//     return 0;
//   }
//
//   static bool isEqual(const circt::om::Evaluator::Key *lhsC,
//                       const circt::om::Evaluator::Key *rhsC) {
//     return true;
//   }
// };

#endif // CIRCT_DIALECT_OM_EVALUATOR_EVALUATOR_H
