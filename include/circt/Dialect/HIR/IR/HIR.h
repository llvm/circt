#ifndef HIR_HIR_H
#define HIR_HIR_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace circt {
namespace hir {
#include "circt/Dialect/HIR/IR/HIRInterfaces.h.inc"
enum PortKind { rd = 0, wr = 1, rw = 2 };
enum BusDirection { SAME = 0, FLIP = 1 };

enum DimKind { ADDR = 0, BANK = 1 };
namespace Details {

/// Storage class for MemrefType.
struct MemrefTypeStorage : public TypeStorage {
  MemrefTypeStorage(ArrayRef<int64_t> shape, Type elementType,
                    ArrayRef<DimKind> dimKinds)
      : shape(shape), elementType(elementType), dimKinds(dimKinds) {}

  using KeyTy = std::tuple<ArrayRef<int64_t>, Type, ArrayRef<DimKind>>;

  /// Define the comparison function for the key type.
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(shape, elementType, dimKinds);
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(std::get<0>(key), std::get<1>(key),
                              std::get<2>(key));
  }

  /// Define a construction function for the key type.
  static KeyTy getKey(ArrayRef<int64_t> shape, Type elementType,
                      ArrayRef<DimKind> dimKinds) {
    return KeyTy(shape, elementType, dimKinds);
  }

  /// Define a construction method for creating a new instance of this storage.
  static MemrefTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    ArrayRef<int64_t> shape = allocator.copyInto(std::get<0>(key));
    Type elementType = std::get<1>(key);
    ArrayRef<DimKind> dimKinds = allocator.copyInto(std::get<2>(key));
    return new (allocator.allocate<MemrefTypeStorage>())
        MemrefTypeStorage(shape, elementType, dimKinds);
  }

  ArrayRef<int64_t> shape;
  Type elementType;
  ArrayRef<DimKind> dimKinds;
};

/// Storage class for FuncType.
struct FuncTypeStorage : public TypeStorage {
  FuncTypeStorage(ArrayRef<Type> inputTypes,
                  ArrayRef<DictionaryAttr> inputAttrs,
                  ArrayRef<Type> resultTypes,
                  ArrayRef<DictionaryAttr> resultAttrs)
      : inputTypes(inputTypes), inputAttrs(inputAttrs),
        resultTypes(resultTypes), resultAttrs(resultAttrs) {}

  using KeyTy = std::tuple<ArrayRef<Type>, ArrayRef<DictionaryAttr>,
                           ArrayRef<Type>, ArrayRef<DictionaryAttr>>;

  /// Define the comparison function for the key type.
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(inputTypes, inputAttrs, resultTypes, resultAttrs);
  }

  /// Define a hash function for the key type.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(std::get<0>(key), std::get<1>(key));
  }

  /// Define a construction function for the key type.
  static KeyTy getKey(ArrayRef<Type> inputTypes,
                      ArrayRef<DictionaryAttr> inputAttrs,
                      ArrayRef<Type> resultTypes,
                      ArrayRef<DictionaryAttr> resultAttrs) {
    return KeyTy(inputTypes, inputAttrs, resultTypes, resultAttrs);
  }

  /// Define a construction method for creating a new instance of this storage.
  static FuncTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                    const KeyTy &key) {
    auto inputTypes = allocator.copyInto(std::get<0>(key));
    auto inputAttrs = allocator.copyInto(std::get<1>(key));
    auto resultTypes = allocator.copyInto(std::get<2>(key));
    auto resultAttrs = allocator.copyInto(std::get<3>(key));
    return new (allocator.allocate<FuncTypeStorage>())
        FuncTypeStorage(inputTypes, inputAttrs, resultTypes, resultAttrs);
  }

  ArrayRef<Type> inputTypes;
  ArrayRef<DictionaryAttr> inputAttrs;
  ArrayRef<Type> resultTypes;
  ArrayRef<DictionaryAttr> resultAttrs;
};

struct ArrayTypeStorage : public TypeStorage {
  ArrayTypeStorage(ArrayRef<int64_t> dims, Type elementType, Attribute attr)
      : dims(dims), elementType(elementType), attr(attr) {}

  using KeyTy = std::tuple<ArrayRef<int64_t>, Type, Attribute>;

  /// Define the comparison function for the key type.
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(dims, elementType, attr);
  }

  /// Define a hash function for the key type.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(std::get<0>(key), std::get<1>(key));
  }

  /// Define a construction function for the key type.
  static KeyTy getKey(ArrayRef<int64_t> dims, Type elementType,
                      Attribute attr) {
    return KeyTy(dims, elementType, attr);
  }

  /// Define a construction method for creating a new instance of this storage.
  static ArrayTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    ArrayRef<int64_t> dims = allocator.copyInto(std::get<0>(key));
    Type elementType = std::get<1>(key);
    Attribute attr = std::get<2>(key);
    return new (allocator.allocate<ArrayTypeStorage>())
        ArrayTypeStorage(dims, elementType, attr);
  }

  ArrayRef<int64_t> dims;
  Type elementType;
  Attribute attr;
};

struct GroupTypeStorage : public TypeStorage {
  GroupTypeStorage(ArrayRef<Type> elementTypes, ArrayRef<Attribute> attrs)
      : elementTypes(elementTypes), attrs(attrs) {}

  using KeyTy = std::tuple<ArrayRef<Type>, ArrayRef<Attribute>>;

  /// Define the comparison function for the key type.
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(elementTypes, attrs);
  }

  /// Define a hash function for the key type.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(std::get<0>(key), std::get<1>(key));
  }

  /// Define a construction function for the key type.
  static KeyTy getKey(ArrayRef<Type> elementTypes, ArrayRef<Attribute> attrs) {
    return KeyTy(elementTypes, attrs);
  }

  /// Define a construction method for creating a new instance of this storage.
  static GroupTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    ArrayRef<Type> elementTypes = allocator.copyInto(std::get<0>(key));
    ArrayRef<Attribute> attrs = allocator.copyInto(std::get<1>(key));
    return new (allocator.allocate<GroupTypeStorage>())
        GroupTypeStorage(elementTypes, attrs);
  }

  ArrayRef<Type> elementTypes;
  ArrayRef<Attribute> attrs;
};

/// Storage class for BusType.
struct BusTypeStorage : public TypeStorage {
  BusTypeStorage(ArrayRef<Type> elementTypes, ArrayRef<BusDirection> directions)
      : elementTypes(elementTypes), directions(directions) {}

  using KeyTy = std::tuple<ArrayRef<Type>, ArrayRef<BusDirection>>;

  /// Define the comparison function for the key type.
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(elementTypes, directions);
  }

  /// Define a hash function for the key type.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(std::get<0>(key), std::get<1>(key));
  }

  /// Define a construction function for the key type.
  static KeyTy getKey(ArrayRef<Type> elementTypes,
                      ArrayRef<BusDirection> directions) {
    return KeyTy(elementTypes, directions);
  }

  /// Define a construction method for creating a new instance of this storage.
  static BusTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                   const KeyTy &key) {
    auto elementTypes = allocator.copyInto(std::get<0>(key));
    auto directions = allocator.copyInto(std::get<1>(key));
    return new (allocator.allocate<BusTypeStorage>())
        BusTypeStorage(elementTypes, directions);
  }

  ArrayRef<Type> elementTypes;
  ArrayRef<BusDirection> directions;
};
} // namespace Details.

/// This class defines hir.time type in the dialect.
class TimeType : public Type::TypeBase<TimeType, Type, TypeStorage> {
public:
  using Base::Base;

  // static bool kindof(unsigned kind) { return kind == TimeKind; }
  static StringRef getKeyword() { return "time"; }
};

/// This class defines hir.memref type in the dialect.
class MemrefType
    : public Type::TypeBase<MemrefType, Type, Details::MemrefTypeStorage> {
public:
  using Base::Base;

  // static bool kindof(unsigned kind) { return kind == MemrefKind; }
  static StringRef getKeyword() { return "memref"; }
  static MemrefType get(MLIRContext *context, ArrayRef<int64_t> shape,
                        Type elementType, ArrayRef<DimKind> dimKinds) {
    assert(dimKinds.size() == shape.size());
    return Base::get(context, shape, elementType, dimKinds);
  }

  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                              ArrayRef<int64_t>, Type, ArrayRef<DimKind>);
  ArrayRef<int64_t> getShape() { return getImpl()->shape; }
  Type getElementType() { return getImpl()->elementType; }
  ArrayRef<DimKind> getDimKinds() { return getImpl()->dimKinds; }

  int getNumElementsPerBank() {
    int count = 1;
    auto dimKinds = getDimKinds();
    auto shape = getShape();
    for (size_t i = 0; i < shape.size(); i++) {
      if (dimKinds[i] == ADDR)
        count *= shape[i];
    }
    return count;
  }

  uint64_t getNumBanks() {
    uint64_t count = 1;
    auto dimKinds = getDimKinds();
    auto shape = getShape();
    for (size_t i = 0; i < shape.size(); i++) {
      if (dimKinds[i] == BANK) {
        count *= shape[i];
      }
    }
    return count;
  }

  SmallVector<int64_t, 4> filterShape(DimKind dimKind) {
    auto shape = getShape();
    auto dimKinds = getDimKinds();
    SmallVector<int64_t, 4> bankShape;
    for (size_t i = 0; i < getShape().size(); i++) {
      if (dimKinds[i] == dimKind)
        bankShape.push_back(shape[i]);
    }
    return bankShape;
  }
};

/// This class defines !hir.func type in the dialect.
class FuncType
    : public Type::TypeBase<FuncType, Type, Details::FuncTypeStorage> {
public:
  using Base::Base;

  static StringRef getKeyword() { return "func"; }

  // Depricated.
  // static FuncType get(MLIRContext *context, FunctionType functionTy,
  //                    ArrayRef<DictionaryAttr> inputAttrs,
  //                    ArrayRef<DictionaryAttr> resultAttrs) {
  //  return Base::get(context, functionTy, inputAttrs, resultAttrs);
  //}

  /// Build a new FuncType from the given attributes.
  static FuncType get(MLIRContext *context, ArrayRef<Type> inputTypes,
                      ArrayRef<DictionaryAttr> inputAttrs,
                      ArrayRef<Type> resultTypes,
                      ArrayRef<DictionaryAttr> resultAttrs) {
    return Base::get(context, inputTypes, inputAttrs, resultTypes, resultAttrs);
  }

  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                              ArrayRef<Type> inputTypes,
                              ArrayRef<DictionaryAttr> inputAttrs,
                              ArrayRef<Type> resultTypes,
                              ArrayRef<DictionaryAttr> resultAttrs);
  FunctionType getFunctionType() {
    return FunctionType::get(getContext(), getImpl()->inputTypes,
                             getImpl()->resultTypes);
  }
  ArrayRef<Type> getInputTypes() { return getImpl()->inputTypes; }
  ArrayRef<DictionaryAttr> getInputAttrs() { return getImpl()->inputAttrs; }
  ArrayRef<Type> getResultTypes() { return getImpl()->resultTypes; }
  ArrayRef<DictionaryAttr> getResultAttrs() { return getImpl()->resultAttrs; }
};

/// This class defines array type.
class ArrayType
    : public Type::TypeBase<ArrayType, Type, Details::ArrayTypeStorage> {
public:
  using Base::Base;

  static StringRef getKeyword() { return "array"; }

  static ArrayType get(MLIRContext *context, ArrayRef<int64_t> dims,
                       Type elementType, Attribute attr) {
    return Base::get(context, dims, elementType, attr);
  }

  ArrayRef<int64_t> dims;
  Type getElementType() { return getImpl()->elementType; }
  ArrayRef<int64_t> getDimensions() { return getImpl()->dims; }
  Attribute getAttribute() { return getImpl()->attr; }
};

/// This class defines group type.
class GroupType
    : public Type::TypeBase<GroupType, Type, Details::GroupTypeStorage> {
public:
  using Base::Base;

  static StringRef getKeyword() { return "group"; }
  static GroupType get(MLIRContext *context, ArrayRef<Type> elementTypes,
                       ArrayRef<Attribute> attrs) {
    return Base::get(context, elementTypes, attrs);
  }
  ArrayRef<Type> getElementTypes() { return getImpl()->elementTypes; }
  ArrayRef<Attribute> getAttributes() { return getImpl()->attrs; }
};

/// This class defines bus type.
class BusType : public Type::TypeBase<BusType, Type, Details::BusTypeStorage> {
public:
  using Base::Base;

  static StringRef getKeyword() { return "bus"; }
  static BusType get(MLIRContext *context, ArrayRef<Type> elementTypes,
                     ArrayRef<BusDirection> directions) {
    return Base::get(context, elementTypes, directions);
  }
  ArrayRef<Type> getElementTypes() { return getImpl()->elementTypes; }
  ArrayRef<BusDirection> getElementDirections() {
    return getImpl()->directions;
  }
};

} // namespace hir.
} // namespace circt
#define GET_OP_CLASSES
#include "circt/Dialect/HIR/IR/HIR.h.inc"

#endif // HIR_HIR_H.
