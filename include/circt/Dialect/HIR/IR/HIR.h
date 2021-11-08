#ifndef HIR_HIR_H
#define HIR_HIR_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace circt {
namespace hir {
enum PortKind { rd = 0, wr = 1, rw = 2 };
enum BusDirection { SAME = 0, FLIPPED = 1 };

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

/// Storage class for BusType.
struct BusTypeStorage : public TypeStorage {
  BusTypeStorage(ArrayRef<StringAttr> memberNames, ArrayRef<Type> memberTypes,
                 ArrayRef<BusDirection> memberDirections)
      : memberNames(memberNames), memberTypes(memberTypes),
        memberDirections(memberDirections) {}

  using KeyTy =
      std::tuple<ArrayRef<StringAttr>, ArrayRef<Type>, ArrayRef<BusDirection>>;

  /// Define the comparison function for the key type.
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(memberNames, memberTypes, memberDirections);
  }

  /// Define a hash function for the key type.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(std::get<0>(key), std::get<1>(key));
  }

  /// Define a construction function for the key type.
  static KeyTy getKey(ArrayRef<StringAttr> memberNames,
                      ArrayRef<Type> memberTypes,
                      ArrayRef<BusDirection> memberDirections) {
    return KeyTy(memberNames, memberTypes, memberDirections);
  }

  /// Define a construction method for creating a new instance of this storage.
  static BusTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                   const KeyTy &key) {
    auto memberNames = allocator.copyInto(std::get<0>(key));
    auto memberTypes = allocator.copyInto(std::get<1>(key));
    auto memberDirections = allocator.copyInto(std::get<2>(key));
    return new (allocator.allocate<BusTypeStorage>())
        BusTypeStorage(memberNames, memberTypes, memberDirections);
  }

  ArrayRef<StringAttr> memberNames;
  ArrayRef<Type> memberTypes;
  ArrayRef<BusDirection> memberDirections;
};

/// Storage class for BusTensorType.
struct BusTensorTypeStorage : public TypeStorage {
  BusTensorTypeStorage(ArrayRef<int64_t> shape, Type elementTy)
      : shape(shape), elementTy(elementTy) {}

  using KeyTy = std::tuple<ArrayRef<int64_t>, Type>;

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(shape, elementTy);
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(std::get<0>(key), std::get<1>(key));
  }

  /// Define a construction function for the key type.
  static KeyTy getKey(ArrayRef<int64_t> shape, Type elementTy) {
    return KeyTy(shape, elementTy);
  }

  /// Define a construction method for creating a new instance of this storage.
  static BusTensorTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                         const KeyTy &key) {
    auto shape = allocator.copyInto(std::get<0>(key));
    auto elementTy = std::get<1>(key);
    return new (allocator.allocate<BusTensorTypeStorage>())
        BusTensorTypeStorage(shape, elementTy);
  }

  ArrayRef<int64_t> shape;
  Type elementTy;
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

  int64_t getNumElementsPerBank() {
    int count = 1;
    auto dimKinds = getDimKinds();
    auto shape = getShape();
    for (size_t i = 0; i < shape.size(); i++) {
      if (dimKinds[i] == ADDR)
        count *= shape[i];
    }
    return count;
  }

  int64_t getNumBanks() {
    int64_t count = 1;
    auto dimKinds = getDimKinds();
    auto shape = getShape();
    for (size_t i = 0; i < shape.size(); i++) {
      if (dimKinds[i] == BANK) {
        count *= shape[i];
      }
    }
    return count;
  }

  SmallVector<int64_t> filterShape(DimKind dimKind) {
    auto shape = getShape();
    auto dimKinds = getDimKinds();
    SmallVector<int64_t> bankShape;
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

/// This class defines a bus type.
class BusType : public Type::TypeBase<BusType, Type, Details::BusTypeStorage> {
public:
  using Base::Base;

  static StringRef getKeyword() { return "bus"; }
  static BusType get(MLIRContext *context, Type type) {

    return Base::get(context, StringAttr::get(context, "bus"), type,
                     hir::BusDirection::SAME);
  }

  Type getElementType() { return getImpl()->memberTypes[0]; }
};

/// This class defines a bus_tensor type.
class BusTensorType : public Type::TypeBase<BusTensorType, Type,
                                            Details::BusTensorTypeStorage> {
public:
  using Base::Base;

  static StringRef getKeyword() { return "bus_tensor"; }
  static BusTensorType get(MLIRContext *context, ArrayRef<int64_t> shape,
                           Type type) {

    return Base::get(context, shape, type);
  }

  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                              ArrayRef<int64_t> shape, Type elementTy);
  Type getElementType() { return getImpl()->elementTy; }
  ArrayRef<int64_t> getShape() { return getImpl()->shape; }
  size_t getNumElements() {
    size_t numElements = 1;
    for (auto dim : getShape())
      numElements *= dim;
    return numElements;
  }
};

/// This class defines bus struct type.
class BusStructType
    : public Type::TypeBase<BusStructType, Type, Details::BusTypeStorage> {
public:
  using Base::Base;

  static StringRef getKeyword() { return "bus_struct"; }
  static BusStructType get(MLIRContext *context,
                           ArrayRef<StringAttr> memberNames,
                           ArrayRef<Type> memberTypes,
                           ArrayRef<BusDirection> memberDirections) {
    return Base::get(context, memberNames, memberTypes, memberDirections);
  }

  ArrayRef<StringAttr> getMemberNames() { return getImpl()->memberNames; }
  ArrayRef<Type> getMemberTypes() { return getImpl()->memberTypes; }
  ArrayRef<BusDirection> getMemberDirections() {
    return getImpl()->memberDirections;
  }
};

} // namespace hir.
} // namespace circt

#include "circt/Dialect/HIR/IR/HIROpInterfaces.h.inc"
#define GET_OP_CLASSES
#include "circt/Dialect/HIR/IR/HIR.h.inc"

#endif // HIR_HIR_H.
