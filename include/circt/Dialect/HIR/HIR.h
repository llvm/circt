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

namespace mlir {
namespace hir {

enum WireDirection { in = 0, out = 1, inout = 2 };

namespace Details {
enum PortKind { r = 0, w = 1, rw = 2 };

/// Storage class for MemrefType.
struct MemrefTypeStorage : public TypeStorage {
  MemrefTypeStorage(ArrayRef<int64_t> shape, Type elementType,
                    ArrayAttr bankedDims, DictionaryAttr portAttrs)
      : shape(shape), elementType(elementType), bankedDims(bankedDims),
        portAttrs(portAttrs) {}

  /// The hash key for this storage is a pair of the integer and type params.
  using KeyTy = std::tuple<ArrayRef<int64_t>, Type, ArrayAttr, DictionaryAttr>;

  /// Define the comparison function for the key type.
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(shape, elementType, bankedDims, portAttrs);
  }

  /// Define a hash function for the key type.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(std::get<0>(key), std::get<1>(key),
                              std::get<2>(key), std::get<3>(key));
  }

  /// Define a construction function for the key type.
  static KeyTy getKey(ArrayRef<int64_t> shape, Type elementType,
                      ArrayAttr bankedDims, DictionaryAttr portAttrs) {
    return KeyTy(shape, elementType, bankedDims, portAttrs);
  }

  /// Define a construction method for creating a new instance of this storage.
  static MemrefTypeStorage *construct(TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    ArrayRef<int64_t> shape = allocator.copyInto(std::get<0>(key));
    Type elementType = std::get<1>(key);
    ArrayAttr bankedDims = std::get<2>(key);
    DictionaryAttr portAttrs = std::get<3>(key);
    return new (allocator.allocate<MemrefTypeStorage>())
        MemrefTypeStorage(shape, elementType, bankedDims, portAttrs);
  }

  ArrayRef<int64_t> shape;
  Type elementType;
  ArrayAttr bankedDims;
  DictionaryAttr portAttrs;
};

/// Storage class for FuncType.
struct FuncTypeStorage : public TypeStorage {
  FuncTypeStorage(FunctionType functionTy, ArrayAttr inputDelays,
                  ArrayAttr outputDelays)
      : functionTy(functionTy), inputDelays(inputDelays),
        outputDelays(outputDelays) {}

  /// The hash key for this storage is a pair of the integer and type params.
  using KeyTy = std::tuple<FunctionType, ArrayAttr, ArrayAttr>;

  /// Define the comparison function for the key type.
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(functionTy, inputDelays, outputDelays);
  }

  /// Define a hash function for the key type.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(std::get<0>(key), std::get<1>(key));
  }

  /// Define a construction function for the key type.
  static KeyTy getKey(FunctionType functionTy, ArrayAttr inputDelays,
                      ArrayAttr outputDelays) {
    return KeyTy(functionTy, inputDelays, outputDelays);
  }

  /// Define a construction method for creating a new instance of this storage.
  static FuncTypeStorage *construct(TypeStorageAllocator &allocator,
                                    const KeyTy &key) {
    FunctionType functionTy = std::get<0>(key);
    ArrayAttr inputDelays = std::get<1>(key);
    ArrayAttr outputDelays = std::get<2>(key);
    return new (allocator.allocate<FuncTypeStorage>())
        FuncTypeStorage(functionTy, inputDelays, outputDelays);
  }

  FunctionType functionTy;
  ArrayAttr inputDelays;
  ArrayAttr outputDelays;
};

struct ArrayTypeStorage : public TypeStorage {
  ArrayTypeStorage(ArrayRef<int64_t> dims, Type elementType, Attribute attr)
      : dims(dims), elementType(elementType), attr(attr) {}

  /// The hash key for this storage is a pair of the integer and type params.
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
  static ArrayTypeStorage *construct(TypeStorageAllocator &allocator,
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

  /// The hash key for this storage is a pair of the integer and type params.
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
  static GroupTypeStorage *construct(TypeStorageAllocator &allocator,
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
  BusTypeStorage(ArrayRef<Type> elementTypes,
                 ArrayRef<WireDirection> directions, DictionaryAttr proto)
      : elementTypes(elementTypes), directions(directions), proto(proto) {}

  /// The hash key for this storage is a pair of the integer and type params.
  using KeyTy =
      std::tuple<ArrayRef<Type>, ArrayRef<WireDirection>, DictionaryAttr>;

  /// Define the comparison function for the key type.
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(elementTypes, directions, proto);
  }

  /// Define a hash function for the key type.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(std::get<0>(key), std::get<1>(key),
                              std::get<2>(key));
  }

  /// Define a construction function for the key type.
  static KeyTy getKey(ArrayRef<Type> elementTypes,
                      ArrayRef<WireDirection> directions,
                      DictionaryAttr proto) {
    return KeyTy(elementTypes, directions, proto);
  }

  /// Define a construction method for creating a new instance of this storage.
  static BusTypeStorage *construct(TypeStorageAllocator &allocator,
                                   const KeyTy &key) {
    ArrayRef<Type> elementTypes = allocator.copyInto(std::get<0>(key));
    ArrayRef<WireDirection> directions = allocator.copyInto(std::get<1>(key));
    DictionaryAttr proto = std::get<2>(key);
    return new (allocator.allocate<BusTypeStorage>())
        BusTypeStorage(elementTypes, directions, proto);
  }

  ArrayRef<Type> elementTypes;
  ArrayRef<WireDirection> directions;
  DictionaryAttr proto;
};
} // namespace Details.

/// This class defines hir.time type in the dialect.
class TimeType : public Type::TypeBase<TimeType, Type, TypeStorage> {
public:
  using Base::Base;

  // static bool kindof(unsigned kind) { return kind == TimeKind; }
  static StringRef getKeyword() { return "time"; }
};

/// This class defines hir.const type in the dialect.
class ConstType : public Type::TypeBase<ConstType, Type, TypeStorage> {
public:
  using Base::Base;
  // static bool kindof(unsigned kind) { return kind == ConstKind; }
  static StringRef getKeyword() { return "const"; }
  // static ConstType get(MLIRContext *context) {
  //  return Base::get(context, ConstKind);
  //}
};

/// This class defines hir.memref type in the dialect.
class MemrefType
    : public Type::TypeBase<MemrefType, Type, Details::MemrefTypeStorage> {
public:
  using Base::Base;

  // static bool kindof(unsigned kind) { return kind == MemrefKind; }
  static StringRef getKeyword() { return "memref"; }
  static MemrefType get(MLIRContext *context, ArrayRef<int64_t> shape,
                        Type elementType, ArrayAttr bankedDims,
                        DictionaryAttr portDims) {
    assert(bankedDims);
    assert(portDims);
    return Base::get(context, shape, elementType, bankedDims, portDims);
  }

  ArrayRef<int64_t> getShape() { return getImpl()->shape; }
  Type getElementType() { return getImpl()->elementType; }
  ArrayAttr getBankedDims() { return getImpl()->bankedDims; }
  DictionaryAttr getPortAttrs() { return getImpl()->portAttrs; }
  Details::PortKind getPort() {
    DictionaryAttr portAttrs = getPortAttrs();
    auto rd = portAttrs.getNamed("rd");
    auto wr = portAttrs.getNamed("wr");
    if (rd && wr)
      return Details::rw;
    if (rd)
      return Details::r;
    return Details::w;
  }

  SmallVector<int, 4> getPacking() {
    SmallVector<int, 4> packedDims;
    for (size_t i = 0; i < getShape().size(); i++) {
      bool isBankedDim = false;
      for (auto dim : getBankedDims())
        if (i == (size_t)dim.dyn_cast<IntegerAttr>().getInt())
          isBankedDim = true;
      if (!isBankedDim)
        packedDims.push_back(i);
    }
    return packedDims;
  }
};

/// This class defines !hir.func type in the dialect.
class FuncType
    : public Type::TypeBase<FuncType, Type, Details::FuncTypeStorage> {
public:
  using Base::Base;

  static StringRef getKeyword() { return "func"; }
  static FuncType get(MLIRContext *context, FunctionType functionTy,
                      ArrayAttr inputDelays, ArrayAttr outputDelays) {
    return Base::get(context, functionTy, inputDelays, outputDelays);
  }

  FunctionType getFunctionType() { return getImpl()->functionTy; }
  ArrayAttr getInputDelays() { return getImpl()->inputDelays; }
  ArrayAttr getOutputDelays() { return getImpl()->outputDelays; }
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
                     ArrayRef<WireDirection> directions, DictionaryAttr proto) {
    return Base::get(context, elementTypes, directions, proto);
  }
  ArrayRef<Type> getElementTypes() { return getImpl()->elementTypes; }
  ArrayRef<WireDirection> getElementDirections() {
    return getImpl()->directions;
  }
  DictionaryAttr getProto() { return getImpl()->proto; }
};

} // namespace hir.
#define GET_OP_CLASSES
#include "circt/Dialect/HIR/HIR.h.inc"
} // namespace mlir.

#endif // HIR_HIR_H.
