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
#include "circt/Dialect/HIR/HIRInterfaces.h.inc"
enum PortKind { rd = 0, wr = 1, rw = 2 };

namespace Details {

/// Storage class for MemrefType.
struct MemrefTypeStorage : public TypeStorage {
  MemrefTypeStorage(ArrayRef<int64_t> shape, Type elementType,
                    ArrayAttr bankDims, DictionaryAttr portAttrs)
      : shape(shape), elementType(elementType), bankDims(bankDims),
        portAttrs(portAttrs) {}

  /// The hash key for this storage is a pair of the integer and type params.
  using KeyTy = std::tuple<ArrayRef<int64_t>, Type, ArrayAttr, DictionaryAttr>;

  /// Define the comparison function for the key type.
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(shape, elementType, bankDims, portAttrs);
  }

  /// Define a hash function for the key type.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(std::get<0>(key), std::get<1>(key),
                              std::get<2>(key), std::get<3>(key));
  }

  /// Define a construction function for the key type.
  static KeyTy getKey(ArrayRef<int64_t> shape, Type elementType,
                      ArrayAttr bankDims, DictionaryAttr portAttrs) {
    return KeyTy(shape, elementType, bankDims, portAttrs);
  }

  /// Define a construction method for creating a new instance of this storage.
  static MemrefTypeStorage *construct(TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    ArrayRef<int64_t> shape = allocator.copyInto(std::get<0>(key));
    Type elementType = std::get<1>(key);
    ArrayAttr bankDims = std::get<2>(key);
    DictionaryAttr portAttrs = std::get<3>(key);
    return new (allocator.allocate<MemrefTypeStorage>())
        MemrefTypeStorage(shape, elementType, bankDims, portAttrs);
  }

  ArrayRef<int64_t> shape;
  Type elementType;
  ArrayAttr bankDims;
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
  BusTypeStorage(ArrayRef<Type> elementTypes, ArrayRef<PortKind> directions,
                 DictionaryAttr proto)
      : elementTypes(elementTypes), directions(directions), proto(proto) {}

  /// The hash key for this storage is a pair of the integer and type params.
  using KeyTy = std::tuple<ArrayRef<Type>, ArrayRef<PortKind>, DictionaryAttr>;

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
                      ArrayRef<PortKind> directions, DictionaryAttr proto) {
    return KeyTy(elementTypes, directions, proto);
  }

  /// Define a construction method for creating a new instance of this storage.
  static BusTypeStorage *construct(TypeStorageAllocator &allocator,
                                   const KeyTy &key) {
    ArrayRef<Type> elementTypes = allocator.copyInto(std::get<0>(key));
    ArrayRef<PortKind> directions = allocator.copyInto(std::get<1>(key));
    DictionaryAttr proto = std::get<2>(key);
    return new (allocator.allocate<BusTypeStorage>())
        BusTypeStorage(elementTypes, directions, proto);
  }

  ArrayRef<Type> elementTypes;
  ArrayRef<PortKind> directions;
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

/// This class defines hir.memref type in the dialect.
class MemrefType
    : public Type::TypeBase<MemrefType, Type, Details::MemrefTypeStorage> {
public:
  using Base::Base;

  // static bool kindof(unsigned kind) { return kind == MemrefKind; }
  static StringRef getKeyword() { return "memref"; }
  static MemrefType get(MLIRContext *context, ArrayRef<int64_t> shape,
                        Type elementType, ArrayAttr bankDims,
                        DictionaryAttr portDims) {
    assert(bankDims);
    assert(portDims);
    return Base::get(context, shape, elementType, bankDims, portDims);
  }

  ArrayRef<int64_t> getShape() { return getImpl()->shape; }
  Type getElementType() { return getImpl()->elementType; }
  ArrayAttr getBankDims() { return getImpl()->bankDims; }
  DictionaryAttr getPortAttrs() { return getImpl()->portAttrs; }
  PortKind getPort() {
    DictionaryAttr portAttrs = getPortAttrs();
    auto rdAttr = portAttrs.getNamed("rd");
    auto wrAttr = portAttrs.getNamed("wr");
    if (rdAttr && wrAttr)
      return PortKind::rw;
    if (rdAttr)
      return PortKind::rd;
    return PortKind::wr;
  }

  int getDepth() {
    int depth = 1;
    auto addrDims = getAddrDims();
    auto shape = getShape();
    for (auto dim : addrDims) {
      depth *= shape[shape.size() - 1 - dim];
    }
    return depth;
  }

  int getNumBanks() {
    int numBanks = 1;
    auto bankDims = getBankDims();
    auto shape = getShape();
    for (auto dim : bankDims) {
      numBanks *=
          shape[shape.size() - 1 - dim.dyn_cast<IntegerAttr>().getInt()];
    }
    return numBanks;
  }
  SmallVector<int64_t, 4> getBankShape() {
    auto shape = getShape();
    SmallVector<int64_t, 4> bankShape;
    for (size_t i = 0; i < getShape().size(); i++) {
      bool isBankDim = false;
      for (auto dim : getBankDims())
        if (i == (size_t)dim.dyn_cast<IntegerAttr>().getInt())
          isBankDim = true;
      if (isBankDim)
        bankShape.push_back(shape[shape.size() - 1 - i]);
    }
    return bankShape;
  }

  SmallVector<int64_t, 4> getAddrShape() {
    auto shape = getShape();
    SmallVector<int64_t, 4> addrShape;
    for (size_t i = 0; i < getShape().size(); i++) {
      bool isAddrDim = false;
      for (auto dim : getAddrDims())
        if (i == (size_t)dim)
          isAddrDim = true;
      if (isAddrDim)
        addrShape.push_back(shape[shape.size() - 1 - i]);
    }
    return addrShape;
  }

  SmallVector<int, 4> getAddrDims() {
    SmallVector<int, 4> addrDims;
    for (size_t i = 0; i < getShape().size(); i++) {
      bool isBankDim = false;
      for (auto dim : getBankDims())
        if (i == (size_t)dim.dyn_cast<IntegerAttr>().getInt())
          isBankDim = true;
      if (!isBankDim)
        addrDims.push_back(i);
    }
    return addrDims;
  }
  enum DimKind { ADDR = 0, BANK = 1 };
  SmallVector<DimKind, 4> getDimensionKinds() {
    SmallVector<DimKind, 4> dimensionKinds;
    auto bankDims = getBankDims();
    for (auto i = 0; i < (int)getShape().size(); i++) {
      bool isBankDim = false;
      for (auto bankDim : bankDims) {
        if (i == bankDim.dyn_cast<IntegerAttr>().getInt())
          isBankDim = true;
      }
      if (isBankDim)
        dimensionKinds.push_back(BANK);
      else
        dimensionKinds.push_back(ADDR);
    }
    return dimensionKinds;
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
                     ArrayRef<PortKind> directions, DictionaryAttr proto) {
    return Base::get(context, elementTypes, directions, proto);
  }
  ArrayRef<Type> getElementTypes() { return getImpl()->elementTypes; }
  ArrayRef<PortKind> getElementDirections() { return getImpl()->directions; }
  DictionaryAttr getProto() { return getImpl()->proto; }
};

} // namespace hir.
#define GET_OP_CLASSES
#include "circt/Dialect/HIR/HIR.h.inc"
} // namespace mlir.

#endif // HIR_HIR_H.
