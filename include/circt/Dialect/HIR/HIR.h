#ifndef HIR_HIR_H
#define HIR_HIR_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace hir {
enum Kinds {
  TimeKind = Type::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE,
  ConstKind,
  MemrefKind,
  WireKind
};

class TimeType : public Type::TypeBase<TimeType, Type, DefaultTypeStorage> {
  /**
   * This class defines hir.time type in the dialect.
   */
public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == TimeKind; }
  static llvm::StringRef getKeyword() { return "time"; }
  static TimeType get(MLIRContext *context) {
    return Base::get(context, TimeKind);
  }
};

class ConstType : public Type::TypeBase<ConstType, Type, DefaultTypeStorage> {
  /**
   * This class defines hir.const type in the dialect.
   */

public:
  using Base::Base;
  static bool kindof(unsigned kind) { return kind == ConstKind; }
  static llvm::StringRef getKeyword() { return "const"; }
  static ConstType get(MLIRContext *context) {
    return Base::get(context, ConstKind);
  }
};

namespace MemrefDetails {
enum PortKind { r = 0, w = 1, rw = 2 };

struct MemrefTypeStorage : public TypeStorage {
  MemrefTypeStorage(ArrayRef<unsigned> shape, Type elementType,
                    ArrayRef<unsigned> packing, PortKind port = rw)
      : shape(shape), elementType(elementType), packing(packing), port(port) {}

  /// The hash key for this storage is a pair of the integer and type params.
  using KeyTy =
      std::tuple<ArrayRef<unsigned>, Type, ArrayRef<unsigned>, PortKind>;

  /// Define the comparison function for the key type.
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(shape, elementType, packing, port);
  }

  /// Define a hash function for the key type.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(std::get<0>(key), std::get<1>(key),
                              std::get<2>(key), std::get<3>(key));
  }

  /// Define a construction function for the key type.
  static KeyTy getKey(ArrayRef<unsigned> shape, Type elementType,
                      ArrayRef<unsigned> packing, PortKind port = rw) {
    return KeyTy(shape, elementType, packing, port);
  }

  /// Define a construction method for creating a new instance of this storage.
  static MemrefTypeStorage *construct(TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    ArrayRef<unsigned> shape = allocator.copyInto(std::get<0>(key));
    Type elementType = std::get<1>(key);
    ArrayRef<unsigned> packing = allocator.copyInto(std::get<2>(key));
    MemrefDetails::PortKind port = std::get<3>(key);
    return new (allocator.allocate<MemrefTypeStorage>())
        MemrefTypeStorage(shape, elementType, packing, port);
  }

  ArrayRef<unsigned> shape;
  ArrayRef<unsigned> packing;
  Type elementType;
  PortKind port;
};
} // namespace MemrefDetails

class MemrefType : public Type::TypeBase<MemrefType, Type,
                                         MemrefDetails::MemrefTypeStorage> {
  /**
   * This class defines hir.memref type in the dialect.
   */
public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == MemrefKind; }
  static llvm::StringRef getKeyword() { return "memref"; }
  static MemrefType get(MLIRContext *context, ArrayRef<unsigned> shape,
                        Type elementType, ArrayRef<unsigned> packing,
                        MemrefDetails::PortKind port) {
    return Base::get(context, MemrefKind, shape, elementType, packing, port);
  }
  ArrayRef<unsigned> getShape() { return getImpl()->shape; }
  Type getElementType() { return getImpl()->elementType; }
  ArrayRef<unsigned> getPacking() { return getImpl()->packing; }
  MemrefDetails::PortKind getPort() { return getImpl()->port; }
};

class WireType : public Type::TypeBase<WireType, Type, DefaultTypeStorage> {
  /**
   * This class defines hir.wire type in the dialect.
   */
public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == WireKind; }
  static llvm::StringRef getKeyword() { return "wire"; }
  static WireType get(MLIRContext *context) {
    return Base::get(context, WireKind);
  }
};

#define GET_OP_CLASSES
#include "circt/Dialect/HIR/HIR.h.inc"

} // namespace hir
} // namespace mlir

#endif // HIR_HIR_H
