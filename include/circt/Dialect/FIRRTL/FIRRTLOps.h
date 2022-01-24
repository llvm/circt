//===- FIRRTLOps.h - Declare FIRRTL dialect operations ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation class for the FIRRTL IR.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_OPS_H
#define CIRCT_DIALECT_FIRRTL_OPS_H

#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOpInterfaces.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/FieldRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace circt {
namespace firrtl {

/// Return true if the specified operation is a firrtl expression.
bool isExpression(Operation *op);

/// Return the number of ports in a module-like thing (modules, memories, etc)
size_t getNumPorts(Operation *op);

/// Return true if the specified operation has a constant value. This trivially
/// checks for `firrtl.constant` and friends, but also looks through subaccesses
/// and correctly handles wires driven with only constant values.
bool isConstant(Operation *op);
bool isConstant(Value value);

/// Returns true if the value results from an expression with duplex flow.
/// Duplex values have special treatment in bundle connect operations, and
/// their flip orientation is not used to determine the direction of each
/// pairwise connect.
bool isDuplexValue(Value val);

enum class Flow { Source, Sink, Duplex };

/// Get a flow's reverse.
Flow swapFlow(Flow flow);

/// Compute the flow for a Value, \p val, as determined by the FIRRTL
/// specification.  This recursively walks backwards from \p val to the
/// declaration.  The resulting flow is a combination of the declaration flow
/// (output ports and instance inputs are sinks, registers and wires are
/// duplex, anything else is a source) and the number of intermediary flips.
/// An even number of flips will result in the same flow as the declaration.
/// An odd number of flips will result in reversed flow being returned.  The
/// reverse of source is sink.  The reverse of sink is source.  The reverse of
/// duplex is duplex.  The \p accumulatedFlow parameter sets the initial flow.
/// A user should normally \a not have to change this from its default of \p
/// Flow::Source.
Flow foldFlow(Value val, Flow accumulatedFlow = Flow::Source);

enum class DeclKind { Port, Instance, Other };

DeclKind getDeclarationKind(Value val);

enum class ReadPortSubfield { addr, en, clk, data };
enum class WritePortSubfield { addr, en, clk, data, mask };
enum class ReadWritePortSubfield { addr, en, clk, rdata, wmode, wdata, wmask };

/// Allow 'or'ing MemDirAttr.  This allows combining Read and Write into
/// ReadWrite.
inline MemDirAttr operator|(MemDirAttr lhs, MemDirAttr rhs) {
  return static_cast<MemDirAttr>(
      static_cast<std::underlying_type<MemDirAttr>::type>(lhs) |
      static_cast<std::underlying_type<MemDirAttr>::type>(rhs));
}

inline MemDirAttr &operator|=(MemDirAttr &lhs, MemDirAttr rhs) {
  lhs = lhs | rhs;
  return lhs;
}

/// Check whether a block argument ("port") or the operation defining a value
/// has a `DontTouch` annotation, or a symbol that should prevent certain types
/// of canonicalizations.
bool hasDontTouch(Value value);

/// Check whether an operation has a `DontTouch` annotation, or a symbol that
/// should prevent certain types of canonicalizations.
bool hasDontTouch(Operation *op);

// Out-of-line implementation of various trait verification methods and
// functions commonly used among operations.
namespace impl {
LogicalResult verifySameOperandsIntTypeKind(Operation *op);

// Type inference adaptor for FIRRTL operations.
LogicalResult inferReturnTypes(
    MLIRContext *context, Optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::RegionRange regions,
    SmallVectorImpl<Type> &results,
    llvm::function_ref<FIRRTLType(ValueRange, ArrayRef<NamedAttribute>,
                                  Optional<Location>)>
        callback);

// Common type inference functions.
FIRRTLType inferAddSubResult(FIRRTLType lhs, FIRRTLType rhs,
                             Optional<Location> loc);
FIRRTLType inferBitwiseResult(FIRRTLType lhs, FIRRTLType rhs,
                              Optional<Location> loc);
FIRRTLType inferComparisonResult(FIRRTLType lhs, FIRRTLType rhs,
                                 Optional<Location> loc);
FIRRTLType inferReductionResult(FIRRTLType arg, Optional<Location> loc);

// Common parsed argument validation functions.
LogicalResult validateBinaryOpArguments(ValueRange operands,
                                        ArrayRef<NamedAttribute> attrs,
                                        Location loc);
LogicalResult validateUnaryOpArguments(ValueRange operands,
                                       ArrayRef<NamedAttribute> attrs,
                                       Location loc);
LogicalResult validateOneOperandOneConst(ValueRange operands,
                                         ArrayRef<NamedAttribute> attrs,
                                         Location loc);
} // namespace impl

/// A binary operation where the operands have the same integer kind.
template <typename ConcreteOp>
class SameOperandsIntTypeKind
    : public OpTrait::TraitBase<ConcreteOp, SameOperandsIntTypeKind> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifySameOperandsIntTypeKind(op);
  }
};

// This is a summary of a FIRRTL::MemOp. It defines the relevant properties of
// the FIRRTL memory, and can be constructed by parsing its attributes.
struct FirMemory {
  size_t numReadPorts;
  size_t numWritePorts;
  size_t numReadWritePorts;
  size_t dataWidth;
  size_t depth;
  size_t readLatency;
  size_t writeLatency;
  size_t maskBits;
  size_t readUnderWrite;
  hw::WUW writeUnderWrite;
  SmallVector<int32_t> writeClockIDs;
  StringAttr modName;

  // Location is carried along but not considered part of the identity of this.
  Location loc;

  std::tuple<size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t,
             size_t, hw::WUW, SmallVector<int32_t>>
  getTuple() const {
    return std::tie(numReadPorts, numWritePorts, numReadWritePorts, dataWidth,
                    depth, readLatency, writeLatency, maskBits, readUnderWrite,
                    writeUnderWrite, writeClockIDs);
  }
  bool operator<(const FirMemory &rhs) const {
    return getTuple() < rhs.getTuple();
  }
  bool operator==(const FirMemory &rhs) const {
    return getTuple() == rhs.getTuple();
  }
  StringAttr getFirMemoryName() const;
};
} // namespace firrtl
} // namespace circt

#define GET_OP_CLASSES
#include "circt/Dialect/FIRRTL/FIRRTL.h.inc"

//===----------------------------------------------------------------------===//
// Traits
//===----------------------------------------------------------------------===//

namespace llvm {
template <>
struct DenseMapInfo<circt::firrtl::FModuleOp> {
  using Operation = mlir::Operation;
  using FModuleOp = circt::firrtl::FModuleOp;
  static inline FModuleOp getEmptyKey() {
    return FModuleOp::getFromOpaquePointer(
        DenseMapInfo<Operation *>::getEmptyKey());
  }
  static inline FModuleOp getTombstoneKey() {
    return FModuleOp::getFromOpaquePointer(
        DenseMapInfo<Operation *>::getTombstoneKey());
  }
  static unsigned getHashValue(const FModuleOp &val) {
    return DenseMapInfo<Operation *>::getHashValue(val);
  }
  static bool isEqual(const FModuleOp &lhs, const FModuleOp &rhs) {
    return lhs == rhs;
  }
};
} // end namespace llvm

#endif // CIRCT_DIALECT_FIRRTL_OPS_H
