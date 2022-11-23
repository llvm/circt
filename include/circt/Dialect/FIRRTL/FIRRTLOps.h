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
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/FieldRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace circt {
namespace firrtl {

class StrictConnectOp;

// is the name useless?
bool isUselessName(circt::StringRef name);

// works for regs, nodes, and wires
bool hasDroppableName(Operation *op);

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

/// Return the StringAttr for the inner_sym name, if it exists.
inline StringAttr getInnerSymName(Operation *op) {
  auto s = op->getAttrOfType<hw::InnerSymAttr>(
      InnerSymbolTable::getInnerSymbolAttrName());
  if (s)
    return s.getSymName();
  return StringAttr();
}

/// Check whether a block argument ("port") or the operation defining a value
/// has a `DontTouch` annotation, or a symbol that should prevent certain types
/// of canonicalizations.
bool hasDontTouch(Value value);

/// Check whether an operation has a `DontTouch` annotation, or a symbol that
/// should prevent certain types of canonicalizations.
bool hasDontTouch(Operation *op);

/// Scan all the uses of the specified value, checking to see if there is
/// exactly one connect that has the value as its destination. This returns the
/// operation if found and if all the other users are "reads" from the value.
/// Returns null if there are no connects, or multiple connects to the value, or
/// if the value is involved in an `AttachOp`.
///
/// Note that this will simply return the connect, which is located *anywhere*
/// after the definition of the value. Users of this function are likely
/// interested in the source side of the returned connect, the definition of
/// which does likely not dominate the original value.
StrictConnectOp getSingleConnectUserOf(Value value);

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
  bool isMasked;
  uint32_t groupID;

  // Location is carried along but not considered part of the identity of this.
  Location loc;
  // Flag to indicate if the memory was under the DUT hierarchy, only used in
  // LowerToHW. Not part of the identity.
  bool isInDut = false;
  // The original MemOp, only used in LowerToHW.  Also not part of the identity.
  Operation *op = nullptr;

  std::tuple<size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t,
             size_t, hw::WUW, SmallVector<int32_t>, uint32_t>
  getTuple() const {
    return std::tie(numReadPorts, numWritePorts, numReadWritePorts, dataWidth,
                    depth, readLatency, writeLatency, maskBits, readUnderWrite,
                    writeUnderWrite, writeClockIDs, groupID);
  }
  bool operator<(const FirMemory &rhs) const {
    return getTuple() < rhs.getTuple();
  }
  bool operator==(const FirMemory &rhs) const {
    return getTuple() == rhs.getTuple();
  }
  StringAttr getFirMemoryName() const;
};

// Record of the inner sym name, the module name and the corresponding
// operation. Also the port index, if the symbol is on a module port.
// This is used to record the operations or ports, that have an inner sym.
// The operation and portIdx, can be null, when we have an InnerRefAttr, that is
// the module name and sym name, but we don't yet have a handle to the operation
// which has the Reference. So, InnerRefRecord can be used to construct illegal
// InnerRefAttr, which do not exist in the circt. That is the reason, the
// comparison operators here, only care for module name and symbol name.
struct InnerRefRecord {
  mlir::StringAttr mod, innerSym;
  mlir::Operation *op = nullptr;
  unsigned portIdx = 0;
  InnerRefRecord(StringAttr mod, StringAttr innerSym, Operation *op)
      : mod(mod), innerSym(innerSym), op(op) {}
  InnerRefRecord(StringAttr mod, StringAttr innerSym, Operation *op,
                 unsigned portIdx)
      : mod(mod), innerSym(innerSym), op(op), portIdx(portIdx) {}
  InnerRefRecord(hw::InnerRefAttr ref)
      : mod(ref.getModule()), innerSym(ref.getName()) {}
  bool operator<(const InnerRefRecord &rhs) const {
    return (innerSym.getValue() < rhs.innerSym.getValue() ||
            (innerSym == rhs.innerSym && mod.getValue() < rhs.mod));
  }
  bool operator==(const InnerRefRecord &rhs) const {
    return (innerSym == rhs.innerSym && mod == rhs.mod);
  }
  bool operator!=(const InnerRefRecord &rhs) const { return !(*this == rhs); }
};

// A data structure to record and lookup an InnerSym and the corresponding
// operation. Can be used when the list is populated first, then sorted and then
// only used for lookup. This does not handle duplicate entries explicitly. The
// list must be sorted, before any lookup. This is based on an observation that
// Dense arrays can be efficient lookup structures. Especially when we have
// insert-phase and lookup-phase based code.
// TODO: Generalize this data structure.
struct InnerRefList {
  InnerRefList(MLIRContext *context)
      : InnerSymAttr(StringAttr::get(context, "inner_sym")) {}

  void sort() {
    llvm::sort(list);
    sorted = true;
  }
  int search(const InnerRefRecord &key) const {
    assert(sorted && "Sort the list before search");
    if (!sorted || list.empty())
      return -1;
    const auto *iter = std::lower_bound(list.begin(), list.end(), key);
    if (iter == list.end())
      return -1;
    return (iter - list.begin());
  }
  bool exists(const InnerRefRecord &key) const { return search(key) != -1; }
  Operation *getOpIfExists(const hw::InnerRefAttr ref) const {
    auto index = search(InnerRefRecord(ref));
    if (index != -1 && list[index].op != nullptr)
      return list[index].op;
    return nullptr;
  }
  const InnerRefRecord *getRecordIfExists(const hw::InnerRefAttr ref) const {
    auto index = search(InnerRefRecord(ref));
    if (index != -1)
      return &list[index];
    return nullptr;
  }
  void pushBack(InnerRefRecord &key) {
    list.push_back(key);
    sorted = false;
  }
  // Inesrt the op with the module modName, if it has an inner sym.
  bool insert(Operation *op, StringAttr modName) {
    if (op == nullptr)
      return false;
    auto innerSym = op->getAttrOfType<StringAttr>(InnerSymAttr);
    if (!innerSym)
      return false;
    list.emplace_back(modName, innerSym, op);
    sorted = false;
    return true;
  }
  // Insert all the ports for the op, if they have the inner sym.
  bool insert(FModuleLike op) {
    StringAttr modName = op.moduleNameAttr();
    bool inserted = false;
    for (auto sym : llvm::enumerate(op.getPortSymbols()))
      if (sym.value()) {
        list.emplace_back(modName, sym.value().cast<StringAttr>(), op,
                          sym.index());
        inserted = true;
      }
    sorted = !inserted;
    return inserted;
  }

private:
  StringAttr InnerSymAttr;
  SmallVector<InnerRefRecord> list;
  bool sorted = false;
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
