//===- CalyxLoweringUtils.h - Calyx lowering utility methods ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines various lowering utility methods for converting to
// and from Calyx programs.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_CALYX_CALYXLOWERINGUTILS_H
#define CIRCT_DIALECT_CALYX_CALYXLOWERINGUTILS_H

#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/PatternMatch.h"

#include <variant>

namespace circt {
namespace calyx {

// A structure representing a set of ports which act as a memory interface for
// external memories.
struct MemoryPortsImpl {
  Value readData;
  Value done;
  Value writeData;
  SmallVector<Value> addrPorts;
  Value writeEn;
};

// Represents the interface of memory in Calyx. The various lowering passes
// are agnostic wrt. whether working with a calyx::MemoryOp (internally
// allocated memory) or MemoryPortsImpl (external memory).
struct MemoryInterface {
  MemoryInterface();
  explicit MemoryInterface(const MemoryPortsImpl &ports);
  explicit MemoryInterface(calyx::MemoryOp memOp);

  // Getter methods for each memory interface port.
  Value readData();
  Value done();
  Value writeData();
  Value writeEn();
  ValueRange addrPorts();

private:
  std::variant<calyx::MemoryOp, MemoryPortsImpl> impl;
};

// Provides an interface for the control flow `while` operation across different
// dialects.
template <typename T>
class WhileOpInterface {
  static_assert(std::is_convertible_v<T, Operation *>);

public:
  explicit WhileOpInterface(T op) : impl(op) {}
  explicit WhileOpInterface(Operation *op) : impl(dyn_cast_or_null<T>(op)) {}

  virtual ~WhileOpInterface() = default;

  // Returns the arguments to this while operation.
  virtual Block::BlockArgListType getBodyArgs() = 0;

  // Returns body of this while operation.
  virtual Block *getBodyBlock() = 0;

  // Returns the Block in which the condition exists.
  virtual Block *getConditionBlock() = 0;

  // Returns the condition as a Value.
  virtual Value getConditionValue() = 0;

  // Returns the number of iterations the while loop will conduct if known.
  virtual Optional<uint64_t> getBound() = 0;

  // Returns the operation.
  T getOperation() { return impl; }

  // Returns the source location of the operation.
  Location getLoc() { return impl->getLoc(); }

private:
  T impl;
};

// Walks the control of this component, and appends source information for leaf
// nodes. It also appends a position attribute that connects the source location
// metadata to the corresponding control operation.
WalkResult
getCiderSourceLocationMetadata(calyx::ComponentOp component,
                               SmallVectorImpl<Attribute> &sourceLocations);

// Tries to match a constant value defined by op. If the match was
// successful, returns true and binds the constant to 'value'. If unsuccessful,
// the value is unmodified.
bool matchConstantOp(Operation *op, APInt &value);

// Returns true if there exists only a single memref::LoadOp which loads from
// the memory referenced by loadOp.
bool singleLoadFromMemory(Value memoryReference);

// Returns true if there are no memref::StoreOp uses with the referenced
// memory.
bool noStoresToMemory(Value memoryReference);

// Get the index'th output port of compOp.
Value getComponentOutput(calyx::ComponentOp compOp, unsigned outPortIdx);

// If the provided type is an index type, converts it to i32, else, returns the
// unmodified type.
Type convIndexType(PatternRewriter &rewriter, Type type);

// Creates a new calyx::CombGroupOp or calyx::GroupOp group within compOp.
template <typename TGroup>
TGroup createGroup(PatternRewriter &rewriter, calyx::ComponentOp compOp,
                   Location loc, Twine uniqueName) {
  mlir::IRRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(compOp.getWiresOp().getBody());
  return rewriter.create<TGroup>(loc, uniqueName.str());
}

} // namespace calyx
} // namespace circt

#endif // CIRCT_DIALECT_CALYX_CALYXLOWERINGUTILS_H
