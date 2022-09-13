//===- RTLBuilder.h - CIRCT core RTL builder sugar --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A convenience builder for building CIRCT RTL operations.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_RTLBUILDER_H
#define CIRCT_SUPPORT_RTLBUILDER_H

#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class OpBuilder;
class PatternRewriter;
class Operation;
class ImplicitLocOpBuilder;
} // namespace mlir

namespace circt {

// A class containing a bunch of syntactic sugar to reduce builder function
// verbosity when building core RTL dialect operations.
class RTLBuilder {
public:
  RTLBuilder(mlir::OpBuilder &builder, mlir::Location loc,
             mlir::Value clk = mlir::Value(), mlir::Value rst = mlir::Value());
  RTLBuilder(mlir::ImplicitLocOpBuilder &builder,
             mlir::Value clk = mlir::Value(), mlir::Value rst = mlir::Value());

  // Return a constant value of the specified width and value.
  mlir::Value constant(unsigned width, int64_t value,
                       mlir::Location *extLoc = nullptr);

  // Create a register on the 'in' value and return the registered value.
  // If the RTLBuilder was created with a clock and reset, clock and reset
  // signals may be omitted from this function.
  mlir::Value reg(mlir::StringRef name, mlir::Value in, mlir::Value rstValue,
                  mlir::Value clk = mlir::Value(),
                  mlir::Value rst = mlir::Value(),
                  mlir::Location *extLoc = nullptr);

  // Bitwise AND.
  mlir::Value bAnd(mlir::ValueRange values, mlir::Location *extLoc = nullptr);
  // Bitwise NOT.
  mlir::Value bNot(mlir::Value value, mlir::Location *extLoc = nullptr);
  // Bitwise OR.
  mlir::Value bOr(mlir::ValueRange values, mlir::Location *extLoc = nullptr);
  // Bitwise XOR.
  mlir::Value bXor(mlir::ValueRange values, mlir::Location *extLoc = nullptr);

  mlir::OpBuilder &b;

protected:
  mlir::Location getLoc(mlir::Location *extLoc = nullptr) {
    return extLoc ? *extLoc : loc;
  }

  mlir::Location loc;
  mlir::Value clk, rst;
};

} // namespace circt

#endif // CIRCT_SUPPORT_RTLBUILDER_H
