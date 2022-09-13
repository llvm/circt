//===- RTLBuilder.cpp - CIRCT core RTL builder sugar ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/RTLBuilder.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace mlir;
using namespace circt;

RTLBuilder::RTLBuilder(mlir::OpBuilder &builder, mlir::Location loc, Value clk,
                       Value rst)
    : b(builder), loc(loc), clk(clk), rst(rst) {}

RTLBuilder::RTLBuilder(mlir::ImplicitLocOpBuilder &builder, Value clk,
                       Value rst)
    : b(builder), loc(builder.getLoc()), clk(clk), rst(rst) {}

Value RTLBuilder::constant(unsigned width, int64_t value, Location *extLoc) {
  return b.create<hw::ConstantOp>(getLoc(extLoc), APInt(width, value));
}

Value RTLBuilder::reg(StringRef name, Value in, Value rstValue, Value clk,
                      Value rst, Location *extLoc) {
  Value resolvedClk = clk ? clk : this->clk;
  Value resolvedRst = rst ? rst : this->rst;
  assert(resolvedClk && "No global clock provided to this RTLBuilder - a clock "
                        "signal must be provided to the reg(...) function.");
  assert(resolvedRst && "No global reset provided to this RTLBuilder - a reset "
                        "signal must be provided to the reg(...) function.");

  return b.create<seq::CompRegOp>(getLoc(extLoc), in.getType(), in, resolvedClk,
                                  name, resolvedRst, rstValue,
                                  mlir::StringAttr());
}

Value RTLBuilder::bAnd(ValueRange values, Location *extLoc) {
  return b.create<comb::AndOp>(getLoc(extLoc), values).getResult();
}

Value RTLBuilder::bNot(Value value, Location *extLoc) {
  return comb::createOrFoldNot(getLoc(extLoc), value, b);
}

Value RTLBuilder::bOr(ValueRange values, Location *extLoc) {
  return b.create<comb::OrOp>(getLoc(extLoc), values).getResult();
}

Value RTLBuilder::bXor(ValueRange values, Location *extLoc) {
  return b.create<comb::XorOp>(getLoc(extLoc), values).getResult();
}
