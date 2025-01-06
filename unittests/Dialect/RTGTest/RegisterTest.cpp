//===- RegisterTest.cpp - RTGTest register unit tests ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTGTest/IR/RTGTestOps.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;
using namespace rtgtest;

namespace {

TEST(RegisterInterfaceTest, IntegerRegisters) {
  MLIRContext context;
  context.loadDialect<RTGTestDialect>();
  Location loc(UnknownLoc::get(&context));

  SmallVector<std::tuple<Registers, std::string, unsigned>> regs{
      {Registers::zero, "zero", 0}, {Registers::ra, "ra", 1},
      {Registers::sp, "sp", 2},     {Registers::gp, "gp", 3},
      {Registers::tp, "tp", 4},     {Registers::t0, "t0", 5},
      {Registers::t1, "t1", 6},     {Registers::t2, "t2", 7},
      {Registers::s0, "s0", 8},     {Registers::s1, "s1", 9},
      {Registers::a0, "a0", 10},    {Registers::a1, "a1", 11},
      {Registers::a2, "a2", 12},    {Registers::a3, "a3", 13},
      {Registers::a4, "a4", 14},    {Registers::a5, "a5", 15},
      {Registers::a6, "a6", 16},    {Registers::a7, "a7", 17},
      {Registers::s2, "s2", 18},    {Registers::s3, "s3", 19},
      {Registers::s4, "s4", 20},    {Registers::s5, "s5", 21},
      {Registers::s6, "s6", 22},    {Registers::s7, "s7", 23},
      {Registers::s8, "s8", 24},    {Registers::s9, "s9", 25},
      {Registers::s10, "s10", 26},  {Registers::s11, "s11", 27},
      {Registers::t3, "t3", 28},    {Registers::t4, "t4", 29},
      {Registers::t5, "t5", 30},    {Registers::t6, "t6", 31}};

  auto moduleOp = ModuleOp::create(loc);
  OpBuilder builder = OpBuilder::atBlockBegin(moduleOp.getBody());
  auto regOp = builder.create<RegisterOp>(loc, Registers::Virtual);
  ASSERT_EQ(regOp.getAllowedRegs(),
            llvm::BitVector(getMaxEnumValForRegisters(), true));
  ASSERT_EQ(regOp.getFixedReg(), ~0U);

  for (auto [reg, str, idx] : regs) {
    regOp.setFixedReg(idx);
    ASSERT_EQ(regOp.getClassIndex(), idx);
    ASSERT_EQ(regOp.getClassIndexBinary(), APInt(5, idx));
    ASSERT_EQ(regOp.getRegisterAssembly(), str);
    ASSERT_EQ(regOp.getAllowedRegs(),
              llvm::BitVector(getMaxEnumValForRegisters(), false).set(idx));
    ASSERT_EQ(regOp.getFixedReg(), idx);
  }
}

} // namespace
