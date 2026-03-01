//===- RegisterAllocationInterfaceTest.cpp - Register Allocation tests ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTG/IR/RTGISAAssemblyOpInterfaces.h"
#include "circt/Dialect/RTG/IR/RTGOps.h"
#include "circt/Dialect/RTGTest/IR/RTGTestAttributes.h"
#include "circt/Dialect/RTGTest/IR/RTGTestDialect.h"
#include "circt/Dialect/RTGTest/IR/RTGTestOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;
using namespace rtg;
using namespace rtgtest;

namespace {

class RegisterAllocationInterfaceTest : public ::testing::Test {
protected:
  RegisterAllocationInterfaceTest() : builder(&context) {}

  void SetUp() override {
    context.loadDialect<RTGTestDialect>();
    context.loadDialect<RTGDialect>();
    loc = UnknownLoc::get(&context);
    moduleOp = ModuleOp::create(loc);
    builder.setInsertionPointToStart(moduleOp.getBody());

    rd = RegRaAttr::get(&context);
    rs1 = RegS0Attr::get(&context);
    rs2 = RegS1Attr::get(&context);

    imm12 = ImmediateAttr::get(&context, APInt(12, 0));
    imm13 = ImmediateAttr::get(&context, APInt(13, 0));
    imm21 = ImmediateAttr::get(&context, APInt(21, 0));
    imm32 = ImmediateAttr::get(&context, APInt(32, 0));
    imm5 = ImmediateAttr::get(&context, APInt(5, 0));
  }

  Value createConstant(Attribute attr) {
    auto typedAttr = cast<TypedAttr>(attr);
    return ConstantOp::create(builder, loc, typedAttr.getType(), typedAttr);
  }

  MLIRContext context;
  Location loc = UnknownLoc::get(&context);
  ModuleOp moduleOp;
  OpBuilder builder;
  Attribute rd, rs1, rs2;
  Attribute imm12, imm13, imm21, imm32, imm5;
};

TEST_F(RegisterAllocationInterfaceTest, RFormatInstruction) {
  auto rdVal = createConstant(rd);
  auto rs1Val = createConstant(rs1);
  auto rs2Val = createConstant(rs2);

  auto addOp = ADD::create(builder, loc, rdVal, rs1Val, rs2Val);
  auto iface = cast<RegisterAllocationOpInterface>(addOp.getOperation());

  EXPECT_TRUE(iface.isDestinationRegister(0));
  EXPECT_FALSE(iface.isSourceRegister(0));

  EXPECT_TRUE(iface.isSourceRegister(1));
  EXPECT_FALSE(iface.isDestinationRegister(1));

  EXPECT_TRUE(iface.isSourceRegister(2));
  EXPECT_FALSE(iface.isDestinationRegister(2));

  SmallVector<unsigned> sourceIndices;
  iface.getSourceRegisterIndices(sourceIndices);
  EXPECT_EQ(sourceIndices.size(), 2u);
  EXPECT_TRUE(llvm::is_contained(sourceIndices, 1u));
  EXPECT_TRUE(llvm::is_contained(sourceIndices, 2u));

  SmallVector<unsigned> destIndices;
  iface.getDestinationRegisterIndices(destIndices);
  EXPECT_EQ(destIndices.size(), 1u);
  EXPECT_EQ(destIndices[0], 0u);
}

TEST_F(RegisterAllocationInterfaceTest, OutOfBoundsIndices) {
  auto rdVal = createConstant(rd);
  auto rs1Val = createConstant(rs1);
  auto rs2Val = createConstant(rs2);

  auto addOp = ADD::create(builder, loc, rdVal, rs1Val, rs2Val);

  EXPECT_FALSE(addOp.isSourceRegister(3));
  EXPECT_FALSE(addOp.isSourceRegister(100));
  EXPECT_FALSE(addOp.isDestinationRegister(3));
  EXPECT_FALSE(addOp.isDestinationRegister(100));
}

TEST_F(RegisterAllocationInterfaceTest, NoRegisterInstruction) {
  auto ebreakOp = EBREAKOp::create(builder, loc);
  auto iface = cast<RegisterAllocationOpInterface>(ebreakOp.getOperation());

  SmallVector<unsigned> sourceIndices;
  iface.getSourceRegisterIndices(sourceIndices);
  EXPECT_EQ(sourceIndices.size(), 0u);

  SmallVector<unsigned> destIndices;
  iface.getDestinationRegisterIndices(destIndices);
  EXPECT_EQ(destIndices.size(), 0u);
}

} // namespace
