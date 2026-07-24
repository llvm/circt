//===- SystemCOpsTest.cpp - SystemC op unit tests -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SystemC/SystemCOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;
using namespace hw;
using namespace systemc;

namespace {

TEST(SCModuleOpTest, GetPortListArgNumIsDirectionRelative) {
  MLIRContext context;
  context.loadDialect<SystemCDialect, HWDialect>();
  LocationAttr loc = UnknownLoc::get(&context);
  auto module = ModuleOp::create(loc);
  auto builder = ImplicitLocOpBuilder::atBlockEnd(loc, module.getBody());

  auto i1 = builder.getIntegerType(1);
  SmallVector<PortInfo> ports;
  ports.push_back(
      {{builder.getStringAttr("in0"), i1, ModulePort::Direction::Input}});
  ports.push_back(
      {{builder.getStringAttr("out0"), i1, ModulePort::Direction::Output}});
  ports.push_back(
      {{builder.getStringAttr("inout0"), i1, ModulePort::Direction::InOut}});
  ports.push_back(
      {{builder.getStringAttr("in1"), i1, ModulePort::Direction::Input}});
  ports.push_back(
      {{builder.getStringAttr("out1"), i1, ModulePort::Direction::Output}});

  auto scModule =
      SCModuleOp::create(builder, builder.getStringAttr("Top"), ports);

  auto portList = scModule.getPortList();
  ASSERT_EQ(portList.size(), 5u);

  EXPECT_EQ(portList[0].name, builder.getStringAttr("in0"));
  EXPECT_EQ(portList[0].dir, ModulePort::Direction::Input);
  EXPECT_EQ(portList[0].argNum, 0u);

  EXPECT_EQ(portList[1].name, builder.getStringAttr("out0"));
  EXPECT_EQ(portList[1].dir, ModulePort::Direction::Output);
  EXPECT_EQ(portList[1].argNum, 0u);

  EXPECT_EQ(portList[2].name, builder.getStringAttr("inout0"));
  EXPECT_EQ(portList[2].dir, ModulePort::Direction::InOut);
  EXPECT_EQ(portList[2].argNum, 1u);

  EXPECT_EQ(portList[3].name, builder.getStringAttr("in1"));
  EXPECT_EQ(portList[3].dir, ModulePort::Direction::Input);
  EXPECT_EQ(portList[3].argNum, 2u);

  EXPECT_EQ(portList[4].name, builder.getStringAttr("out1"));
  EXPECT_EQ(portList[4].dir, ModulePort::Direction::Output);
  EXPECT_EQ(portList[4].argNum, 1u);
}

} // namespace
