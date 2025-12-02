//===- PortsTest.cpp - Port accessor tests --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOpInterfaces.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "mlir/IR/Builders.h"

#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;
using namespace firrtl;

namespace {

//===----------------------------------------------------------------------===//
// Port accessor tests
//===----------------------------------------------------------------------===//

class PortsTest : public ::testing::Test {
protected:
  void SetUp() override { context.loadDialect<FIRRTLDialect>(); }

  MLIRContext context;
  OpBuilder builder{&context};

  template <typename T, typename... Args>
  void checkPortsAgainstHWPortInfo(const Twine &name, Args &&...args) {
    auto mod =
        T::create(builder, UnknownLoc::get(&context),
                  builder.getStringAttr(name), std::forward<Args>(args)...);
    ASSERT_TRUE(isa<FModuleLike>(mod.getOperation()));
    auto allFPorts = mod.getPorts();
    auto allHWPorts = mod.getPortList();
    ASSERT_EQ(allFPorts.size(), allHWPorts.size());

    size_t inputs = 0, outputs = 0;
    for (auto [fport, hwport] : llvm::zip_equal(allFPorts, allHWPorts)) {
      EXPECT_EQ(fport.name, hwport.name);
      EXPECT_EQ(fport.isInput(), hwport.isInput());
      EXPECT_EQ(fport.isOutput(), hwport.isOutput());
      EXPECT_EQ(fport.isInOut(), hwport.isInOut());
      EXPECT_EQ(fport.loc, hwport.loc);
      EXPECT_EQ(fport.sym, hwport.getSym());
      EXPECT_EQ(hwport.attrs.size(), (fport.sym ? 1 : 0))
          << "hw::PortInfo attributes should only contain inner symbol?";

      if (fport.isInput())
        ++inputs;
      if (fport.isOutput())
        ++outputs;
    }
    EXPECT_EQ(inputs, mod.getNumInputPorts());
    EXPECT_EQ(outputs, mod.getNumOutputPorts());
    EXPECT_EQ(inputs + outputs, mod.getNumPorts());
  }
};

// https://github.com/llvm/circt/issues/9278
TEST_F(PortsTest, GetClassPortNoSym_Issue9278) {
  auto strType = StringType::get(&context);
  StringAttr name1 = builder.getStringAttr("name1");
  StringAttr name2 = builder.getStringAttr("name2");

  SmallVector<PortInfo, 2> ports = {{name1, strType, Direction::Out},
                                    {name2, strType, Direction::In}};

  auto clazz =
      ClassOp::create(builder, UnknownLoc::get(&context),
                      builder.getStringAttr("ClassWithoutPortSym"), ports);

  auto allPorts = clazz.getPorts();

  ASSERT_TRUE(allPorts.size() == 2);
  EXPECT_EQ(allPorts[0].getName(), name1.strref());
  EXPECT_EQ(allPorts[0].sym, clazz.getPort(0).getSym());
  EXPECT_EQ(allPorts[1].getName(), name2.strref());
  EXPECT_EQ(allPorts[1].sym, clazz.getPort(1).getSym());
}

TEST_F(PortsTest, CheckModPortsNoSyms) {
  auto strType = StringType::get(&context);
  auto intType = FIntegerType::get(&context);
  StringAttr name1 = builder.getStringAttr("name1");
  StringAttr name2 = builder.getStringAttr("name2");
  Location loc1 = NameLoc::get(builder.getStringAttr("loc1"));
  SmallVector<PortInfo, 2> ports = {
      {name1, strType, Direction::Out, /*symName=*/{}, loc1},
      {name2, intType, Direction::In}};

  checkPortsAgainstHWPortInfo<ClassOp>("clazz", ports);
  checkPortsAgainstHWPortInfo<ExtClassOp>("extclazz", ports);
  checkPortsAgainstHWPortInfo<FModuleOp>(
      "modular", ConventionAttr::get(&context, Convention::Internal), ports);
  checkPortsAgainstHWPortInfo<FExtModuleOp>(
      "extmodular", ConventionAttr::get(&context, Convention::Scalarized),
      ports, /*knownLayers=*/ArrayAttr());
}

TEST_F(PortsTest, CheckModPortsSyms) {
  auto strType = StringType::get(&context);
  auto intType = FIntegerType::get(&context);
  Location loc1 = NameLoc::get(builder.getStringAttr("loc1"));
  Location loc2 = NameLoc::get(builder.getStringAttr("loc2"));
  StringAttr name1 = builder.getStringAttr("name1");
  StringAttr name2 = builder.getStringAttr("name2");
  SmallVector<PortInfo, 2> ports = {
      {name1, strType, Direction::Out, builder.getStringAttr("foo"), loc1},
      {name2, intType, Direction::In, builder.getStringAttr("bar"), loc2}};

  checkPortsAgainstHWPortInfo<ClassOp>("clazz", ports);
  checkPortsAgainstHWPortInfo<ExtClassOp>("extclazz", ports);
  checkPortsAgainstHWPortInfo<FModuleOp>(
      "modular", ConventionAttr::get(&context, Convention::Internal), ports);
  checkPortsAgainstHWPortInfo<FExtModuleOp>(
      "extmodular", ConventionAttr::get(&context, Convention::Scalarized),
      ports, /*knownLayers=*/ArrayAttr());
}

TEST_F(PortsTest, CheckNoPorts) {
  ArrayRef<PortInfo> ports;

  checkPortsAgainstHWPortInfo<ClassOp>("clazz", ports);
  checkPortsAgainstHWPortInfo<ExtClassOp>("extclazz", ports);
  checkPortsAgainstHWPortInfo<FModuleOp>(
      "modular", ConventionAttr::get(&context, Convention::Internal), ports);
  checkPortsAgainstHWPortInfo<FExtModuleOp>(
      "extmodular", ConventionAttr::get(&context, Convention::Scalarized),
      ports, /*knownLayers=*/ArrayAttr());
}

} // end namespace
