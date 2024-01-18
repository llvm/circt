//===- HWModuleTest.cpp - HW Module utility tests -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;
using namespace hw;

namespace {

TEST(HWModuleOpTest, AddOutputs) {
  // Create a hw.module with no ports.
  MLIRContext context;
  context.loadDialect<HWDialect>();
  LocationAttr loc = UnknownLoc::get(&context);
  auto module = ModuleOp::create(loc);
  auto builder = ImplicitLocOpBuilder::atBlockEnd(loc, module.getBody());
  auto top = builder.create<HWModuleOp>(StringAttr::get(&context, "Top"),
                                        ArrayRef<PortInfo>{});

  builder.setInsertionPointToStart(top.getBodyBlock());
  auto wireTy = builder.getIntegerType(2);

  // Add two ports.
  SmallVector<std::pair<StringAttr, Value>> appendPorts;
  auto wireA = builder.create<ConstantOp>(wireTy, 0);
  appendPorts.emplace_back(builder.getStringAttr("a"), wireA);
  auto wireD = builder.create<ConstantOp>(wireTy, 1);
  appendPorts.emplace_back(builder.getStringAttr("d"), wireD);
  top.appendOutputs(appendPorts);

  SmallVector<std::pair<StringAttr, Value>> insertPorts;
  auto wireB = builder.create<ConstantOp>(wireTy, 2);
  insertPorts.emplace_back(builder.getStringAttr("b"), wireB);
  auto wireC = builder.create<ConstantOp>(wireTy, 3);
  insertPorts.emplace_back(builder.getStringAttr("c"), wireC);
  top.insertOutputs(1, insertPorts);

  // Convenience methods.
  auto wireF = builder.create<hw::ConstantOp>(APInt(2, 0));
  top.appendOutput("f", wireF);
  auto wireQ = builder.create<hw::ConstantOp>(APInt(2, 0));
  top.prependOutput("q", wireQ);

  auto ports = top.getPortList();
  ASSERT_EQ(ports.size(), 6u);

  EXPECT_EQ(ports[0].name, builder.getStringAttr("q"));
  EXPECT_EQ(ports[0].dir, ModulePort::Direction::Output);
  EXPECT_EQ(ports[0].type, wireTy);

  EXPECT_EQ(ports[1].name, builder.getStringAttr("a"));
  EXPECT_EQ(ports[1].dir, ModulePort::Direction::Output);
  EXPECT_EQ(ports[1].type, wireTy);

  EXPECT_EQ(ports[2].name, builder.getStringAttr("b"));
  EXPECT_EQ(ports[2].dir, ModulePort::Direction::Output);
  EXPECT_EQ(ports[2].type, wireTy);

  EXPECT_EQ(ports[3].name, builder.getStringAttr("c"));
  EXPECT_EQ(ports[3].dir, ModulePort::Direction::Output);
  EXPECT_EQ(ports[3].type, wireTy);

  EXPECT_EQ(ports[4].name, builder.getStringAttr("d"));
  EXPECT_EQ(ports[4].dir, ModulePort::Direction::Output);
  EXPECT_EQ(ports[4].type, wireTy);

  EXPECT_EQ(ports[5].name, builder.getStringAttr("f"));
  EXPECT_EQ(ports[5].dir, ModulePort::Direction::Output);
  EXPECT_EQ(ports[5].type, wireTy);

  auto output = cast<OutputOp>(top.getBodyBlock()->getTerminator());
  ASSERT_EQ(output->getNumOperands(), 6u);

  EXPECT_EQ(output->getOperand(0), wireQ.getResult());
  EXPECT_EQ(output->getOperand(1), wireA.getResult());
  EXPECT_EQ(output->getOperand(2), wireB.getResult());
  EXPECT_EQ(output->getOperand(3), wireC.getResult());
  EXPECT_EQ(output->getOperand(4), wireD.getResult());
  EXPECT_EQ(output->getOperand(5), wireF.getResult());
}

TEST(HWModuleOpTest, AddInputs) {
  // Create a hw.module with no ports.
  MLIRContext context;
  context.loadDialect<HWDialect>();
  LocationAttr loc = UnknownLoc::get(&context);
  auto module = ModuleOp::create(loc);
  auto builder = ImplicitLocOpBuilder::atBlockEnd(loc, module.getBody());
  auto top = builder.create<HWModuleOp>(StringAttr::get(&context, "Top"),
                                        ArrayRef<PortInfo>{});

  builder.setInsertionPointToStart(top.getBodyBlock());

  // Convenience methods.
  auto tyF = builder.getIntegerType(6);
  top.appendInput("f", tyF);
  auto tyQ = builder.getIntegerType(1);
  top.prependInput("q", tyQ);

  auto ports = top.getPortList();
  ASSERT_EQ(ports.size(), 2u);

  EXPECT_EQ(ports[0].name, builder.getStringAttr("q"));
  EXPECT_EQ(ports[0].dir, ModulePort::Direction::Input);
  EXPECT_EQ(ports[0].type, tyQ);

  EXPECT_EQ(ports[1].name, builder.getStringAttr("f"));
  EXPECT_EQ(ports[1].dir, ModulePort::Direction::Input);
  EXPECT_EQ(ports[1].type, tyF);
}

TEST(HWModuleOpTest, SetPortAttr) {
  MLIRContext context;
  context.loadDialect<HWDialect>();
  LocationAttr loc = UnknownLoc::get(&context);
  auto module = ModuleOp::create(loc);
  auto builder = ImplicitLocOpBuilder::atBlockEnd(loc, module.getBody());

  SmallVector<PortInfo, 3> ports;
  auto ty = builder.getIntegerType(1);

  ports.emplace_back(PortInfo{
      {builder.getStringAttr("foo"), ty, hw::ModulePort::Direction::Input},
      0,
      {}});

  ports.emplace_back(PortInfo{
      {builder.getStringAttr("bar"), ty, hw::ModulePort::Direction::Input},
      1,
      {}});

  ports.emplace_back(PortInfo{
      {builder.getStringAttr("baz"), ty, hw::ModulePort::Direction::Output},
      0,
      {}});

  auto top = builder.create<HWModuleOp>(builder.getStringAttr("Top"), ports);

  auto emptyArrayAttr = ArrayAttr::get(&context, ArrayRef<Attribute>());
  EXPECT_EQ(top.getPerPortAttrsAttr(), emptyArrayAttr);

  auto fooStrAttr = builder.getStringAttr("fooAttr");
  auto bazStrAttr = builder.getStringAttr("bazAttr");

  NamedAttrList namedAttrs;
  namedAttrs.append(fooStrAttr, UnitAttr::get(&context));

  top.setPortAttrs(top.getPortIdForInputId(0),
                   namedAttrs.getDictionary(&context));

  top.setPortAttr(top.getPortIdForOutputId(0), bazStrAttr,
                  UnitAttr::get(&context));

  // Check attribute constraints
  auto adaptor = HWModuleOpAdaptor(top);
  EXPECT_TRUE(adaptor.verify(loc).succeeded());

  auto fooAttrs = top.getInputAttrs(0);
  auto barAttrs = top.getInputAttrs(1);
  auto bazAttrs = top.getOutputAttrs(0);

  auto emptyDictAttr = DictionaryAttr::get(&context);
  EXPECT_EQ(barAttrs, emptyDictAttr);

  ASSERT_TRUE(llvm::isa<DictionaryAttr>(fooAttrs));
  ASSERT_TRUE(llvm::isa<DictionaryAttr>(bazAttrs));

  auto fooDict = llvm::cast<DictionaryAttr>(fooAttrs);
  auto bazDict = llvm::cast<DictionaryAttr>(bazAttrs);

  EXPECT_TRUE(fooDict.contains(fooStrAttr));
  EXPECT_FALSE(fooDict.contains(bazStrAttr));

  EXPECT_FALSE(bazDict.contains(fooStrAttr));
  EXPECT_TRUE(bazDict.contains(bazStrAttr));

  top.setPortAttrs(top.getPortIdForInputId(0), emptyDictAttr);
  EXPECT_NE(top.getPerPortAttrsAttr(), emptyArrayAttr);
  EXPECT_TRUE(adaptor.verify(loc).succeeded());

  top.setPortAttrs(top.getPortIdForOutputId(0), emptyDictAttr);
  EXPECT_EQ(top.getPerPortAttrsAttr(), emptyArrayAttr);
  EXPECT_TRUE(adaptor.verify(loc).succeeded());
}

} // namespace
