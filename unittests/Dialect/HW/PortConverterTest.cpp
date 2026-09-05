//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/PortConverter.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Verifier.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;
using namespace hw;

namespace {

class EraseOutput : public PortConversion {
public:
  using PortConversion::PortConversion;

  void mapInputSignals(OpBuilder &, Operation *, Value,
                       SmallVectorImpl<Value> &, ArrayRef<Backedge>) override {
    llvm_unreachable("input port unexpectedly selected for removal");
  }

  void mapOutputSignals(OpBuilder &, Operation *, Value instanceResult,
                        SmallVectorImpl<Value> &, ArrayRef<Backedge>) override {
    assert(instanceResult.use_empty() && "removed result must have no uses");
  }

private:
  void buildInputSignals() override {
    llvm_unreachable("input port unexpectedly selected for removal");
  }
  void buildOutputSignals() override {}
};

class TestPortConversionBuilder : public PortConversionBuilder {
public:
  using PortConversionBuilder::PortConversionBuilder;

  FailureOr<std::unique_ptr<PortConversion>> build(PortInfo port) override {
    if (port.isOutput() && port.getName() == "removed")
      return {std::make_unique<EraseOutput>(converter, port)};
    return PortConversionBuilder::build(port);
  }
};

TEST(PortConverterTest, PreserveUntouchedPortAndInstanceAttributes) {
  MLIRContext context;
  context.loadDialect<HWDialect>();
  auto loc = UnknownLoc::get(&context);
  auto circuit = ModuleOp::create(loc);
  auto builder = ImplicitLocOpBuilder::atBlockEnd(loc, circuit.getBody());
  auto i8 = builder.getI8Type();

  auto keptPortSym = builder.getStringAttr("keptPort");
  auto keptPortAttrs = builder.getDictionaryAttr({
      builder.getNamedAttr("hw.exportPort", InnerSymAttr::get(keptPortSym)),
      builder.getNamedAttr("hw.verilogName",
                           builder.getStringAttr("kept_name")),
  });
  SmallVector<PortInfo> childPorts = {
      {{builder.getStringAttr("in"), i8, ModulePort::Input}, 0},
      {{builder.getStringAttr("kept"), i8, ModulePort::Output},
       0,
       keptPortAttrs},
      {{builder.getStringAttr("removed"), i8, ModulePort::Output}, 1},
  };
  auto child = HWModuleOp::create(
      builder, builder.getStringAttr("Child"), ModulePortInfo(childPorts),
      [&](OpBuilder &, HWModulePortAccessor &ports) {
        ports.setOutput("kept", ports.getInput("in"));
        ports.setOutput("removed", ports.getInput("in"));
      });
  child.setVisibility(SymbolTable::Visibility::Private);

  SmallVector<PortInfo> topPorts = {
      {{builder.getStringAttr("in"), i8, ModulePort::Input}, 0},
      {{builder.getStringAttr("out"), i8, ModulePort::Output}, 0},
  };
  HWModuleOp::create(
      builder, builder.getStringAttr("Top"), ModulePortInfo(topPorts),
      [&](OpBuilder &builder, HWModulePortAccessor &ports) {
        auto instance = InstanceOp::create(
            builder, loc, child, "child",
            SmallVector<Value>{ports.getInput("in")}, {},
            InnerSymAttr::get(builder.getStringAttr("childInst")));
        instance.setDoNotPrintAttr(builder.getUnitAttr());
        instance->setAttr("hw.verilogName",
                          builder.getStringAttr("child_name"));
        ports.setOutput("out", instance.getResult(0));
      });

  auto keptPortRef = InnerRefAttr::get(child.getModuleNameAttr(), keptPortSym);
  HierPathOp::create(builder, builder.getStringAttr("keptPortPath"),
                     builder.getArrayAttr({keptPortRef}));

  ASSERT_TRUE(succeeded(verify(circuit.getOperation())));

  InstanceGraph instanceGraph(circuit);
  auto mutableChild = cast<HWMutableModuleLike>(child.getOperation());
  ASSERT_TRUE(succeeded(
      PortConverter<TestPortConversionBuilder>(instanceGraph, mutableChild)
          .run()));

  auto childPortList = child.getPortList();
  ASSERT_EQ(childPortList.size(), 2u);
  auto keptPort = llvm::find_if(
      childPortList, [](PortInfo port) { return port.getName() == "kept"; });
  ASSERT_NE(keptPort, childPortList.end());
  EXPECT_EQ(keptPort->getSym().getSymName(), keptPortSym);
  EXPECT_EQ(keptPort->attrs.get("hw.verilogName"),
            builder.getStringAttr("kept_name"));

  auto top = circuit.lookupSymbol<HWModuleOp>("Top");
  ASSERT_TRUE(top);
  auto instances = top.getOps<InstanceOp>();
  ASSERT_TRUE(llvm::hasSingleElement(instances));
  auto instance = *instances.begin();
  ASSERT_EQ(instance.getNumResults(), 1u);
  EXPECT_EQ(instance.getResultNames()[0], builder.getStringAttr("kept"));
  EXPECT_EQ(instance.getInnerSymAttr().getSymName(),
            builder.getStringAttr("childInst"));
  EXPECT_TRUE(instance.getDoNotPrint());
  EXPECT_EQ(instance->getAttr("hw.verilogName"),
            builder.getStringAttr("child_name"));
  EXPECT_EQ(cast<OutputOp>(top.getBodyBlock()->getTerminator()).getOperand(0),
            instance.getResult(0));

  EXPECT_TRUE(succeeded(verify(circuit.getOperation())));
}

} // namespace
