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
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
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
  const char *ir = R"MLIR(
module {
  hw.module private @Child(
      in %in: i8 {
        hw.exportPort = #hw<innerSym@inputPort>,
        hw.verilogName = "input_name"
      },
      out kept: i8 {
        hw.exportPort = #hw<innerSym@keptPort>,
        hw.verilogName = "kept_name"
      },
      out removed: i8) {
    hw.output %in, %in : i8, i8
  }
  hw.module @Top(in %in: i8, out out: i8) {
    %kept, %removed = hw.instance "child" sym @childInst @Child(
        in: %in: i8) -> (kept: i8, removed: i8) {
      doNotPrint,
      hw.verilogName = "child_name"
    }
    hw.output %kept : i8
  }
  hw.hierpath @inputPortPath [@Child::@inputPort]
  hw.hierpath @keptPortPath [@Child::@keptPort]
}
)MLIR";

  OwningOpRef<ModuleOp> circuit = parseSourceString<ModuleOp>(ir, &context);
  ASSERT_TRUE(circuit);
  ASSERT_TRUE(succeeded(verify(circuit->getOperation())));

  SymbolTable symbolTable(circuit.get());
  auto child = symbolTable.lookup<HWModuleOp>("Child");
  ASSERT_TRUE(child);
  auto inputPortSym = StringAttr::get(&context, "inputPort");
  auto keptPortSym = StringAttr::get(&context, "keptPort");

  InstanceGraph instanceGraph(circuit.get());
  auto mutableChild = cast<HWMutableModuleLike>(child.getOperation());
  ASSERT_TRUE(succeeded(
      PortConverter<TestPortConversionBuilder>(instanceGraph, mutableChild)
          .run()));

  auto childPortList = child.getPortList();
  ASSERT_EQ(childPortList.size(), 2u);
  auto *inputPort = llvm::find_if(
      childPortList, [](PortInfo port) { return port.getName() == "in"; });
  ASSERT_NE(inputPort, childPortList.end());
  ASSERT_TRUE(inputPort->getSym());
  EXPECT_EQ(inputPort->getSym().getSymName(), inputPortSym);
  EXPECT_EQ(inputPort->attrs.get("hw.verilogName"),
            StringAttr::get(&context, "input_name"));

  auto *keptPort = llvm::find_if(
      childPortList, [](PortInfo port) { return port.getName() == "kept"; });
  ASSERT_NE(keptPort, childPortList.end());
  ASSERT_TRUE(keptPort->getSym());
  EXPECT_EQ(keptPort->getSym().getSymName(), keptPortSym);
  EXPECT_EQ(keptPort->attrs.get("hw.verilogName"),
            StringAttr::get(&context, "kept_name"));

  auto top = symbolTable.lookup<HWModuleOp>("Top");
  ASSERT_TRUE(top);
  auto instances = top.getOps<InstanceOp>();
  ASSERT_TRUE(llvm::hasSingleElement(instances));
  auto instance = *instances.begin();
  ASSERT_EQ(instance.getNumOperands(), 1u);
  EXPECT_EQ(instance.getArgNames()[0], StringAttr::get(&context, "in"));
  EXPECT_EQ(instance.getInputs()[0], top.getBodyBlock()->getArgument(0));
  ASSERT_EQ(instance.getNumResults(), 1u);
  EXPECT_EQ(instance.getResultNames()[0], StringAttr::get(&context, "kept"));
  EXPECT_EQ(instance.getInnerSymAttr().getSymName(),
            StringAttr::get(&context, "childInst"));
  EXPECT_TRUE(instance.getDoNotPrint());
  EXPECT_EQ(instance->getAttr("hw.verilogName"),
            StringAttr::get(&context, "child_name"));
  EXPECT_EQ(cast<OutputOp>(top.getBodyBlock()->getTerminator()).getOperand(0),
            instance.getResult(0));

  EXPECT_TRUE(succeeded(verify(circuit->getOperation())));
}

} // namespace
