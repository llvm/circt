//===- TestPasses.cpp - Test passes for the support infrastructure -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements test passes for the support infrastructure.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Support/CPPCDE.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace circt;
using namespace cppcde;

//===----------------------------------------------------------------------===//
// CPPCDE passes.
//===----------------------------------------------------------------------===//

namespace {
struct TestCPPCDEPass
    : public PassWrapper<TestCPPCDEPass, OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestCPPCDEPass)

  void runOnOperation() override;
  StringRef getArgument() const override { return "test-cppcde"; }
  StringRef getDescription() const override {
    return "Runs various built-in cppcde tests.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<hw::HWDialect, comb::CombDialect, seq::SeqDialect,
                    esi::ESIDialect>();
  }
};
} // namespace

class MyAdder : public HWModuleGenerator<DefaultCDEValue> {
public:
  using HWModuleGenerator::HWModuleGenerator;
  std::string getName() override { return "MyAdder"; }
  void createIO() override {
    input("in0", ctx.b.getIntegerType(32));
    input("in1", ctx.b.getIntegerType(32));
    clock();
    output("out0", ctx.b.getIntegerType(32));
  }

  void generate(CDEPorts &ports) override {
    ports["out0"].assign((ports["in0"] + ports["in1"]).reg("add_reg"));
  }
};

// Example of specializing the generator with a custom value type that extends
// the default value type with a few extra operators.
class MyESIAdder : public HWModuleGenerator<ESICDEValue> {
public:
  using HWModuleGenerator::HWModuleGenerator;
  std::string getName() override { return "MyESIAdder"; }
  void createIO() override {
    Type channelType =
        esi::ChannelType::get(ctx.b.getContext(), ctx.b.getIntegerType(32));
    input("in0", channelType);
    input("in1", channelType);
    clock();
    output("out0", channelType);
  }

  void generate(CDEPorts &ports) override {
    auto c1_i1 = cint(1, 1);
    auto valid = wire(ctx.b.getI1Type());
    auto [inner1, valid1] = ports["in0"].unwrap(valid);
    auto [inner2, valid2] = ports["in1"].unwrap(valid);
    auto res = inner1 + inner2;
    auto [outCh, outReady] = res.wrap(valid1 & valid2);
    valid.assign(outReady);
    ports["out0"].assign(outCh);
  }
};

void TestCPPCDEPass::runOnOperation() {
  MLIRContext *context = &getContext();
  Location loc = getOperation().getLoc();
  OpBuilder b(context);
  b.setInsertionPointToStart(getOperation().getBody());

  if (failed((MyAdder(loc, b))()) || failed((MyESIAdder(loc, b))())) {
    signalPassFailure();
    return;
  }
}

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace circt {
namespace test {
void registerSupportTestPasses() {
  mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return std::make_unique<TestCPPCDEPass>();
  });
}

} // namespace test
} // namespace circt
