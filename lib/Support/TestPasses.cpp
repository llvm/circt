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

// A width and value parameterized adder.
template <typename CDEValue = DefaultCDEValue>
class MyAdder : public HWModuleGenerator<MyAdder<CDEValue>, CDEValue> {
public:
  using HWModuleGenerator<MyAdder<CDEValue>, CDEValue>::HWModuleGenerator;
  std::string getName(int width) { return "MyAdder_" + std::to_string(width); }
  void createIO(int width) {
    this->input("in0", this->template type<IntegerType>(width));
    this->input("in1", this->template type<IntegerType>(width));
    this->clock();
    this->output("out0", this->template type<IntegerType>(width));
  }

  void generate(CDEModulePorts<CDEValue> &ports, int width) {
    ports["out0"].assign((ports["in0"] + ports["in1"]).reg("add_reg"));
  }
};

// Example of specializing the generator with a custom value type that extends
// the default value type with a few extra operators.
class MyESIAdder : public HWModuleGenerator<MyESIAdder, ESICDEValue> {
public:
  using HWModuleGenerator::HWModuleGenerator;

  // Alternative to duplicating args in createIO, generate and getName is to
  // pack it beforehand.
  struct Args {
    int width;
  };

  std::string getName(Args &args) { return "MyESIAdder"; }
  void createIO(Args &args) {
    Type channelType = type<esi::ChannelType>(type<IntegerType>(args.width));
    input("in0", channelType);
    input("in1", channelType);
    clock();
    output("out0", channelType);
  }

  void generate(CDEModulePorts &ports, Args &args) {
    auto valid = wire(ctx.b.getI1Type());
    auto [inner1, valid1] = ports["in0"].unwrap(valid);
    auto [inner2, valid2] = ports["in1"].unwrap(valid);

    // TODO: some form of memoization is needed to avoid multiple definitions of
    // myAdder if this generator is called multiple times - but should this be
    // done by CPPCDE or the programmer? Since this is a library, probably the
    // latter. Could even pass in the adder through a reference in the args!
    auto myAdder = (MyAdder<ESICDEValue>(ctx.loc, ctx.b))(32);
    auto adderInstance =
        myAdder.value().instantiate("myAdder", {inner1, inner2, ports["clk"]});

    auto [outCh, outReady] = adderInstance["out0"].wrap(valid1 & valid2);
    valid.assign(outReady);
    ports["out0"].assign(outCh);
  }
};

void TestCPPCDEPass::runOnOperation() {
  MLIRContext *context = &getContext();
  Location loc = getOperation().getLoc();
  OpBuilder b(context);
  b.setInsertionPointToStart(getOperation().getBody());

  auto myESIAdder = (MyESIAdder(loc, b))(MyESIAdder::Args{32});
  if (failed(myESIAdder))
    return signalPassFailure();
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
