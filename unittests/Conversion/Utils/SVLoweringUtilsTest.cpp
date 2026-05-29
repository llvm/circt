//===- SVLoweringUtilsTest.cpp - SV lowering utility tests ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/SVLoweringUtils.h"
#include "circt/Dialect/Emit/EmitDialect.h"
#include "circt/Dialect/Emit/EmitOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "gtest/gtest.h"

using namespace circt;
using namespace mlir;

namespace {

class SVLoweringUtilsTest : public ::testing::Test {
protected:
  SVLoweringUtilsTest() {
    context.loadDialect<emit::EmitDialect, hw::HWDialect, sv::SVDialect>();
    loc = UnknownLoc::get(&context);
    module = ModuleOp::create(loc);
    builder = std::make_unique<ImplicitLocOpBuilder>(
        ImplicitLocOpBuilder::atBlockEnd(loc, module.getBody()));
  }

  MLIRContext context;
  LocationAttr loc;
  ModuleOp module;
  std::unique_ptr<ImplicitLocOpBuilder> builder;
};

TEST_F(SVLoweringUtilsTest, FileDescriptorRuntimeUsesResolvedMacroSymbol) {
  hw::HWModuleOp::create(*builder, builder->getStringAttr("SYNTHESIS"),
                         ArrayRef<hw::PortInfo>{});

  sv::emitFileDescriptorRuntime(module, *builder);

  auto synthesisMacro = module.lookupSymbol<sv::MacroDeclOp>("SYNTHESIS_0");
  ASSERT_TRUE(synthesisMacro);
  EXPECT_EQ(synthesisMacro.getMacroIdentifier(), "SYNTHESIS");

  auto fragment =
      module.lookupSymbol<emit::FragmentOp>("CIRCT_LIB_LOGGING_FRAGMENT");
  ASSERT_TRUE(fragment);
  auto ifdef = dyn_cast<sv::IfDefOp>(&fragment.getBody()->front());
  ASSERT_TRUE(ifdef);
  EXPECT_EQ(ifdef.getCond().getName(), "SYNTHESIS_0");
}

} // namespace
