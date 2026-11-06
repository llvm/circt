//===- HWModuleTest.cpp - HW Module utility tests -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWAttributes.h"
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
  auto top = HWModuleOp::create(builder, StringAttr::get(&context, "Top"),
                                ArrayRef<PortInfo>{});

  builder.setInsertionPointToStart(top.getBodyBlock());
  auto wireTy = builder.getIntegerType(2);

  // Add two ports.
  SmallVector<std::pair<StringAttr, Value>> appendPorts;
  auto wireA = ConstantOp::create(builder, wireTy, 0);
  appendPorts.emplace_back(builder.getStringAttr("a"), wireA);
  auto wireD = ConstantOp::create(builder, wireTy, 1);
  appendPorts.emplace_back(builder.getStringAttr("d"), wireD);
  top.appendOutputs(appendPorts);

  SmallVector<std::pair<StringAttr, Value>> insertPorts;
  auto wireB = ConstantOp::create(builder, wireTy, 2);
  insertPorts.emplace_back(builder.getStringAttr("b"), wireB);
  auto wireC = ConstantOp::create(builder, wireTy, 3);
  insertPorts.emplace_back(builder.getStringAttr("c"), wireC);
  top.insertOutputs(1, insertPorts);

  // Convenience methods.
  auto wireF = hw::ConstantOp::create(builder, APInt(2, 0));
  top.appendOutput("f", wireF);
  auto wireQ = hw::ConstantOp::create(builder, APInt(2, 0));
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
  auto top = HWModuleOp::create(builder, StringAttr::get(&context, "Top"),
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

TEST(HWSpecializationAttrTest, VerifyAndHelpers) {
  MLIRContext context;
  context.loadDialect<HWDialect>();
  std::string capturedMsg;
  auto captureEmit = [&]() -> InFlightDiagnostic {
    capturedMsg.clear();
    return context.getDiagEngine().emit(UnknownLoc::get(&context),
                                        DiagnosticSeverity::Error);
  };
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diag) {
    capturedMsg = diag.str();
    return success();
  });

  StringAttr origName = StringAttr::get(&context, "Orig");
  IntegerType i32Ty = IntegerType::get(&context, 32);
  IntegerType i16Ty = IntegerType::get(&context, 16);
  NoneType noneTy = NoneType::get(&context);

  // Valid attribute round-trips through getSpecialization / setSpecialization.
  auto validParam = ParamDeclAttr::get(StringAttr::get(&context, "W"),
                                       IntegerAttr::get(i32Ty, 16));
  auto validSpec = SpecializationAttr::getChecked(
      captureEmit, &context, origName, ArrayRef<ParamDeclAttr>{validParam},
      /*boundaryAltered=*/true);
  ASSERT_TRUE(static_cast<bool>(validSpec));
  EXPECT_EQ(validSpec.getOrigName(), origName);
  EXPECT_TRUE(validSpec.getBoundaryAltered());

  // Null operation handling in helpers.
  EXPECT_FALSE(getSpecialization(nullptr));
  setSpecialization(nullptr, validSpec);

  Location loc = UnknownLoc::get(&context);
  OwningOpRef<ModuleOp> root(ModuleOp::create(loc));
  auto builder = ImplicitLocOpBuilder::atBlockEnd(loc, root->getBody());
  auto mod = HWModuleOp::create(builder, StringAttr::get(&context, "M"),
                                ArrayRef<PortInfo>{});
  setSpecialization(mod, validSpec);
  EXPECT_EQ(getSpecialization(mod), validSpec);
  setSpecialization(mod, SpecializationAttr());
  EXPECT_FALSE(getSpecialization(mod));

  // Unreachable-from-parser verifier branches.
  EXPECT_FALSE(SpecializationAttr::getChecked(
      captureEmit, &context, StringAttr(), ArrayRef<ParamDeclAttr>{},
      /*boundaryAltered=*/false));
  EXPECT_EQ(capturedMsg,
            "specialization original module name must not be empty");

  EXPECT_FALSE(SpecializationAttr::getChecked(
      captureEmit, &context, origName, ArrayRef<ParamDeclAttr>{ParamDeclAttr()},
      /*boundaryAltered=*/false));
  EXPECT_EQ(capturedMsg, "specialization parameter entry must not be null");

  auto nullNameParam =
      ParamDeclAttr::get(&context, StringAttr(), noneTy, Attribute());
  EXPECT_FALSE(SpecializationAttr::getChecked(
      captureEmit, &context, origName, ArrayRef<ParamDeclAttr>{nullNameParam},
      /*boundaryAltered=*/false));
  EXPECT_EQ(capturedMsg, "specialization parameter name must not be empty");

  auto untypedValParam =
      ParamDeclAttr::get(&context, StringAttr::get(&context, "U"), noneTy,
                         UnitAttr::get(&context));
  EXPECT_FALSE(SpecializationAttr::getChecked(
      captureEmit, &context, origName, ArrayRef<ParamDeclAttr>{untypedValParam},
      /*boundaryAltered=*/false));
  EXPECT_NE(capturedMsg.find("must have a typed value, got"),
            std::string::npos);

  auto mismatchedTypeParam =
      ParamDeclAttr::get(&context, StringAttr::get(&context, "M"), i32Ty,
                         IntegerAttr::get(i16Ty, 1));
  EXPECT_FALSE(SpecializationAttr::getChecked(
      captureEmit, &context, origName,
      ArrayRef<ParamDeclAttr>{mismatchedTypeParam}, /*boundaryAltered=*/false));
  EXPECT_NE(capturedMsg.find("must have type 'i32', got 'i16'"),
            std::string::npos);
}

} // namespace
