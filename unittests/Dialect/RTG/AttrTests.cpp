//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTG/IR/RTGAttributes.h"
#include "circt/Dialect/RTG/IR/RTGDialect.h"
#include "circt/Dialect/RTGTest/IR/RTGTestAttributes.h"
#include "circt/Dialect/RTGTest/IR/RTGTestDialect.h"
#include "mlir/IR/Builders.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;
using namespace rtg;
using namespace rtgtest;

namespace {

// Helper class to capture diagnostic messages
class DiagnosticCapture {
public:
  DiagnosticCapture(MLIRContext *context) : context(context) {
    handler =
        context->getDiagEngine().registerHandler([this](Diagnostic &diag) {
          messages.push_back(diag.str());
          return success();
        });
  }

  ~DiagnosticCapture() { context->getDiagEngine().eraseHandler(handler); }

  const SmallVector<std::string> &getMessages() const { return messages; }

  void clear() { messages.clear(); }

private:
  MLIRContext *context;
  DiagnosticEngine::HandlerID handler;
  SmallVector<std::string> messages;
};

TEST(VirtualRegisterConfigAttrTests, EmptyAllowedRegs) {
  MLIRContext context;
  context.loadDialect<RTGDialect>();
  context.loadDialect<RTGTestDialect>();

  DiagnosticCapture capture(&context);

  VirtualRegisterConfigAttr::getChecked(
      [&]() { return emitError(UnknownLoc::get(&context)); }, &context, {});

  EXPECT_EQ(capture.getMessages().size(), 1UL);
  EXPECT_STREQ(capture.getMessages()[0].c_str(),
               "must have at least one allowed register");
}

TEST(VirtualRegisterConfigAttrTests, SameTypeAllowedRegs) {
  MLIRContext context;
  context.loadDialect<RTGDialect>();
  context.loadDialect<RTGTestDialect>();

  DiagnosticCapture capture(&context);

  auto regA0 = RegA0Attr::get(&context);
  auto regF0 = RegF0Attr::get(&context);

  VirtualRegisterConfigAttr::getChecked(
      [&]() { return emitError(UnknownLoc::get(&context)); }, &context,
      {regA0, regF0});

  EXPECT_EQ(capture.getMessages().size(), 1UL);
  EXPECT_STREQ(capture.getMessages()[0].c_str(),
               "all allowed registers must be of the same type");
}

} // namespace
