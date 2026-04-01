//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lib/Conversion/ImportVerilog/CaptureAnalysis.h"
#include "slang/ast/ASTVisitor.h"
#include "slang/ast/Compilation.h"
#include "slang/syntax/SyntaxTree.h"
#include "llvm/Support/Valgrind.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace slang;
using namespace slang::ast;
using namespace slang::syntax;
using namespace circt::ImportVerilog;
using testing::ElementsAre;
using testing::IsEmpty;

namespace {

/// Maps function names to sorted, unique lists of captured variable names.
using NamedCaptures = std::map<std::string, std::vector<std::string>>;

/// Test fixture that skips all tests when running under Valgrind, since slang
/// triggers Valgrind's uninitialized value detection.
class CaptureAnalysisTest : public testing::Test {
protected:
  void SetUp() override {
    if (llvm::sys::RunningOnValgrind())
      GTEST_SKIP() << "Slang triggers Valgrind false positives";
  }

  /// Parse Verilog, run capture analysis, and return a map from function name
  /// to sorted capture variable names. All AST pointers are resolved to strings
  /// before the compilation is destroyed.
  NamedCaptures analyze(std::string_view code) {
    auto tree = SyntaxTree::fromText(code);
    Compilation compilation;
    compilation.addSyntaxTree(tree);
    auto captures = analyzeFunctionCaptures(compilation.getRoot());

    NamedCaptures result;
    for (auto &[func, caps] : captures) {
      auto &names = result[std::string(func->name)];
      for (auto *var : caps)
        names.emplace_back(var->name);
      std::sort(names.begin(), names.end());
      names.erase(std::unique(names.begin(), names.end()), names.end());
    }
    return result;
  }
};

// Virtual interface member accesses should not be treated as captures.
// Slang resolves `vif.data` directly to a NamedValueExpression for the
// interface signal, but this is lowered through the virtual interface
// mechanism, not through capture parameters.
TEST_F(CaptureAnalysisTest, VirtualInterfaceMemberNotCaptured) {
  auto captures = analyze(R"(
    interface MyIf;
      logic data;
    endinterface

    class consumer;
      virtual MyIf vif;
      function void drive(logic val);
        vif.data = val;
      endfunction
    endclass

    module top;
      MyIf intf();
      initial begin
        consumer c = new;
        c.vif = intf;
        c.drive(1'b1);
      end
    endmodule
  )");
  EXPECT_THAT(captures["drive"], IsEmpty());
}

// Parameters are compile-time constants and should not be captured.
TEST_F(CaptureAnalysisTest, ParameterNotCaptured) {
  auto captures = analyze(R"(
    module top;
      parameter MAX = 256;
      function automatic int scale();
        return $random % MAX;
      endfunction
    endmodule
  )");
  EXPECT_THAT(captures["scale"], IsEmpty());
}

// Localparams are compile-time constants and should not be captured.
TEST_F(CaptureAnalysisTest, LocalparamNotCaptured) {
  auto captures = analyze(R"(
    module top;
      localparam OFFSET = 42;
      function automatic int get();
        return $random + OFFSET;
      endfunction
    endmodule
  )");
  EXPECT_THAT(captures["get"], IsEmpty());
}

// Enum values are compile-time constants and should not be captured.
TEST_F(CaptureAnalysisTest, EnumValueNotCaptured) {
  auto captures = analyze(R"(
    module top;
      typedef enum logic [1:0] { A, B, C } state_t;
      task automatic check(state_t s);
        if (s === B)
          $display("ok");
      endtask
    endmodule
  )");
  EXPECT_THAT(captures["check"], IsEmpty());
}

// Class type parameters are compile-time constants and should not be captured.
TEST_F(CaptureAnalysisTest, ClassTypeParamNotCaptured) {
  auto captures = analyze(R"(
    module top;
      class C #(int N = 10);
        function automatic int get();
          return $random % N;
        endfunction
      endclass
    endmodule
  )");
  EXPECT_THAT(captures["get"], IsEmpty());
}

} // namespace

// A function referencing a module-level variable should capture it.
TEST_F(CaptureAnalysisTest, DirectCapture) {
  auto captures = analyze(R"(
    module top;
      int x;
      function void foo();
        x = 42;
      endfunction
    endmodule
  )");
  EXPECT_THAT(captures["foo"], ElementsAre("x"));
}

// A function referencing only local variables should have no captures.
TEST_F(CaptureAnalysisTest, NoCapture) {
  auto captures = analyze(R"(
    module top;
      function int foo();
        int y;
        y = 42;
        return y;
      endfunction
    endmodule
  )");
  EXPECT_THAT(captures["foo"], IsEmpty());
}

// Package-scope variables are global and should not be captured.
TEST_F(CaptureAnalysisTest, GlobalVariableNotCaptured) {
  auto captures = analyze(R"(
    package pkg;
      int g;
      function int foo();
        return g;
      endfunction
    endpackage
    module top;
      import pkg::*;
    endmodule
  )");
  EXPECT_THAT(captures["foo"], IsEmpty());
}

// Transitive capture: foo calls bar, bar captures x, so foo should too.
TEST_F(CaptureAnalysisTest, TransitiveCapture) {
  auto captures = analyze(R"(
    module top;
      int x;
      function void bar();
        x = 1;
      endfunction
      function void foo();
        bar();
      endfunction
    endmodule
  )");
  EXPECT_THAT(captures["bar"], ElementsAre("x"));
  EXPECT_THAT(captures["foo"], ElementsAre("x"));
}

// A variable defined by the caller should not propagate as a capture of the
// caller, even though the callee captures it.
TEST_F(CaptureAnalysisTest, TransitiveCaptureStopsAtDefiner) {
  auto captures = analyze(R"(
    module top;
      function void inner(int x);
      endfunction
      function void outer();
        int x;
        inner(x);
      endfunction
    endmodule
  )");
  EXPECT_THAT(captures["inner"], IsEmpty());
  EXPECT_THAT(captures["outer"], IsEmpty());
}

// A shadowing local in the caller does not prevent transitive propagation,
// because the callee captures a different symbol (the module-level one).
TEST_F(CaptureAnalysisTest, ShadowingDoesNotPreventCapture) {
  auto captures = analyze(R"(
    module top;
      int x;
      function void inner();
        x = 1;
      endfunction
      function void outer();
        int x;
        inner();
      endfunction
    endmodule
  )");
  EXPECT_THAT(captures["inner"], ElementsAre("x"));
  EXPECT_THAT(captures["outer"], ElementsAre("x"));
}

// Multiple variables captured by the same function.
TEST_F(CaptureAnalysisTest, MultipleCaptures) {
  auto captures = analyze(R"(
    module top;
      int a, b, c;
      function void foo();
        a = b + c;
      endfunction
    endmodule
  )");
  EXPECT_THAT(captures["foo"], ElementsAre("a", "b", "c"));
}

// Function arguments should not be captured.
TEST_F(CaptureAnalysisTest, ArgumentsNotCaptured) {
  auto captures = analyze(R"(
    module top;
      function int foo(int a, int b);
        return a + b;
      endfunction
    endmodule
  )");
  EXPECT_THAT(captures["foo"], IsEmpty());
}

// Recursive functions should not cause infinite loops.
TEST_F(CaptureAnalysisTest, RecursiveFunction) {
  auto captures = analyze(R"(
    module top;
      int x;
      function void foo();
        x = 1;
        foo();
      endfunction
    endmodule
  )");
  EXPECT_THAT(captures["foo"], ElementsAre("x"));
}

// Deep transitive capture chain: a -> b -> c, c captures x.
TEST_F(CaptureAnalysisTest, DeepTransitiveCapture) {
  auto captures = analyze(R"(
    module top;
      int x;
      function void c();
        x = 1;
      endfunction
      function void b();
        c();
      endfunction
      function void a();
        b();
      endfunction
    endmodule
  )");
  for (auto *name : {"a", "b", "c"})
    EXPECT_THAT(captures[name], ElementsAre("x")) << "function " << name;
}
