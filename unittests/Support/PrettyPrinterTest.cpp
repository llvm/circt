//===- PrettyPrinterTest.cpp - Pretty printer unit tests ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/PrettyPrinter.h"
#include "circt/Support/PrettyPrinterBuilder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace circt;
using namespace pretty;

namespace {

class FuncTest : public testing::Test {
protected:
  // Test inputs.
  SmallVector<Token> funcTokens;
  SmallVector<Token> nestedTokens;
  SmallVector<Token> indentNestedTokens;

  /// Scratch buffer used by print.
  SmallString<256> out;

  SmallVector<Token> argTokens;
  void buildArgs() {
    // Build argument list with comma + break between tokens.
    auto args = {"int a",
                 "int b",
                 "int a1",
                 "int b1",
                 "int a2",
                 "int b2",
                 "int a3",
                 "int b3",
                 "int a4",
                 "int b4",
                 "int a5",
                 "int b5",
                 "int a6",
                 "int b6",
                 "int a7",
                 "int b7",
                 "float xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"};

    llvm::interleave(
        args, [&](auto &arg) { argTokens.push_back(StringToken(arg)); },
        [&]() {
          argTokens.push_back(StringToken(","));
          argTokens.push_back(BreakToken());
        });
  }

  void SetUp() override {

    buildArgs();
    {
      // foooooo(ARGS)
      // With ARGS in an ibox.
      funcTokens.append({StringToken("foooooo"), StringToken("("),
                         BeginToken(0, Breaks::Inconsistent), BreakToken(0)});
      funcTokens.append(argTokens);
      funcTokens.append({BreakToken(0), EndToken(), StringToken(");"),
                         BreakToken(PrettyPrinter::kInfinity)});
    }
    {
      // baroo(AR..  barooga(ARGS) .. GS)
      // Nested function call, nested method wrapped in cbox(0) w/breaks.
      nestedTokens.append({StringToken("baroo"), StringToken("("),
                           BeginToken(0, Breaks::Inconsistent), BreakToken(0)});
      SmallVectorImpl<Token>::iterator argMiddle =
          argTokens.begin() + argTokens.size() / 2;
      nestedTokens.append(argTokens.begin(), argMiddle);

      nestedTokens.append({
          BeginToken(0, Breaks::Consistent),
          StringToken("barooga"),
          StringToken("("),
          BeginToken(0, Breaks::Inconsistent),
          BreakToken(0),
      });
      nestedTokens.append(argTokens);
      nestedTokens.append({BreakToken(0), EndToken(), StringToken("),"),
                           BreakToken(), EndToken(),
                           /* BreakToken(0), */});
      nestedTokens.append(argMiddle, argTokens.end());
      nestedTokens.append({BreakToken(0), EndToken(), StringToken(");"),
                           BreakToken(PrettyPrinter::kInfinity)});
    }
    {
      // wahoo(ARGS)
      // If wrap args, indent on next line
      indentNestedTokens.append({
          BeginToken(2, Breaks::Consistent),
          StringToken("wahoo"),
          StringToken("("),
          BreakToken(0),
          BeginToken(0, Breaks::Inconsistent),
      });

      SmallVectorImpl<Token>::iterator argMiddle =
          argTokens.begin() + argTokens.size() / 2;
      indentNestedTokens.append(argTokens.begin(), argMiddle);

      indentNestedTokens.append({
          BeginToken(0, Breaks::Consistent),
          StringToken("yahooooooo"),
          StringToken("("),
          BeginToken(0, Breaks::Inconsistent),
          BreakToken(0),
      });
      indentNestedTokens.append(argTokens);
      indentNestedTokens.append({
          BreakToken(0), EndToken(), StringToken("),"), BreakToken(),
          EndToken(), /* BreakToken(0), */
      });
      indentNestedTokens.append(argMiddle, argTokens.end());
      indentNestedTokens.append({EndToken(), BreakToken(0, -2),
                                 StringToken(");"), EndToken(),
                                 BreakToken(PrettyPrinter::kInfinity)});
    }
  }

  void print(SmallVectorImpl<Token> &tokens, size_t margin) {
    out = "\n";
    raw_svector_ostream os(out);
    PrettyPrinter pp(os, margin);
    pp.addTokens(tokens);
    pp.eof();
  }
};

TEST_F(FuncTest, Margin20) {
  auto constexpr margin = 20;
  {
    print(funcTokens, margin);
    EXPECT_EQ(out.str(), StringRef(R"""(
foooooo(int a,
        int b,
        int a1,
        int b1,
        int a2,
        int b2,
        int a3,
        int b3,
        int a4,
        int b4,
        int a5,
        int b5,
        int a6,
        int b6,
        int a7,
        int b7,
        float xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        );
)"""));
  }
  {
    print(nestedTokens, margin);
    EXPECT_EQ(out.str(), StringRef(R"""(
baroo(int a, int b,
      int a1,
      int b1,
      int a2,
      int b2,
      int a3,
      int b3,
      barooga(int a,
              int b,
              int a1,
              int b1,
              int a2,
              int b2,
              int a3,
              int b3,
              int a4,
              int b4,
              int a5,
              int b5,
              int a6,
              int b6,
              int a7,
              int b7,
              float xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
              ),
      int a4,
      int b4,
      int a5,
      int b5,
      int a6,
      int b6,
      int a7,
      int b7,
      float xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
      );
)"""));
  }
  {
    print(indentNestedTokens, margin);
    EXPECT_EQ(out.str(), StringRef(R"""(
wahoo(
  int a, int b,
  int a1, int b1,
  int a2, int b2,
  int a3, int b3,
  yahooooooo(int a,
             int b,
             int a1,
             int b1,
             int a2,
             int b2,
             int a3,
             int b3,
             int a4,
             int b4,
             int a5,
             int b5,
             int a6,
             int b6,
             int a7,
             int b7,
             float xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
             ),
  int a4, int b4,
  int a5, int b5,
  int a6, int b6,
  int a7, int b7,
  float xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
);
)"""));
  }
}

TEST_F(FuncTest, Margin50) {
  auto constexpr margin = 50;
  {
    print(funcTokens, margin);
    EXPECT_EQ(out.str(), StringRef(R"""(
foooooo(int a, int b, int a1, int b1, int a2,
        int b2, int a3, int b3, int a4, int b4,
        int a5, int b5, int a6, int b6, int a7,
        int b7,
        float xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        );
)"""));
  }
  {
    print(nestedTokens, margin);
    EXPECT_EQ(out.str(), StringRef(R"""(
baroo(int a, int b, int a1, int b1, int a2,
      int b2, int a3, int b3,
      barooga(int a, int b, int a1, int b1,
              int a2, int b2, int a3, int b3,
              int a4, int b4, int a5, int b5,
              int a6, int b6, int a7, int b7,
              float xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
              ),
      int a4, int b4, int a5, int b5, int a6,
      int b6, int a7, int b7,
      float xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
      );
)"""));
  }
  {
    print(indentNestedTokens, margin);
    EXPECT_EQ(out.str(), StringRef(R"""(
wahoo(
  int a, int b, int a1, int b1, int a2, int b2,
  int a3, int b3,
  yahooooooo(int a, int b, int a1, int b1, int a2,
             int b2, int a3, int b3, int a4,
             int b4, int a5, int b5, int a6,
             int b6, int a7, int b7,
             float xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
             ),
  int a4, int b4, int a5, int b5, int a6, int b6,
  int a7, int b7,
  float xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
);
)"""));
  }
}

TEST_F(FuncTest, Margin2048) {
  auto constexpr margin = 2048;
  {
    print(funcTokens, margin);
    EXPECT_EQ(out.str(), StringRef(R"""(
foooooo(int a, int b, int a1, int b1, int a2, int b2, int a3, int b3, int a4, int b4, int a5, int b5, int a6, int b6, int a7, int b7, float xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx);
)"""));
  }
  {
    print(nestedTokens, margin);
    EXPECT_EQ(out.str(), StringRef(R"""(
baroo(int a, int b, int a1, int b1, int a2, int b2, int a3, int b3, barooga(int a, int b, int a1, int b1, int a2, int b2, int a3, int b3, int a4, int b4, int a5, int b5, int a6, int b6, int a7, int b7, float xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx), int a4, int b4, int a5, int b5, int a6, int b6, int a7, int b7, float xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx);
)"""));
  }
  {
    print(indentNestedTokens, margin);
    EXPECT_EQ(out.str(), StringRef(R"""(
wahoo(int a, int b, int a1, int b1, int a2, int b2, int a3, int b3, yahooooooo(int a, int b, int a1, int b1, int a2, int b2, int a3, int b3, int a4, int b4, int a5, int b5, int a6, int b6, int a7, int b7, float xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx), int a4, int b4, int a5, int b5, int a6, int b6, int a7, int b7, float xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx);
)"""));
  }
}

TEST(CIRCTSupportTests, TrailingSpace) {
  SmallString<128> out;
  raw_svector_ostream os(out);

  PrettyPrinter pp(os, 20);
  SmallVector<Token> tokens = {BeginToken(2),
                               StringToken("test"),
                               BreakToken(),
                               StringToken("test"),
                               BreakToken(PrettyPrinter::kInfinity),
                               EndToken()};
  pp.addTokens(tokens);
  pp.eof();
  EXPECT_EQ(out.str(), StringRef("test test\n"));
}

TEST(CIRCTSupportTests, Builder) {
  SmallString<128> out;
  raw_svector_ostream os(out);

  PPBuilder b(os, 7);

  {
    auto ib = b.scopedIBox();
    b.literal("test");
    b.space();
    b.literal("test");
    b.space();
    b.literal("test");
  }
  EXPECT_EQ(out.str(), StringRef("test\ntest\ntest"));
}

TEST(CIRCTSupportTests, Stream) {
  SmallString<128> out;
  raw_svector_ostream os(out);

  PPStream ps(os, 20);
  {
    auto ib = ps.scopedIBox();
    ps << "test" << PP::space << "test" << PP::space << "test";
  }
  ps << PP::eof;
  EXPECT_EQ(out.str(), StringRef("test test test"));
}

TEST(CIRCTSupportTests, StreamQuoted) {
  SmallString<128> out;
  raw_svector_ostream os(out);

  PPStream ps(os, 20);
  out = "\n";
  {
    auto ib = ps.scopedIBox(2);
    ps << "test" << PP::space;
    ps.writeQuotedEscaped("quote\"me");
    ps << PP::space << "test";
  }
  ps << PP::newline << PP::eof;
  EXPECT_EQ(out.str(), StringRef(R"""(
test "quote\"me"
  test
)"""));
}

TEST(CIRCTSupportTests, Expr) {
  SmallString<128> out;
  raw_svector_ostream os(out);

  auto sumExpr = [](auto &ps) {
    ps << "(";
    {
      auto ib = ps.scopedIBox(0);
      auto vars = {"a", "b", "c", "d", "e", "f"};
      llvm::interleave(
          vars, [&](const char *each) { ps << each; },
          [&]() { ps << PP::space << "+" << PP::space; });
    }
    ps << ")";
  };

  auto test = [&](const char *id, auto margin) {
    PPStream ps(os, margin);
    out = "\n";
    {
      auto ib = ps.scopedIBox(2);
      {
        // TODO: let this wrap.
        ps << "assign" << PP::nbsp << id << PP::nbsp << "=";
      }
      ps << PP::space;
      auto ib3 = ps.scopedIBox(0);
      sumExpr(ps);
      ps << PP::space << "*" << PP::space;
      sumExpr(ps);
      ps << ";";
    }
    ps << PP::newline << PP::eof;
  };

  test("foo", 8);
  EXPECT_EQ(out.str(), StringRef(R"""(
assign foo =
  (a + b
   + c +
   d + e
   + f)
  *
  (a + b
   + c +
   d + e
   + f);
)"""));
  test("foo", 12);
  EXPECT_EQ(out.str(), StringRef(R"""(
assign foo =
  (a + b + c
   + d + e +
   f) *
  (a + b + c
   + d + e +
   f);
)"""));
  test("foo", 30);
  EXPECT_EQ(out.str(), StringRef(R"""(
assign foo =
  (a + b + c + d + e + f) *
  (a + b + c + d + e + f);
)"""));
  test("foo", 80);
  EXPECT_EQ(out.str(), StringRef(R"""(
assign foo = (a + b + c + d + e + f) * (a + b + c + d + e + f);
)"""));
}

} // end anonymous namespace
