//===- PrettyPrinterTest.cpp - Pretty printer unit tests ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/PrettyPrinter.h"
#include "circt/Support/PrettyPrinterHelpers.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace circt;
using namespace pretty;

namespace {

using CallbackType = Token::CallbackInfo::CallbackTy;

class FuncTest : public testing::Test {
protected:
  // Test inputs.
  SmallVector<Token> funcTokens;
  SmallVector<Token> nestedTokens;
  SmallVector<Token> indentNestedTokens;
  formatted_raw_ostream *fStream = nullptr;
  llvm::BumpPtrAllocator allocator;
  TokenStringAndCallbackSaver callbacks;

  /// Scratch buffer used by print.
  SmallString<256> out;
  SmallString<8> locationOut1, locationOut2, locationOut3;

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

  CallbackType *getCallback(SmallString<8> &str) {
    // Note, the `fStream` is not yet initialized, it is null. Store the address
    // of fStream, such that when it is initialized, the callback can access it.
    formatted_raw_ostream **formattedStream = &fStream;
    auto *locationStr = &str;
    // Note, the callback captures the local variables by value. The value is an
    // address, which should be safe to access even out of scope of this
    // function. Also, the not-null check cannot be moved out of the lambda,
    // since it is null when the lambda is created.
    return callbacks.save([formattedStream, locationStr]() {
      if (*formattedStream)
        locationStr->append(("line <" + Twine((**formattedStream).getLine()) +
                             "," + Twine((**formattedStream).getColumn()) +
                             ">,")
                                .str());
    });
  }
  void SetUp() override {

    buildArgs();
    {
      // foooooo(ARGS)
      // With ARGS in an ibox.
      funcTokens.append({StringToken("foooooo"), StringToken("("),
                         BeginToken(0, Breaks::Inconsistent), BreakToken(0)});
      auto *callbackPtr = getCallback(locationOut1);

      funcTokens.append({CallbackToken(callbackPtr)});
      funcTokens.append(argTokens);
      funcTokens.append({CallbackToken(callbackPtr)});
      funcTokens.append({BreakToken(0), EndToken(), StringToken(");"),
                         BreakToken(PrettyPrinter::kInfinity)});
      funcTokens.append({CallbackToken(callbackPtr)});
    }
    {
      auto *callbackPtr = getCallback(locationOut2);
      // baroo(AR..  barooga(ARGS) .. GS)
      // Nested function call, nested method wrapped in cbox(0) w/breaks.
      nestedTokens.append({StringToken("baroo"), StringToken("("),
                           BeginToken(0, Breaks::Inconsistent), BreakToken(0)});
      nestedTokens.append({CallbackToken(callbackPtr)});
      SmallVectorImpl<Token>::iterator argMiddle =
          argTokens.begin() + argTokens.size() / 2;
      nestedTokens.append(argTokens.begin(), argMiddle);
      nestedTokens.append({CallbackToken(callbackPtr)});

      nestedTokens.append({
          BeginToken(0, Breaks::Consistent),
          StringToken("barooga"),
          StringToken("("),
          CallbackToken(callbackPtr),
          BeginToken(0, Breaks::Inconsistent),
          BreakToken(0),
      });
      nestedTokens.append({CallbackToken(callbackPtr)});
      nestedTokens.append(argTokens);
      nestedTokens.append({BreakToken(0), CallbackToken(callbackPtr),
                           EndToken(), StringToken("),"), BreakToken(),
                           EndToken(),
                           /* BreakToken(0), */});
      nestedTokens.append(argMiddle, argTokens.end());
      nestedTokens.append({BreakToken(0), EndToken(), StringToken(");"),
                           BreakToken(PrettyPrinter::kInfinity)});
    }
    {
      auto *callbackPtr = getCallback(locationOut3);
      // wahoo(ARGS)
      // If wrap args, indent on next line
      indentNestedTokens.append(
          {CallbackToken(callbackPtr), BeginToken(2, Breaks::Consistent),
           /*No change in location*/ CallbackToken(callbackPtr),
           StringToken("wahoo"), CallbackToken(callbackPtr), StringToken("("),
           BreakToken(0), CallbackToken(callbackPtr),
           BeginToken(0, Breaks::Inconsistent),
           /*No change in location*/ CallbackToken(callbackPtr)});

      SmallVectorImpl<Token>::iterator argMiddle =
          argTokens.begin() + argTokens.size() / 2;
      indentNestedTokens.append(argTokens.begin(), argMiddle);

      indentNestedTokens.append({
          CallbackToken(callbackPtr),
          BeginToken(0, Breaks::Consistent),
          CallbackToken(callbackPtr),
          StringToken("yahooooooo"),
          StringToken("("),
          BeginToken(0, Breaks::Inconsistent),
          CallbackToken(callbackPtr),
          BreakToken(0),
      });
      indentNestedTokens.append(argTokens);
      indentNestedTokens.append({
          CallbackToken(callbackPtr), BreakToken(0), EndToken(),
          CallbackToken(callbackPtr), StringToken("),"), BreakToken(),
          CallbackToken(callbackPtr), EndToken(),
          CallbackToken(callbackPtr), /* BreakToken(0), */
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
    formatted_raw_ostream formattedStream(os);
    fStream = &formattedStream;
    locationOut1.clear();
    PrettyPrinter pp(formattedStream, margin);
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

    EXPECT_EQ(locationOut1.str(),
              StringRef("line <0,8>,line <16,55>,line <18,0>,"));
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
    EXPECT_EQ(
        locationOut2.str(),
        StringRef(
            "line <0,6>,line <7,6>,line <7,14>,line <7,14>,line <24,14>,"));
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
    EXPECT_EQ(locationOut3.str(),
              StringRef("line <0,0>,line <0,0>,line <0,5>,line <1,2>,line "
                        "<1,2>,line <5,2>,line <5,2>,line <5,13>,line "
                        "<21,60>,line <22,13>,line <23,2>,line <23,2>,"));
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

TEST(PrettyPrinterTest, TrailingSpace) {
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

TEST(PrettyPrinterTest, Builder) {
  SmallString<128> out;
  raw_svector_ostream os(out);

  PrettyPrinter pp(os, 7);
  TokenBuilder<> b(pp);
  {
    b.ibox();
    b.literal("test");
    b.space();
    b.literal("test");
    b.space();
    b.literal("test");
    b.end();
  }
  EXPECT_EQ(out.str(), StringRef("test\ntest\ntest"));
}

TEST(PrettyPrinterTest, Stream) {
  SmallString<128> out;
  raw_svector_ostream os(out);

  PrettyPrinter pp(os, 20);
  TokenStringSaver saver;
  TokenStream<> ps(pp, saver);
  ps.scopedBox(PP::ibox0, [&]() {
    ps << "test" << PP::space << "test" << PP::space << "test";
  });
  ps << PP::eof;
  EXPECT_EQ(out.str(), StringRef("test test test"));
}

TEST(PrettyPrinterTest, StreamQuoted) {
  SmallString<128> out;
  raw_svector_ostream os(out);

  PrettyPrinter pp(os, 20);
  TokenStringSaver saver;
  TokenStream<> ps(pp, saver);
  ps.writeQuotedEscaped("quote\"me", false, "'", "'");
  EXPECT_EQ(out.str(), "'quote\\\"me'");
}

TEST(PrettyPrinterTest, LargeStream) {
  PrettyPrinter pp(llvm::nulls(), 20);
  TokenStringSaver saver;
  TokenStream<> ps(pp, saver);

  for (uint32_t i = 1U << 20; i; --i)
    ps << "testingtesting" << PP::space;
}

TEST(PrettyPrinterTest, LargeStreamScan) {
  PrettyPrinter pp(llvm::nulls(), 20);
  TokenStringSaver saver;
  TokenStream<> ps(pp, saver);

  // This triggers an assert w/o "rebase" support.
  ps << PP::cbox0;
  for (uint32_t i = 1U << 20; i; --i)
    ps << BeginToken(0, Breaks::Never) << "testingtesting" << PP::end
       << PP::newline;
  ps << PP::end;
}

TEST(PrettyPrinterTest, IndentStyle) {
  SmallString<128> out;
  raw_svector_ostream os(out);

  auto test = [&](auto margin, auto style) {
    PrettyPrinter pp(os, margin);
    TokenStringSaver saver;
    TokenStream<> ps(pp, saver);
    out = "\n";
    {
      ps << "start" << PP::nbsp;
      ps << BeginToken(2, Breaks::Inconsistent, style);
      ps << "open";
      ps << PP::space << "test" << PP::space << "test";
      ps << PP::space << "test" << PP::space << "test";
      ps << PP::end;
    }
    ps << PP::newline << PP::eof;
  };
  test(10, IndentStyle::Block);
  EXPECT_EQ(out.str(), StringRef(R"""(
start open
  test
  test
  test
  test
)"""));
  test(15, IndentStyle::Block);
  EXPECT_EQ(out.str(), StringRef(R"""(
start open test
  test test
  test
)"""));
  test(10, IndentStyle::Visual);
  EXPECT_EQ(out.str(), StringRef(R"""(
start open
        test
        test
        test
        test
)"""));
  test(15, IndentStyle::Visual);
  EXPECT_EQ(out.str(), StringRef(R"""(
start open test
        test
        test
        test
)"""));
}

TEST(PrettyPrinterTest, FuncArgsBlock) {
  SmallString<128> out;
  raw_svector_ostream os(out);

  auto test = [&](auto margin) {
    PrettyPrinter pp(os, margin);
    TokenStringSaver saver;
    TokenStream<> ps(pp, saver);
    out = "\n";
    {
      ps << "foo(";
      ps << BeginToken(2, Breaks::Consistent, IndentStyle::Block);
      ps << PP::zerobreak;
      ps << BeginToken(0);
      auto args = {"a", "b", "c", "d", "e"};
      llvm::interleave(
          args, [&](auto &arg) { ps << arg; },
          [&]() { ps << "," << PP::space; });
      ps << PP::end;
      ps << BreakToken(0, -2);
      ps << ");";
      ps << PP::end;
    }
    ps << PP::newline << PP::eof;
  };
  test(10);
  EXPECT_EQ(out.str(), StringRef(R"""(
foo(
  a, b, c,
  d, e
);
)"""));
  test(15);
  EXPECT_EQ(out.str(), StringRef(R"""(
foo(
  a, b, c, d, e
);
)"""));
  test(20);
  EXPECT_EQ(out.str(), StringRef(R"""(
foo(a, b, c, d, e);
)"""));
}

TEST(PrettyPrinterTest, FuncArgsVisual) {
  SmallString<128> out;
  raw_svector_ostream os(out);

  auto test = [&](auto margin) {
    PrettyPrinter pp(os, margin);
    TokenStringSaver saver;
    TokenStream<> ps(pp, saver);
    out = "\n";
    {
      ps << "foo(";
      ps << BeginToken(0, Breaks::Inconsistent, IndentStyle::Visual);
      auto args = {"a", "b", "c", "d", "e"};
      llvm::interleave(
          args, [&](auto &arg) { ps << arg; },
          [&]() { ps << "," << PP::space; });
      ps << ");";
      ps << PP::end;
    }
    ps << PP::newline << PP::eof;
  };
  test(10);
  EXPECT_EQ(out.str(), StringRef(R"""(
foo(a, b,
    c, d,
    e);
)"""));
  test(15);
  EXPECT_EQ(out.str(), StringRef(R"""(
foo(a, b, c, d,
    e);
)"""));
  test(20);
  EXPECT_EQ(out.str(), StringRef(R"""(
foo(a, b, c, d, e);
)"""));
}

TEST(PrettyPrinterTest, Expr) {
  SmallString<128> out;
  raw_svector_ostream os(out);
  formatted_raw_ostream formattedStream(os);
  SmallString<8> locationOut;
  auto callback = [&formattedStream, &locationOut]() {
    locationOut.append(("line <" + Twine(formattedStream.getLine()) + "," +
                        Twine(formattedStream.getColumn()) + ">,")
                           .str());
  };
  TokenStringAndCallbackSaver callbacks;
  CallbackType *callbackPtr = callbacks.save(callback);

  auto sumExpr = [&callbackPtr](auto &ps) {
    ps << CallbackToken(callbackPtr) << "(";
    {
      ps << PP::ibox0;
      auto vars = {"a", "b", "c", "d", "e", "f"};
      llvm::interleave(
          vars, [&](const char *each) { ps << each; },
          [&]() {
            ps << PP::space << "+" << CallbackToken(callbackPtr) << PP::space;
          });
      ps << PP::end << CallbackToken(callbackPtr);
    }
    ps << ")" << CallbackToken(callbackPtr);
  };

  auto test = [&](const char *id, auto margin) {
    PrettyPrinter pp(formattedStream, margin);
    TokenStringSaver saver;
    TokenStream<> ps(pp, saver);
    out = "\n";
    ps.scopedBox(PP::ibox2, [&]() {
      ps << CallbackToken(callbackPtr);
      ps << "assign" << PP::nbsp << id << PP::nbsp << "=";
      ps << PP::space;
      ps.scopedBox(PP::ibox0, [&]() {
        sumExpr(ps);
        ps << PP::space << "*" << PP::space;
        sumExpr(ps);
        ps << ";";
      });
    });
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
  EXPECT_EQ(
      locationOut.str(),
      StringRef(
          "line <0,0>,line <1,2>,line <1,6>,line <2,4>,line <2,8>,line "
          "<3,6>,line <4,4>,line <4,6>,line <4,7>,line <6,2>,line <6,6>,line "
          "<7,4>,line <7,8>,line <8,6>,line <9,4>,line <9,6>,line <9,7>,"));
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

TEST(PrettyPrinterTest, InitWithBaseAndCurrentIndent) {
  SmallString<128> out;
  raw_svector_ostream os(out);

  SmallVector<Token> tokens = {
      StringToken("xxxxxxxxxxxxxxx"), BreakToken(),
      StringToken("yyyyyyyyyyyyyyy"), BreakToken(),
      StringToken("zzzzzzzzzzzzzzz"), BreakToken(PrettyPrinter::kInfinity)};

  auto test = [&](auto base, auto current, bool populate = true,
                  bool group = true) {
    out = "\n";
    for (int i = 0; populate && i < current; ++i)
      os << ">";
    PrettyPrinter pp(os, 35, base, current);
    if (group)
      pp.add(BeginToken(2));
    pp.addTokens(tokens);
    if (group)
      pp.add(EndToken());
    pp.eof();
  };
  test(0, 0);
  EXPECT_EQ(out.str(), StringRef(R"""(
xxxxxxxxxxxxxxx yyyyyyyyyyyyyyy
  zzzzzzzzzzzzzzz
)"""));

  // Base = current.
  test(2, 2);
  EXPECT_EQ(out.str(), StringRef(R"""(
>>xxxxxxxxxxxxxxx yyyyyyyyyyyyyyy
    zzzzzzzzzzzzzzz
)"""));
  test(2, 2, false); // wrong 'current' column.
  EXPECT_EQ(out.str(), StringRef(R"""(
xxxxxxxxxxxxxxx yyyyyyyyyyyyyyy
    zzzzzzzzzzzzzzz
)"""));

  // Base < current.
  test(2, 6);
  EXPECT_EQ(out.str(), StringRef(R"""(
>>>>>>xxxxxxxxxxxxxxx
        yyyyyyyyyyyyyyy
        zzzzzzzzzzzzzzz
)"""));
  // Check behavior w/o grouping, respect 'base'.
  test(2, 6, true, false);
  EXPECT_EQ(out.str(), StringRef(R"""(
>>>>>>xxxxxxxxxxxxxxx
  yyyyyyyyyyyyyyy zzzzzzzzzzzzzzz
)"""));

  // Base > current.  PP should add whitespace.
  test(6, 3);
  EXPECT_EQ(out.str(), StringRef(R"""(
>>>   xxxxxxxxxxxxxxx
        yyyyyyyyyyyyyyy
        zzzzzzzzzzzzzzz
)"""));
  // Check behavior w/o any group (default group).
  test(6, 3, true, false);
  EXPECT_EQ(out.str(), StringRef(R"""(
>>>   xxxxxxxxxxxxxxx
      yyyyyyyyyyyyyyy
      zzzzzzzzzzzzzzz
)"""));
}

TEST(PrettyPrinterTest, NeverBreak) {
  SmallString<128> out;
  raw_svector_ostream os(out);

  auto test = [&](auto neverbreak) {
    out = "\n";
    PrettyPrinter pp(os, 8);
    pp.add(BeginToken(2));
    pp.add(StringToken("test"));
    pp.add(BreakToken(1, 0, neverbreak));
    pp.add(StringToken("test"));
    pp.add(EndToken());
    pp.add(BreakToken());
    pp.add(StringToken("test"));
    pp.add(BreakToken(PrettyPrinter::kInfinity));
    pp.eof();
  };

  test(false);
  EXPECT_EQ(out.str(), StringRef(R"""(
test
  test
test
)"""));

  test(true);
  EXPECT_EQ(out.str(), StringRef(R"""(
test test
test
)"""));
}

TEST(PrettyPrinterTest, NeverBreakGroup) {
  SmallString<128> out;
  raw_svector_ostream os(out);
  formatted_raw_ostream formattedStream(os);
  SmallString<8> locationOut;
  TokenStringAndCallbackSaver callbacks;
  // Mostly checking location after break tokens.
  CallbackType *callbackPtr =
      callbacks.save([&formattedStream, &locationOut]() {
        locationOut.append(("line <" + Twine(formattedStream.getLine()) + "," +
                            Twine(formattedStream.getColumn()) + ">,")
                               .str());
      });

  auto test = [&](Breaks breaks1, Breaks breaks2) {
    out = "\n";
    PrettyPrinter pp(formattedStream, 8);
    pp.add(CallbackToken(callbackPtr));
    pp.add(BeginToken(2, breaks1));
    pp.add(StringToken("test"));
    pp.add(CallbackToken(callbackPtr));
    pp.add(BreakToken());
    pp.add(CallbackToken(callbackPtr));
    pp.add(StringToken("test"));
    {
      pp.add(BeginToken(2, breaks2));
      pp.add(CallbackToken(callbackPtr));
      pp.add(BreakToken());
      pp.add(CallbackToken(callbackPtr));
      pp.add(StringToken("test"));
      pp.add(BreakToken());
      pp.add(CallbackToken(callbackPtr));
      pp.add(StringToken("test"));
      pp.add(EndToken());
      pp.add(CallbackToken(callbackPtr));
    }
    pp.add(BreakToken());
    pp.add(StringToken("test"));
    pp.add(EndToken());
    pp.add(BreakToken(PrettyPrinter::kInfinity));
    pp.eof();
  };

  test(Breaks::Inconsistent, Breaks::Inconsistent);
  EXPECT_EQ(out.str(), StringRef(R"""(
test
  test
        test
        test
  test
)"""));

  EXPECT_EQ(locationOut.str(),
            StringRef("line <0,0>,line <0,4>,line <1,2>,line <1,6>,line "
                      "<2,8>,line <3,8>,line <3,12>,"));
  test(Breaks::Inconsistent, Breaks::Never);
  EXPECT_EQ(out.str(), StringRef(R"""(
test
  test test test
  test
)"""));

  test(Breaks::Never, Breaks::Inconsistent);
  EXPECT_EQ(out.str(), StringRef(R"""(
test test test test test
)"""));

  test(Breaks::Never, Breaks::Never);
  EXPECT_EQ(out.str(), StringRef(R"""(
test test test test test
)"""));
}

TEST(PrettyPrinterTest, MaxStartingIndent) {
  SmallString<128> out;
  raw_svector_ostream os(out);
  formatted_raw_ostream formattedStream(os);
  SmallString<8> locationOut;

  TokenStringAndCallbackSaver callbacks;
  CallbackType *callbackPtr =
      callbacks.save([&formattedStream, &locationOut]() {
        locationOut.append(("line <" + Twine(formattedStream.getLine()) + "," +
                            Twine(formattedStream.getColumn()) + ">,")
                               .str());
      });

  auto test = [&](PrettyPrinter &pp) {
    out = "\n";
    pp.add(CallbackToken(callbackPtr));
    pp.add(BeginToken(2));
    pp.add(StringToken("test"));
    pp.add(CallbackToken(callbackPtr));
    pp.add(BreakToken());
    pp.add(BeginToken(2));
    pp.add(StringToken("test"));
    pp.add(BreakToken());
    pp.add(BeginToken(2));
    pp.add(CallbackToken(callbackPtr));
    pp.add(StringToken("test"));
    pp.add(CallbackToken(callbackPtr));
    pp.add(BreakToken());
    pp.add(BeginToken(2));
    pp.add(StringToken("test"));
    pp.add(CallbackToken(callbackPtr));
    pp.add(BreakToken());
    pp.add(StringToken("test"));
    pp.add(CallbackToken(callbackPtr));
    pp.add(EndToken());
    pp.add(EndToken());
    pp.add(EndToken());
    pp.add(EndToken());
    pp.add(BreakToken(PrettyPrinter::kInfinity));
  };
  auto testDefault = [&]() {
    PrettyPrinter pp(formattedStream, 4);
    test(pp);
  };
  auto testValue = [&](auto maxStartingIndent) {
    PrettyPrinter pp(formattedStream, 4, 0, 0, maxStartingIndent);
    test(pp);
  };

  // Limit max starting position to margin (=4).
  testValue(4);
  EXPECT_EQ(out.str(), StringRef(R"""(
test
  test
    test
    test
    test
)"""));
  EXPECT_EQ(locationOut.str(),
            StringRef("line <0,0>,line <0,4>,line <2,4>,line <2,8>,line "
                      "<3,8>,line <4,8>,"));
  locationOut.clear();
  // Limit max starting position to one past margin,
  // neither margin nor where the indent wants to place it.
  testValue(5);
  EXPECT_EQ(out.str(), StringRef(R"""(
test
  test
    test
     test
     test
)"""));

  // Continued line number from last print.
  EXPECT_EQ(locationOut.str(),
            StringRef("line <5,0>,line <5,4>,line <7,4>,line <7,8>,line "
                      "<8,9>,line <9,9>,"));
  // Check large limit allows repeated indent past margin.
  testValue(100);
  EXPECT_EQ(out.str(), StringRef(R"""(
test
  test
    test
      test
        test
)"""));

  // Check that default value does not (easily) limit indent start.
  testDefault();
  EXPECT_EQ(out.str(), StringRef(R"""(
test
  test
    test
      test
        test
)"""));
}

class TokenStreamCompareTest : public testing::Test {
protected:
  SmallString<128> out, compare;
  raw_svector_ostream ppOS = raw_svector_ostream(out);
  raw_svector_ostream os = raw_svector_ostream(compare);

  TokenStringAndCallbackSaver callbacks;
  TokenStringSaver saver;
  CallbackType *callbackPtr = nullptr;

  template <typename Callable>
  void testStreams(Callable &&test,
                   std::optional<StringRef> data = std::nullopt,
                   unsigned margin = 10) {
    formatted_raw_ostream formattedStream(ppOS);
    SmallString<8> locationOut;

    callbackPtr = callbacks.save([&formattedStream, &locationOut]() {
      locationOut.append(("line <" + Twine(formattedStream.getLine()) + "," +
                          Twine(formattedStream.getColumn()) + ">,")
                             .str());
    });
    out.clear();
    compare.clear();
    PrettyPrinter pp(formattedStream, margin);
    TokenStream<> ps(pp, saver);

    std::invoke(test, ps, os);

    if (data) {
      EXPECT_EQ(compare.str(), *data);
      EXPECT_EQ(out.str(), *data);
    }
    EXPECT_EQ(out.str(), compare.str());
  }

  template <typename Callable>
  void testStreamsSame(Callable &&test,
                       std::optional<StringRef> data = std::nullopt,
                       unsigned margin = 10) {
    testStreams(
        [&](auto &ps, auto &os) {
          test(ps);
          test(os);
        },
        data, margin);
  }
};

TEST_F(TokenStreamCompareTest, Strings) {
  testStreamsSame([&](auto &os) { os << "testing"; });
  testStreamsSame([&](auto &os) { os << StringRef("foo"); });
  testStreamsSame([&](auto &os) { os << StringRef("foobar", 3); });
  testStreamsSame([&](auto &os) { os << std::string("test"); });
  testStreamsSame([&](auto &os) {
    SmallString<10> ss;
    ss = "test";
    os << ss;
  });
  // (Twine not directly supported, use Twine::str())
}

TEST_F(TokenStreamCompareTest, NBSPs) {
  for (auto i : {0, 1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 511, 512, 513, 2048})
    testStreams([&](auto &ps, auto &os) {
      ps.nbsp(i);
      // This just checks there is no change in output stream.
      ps << CallbackToken(callbackPtr);
      os.indent(i);
    });
}

TEST_F(TokenStreamCompareTest, Numeric) {
  auto test = [&](auto x, std::optional<StringRef> data = std::nullopt) {
    testStreams(
        [&](auto &ps, auto &os) {
          ps.addAsString(x);
          os << x;
        },
        data);
  };

  // The exact output isn't quite as important as ensuring we are a suitable
  // replacement for llvm::raw_ostream.
  test(0);
  test(-123);
  test(4321);
  test(int{0});
  test(char{0}, StringRef("\0", 1));
  test(0L);
  test(0U);
  test(~0U);
  test(INT32_MIN);
  test(UINT64_MAX);

  test(0.);
  test(0.f);
  test(1.5);
  test(1.2);
  test(1. / 3.);
}

} // end anonymous namespace
