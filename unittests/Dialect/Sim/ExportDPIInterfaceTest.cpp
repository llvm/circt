//===- ExportDPIInterfaceTest.cpp - DPI ABI tests ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/ExportDPIInterface.h"
#include "circt/Conversion/ExportVerilog.h"
#include "circt/Conversion/SimToSV.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/Moore/MooreDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Sim/SimDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;

namespace {

class ExportDPIInterfaceTest : public testing::Test {
protected:
  ExportDPIInterfaceTest() {
    context.loadDialect<sim::SimDialect, hw::HWDialect, moore::MooreDialect,
                        sv::SVDialect, LLVM::LLVMDialect>();
  }

  void expectSchema(llvm::StringRef ir, llvm::StringRef expectedJSON,
                    llvm::StringRef expectedSV = {}) {
    auto module = parseSourceString<ModuleOp>(ir, &context);
    ASSERT_TRUE(module);
    std::string output;
    llvm::raw_string_ostream stream(output);
    ASSERT_TRUE(succeeded(sim::exportDPIInterface(*module, stream)));

    auto actual = llvm::json::parse(output);
    ASSERT_TRUE(bool(actual)) << llvm::toString(actual.takeError());
    auto expected = llvm::json::parse(expectedJSON);
    ASSERT_TRUE(bool(expected)) << llvm::toString(expected.takeError());
    EXPECT_EQ(*actual, *expected);

    if (expectedSV.empty())
      return;
    OwningOpRef<ModuleOp> lowered(cast<ModuleOp>((*module)->clone()));
    PassManager pm(&context);
    pm.addPass(createLowerSimToSVPass());
    ASSERT_TRUE(succeeded(pm.run(*lowered)));
    (*lowered)->setAttr("circt.loweringOptions",
                        StringAttr::get(&context, "disallowPortDeclSharing"));
    std::string verilog;
    llvm::raw_string_ostream verilogStream(verilog);
    ASSERT_TRUE(succeeded(exportVerilog(*lowered, verilogStream)));
    auto compact = [](StringRef text) {
      std::string result;
      for (char c : text)
        if (!llvm::isSpace(c))
          result += c;
      return result;
    };
    SmallVector<StringRef> lines;
    expectedSV.split(lines, '\n');
    auto compactVerilog = compact(verilog);
    for (auto line : lines) {
      if (line.trim().empty())
        continue;
      EXPECT_NE(compactVerilog.find(compact(line)), std::string::npos)
          << "Missing SV declaration: " << line << "\n"
          << verilog;
    }
  }

  void expectFailure(llvm::StringRef ir, llvm::StringRef expectedDiagnostic) {
    auto module = parseSourceString<ModuleOp>(ir, &context);
    ASSERT_TRUE(module);
    std::string diagnostic;
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diag) {
      llvm::raw_string_ostream stream(diagnostic);
      diag.print(stream);
      return success();
    });

    std::string output = "existing output";
    llvm::raw_string_ostream stream(output);
    EXPECT_TRUE(failed(sim::exportDPIInterface(*module, stream)));
    EXPECT_EQ(output, "existing output");
    EXPECT_NE(diagnostic.find(expectedDiagnostic.str()), std::string::npos)
        << diagnostic;
  }

  MLIRContext context;
};

TEST_F(ExportDPIInterfaceTest, NamesDirectionsAndSignedness) {
  expectSchema(R"mlir(
    module {
      sim.func.dpi @step(in %cycle: i32, out drive: ui8,
                        inout %state: si16, return status: si32)
          attributes {verilogName = "dpi_step"}
      hw.module @Top() attributes {emit.fragments = [@step_dpi_import_fragument]} {
        hw.output
      }
    }
  )mlir",
               R"json({
    "dpi_functions": [{
      "function": "dpi_step",
      "arguments": [
        {"name": "cycle", "direction": "in", "width": 32,
         "signed": true},
        {"name": "drive", "direction": "out", "width": 8,
         "signed": true},
        {"name": "state", "direction": "inout", "width": 16,
         "signed": true},
        {"name": "status", "direction": "return", "width": 32,
         "signed": true}
      ]
    }]
  })json",
               R"sv(
    import "DPI-C" context function int dpi_step(
    input int cycle,
    output byte drive,
    inout shortint state
  )sv");
}

TEST_F(ExportDPIInterfaceTest, IntegerTypesMatchSV) {
  for (unsigned width : {1, 7, 8, 16, 32, 64, 65, 1024}) {
    for (const char *prefix : {"i", "si", "ui"}) {
      std::string type = std::string(prefix) + std::to_string(width);
      SCOPED_TRACE(type);
      std::string ir = "module { sim.func.dpi @probe(in %value: " + type +
                       ", return result: i64) hw.module @Top() attributes "
                       "{emit.fragments = [@probe_dpi_import_fragument]} "
                       "{ hw.output } }";
      bool native = width == 8 || width == 16 || width == 32 || width == 64;
      std::string expected =
          "{\"dpi_functions\": [{\"function\": \"probe\", \"arguments\": ["
          "{\"name\": \"value\", \"direction\": \"in\", \"width\": " +
          std::to_string(width) +
          ", \"signed\": " + (native ? "true" : "false") +
          "}, {\"name\": \"result\", \"direction\": \"return\", \"width\": 64, "
          "\"signed\": true}]}]}";
      std::string svType = "bit";
      switch (width) {
      case 8:
        svType = "byte";
        break;
      case 16:
        svType = "shortint";
        break;
      case 32:
        svType = "int";
        break;
      case 64:
        svType = "longint";
        break;
      default:
        if (width != 1)
          svType += " [" + std::to_string(width - 1) + ":0]";
      }
      expectSchema(ir, expected,
                   "import \"DPI-C\" context function longint probe(\ninput " +
                       svType + " value\n);");
    }
  }
}

TEST_F(ExportDPIInterfaceTest, WidthsAndFunctionOrder) {
  expectSchema(R"mlir(
    module {
      sim.func.dpi @tick(in %edge: i1)
      sim.func.dpi @read(out value: i7, in %byte: i8, in %half: i16,
                        in %word: i32, in %wide: i63, return status: i64)
      sim.func.dpi @finish()
    }
  )mlir",
               R"json({
    "dpi_functions": [
      {"function": "tick", "arguments": [
        {"name": "edge", "direction": "in", "width": 1,
         "signed": false}
      ]},
      {"function": "read", "arguments": [
        {"name": "value", "direction": "out", "width": 7,
         "signed": false},
        {"name": "byte", "direction": "in", "width": 8,
         "signed": true},
        {"name": "half", "direction": "in", "width": 16,
         "signed": true},
        {"name": "word", "direction": "in", "width": 32,
         "signed": true},
        {"name": "wide", "direction": "in", "width": 63,
         "signed": false},
        {"name": "status", "direction": "return", "width": 64,
         "signed": true}
      ]},
      {"function": "finish", "arguments": []}
    ]
  })json");
}

TEST_F(ExportDPIInterfaceTest, DoesNotModifyIR) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module {
      sim.func.dpi @step(in %cycle: i32, in %data: i1024,
                        return result: i64)
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  std::string before;
  llvm::raw_string_ostream beforeStream(before);
  module->print(beforeStream);

  std::string output;
  llvm::raw_string_ostream stream(output);
  ASSERT_TRUE(succeeded(sim::exportDPIInterface(*module, stream)));
  std::string after;
  llvm::raw_string_ostream afterStream(after);
  module->print(afterStream);
  EXPECT_EQ(before, after);

  std::string repeatedOutput;
  llvm::raw_string_ostream repeatedStream(repeatedOutput);
  ASSERT_TRUE(succeeded(sim::exportDPIInterface(*module, repeatedStream)));
  EXPECT_EQ(output, repeatedOutput);
}

TEST_F(ExportDPIInterfaceTest, RejectsMissingFunctions) {
  expectFailure("module {}", "no sim.func.dpi operations to export");
}

TEST_F(ExportDPIInterfaceTest, ExportsWideIntegers) {
  expectSchema(R"mlir(
    module {
      sim.func.dpi @transfer(in %address: i65, in %data: i1024,
                            out response: si1024)
    }
  )mlir",
               R"json({
    "dpi_functions": [{
      "function": "transfer",
      "arguments": [
        {"name": "address", "direction": "in", "width": 65,
         "signed": false},
        {"name": "data", "direction": "in", "width": 1024,
         "signed": false},
        {"name": "response", "direction": "out", "width": 1024,
         "signed": false}
      ]
    }]
  })json");
}

TEST_F(ExportDPIInterfaceTest,
       RejectsTypesUnsupportedBySVWithoutPartialOutput) {
  for (const char *type :
       {"f32", "f64", "!hw.string", "!sim.dstring", "!moore.i32",
        "!moore.l1024", "!moore.f32", "!moore.f64", "!moore.string",
        "!moore.chandle", "!moore.time", "!moore.array<2 x l8>",
        "!moore.ustruct<{data: i32}>"}) {
    SCOPED_TRACE(type);
    expectFailure((llvm::Twine("module { sim.func.dpi @valid(in %cycle: i32) "
                               "sim.func.dpi @unsupported(in %value: ") +
                   type + ") }")
                      .str(),
                  "unsupported DPI schema argument \"value\"");
  }
}

TEST_F(ExportDPIInterfaceTest, RejectsZeroWidthWithoutPartialOutput) {
  expectFailure(R"mlir(
    module {
      sim.func.dpi @valid(in %cycle: i32)
      sim.func.dpi @step(in %value: i0)
    }
  )mlir",
                "unsupported DPI schema argument \"value\"");
}

TEST_F(ExportDPIInterfaceTest, RejectsInvalidAndDuplicateFunctionNames) {
  expectFailure(R"mlir(module {
    sim.func.dpi @"not-a-c-name"()
  })mlir",
                "is not a valid C/SV identifier");
  expectFailure(R"mlir(module {
    sim.func.dpi @first() attributes {verilogName = "shared"}
    sim.func.dpi @second() attributes {verilogName = "shared"}
  })mlir",
                "duplicate DPI function name");
  expectFailure(R"mlir(module {
    sim.func.dpi @first() attributes {verilogName = "module"}
  })mlir",
                "is not a valid C/SV identifier");
}

TEST_F(ExportDPIInterfaceTest, RejectsRefArguments) {
  expectFailure(R"mlir(
    module {
      sim.func.dpi @valid(in %cycle: i32)
      sim.func.dpi @invalid(ref %buffer: !llvm.ptr)
    }
  )mlir",
                "IEEE 1800 DPI does not allow ref arguments");
}

TEST_F(ExportDPIInterfaceTest, RejectsAmbiguousPointers) {
  expectFailure("module { sim.func.dpi @pointer(in %value: !llvm.ptr) }",
                "unsupported DPI schema argument \"value\"");
}

TEST_F(ExportDPIInterfaceTest, RejectsNonDPITypes) {
  for (const char *type : {"f16", "bf16", "f80", "index", "!moore.event",
                           "!moore.queue<i8, 4>", "!moore.uunion<{a: i8}>"}) {
    SCOPED_TRACE(type);
    expectFailure((llvm::Twine("module { sim.func.dpi @invalid(in %value: ") +
                   type + ") }")
                      .str(),
                  "unsupported DPI schema argument \"value\"");
  }
}

TEST_F(ExportDPIInterfaceTest, RejectsInvalidReturnTypes) {
  for (const char *type :
       {"i7", "i1024", "!moore.l32", "!moore.time", "!hw.array<2xi8>",
        "!sv.open_uarray<i32>", "!hw.struct<data: i32>", "!hw.enum<A, B>"}) {
    SCOPED_TRACE(type);
    expectFailure(
        (llvm::Twine("module { sim.func.dpi @invalid(return result: ") + type +
         ") }")
            .str(),
        "unsupported DPI schema argument \"result\"");
  }
}

TEST_F(ExportDPIInterfaceTest, RejectsUnsupportedNestedMembers) {
  expectFailure(R"mlir(
    module {
      sim.func.dpi @nested(in %value: !moore.ustruct<{data: queue<i8, 4>}>)
    }
  )mlir",
                "unsupported DPI schema argument \"value\"");
}

} // namespace
