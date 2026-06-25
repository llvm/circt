#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Sim/SimDialect.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Dialect/Sim/SimUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;

namespace {

const char *irSimpleChain = R"MLIR(
module {
  hw.module @test(in %clk : !seq.clock, in %cond : i1) {
    %lit = sim.fmt.literal "hello"
    %fmt = sim.fmt.concat (%lit)
    sim.print %fmt on %clk if %cond
    hw.output
  }
}
)MLIR";

const char *irSharedInput = R"MLIR(
module {
  hw.module @test(in %clk : !seq.clock, in %cond : i1) {
    %lit = sim.fmt.literal "shared"
    %fmt0 = sim.fmt.concat (%lit, %lit)
    %fmt1 = sim.fmt.concat (%lit)
    sim.print %fmt0 on %clk if %cond
    sim.print %fmt1 on %clk if %cond
    hw.output
  }
}
)MLIR";

const char *irConditionFromConstant = R"MLIR(
module {
  hw.module @test(in %clk : !seq.clock) {
    %cond = hw.constant true
    %lit = sim.fmt.literal "hello"
    %fmt = sim.fmt.concat (%lit)
    sim.print %fmt on %clk if %cond
    hw.output
  }
}
)MLIR";

const char *irDuplicateProducerUse = R"MLIR(
module {
  hw.module @test(in %clk : !seq.clock, in %cond : i1) {
    %lit = sim.fmt.literal "dup"
    %fmt = sim.fmt.concat (%lit, %lit)
    sim.print %fmt on %clk if %cond
    hw.output
  }
}
)MLIR";

const char *irProcPrintGetFile = R"MLIR(
module {
  hw.module @test(in %trigger : i1) {
    hw.triggered posedge %trigger {
      %msg = sim.fmt.literal "hello"
      %prefix = sim.fmt.literal "out_"
      %suffix = sim.fmt.literal ".log"
      %fname = sim.fmt.concat (%prefix, %suffix)
      %file = sim.get_file %fname
      sim.proc.print %msg to %file
    }
    hw.output
  }
}
)MLIR";

class SimCascadeDeleteTest : public ::testing::Test {
protected:
  MLIRContext context;

  void SetUp() override {
    context.loadDialect<hw::HWDialect>();
    context.loadDialect<seq::SeqDialect>();
    context.loadDialect<sim::SimDialect>();
  }

  OwningOpRef<ModuleOp> parseTestModule(const char *source) {
    auto module = parseSourceString<ModuleOp>(source, &context);
    EXPECT_TRUE(module);
    return module;
  }

  static SmallVector<sim::PrintFormattedOp> collectPrints(ModuleOp module) {
    SmallVector<sim::PrintFormattedOp> prints;
    module.walk([&](sim::PrintFormattedOp op) { prints.push_back(op); });
    return prints;
  }

  static sim::PrintFormattedOp findSinglePrint(ModuleOp module) {
    auto prints = collectPrints(module);
    EXPECT_EQ(prints.size(), 1u);
    return prints.front();
  }

  static sim::PrintFormattedProcOp findSingleProcPrint(ModuleOp module) {
    sim::PrintFormattedProcOp result;
    module.walk([&](sim::PrintFormattedProcOp op) {
      EXPECT_FALSE(result);
      result = op;
    });
    EXPECT_TRUE(result);
    return result;
  }

  template <typename OpTy>
  static unsigned countOps(Operation *root) {
    unsigned count = 0;
    root->walk([&](OpTy) { ++count; });
    return count;
  }

  void erasePrint(sim::PrintFormattedOp printOp) {
    IRRewriter rewriter(&context);
    sim::cascadeErasePrint(printOp, rewriter);
  }

  void erasePrint(sim::PrintFormattedProcOp printOp) {
    IRRewriter rewriter(&context);
    sim::cascadeErasePrint(printOp, rewriter);
  }
};

TEST_F(SimCascadeDeleteTest, SimpleChain) {
  auto module = parseTestModule(irSimpleChain);
  ASSERT_TRUE(module);

  auto printOp = findSinglePrint(module.get());
  ASSERT_TRUE(printOp);

  erasePrint(printOp);

  ASSERT_EQ(countOps<sim::PrintFormattedOp>(module.get()), 0);
  ASSERT_EQ(countOps<sim::FormatStringConcatOp>(module.get()), 0);
  ASSERT_EQ(countOps<sim::FormatLiteralOp>(module.get()), 0);
}

TEST_F(SimCascadeDeleteTest, SharedInputStaysAlive) {
  auto module = parseTestModule(irSharedInput);
  ASSERT_TRUE(module);

  auto prints = collectPrints(module.get());
  ASSERT_EQ(prints.size(), 2u);

  erasePrint(prints.front());

  ASSERT_EQ(countOps<sim::PrintFormattedOp>(module.get()), 1);
  ASSERT_EQ(countOps<sim::FormatStringConcatOp>(module.get()), 1);
  ASSERT_EQ(countOps<sim::FormatLiteralOp>(module.get()), 1);
}

TEST_F(SimCascadeDeleteTest, KeepsNonCascadableConditionProducer) {
  auto module = parseTestModule(irConditionFromConstant);
  ASSERT_TRUE(module);

  auto printOp = findSinglePrint(module.get());
  ASSERT_TRUE(printOp);

  erasePrint(printOp);

  ASSERT_EQ(countOps<sim::PrintFormattedOp>(module.get()), 0);
  ASSERT_EQ(countOps<sim::FormatStringConcatOp>(module.get()), 0);
  ASSERT_EQ(countOps<sim::FormatLiteralOp>(module.get()), 0);
  ASSERT_EQ(countOps<hw::ConstantOp>(module.get()), 1);
}

TEST_F(SimCascadeDeleteTest, DuplicateProducerUseBySingleConsumer) {
  auto module = parseTestModule(irDuplicateProducerUse);
  ASSERT_TRUE(module);

  auto printOp = findSinglePrint(module.get());
  ASSERT_TRUE(printOp);

  erasePrint(printOp);

  ASSERT_EQ(countOps<sim::PrintFormattedOp>(module.get()), 0);
  ASSERT_EQ(countOps<sim::FormatStringConcatOp>(module.get()), 0);
  ASSERT_EQ(countOps<sim::FormatLiteralOp>(module.get()), 0);
}

TEST_F(SimCascadeDeleteTest, ProcPrintAlsoCascadesGetFileChain) {
  auto module = parseTestModule(irProcPrintGetFile);
  ASSERT_TRUE(module);

  auto printOp = findSingleProcPrint(module.get());
  ASSERT_TRUE(printOp);

  erasePrint(printOp);

  ASSERT_EQ(countOps<sim::PrintFormattedProcOp>(module.get()), 0);
  ASSERT_EQ(countOps<sim::GetFileOp>(module.get()), 0);
  ASSERT_EQ(countOps<sim::FormatStringConcatOp>(module.get()), 0);
  ASSERT_EQ(countOps<sim::FormatLiteralOp>(module.get()), 0);
}

} // namespace
