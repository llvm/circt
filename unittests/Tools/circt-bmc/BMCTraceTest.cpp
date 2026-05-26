//===- BMCTraceTest.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Tools/circt-bmc/BMCTrace.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/raw_ostream.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <optional>

using namespace circt::bmc;
using testing::HasSubstr;

namespace {

static std::optional<llvm::APInt> evaluateWord(BMCTrace::Handle handle,
                                               unsigned width) {
  auto value = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(handle));
  return llvm::APInt(width, value);
}

TEST(BMCTraceTest, RecordsSignalsByStep) {
  BMCTrace trace("top");
  auto dataIn = trace.addSignal("data_in", 8);
  auto stateQ = trace.addSignal("state_q", 8);

  trace.record(
      0, dataIn,
      reinterpret_cast<BMCTrace::Handle>(static_cast<uintptr_t>(0xff)));
  trace.record(
      0, stateQ,
      reinterpret_cast<BMCTrace::Handle>(static_cast<uintptr_t>(0x00)));
  trace.record(
      1, dataIn,
      reinterpret_cast<BMCTrace::Handle>(static_cast<uintptr_t>(0x12)));
  trace.record(
      1, stateQ,
      reinterpret_cast<BMCTrace::Handle>(static_cast<uintptr_t>(0xff)));

  EXPECT_EQ(trace.getSignals().size(), 2u);
  EXPECT_EQ(trace.getNumSteps(), 2u);
  EXPECT_EQ(trace.lookup(0, dataIn),
            reinterpret_cast<BMCTrace::Handle>(static_cast<uintptr_t>(0xff)));
  EXPECT_EQ(trace.lookup(1, stateQ),
            reinterpret_cast<BMCTrace::Handle>(static_cast<uintptr_t>(0xff)));
}

TEST(BMCTraceTest, PrintsTextTrace) {
  BMCTrace trace("top");
  auto dataIn = trace.addSignal("data_in", 8);
  auto stateQ = trace.addSignal("state_q", 8);

  trace.record(
      0, dataIn,
      reinterpret_cast<BMCTrace::Handle>(static_cast<uintptr_t>(0xff)));
  trace.record(
      0, stateQ,
      reinterpret_cast<BMCTrace::Handle>(static_cast<uintptr_t>(0x00)));
  trace.record(
      1, dataIn,
      reinterpret_cast<BMCTrace::Handle>(static_cast<uintptr_t>(0x12)));
  trace.record(
      1, stateQ,
      reinterpret_cast<BMCTrace::Handle>(static_cast<uintptr_t>(0xff)));

  std::string output;
  llvm::raw_string_ostream os(output);
  ASSERT_TRUE(trace.printTextTrace(os, evaluateWord));
  os.flush();

  EXPECT_THAT(output, HasSubstr("counterexample for top:\n"));
  EXPECT_THAT(output,
              HasSubstr("cycle 0:\n  data_in = 0xff\n  state_q = 0x0\n"));
  EXPECT_THAT(output,
              HasSubstr("cycle 1:\n  data_in = 0x12\n  state_q = 0xff\n"));
}

TEST(BMCTraceTest, SupportsZeroWidthSignals) {
  BMCTrace trace("top");
  auto empty = trace.addSignal("empty", 0);

  trace.record(0, empty,
               reinterpret_cast<BMCTrace::Handle>(static_cast<uintptr_t>(0x0)));

  std::string output;
  llvm::raw_string_ostream os(output);
  ASSERT_TRUE(trace.printTextTrace(os, evaluateWord));
  os.flush();

  EXPECT_THAT(output, HasSubstr("cycle 0:\n  empty = 0x0\n"));
}

} // namespace
