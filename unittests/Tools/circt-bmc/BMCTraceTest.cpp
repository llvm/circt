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

TEST(BMCTraceTest, RuntimeCallbackRegistersAndRecordsSignals) {
  BMCTrace trace("top");
  setActiveBMCTrace(&trace);

  auto data0 = reinterpret_cast<BMCTrace::Handle>(static_cast<uintptr_t>(0x12));
  auto state0 =
      reinterpret_cast<BMCTrace::Handle>(static_cast<uintptr_t>(0x34));
  auto data1 = reinterpret_cast<BMCTrace::Handle>(static_cast<uintptr_t>(0x56));
  circt_bmc_record_trace(0, "data_in", 8, data0);
  circt_bmc_record_trace(0, "state_q", 8, state0);
  circt_bmc_record_trace(1, "data_in", 8, data1);
  setActiveBMCTrace(nullptr);

  ASSERT_EQ(trace.getSignals().size(), 2u);
  EXPECT_EQ(trace.getSignals()[0].name, "data_in");
  EXPECT_EQ(trace.getSignals()[0].width, 8u);
  EXPECT_EQ(trace.getSignals()[1].name, "state_q");
  EXPECT_EQ(trace.lookup(0, 0), data0);
  EXPECT_EQ(trace.lookup(0, 1), state0);
  EXPECT_EQ(trace.lookup(1, 0), data1);
}

} // namespace
