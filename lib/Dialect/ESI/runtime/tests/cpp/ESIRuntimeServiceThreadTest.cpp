//===- ESIRuntimeServiceThreadTest.cpp - Service thread tests -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "esi/Accelerator.h"
#include "esi/Context.h"
#include "esi/Logging.h"
#include "gtest/gtest.h"

#include <any>
#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace esi;

namespace {

/// Records the error messages logged to it.
struct RecordingLogger : public Logger {
  RecordingLogger() : Logger(false, false) {}
  void log(Level level, const std::string &, const std::string &msg,
           const std::map<std::string, std::any> *) override {
    std::lock_guard<std::mutex> lock(mutex);
    if (level == Level::Error)
      errors.push_back(msg);
  }
  std::vector<std::string> getErrors() {
    std::lock_guard<std::mutex> lock(mutex);
    return errors;
  }
  std::mutex mutex;
  std::vector<std::string> errors;
};

TEST(AcceleratorServiceThreadTest, TaskExceptionsAreLogged) {
  Context ctxt(std::make_unique<RecordingLogger>());
  auto ownedLogger = std::make_unique<RecordingLogger>();
  RecordingLogger &logger = *ownedLogger;
  std::atomic<int> throwingRuns = 0, otherRuns = 0;
  {
    AcceleratorServiceThread serviceThread(&ctxt);
    // The context's current logger is used, even if it was replaced.
    ctxt.setLogger(std::move(ownedLogger));
    serviceThread.addTask([&]() {
      if (throwingRuns++ == 0)
        throw std::runtime_error("task failure");
    });
    serviceThread.addTask([&]() { ++otherRuns; });

    // The thread carries on running both tasks.
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while ((throwingRuns < 2 || otherRuns < 2) &&
           std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_GE(throwingRuns, 2);
  EXPECT_GE(otherRuns, 2);
  std::vector<std::string> errors = logger.getErrors();
  ASSERT_EQ(errors.size(), 1u);
  EXPECT_NE(errors[0].find("task failure"), std::string::npos);
}

} // namespace
