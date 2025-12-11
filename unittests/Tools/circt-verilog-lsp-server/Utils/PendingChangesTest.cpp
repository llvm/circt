//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "llvm/Support/LSP/Protocol.h"

#include "PendingChanges.h"

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

using namespace circt::lsp;
using namespace std::chrono_literals;

namespace {

/// Manual, monotonic clock for tests. Time is advanced manually to avoid race
/// conditions in threads
struct ManualClock {
  SteadyClock::time_point t = SteadyClock::time_point{}; // start at epoch
  SteadyClock::time_point now() const { return t; }
  void advanceMs(std::chrono::milliseconds ms) { t += ms; }
};

static inline void advanceTestTime(ManualClock &clock,
                                   std::chrono::milliseconds ms) {
  clock.advanceMs(ms);
  std::this_thread::yield(); // let the worker run
}

/// Tiny helper to await a single callback result in tests.
struct CallbackCapture {
  std::mutex m;
  std::condition_variable cv;
  std::unique_ptr<PendingChanges> got; // nullptr => obsolete
  bool called = false;

  void set(std::unique_ptr<PendingChanges> r) {
    {
      std::scoped_lock<std::mutex> lock(m);
      got = std::move(r);
      called = true;
    }
    cv.notify_all();
  }

  // Wait up to timeout for a callback; returns true if called.
  bool waitFor() {
    std::unique_lock<std::mutex> lock(m);
    // Keep the timeout very small, as all time references are anyway driven
    // from the unit tests
    return cv.wait_for(lock, 1ms, [&] { return called; });
  }
};

// Absolute fake path -> file URI. No filesystem access; it's just a key.
static inline llvm::lsp::URIForFile makeMemURI(llvm::StringRef basename) {
  // Build a POSIX-like absolute path key; URIForFile only cares it’s absolute.
  std::string path = ("/test://" + basename.str());
  return llvm::cantFail(llvm::lsp::URIForFile::fromFile(path, "file"));
}

static inline llvm::lsp::DidChangeTextDocumentParams
makeChangeParams(llvm::StringRef memName, int64_t version,
                 llvm::StringRef text = "x") {
  llvm::lsp::DidChangeTextDocumentParams params;
  params.textDocument.uri = makeMemURI(memName); // e.g. "a.sv"
  params.textDocument.version = version;

  llvm::lsp::TextDocumentContentChangeEvent ev;
  ev.text = text.str();
  params.contentChanges = {ev};
  return params;
}

/// Check that changes are done immediately if debounce is disabled
TEST(PendingChangesMapTest, ImmediateFlushWhenDisabled) {
#if defined(__APPLE__)
  // See https://github.com/llvm/circt/issues/9292.
  //() << "flaky on macOS";
#endif

  ManualClock clock;
  PendingChangesMap pcm(/*maxThreads=*/2, [&] { return clock.now(); });
  DebounceOptions opt;
  opt.disableDebounce = true;
  opt.debounceMinMs = 500;
  opt.debounceMaxMs = 1000;

  auto p = makeChangeParams("/tmp/a.sv", 1);
  pcm.enqueueChange(p);

  CallbackCapture cap;
  pcm.debounceAndThen(p, opt, [&](std::unique_ptr<PendingChanges> r) {
    cap.set(std::move(r));
  });

  ASSERT_TRUE(cap.waitFor());
  ASSERT_TRUE(cap.got != nullptr);
  EXPECT_EQ(cap.got->version, 1);
  ASSERT_EQ(cap.got->changes.size(), 1u);
  pcm.abort();
}

/// Check that changes are applied once debounce delay window is passed
TEST(PendingChangesMapTest, FlushAfterQuietMinWindow) {
#if defined(__APPLE__)
  // See https://github.com/llvm/circt/issues/9292.
  //() << "flaky on macOS";
#endif

  ManualClock clock;
  PendingChangesMap pcm(/*maxThreads=*/2, [&] { return clock.now(); });

  DebounceOptions opt;
  opt.disableDebounce = false;
  opt.debounceMinMs = 60; // small debounce
  opt.debounceMaxMs = 0;  // no cap

  auto p = makeChangeParams("/tmp/b.sv", 1);
  pcm.enqueueChange(p);

  CallbackCapture cap;
  pcm.debounceAndThen(p, opt, [&](std::unique_ptr<PendingChanges> r) {
    cap.set(std::move(r));
  });

  // Let the worker's sleep elapse, and advance the manual clock accordingly.
  advanceTestTime(clock, 60ms);

  ASSERT_TRUE(cap.waitFor());
  ASSERT_TRUE(cap.got != nullptr);
  EXPECT_EQ(cap.got->version, 1);
  EXPECT_EQ(cap.got->changes.size(), 1u);

  // Signal all to all scheduled threads they are out of time.
  advanceTestTime(clock, 1000ms);
  pcm.abort();
}

/// Check that outdated edits are not applied and the thread is discarded.
TEST(PendingChangesMapTest, ObsoleteWhenNewerEditsArrive) {
#if defined(__APPLE__)
  // See https://github.com/llvm/circt/issues/9292.
  //() << "flaky on macOS";
#endif

  ManualClock clock;
  PendingChangesMap pcm(/*maxThreads=*/2, [&] { return clock.now(); });
  DebounceOptions opt;
  opt.disableDebounce = false;
  opt.debounceMinMs = 80; // first task sleeps 80ms
  opt.debounceMaxMs = 0;

  const auto *key = "/tmp/c.sv";
  auto p1 = makeChangeParams(key, 1);
  pcm.enqueueChange(p1);

  CallbackCapture first;
  pcm.debounceAndThen(p1, opt, [&](std::unique_ptr<PendingChanges> r) {
    first.set(std::move(r));
  });

  // While the first sleeper is waiting, enqueue a new edit to reset lastChange.
  advanceTestTime(clock, 30ms);
  auto p2 = makeChangeParams(key, 2);
  pcm.enqueueChange(p2);

  // Advance to the end of the quiet window so the first check runs.
  advanceTestTime(clock, 50ms);

  // The first scheduled check should now be obsolete -> nullptr.
  ASSERT_TRUE(first.waitFor());
  ASSERT_EQ(first.got, nullptr);

  // Now schedule another check; with no more edits, it should flush.
  CallbackCapture second;
  pcm.debounceAndThen(p2, opt, [&](std::unique_ptr<PendingChanges> r) {
    second.set(std::move(r));
  });

  advanceTestTime(clock, 80ms);

  ASSERT_TRUE(second.waitFor());
  ASSERT_TRUE(second.got != nullptr);
  // Both edits should be present (burst merged)
  EXPECT_EQ(second.got->version, 2);
  EXPECT_EQ(second.got->changes.size(), 2u);

  // Signal all to all scheduled threads they are out of time.
  advanceTestTime(clock, 1000ms);
  pcm.abort();
}

/// Check that no update is queued if the change map is empty.
TEST(PendingChangesMapTest, MissingKeyYieldsNullptr) {
#if defined(__APPLE__)
  // See https://github.com/llvm/circt/issues/9292.
  //() << "flaky on macOS";
#endif

  ManualClock clock;
  PendingChangesMap pcm(/*maxThreads=*/2, [&] { return clock.now(); });
  DebounceOptions opt;
  opt.disableDebounce = false;
  opt.debounceMinMs = 30;
  opt.debounceMaxMs = 0;

  auto p = makeChangeParams("/tmp/e.sv", 1);

  // Intentionally do NOT enqueueChange(). Debouncer should find no entry.
  CallbackCapture cap;
  pcm.debounceAndThen(p, opt, [&](std::unique_ptr<PendingChanges> r) {
    cap.set(std::move(r));
  });

  advanceTestTime(clock, 30ms);

  ASSERT_TRUE(cap.waitFor());
  EXPECT_EQ(cap.got, nullptr);

  // Signal all to all scheduled threads they are out of time.
  advanceTestTime(clock, 1000ms);
}

TEST(PendingChangesMapTest, EraseByKeyAndUriAreIdempotent) {
#if defined(__APPLE__)
  // See https://github.com/llvm/circt/issues/9292.
  //() << "flaky on macOS";
#endif

  ManualClock clock;
  PendingChangesMap pcm(/*maxThreads=*/2, [&] { return clock.now(); });

  DebounceOptions opt;
  opt.disableDebounce = true; // immediate path exercises the keyed lookup

  // Enqueue for "a.sv"
  auto a = makeChangeParams("a.sv", /*version=*/1);
  pcm.enqueueChange(a);

  // Erase by key (StringRef)
  pcm.erase(a.textDocument.uri.file());
  {
    CallbackCapture cap;
    pcm.debounceAndThen(a, opt, [&](std::unique_ptr<PendingChanges> r) {
      cap.set(std::move(r));
    });
    ASSERT_TRUE(cap.waitFor());
    // Nothing pending after erase -> nullptr
    EXPECT_EQ(cap.got, nullptr);
  }

  // Re-enqueue and erase by URI overload
  auto b = makeChangeParams("b.sv", /*version=*/2);
  pcm.enqueueChange(b);
  pcm.erase(b.textDocument.uri); // URI overload

  {
    CallbackCapture cap;
    pcm.debounceAndThen(b, opt, [&](std::unique_ptr<PendingChanges> r) {
      cap.set(std::move(r));
    });
    ASSERT_TRUE(cap.waitFor());
    EXPECT_EQ(cap.got, nullptr);
  }

  // Erasing a missing key should be a no-op (idempotent)
  pcm.erase(makeMemURI("nonexistent.sv"));
  // And should not affect other keys: add "c.sv", don't erase it
  auto c = makeChangeParams("c.sv", /*version=*/3);
  pcm.enqueueChange(c);
  {
    CallbackCapture cap;
    pcm.debounceAndThen(c, opt, [&](std::unique_ptr<PendingChanges> r) {
      cap.set(std::move(r));
    });
    ASSERT_TRUE(cap.waitFor());
    ASSERT_NE(cap.got, nullptr);
    EXPECT_EQ(cap.got->version, 3);
    EXPECT_EQ(cap.got->changes.size(), 1u);
  }

  // Signal all to all scheduled threads they are out of time.
  advanceTestTime(clock, 1000ms);
  pcm.abort();
}

TEST(PendingChangesMapTest, AbortClearsAll) {
#if defined(__APPLE__)
  // See https://github.com/llvm/circt/issues/9292.
  //() << "flaky on macOS";
#endif

  ManualClock clock;
  PendingChangesMap pcm(/*maxThreads=*/2, [&] { return clock.now(); });

  // Debounced path to also cover the async branch
  DebounceOptions opt;
  opt.disableDebounce = false;
  opt.debounceMinMs = 50; // small delay
  opt.debounceMaxMs = 0;  // no cap

  auto a = makeChangeParams("x1.sv", 1);
  auto b = makeChangeParams("x2.sv", 1);
  pcm.enqueueChange(a);
  pcm.enqueueChange(b);

  // Abort immediately — this should erase all keys.
  pcm.abort();

  // After abort: scheduling a debounce for old URIs should find no pending
  // entry.
  {
    CallbackCapture capA, capB;

    pcm.debounceAndThen(a, opt, [&](std::unique_ptr<PendingChanges> r) {
      capA.set(std::move(r));
    });
    pcm.debounceAndThen(b, opt, [&](std::unique_ptr<PendingChanges> r) {
      capB.set(std::move(r));
    });

    advanceTestTime(clock, 50ms);

    ASSERT_TRUE(capA.waitFor());
    ASSERT_TRUE(capB.waitFor());

    EXPECT_EQ(capA.got, nullptr);
    EXPECT_EQ(capB.got, nullptr);
  }

  // Also test immediate path after abort (disableDebounce=true)
  opt.disableDebounce = true;
  auto c = makeChangeParams("x3.sv", 2);
  {
    CallbackCapture cap;
    pcm.debounceAndThen(c, opt, [&](std::unique_ptr<PendingChanges> r) {
      cap.set(std::move(r));
    });
    ASSERT_TRUE(cap.waitFor());
    // Still nullptr because we didn't enqueue after abort.
    EXPECT_EQ(cap.got, nullptr);
  }

  // If we enqueue new changes after abort, things work as usual.
  pcm.enqueueChange(c);
  {
    CallbackCapture cap;
    pcm.debounceAndThen(c, opt, [&](std::unique_ptr<PendingChanges> r) {
      cap.set(std::move(r));
    });
    ASSERT_TRUE(cap.waitFor());
    ASSERT_NE(cap.got, nullptr);
    EXPECT_EQ(cap.got->version, 2);
    EXPECT_EQ(cap.got->changes.size(), 1u);
  }

  // Signal all to all scheduled threads they are out of time.
  advanceTestTime(clock, 1000ms);
}

} // namespace
