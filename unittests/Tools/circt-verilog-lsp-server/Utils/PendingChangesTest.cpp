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
  bool waitFor(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(m);
    return cv.wait_for(lock, timeout, [&] { return called; });
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
  // See https://github.com/llvm/circt/issues/9292.
  GTEST_SKIP() << "flaky on macOS";

  PendingChangesMap pcm(/*maxThreads=*/2);
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

  ASSERT_TRUE(cap.waitFor(200ms));
  ASSERT_TRUE(cap.got != nullptr);
  EXPECT_EQ(cap.got->version, 1);
  ASSERT_EQ(cap.got->changes.size(), 1u);
}

/// Check that changes are applied once debounce delay window is passed
TEST(PendingChangesMapTest, FlushAfterQuietMinWindow) {
  // See https://github.com/llvm/circt/issues/9292.
  GTEST_SKIP() << "flaky on macOS";

  PendingChangesMap pcm(2);
  DebounceOptions opt;
  opt.disableDebounce = false;
  opt.debounceMinMs = 60; // small debounce
  opt.debounceMaxMs = 0;  // no cap

  auto p = makeChangeParams("/tmp/b.sv", 1);
  pcm.enqueueChange(p);

  CallbackCapture cap;
  auto start = std::chrono::steady_clock::now();
  pcm.debounceAndThen(p, opt, [&](std::unique_ptr<PendingChanges> r) {
    cap.set(std::move(r));
  });

  // Expect callback to arrive after ~debounceMinMs; allow headroom.
  ASSERT_TRUE(cap.waitFor(800ms));
  ASSERT_TRUE(cap.got != nullptr);
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start);
  // Should be at least the min window (allowing scheduling jitter).
  EXPECT_GE(elapsed.count(), static_cast<long long>(opt.debounceMinMs) - 5);
}

/// Check that outdated edits are not applied and the thread is discarded.
TEST(PendingChangesMapTest, ObsoleteWhenNewerEditsArrive) {
  // See https://github.com/llvm/circt/issues/9292.
  GTEST_SKIP() << "flaky on macOS";

  PendingChangesMap pcm(2);
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
  std::this_thread::sleep_for(30ms);
  auto p2 = makeChangeParams(key, 2);
  pcm.enqueueChange(p2);

  // The first scheduled check should now be obsolete -> nullptr.
  ASSERT_TRUE(first.waitFor(800ms));
  ASSERT_EQ(first.got, nullptr);

  // Now schedule another check; with no more edits, it should flush.
  CallbackCapture second;
  pcm.debounceAndThen(p2, opt, [&](std::unique_ptr<PendingChanges> r) {
    second.set(std::move(r));
  });

  ASSERT_TRUE(second.waitFor(800ms));
  ASSERT_TRUE(second.got != nullptr);
  // Both edits should be present (burst merged)
  EXPECT_EQ(second.got->version, 2);
  EXPECT_EQ(second.got->changes.size(), 2u);
}

TEST(PendingChangesMapTest, MaxCapForcesFlushDuringContinuousTyping) {
  // See https://github.com/llvm/circt/issues/9292.
  GTEST_SKIP() << "flaky on macOS";

  PendingChangesMap pcm(2);
  DebounceOptions opt;
  opt.disableDebounce = false;
  opt.debounceMinMs = 100; // quiet 100ms
  opt.debounceMaxMs = 500; // cap 500ms

  const auto *key = "d.sv";
  auto p = makeChangeParams(key, 1);
  pcm.enqueueChange(p);

  // 1) First check at ~100ms: will be obsolete (new edits arrived).
  CallbackCapture first;
  pcm.debounceAndThen(p, opt, [&](std::unique_ptr<PendingChanges> r) {
    first.set(std::move(r));
  });

  // Immediately enqueue a new edit to represent continuous typing.
  pcm.enqueueChange(makeChangeParams(key, 2));

  ASSERT_TRUE(first.waitFor(std::chrono::milliseconds(800)));
  EXPECT_EQ(first.got, nullptr); // expected obsolete

  // 2) After max-cap has passed, schedule another check -> must flush.
  std::this_thread::sleep_for(
      std::chrono::milliseconds(550)); // Total > 500ms cap
  CallbackCapture second;
  // Use the latest version for the schedule key; any params with same URI key
  // is fine
  auto p2 = makeChangeParams(key, /*version=*/999);
  pcm.debounceAndThen(p2, opt, [&](std::unique_ptr<PendingChanges> r) {
    second.set(std::move(r));
  });

  ASSERT_TRUE(second.waitFor(std::chrono::milliseconds(1500)));

  ASSERT_TRUE(second.got != nullptr);
  EXPECT_GE(second.got->version, 2);
  EXPECT_GE(second.got->changes.size(), 1u);
}

/// Check that no update is queued if the change map is empty.
TEST(PendingChangesMapTest, MissingKeyYieldsNullptr) {
  // See https://github.com/llvm/circt/issues/9292.
  GTEST_SKIP() << "flaky on macOS";

  PendingChangesMap pcm(2);
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

  ASSERT_TRUE(cap.waitFor(400ms));
  EXPECT_EQ(cap.got, nullptr);
}

TEST(PendingChangesMapTest, EraseByKeyAndUriAreIdempotent) {
  // See https://github.com/llvm/circt/issues/9292.
  GTEST_SKIP() << "flaky on macOS";

  PendingChangesMap pcm(/*maxThreads=*/2);

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
    ASSERT_TRUE(cap.waitFor(std::chrono::milliseconds(200)));
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
    ASSERT_TRUE(cap.waitFor(std::chrono::milliseconds(200)));
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
    ASSERT_TRUE(cap.waitFor(std::chrono::milliseconds(200)));
    ASSERT_NE(cap.got, nullptr);
    EXPECT_EQ(cap.got->version, 3);
    EXPECT_EQ(cap.got->changes.size(), 1u);
  }
}

TEST(PendingChangesMapTest, AbortClearsAll) {
  // See https://github.com/llvm/circt/issues/9292.
  GTEST_SKIP() << "flaky on macOS";

  PendingChangesMap pcm(2);

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

    ASSERT_TRUE(capA.waitFor(std::chrono::milliseconds(500)));
    ASSERT_TRUE(capB.waitFor(std::chrono::milliseconds(500)));

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
    ASSERT_TRUE(cap.waitFor(std::chrono::milliseconds(200)));
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
    ASSERT_TRUE(cap.waitFor(std::chrono::milliseconds(200)));
    ASSERT_NE(cap.got, nullptr);
    EXPECT_EQ(cap.got->version, 2);
    EXPECT_EQ(cap.got->changes.size(), 1u);
  }
}

} // namespace
