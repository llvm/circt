//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LSPServer.h"
#include "VerilogServerImpl/VerilogServer.h"
#include "circt/Tools/circt-verilog-lsp-server/CirctVerilogLspServerMain.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/LSP/Protocol.h"
#include "llvm/Support/LSP/Transport.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <thread>

#define DEBUG_TYPE "circt-verilog-lsp-server"

using namespace llvm;
using namespace llvm::lsp;

class Debouncer {
public:
  explicit Debouncer(
      std::chrono::steady_clock::duration delay,
      std::optional<std::chrono::steady_clock::duration> maxWait = std::nullopt)
      : delay(delay), maxWait(maxWait), worker(&Debouncer::run, this) {}

  ~Debouncer() {
    {
      std::lock_guard<std::mutex> lk(mutex);
      stop = true;
      ++version;
    }
    cv.notify_all();
    if (worker.joinable())
      worker.join();
  }

  /// Schedule fn to run after delay. Replaces any pending callback.
  void schedule(std::function<void()> fn) {
    std::lock_guard<std::mutex> lk(mutex);
    callback = std::move(fn);
    lastSchedule = std::chrono::steady_clock::now();
    if (!firstSchedule)
      firstSchedule = *lastSchedule;
    ++version;
    cv.notify_all();
  }

  /// Force the pending callback to run immediately.
  void flush() {
    {
      std::lock_guard<std::mutex> lk(mutex);
      if (!callback)
        return;
      lastSchedule = std::chrono::steady_clock::now() - delay; // backdate
      ++version;
    }
    cv.notify_all();
  }

  /// Cancel the pending callback, if any.
  void cancel() {
    std::lock_guard<std::mutex> lk(mutex);
    callback.reset();
    firstSchedule.reset();
    ++version;
    cv.notify_all();
  }

private:
  void run() {
    std::unique_lock<std::mutex> lk(mutex);
    while (!stop) {
      if (!callback) {
        cv.wait(lk, [&] { return stop || callback.has_value(); });
        if (stop)
          break;
      }

      assert(lastSchedule.has_value() && "Debouncer invariant violated");

      auto v = version.load(std::memory_order_acquire);
      auto now = std::chrono::steady_clock::now();
      auto nextTime = *lastSchedule + delay;

      if (maxWait && firstSchedule) {
        auto maxTime = *firstSchedule + *maxWait;
        if (maxTime < nextTime)
          nextTime = maxTime;
      }

      if (now < nextTime) {
        cv.wait_until(lk, nextTime, [&] {
          return stop || version.load(std::memory_order_acquire) != v ||
                 !callback.has_value();
        });
        continue;
      }

      auto fn = std::move(*callback);
      callback.reset();
      firstSchedule.reset();
      lk.unlock();
      std::thread(std::move(fn))
          .detach(); // fn might take a long time, so offload and re-arm timer.
      lk.lock();
    }
  }

  const std::chrono::steady_clock::duration delay;
  const std::optional<std::chrono::steady_clock::duration> maxWait;

  std::mutex mutex;
  std::condition_variable cv;
  std::optional<std::chrono::steady_clock::time_point> firstSchedule;
  std::optional<std::chrono::steady_clock::time_point> lastSchedule;
  std::optional<std::function<void()>> callback;
  std::atomic<uint64_t> version{0};
  bool stop = false;

  std::thread worker;
};

struct ChangeBuffer {
  mutable std::mutex mu;
  std::vector<llvm::lsp::TextDocumentContentChangeEvent>
      pending;                // buffered incremental changes (in order)
  int64_t pendingVersion = 0; // last seen version for these changes

  // Add one batch (from one didChange). Assumes versions are non-decreasing.
  void add(llvm::ArrayRef<llvm::lsp::TextDocumentContentChangeEvent> changes,
           int64_t version) {
    std::lock_guard<std::mutex> lk(mu);
    // Sometimes a change is the full replacement of a file; discard previous
    // changes in such a case.
    if (version < pendingVersion)
      return; // ignore stale batch
    bool fullReplace = llvm::any_of(
        changes, [](const llvm::lsp::TextDocumentContentChangeEvent &c) {
          return !c.range.has_value();
        });
    if (fullReplace)
      pending.clear();
    pending.insert(pending.end(), changes.begin(), changes.end());
    pendingVersion = version;
  }

  // Drain all buffered changes atomically.
  std::vector<llvm::lsp::TextDocumentContentChangeEvent>
  drain(int64_t &versionOut) {
    std::lock_guard<std::mutex> lk(mu);
    std::vector<llvm::lsp::TextDocumentContentChangeEvent> out;
    out.swap(pending);
    versionOut = pendingVersion;
    return out;
  }

  // Snapshot-only (no drain)
  bool empty() const {
    std::lock_guard<std::mutex> lk(mu);
    return pending.empty();
  }
};

struct FileBucket {
  // Full constructor: configure debounce
  FileBucket(bool useDb, uint32_t minMs, uint32_t maxMs)
      : useDebounce(useDb),
        deb(std::chrono::milliseconds(minMs),
            maxMs ? std::optional{std::chrono::milliseconds(maxMs)}
                  : std::nullopt) {}

  // Disabled debounce constructor
  FileBucket()
      : useDebounce(false), deb(std::chrono::milliseconds(0), std::nullopt) {}

  ChangeBuffer changeBuf;        // Collects buffer updates to apply batched
  std::atomic<bool> alive{true}; // Closed state
  std::atomic<bool> building{false};
  std::atomic<bool> rerun{false};

  bool useDebounce; // Whether to debounce updates
  Debouncer deb;    // Debounce didChange to single trailing rebuild
};

//===----------------------------------------------------------------------===//
// LSPServer
//===----------------------------------------------------------------------===//

namespace {
struct LSPServer {
  LSPServer(const circt::lsp::LSPServerOptions &options,
            circt::lsp::VerilogServer &server, JSONTransport &transport)
      : server(server), transport(transport), options(options) {}

  //===--------------------------------------------------------------------===//
  // Initialization
  //===--------------------------------------------------------------------===//

  void onInitialize(const InitializeParams &params,
                    Callback<json::Value> reply);
  void onInitialized(const InitializedParams &params);
  void onShutdown(const NoParams &params, Callback<std::nullptr_t> reply);

  //===--------------------------------------------------------------------===//
  // Document Change
  //===--------------------------------------------------------------------===//

  void onDocumentDidOpen(const DidOpenTextDocumentParams &params);
  void onDocumentDidClose(const DidCloseTextDocumentParams &params);
  void onDocumentDidChange(const DidChangeTextDocumentParams &params);
  void onDocumentDidChangeDebounce(const DidChangeTextDocumentParams &params);

  //===--------------------------------------------------------------------===//
  // Definitions and References
  //===--------------------------------------------------------------------===//

  void onGoToDefinition(const TextDocumentPositionParams &params,
                        Callback<std::vector<Location>> reply);
  void onReference(const ReferenceParams &params,
                   Callback<std::vector<Location>> reply);

  //===--------------------------------------------------------------------===//
  // Fields
  //===--------------------------------------------------------------------===//

  circt::lsp::VerilogServer &server;
  JSONTransport &transport;

  /// An outgoing notification used to send diagnostics to the client when they
  /// are ready to be processed.
  OutgoingNotification<PublishDiagnosticsParams> publishDiagnostics;

  /// Used to indicate that the 'shutdown' request was received from the
  /// Language Server client.
  std::atomic<bool> shutdownRequestReceived{false};

private:
  /// The options for this LSP Server
  const circt::lsp::LSPServerOptions &options;

  /// A mutex to serialize access to publishing diagnostics
  std::mutex diagMu;
  /// A thread-safe version of `publishDiagnostics`
  void sendDiagnostics(const PublishDiagnosticsParams &p);

  //===--------------------------------------------------------------------===//
  // FileBucket Accessors & Setters
  //===--------------------------------------------------------------------===//

  mutable std::mutex bucketsMu;
  llvm::StringMap<std::shared_ptr<FileBucket>>
      buckets; // shared_ptr owns lifetime

  std::shared_ptr<FileBucket> getOrCreateBucket(llvm::StringRef key);
  std::shared_ptr<FileBucket> findBucket(llvm::StringRef key) const;
  void eraseBucket(llvm::StringRef key);
  void cancelAllBuckets();
};
} // namespace

void LSPServer::sendDiagnostics(const PublishDiagnosticsParams &p) {
  std::lock_guard<std::mutex> lk(diagMu);
  publishDiagnostics(p); // serialize the write
}

//===--------------------------------------------------------------------===//
// FileBucket Accessors & Setters
//===--------------------------------------------------------------------===//

std::shared_ptr<FileBucket> LSPServer::getOrCreateBucket(llvm::StringRef key) {
  std::lock_guard<std::mutex> lk(bucketsMu);
  auto it = buckets.find(key);
  if (it != buckets.end())
    return it->second;

  std::shared_ptr<FileBucket> fb;
  if (options.disableDebounce)
    fb = std::make_shared<FileBucket>();
  else
    fb = std::make_shared<FileBucket>(
        options.disableDebounce, options.debounceMinMs, options.debounceMaxMs);
  buckets.try_emplace(key, fb);
  return fb;
}

std::shared_ptr<FileBucket> LSPServer::findBucket(llvm::StringRef key) const {
  std::lock_guard<std::mutex> lk(bucketsMu);
  auto it = buckets.find(key);
  return (it == buckets.end()) ? nullptr : it->second;
}

void LSPServer::eraseBucket(llvm::StringRef key) {
  std::shared_ptr<FileBucket> fb;
  {
    std::lock_guard<std::mutex> lk(bucketsMu);
    auto it = buckets.find(key);
    if (it == buckets.end())
      return;
    fb = it->second; // keep a ref while we cancel
    buckets.erase(it);
  }
  fb->alive.store(false, std::memory_order_release);
  fb->deb.cancel(); // cancel outside the map lock
}

void LSPServer::cancelAllBuckets() {
  std::vector<std::shared_ptr<FileBucket>> snapshot;
  {
    std::lock_guard<std::mutex> lk(bucketsMu);
    snapshot.reserve(buckets.size());
    for (auto &kv : buckets)
      snapshot.push_back(kv.second);
    buckets.clear();
  }
  for (auto &fb : snapshot)
    fb->deb.cancel();
}

//===----------------------------------------------------------------------===//
// Initialization
//===----------------------------------------------------------------------===//

void LSPServer::onInitialize(const InitializeParams &params,
                             Callback<json::Value> reply) {
  // Send a response with the capabilities of this server.
  json::Object serverCaps{
      {
          "textDocumentSync",
          llvm::json::Object{
              {"openClose", true},
              {"change", (int)TextDocumentSyncKind::Incremental},
              {"save", true},

          },

      },
      {"definitionProvider", true},
      {"referencesProvider", true},
  };

  json::Object result{
      {{"serverInfo", json::Object{{"name", "circt-verilog-lsp-server"},
                                   {"version", "0.0.1"}}},
       {"capabilities", std::move(serverCaps)}}};
  reply(std::move(result));
}
void LSPServer::onInitialized(const InitializedParams &) {}
void LSPServer::onShutdown(const NoParams &, Callback<std::nullptr_t> reply) {
  shutdownRequestReceived.store(true, std::memory_order_relaxed);
  cancelAllBuckets();
  reply(nullptr);
}

//===----------------------------------------------------------------------===//
// Document Change
//===----------------------------------------------------------------------===//

void LSPServer::onDocumentDidOpen(const DidOpenTextDocumentParams &params) {
  PublishDiagnosticsParams diagParams(params.textDocument.uri,
                                      params.textDocument.version);
  server.addDocument(params.textDocument.uri, params.textDocument.text,
                     params.textDocument.version, diagParams.diagnostics);

  // Publish any recorded diagnostics.
  sendDiagnostics(diagParams);
}

void LSPServer::onDocumentDidClose(const DidCloseTextDocumentParams &params) {
  std::optional<int64_t> version =
      server.removeDocument(params.textDocument.uri);
  if (!version)
    return;
  eraseBucket(params.textDocument.uri.file());

  // Empty out the diagnostics shown for this document. This will clear out
  // anything currently displayed by the client for this document (e.g. in the
  // "Problems" pane of VSCode).
  sendDiagnostics(PublishDiagnosticsParams(params.textDocument.uri, *version));
}

void LSPServer::onDocumentDidChange(const DidChangeTextDocumentParams &params) {
  PublishDiagnosticsParams diagParams(params.textDocument.uri,
                                      params.textDocument.version);
  server.updateDocument(params.textDocument.uri, params.contentChanges,
                        params.textDocument.version, diagParams.diagnostics);

  // Publish any recorded diagnostics.
  sendDiagnostics(diagParams);
}

void LSPServer::onDocumentDidChangeDebounce(
    const DidChangeTextDocumentParams &params) {
  auto fb = getOrCreateBucket(params.textDocument.uri.file());
  // Buffer the raw change events (no text mutation yet)
  fb->changeBuf.add(params.contentChanges, params.textDocument.version);
  auto uri = params.textDocument.uri;
  fb->deb.schedule([this, fb, uri]() {
    if (shutdownRequestReceived.load(std::memory_order_relaxed))
      return;
    if (!fb->alive.load(std::memory_order_acquire))
      return; // We've been closed in the meantime

    int64_t toVersion = 0;
    // Early exit: nothing buffered
    auto batch = fb->changeBuf.drain(toVersion);
    if (batch.empty())
      return; // nothing to do

    if (fb->building.exchange(true, std::memory_order_acq_rel)) {
      fb->rerun.store(true, std::memory_order_release);
      return;
    }

    do {
      fb->rerun.store(false, std::memory_order_release);
      PublishDiagnosticsParams diagParams(uri, toVersion);
      // No lock needed here; updates on VerilogDocument are synchronized in
      // VerilogTextFile.
      server.updateDocument(diagParams.uri, batch, diagParams.version,
                            diagParams.diagnostics);
      // Publish any recorded diagnostics.
      sendDiagnostics(diagParams);
    } while (fb->rerun.load(std::memory_order_acquire));
    fb->building.store(false, std::memory_order_release);
  });
}

//===----------------------------------------------------------------------===//
// Definitions and References
//===----------------------------------------------------------------------===//

void LSPServer::onGoToDefinition(const TextDocumentPositionParams &params,
                                 Callback<std::vector<Location>> reply) {
  std::vector<Location> locations;
  server.getLocationsOf(params.textDocument.uri, params.position, locations);
  reply(std::move(locations));
}

void LSPServer::onReference(const ReferenceParams &params,
                            Callback<std::vector<Location>> reply) {
  std::vector<Location> locations;
  server.findReferencesOf(params.textDocument.uri, params.position, locations);
  reply(std::move(locations));
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

LogicalResult
circt::lsp::runVerilogLSPServer(const circt::lsp::LSPServerOptions &options,
                                VerilogServer &server,
                                JSONTransport &transport) {
  LSPServer lspServer(options, server, transport);
  MessageHandler messageHandler(transport);

  // Initialization
  messageHandler.method("initialize", &lspServer, &LSPServer::onInitialize);
  messageHandler.notification("initialized", &lspServer,
                              &LSPServer::onInitialized);
  messageHandler.method("shutdown", &lspServer, &LSPServer::onShutdown);

  // Document Changes
  messageHandler.notification("textDocument/didOpen", &lspServer,
                              &LSPServer::onDocumentDidOpen);
  messageHandler.notification("textDocument/didClose", &lspServer,
                              &LSPServer::onDocumentDidClose);

  if (options.disableDebounce)
    messageHandler.notification("textDocument/didChange", &lspServer,
                                &LSPServer::onDocumentDidChange);
  else
    messageHandler.notification("textDocument/didChange", &lspServer,
                                &LSPServer::onDocumentDidChangeDebounce);
  // Definitions and References
  messageHandler.method("textDocument/definition", &lspServer,
                        &LSPServer::onGoToDefinition);
  messageHandler.method("textDocument/references", &lspServer,
                        &LSPServer::onReference);

  // Diagnostics
  lspServer.publishDiagnostics =
      messageHandler.outgoingNotification<PublishDiagnosticsParams>(
          "textDocument/publishDiagnostics");

  // Run the main loop of the transport.
  if (Error error = transport.run(messageHandler)) {
    Logger::error("Transport error: {0}", error);
    consumeError(std::move(error));
    return failure();
  }

  return success(lspServer.shutdownRequestReceived);
}
