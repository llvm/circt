//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides an abstract TraceEncoder class implementing the trace buffer
// management and communication between the simulation thread and the trace
// encoder thread.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_RUNTIME_TRACEENCODER_H
#define CIRCT_DIALECT_ARC_RUNTIME_TRACEENCODER_H

#include "circt/Dialect/Arc/Runtime/Common.h"
#include "circt/Dialect/Arc/Runtime/Internal.h"

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <vector>

namespace circt::arc::runtime::impl {

/// Helper for marking time steps within a trace buffer
struct TraceBufferMarker {
  TraceBufferMarker() = delete;
  explicit TraceBufferMarker(uint32_t offset) : offset(offset), numSteps(1) {}
  /// Offset within the buffer in number of elements
  uint32_t offset;
  /// Number of time steps to advance at the given offset
  uint32_t numSteps;
};

/// A heap allocated buffer containing raw trace data and time step markers
struct TraceBuffer {

public:
  TraceBuffer() = delete;
  explicit TraceBuffer(uint32_t capacity) : capacity(capacity) {
    storage = std::make_unique<uint64_t[]>(capacity + 1);
    storage[capacity] = sentinelValue;
  }

  // Never copy a trace buffer
  TraceBuffer(const TraceBuffer &other) = delete;
  TraceBuffer operator=(const TraceBuffer &other) = delete;

  TraceBuffer(TraceBuffer &&other) noexcept = default;
  TraceBuffer &operator=(TraceBuffer &&other) noexcept = default;

  /// Available storage in number of elements.
  /// Note: Actual capacity is +1 for sentinel
  uint32_t capacity;
  /// Number of valid elements, set on dispatch
  uint32_t size = 0;
  /// Time step of the buffer's first entry
  int64_t firstStep = -1;
  /// Time step markers
  std::vector<TraceBufferMarker> stepMarkers;

  /// Get the pointer to the buffer's storage
  uint64_t *getData() const { return storage.get(); }

  /// Reset the buffer
  void clear() {
    size = 0;
    firstStep = -1;
    stepMarkers.clear();
  }

  /// Assert that the buffer's sentinel value has not been overwritten
  void assertSentinel() const {
    if (storage[capacity] != sentinelValue)
      impl::fatalError("Trace buffer sentinel overwritten");
  }

private:
  static constexpr uint64_t sentinelValue = UINT64_C(0xABCD1234EFBA8765);
  std::unique_ptr<uint64_t[]> storage;
};

/// Abstract TraceEncoder managing trace buffers and the encoder thread
class TraceEncoder {
public:
  TraceEncoder() = delete;
  /// Construct a trace encoder for the given model and the given number of
  /// buffers. Initializes the state with the first buffer.
  TraceEncoder(const ArcRuntimeModelInfo *modelInfo, ArcState *state,
               unsigned numBuffers, bool debug);

  virtual ~TraceEncoder() = default;

  /// Begin tracing
  void run(ArcState *state);
  /// Dispatch the currently active trace buffer containing `oldBufferSize`
  /// valid entries to the encoder thread and return the storage of the
  /// new active buffer.
  /// Blocks execution until a new buffer is available.
  uint64_t *dispatch(uint32_t oldBufferSize);
  /// Signal the start of a new simulation time step
  void step(const ArcState *state);
  /// Stop tracing
  void finish(const ArcState *state);

  /// Number of trace buffers in rotation
  const unsigned numBuffers;

protected:
  // Virtual methods to be implemented by encoder backends.

  /// Set-up the encoder before starting the worker thread. May return `false`
  /// to indicate failure. In this case, no worker thread will be created.
  /// Called by the simulation thread.
  virtual bool initialize(const ArcState *state) = 0;

  /// Called by the worker thread before entering the encode loop
  virtual void startUpWorker(){};
  /// Encode the given trace buffer. Called by the worker thread.
  virtual void encode(TraceBuffer &work){};
  /// Called by the worker thread after leaving the encode loop
  virtual void windDownWorker(){};
  /// Finish trace encoding. Called by the simulation thread.
  virtual void finalize(const ArcState *state){};

  /// Metadata of the traced model
  const ArcRuntimeModelInfo *const modelInfo;
  /// Debug mode flag
  const bool debug;

private:
  /// The encoder thread's work loop
  void workLoop();

  void enqueueBuffer(TraceBuffer &&buffer);
  TraceBuffer getBuffer();

  /// Current simulation time step
  int64_t timeStep;

  /// Trace encoder worker thread. If empty, tracing is disabled.
  std::optional<std::thread> worker;

  /// Queue and synchronization primitives for buffers to be processed by the
  /// encoder thread
  std::mutex bufferQueueMutex;
  std::condition_variable bufferQueueCv;
  std::queue<TraceBuffer> bufferQueue;

  /// Return stack and synchronization primitives for processed buffers
  std::mutex availableBuffersMutex;
  std::condition_variable availableBuffersCv;
  std::vector<TraceBuffer> availableBuffers;

  /// Flag signaling that no more buffers will be enqueued
  std::atomic<bool> isFinished;

  /// Trace buffer currently in use by the hardware model
  TraceBuffer activeBuffer;
};

/// Dummy encoder discarding all produced trace data
class DummyTraceEncoder final : public TraceEncoder {
public:
  DummyTraceEncoder(const ArcRuntimeModelInfo *modelInfo, ArcState *state)
      : TraceEncoder(modelInfo, state, 1, false){};

  ~DummyTraceEncoder() override = default;

protected:
  bool initialize(const ArcState *state) override { return false; }
};

} // namespace circt::arc::runtime::impl

#endif // CIRCT_DIALECT_ARC_RUNTIME_TRACEENCODER_H
