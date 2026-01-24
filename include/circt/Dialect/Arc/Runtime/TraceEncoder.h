#ifndef CIRCT_DIALECT_ARC_RUNTIME_TRACEENCODER_H
#define CIRCT_DIALECT_ARC_RUNTIME_TRACEENCODER_H

#include "circt/Dialect/Arc/Runtime/Common.h"
#include "circt/Dialect/Arc/Runtime/Internal.h"

#include <cassert>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace circt::arc::runtime::impl {

enum class TraceMode { DUMMY };

struct TraceBufferOffset {
  TraceBufferOffset() = delete;
  explicit TraceBufferOffset(uint32_t offset) : offset(offset), numSteps(1) {}
  uint32_t offset;
  uint32_t numSteps;
};

struct TraceBuffer {

public:
  TraceBuffer() = delete;
  explicit TraceBuffer(uint32_t capacity) : capacity(capacity) {
    storage = std::make_unique<uint64_t[]>(capacity + 1);
    storage[capacity] = sentinelValue;
  }

  TraceBuffer(const TraceBuffer &other) = delete;
  TraceBuffer operator=(const TraceBuffer &other) = delete;

  TraceBuffer(TraceBuffer &&other) noexcept = default;
  TraceBuffer &operator=(TraceBuffer &&other) noexcept = default;

  // Note: Actual capacity is +1 for sentinel
  uint32_t capacity;
  uint32_t size = 0;
  int64_t firstStep = -1;
  std::vector<TraceBufferOffset> stepOffsets;

  uint64_t *getData() const { return storage.get(); }

  void clear() {
    size = 0;
    firstStep = -1;
    stepOffsets.clear();
  }

  void assertSentinel() const {
    if (storage[capacity] != sentinelValue)
      impl::fatalError("Trace buffer sentinel overwritten");
  }

private:
  static constexpr uint64_t sentinelValue = UINT64_C(0xABCD1234EFBA8765);
  std::unique_ptr<uint64_t[]> storage;
};

class TraceEncoder {
public:
  TraceEncoder() = delete;
  TraceEncoder(const ArcRuntimeModelInfo *modelInfo, ArcState *state,
               unsigned numBuffers, bool debug);

  virtual ~TraceEncoder() = default;

  void run(ArcState const *state);
  uint64_t *dispatch(uint32_t oldBufferSize);
  void step(const ArcState *state);
  void finish(const ArcState *state);

  const unsigned numBuffers;

protected:
  virtual bool initialize(const ArcState *state) = 0;
  virtual void startUpWorker() {};
  virtual void encode(TraceBuffer &work) {};
  virtual void windDownWorker() {};
  virtual void finalize(const ArcState *state) {};

  const ArcRuntimeModelInfo *const modelInfo;
  const bool debug;

private:
  void workLoop();

  void enqueueBuffer(TraceBuffer &&buffer);
  TraceBuffer getBuffer();

  std::optional<std::thread> worker;
  std::mutex availableBuffersMutex;
  std::condition_variable availableBuffersCv;
  std::vector<TraceBuffer> availableBuffers;

  std::mutex bufferQueueMutex;
  std::condition_variable bufferQueueCv;
  std::queue<TraceBuffer> bufferQueue;

  std::atomic<bool> isFinished;

  TraceBuffer activeBuffer;
  int64_t timeStep;
};

class DummyTraceEncoder final : public TraceEncoder {
public:
  DummyTraceEncoder(const ArcRuntimeModelInfo *modelInfo, ArcState *state)
      : TraceEncoder(modelInfo, state, 1, false) {};

  virtual ~DummyTraceEncoder() = default;

protected:
  virtual bool initialize(const ArcState *state) override { return false; }
};

} // namespace circt::arc::runtime::impl

#endif // CIRCT_DIALECT_ARC_RUNTIME_TRACEENCODER_H
