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

#include "circt/Dialect/Arc/Runtime/TraceEncoder.h"

#include "circt/Dialect/Arc/Runtime/TraceTaps.h"

#include <iostream>

namespace circt::arc::runtime::impl {

TraceEncoder::TraceEncoder(const ArcRuntimeModelInfo *modelInfo,
                           ArcState *state, unsigned numBuffers,
                           bool debug = false)
    : numBuffers(numBuffers), modelInfo(modelInfo), debug(debug),
      activeBuffer(modelInfo->traceInfo->traceBufferCapacity) {
  assert(numBuffers > 0);
  assert(!!modelInfo->traceInfo && modelInfo->traceInfo->numTraceTaps > 0);
  timeStep = 0;
  isFinished = false;
  worker = {};
  // Put the first buffer in place
  state->traceBuffer = activeBuffer.getData();
  // Prime the buffer return stack with additional buffers
  for (unsigned i = 1; i < numBuffers; ++i)
    availableBuffers.emplace_back(modelInfo->traceInfo->traceBufferCapacity);
}

// Enqueue a trace buffer for processing by the worker thread.
void TraceEncoder::enqueueBuffer(TraceBuffer &&buffer) {
  assert(worker.has_value());
  std::unique_lock<std::mutex> lock(bufferQueueMutex);
  bufferQueue.emplace(std::move(buffer));
  lock.unlock();
  bufferQueueCv.notify_one();
}

// Retrieve a buffer from the stack of processed buffers.
TraceBuffer TraceEncoder::getBuffer() {
  assert(worker.has_value());
  std::unique_lock<std::mutex> lock(availableBuffersMutex);
  availableBuffersCv.wait(lock, [this]() { return !availableBuffers.empty(); });
  auto buffer = std::move(availableBuffers.back());
  availableBuffers.pop_back();
  return buffer;
}

void TraceEncoder::run(ArcState *state) {
  if (worker.has_value() || !initialize(state))
    return;
  // Flush the trace buffer which may contain data from the init function
  state->traceBufferSize = 0;
  timeStep = 0;
  activeBuffer.firstStep = 0;
  assert(activeBuffer.stepMarkers.empty());
  assert(bufferQueue.empty());
  // Start the worker thread
  if (debug)
    std::cout << "[ArcRuntime] Starting trace worker thread." << std::endl;
  std::thread newWorker([this]() { this->workLoop(); });
  worker = std::move(newWorker);
}

void TraceEncoder::workLoop() {
  startUpWorker();
  std::optional<TraceBuffer> work = {};
  while (true) {
    {
      // Take buffer
      std::unique_lock<std::mutex> lock(bufferQueueMutex);
      bufferQueueCv.wait(
          lock, [this]() { return isFinished || !bufferQueue.empty(); });

      // Still work to do?
      if (bufferQueue.empty() && isFinished)
        break; // Release bufferQueueMutex

      assert(!bufferQueue.empty());
      work = std::move(bufferQueue.front());
      bufferQueue.pop();
    } // Release bufferQueueMutex

    // Process the taken buffer
    assert(work->size > 0);
    encode(*work);
    work->clear();

    {
      // Return buffer
      std::lock_guard<std::mutex> lock(availableBuffersMutex);
      availableBuffers.emplace_back(std::move(*work));
    } // Release availableBuffersMutex
    work.reset();
    availableBuffersCv.notify_one();
  }
  windDownWorker();
}

void TraceEncoder::step(const ArcState *state) {
  activeBuffer.assertSentinel();
  ++timeStep;
  if (!worker)
    return;
  assert(activeBuffer.getData() == state->traceBuffer);
  if (state->traceBufferSize > 0) {
    // Mark the beginning of the new step in the active trace buffer
    auto &offsets = activeBuffer.stepMarkers;
    if (!offsets.empty() && offsets.back().offset == state->traceBufferSize) {
      // No new data produced since last time: Bump the existing last step.
      offsets.back().numSteps++;
    } else {
      // Store the current offset
      offsets.emplace_back(state->traceBufferSize);
    }
  } else {
    // Untouched buffer: Adjust the first step value.
    assert(activeBuffer.stepMarkers.empty());
    activeBuffer.firstStep = timeStep;
  }
}

void TraceEncoder::finish(const ArcState *state) {
  activeBuffer.assertSentinel();
  if (worker.has_value()) {
    // Dispatch the final buffer and request the worker to finish.
    if (state->traceBufferSize > 0)
      dispatch(state->traceBufferSize);
    {
      std::lock_guard<std::mutex> lock(bufferQueueMutex);
      isFinished = true;
    } // Release bufferQueueMutex
    bufferQueueCv.notify_one();
    // Wait for the worker to finish.
    worker->join();
    worker.reset();
    if (debug)
      std::cout << "[ArcRuntime] Trace worker thread finished." << std::endl;
  }
  finalize(state);
}

uint64_t *TraceEncoder::dispatch(uint32_t oldBufferSize) {
  activeBuffer.assertSentinel();
  if (oldBufferSize == 0)
    impl::fatalError("Trace dispatch called on an empty buffer");
  if (worker.has_value()) {
    activeBuffer.size = oldBufferSize;
    enqueueBuffer(std::move(activeBuffer));
    activeBuffer = getBuffer();
    activeBuffer.firstStep = timeStep;
  }
  return activeBuffer.getData();
}

} // namespace circt::arc::runtime::impl
