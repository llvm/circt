//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TraceEncoder subclass converting and outputting a stream of raw trace buffers
// to an FST file.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_RUNTIME_FSTTRACEENCODER_H
#define CIRCT_DIALECT_ARC_RUNTIME_FSTTRACEENCODER_H

#include "circt/Dialect/Arc/Runtime/Common.h"
#include "circt/Dialect/Arc/Runtime/TraceEncoder.h"
#include "circt/Dialect/Arc/Runtime/TraceTaps.h"

#include <filesystem>
#include <string>
#include <vector>

struct fstWriterContext;

namespace circt::arc::runtime::impl {

/// A traced signal in the FST file
struct FSTSignalTableEntry {
  FSTSignalTableEntry(uint64_t index, uint64_t stateOffset, uint32_t numBits)
      : handle(0), stateOffset(stateOffset), numBits(numBits) {}

  /// Get the number of words occupied by the signal value in the trace buffer
  inline unsigned getStride() const { return (numBits + 63) / 64; }

  /// FST signal handle
  uint32_t handle;
  /// Offest of the signal in the model's simulation state
  uint64_t stateOffset;
  /// Bit width of the signal
  uint32_t numBits;
};

/// Trace buffer encoder producing an FST file
class FSTTraceEncoder final : public TraceEncoder {
public:
  static constexpr unsigned numTraceBuffers = 4;

  FSTTraceEncoder(const ArcRuntimeModelInfo *modelInfo, ArcState *state,
                  const std::filesystem::path &outFilePath, bool debug);

protected:
  bool initialize(const ArcState *state) override;
  void startUpWorker() override;
  void encode(TraceBuffer &work) override;
  void windDownWorker() override;
  void finalize(const ArcState *state) override;

private:
  /// Create the table of signals
  void initSignalTable();
  /// Build and dump the signal hierarchy
  void createHierarchy();

  /// Path to the output file
  const std::filesystem::path outFilePath;
  /// Table of signals: The index matches their Trace Tap ID.
  std::vector<FSTSignalTableEntry> signalTable;
  /// FST writer context
  struct ::fstWriterContext *fstWriter = nullptr;
  /// Current time step of the worker thread
  int64_t workerStep;
};

} // namespace circt::arc::runtime::impl

#endif // CIRCT_DIALECT_ARC_RUNTIME_FSTTRACEENCODER_H
