//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TraceEncoder subclass converting and outputting a stream of raw trace buffers
// to a VCD file.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_RUNTIME_VCDTRACEENCODER_H
#define CIRCT_DIALECT_ARC_RUNTIME_VCDTRACEENCODER_H

#include "circt/Dialect/Arc/Runtime/Common.h"
#include "circt/Dialect/Arc/Runtime/TraceEncoder.h"
#include "circt/Dialect/Arc/Runtime/TraceTaps.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

namespace circt::arc::runtime::impl {

/// String identifier for a signal in a VCD file consisting of characters in
/// the ASCII range from '!' to '~'.
struct VCDSignalId {
  VCDSignalId() = delete;
  /// Create the string ID from an integer ID
  explicit VCDSignalId(uint64_t index);

  /// Get the ID as null terminated string
  inline const char *cStr() const { return raw.data(); }
  /// Get the number of characters in the ID
  inline unsigned getNumChars() const { return numChars; }

private:
  unsigned numChars;
  std::array<char, 16> raw;
};

/// A traced signal in the VCD file
struct VCDSignalTableEntry {
  VCDSignalTableEntry(uint64_t index, uint64_t stateOffset, uint32_t numBits)
      : id(VCDSignalId(index)), stateOffset(stateOffset), numBits(numBits) {}

  /// Get the number of characters required to dump this signal's ID and value
  inline unsigned getDumpSize() const {
    return id.getNumChars() + numBits + ((numBits > 1) ? 3 : 1);
  }

  /// Get the number of words occupied by the signal value in the trace buffer
  inline unsigned getStride() const { return (numBits + 63) / 64; }

  /// VCD signal ID
  VCDSignalId id;
  /// Offest of the signal in the model's simulation state
  uint64_t stateOffset;
  /// Bit width of the signal
  uint32_t numBits;
};

/// Trace buffer encoder producing a VCD file
class VCDTraceEncoder final : public TraceEncoder {
public:
  static constexpr unsigned numTraceBuffers = 4;

  VCDTraceEncoder(const ArcRuntimeModelInfo *modelInfo, ArcState *state,
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
  std::vector<VCDSignalTableEntry> signalTable;
  /// Output file stream
  std::ofstream outFile;
  /// Internal buffer of the output stream.
  /// Do not access directly. Do not release until outFile is closed.
  std::unique_ptr<char[]> fileBuffer;
  /// Concatenation buffer of the worker thread
  std::vector<char> workerOutBuffer;
  /// Current time step of the worker thread
  int64_t workerStep;
};

} // namespace circt::arc::runtime::impl

#endif // CIRCT_DIALECT_ARC_RUNTIME_VCDTRACEENCODER_H
