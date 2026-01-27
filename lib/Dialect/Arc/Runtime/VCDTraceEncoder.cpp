//===- VCDTraceEncoder.cpp - VCD Trace Encoder ----------------------------===//
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

#include "circt/Dialect/Arc/Runtime/VCDTraceEncoder.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <iostream>
#include <ostream>
#include <string>
#include <string_view>

using namespace circt::arc::runtime::impl;

namespace {

// Helper for referencing the name of a signal within the model's name array
struct SignalNameRef {
  SignalNameRef(uint64_t signalIndex, uint64_t nameOffset)
      : signalIndex(signalIndex), nameOffset(nameOffset) {}

  uint64_t signalIndex;
  uint64_t nameOffset;

  std::string_view getStringView(const char *nameBlob) const {
    return std::string_view(nameBlob + nameOffset);
  }
};

// Write the given signal and its value to the ASCII data buffer.
// Note: This is the encoder's hot spot.
static inline void dumpSignal(const VCDSignalTableEntry &signal, char *dest,
                              const uint8_t *data) {
  const auto *sigStr = signal.id.cStr();
  if (signal.numBits > 1)
    *(dest++) = 'b';
  for (unsigned n = signal.numBits; n > 0; --n)
    *(dest++) = (data[(n - 1) / 8] & (1 << ((n - 1) % 8)) ? '1' : '0');
  if (signal.numBits > 1)
    *(dest++) = ' ';
  for (unsigned n = 0; n < signal.id.getNumChars(); ++n)
    *(dest++) = *(sigStr++);
  *(dest) = '\n';
}

static inline void dumpSignalToString(const VCDSignalTableEntry &signal,
                                      std::string &dest, const uint8_t *data) {
  auto strOffset = dest.size();
  dest.resize(strOffset + signal.getDumpSize());
  dumpSignal(signal, &dest[strOffset], data);
}

static void writeVCDHeader(std::basic_ostream<char> &os) {
  // TODO: Add the current date to the header. For now, keep the output
  // stable to facilitate comparisons.
  os << "$version\n    Some cryptic ArcRuntime magic\n$end\n";
  os << "$timescale 1ns $end\n";
}

} // namespace

namespace circt::arc::runtime::impl {

VCDSignalId::VCDSignalId(uint64_t index) {
  raw.fill('\0');
  unsigned pos = 0;
  const char base = ('~' - '!') + 1;
  do {
    assert(pos < (sizeof(raw) - 1) && "Signal ID out of range");
    raw[pos] = '!' + static_cast<char>(index % base);
    index /= base;
    ++pos;
  } while (index != 0);
  numChars = pos;
}

VCDTraceEncoder::VCDTraceEncoder(const ArcRuntimeModelInfo *modelInfo,
                                 ArcState *state,
                                 const std::filesystem::path &outFilePath,
                                 bool debug)
    : TraceEncoder(modelInfo, state, numTraceBuffers, debug),
      outFilePath(outFilePath) {}

void VCDTraceEncoder::initSignalTable() {
  const auto *info = modelInfo->traceInfo;
  signalTable.reserve(info->numTraceTaps);
  for (uint64_t i = 0; i < info->numTraceTaps; ++i)
    signalTable.emplace_back(i, info->traceTaps[i].stateOffset,
                             info->traceTaps[i].typeBits);
}

static void appendLegalizedName(std::string &buffer,
                                const std::string_view &name) {
  if (name.empty()) {
    buffer.append("<EMPTY>");
    return;
  }
  for (auto c : name) {
    // TODO: Escape illegal characters
    if (c == ' ')
      buffer.push_back('_');
    else if (std::isprint(c))
      buffer.push_back(c);
  }
}

void VCDTraceEncoder::createHierarchy() {
  assert(!signalTable.empty());
  const auto *info = modelInfo->traceInfo;
  std::string vcdHdr;
  const char *sigNameBlob = info->traceTapNames;
  auto nameRefVector = std::vector<SignalNameRef>();
  // Start of the name/alias
  uint64_t sigOffset = 0;
  // Start of the next signal's first name
  uint64_t sigOffsetNext = info->traceTaps[0].nameOffset + 1;
  uint64_t signalIndex = 0;
  while (signalIndex < info->numTraceTaps) {
    nameRefVector.emplace_back(signalIndex, sigOffset);
    auto nameLen = nameRefVector.back().getStringView(sigNameBlob).length();
    // Advance to the next name/alias
    sigOffset += nameLen + 1;
    if (sigOffset >= sigOffsetNext) {
      // We've reached the next signal
      assert(sigOffset == sigOffsetNext);
      ++signalIndex;
      // Bump to the new next signal
      if (signalIndex < info->numTraceTaps)
        sigOffsetNext = info->traceTaps[signalIndex].nameOffset + 1;
    }
  }

  // Sort lexicographically
  std::stable_sort(
      nameRefVector.begin(), nameRefVector.end(),
      [sigNameBlob](const SignalNameRef &a, const SignalNameRef &b) {
        return a.getStringView(sigNameBlob) < b.getStringView(sigNameBlob);
      });

  std::vector<std::string_view> currentScope;
  for (auto &name : nameRefVector) {
    auto nameStr = name.getStringView(sigNameBlob);
    std::vector<std::string_view> sigScope;
    size_t charOffset = 0;
    // Push the signal's scope names onto the stack
    auto newOffset = std::string::npos;
    while ((newOffset = nameStr.find('/', charOffset)) != std::string::npos) {
      sigScope.push_back(nameStr.substr(charOffset, newOffset - charOffset));
      charOffset = newOffset + 1;
    }
    // Count how many scopes match the current scope
    unsigned commonScopes = 0;
    for (size_t i = 0; i < std::min(currentScope.size(), sigScope.size());
         ++i) {
      if (sigScope[i] == currentScope[i])
        ++commonScopes;
      else
        break;
    }
    // Pop scopes until we have reached the first common scope
    while (commonScopes < currentScope.size()) {
      currentScope.pop_back();
      vcdHdr.append(currentScope.size() + 1, ' ');
      vcdHdr.append("$upscope $end\n");
    }
    // Push the new scopes
    while (sigScope.size() > currentScope.size()) {
      vcdHdr.append(currentScope.size() + 1, ' ');
      vcdHdr.append("$scope module ");
      appendLegalizedName(vcdHdr, sigScope[currentScope.size()]);
      vcdHdr.append(" $end\n");
      currentScope.push_back(sigScope[currentScope.size()]);
    }
    // Write the signal declaration
    vcdHdr.append(currentScope.size() + 1, ' ');
    vcdHdr.append("$var wire ");
    vcdHdr.append(std::to_string(signalTable[name.signalIndex].numBits));
    vcdHdr.append(" ");
    vcdHdr.append(signalTable[name.signalIndex].id.cStr());
    vcdHdr.append(" ");
    appendLegalizedName(vcdHdr, nameStr.substr(charOffset));
    vcdHdr.append(" $end\n");

    outFile << vcdHdr;
    vcdHdr.clear();
  }
}

static inline void writeTimestepToBuffer(int64_t currentStep,
                                         std::vector<char> &buf) {
  buf.push_back('#');
  auto stepStr = std::to_string(currentStep);
  std::copy(&stepStr.c_str()[0], &stepStr.c_str()[stepStr.size()],
            std::back_inserter(buf));
  buf.push_back('\n');
}

bool VCDTraceEncoder::initialize(const ArcState *state) {
  // Use a large 32K IO buffer as we might be writing gigabytes of data
  const size_t fileBufferCapacity = 32 * 1024;
  fileBuffer = std::unique_ptr<char[]>(new char[fileBufferCapacity]);
  outFile.rdbuf()->pubsetbuf(fileBuffer.get(), fileBufferCapacity);
  outFile.open(outFilePath, std::ios::out | std::ios::trunc);
  if (!outFile.is_open()) {
    std::cerr << "[ArcRuntime] WARNING: Unable to open VCD trace file "
              << outFilePath << " for writing. No trace will be produced."
              << std::endl;
    return false;
  }
  if (debug)
    std::cout << "[ArcRuntime] Created VCD trace file: " << outFilePath
              << std::endl;

  // Create the signal table from the model's metadata
  initSignalTable();

  writeVCDHeader(outFile);
  // Write out the signal hierarchy
  outFile << "$scope module " << modelInfo->modelName << " $end\n";
  createHierarchy();
  outFile << "$upscope $end\n";
  outFile << "$enddefinitions $end\n";
  // Dump the entire initial state
  outFile << "#0\n";
  std::string outstring;
  for (const auto &sig : signalTable) {
    assert(sig.stateOffset < modelInfo->numStateBytes);
    dumpSignalToString(sig, outstring, &state->modelState[sig.stateOffset]);
  }
  outFile << outstring;

  outFile.flush();
  return true;
}

void VCDTraceEncoder::startUpWorker() { workerStep = -1; }

void VCDTraceEncoder::encode(TraceBuffer &work) {
  size_t offset = 0;
  assert(workerStep <= work.firstStep);
  if (workerStep != work.firstStep) {
    // The new buffer starts at a different step than the previous one ended on
    workerStep = work.firstStep;
    writeTimestepToBuffer(workerStep, workerOutBuffer);
  }
  // Walk through the the buffer
  const uint64_t *basePtr = work.getData();
  auto stepIter = work.stepMarkers.begin();
  while (offset < work.size) {
    // The trace tap ID and index in the signal table
    auto idx = basePtr[offset];
    ++offset;
    // Pointer to the signal value
    const void *voidPtr = static_cast<const void *>(basePtr + offset);
    // Dump the signal and advance the offset
    auto numChars = signalTable[idx].getDumpSize();
    auto strOffset = workerOutBuffer.size();
    workerOutBuffer.resize(strOffset + numChars);
    dumpSignal(signalTable[idx], &workerOutBuffer[strOffset],
               static_cast<const uint8_t *>(voidPtr));
    offset += signalTable[idx].getStride();
    assert(offset <= work.size);
    // Check if we have reached a new time step marker
    if (stepIter != work.stepMarkers.end() && stepIter->offset == offset) {
      workerStep += stepIter->numSteps;
      writeTimestepToBuffer(workerStep, workerOutBuffer);
      stepIter++;
    }
  }
  assert(offset == work.size);
  assert(stepIter == work.stepMarkers.end());
  // Write out our buffer
  outFile.write(workerOutBuffer.data(), workerOutBuffer.size());
  workerOutBuffer.clear();
}

void VCDTraceEncoder::windDownWorker() {
  assert(workerOutBuffer.empty());
  // Terminate and flush the file
  workerStep++;
  writeTimestepToBuffer(workerStep, workerOutBuffer);
  outFile.write(workerOutBuffer.data(), workerOutBuffer.size());
  outFile.flush();
  workerOutBuffer.clear();
}

void VCDTraceEncoder::finalize(const ArcState *state) {
  if (outFile.is_open())
    outFile.close();
}

} // namespace circt::arc::runtime::impl
