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

#include "circt/Dialect/Arc/Runtime/FSTTraceEncoder.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <iostream>
#include <string>
#include <string_view>

#include "third_party/libfst/fstapi.h"

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

static void appendLegalizedName(std::string &buffer,
                                const std::string_view &name) {
  if (name.empty()) {
    buffer.append("<EMPTY>");
    return;
  }
  for (auto c : name) {
    if (c == ' ')
      buffer.push_back('_');
    else if (std::isprint(c))
      buffer.push_back(c);
  }
}

} // namespace

namespace circt::arc::runtime::impl {

FSTTraceEncoder::FSTTraceEncoder(const ArcRuntimeModelInfo *modelInfo,
                                 ArcState *state,
                                 const std::filesystem::path &outFilePath,
                                 bool debug)
    : TraceEncoder(modelInfo, state, numTraceBuffers, debug),
      outFilePath(outFilePath) {}

void FSTTraceEncoder::initSignalTable() {
  const auto *info = modelInfo->traceInfo;
  signalTable.reserve(info->numTraceTaps);
  for (uint64_t i = 0; i < info->numTraceTaps; ++i)
    signalTable.emplace_back(i, info->traceTaps[i].stateOffset,
                             info->traceTaps[i].typeBits);
}

void FSTTraceEncoder::createHierarchy() {
  assert(!signalTable.empty());
  const auto *info = modelInfo->traceInfo;
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

  std::vector<std::string> currentScope;
  for (auto &name : nameRefVector) {
    auto nameStr = name.getStringView(sigNameBlob);
    std::vector<std::string> sigScope;
    size_t charOffset = 0;
    // Push the signal's scope names onto the stack
    auto newOffset = std::string::npos;
    while ((newOffset = nameStr.find('/', charOffset)) != std::string::npos) {
      std::string scopeName(nameStr.substr(charOffset, newOffset - charOffset));
      std::string legalizedScope;
      appendLegalizedName(legalizedScope, scopeName);
      sigScope.push_back(legalizedScope);
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
      fstWriterSetUpscope(fstWriter);
    }
    // Push the new scopes
    while (sigScope.size() > currentScope.size()) {
      fstWriterSetScope(fstWriter, FST_ST_VCD_MODULE,
                        sigScope[currentScope.size()].c_str(), nullptr);
      currentScope.push_back(sigScope[currentScope.size()]);
    }
    // Write the signal declaration
    std::string sigName;
    appendLegalizedName(sigName, nameStr.substr(charOffset));
    uint32_t handle = fstWriterCreateVar(
        fstWriter, FST_VT_VCD_WIRE, FST_VD_IMPLICIT,
        signalTable[name.signalIndex].numBits, sigName.c_str(),
        signalTable[name.signalIndex].handle);
    assert(handle != 0);
    signalTable[name.signalIndex].handle = handle;
  }
}

bool FSTTraceEncoder::initialize(const ArcState *state) {
  fstWriter = fstWriterCreate(outFilePath.string().c_str(), 1);
  if (!fstWriter) {
    std::cerr << "[ArcRuntime] WARNING: Unable to open FST trace file "
              << outFilePath << " for writing. No trace will be produced."
              << std::endl;
    return false;
  }

  fstWriterSetPackType(fstWriter, FST_WR_PT_LZ4);
  fstWriterSetTimescaleFromString(fstWriter, "1 fs");

  if (debug)
    std::cout << "[ArcRuntime] Created FST trace file: " << outFilePath
              << std::endl;

  initSignalTable();

  fstWriterSetScope(fstWriter, FST_ST_VCD_MODULE, modelInfo->modelName,
                    nullptr);
  createHierarchy();
  fstWriterSetUpscope(fstWriter);

  // Dump the entire initial state
  fstWriterEmitTimeChange(fstWriter, 0);
  std::string valStr;
  for (const auto &sig : signalTable) {
    assert(sig.stateOffset < modelInfo->numStateBytes);
    valStr.clear();
    const uint8_t *data = &state->modelState[sig.stateOffset];
    for (unsigned n = sig.numBits; n > 0; --n)
      valStr.push_back((data[(n - 1) / 8] & (1 << ((n - 1) % 8))) ? '1' : '0');
    fstWriterEmitValueChange(fstWriter, sig.handle, valStr.c_str());
  }

  return true;
}

void FSTTraceEncoder::startUpWorker() { workerStep = -1; }

void FSTTraceEncoder::encode(TraceBuffer &work) {
  size_t offset = 0;
  assert(workerStep <= work.firstStep);
  if (workerStep != work.firstStep) {
    workerStep = work.firstStep;
    fstWriterEmitTimeChange(fstWriter, work.firstSimTime);
  }

  const uint64_t *basePtr = work.getData();
  auto stepIter = work.stepMarkers.begin();

  std::string valStr;
  while (offset < work.size) {
    auto idx = basePtr[offset];
    ++offset;
    const void *voidPtr = static_cast<const void *>(basePtr + offset);

    valStr.clear();
    const uint8_t *data = static_cast<const uint8_t *>(voidPtr);
    for (unsigned n = signalTable[idx].numBits; n > 0; --n)
      valStr.push_back((data[(n - 1) / 8] & (1 << ((n - 1) % 8))) ? '1' : '0');

    fstWriterEmitValueChange(fstWriter, signalTable[idx].handle,
                             valStr.c_str());
    offset += signalTable[idx].getStride();
    assert(offset <= work.size);
    // Check if we have reached a new time step marker
    if (stepIter != work.stepMarkers.end() && stepIter->offset == offset) {
      workerStep += stepIter->numSteps;
      fstWriterEmitTimeChange(fstWriter, stepIter->simTime);
      stepIter++;
    }
  }
  assert(offset == work.size);
  assert(stepIter == work.stepMarkers.end());
}

void FSTTraceEncoder::windDownWorker() {}

void FSTTraceEncoder::finalize(const ArcState *state) {
  if (fstWriter) {
    // Finalize the trace file with the final simulation time
    assert(workerStep <= getTimeStep());
    uint64_t finalSimTime =
        *reinterpret_cast<const uint64_t *>(state->modelState);
    fstWriterEmitTimeChange(fstWriter, finalSimTime);
    fstWriterClose(fstWriter);
    fstWriter = nullptr;
  }
}

} // namespace circt::arc::runtime::impl
