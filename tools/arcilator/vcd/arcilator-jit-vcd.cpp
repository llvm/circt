#include "arcilator-jit-vcd.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <fstream>
#include <iostream>
#include <ostream>
#include <set>
#include <string>
#include <string_view>
#include <vector>

// #define ARC_JIT_TRACE_DEBUG_PRINT
// #define ARC_JIT_VCD_TIMING

#ifdef ARC_JIT_VCD_TIMING
#include <chrono>
#endif

namespace arcilator {
namespace vcd {

struct VCDSignalId {
public:
  explicit VCDSignalId(uint64_t num) {
    raw.fill('\0');
    unsigned index = 0;
    const char base = ('~' - '!') + 1;
    do {
      assert(index < 7 && "Signal ID out of range");
      raw[index] = '!' + static_cast<char>(num % base);
      num /= base;
      ++index;
    } while (num != 0);
  }

  const char *cStr() const { return raw.data(); }

private:
  std::array<char, 8> raw;
};

#ifdef ARC_JIT_VCD_TIMING

struct VCDTracerTiming {
  struct TimeStamp {
    TimeStamp() { valid = false; }
    TimeStamp(const std::chrono::steady_clock::time_point &tp) : tp(tp) {
      valid = true;
    }
    std::chrono::steady_clock::time_point tp;
    bool valid;
  };

  using Duration = std::chrono::nanoseconds;

  static TimeStamp getNow() {
    return TimeStamp(std::chrono::high_resolution_clock::now());
  }

  static auto diffNs(const TimeStamp &start, const TimeStamp &end) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end.tp -
                                                                start.tp)
        .count();
  }

  TimeStamp modelCreation;
  TimeStamp modelDeletion;

  TimeStamp vcdInitStart;
  TimeStamp vcdInitEnd;
  TimeStamp prevStepStart;
  Duration totalSimTime;
  TimeStamp prevVcdProcessStart;
  Duration totalVcdTime;

  int64_t step = -1;
  uint64_t evenChanges;
  uint64_t oddChanges;
  std::array<TimeStamp, 8> startStepStamps;
  std::array<TimeStamp, 8> endStepStamps;

  void dump();

  void stepStart() {
    auto now = getNow();
    ++step;
    if (step >= 0 && step < static_cast<int64_t>(startStepStamps.size()))
      startStepStamps[step] = now;
    prevStepStart = now;
  }

  void stepEnd() {
    auto now = getNow();
    if (step >= 0 && step < static_cast<int64_t>(endStepStamps.size()))
      endStepStamps[step] = now;
    if (prevStepStart.valid) {
      totalSimTime += (now.tp - prevStepStart.tp);
    }
  }

  void vcdProcessStart() { prevVcdProcessStart = getNow(); }

  void vcdProcessEnd() {
    assert(prevStepStart.valid);
    auto diff = getNow().tp - prevVcdProcessStart.tp;
    totalVcdTime += diff;
  }

  void signalsChanged(uint64_t numChanges) {
    if (step < 6)
      return;
    if (step % 2)
      evenChanges += numChanges;
    else
      oddChanges += numChanges;
  }
};
#endif

struct VCDSignalTableEntry {
  VCDSignalTableEntry(uint64_t num, uint64_t stateOffset,
                      uint32_t typeDescriptor)
      : signalId(num), stateOffset(stateOffset),
        typeDescriptor(typeDescriptor) {}

  VCDSignalId signalId;
  uint64_t stateOffset;
  uint32_t typeDescriptor;

  void dumpSignalFromState(std::basic_ostream<char> &os,
                           const uint8_t *statePtr) const;
  unsigned dumpSignalFromTraceBuffer(std::basic_ostream<char> &os,
                                     const uint8_t *buffer) const;
};

struct VcdModelTracer {

  VcdModelTracer() {}

  bool init(const ArcTraceModelInfo *modelInfo);
  void close(ArcTracerState *state);
  void *swapBuffer(void *oldBuffer, uint64_t oldBufferSize);
  void step(ArcTracerState *state);

  void initSignalTable(const ArcTraceModelInfo *info);
  void createHierarchy(const ArcTraceModelInfo *info);
  void processBuffer(const uint64_t *traceBuffer, uint64_t bufferSize);

  std::vector<VCDSignalTableEntry> signalTable;
  std::ofstream outFile;
  int64_t timestep;

#ifdef ARC_JIT_VCD_TIMING
  VCDTracerTiming timing;
#endif
};

struct VcdTraceLibrary {
  static void *initModelCb(const ArcTraceModelInfo *modelInfo);
  static void closeModelCb(ArcTracerState *state);
  static void *swapBufferCb(void *oldBuffer, uint64_t oldBufferSize,
                            void *user);
  static void stepCb(ArcTracerState *state);

  VcdTraceLibrary() {
    library.initModel = &VcdTraceLibrary::initModelCb;
    library.closeModel = &VcdTraceLibrary::closeModelCb;
    library.swapBuffer = &VcdTraceLibrary::swapBufferCb;
    library.step = &VcdTraceLibrary::stepCb;
  }

  ~VcdTraceLibrary() { assert(liveInstances.empty()); }

  void *initModel(const ArcTraceModelInfo *modelInfo) {
    auto *newTracer = new VcdModelTracer();
    if (newTracer->init(modelInfo)) {
      liveInstances.insert(newTracer);
      return static_cast<void *>(newTracer);
    }
    delete newTracer;
    return nullptr;
  }

  void closeModel(ArcTracerState *state) {
    if (!state->user)
      return;
    auto *tracer = static_cast<VcdModelTracer *>(state->user);
    assert(liveInstances.count(tracer) != 0);
    tracer->close(state);
    liveInstances.erase(tracer);
    delete tracer;
  }

  void *swapBuffer(void *oldBuffer, uint64_t oldBufferSize, void *user) {
    if (!user)
      return nullptr;
    auto *tracer = static_cast<VcdModelTracer *>(user);
    assert(liveInstances.count(tracer) != 0);
    return tracer->swapBuffer(oldBuffer, oldBufferSize);
  }

  void step(ArcTracerState *state) {
    if (!state->user)
      return;
    auto *tracer = static_cast<VcdModelTracer *>(state->user);
    assert(liveInstances.count(tracer) != 0);
    tracer->step(state);
  }

  std::set<VcdModelTracer *> liveInstances;

  arcilator::ArcTraceLibrary library;
};

struct SignalNameRef {

  SignalNameRef(uint64_t signalIndex, uint64_t nameOffset)
      : signalIndex(signalIndex), nameOffset(nameOffset) {}

  uint64_t signalIndex;
  uint64_t nameOffset;

  std::string_view getStringView(const char *nameBlob) const {
    return std::string_view(nameBlob + nameOffset);
  }
};

} // namespace vcd
} // namespace arcilator

using namespace arcilator;
using namespace vcd;

static VcdTraceLibrary globalLibInstance;

void *VcdTraceLibrary::initModelCb(const ArcTraceModelInfo *modelInfo) {
#ifdef ARC_JIT_TRACE_DEBUG_PRINT
  std::cerr << "VCD: initModel called" << std::endl;
#endif
  return globalLibInstance.initModel(modelInfo);
}

void VcdTraceLibrary::closeModelCb(ArcTracerState *state) {
#ifdef ARC_JIT_TRACE_DEBUG_PRINT
  std::cerr << "VCD: closeModel called" << std::endl;
#endif
  globalLibInstance.closeModel(state);
}

void *VcdTraceLibrary::swapBufferCb(void *oldBuffer, uint64_t oldBufferSize,
                                    void *user) {
#ifdef ARC_JIT_TRACE_DEBUG_PRINT
  std::cerr << "VCD: swapBuffer called" << std::endl;
#endif
  return globalLibInstance.swapBuffer(oldBuffer, oldBufferSize, user);
}

void VcdTraceLibrary::stepCb(ArcTracerState *state) {
#ifdef ARC_JIT_TRACE_DEBUG_PRINT
  std::cerr << "VCD: step called" << std::endl;
#endif
  globalLibInstance.step(state);
}

const arcilator::ArcTraceLibrary *arcilator::vcd::getVcdTraceLibrary() {
  return &(globalLibInstance.library);
}

static void writeVCDHeader(std::basic_ostream<char> &os) {
  os << "$date\n    October 21, 2015\n$end\n";
  os << "$version\n    Some cryptic JIT MLIR magic\n$end\n";
  os << "$timescale 1ns $end\n";
}

void VcdModelTracer::initSignalTable(const ArcTraceModelInfo *info) {
  signalTable.reserve(info->numTraceTaps);
  for (uint64_t i = 0; i < info->numTraceTaps; ++i) {
    signalTable.emplace_back(VCDSignalTableEntry(i, info->stateOffsets[i],
                                                 info->typeDescriptors[i]));
  }
}

void VcdModelTracer::createHierarchy(const ArcTraceModelInfo *info) {
  std::string vcdHdr;
  auto *sigNameBlob = info->signalNames;
  auto nameRefVector = std::vector<SignalNameRef>();
  uint64_t sigOffset = 0;
  uint64_t sigOffsetNext = info->tapNameOffsets[0];
  uint64_t signalIndex = 0;
  while (signalIndex < info->numTraceTaps) {
    nameRefVector.emplace_back(signalIndex, sigOffset);
    auto nameLen = nameRefVector.back().getStringView(sigNameBlob).length();
    sigOffset += nameLen + 1;
    if (sigOffset >= sigOffsetNext) {
      assert(sigOffset == sigOffsetNext);
      ++signalIndex;
      if (signalIndex < info->numTraceTaps)
        sigOffsetNext = info->tapNameOffsets[signalIndex];
    }
  }

  std::sort(nameRefVector.begin(), nameRefVector.end(),
            [sigNameBlob](const SignalNameRef &a, const SignalNameRef &b) {
              return a.getStringView(sigNameBlob) <
                     b.getStringView(sigNameBlob);
            });

  std::vector<std::string_view> scopeStack;
  for (auto &name : nameRefVector) {
    auto nameStr = name.getStringView(sigNameBlob);
    std::vector<std::string_view> sigScope;
    size_t charOffset = 0;
    auto newOffset = std::string::npos;
    while ((newOffset = nameStr.find('/', charOffset)) != std::string::npos) {
      sigScope.push_back(nameStr.substr(charOffset, newOffset - charOffset));
      charOffset = newOffset + 1;
    }
    unsigned commonScopes = 0;
    for (size_t i = 0; i < std::min(scopeStack.size(), sigScope.size()); ++i) {
      if (sigScope[i] == scopeStack[i])
        ++commonScopes;
      else
        break;
    }

    while (commonScopes < scopeStack.size()) {
      scopeStack.pop_back();
      vcdHdr.append(scopeStack.size() + 1, ' ');
      vcdHdr.append("$upscope $end\n");
    }
    while (sigScope.size() > scopeStack.size()) {
      vcdHdr.append(scopeStack.size() + 1, ' ');
      vcdHdr.append("$scope module ");
      vcdHdr.append(sigScope[scopeStack.size()]);
      vcdHdr.append(" $end\n");
      scopeStack.push_back(sigScope[scopeStack.size()]);
    }
    vcdHdr.append(scopeStack.size() + 1, ' ');
    vcdHdr.append("$var wire ");
    vcdHdr.append(std::to_string(signalTable[name.signalIndex].typeDescriptor));
    vcdHdr.append(" ");
    vcdHdr.append(signalTable[name.signalIndex].signalId.cStr());
    vcdHdr.append(" ");
    vcdHdr.append(nameStr.substr(charOffset));
    vcdHdr.append(" $end\n");

    outFile << vcdHdr;
    vcdHdr.clear();
  }
}

unsigned
VCDSignalTableEntry::dumpSignalFromTraceBuffer(std::basic_ostream<char> &os,
                                               const uint8_t *buffer) const {
  assert(typeDescriptor > 0);
  if (typeDescriptor > 1)
    os << 'b';
  for (unsigned n = typeDescriptor; n > 0; --n)
    os << (buffer[(n - 1) / 8] & (1 << ((n - 1) % 8)) ? '1' : '0');
  if (typeDescriptor > 1)
    os << ' ';
  os << signalId.cStr() << "\n";
  return (typeDescriptor % 64 == 0) ? (typeDescriptor / 64)
                                    : (typeDescriptor / 64) + 1;
}

void VCDSignalTableEntry::dumpSignalFromState(std::basic_ostream<char> &os,
                                              const uint8_t *statePtr) const {
  const uint8_t *u8Ptr = statePtr + stateOffset;
  dumpSignalFromTraceBuffer(os, u8Ptr);
}

bool VcdModelTracer::init(const ArcTraceModelInfo *modelInfo) {
  assert(modelInfo->numTraceTaps > 0);
#ifdef ARC_JIT_VCD_TIMING
  timing.modelCreation = VCDTracerTiming::getNow();
#endif
  timestep = 0;
  std::string filename(modelInfo->modelName);
  filename += ".vcd";
  outFile.open(filename, std::ios::out | std::ios::trunc);
  if (!outFile.is_open()) {
    std::cerr << "VCD-Trace: Unable to open trace file " << filename
              << " for writing" << std::endl;
    return false;
  }

#ifdef ARC_JIT_VCD_TIMING
  timing.vcdInitStart = VCDTracerTiming::getNow();
#endif

  initSignalTable(modelInfo);
  writeVCDHeader(outFile);
  outFile << "$scope module " << modelInfo->modelName << " $end\n";
  createHierarchy(modelInfo);
  outFile << "$upscope $end\n";
  outFile << "$enddefinitions $end\n";

#ifdef ARC_JIT_VCD_TIMING
  timing.vcdInitEnd = VCDTracerTiming::getNow();
#endif

  return true;
}

void *VcdModelTracer::swapBuffer(void *oldBuffer, uint64_t oldBufferSize) {
  // TODO: Proper buffer handling.
  // For now we just create one buffer, hope it's large enough,
  // and flush it every step.
  assert(false && "Trace buffer overflow.");
  return nullptr;
}

void VcdModelTracer::close(ArcTracerState *state) {
#ifdef ARC_JIT_VCD_TIMING
  timing.stepEnd();
#endif

  if (outFile.is_open()) {
    // Dump the last step
    if (state->size > 0) {
      outFile << "#" << timestep << "\n";
      processBuffer(static_cast<uint64_t *>(state->buffer), state->size);
    }
    state->size = 0;
    ++timestep;
    outFile << "#" << timestep << "\n";
  }

  if (outFile.is_open())
    outFile.close();

#ifdef ARC_JIT_VCD_TIMING
  timing.modelDeletion = VCDTracerTiming::getNow();
  timing.dump();
#endif

  if (state->buffer != nullptr) {
    delete[] static_cast<uint64_t *>(state->buffer);
    state->buffer = nullptr;
    state->capacity = 0;
  }
}

void VcdModelTracer::processBuffer(const uint64_t *traceBuffer,
                                   uint64_t bufferSize) {
  assert(bufferSize > 1);
#ifdef ARC_JIT_VCD_TIMING
  timing.vcdProcessStart();
#endif
  uint64_t offset = 0;
  uint64_t numChanges = 0;
  while (offset < bufferSize) {
    auto idx = traceBuffer[offset];
    ++offset;
    const void *voidPtr = static_cast<const void *>(traceBuffer + offset);
    ++numChanges;
    auto stride = signalTable[idx].dumpSignalFromTraceBuffer(
        outFile, static_cast<const uint8_t *>(voidPtr));
    offset += stride;
    assert(offset <= bufferSize);
  }
  assert(offset == bufferSize);
#ifdef ARC_JIT_VCD_TIMING
  timing.vcdProcessEnd();
  timing.signalsChanged(numChanges);
#else
  (void)numChanges;
#endif
}

void VcdModelTracer::step(ArcTracerState *state) {
#ifdef ARC_JIT_VCD_TIMING
  timing.stepEnd();
#endif

  if (state->buffer == nullptr) {
    state->buffer = new uint64_t[1024 * 64];
    state->capacity = 1024 * 64;
  }

  // Dump the changes of the previous time step.
  // Do a full dump on the initial step.
  if (timestep == 0) {
    outFile << "#0\n";
    for (auto sig : signalTable)
      sig.dumpSignalFromState(outFile, state->simulationState);
  } else if (state->size > 0) {
    outFile << "#" << timestep << "\n";
    processBuffer(static_cast<uint64_t *>(state->buffer), state->size);
    state->size = 0;
  }
#ifdef ARC_JIT_VCD_TIMING
  timing.stepStart();
#endif
  ++timestep;
}

#ifdef ARC_JIT_VCD_TIMING
void VCDTracerTiming::dump() {
  std::cerr << "\n --- VCDTracerTiming Dump --- \n";
  if (!modelCreation.valid || step < 0) {
    std::cerr << "(Empty)\n";
    return;
  }

  std::cerr << "Total Simulation Steps\t\t" << std::setw(16) << (step + 1)
            << "\n";
  if (step > 7) {
    auto numEvenSteps = static_cast<double>((step - 5) / 2);
    auto numOddSteps = static_cast<double>((step - 6) / 2);
    std::cerr << "Average Even Changes\t\t" << std::setw(16)
              << (evenChanges / numEvenSteps) << "\n";
    std::cerr << "Average Odd  Changes\t\t" << std::setw(16)
              << (oddChanges / numOddSteps) << "\n";
  }
  std::cerr << "Total lifetime\t\t\t" << std::setw(16)
            << diffNs(modelCreation, modelDeletion) << " ns\n";

  if (vcdInitEnd.valid) {
    assert(vcdInitStart.valid);
    std::cerr << "VCD Tracer Init time\t\t" << std::setw(16)
              << diffNs(vcdInitStart, vcdInitEnd) << " ns\n";
  }

  std::cerr << "Buffer processing time\t\t" << std::setw(16)
            << totalVcdTime.count() << " ns\n";
  std::cerr << "Total Sim time\t\t\t" << std::setw(16) << totalSimTime.count()
            << " ns\n";

  auto averageTotal = totalSimTime;
  for (unsigned i = 0; i < startStepStamps.size(); ++i) {
    if (!endStepStamps[i].valid)
      break;
    assert(startStepStamps[i].valid);
    averageTotal -= (endStepStamps[i].tp - startStepStamps[i].tp);
    std::cerr << "Simulation Step " << std::setw(2) << i << "\t\t"
              << std::setw(16) << diffNs(startStepStamps[i], endStepStamps[i])
              << " ns\n";
  }
  if (step >= static_cast<int64_t>(startStepStamps.size())) {
    auto avg = averageTotal.count() / (step + 1 - startStepStamps.size());
    std::cerr << "Remaining Average\t\t" << std::setw(16) << avg << " ns\n";
  }

  std::cerr << "\n";
}
#endif