// NOLINTBEGIN
#pragma once
#include <array>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <ostream>
#include <vector>

// Sanity checks for binary compatibility
#ifdef __BYTE_ORDER__
#if (__BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__)
#error Unsupported endianess
#endif
#endif
static_assert(sizeof(int) == 4, "Unsupported ABI");
static_assert(sizeof(long long) == 8, "Unsupported ABI");

// ---  Exports to the IR ---

#ifdef _WIN32
#define ARC_EXPORT extern "C" __declspec(dllexport)
#else
#define ARC_EXPORT extern "C" __attribute__((visibility("default")))
#endif

#ifndef ARC_NO_LIBC_EXPORTS

// libc Adapters
ARC_EXPORT int _arc_libc_fprintf(FILE *stream, const char *format, ...) {
  int result;
  va_list args;
  va_start(args, format);
  result = vfprintf(stream, format, args);
  va_end(args);
  return result;
}

ARC_EXPORT int _arc_libc_fputs(const char *str, FILE *stream) {
  return fputs(str, stream);
}

ARC_EXPORT int _arc_libc_fputc(int ch, FILE *stream) {
  return fputc(ch, stream);
}

#endif // ARC_NO_LIBC_EXPORTS

// Runtime Environment calls

#define ARC_ENV_DECL_GET_PRINT_STREAM(idarg)                                   \
  ARC_EXPORT FILE *_arc_env_get_print_stream(uint32_t idarg)

#ifndef ARC_NO_DEFAULT_GET_PRINT_STREAM
ARC_ENV_DECL_GET_PRINT_STREAM(id) {
  (void)id;
  return stderr;
}
#endif // ARC_NO_DEFAULT_GET_PRINT_STREAM

// Handler for out-of-bounds index on an array_get operation
// base:    Array base address
// size:    Number of array elements
// eltBits: Number of bits per element
// oobAddr: Referenced address outside of array bounds
// oobIdx:  Out-of-bounds index
// Return value: Pointer to new the result value of the array_get operation
#define ARC_HANDLER_ARRAY_GET_OOB_HANDLER(base, size, eltBits, oobAddr,        \
                                          oobIdx)                              \
  const void *_arc_env_array_get_oob_handler(                                  \
      const void *base, uint64_t size, uint32_t eltBits, const void *oobAddr,  \
      uint64_t oobIdx)

#ifndef ARC_NO_DEFAULT_ARRAY_GET_OOB_HANDLER
#include <iostream>
extern "C" {
ARC_HANDLER_ARRAY_GET_OOB_HANDLER(base, as, eb, oa, oi) {
  (void)eb;
  (void)oa;
  std::cerr << "ARCENV-WARNING: Out-of-bounds array access caught: Index = "
            << oi << ", Size = " << as << ", Base = " << base << std::endl;
  return base;
}
}
#endif // ARC_NO_DEFAULT_ARRAY_GET_OOB_HANDLER

// ----------------

struct Signal {
  const char *name;
  unsigned offset;
  unsigned numBits;
  enum Type { Input, Output, Register, Memory, Wire } type;
  // for memories:
  unsigned stride;
  unsigned depth;
};

struct Hierarchy {
  const char *name;
  unsigned numStates;
  unsigned numChildren;
  Signal *states;
  Hierarchy *children;
};

template <unsigned N>
struct Bytes {
  uint8_t byte[N];
};
template <typename T, unsigned Stride, unsigned Depth>
struct Memory {
  union {
    T data;
    uint8_t stride[Stride];
  } words[Depth];
};

template <class ModelLayout>
class ValueChangeDump {
public:
  ValueChangeDump(std::basic_ostream<char> &os, const uint8_t *state)
      : os(os), state(state) {}

  void writeHeader(bool withHierarchy = true) {
    os << "$date\n    October 21, 2015\n$end\n";
    os << "$version\n    Some cryptic MLIR magic\n$end\n";
    os << "$timescale 1ns $end\n";

    os << "$scope module " << ModelLayout::name << " $end\n";

    auto writeSignal = [&](const Signal &state) {
      if (state.type != Signal::Memory) {
        auto &signal =
            allocSignal(state, state.offset, (state.numBits + 7) / 8);
        if (state.type == Signal::Register) {
          os << "$var reg " << state.numBits << " " << signal.abbrev << " "
             << state.name;
        } else {
          os << "$var wire " << state.numBits << " " << signal.abbrev << " "
             << state.name;
        }
        if (state.numBits > 1)
          os << " [" << (state.numBits - 1) << ":0]";
        os << " $end\n";
      } else {
        for (unsigned i = 0; i < state.depth; ++i) {
          auto &signal = allocSignal(state, state.offset + i * state.stride,
                                     (state.numBits + 7) / 8);
          os << "$var reg " << state.numBits << " " << signal.abbrev << " "
             << state.name << "[" << i << "]";
          if (state.numBits > 1)
            os << " [" << (state.numBits - 1) << ":0]";
          os << " $end\n";
        }
      }
    };

    std::function<void(const Hierarchy &)> writeHierarchy =
        [&](const Hierarchy &hierarchy) {
          os << "$scope module " << hierarchy.name << " $end\n";
          for (unsigned i = 0; i < hierarchy.numStates; ++i)
            writeSignal(hierarchy.states[i]);
          for (unsigned i = 0; i < hierarchy.numChildren; ++i)
            writeHierarchy(hierarchy.children[i]);
          os << "$upscope $end\n";
        };

    for (auto &port : ModelLayout::io)
      writeSignal(port);
    if (withHierarchy)
      writeHierarchy(ModelLayout::hierarchy);

    os << "$upscope $end\n";
    os << "$enddefinitions $end\n";
  }

  void writeValues(bool includeUnchanged = false) {
    for (auto &signal : signals) {
      const uint8_t *valNew = state + signal.offset;
      uint8_t *valOld = &previousValues[0] + signal.previousOffset;
      size_t numBytes = (signal.state.numBits + 7) / 8;
      bool unchanged = std::equal(valNew, valNew + numBytes, valOld);
      if (unchanged && !includeUnchanged)
        continue;
      if (signal.state.numBits > 1)
        os << 'b';
      for (unsigned n = signal.state.numBits; n > 0; --n)
        os << (valNew[(n - 1) / 8] & (1 << ((n - 1) % 8)) ? '1' : '0');
      if (signal.state.numBits > 1)
        os << ' ';
      os << signal.abbrev << "\n";
      std::copy(valNew, valNew + numBytes, valOld);
    }
  }

  void writeDumpvars() {
    os << "$dumpvars\n";
    writeValues(true);
  }

  void writeTimestep(size_t timeIncrement) {
    time += timeIncrement;
    os << "#" << time << "\n";
    writeValues();
  }

  size_t time = 0;

private:
  struct VcdSignal {
    std::string abbrev;
    unsigned offset;
    const Signal &state;
    unsigned previousOffset;
  };

  VcdSignal &allocSignal(const Signal &state, unsigned offset,
                         unsigned numBytes) {
    std::string abbrev;
    unsigned rest = signals.size() + 1;
    while (rest != 0) {
      uint8_t c = (rest % 84) + 33;
      if (c >= '0')
        c += 10;
      abbrev += c;
      rest /= 84;
    }
    signals.push_back(
        VcdSignal{abbrev, offset, state, unsigned(previousValues.size())});
    previousValues.resize(previousValues.size() + numBytes);
    return signals.back();
  }

  std::basic_ostream<char> &os;
  const uint8_t *state;
  std::vector<VcdSignal> signals;
  std::vector<uint8_t> previousValues;
};

// NOLINTEND
