//===- SVRuntime.cpp - SystemVerilog execution runtime --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the SystemVerilog execution runtime. See SVRuntime.h. Functions
// here are plain `extern "C"` leaf utilities; the JIT-binding table at the
// bottom is only compiled into the copy linked into arcilator
// (ARC_RUNTIME_JIT_BIND).
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/Runtime/SVRuntime.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" int32_t circt_sv_string_len(const char *str) {
  if (!str)
    return 0;
  return static_cast<int32_t>(std::strlen(str));
}

extern "C" int32_t circt_sv_strcmp(const char *lhs, const char *rhs) {
  if (!lhs)
    lhs = "";
  if (!rhs)
    rhs = "";
  return std::strcmp(lhs, rhs);
}

extern "C" uint8_t circt_sv_string_getc(const char *str, int32_t idx) {
  if (!str || idx < 0)
    return 0;
  size_t len = std::strlen(str);
  size_t pos = static_cast<size_t>(idx);
  if (pos >= len)
    return 0;
  return static_cast<uint8_t>(static_cast<unsigned char>(str[pos]));
}

extern "C" const char *circt_sv_string_substr(const char *str, int32_t start,
                                              int32_t end) {
  if (!str)
    str = "";
  if (start < 0)
    start = 0;
  if (end < start)
    return "";

  size_t len = std::strlen(str);
  size_t startPos = static_cast<size_t>(start);
  if (startPos >= len)
    return "";

  size_t endPos = static_cast<size_t>(end);
  if (endPos >= len)
    endPos = len - 1;

  size_t outLen = endPos - startPos + 1;
  char *out = static_cast<char *>(std::malloc(outLen + 1));
  if (!out)
    return "";
  std::memcpy(out, str + startPos, outLen);
  out[outLen] = '\0';
  return out;
}

//===----------------------------------------------------------------------===//
// Integer formatting (`$display`-style)
//===----------------------------------------------------------------------===//

namespace {

// Backend-defined formatting flags; keep in sync with the console lowering.
constexpr int32_t kFmtUppercase = 1 << 0;
constexpr int32_t kFmtLeftJustify = 1 << 1;
constexpr int32_t kFmtPadZero = 1 << 2;
constexpr int32_t kFmtSigned = 1 << 3;

uint8_t getBitLE(const uint8_t *bytes, uint32_t bitIndex) {
  return (bytes[bitIndex / 8] >> (bitIndex % 8)) & 1u;
}

uint32_t getBitsLEMasked(const uint8_t *bytes, uint32_t bitWidth,
                         uint32_t bitIndex, uint32_t bitCount) {
  uint32_t out = 0;
  for (uint32_t i = 0; i < bitCount; ++i) {
    uint32_t b = bitIndex + i;
    if (b >= bitWidth)
      break;
    out |= static_cast<uint32_t>(getBitLE(bytes, b)) << i;
  }
  return out;
}

// Emit `out` to stdout with the requested minimum field width / justification,
// after trimming the leading zeros produced by the per-base digit loops.
void emitPadded(std::string &out, int32_t minWidth, bool leftJustify,
                char padChar) {
  size_t firstNonZero = out.find_first_not_of('0');
  if (firstNonZero == std::string::npos)
    out.erase(0, out.size() - 1);
  else if (firstNonZero > 0)
    out.erase(0, firstNonZero);

  int32_t padCount = 0;
  if (minWidth > 0 && static_cast<int32_t>(out.size()) < minWidth)
    padCount = minWidth - static_cast<int32_t>(out.size());
  if (!leftJustify)
    for (int32_t i = 0; i < padCount; ++i)
      std::fputc(padChar, stdout);
  std::fwrite(out.data(), 1, out.size(), stdout);
  if (leftJustify)
    for (int32_t i = 0; i < padCount; ++i)
      std::fputc(padChar, stdout);
}

} // namespace

extern "C" void circt_sv_print_int(const void *data, int32_t bitWidth,
                                   int32_t base, int32_t minWidth,
                                   int32_t flags) {
  if (bitWidth < 0)
    bitWidth = 0;
  if (minWidth < 0)
    minWidth = 0;

  const bool uppercase = (flags & kFmtUppercase) != 0;
  const bool leftJustify = (flags & kFmtLeftJustify) != 0;
  const bool padZero = (flags & kFmtPadZero) != 0;
  const bool isSigned = (flags & kFmtSigned) != 0;
  char padChar = padZero ? '0' : ' ';

  // Handle zero-width integers.
  if (bitWidth == 0) {
    if (base == 10)
      std::fputc('0', stdout);
    return;
  }

  uint32_t bw = static_cast<uint32_t>(bitWidth);
  auto numBytes = static_cast<uint32_t>((bw + 7u) / 8u);
  const auto *bytes = static_cast<const uint8_t *>(data);
  if (!bytes) {
    static const uint8_t zero = 0;
    bytes = &zero;
    numBytes = 1;
  }

  std::string out;
  switch (base) {
  case 2:
    out.reserve(bw);
    for (uint32_t b = bw; b > 0; --b)
      out.push_back(getBitLE(bytes, b - 1) ? '1' : '0');
    break;
  case 8: {
    uint32_t digits = (bw + 2u) / 3u;
    out.reserve(digits);
    for (uint32_t d = digits; d > 0; --d)
      out.push_back(static_cast<char>(
          '0' + getBitsLEMasked(bytes, bw, (d - 1) * 3u, 3u)));
    break;
  }
  case 16: {
    uint32_t digits = (bw + 3u) / 4u;
    out.reserve(digits);
    for (uint32_t d = digits; d > 0; --d) {
      uint32_t digit = getBitsLEMasked(bytes, bw, (d - 1) * 4u, 4u);
      if (digit < 10)
        out.push_back(static_cast<char>('0' + digit));
      else
        out.push_back(
            static_cast<char>((uppercase ? 'A' : 'a') + (digit - 10)));
    }
    break;
  }
  case 10: {
    uint32_t limbs = (bw + 31u) / 32u;
    std::vector<uint32_t> value(limbs, 0);
    for (uint32_t i = 0; i < limbs; ++i) {
      uint32_t limb = 0;
      for (uint32_t b = 0; b < 4; ++b) {
        uint32_t byteIdx = i * 4u + b;
        if (byteIdx < numBytes)
          limb |= static_cast<uint32_t>(bytes[byteIdx]) << (8u * b);
      }
      value[i] = limb;
    }
    auto maskTop = [&]() {
      if (bw % 32u) {
        uint32_t bitsInTop = bw % 32u;
        value.back() &= (1u << bitsInTop) - 1u;
      }
    };
    maskTop();

    bool neg = isSigned && getBitLE(bytes, bw - 1u) != 0;
    if (neg) {
      for (auto &limb : value)
        limb = ~limb;
      maskTop();
      uint64_t carry = 1;
      for (auto &limb : value) {
        uint64_t sum = static_cast<uint64_t>(limb) + carry;
        limb = static_cast<uint32_t>(sum);
        carry = sum >> 32;
        if (!carry)
          break;
      }
    }

    auto isZero = [&]() {
      for (auto limb : value)
        if (limb != 0)
          return false;
      return true;
    };

    std::string digits;
    if (isZero()) {
      digits = "0";
    } else {
      while (!isZero()) {
        uint64_t rem = 0;
        for (uint32_t i = value.size(); i > 0; --i) {
          uint64_t cur = (rem << 32) | value[i - 1];
          value[i - 1] = static_cast<uint32_t>(cur / 10);
          rem = cur % 10;
        }
        digits.push_back(static_cast<char>('0' + rem));
      }
      std::reverse(digits.begin(), digits.end());
    }

    if (neg)
      out.push_back('-');
    out.append(digits);
    break;
  }
  default:
    std::fputs("<unsupported base>", stdout);
    return;
  }

  emitPadded(out, minWidth, leftJustify, padChar);
}

extern "C" void circt_sv_print_fvint(const void *valueData,
                                     const void *unknownData, int32_t bitWidth,
                                     int32_t base, int32_t minWidth,
                                     int32_t flags) {
  if (bitWidth < 0)
    bitWidth = 0;
  if (minWidth < 0)
    minWidth = 0;

  const bool uppercase = (flags & kFmtUppercase) != 0;
  const bool leftJustify = (flags & kFmtLeftJustify) != 0;
  const bool padZero = (flags & kFmtPadZero) != 0;
  char padChar = padZero ? '0' : ' ';

  // Handle zero-width integers.
  if (bitWidth == 0) {
    if (base == 10)
      std::fputc('0', stdout);
    return;
  }

  uint32_t bw = static_cast<uint32_t>(bitWidth);
  uint32_t numBytes = static_cast<uint32_t>((bw + 7u) / 8u);
  const auto *valueBytes = static_cast<const uint8_t *>(valueData);
  const auto *unknownBytes = static_cast<const uint8_t *>(unknownData);
  static const uint8_t zero = 0;
  if (!valueBytes)
    valueBytes = &zero;
  if (!unknownBytes)
    unknownBytes = &zero;

  bool anyUnknown = false;
  for (uint32_t i = 0; i < numBytes; ++i)
    if (unknownBytes[i] != 0) {
      anyUnknown = true;
      break;
    }

  // Fast path: no unknown bits, so it formats like a two-state value.
  if (!anyUnknown) {
    circt_sv_print_int(valueBytes, bitWidth, base, minWidth, flags);
    return;
  }

  std::string out;
  switch (base) {
  case 2:
    out.reserve(bw);
    for (uint32_t b = bw; b > 0; --b) {
      if (!getBitLE(unknownBytes, b - 1)) {
        out.push_back(getBitLE(valueBytes, b - 1) ? '1' : '0');
      } else {
        bool isZ = getBitLE(valueBytes, b - 1) != 0;
        out.push_back(isZ ? (uppercase ? 'Z' : 'z') : (uppercase ? 'X' : 'x'));
      }
    }
    break;
  case 8:
  case 16: {
    const uint32_t groupBits = base == 8 ? 3u : 4u;
    uint32_t digits = (bw + (groupBits - 1u)) / groupBits;
    out.reserve(digits);
    for (uint32_t d = digits; d > 0; --d) {
      uint32_t startBit = (d - 1u) * groupBits;
      uint32_t bitsThisDigit =
          std::min(groupBits, bw > startBit ? (bw - startBit) : 0u);
      if (bitsThisDigit == 0) {
        out.push_back('0');
        continue;
      }
      uint32_t digitUnknown =
          getBitsLEMasked(unknownBytes, bw, startBit, groupBits);
      if (digitUnknown == 0) {
        uint32_t digitVal =
            getBitsLEMasked(valueBytes, bw, startBit, groupBits);
        if (digitVal < 10)
          out.push_back(static_cast<char>('0' + digitVal));
        else
          out.push_back(
              static_cast<char>((uppercase ? 'A' : 'a') + (digitVal - 10)));
        continue;
      }
      uint32_t mask =
          bitsThisDigit == 32u ? 0xFFFFFFFFu : ((1u << bitsThisDigit) - 1u);
      uint32_t digitVal =
          getBitsLEMasked(valueBytes, bw, startBit, groupBits) & mask;
      digitUnknown &= mask;
      // A fully-unknown digit preserves `z` when all bits are Z.
      if (digitUnknown == mask) {
        if (digitVal == mask) {
          out.push_back(uppercase ? 'Z' : 'z');
          continue;
        }
        if (digitVal == 0) {
          out.push_back(uppercase ? 'X' : 'x');
          continue;
        }
      }
      out.push_back(uppercase ? 'X' : 'x');
    }
    break;
  }
  case 10: {
    // Decimal with any unknown bit prints `x` (or `z` if every bit is Z).
    bool allUnknown = true;
    bool allZ = true;
    for (uint32_t b = 0; b < bw; ++b) {
      if (!getBitLE(unknownBytes, b)) {
        allUnknown = false;
        break;
      }
      if (getBitLE(valueBytes, b) == 0)
        allZ = false;
    }
    std::fputc(allUnknown && allZ ? (uppercase ? 'Z' : 'z')
                                  : (uppercase ? 'X' : 'x'),
               stdout);
    return;
  }
  default:
    std::fputs("<unsupported base>", stdout);
    return;
  }

  emitPadded(out, minWidth, leftJustify, padChar);
}

#ifdef ARC_RUNTIME_JIT_BIND
namespace circt {
namespace arc {
namespace runtime {

static const SVRuntimeSymbol svRuntimeSymbols[] = {
    {"circt_sv_string_len", reinterpret_cast<void (*)()>(&circt_sv_string_len)},
    {"circt_sv_strcmp", reinterpret_cast<void (*)()>(&circt_sv_strcmp)},
    {"circt_sv_string_getc",
     reinterpret_cast<void (*)()>(&circt_sv_string_getc)},
    {"circt_sv_string_substr",
     reinterpret_cast<void (*)()>(&circt_sv_string_substr)},
    {"circt_sv_print_int", reinterpret_cast<void (*)()>(&circt_sv_print_int)},
    {"circt_sv_print_fvint",
     reinterpret_cast<void (*)()>(&circt_sv_print_fvint)},
    {nullptr, nullptr},
};

const SVRuntimeSymbol *getSVRuntimeSymbols() { return svRuntimeSymbols; }

} // namespace runtime
} // namespace arc
} // namespace circt
#endif // ARC_RUNTIME_JIT_BIND
