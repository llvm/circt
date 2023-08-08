//===- HWRLELogic.h - Run-length encoded logic ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the RLELogic class, a compressed container for arbitrarily
// long sequences of multi-valued logic digits.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_HWRLELOGIC_H
#define CIRCT_DIALECT_HW_HWRLELOGIC_H

#include "circt/Dialect/HW/HWLogicDigits.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstring>

namespace circt {
namespace hw {

/// Container for sequences of multi-valued logic digits.
/// RLELogic uses run-length encoding to reduce the encoded size of values
/// containing long stretches of identical digits.
/// The run-length of the most siginificant encoded digit is always implicitly
/// indefinite. Thus, an RLELogic always has either infinite length
/// or zero length for special non-value encodings. E.g., the RLELogic values
/// "[msb] ...XXXXXXXXL01 [lsb]" and "[msb] ...XL01 [lsb]" are equal.
/// Encodings that do not require more than eight bytes are stored directly
/// in the instance. Larger buffers are allocated on the heap.
/// RLELogic also stores a bit mask indicating which digits are present at least
/// once in the encoded value, allowing for shortcuts on certain operations.

class RLELogic {
public:
  using RunLengthCode = uint8_t;
  using RawType = uintptr_t;
  using SizeType = unsigned int;

  /// Tag type to discriminate between various (zero-length) non-value encodings
  enum InvalidTag : RawType {
    Deleted = 0,  // Value has been deallocated or moved
    ZeroLength,   // Encoding of a zero-length logic value
    InvalidDigit, // Encoded value contained at least one invalid digit

    KeyTombstone = ~((RawType)1), // DenseMap tombstone key
    KeyEmpty = ~((RawType)0)      // DenseMap empty key
  };

  /// Container for the results of a pre-encoding analysis run
  struct AnalyzePassResult {
    SizeType requiredBytes = 0;
    SizeType lastDigitTrim = 0;
    uint16_t foundDigitsMask = 0;

    bool isInteger() const {
      return containsOnly(foundDigitsMask, logicdigits::LogicDigit::LD_0,
                          logicdigits::LogicDigit::LD_1);
    };

    bool isValid() const { return requiredBytes != 0; };
  };

  /// Container for a digit offset within an RLELogic instance
  struct Offset {
    SizeType bytes = 0;
    SizeType runLength = 0;
  };

  /// Deleted default constructor
  RLELogic() = delete;

  /// Copy constructor
  RLELogic(const RLELogic &rlelog) {
    if (rlelog.isSelfContained()) {
      valPtrUnion.raw = rlelog.valPtrUnion.raw;
    } else {
      valPtrUnion.ptr = new RunLengthCode[rlelog.byteCount];
      memcpy(valPtrUnion.ptr, rlelog.valPtrUnion.ptr, rlelog.byteCount);
    }
    byteCount = rlelog.byteCount;
    digitMask = rlelog.digitMask;
  }

  /// Move constructor
  RLELogic(RLELogic &&rlelog)
      : valPtrUnion({rlelog.valPtrUnion.raw}), byteCount(rlelog.byteCount),
        digitMask(rlelog.digitMask) {
    rlelog.invalidate(false);
  }

  /// Constructor for zero-length encodings
  explicit RLELogic(InvalidTag invalidTag)
      : valPtrUnion({(RawType)invalidTag}), byteCount(0), digitMask(0){};

  /// Destructor
  ~RLELogic() { invalidate(true); }

  /// Returns an RLELogic filled entirely with the logic digit represented
  /// by the given character
  template <char C>
  typename std::enable_if<
      logicdigits::isValidLogicDigit(logicdigits::charToLogicDigit(C)),
      RLELogic>::type static filled() {
    return RLELogic(logicdigits::charToLogicDigit(C));
  };

  /// Returns an RLELogic filled entirely with the  given logic digit
  static RLELogic filled(logicdigits::LogicDigit logdigit) {
    if (!isValidLogicDigit(logdigit))
      return RLELogic(InvalidTag::Deleted);
    return RLELogic(logdigit);
  }

  /// Copy assignment operator
  RLELogic &operator=(const RLELogic &rlelog) {
    if (this == &rlelog)
      return *this;

    RLELogic rleCopy(rlelog);
    swap(rleCopy);
    return *this;
  }

  /// Move assignment operator
  RLELogic &operator=(RLELogic &&rlelog) {
    if (this == &rlelog)
      return *this;

    invalidate(true);
    swap(rlelog);
    return *this;
  }

  /// Equals operator. Two RLELogic instances are equal iff they
  /// encode the same infinite-length sequence of logic digits or
  /// the same zero-length non-value encoding.
  bool operator==(const RLELogic &rlelog) const {
    if (byteCount != rlelog.byteCount)
      return false;
    if (isSelfContained())
      return valPtrUnion.raw == rlelog.valPtrUnion.raw;
    return memcmp(valPtrUnion.ptr, rlelog.valPtrUnion.ptr, byteCount) == 0;
  }

  /// Not-equals operator
  bool operator!=(const RLELogic &rlelog) const { return !(*this == rlelog); }

  /// Helper to get the run-length specified by the given code byte
  static uint8_t getRunLength(RunLengthCode code) { return (code >> 4) + 1; }

  /// Helper to get the logic digit specified by the given code byte
  static logicdigits::LogicDigit getDigit(RunLengthCode code) {
    return static_cast<logicdigits::LogicDigit>(code & 0xF);
  }

  /// Returns true iff the current instance contains an infinite length logic
  /// value
  bool isValid() const { return byteCount != 0; }

  /// Retuns true iff the encoded value contains any of the given character-
  /// represented logic digits at least once.
  template <char... Cs>
  typename std::enable_if<
      (logicdigits::isValidLogicDigit(logicdigits::charToLogicDigit(Cs)) &&
       ...),
      bool>::type
  containsAny() const {
    if (!isValid())
      return false;
    uint16_t mask = (makeDigitMask(logicdigits::charToLogicDigit(Cs)) | ...);
    return (digitMask & mask) != 0;
  }

  /// Retuns true iff the encoded value contains any of the given logic digits
  /// at least once.
  template <typename... Args>
  static bool containsAny(uint16_t digMask, logicdigits::LogicDigit logDig,
                          Args... args) {
    return (digMask & makeDigitMask(logDig, args...)) != 0;
  };

  /// Retuns true iff the encoded value contains no other than the given
  /// character-represented logic digits.
  template <char... Cs>
  typename std::enable_if<
      (logicdigits::isValidLogicDigit(logicdigits::charToLogicDigit(Cs)) &&
       ...),
      bool>::type
  containsOnly() const {
    if (!isValid())
      return false;
    uint16_t mask = (makeDigitMask(logicdigits::charToLogicDigit(Cs)) | ...);
    return (digitMask & ~mask) == 0;
  }

  /// Retuns true iff the encoded value contains no other than the given logic
  /// digits.
  template <typename... Args>
  static bool containsOnly(uint16_t digMask, logicdigits::LogicDigit logDig,
                           Args... args) {
    return (digMask & ~makeDigitMask(logDig, args...)) == 0;
  };

  /// Retuns true iff the encoded value is an integer. I.e., it contains only
  /// '0' or '1' digits.
  bool isInteger() const { return containsOnly<'0', '1'>(); }

  /// Retuns true iff the encoded value is integer-like. I.e., it contains only
  /// '0' 'L', 'H' or '1' digits.
  bool isIntegerLike() const { return containsOnly<'0', '1', 'L', 'H'>(); }

  /// Retuns true iff the encoded value contains at least one digit representing
  /// an unknown state.
  bool containsAnyUnknownDigits() const {
    return containsAny<'U', 'X', 'W', 'Z', '-'>();
  }

  /// Retuns true iff the encoded value contains only digits representing
  /// an unknown state.
  bool containsOnlyUnknownDigits() const {
    return containsOnly<'U', 'X', 'W', 'Z', '-'>();
  }

  /// Returns a pointer to the run-length encoded buffer
  const RunLengthCode *getCodePointer() const {
    return isSelfContained() ? valPtrUnion.digits : valPtrUnion.ptr;
  }

  /// Returns the size of the encoded value in bytes or zero for non-values.
  SizeType getByteCount() const { return byteCount; }

  /// Returns a bit mask indicating wich logic digits are present at least once
  /// in the encoded value
  uint16_t getDigitMask() const { return digitMask; }

  /// Runs an analysis pass on a sequence of logic digits as preparation for
  /// encoding. Determines the exact number of bytes required for the encoded
  /// value.
  static AnalyzePassResult
  analyze(llvm::ArrayRef<logicdigits::LogicDigit> digits);

  /// Encode the given sequence of logic digits. If an analysis pass has already
  /// been run on the same sequence it can optionally be passed to this method.
  static RLELogic encode(llvm::ArrayRef<logicdigits::LogicDigit> digits,
                         std::optional<AnalyzePassResult> analysisResult = {});

  /// Parse an RLELogic value from a string
  static RLELogic encode(llvm::StringRef str) {
    // Could be done more nicely with rages/views in C++20
    llvm::SmallVector<logicdigits::LogicDigit, 32> digitVector;
    const auto length = str.size();
    digitVector.resize_for_overwrite(length);
    for (size_t i = 0; i < length; i++)
      digitVector[length - i - 1] = logicdigits::charToLogicDigit(str[i]);
    return encode(llvm::ArrayRef(digitVector), {});
  }

  /// Unrolls 'length' digits of the encoded value into the given buffer, LSB
  /// first.
  void unroll(logicdigits::LogicDigit *dest, size_t length) const {
    auto iter = infiniteIterator();
    for (size_t i = 0; i < length; i++) {
      dest[i] = *iter;
      iter++;
    }
  }

  /// Convert 'length' digits of the encoded value to a human-readable string
  /// representation, MSB to LSB.
  std::string toString(unsigned length) const {
    assert(isValid() && "invalid RLELogic should not be converted to string");
    if (!isValid())
      return "";
    std::string str(length, '\0');
    auto iter = infiniteIterator();
    for (unsigned i = 0; i < length; i++) {
      str[length - i - 1] = logicDigitToChar(*iter);
      iter++;
    }
    return str;
  }

  /// Applies an unary-operation LUT to 'length' digits of the given
  /// RLELogic value. The unrolled result is appended to the given buffer.
  template <unsigned N>
  static void unaryOp(const logicdigits::UnaryLogicLUT &lut,
                      llvm::SmallVector<logicdigits::LogicDigit, N> &dest,
                      unsigned length, const RLELogic &operand) {
    for (auto digit : operand.boundedIterator(length))
      dest.push_back(lut[(unsigned)digit]);
  }

  /// Applies a binary-operation LUT to 'length' digits of the given
  /// RLELogic values. The unrolled result is appended to the given buffer.
  template <unsigned N>
  static void binaryOp(const logicdigits::BinaryLogicLUT &lut,
                       llvm::SmallVector<logicdigits::LogicDigit, N> &dest,
                       unsigned length, const RLELogic &operandA,
                       const RLELogic &operandB) {
    auto iterA = operandA.infiniteIterator();
    auto iterB = operandB.infiniteIterator();
    for (unsigned i = 0; i < length; i++) {
      dest.push_back(lut[(unsigned)*iterA][(unsigned)*iterB]);
      iterA++;
      iterB++;
    }
  }

  /// Applies a binary-operation LUT operation inplace on the given buffer
  /// using 'length' digits of this instance's value as second operand.
  void binaryOpInplace(const logicdigits::BinaryLogicLUT &lut,
                       logicdigits::LogicDigit *destAndOperandA,
                       unsigned length) const {
    auto iter = infiniteIterator();
    for (unsigned i = 0; i < length; i++) {
      destAndOperandA[i] = lut[(unsigned)destAndOperandA[i]][(unsigned)*iter];
      iter++;
    }
  }

  /// Advance an offset struct by the given amount of digits
  void seek(SizeType digitSkip, Offset &offset) const;

  /// Iterator over the encoded digits with an optional bound
  struct DigitIterator
      : public llvm::iterator_facade_base<
            DigitIterator, std::forward_iterator_tag, logicdigits::LogicDigit> {

    DigitIterator() = delete;
    DigitIterator(const RunLengthCode *ptr, SizeType bytes, Offset startOffset,
                  std::optional<SizeType> limit)
        : basePtr(ptr), totalBytes(bytes), offset(startOffset),
          digitLimit(limit){};

    using llvm::iterator_facade_base<DigitIterator, std::forward_iterator_tag,
                                     logicdigits::LogicDigit>::operator++;

    logicdigits::LogicDigit operator*() const {
      if (totalBytes == 0)
        return logicdigits::LogicDigit::Invalid;
      return getDigit(basePtr[offset.bytes]);
    }

    DigitIterator &operator++() {

      if (digitLimit.has_value()) {
        assert(*digitLimit > 0 && "incrementing past end");
        (*digitLimit)--;
      }

      if (totalBytes == 0)
        return *this;

      offset.runLength++;

      if (offset.bytes == totalBytes - 1)
        return *this;

      if (offset.runLength == getRunLength(basePtr[offset.bytes])) {
        offset.bytes++;
        offset.runLength = 0;
      }

      return *this;
    }

    bool operator==(const DigitIterator &other) const {
      if (digitLimit.has_value() != other.digitLimit.has_value())
        return false;
      if (digitLimit.has_value())
        return *digitLimit == *other.digitLimit;
      return (offset.bytes == other.offset.bytes) &&
             (offset.runLength == other.offset.runLength);
    }

    Offset currentOffset() const { return offset; }

  private:
    const RunLengthCode *basePtr;
    const SizeType totalBytes;
    Offset offset;
    std::optional<SizeType> digitLimit;
  };

  /// Returns an unbounded iterator allowing to step over the encoded logic
  /// digits. A start offset from the least significant digit can be optionally
  /// specified.
  DigitIterator infiniteIterator(Offset startOffset = {0, 0}) const {
    assert((!isValid() || startOffset.bytes < byteCount) &&
           "start byte offset exceeeds bounds");
    assert((!isValid() || (startOffset.bytes == byteCount - 1) ||
            (startOffset.runLength <
             getRunLength(getCodePointer()[startOffset.bytes]))) &&
           "start run-lenght offset exceeeds bounds");
    return DigitIterator(getCodePointer(), byteCount, startOffset, {});
  }

  /// Returns a bounded iterator range allowing to step over 'digitCount'
  /// encoded digits. A start offset from the least significant digit can be
  /// optionally specified.
  llvm::iterator_range<DigitIterator>
  boundedIterator(unsigned digitCount, Offset startOffset = {0, 0}) const {
    assert((!isValid() || startOffset.bytes < byteCount) &&
           "start byte offset exceeeds bounds");
    assert((!isValid() || (startOffset.bytes == byteCount - 1) ||
            (startOffset.runLength <
             getRunLength(getCodePointer()[startOffset.bytes]))) &&
           "start run-lenght offset exceeeds bounds");
    return llvm::make_range(
        DigitIterator(getCodePointer(), byteCount, startOffset, digitCount),
        DigitIterator(getCodePointer(), byteCount, Offset(), 0));
  }

  // Helpers for storing an RLELogic in attributes
  friend struct llvm::DenseMapInfo<RLELogic, void>;
  friend llvm::hash_code hashValue(const RLELogic &rlelog);

private:
  static constexpr SizeType maxSelfContainedBytes = sizeof(RawType);

  static constexpr bool isSelfContained(SizeType byteCount) {
    return byteCount <= maxSelfContainedBytes;
  };
  /// Returns true iff the encoded value does not require a heap allocation
  bool isSelfContained() const { return isSelfContained(byteCount); };

  /// Raw constructor
  RLELogic(RawType raw, SizeType size, uint16_t mask)
      : valPtrUnion({raw}), byteCount(size), digitMask(mask){};

  /// Construct a filled value
  explicit RLELogic(logicdigits::LogicDigit fillDigit)
      : valPtrUnion({(RawType)fillDigit}), byteCount(1),
        digitMask(makeDigitMask(fillDigit)){};

  /// Invalidates the instance and optionally deallocates its heap buffer.
  void invalidate(bool freePtr) {
    if (freePtr && !isSelfContained())
      delete[] valPtrUnion.ptr;
    valPtrUnion.raw = (RawType)InvalidTag::Deleted;
    byteCount = 0;
    digitMask = 0;
  }

  /// Swaps the values of this instance and the given instance
  void swap(RLELogic &rlelog) {
    std::swap(valPtrUnion.raw, rlelog.valPtrUnion.raw);
    std::swap(byteCount, rlelog.byteCount);
    std::swap(digitMask, rlelog.digitMask);
  };

  /// Helper to combine a digit and a run-length into a code byte
  static RunLengthCode toCode(logicdigits::LogicDigit digit,
                              uint8_t runLength) {
    assert((runLength > 0 && runLength <= 16) && "invalid run-length");
    assert(isValidLogicDigit(digit) && "attemtmpt to encode invalid digit");
    return static_cast<RunLengthCode>(digit) | ((runLength - 1) << 4);
  }

  /// Creates a bit mask of the given logic digit
  static constexpr uint16_t makeDigitMask(logicdigits::LogicDigit logDig) {
    return (1 << ((uint8_t)logDig - 1));
  };

  /// Creates a bit mask of the given logic digits
  template <typename... Args>
  static constexpr uint16_t makeDigitMask(logicdigits::LogicDigit logDig,
                                          Args... args) {
    return makeDigitMask(logDig) | makeDigitMask(args...);
  };

  /// Value/Pointer union containing the value buffer directly or providing a
  /// pointer to it.
  union {
    RawType raw;
    RunLengthCode digits[maxSelfContainedBytes];
    RunLengthCode *ptr;
  } valPtrUnion;
  static_assert(sizeof(valPtrUnion.raw) == sizeof(valPtrUnion.digits));
  static_assert(sizeof(valPtrUnion.raw) == sizeof(valPtrUnion.ptr));

  /// Size of the value buffer in bytes
  SizeType byteCount;

  /// Bit mask indicating the contained logic digits
  uint16_t digitMask;
};

llvm::hash_code hashValue(const RLELogic &rlelog);

} // namespace hw
} // namespace circt

/// Provide DenseMapInfo for RLELogic.
namespace llvm {

template <>
struct DenseMapInfo<circt::hw::RLELogic, void> {
  static inline circt::hw::RLELogic getEmptyKey() {
    return circt::hw::RLELogic(circt::hw::RLELogic::InvalidTag::KeyEmpty);
  }

  static inline circt::hw::RLELogic getTombstoneKey() {
    return circt::hw::RLELogic(circt::hw::RLELogic::InvalidTag::KeyTombstone);
  }

  static unsigned getHashValue(const circt::hw::RLELogic &key) {
    return static_cast<unsigned>(circt::hw::hashValue(key));
  }

  static bool isEqual(const circt::hw::RLELogic &lhs,
                      const circt::hw::RLELogic &rhs) {
    return lhs == rhs;
  }
};

} // namespace llvm
#endif // CIRCT_DIALECT_HW_HWRLELOGIC_H
