//===- TypedPorts.h - Strongly-typed ESI port wrappers ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DO NOT EDIT!
// This file is distributed as part of an ESI package. The source for this file
// should always be modified within CIRCT.
//
//===----------------------------------------------------------------------===//
//
// Thin, non-owning wrappers around WriteChannelPort / ReadChannelPort that
// verify type compatibility at connect() time and provide strongly-typed
// write/read APIs. Purely additive — no changes to the untyped port classes.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef ESI_TYPED_PORTS_H
#define ESI_TYPED_PORTS_H

#include "esi/Design.h"
#include "esi/Ports.h"
#include "esi/Services.h"
#include "esi/Types.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <typeinfo>

namespace esi {

//===----------------------------------------------------------------------===//
// AcceleratorMismatchError — thrown for type mismatches and port-not-found.
//===----------------------------------------------------------------------===//

class AcceleratorMismatchError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

//===----------------------------------------------------------------------===//
// Helpers: unwrap TypeAliasType and width-aware serialization.
//===----------------------------------------------------------------------===//

/// Unwrap TypeAliasType (possibly recursively) to get the underlying type.
inline const Type *unwrapTypeAlias(const Type *t) {
  while (auto *alias = dynamic_cast<const TypeAliasType *>(t))
    t = alias->getInnerType();
  return t;
}

/// Compute the wire byte count for a port type. Returns 0 if not a
/// BitVectorType (meaning sizeof(T) should be used instead).
/// Wire format info for a port type, cached at connect() time.
struct WireInfo {
  size_t bytes = 0; // (bitWidth+7)/8, or 0 if not a BitVectorType.
  size_t bitWidth = 0;
};

inline WireInfo getWireInfo(const Type *portType) {
  const Type *inner = unwrapTypeAlias(portType);
  if (auto *bv = dynamic_cast<const BitVectorType *>(inner))
    return {(bv->getWidth() + 7) / 8, bv->getWidth()};
  return {};
}

/// Pack a C++ integral value into a MessageData with the given wire byte count.
/// If wi.bytes is 0, falls back to sizeof(T).
template <typename T>
MessageData toMessageData(const T &data, WireInfo wi) {
  if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
    if (wi.bytes > 0 && wi.bytes != sizeof(T)) {
      std::vector<uint8_t> buf(wi.bytes, 0);
      std::memcpy(buf.data(), &data, std::min(wi.bytes, sizeof(T)));
      return MessageData(std::move(buf));
    }
  } else if constexpr (std::is_base_of_v<SegmentedMessageData, T>) {
    return data.toMessageData();
  }
  return MessageData(reinterpret_cast<const uint8_t *>(&data), sizeof(T));
}

template <typename T>
MessageData toMessageData(const T &data, const Type *portType) {
  return toMessageData(data, getWireInfo(portType));
}

/// Unpack a MessageData into a C++ integral value with the given wire info.
/// If the wire size differs from sizeof(T), copies available bytes into
/// a zero-initialized value and sign-extends for signed types using the
/// actual bit width to locate the sign bit.
template <typename T>
T fromMessageData(const MessageData &msg, WireInfo wi) {
  if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
    if (wi.bytes > 0 && msg.getSize() == wi.bytes &&
        wi.bitWidth < sizeof(T) * 8) {
      // Copy wire bytes into a zero-initialized value.
      T val = 0;
      std::memcpy(&val, msg.getBytes(), std::min(wi.bytes, sizeof(T)));
      // Sign-extend for signed types if the sign bit is set.
      if constexpr (std::is_signed_v<T>) {
        size_t signBit = wi.bitWidth - 1;
        size_t signByte = signBit / 8;
        uint8_t signMask = uint8_t(1) << (signBit % 8);
        if (signByte < wi.bytes && (msg.getBytes()[signByte] & signMask)) {
          // Set all bits above the sign bit to 1.
          if (wi.bitWidth < sizeof(T) * 8)
            val |= static_cast<T>(~T(0)) << wi.bitWidth;
        }
      }
      return val;
    }
  }
  return *msg.as<T>();
}

template <typename T>
T fromMessageData(const MessageData &msg, const Type *portType) {
  return fromMessageData<T>(msg, getWireInfo(portType));
}

namespace detail {

/// Owning callback used by typed read deserializers.
///
/// Returning `false` means the callee did not accept the object and wants the
/// exact same owned value retried later.
template <typename T>
using TypedReadOwnedCallback = std::function<bool(std::unique_ptr<T> &)>;

template <typename T, typename = void>
struct has_type_deserializer : std::false_type {};

template <typename T>
struct has_type_deserializer<T, std::void_t<typename T::TypeDeserializer>>
    : std::true_type {};

template <typename T>
inline constexpr bool has_type_deserializer_v = has_type_deserializer<T>::value;

template <typename T>
const MessageData &getMessageDataRef(const SegmentedMessageData &msg,
                                     MessageData &scratch) {
  if (auto *flat = dynamic_cast<const MessageData *>(&msg))
    return *flat;
  scratch = msg.toMessageData();
  return scratch;
}

/// Default deserializer for simple 1:1 typed reads.
///
/// This path converts one raw message into one typed value and forwards it to
/// the typed callback. If the callback rejects the decoded object, preserve it
/// until the raw message is retried so the same owned object is presented
/// again.
template <typename T>
class PODTypeDeserializer {
public:
  using OutputCallback = TypedReadOwnedCallback<T>;

  PODTypeDeserializer(OutputCallback output, WireInfo wireInfo)
      : output(std::move(output)), wireInfo(wireInfo) {}

  bool push(std::unique_ptr<SegmentedMessageData> &msg) {
    std::scoped_lock<std::mutex> lock(mutex);
    if (!msg)
      throw std::runtime_error("PODTypeDeserializer::push: null message");

    // Flush the buffer, bail if still full.
    poke();
    if (pendingOutput)
      return false;

    // Translate the raw message into a typed object. This always consumes the
    // raw message.
    MessageData scratch;
    const MessageData &flat = getMessageDataRef<T>(*msg, scratch);
    pendingOutput = std::make_unique<T>(fromMessageData<T>(flat, wireInfo));
    msg.reset();
    poke();
    return true;
  }

  bool poke() {
    if (pendingOutput && output(pendingOutput)) {
      pendingOutput.reset();
      return true;
    }
    return false;
  }

private:
  OutputCallback output;
  WireInfo wireInfo;
  std::mutex mutex;
  std::unique_ptr<T> pendingOutput;
};

template <typename T, typename = void>
struct DeserializerSelector {
  using type = PODTypeDeserializer<T>;
};

template <typename T>
struct DeserializerSelector<T, std::void_t<typename T::TypeDeserializer>> {
  using type = typename T::TypeDeserializer;
};

template <typename T>
using DeserializerFor = typename DeserializerSelector<T>::type;

template <typename T>
using DeserializerOutputCallback = typename DeserializerFor<T>::OutputCallback;

} // namespace detail

/// Helper base class for stateful deserializers which may emit zero, one, or
/// many typed outputs for each raw input message.
///
/// Derived classes implement `decode()` and must consume the raw input message
/// before returning. This base class handles retrying blocked typed outputs and
/// preserving them in FIFO order until the client accepts them.
template <typename T>
class QueuedDecodeTypeDeserializer {
public:
  using OutputCallback = detail::TypedReadOwnedCallback<T>;
  using DecodedOutputs = std::vector<std::unique_ptr<T>>;

  explicit QueuedDecodeTypeDeserializer(OutputCallback output)
      : output(std::move(output)) {}

  virtual ~QueuedDecodeTypeDeserializer() = default;

  /// Push one raw message into the deserializer.
  ///
  /// Returns `false` only when previously decoded typed outputs are still
  /// blocked on delivery. In that case `msg` is left untouched so the caller
  /// can retry it later.
  bool push(std::unique_ptr<SegmentedMessageData> &msg) {
    std::scoped_lock<std::mutex> lock(mutex);
    if (!msg)
      throw std::runtime_error(
          "QueuedDecodeTypeDeserializer::push: null message");

    if (!pokeLocked())
      return false;

    DecodedOutputs decoded = decode(msg);
    if (msg)
      throw std::runtime_error(
          "QueuedDecodeTypeDeserializer::push: decode must consume the "
          "message");
    for (std::unique_ptr<T> &value : decoded) {
      if (!value)
        throw std::runtime_error(
            "QueuedDecodeTypeDeserializer::push: null decoded output");
      pendingOutputs.push(std::move(value));
    }

    // Once decode() has consumed the raw message, keep any blocked typed
    // outputs in the pending queue and report success. pokeLocked() still
    // opportunistically drains what it can before returning.
    pokeLocked();
    return true;
  }

  /// Retry delivery of any typed outputs which were previously blocked by the
  /// client callback.
  bool poke() {
    std::scoped_lock<std::mutex> lock(mutex);
    return pokeLocked();
  }

protected:
  /// Decode one raw message into zero or more typed outputs.
  ///
  /// Implementations must consume `msg` before returning, even when zero
  /// outputs are produced.
  virtual DecodedOutputs decode(std::unique_ptr<SegmentedMessageData> &msg) = 0;

private:
  bool pokeLocked() {
    while (!pendingOutputs.empty()) {
      std::unique_ptr<T> &value = pendingOutputs.front();
      if (!value)
        throw std::runtime_error(
            "QueuedDecodeTypeDeserializer::poke: null pending output");
      if (!output(value))
        return false;
      pendingOutputs.pop();
    }
    return true;
  }

  OutputCallback output;
  std::queue<std::unique_ptr<T>> pendingOutputs;
  std::mutex mutex;
};

/// Reusable serial-list window deserializer.
///
/// Walks the serial-list multi-burst protocol used by codegen'd window
/// helpers -- zero or more `header(N>0) + N data` bursts terminated by a
/// `header(0)` footer -- and emits a fully-formed `T` instance built from the
/// accumulated header fields and data frames.
///
/// `T` must be a generated window helper that exposes:
///  - nested `header_frame` / `data_frame` types of identical size,
///  - `count_type`,
///  - `static count_type T::_headerCount(const header_frame &)`,
///  - `static std::unique_ptr<T> T::_fromFrames(const header_frame &,
///        std::vector<data_frame> &&)`.
///
/// `T` typically declares `SerialListTypeDeserializer<T>` as a friend so the
/// template can reach the (private) `header_frame` definition; the helpers
/// themselves can stay private as well.
template <typename T>
class SerialListTypeDeserializer : public QueuedDecodeTypeDeserializer<T> {
public:
  using Base = QueuedDecodeTypeDeserializer<T>;
  using OutputCallback = typename Base::OutputCallback;
  using DecodedOutputs = typename Base::DecodedOutputs;

  explicit SerialListTypeDeserializer(OutputCallback output)
      : Base(std::move(output)) {}

private:
  using header_frame = typename T::header_frame;
  using data_frame = typename T::data_frame;
  using count_type = typename T::count_type;

  static_assert(sizeof(header_frame) == sizeof(data_frame),
                "header and data frames must be the same width");
  static constexpr size_t kFrameSize = sizeof(data_frame);

  DecodedOutputs decode(std::unique_ptr<SegmentedMessageData> &msg) override {
    DecodedOutputs out;
    MessageData scratch;
    const MessageData &flat = detail::getMessageDataRef<T>(*msg, scratch);
    const uint8_t *bytes = flat.getBytes();
    size_t size = flat.getSize();

    size_t offset = 0;
    while (offset < size) {
      size_t needed = kFrameSize - partial_.size();
      size_t chunk = std::min(needed, size - offset);
      partial_.insert(partial_.end(), bytes + offset, bytes + offset + chunk);
      offset += chunk;
      if (partial_.size() != kFrameSize)
        break;

      if (remaining_ == 0) {
        // Header or footer frame. Decode into a local frame first so we can
        // inspect the count without committing. Only the first header of a
        // transaction is guaranteed to carry valid static fields; the static
        // slots of continuation and footer headers may be garbage.
        header_frame frame{};
        std::memcpy(&frame, partial_.data(), kFrameSize);
        partial_.clear();
        count_type batchCount = T::_headerCount(frame);
        if (batchCount == 0) {
          // Footer: emit the accumulated value using the first header's
          // static fields.
          if (!pending_header_)
            throw std::runtime_error(
                "SerialListTypeDeserializer: footer received before any "
                "header");
          out.push_back(
              T::_fromFrames(*pending_header_, std::move(pending_frames_)));
          pending_frames_.clear();
          pending_header_.reset();
          continue;
        }

        if (!pending_header_)
          pending_header_ = frame;
        remaining_ = batchCount;
        continue;
      }

      // Data frame.
      auto &frame = pending_frames_.emplace_back();
      std::memcpy(&frame, partial_.data(), kFrameSize);
      partial_.clear();
      --remaining_;
    }

    msg.reset();
    return out;
  }

  std::vector<uint8_t> partial_;
  std::optional<header_frame> pending_header_;
  std::vector<data_frame> pending_frames_;
  count_type remaining_ = 0;
};

//===----------------------------------------------------------------------===//
// Type-trait: detect T::_ESI_ID (a static constexpr std::string_view).
//===----------------------------------------------------------------------===//

template <typename T, typename = void>
struct has_esi_id : std::false_type {};

template <typename T>
struct has_esi_id<T, std::void_t<decltype(T::_ESI_ID)>>
    : std::is_convertible<decltype(T::_ESI_ID), std::string_view> {};

template <typename T>
inline constexpr bool has_esi_id_v = has_esi_id<T>::value;

// Detect T::_ESI_WINDOW_ID (a static constexpr std::string_view) for
// generated SegmentedMessageData subclasses bound to a specific WindowType.
template <typename T, typename = void>
struct has_esi_window_id : std::false_type {};

template <typename T>
struct has_esi_window_id<T, std::void_t<decltype(T::_ESI_WINDOW_ID)>>
    : std::is_convertible<decltype(T::_ESI_WINDOW_ID), std::string_view> {};

template <typename T>
inline constexpr bool has_esi_window_id_v = has_esi_window_id<T>::value;

//===----------------------------------------------------------------------===//
// verifyTypeCompatibility<T>(const Type *portType)
//
// Checks that the ESI runtime Type is compatible with the C++ type T.
// Dispatch order: _ESI_ID → void → bool → signed int → unsigned int → error.
//===----------------------------------------------------------------------===//

template <typename T>
void verifyTypeCompatibility(const Type *portType) {
  if (!portType)
    throw AcceleratorMismatchError("Port type is null");

  // Unwrap TypeAliasType to get the inner type for verification.
  portType = unwrapTypeAlias(portType);

  // If the port is a windowed type, verify the window id (if T declares one)
  // and then continue checking the inner ('into') type.
  if (auto *windowType = dynamic_cast<const WindowType *>(portType)) {
    if constexpr (has_esi_window_id_v<T>) {
      if (std::string_view(windowType->getID()) != T::_ESI_WINDOW_ID)
        throw AcceleratorMismatchError(
            "ESI window mismatch: C++ type has _ESI_WINDOW_ID '" +
            std::string(T::_ESI_WINDOW_ID) + "' but port window type is '" +
            windowType->toString(/*oneLine=*/true) + "'");
    } else {
      throw AcceleratorMismatchError(
          "ESI type mismatch: port is a window type ('" +
          windowType->toString(/*oneLine=*/true) +
          "') but C++ type has no _ESI_WINDOW_ID");
    }
    portType = unwrapTypeAlias(windowType->getIntoType());
  } else if constexpr (has_esi_window_id_v<T>) {
    throw AcceleratorMismatchError(
        std::string("ESI type mismatch: C++ type has _ESI_WINDOW_ID '") +
        std::string(T::_ESI_WINDOW_ID) + "' but port type is not a window: '" +
        portType->toString(/*oneLine=*/true) + "'");
  }

  if constexpr (has_esi_id_v<T>) {
    // Highest priority: user-defined ESI ID string comparison.
    if (std::string_view(portType->getID()) != T::_ESI_ID)
      throw AcceleratorMismatchError(
          "ESI type mismatch: C++ type has _ESI_ID '" +
          std::string(T::_ESI_ID) + "' but port type is '" +
          portType->toString(/*oneLine=*/true) + "'");
  } else if constexpr (std::is_void_v<T>) {
    if (!dynamic_cast<const VoidType *>(portType))
      throw AcceleratorMismatchError("ESI type mismatch: expected VoidType for "
                                     "void, but port type is '" +
                                     portType->toString(/*oneLine=*/true) +
                                     "'");
  } else if constexpr (std::is_same_v<T, bool>) {
    // bool maps to signless i1, which is BitsType with width <= 1.
    auto *bits = dynamic_cast<const BitsType *>(portType);
    if (!bits || bits->getWidth() > 1)
      throw AcceleratorMismatchError(
          "ESI type mismatch: expected BitsType with width <= 1 for "
          "bool, but port type is '" +
          portType->toString(/*oneLine=*/true) + "'");
  } else if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
    auto *sint = dynamic_cast<const SIntType *>(portType);
    if (!sint)
      throw AcceleratorMismatchError(
          "ESI type mismatch: expected SIntType for signed integer, "
          "but port type is '" +
          portType->toString(/*oneLine=*/true) + "'");
    if (sint->getWidth() > sizeof(T) * 8)
      throw AcceleratorMismatchError(
          "ESI type mismatch: SIntType width " +
          std::to_string(sint->getWidth()) + " does not fit in " +
          std::to_string(sizeof(T) * 8) + "-bit signed integer");
    // Require closest-size match: reject if a smaller C++ type would suffice.
    if (sizeof(T) > 1 && sint->getWidth() <= (sizeof(T) / 2) * 8)
      throw AcceleratorMismatchError("ESI type mismatch: SIntType width " +
                                     std::to_string(sint->getWidth()) +
                                     " should use a smaller C++ type than " +
                                     std::to_string(sizeof(T) * 8) + "-bit");
  } else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T>) {
    // Accept UIntType (uiM) or BitsType (iM, signless).
    auto *uintPort = dynamic_cast<const UIntType *>(portType);
    auto *bits = dynamic_cast<const BitsType *>(portType);
    if (!uintPort && !bits)
      throw AcceleratorMismatchError(
          "ESI type mismatch: expected UIntType or BitsType for unsigned "
          "integer, but port type is '" +
          portType->toString(/*oneLine=*/true) + "'");
    uint64_t width = uintPort ? uintPort->getWidth() : bits->getWidth();
    if (width > sizeof(T) * 8)
      throw AcceleratorMismatchError(
          "ESI type mismatch: bit width " + std::to_string(width) +
          " does not fit in " + std::to_string(sizeof(T) * 8) +
          "-bit unsigned integer");
    // Require closest-size match: reject if a smaller C++ type would suffice.
    if (sizeof(T) > 1 && width <= (sizeof(T) / 2) * 8)
      throw AcceleratorMismatchError("ESI type mismatch: bit width " +
                                     std::to_string(width) +
                                     " should use a smaller C++ type than " +
                                     std::to_string(sizeof(T) * 8) + "-bit");
  } else {
    throw AcceleratorMismatchError(
        std::string("Cannot verify type compatibility for C++ type '") +
        typeid(T).name() + "' against ESI port type '" +
        portType->toString(/*oneLine=*/true) + "'");
  }
}

// Port-aware overload: when checking against a ChannelPort, also reconcile
// the windowed wrapper. ChannelPort::getType() returns the unwrapped 'into'
// type for windowed ports, so we have to consult getWindowType() directly to
// detect the window and forward the original WindowType into the Type-based
// overload above.
template <typename T>
void verifyTypeCompatibility(const ChannelPort *port) {
  if (!port)
    throw AcceleratorMismatchError("Port is null");
  const WindowType *windowType = port->getWindowType();
  if constexpr (has_esi_window_id_v<T>) {
    if (!windowType)
      throw AcceleratorMismatchError(
          std::string("ESI type mismatch: C++ type has _ESI_WINDOW_ID '") +
          std::string(T::_ESI_WINDOW_ID) +
          "' but port is not a window type ('" +
          (port->getType() ? port->getType()->toString(/*oneLine=*/true)
                           : std::string("<null>")) +
          "')");
  } else {
    if (windowType)
      throw AcceleratorMismatchError(
          "ESI type mismatch: port is a window type ('" +
          windowType->toString(/*oneLine=*/true) +
          "') but C++ type has no _ESI_WINDOW_ID");
  }
  const Type *forwardType =
      windowType ? static_cast<const Type *>(windowType) : port->getType();
  verifyTypeCompatibility<T>(forwardType);
}

//===----------------------------------------------------------------------===//
// TypedWritePort<T, SkipTypeCheck = false>
//
// When SkipTypeCheck is false, the `connect` method runs the type check via
// `verifyTypeCompatibility<T>()`. When SkipTypeCheck is true, `connect`
// skips that verification.
//===----------------------------------------------------------------------===//

template <typename T, bool SkipTypeCheck = false>
class TypedWritePort {
public:
  explicit TypedWritePort(WriteChannelPort &port) : inner(&port) {}
  // NOLINTNEXTLINE(google-explicit-constructor)
  TypedWritePort(WriteChannelPort *port) : inner(port) {}

  void connect(const ChannelPort::ConnectOptions &opts = {}) {
    if (!inner)
      throw AcceleratorMismatchError("TypedWritePort: null port pointer");
    if (!SkipTypeCheck)
      verifyTypeCompatibility<T>(inner);
    wireInfo_ = getWireInfo(inner->getType());
    inner->connect(opts);
  }

  void write(const T &data) { inner->write(toMessageData(data, wireInfo_)); }

  /// Write by taking ownership. If T is a SegmentedMessageData, this hands
  /// the message directly to the port's segmented write path.
  void write(std::unique_ptr<T> &data) {
    if (!data)
      throw std::runtime_error("TypedWritePort::write: null unique_ptr");
    if constexpr (std::is_base_of_v<SegmentedMessageData, T>) {
      inner->write(std::move(data));
    } else {
      write(*data);
      data.reset();
    }
  }

  bool tryWrite(const T &data) {
    return inner->tryWrite(toMessageData(data, wireInfo_));
  }

  bool flush() { return inner->flush(); }
  void disconnect() { inner->disconnect(); }
  bool isConnected() const { return inner && inner->isConnected(); }

  WriteChannelPort &raw() { return *inner; }
  const WriteChannelPort &raw() const { return *inner; }

private:
  WriteChannelPort *inner;
  WireInfo wireInfo_;
};

/// Specialization for void — write takes no data argument.
template <>
class TypedWritePort<void> {
public:
  explicit TypedWritePort(WriteChannelPort &port) : inner(&port) {}
  // NOLINTNEXTLINE(google-explicit-constructor)
  TypedWritePort(WriteChannelPort *port) : inner(port) {}

  void connect(const ChannelPort::ConnectOptions &opts = {}) {
    if (!inner)
      throw AcceleratorMismatchError("TypedWritePort: null port pointer");
    verifyTypeCompatibility<void>(inner->getType());
    inner->connect(opts);
  }

  void write() {
    uint8_t zero = 0;
    inner->write(MessageData(&zero, 1));
  }

  bool tryWrite() {
    uint8_t zero = 0;
    return inner->tryWrite(MessageData(&zero, 1));
  }

  bool flush() { return inner->flush(); }
  void disconnect() { inner->disconnect(); }
  bool isConnected() const { return inner && inner->isConnected(); }

  WriteChannelPort &raw() { return *inner; }
  const WriteChannelPort &raw() const { return *inner; }

private:
  WriteChannelPort *inner;
};

//===----------------------------------------------------------------------===//
// TypedReadPort<T>
//===----------------------------------------------------------------------===//

/// Strongly typed wrapper around a raw read channel.
///
/// For scalar/POD-like `T`, this performs a 1:1 conversion from raw messages.
/// If `T` defines a nested `TypeDeserializer`, one instance is created per
/// connection and drives both callback and polling reads through that
/// deserializer.
///
/// Polling reads return `std::unique_ptr<T>` so complex decoded values can be
/// delivered without an extra copy.
template <typename T, bool SkipTypeCheck = false>
class TypedReadPort {
public:
  explicit TypedReadPort(ReadChannelPort &port) : inner(&port) {}
  // NOLINTNEXTLINE(google-explicit-constructor)
  TypedReadPort(ReadChannelPort *port) : inner(port) {}
  TypedReadPort(const TypedReadPort &) = delete;
  TypedReadPort &operator=(const TypedReadPort &) = delete;
  TypedReadPort(TypedReadPort &&) = delete;
  TypedReadPort &operator=(TypedReadPort &&) = delete;

  ~TypedReadPort() {
    if (inner && mode != Mode::Disconnected)
      disconnect();
  }

  /// Connect in polling mode.
  ///
  /// The port installs an internal typed output queue. `read()` and
  /// `readAsync()` consume from that queue.
  void connect(const ChannelPort::ConnectOptions &opts = {}) {
    if (!inner)
      throw AcceleratorMismatchError("TypedReadPort: null port pointer");
    prepareConnect();
    pollingState.emplace(maxDataQueueMsgs);
    emplaceDeserializer([this](std::unique_ptr<T> &value) -> bool {
      return pollingState->enqueue(value);
    });
    try {
      inner->connect(
          [this](std::unique_ptr<SegmentedMessageData> &msg) -> bool {
            assert(deserializer && "Deserializer should be connected");
            return deserializer->push(msg);
          },
          opts);
    } catch (...) {
      resetConnectState();
      throw;
    }
    mode = Mode::Polling;
  }

  /// Connect a non-owning typed callback.
  ///
  /// The callback sees the decoded value by reference. Returning `false`
  /// requests that the same decoded value be retried later.
  void connect(std::function<bool(const T &)> callback,
               const ChannelPort::ConnectOptions &opts = {}) {
    connect([cb = std::move(callback)](
                std::unique_ptr<T> &value) -> bool { return cb(*value); },
            opts);
  }

  /// Connect an owning typed callback.
  ///
  /// This is the typed analogue of `ReadChannelPort::ReadCallback`: the
  /// callback may take ownership of the decoded value or return `false` to
  /// retry delivery later with the same object.
  void connect(detail::TypedReadOwnedCallback<T> callback,
               const ChannelPort::ConnectOptions &opts = {}) {
    if (!inner)
      throw AcceleratorMismatchError("TypedReadPort: null port pointer");
    prepareConnect();
    emplaceDeserializer(std::move(callback));
    try {
      inner->connect(
          [this](std::unique_ptr<SegmentedMessageData> &msg) -> bool {
            assert(deserializer && "Deserializer should be connected");
            return deserializer->push(msg);
          },
          opts);
    } catch (...) {
      resetConnectState();
      throw;
    }
    // TODO: Hook callback-mode custom-deserializer poke() retries into the
    // existing periodic poll/background-worker path.
    mode = Mode::Callback;
  }

  /// Blocking typed read in polling mode.
  std::unique_ptr<T> read() {
    std::future<std::unique_ptr<T>> f = readAsync();
    f.wait();
    return f.get();
  }

  /// Asynchronous typed read in polling mode.
  ///
  /// The returned future yields ownership of the next decoded value.
  std::future<std::unique_ptr<T>> readAsync() {
    if (mode == Mode::Callback)
      throw std::runtime_error(
          "Cannot read from a callback channel. `connect()` without a "
          "callback specified to use polling mode.");
    if (mode == Mode::Disconnected)
      throw std::runtime_error(
          "Cannot read from a disconnected channel. `connect()` first.");

    if (!pollingState)
      throw std::runtime_error(
          "Cannot read from a disconnected channel. `connect()` first.");

    std::future<std::unique_ptr<T>> future = pollingState->readAsync();

    if (deserializer)
      deserializer->poke();
    return future;
  }

  /// Set the maximum number of decoded typed values buffered in polling mode.
  /// `0` means unbounded.
  void setMaxDataQueueMsgs(uint64_t maxMsgs) {
    maxDataQueueMsgs = maxMsgs;
    if (pollingState)
      pollingState->setMaxQueued(maxMsgs);
  }

  /// Disconnect the typed port and abandon any pending polling reads.
  void disconnect() {
    inner->disconnect();
    mode = Mode::Disconnected;
    resetConnectState();
  }
  bool isConnected() const { return inner && inner->isConnected(); }

  ReadChannelPort &raw() { return *inner; }
  const ReadChannelPort &raw() const { return *inner; }

private:
  using Deserializer = detail::DeserializerFor<T>;
  enum Mode { Disconnected, Callback, Polling };
  using PollingState = detail::PollingBuffer<std::unique_ptr<T>>;

  void resetConnectState() {
    deserializer.reset();
    pollingState.reset();
  }

  void emplaceDeserializer(detail::DeserializerOutputCallback<T> callback) {
    if constexpr (detail::has_type_deserializer_v<T>) {
      deserializer.emplace(std::move(callback));
    } else {
      deserializer.emplace(std::move(callback), wireInfo_);
    }
  }

  void prepareConnect() {
    if (mode != Mode::Disconnected)
      throw std::runtime_error("Channel already connected");
    if constexpr (SkipTypeCheck) {
      // Skip verification, but still compute wireInfo for the POD path so
      // small / non-byte-aligned wire widths still encode correctly.
      if constexpr (!detail::has_type_deserializer_v<T>)
        wireInfo_ = getWireInfo(inner->getType());
    } else if constexpr (has_esi_id_v<T>) {
      verifyTypeCompatibility<T>(inner);
    } else if constexpr (!detail::has_type_deserializer_v<T>) {
      verifyTypeCompatibility<T>(inner);
      wireInfo_ = getWireInfo(inner->getType());
    }
  }

  ReadChannelPort *inner;
  WireInfo wireInfo_;
  Mode mode = Mode::Disconnected;
  uint64_t maxDataQueueMsgs = ReadChannelPort::DefaultMaxDataQueueMsgs;
  std::optional<Deserializer> deserializer;
  std::optional<PollingState> pollingState;
};

/// Specialization for void — read discards data and returns nothing.
template <>
class TypedReadPort<void> {
public:
  explicit TypedReadPort(ReadChannelPort &port) : inner(&port) {}
  // NOLINTNEXTLINE(google-explicit-constructor)
  TypedReadPort(ReadChannelPort *port) : inner(port) {}
  TypedReadPort(const TypedReadPort &) = delete;
  TypedReadPort &operator=(const TypedReadPort &) = delete;
  TypedReadPort(TypedReadPort &&) = delete;
  TypedReadPort &operator=(TypedReadPort &&) = delete;

  ~TypedReadPort() {
    if (inner && inner->isConnected())
      disconnect();
  }

  void connect(const ChannelPort::ConnectOptions &opts = {}) {
    if (!inner)
      throw AcceleratorMismatchError("TypedReadPort: null port pointer");
    verifyTypeCompatibility<void>(inner->getType());
    inner->connect(opts);
  }

  void connect(std::function<bool()> callback,
               const ChannelPort::ConnectOptions &opts = {}) {
    if (!inner)
      throw AcceleratorMismatchError("TypedReadPort: null port pointer");
    verifyTypeCompatibility<void>(inner->getType());
    inner->connect(
        [cb = std::move(callback)](MessageData) -> bool { return cb(); }, opts);
  }

  void read() {
    MessageData outData;
    inner->read(outData);
  }

  std::future<void> readAsync() {
    auto innerFuture = inner->readAsync();
    return std::async(
        std::launch::deferred,
        [f = std::move(innerFuture)]() mutable -> void { f.get(); });
  }

  void disconnect() { inner->disconnect(); }
  bool isConnected() const { return inner && inner->isConnected(); }

  ReadChannelPort &raw() { return *inner; }
  const ReadChannelPort &raw() const { return *inner; }

private:
  ReadChannelPort *inner;
};

//===----------------------------------------------------------------------===//
// TypedFunction<ArgT, ResultT>
//
// Strongly-typed function-call wrapper built on top of TypedWritePort and
// TypedReadPort. Implicitly constructible from `FuncService::Function *` (the
// return type of `BundlePort::getAs<FuncService::Function>()`); construction
// resolves the underlying raw "arg" / "result" channels but does not connect
// them until `connect()` is called.
//
// We intentionally bypass `FuncService::Function::call`. That helper returns
// the future from a single `result->readAsync()`, which only sees one raw
// frame and would race when the response is multi-frame or when calls are
// pipelined. Driving a `TypedReadPort<ResultT>` instead reuses its persistent
// per-port deserializer (so partial frames / pending outputs survive between
// calls) and its FIFO polling buffer (so pipelined `readAsync()` futures
// hand back per-call decoded values in call order, even when consumers
// `.get()` them out of order).
//===----------------------------------------------------------------------===//

namespace detail {

/// Standard ConnectOptions for typed function ports: untranslated frames so
/// the deserializer can see raw frame boundaries.
inline ChannelPort::ConnectOptions typedFunctionConnectOptions() {
  return ChannelPort::ConnectOptions(/*bufferSize=*/std::nullopt,
                                     /*translateMessage=*/false);
}

/// Convert a `std::future<std::unique_ptr<T>>` (as returned by
/// `TypedReadPort::readAsync()`) into a `std::future<T>`. Uses a deferred
/// async so `.get()` blocks only when the caller actually waits for the
/// value, preserving the per-call FIFO ordering of the underlying polling
/// buffer. A null `unique_ptr` from a misbehaving deserializer is reported
/// as a runtime error rather than dereferenced.
template <typename T>
std::future<T> awaitDecoded(std::future<std::unique_ptr<T>> inner) {
  return std::async(std::launch::deferred,
                    [fut = std::move(inner)]() mutable -> T {
                      std::unique_ptr<T> v = fut.get();
                      if (!v)
                        throw std::runtime_error(
                            "TypedFunction: deserializer produced a null "
                            "value");
                      return std::move(*v);
                    });
}

/// Throw the standard "null Function pointer" error used by every
/// TypedFunction specialization.
[[noreturn]] inline void throwNullFunction() {
  throw AcceleratorMismatchError(
      "TypedFunction: null Function pointer (getAs failed or wrong type)");
}

/// Throw a clear "not connected" error from `call()` paths.
[[noreturn]] inline void throwNotConnected() {
  throw std::runtime_error("TypedFunction: must be 'connect'ed before "
                           "calling");
}

/// Throw a clear "already connected" error from `connect()` paths.
[[noreturn]] inline void throwAlreadyConnected() {
  throw std::runtime_error("TypedFunction is already connected");
}

} // namespace detail

template <typename ArgT, typename ResultT, bool SkipTypeCheck = false>
class TypedFunction {
public:
  /// Implicit conversion from Function* (returned by getAs<>()).
  // NOLINTNEXTLINE(google-explicit-constructor)
  TypedFunction(services::FuncService::Function *func) : inner(func) {}
  TypedFunction(const TypedFunction &) = delete;
  TypedFunction &operator=(const TypedFunction &) = delete;

  void connect() {
    if (!inner)
      detail::throwNullFunction();
    if (argPort)
      detail::throwAlreadyConnected();
    argPort.emplace(&inner->getRawWrite("arg"));
    resultPort.emplace(&inner->getRawRead("result"));
    auto opts = detail::typedFunctionConnectOptions();
    argPort->connect(opts);
    resultPort->connect(opts);
  }

  std::future<ResultT> call(const ArgT &arg) {
    if (!argPort)
      detail::throwNotConnected();
    // Serialize the per-call write+readAsync pair so the polling buffer's
    // FIFO matches the call FIFO. Pipelined calls are still allowed -- the
    // shared deserializer drains responses in wire order and the FIFO
    // ensures call N's future resolves to call N's result, regardless of
    // the order in which callers `.get()` the futures.
    std::scoped_lock<std::mutex> lock(callMutex);
    argPort->write(arg);
    return detail::awaitDecoded<ResultT>(resultPort->readAsync());
  }

  /// Emplace-style call: constructs `ArgT` in-place from the forwarded
  /// arguments and forwards to `call(const ArgT &)`. SFINAE-disabled for the
  /// single-`ArgT`-argument case so it does not shadow the lvalue overload.
  template <
      typename First, typename... Rest,
      typename = std::enable_if_t<
          std::is_constructible_v<ArgT, First, Rest...> &&
          (!std::is_same_v<std::decay_t<First>, ArgT> || sizeof...(Rest) != 0)>>
  std::future<ResultT> call(First &&first, Rest &&...rest) {
    return call(ArgT(std::forward<First>(first), std::forward<Rest>(rest)...));
  }

  /// Function-call operator overloads: forward to `call()`.
  template <typename... Args>
  auto operator()(Args &&...args)
      -> decltype(this->call(std::forward<Args>(args)...)) {
    return call(std::forward<Args>(args)...);
  }

  services::FuncService::Function &raw() { return *inner; }
  const services::FuncService::Function &raw() const { return *inner; }

private:
  services::FuncService::Function *inner;
  std::optional<TypedWritePort<ArgT, SkipTypeCheck>> argPort;
  std::optional<TypedReadPort<ResultT, SkipTypeCheck>> resultPort;
  std::mutex callMutex;
};

/// Partial specialization: void argument, typed result.
template <typename ResultT, bool SkipTypeCheck>
class TypedFunction<void, ResultT, SkipTypeCheck> {
public:
  // NOLINTNEXTLINE(google-explicit-constructor)
  TypedFunction(services::FuncService::Function *func) : inner(func) {}
  TypedFunction(const TypedFunction &) = delete;
  TypedFunction &operator=(const TypedFunction &) = delete;

  void connect() {
    if (!inner)
      detail::throwNullFunction();
    if (argPort)
      detail::throwAlreadyConnected();
    argPort.emplace(&inner->getRawWrite("arg"));
    resultPort.emplace(&inner->getRawRead("result"));
    auto opts = detail::typedFunctionConnectOptions();
    argPort->connect(opts);
    resultPort->connect(opts);
  }

  std::future<ResultT> call() {
    if (!argPort)
      detail::throwNotConnected();
    std::scoped_lock<std::mutex> lock(callMutex);
    argPort->write();
    return detail::awaitDecoded<ResultT>(resultPort->readAsync());
  }

  /// Function-call operator overload: forwards to `call()`.
  std::future<ResultT> operator()() { return call(); }

  services::FuncService::Function &raw() { return *inner; }
  const services::FuncService::Function &raw() const { return *inner; }

private:
  services::FuncService::Function *inner;
  std::optional<TypedWritePort<void>> argPort;
  std::optional<TypedReadPort<ResultT, SkipTypeCheck>> resultPort;
  std::mutex callMutex;
};

/// Partial specialization: typed argument, void result.
template <typename ArgT, bool SkipTypeCheck>
class TypedFunction<ArgT, void, SkipTypeCheck> {
public:
  // NOLINTNEXTLINE(google-explicit-constructor)
  TypedFunction(services::FuncService::Function *func) : inner(func) {}
  TypedFunction(const TypedFunction &) = delete;
  TypedFunction &operator=(const TypedFunction &) = delete;

  void connect() {
    if (!inner)
      detail::throwNullFunction();
    if (argPort)
      detail::throwAlreadyConnected();
    argPort.emplace(&inner->getRawWrite("arg"));
    resultPort.emplace(&inner->getRawRead("result"));
    auto opts = detail::typedFunctionConnectOptions();
    argPort->connect(opts);
    resultPort->connect(opts);
  }

  std::future<void> call(const ArgT &arg) {
    if (!argPort)
      detail::throwNotConnected();
    std::scoped_lock<std::mutex> lock(callMutex);
    argPort->write(arg);
    return resultPort->readAsync();
  }

  /// Emplace-style call: constructs `ArgT` in-place from the forwarded
  /// arguments and forwards to `call(const ArgT &)`. SFINAE-disabled for the
  /// single-`ArgT`-argument case so it does not shadow the lvalue overload.
  template <
      typename First, typename... Rest,
      typename = std::enable_if_t<
          std::is_constructible_v<ArgT, First, Rest...> &&
          (!std::is_same_v<std::decay_t<First>, ArgT> || sizeof...(Rest) != 0)>>
  std::future<void> call(First &&first, Rest &&...rest) {
    return call(ArgT(std::forward<First>(first), std::forward<Rest>(rest)...));
  }

  /// Function-call operator overloads: forward to `call()`.
  template <typename... Args>
  auto operator()(Args &&...args)
      -> decltype(this->call(std::forward<Args>(args)...)) {
    return call(std::forward<Args>(args)...);
  }

  services::FuncService::Function &raw() { return *inner; }
  const services::FuncService::Function &raw() const { return *inner; }

private:
  services::FuncService::Function *inner;
  std::optional<TypedWritePort<ArgT, SkipTypeCheck>> argPort;
  std::optional<TypedReadPort<void>> resultPort;
  std::mutex callMutex;
};

/// Full specialization: void argument, void result.
template <bool SkipTypeCheck>
class TypedFunction<void, void, SkipTypeCheck> {
public:
  // NOLINTNEXTLINE(google-explicit-constructor)
  TypedFunction(services::FuncService::Function *func) : inner(func) {}
  TypedFunction(const TypedFunction &) = delete;
  TypedFunction &operator=(const TypedFunction &) = delete;

  void connect() {
    if (!inner)
      detail::throwNullFunction();
    if (argPort)
      detail::throwAlreadyConnected();
    argPort.emplace(&inner->getRawWrite("arg"));
    resultPort.emplace(&inner->getRawRead("result"));
    auto opts = detail::typedFunctionConnectOptions();
    argPort->connect(opts);
    resultPort->connect(opts);
  }

  std::future<void> call() {
    if (!argPort)
      detail::throwNotConnected();
    std::scoped_lock<std::mutex> lock(callMutex);
    argPort->write();
    return resultPort->readAsync();
  }

  /// Function-call operator overload: forwards to `call()`.
  std::future<void> operator()() { return call(); }

  services::FuncService::Function &raw() { return *inner; }
  const services::FuncService::Function &raw() const { return *inner; }

private:
  services::FuncService::Function *inner;
  std::optional<TypedWritePort<void>> argPort;
  std::optional<TypedReadPort<void>> resultPort;
  std::mutex callMutex;
};

//===----------------------------------------------------------------------===//
// TypedCallback<ArgT, ResultT>
//
// Strongly-typed accelerator-callback wrapper built on top of TypedReadPort
// and TypedWritePort. Implicitly constructible from `CallService::Callback *`
// (the return type of `BundlePort::getAs<CallService::Callback>()`);
// construction resolves the underlying raw "arg" / "result" channels but
// does not connect them until `connect()` is called.
//
// We bypass `CallService::Callback::connect` for the same reason we bypass
// `FuncService::Function::call`: typed args may span multiple raw frames and
// we need the persistent stateful deserializer that `TypedReadPort<ArgT>`
// already provides.
//
// Note: the `quick` parameter is kept for API compatibility but has no
// effect in this design. The typed user callback is always dispatched inline
// from the read-channel callback thread (matching the previous
// custom-deserializer behavior). If service-thread dispatch is needed in the
// future, that can be layered on top by using `TypedReadPort` polling mode
// plus a worker.
//
// Lifetime: the TypedReadPort callback installed on `connect()` captures
// `this`. The wrapper is non-copyable and non-movable, so its address is
// stable for its entire lifetime. Destruction of the embedded TypedReadPort
// synchronously revokes the callback and waits for any in-flight callback
// dispatch to complete (see ReadChannelPort::disconnect), so the captured
// `this` cannot dangle. Callers must ensure the TypedCallback outlives any
// expected callback dispatch -- using a temporary like
// `TypedCallback<...>(cb).connect(...)` is safe only because the destructor
// runs at end-of-full-expression and disconnects before returning.
//===----------------------------------------------------------------------===//

namespace detail {

/// Throw the standard "null Callback pointer" error used by every
/// TypedCallback specialization.
[[noreturn]] inline void throwNullCallback() {
  throw AcceleratorMismatchError(
      "TypedCallback: null Callback pointer (getAs failed or wrong type)");
}

/// Throw a clear "already connected" error from TypedCallback `connect()`
/// paths.
[[noreturn]] inline void throwCallbackAlreadyConnected() {
  throw std::runtime_error("TypedCallback is already connected");
}

} // namespace detail

template <typename ArgT, typename ResultT, bool SkipTypeCheck = false>
class TypedCallback {
public:
  // NOLINTNEXTLINE(google-explicit-constructor)
  TypedCallback(services::CallService::Callback *cb) : inner(cb) {}
  TypedCallback(const TypedCallback &) = delete;
  TypedCallback &operator=(const TypedCallback &) = delete;

  void connect(std::function<ResultT(const ArgT &)> callback,
               bool quick = false) {
    (void)quick; // See class header.
    if (!inner)
      detail::throwNullCallback();
    if (argPort)
      detail::throwCallbackAlreadyConnected();
    argPort.emplace(&inner->getRawRead("arg"));
    resultPort.emplace(&inner->getRawWrite("result"));
    auto opts = detail::typedFunctionConnectOptions();
    resultPort->connect(opts);
    userCallback = std::move(callback);
    // The TypedReadPort callback runs on the read-channel callback thread
    // for every decoded ArgT. We forward to the user's callback and then
    // serialize the result through the TypedWritePort.
    argPort->connect(
        [this](const ArgT &arg) -> bool {
          ResultT result = userCallback(arg);
          resultPort->write(result);
          return true;
        },
        opts);
  }

  services::CallService::Callback &raw() { return *inner; }
  const services::CallService::Callback &raw() const { return *inner; }

private:
  services::CallService::Callback *inner;
  std::optional<TypedReadPort<ArgT, SkipTypeCheck>> argPort;
  std::optional<TypedWritePort<ResultT, SkipTypeCheck>> resultPort;
  std::function<ResultT(const ArgT &)> userCallback;
};

/// Partial specialization: void argument, typed result.
template <typename ResultT, bool SkipTypeCheck>
class TypedCallback<void, ResultT, SkipTypeCheck> {
public:
  // NOLINTNEXTLINE(google-explicit-constructor)
  TypedCallback(services::CallService::Callback *cb) : inner(cb) {}
  TypedCallback(const TypedCallback &) = delete;
  TypedCallback &operator=(const TypedCallback &) = delete;

  void connect(std::function<ResultT()> callback, bool quick = false) {
    (void)quick;
    if (!inner)
      detail::throwNullCallback();
    if (argPort)
      detail::throwCallbackAlreadyConnected();
    argPort.emplace(&inner->getRawRead("arg"));
    resultPort.emplace(&inner->getRawWrite("result"));
    auto opts = detail::typedFunctionConnectOptions();
    resultPort->connect(opts);
    userCallback = std::move(callback);
    argPort->connect(
        [this]() -> bool {
          ResultT result = userCallback();
          resultPort->write(result);
          return true;
        },
        opts);
  }

  services::CallService::Callback &raw() { return *inner; }
  const services::CallService::Callback &raw() const { return *inner; }

private:
  services::CallService::Callback *inner;
  std::optional<TypedReadPort<void>> argPort;
  std::optional<TypedWritePort<ResultT, SkipTypeCheck>> resultPort;
  std::function<ResultT()> userCallback;
};

/// Partial specialization: typed argument, void result.
template <typename ArgT, bool SkipTypeCheck>
class TypedCallback<ArgT, void, SkipTypeCheck> {
public:
  // NOLINTNEXTLINE(google-explicit-constructor)
  TypedCallback(services::CallService::Callback *cb) : inner(cb) {}
  TypedCallback(const TypedCallback &) = delete;
  TypedCallback &operator=(const TypedCallback &) = delete;

  void connect(std::function<void(const ArgT &)> callback, bool quick = false) {
    (void)quick;
    if (!inner)
      detail::throwNullCallback();
    if (argPort)
      detail::throwCallbackAlreadyConnected();
    argPort.emplace(&inner->getRawRead("arg"));
    resultPort.emplace(&inner->getRawWrite("result"));
    auto opts = detail::typedFunctionConnectOptions();
    resultPort->connect(opts);
    userCallback = std::move(callback);
    argPort->connect(
        [this](const ArgT &arg) -> bool {
          userCallback(arg);
          resultPort->write();
          return true;
        },
        opts);
  }

  services::CallService::Callback &raw() { return *inner; }
  const services::CallService::Callback &raw() const { return *inner; }

private:
  services::CallService::Callback *inner;
  std::optional<TypedReadPort<ArgT, SkipTypeCheck>> argPort;
  std::optional<TypedWritePort<void>> resultPort;
  std::function<void(const ArgT &)> userCallback;
};

/// Full specialization: void argument, void result.
template <bool SkipTypeCheck>
class TypedCallback<void, void, SkipTypeCheck> {
public:
  // NOLINTNEXTLINE(google-explicit-constructor)
  TypedCallback(services::CallService::Callback *cb) : inner(cb) {}
  TypedCallback(const TypedCallback &) = delete;
  TypedCallback &operator=(const TypedCallback &) = delete;

  void connect(std::function<void()> callback, bool quick = false) {
    (void)quick;
    if (!inner)
      detail::throwNullCallback();
    if (argPort)
      detail::throwCallbackAlreadyConnected();
    argPort.emplace(&inner->getRawRead("arg"));
    resultPort.emplace(&inner->getRawWrite("result"));
    auto opts = detail::typedFunctionConnectOptions();
    resultPort->connect(opts);
    userCallback = std::move(callback);
    argPort->connect(
        [this]() -> bool {
          userCallback();
          resultPort->write();
          return true;
        },
        opts);
  }

  services::CallService::Callback &raw() { return *inner; }
  const services::CallService::Callback &raw() const { return *inner; }

private:
  services::CallService::Callback *inner;
  std::optional<TypedReadPort<void>> argPort;
  std::optional<TypedWritePort<void>> resultPort;
  std::function<void()> userCallback;
};

//===----------------------------------------------------------------------===//
// IndexedPorts<T> — immutable wrapper over std::map<int, T>.
//
// Constructed once from a moved std::map (populated via `try_emplace` so T
// need not be movable or default-constructible). After construction the
// contents cannot be altered: `operator[]` delegates to `at()` and only const
// iterators are exposed.
//===----------------------------------------------------------------------===//

template <typename T>
class IndexedPorts {
public:
  explicit IndexedPorts(std::map<int, T> &&ports) : ports_(std::move(ports)) {}
  IndexedPorts(IndexedPorts &&) = default;
  IndexedPorts &operator=(IndexedPorts &&) = default;
  IndexedPorts(const IndexedPorts &) = delete;
  IndexedPorts &operator=(const IndexedPorts &) = delete;

  const T &operator[](int idx) const { return ports_.at(idx); }
  auto begin() const { return ports_.begin(); }
  auto end() const { return ports_.end(); }
  size_t size() const { return ports_.size(); }
  bool contains(int idx) const { return ports_.count(idx) > 0; }

private:
  std::map<int, T> ports_;
};

//===----------------------------------------------------------------------===//
// Port-lookup helpers: findPortOrThrow, findPortAsOrThrow, findPortIndices.
//===----------------------------------------------------------------------===//

/// Look up a BundlePort by AppID in `module`. Throws AcceleratorMismatchError
/// if the port is not found.
inline BundlePort &findPortOrThrow(HWModule *module, const AppID &id) {
  const auto &portIndex = module->getPorts();
  auto it = portIndex.find(id);
  if (it == portIndex.end())
    throw AcceleratorMismatchError("Expected port '" + id.toString() +
                                   "' not found in module");
  return it->second;
}

/// Look up a BundlePort by AppID and cast it to `T`. Throws
/// AcceleratorMismatchError if the port is missing or has the wrong runtime
/// type.
template <typename T>
T *findPortAsOrThrow(HWModule *module, const AppID &id) {
  BundlePort &port = findPortOrThrow(module, id);
  T *result = port.getAs<T>();
  if (!result)
    throw AcceleratorMismatchError("Port '" + id.toString() +
                                   "' has unexpected type (expected " +
                                   typeid(T).name() + ")");
  return result;
}

/// Return a sorted vector of the `idx` values for every port whose AppID name
/// matches `name`. Ports without an index are ignored.
inline std::vector<uint32_t> findPortIndices(HWModule *module,
                                             const std::string &name) {
  std::vector<uint32_t> indices;
  for (const auto &[appid, port] : module->getPorts())
    if (appid.name == name && appid.idx.has_value())
      indices.push_back(appid.idx.value());
  std::sort(indices.begin(), indices.end());
  return indices;
}

} // namespace esi

#endif // ESI_TYPED_PORTS_H
