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

#include "esi/Ports.h"
#include "esi/Services.h"
#include "esi/Types.h"

#include <cstdint>
#include <cstring>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
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
/// the typed callback without maintaining any retry queue of its own.
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

    MessageData scratch;
    const MessageData &flat = getMessageDataRef<T>(*msg, scratch);
    auto typed = std::make_unique<T>(fromMessageData<T>(flat, wireInfo));
    if (!output(typed))
      return false;

    msg.reset();
    return true;
  }

  bool poke() { return true; }

private:
  OutputCallback output;
  WireInfo wireInfo;
  std::mutex mutex;
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

    enqueueDecodedOutputsLocked(decoded);
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

  void enqueueDecodedOutputsLocked(DecodedOutputs &decoded) {
    for (std::unique_ptr<T> &value : decoded) {
      if (!value)
        throw std::runtime_error(
            "QueuedDecodeTypeDeserializer::push: null decoded output");
      pendingOutputs.push(std::move(value));
    }
  }

  OutputCallback output;
  std::queue<std::unique_ptr<T>> pendingOutputs;
  std::mutex mutex;
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
      verifyTypeCompatibility<T>(inner->getType());
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
template <typename T>
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
    auto nextDeserializer =
        makeDeserializer([this](std::unique_ptr<T> &value) -> bool {
          return pollingState->enqueue(value);
        });
    inner->connect(
        [nextDeserializer](std::unique_ptr<SegmentedMessageData> &msg) -> bool {
          return nextDeserializer->push(msg);
        },
        opts);
    deserializer = nextDeserializer;
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
    auto nextDeserializer = makeDeserializer(std::move(callback));
    deserializer = nextDeserializer;
    inner->connect(
        [nextDeserializer](std::unique_ptr<SegmentedMessageData> &msg) -> bool {
          return nextDeserializer->push(msg);
        },
        opts);
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

    auto activeDeserializer = deserializer;
    if (activeDeserializer)
      activeDeserializer->poke();
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
    deserializer.reset();
    pollingState.reset();
  }
  bool isConnected() const { return inner && inner->isConnected(); }

  ReadChannelPort &raw() { return *inner; }
  const ReadChannelPort &raw() const { return *inner; }

private:
  using Deserializer = detail::DeserializerFor<T>;
  enum Mode { Disconnected, Callback, Polling };
  using PollingState = detail::PollingBuffer<std::unique_ptr<T>>;

  void prepareConnect() {
    if (mode != Mode::Disconnected)
      throw std::runtime_error("Channel already connected");
    if constexpr (has_esi_id_v<T>) {
      verifyTypeCompatibility<T>(inner->getType());
    } else if constexpr (!detail::has_type_deserializer_v<T>) {
      verifyTypeCompatibility<T>(inner->getType());
      wireInfo_ = getWireInfo(inner->getType());
    }
  }

  std::shared_ptr<Deserializer>
  makeDeserializer(detail::DeserializerOutputCallback<T> callback) {
    if constexpr (detail::has_type_deserializer_v<T>) {
      return std::make_shared<Deserializer>(std::move(callback));
    } else {
      return std::make_shared<Deserializer>(std::move(callback), wireInfo_);
    }
  }

  ReadChannelPort *inner;
  WireInfo wireInfo_;
  Mode mode = Mode::Disconnected;
  uint64_t maxDataQueueMsgs = ReadChannelPort::DefaultMaxDataQueueMsgs;
  std::shared_ptr<Deserializer> deserializer;
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
// Non-owning wrapper around FuncService::Function that provides strongly-typed
// call() and connect() APIs. Implicitly constructible from Function* (the
// return type of BundlePort::getAs<FuncService::Function>()).
//===----------------------------------------------------------------------===//

template <typename ArgT, typename ResultT, bool SkipTypeCheck = false>
class TypedFunction {
public:
  /// Implicit conversion from Function* (returned by getAs<>()).
  // NOLINTNEXTLINE(google-explicit-constructor)
  TypedFunction(services::FuncService::Function *func) : inner(func) {}

  void connect() {
    if (!inner)
      throw AcceleratorMismatchError(
          "TypedFunction: null Function pointer (getAs failed or wrong type)");
    if constexpr (!SkipTypeCheck) {
      verifyTypeCompatibility<ArgT>(inner->getArgType());
      verifyTypeCompatibility<ResultT>(inner->getResultType());
    }
    argWireInfo_ = getWireInfo(inner->getArgType());
    resWireInfo_ = getWireInfo(inner->getResultType());
    inner->connect(ChannelPort::ConnectOptions(/*bufferSize=*/std::nullopt,
                                               /*translateMessage=*/false));
  }

  std::future<ResultT> call(const ArgT &arg) {
    WireInfo rwb = resWireInfo_;
    auto f = inner->call(toMessageData(arg, argWireInfo_));
    return std::async(std::launch::deferred,
                      [fut = std::move(f), rwb]() mutable -> ResultT {
                        MessageData data = fut.get();
                        return fromMessageData<ResultT>(data, rwb);
                      });
  }

  services::FuncService::Function &raw() { return *inner; }
  const services::FuncService::Function &raw() const { return *inner; }

private:
  services::FuncService::Function *inner;
  WireInfo argWireInfo_;
  WireInfo resWireInfo_;
};

/// Partial specialization: void argument, typed result.
template <typename ResultT, bool SkipTypeCheck>
class TypedFunction<void, ResultT, SkipTypeCheck> {
public:
  // NOLINTNEXTLINE(google-explicit-constructor)
  TypedFunction(services::FuncService::Function *func) : inner(func) {}

  void connect() {
    if (!inner)
      throw AcceleratorMismatchError(
          "TypedFunction: null Function pointer (getAs failed or wrong type)");
    if constexpr (!SkipTypeCheck) {
      verifyTypeCompatibility<void>(inner->getArgType());
      verifyTypeCompatibility<ResultT>(inner->getResultType());
    }
    inner->connect(ChannelPort::ConnectOptions(/*bufferSize=*/std::nullopt,
                                               /*translateMessage=*/false));
  }

  std::future<ResultT> call() {
    uint8_t zero = 0;
    const Type *resType = inner->getResultType();
    auto f = inner->call(MessageData(&zero, 1));
    return std::async(std::launch::deferred,
                      [fut = std::move(f), resType]() mutable -> ResultT {
                        MessageData data = fut.get();
                        return fromMessageData<ResultT>(data, resType);
                      });
  }

  services::FuncService::Function &raw() { return *inner; }
  const services::FuncService::Function &raw() const { return *inner; }

private:
  services::FuncService::Function *inner;
};

/// Partial specialization: typed argument, void result.
template <typename ArgT, bool SkipTypeCheck>
class TypedFunction<ArgT, void, SkipTypeCheck> {
public:
  // NOLINTNEXTLINE(google-explicit-constructor)
  TypedFunction(services::FuncService::Function *func) : inner(func) {}

  void connect() {
    if (!inner)
      throw AcceleratorMismatchError(
          "TypedFunction: null Function pointer (getAs failed or wrong type)");
    if constexpr (!SkipTypeCheck) {
      verifyTypeCompatibility<ArgT>(inner->getArgType());
      verifyTypeCompatibility<void>(inner->getResultType());
    }
    argWireInfo_ = getWireInfo(inner->getArgType());
    inner->connect(ChannelPort::ConnectOptions(/*bufferSize=*/std::nullopt,
                                               /*translateMessage=*/false));
  }

  std::future<void> call(const ArgT &arg) {
    auto f = inner->call(toMessageData(arg, argWireInfo_));
    return std::async(std::launch::deferred,
                      [fut = std::move(f)]() mutable -> void { fut.get(); });
  }

  services::FuncService::Function &raw() { return *inner; }
  const services::FuncService::Function &raw() const { return *inner; }

private:
  services::FuncService::Function *inner;
  WireInfo argWireInfo_;
};

/// Full specialization: void argument, void result.
template <bool SkipTypeCheck>
class TypedFunction<void, void, SkipTypeCheck> {
public:
  // NOLINTNEXTLINE(google-explicit-constructor)
  TypedFunction(services::FuncService::Function *func) : inner(func) {}

  void connect() {
    if (!inner)
      throw AcceleratorMismatchError(
          "TypedFunction: null Function pointer (getAs failed or wrong type)");
    if constexpr (!SkipTypeCheck) {
      verifyTypeCompatibility<void>(inner->getArgType());
      verifyTypeCompatibility<void>(inner->getResultType());
    }
    inner->connect(ChannelPort::ConnectOptions(/*bufferSize=*/std::nullopt,
                                               /*translateMessage=*/false));
  }

  std::future<void> call() {
    uint8_t zero = 0;
    auto f = inner->call(MessageData(&zero, 1));
    return std::async(std::launch::deferred,
                      [fut = std::move(f)]() mutable -> void { fut.get(); });
  }

  services::FuncService::Function &raw() { return *inner; }
  const services::FuncService::Function &raw() const { return *inner; }

private:
  services::FuncService::Function *inner;
};

//===----------------------------------------------------------------------===//
// TypedCallback<ArgT, ResultT>
//
// Non-owning wrapper around CallService::Callback that provides strongly-typed
// connect() and automatic MessageData conversion. Implicitly constructible
// from Callback* (the return type of BundlePort::getAs<Callback>()).
//===----------------------------------------------------------------------===//

template <typename ArgT, typename ResultT, bool SkipTypeCheck = false>
class TypedCallback {
public:
  // NOLINTNEXTLINE(google-explicit-constructor)
  TypedCallback(services::CallService::Callback *cb) : inner(cb) {}

  void connect(std::function<ResultT(const ArgT &)> callback,
               bool quick = false) {
    if (!inner)
      throw AcceleratorMismatchError(
          "TypedCallback: null Callback pointer (getAs failed or wrong type)");
    if constexpr (!SkipTypeCheck) {
      verifyTypeCompatibility<ArgT>(inner->getArgType());
      verifyTypeCompatibility<ResultT>(inner->getResultType());
    }
    inner->connect(
        [cb = std::move(callback), argType = inner->getArgType(),
         resType = inner->getResultType()](
            const MessageData &argData) -> MessageData {
          ResultT result = cb(fromMessageData<ArgT>(argData, argType));
          return toMessageData(result, resType);
        },
        quick,
        ChannelPort::ConnectOptions(/*bufferSize=*/std::nullopt,
                                    /*translateMessage=*/false));
  }

  services::CallService::Callback &raw() { return *inner; }
  const services::CallService::Callback &raw() const { return *inner; }

private:
  services::CallService::Callback *inner;
};

/// Partial specialization: void argument, typed result.
template <typename ResultT, bool SkipTypeCheck>
class TypedCallback<void, ResultT, SkipTypeCheck> {
public:
  // NOLINTNEXTLINE(google-explicit-constructor)
  TypedCallback(services::CallService::Callback *cb) : inner(cb) {}

  void connect(std::function<ResultT()> callback, bool quick = false) {
    if (!inner)
      throw AcceleratorMismatchError(
          "TypedCallback: null Callback pointer (getAs failed or wrong type)");
    if constexpr (!SkipTypeCheck) {
      verifyTypeCompatibility<void>(inner->getArgType());
      verifyTypeCompatibility<ResultT>(inner->getResultType());
    }
    WireInfo rwb = getWireInfo(inner->getResultType());
    inner->connect(
        [cb = std::move(callback), rwb](const MessageData &) -> MessageData {
          ResultT result = cb();
          return toMessageData(result, rwb);
        },
        quick,
        ChannelPort::ConnectOptions(/*bufferSize=*/std::nullopt,
                                    /*translateMessage=*/false));
  }

  services::CallService::Callback &raw() { return *inner; }
  const services::CallService::Callback &raw() const { return *inner; }

private:
  services::CallService::Callback *inner;
};

/// Partial specialization: typed argument, void result.
template <typename ArgT, bool SkipTypeCheck>
class TypedCallback<ArgT, void, SkipTypeCheck> {
public:
  // NOLINTNEXTLINE(google-explicit-constructor)
  TypedCallback(services::CallService::Callback *cb) : inner(cb) {}

  void connect(std::function<void(const ArgT &)> callback, bool quick = false) {
    if (!inner)
      throw AcceleratorMismatchError(
          "TypedCallback: null Callback pointer (getAs failed or wrong type)");
    if constexpr (!SkipTypeCheck) {
      verifyTypeCompatibility<ArgT>(inner->getArgType());
      verifyTypeCompatibility<void>(inner->getResultType());
    }
    inner->connect(
        [cb = std::move(callback), argType = inner->getArgType()](
            const MessageData &argData) -> MessageData {
          cb(fromMessageData<ArgT>(argData, argType));
          uint8_t zero = 0;
          return MessageData(&zero, 1);
        },
        quick,
        ChannelPort::ConnectOptions(/*bufferSize=*/std::nullopt,
                                    /*translateMessage=*/false));
  }

  services::CallService::Callback &raw() { return *inner; }
  const services::CallService::Callback &raw() const { return *inner; }

private:
  services::CallService::Callback *inner;
};

/// Full specialization: void argument, void result.
template <bool SkipTypeCheck>
class TypedCallback<void, void, SkipTypeCheck> {
public:
  // NOLINTNEXTLINE(google-explicit-constructor)
  TypedCallback(services::CallService::Callback *cb) : inner(cb) {}

  void connect(std::function<void()> callback, bool quick = false) {
    if (!inner)
      throw AcceleratorMismatchError(
          "TypedCallback: null Callback pointer (getAs failed or wrong type)");
    if constexpr (!SkipTypeCheck) {
      verifyTypeCompatibility<void>(inner->getArgType());
      verifyTypeCompatibility<void>(inner->getResultType());
    }
    inner->connect(
        [cb = std::move(callback)](const MessageData &) -> MessageData {
          cb();
          uint8_t zero = 0;
          return MessageData(&zero, 1);
        },
        quick,
        ChannelPort::ConnectOptions(/*bufferSize=*/std::nullopt,
                                    /*translateMessage=*/false));
  }

  services::CallService::Callback &raw() { return *inner; }
  const services::CallService::Callback &raw() const { return *inner; }

private:
  services::CallService::Callback *inner;
};

} // namespace esi

#endif // ESI_TYPED_PORTS_H
