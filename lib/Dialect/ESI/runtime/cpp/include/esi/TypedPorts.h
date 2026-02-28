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
#include <functional>
#include <future>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>

namespace esi {

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
    throw std::runtime_error("Port type is null");

  // Unwrap TypeAliasType to get the inner type for verification.
  if (auto *alias = dynamic_cast<const TypeAliasType *>(portType))
    portType = alias->getInnerType();

  if constexpr (has_esi_id_v<T>) {
    // Highest priority: user-defined ESI ID string comparison.
    if (std::string_view(portType->getID()) != T::_ESI_ID)
      throw std::runtime_error("ESI type mismatch: C++ type has _ESI_ID '" +
                               std::string(T::_ESI_ID) +
                               "' but port type is '" +
                               portType->toString(/*oneLine=*/true) + "'");
  } else if constexpr (std::is_void_v<T>) {
    if (!dynamic_cast<const VoidType *>(portType))
      throw std::runtime_error("ESI type mismatch: expected VoidType for "
                               "void, but port type is '" +
                               portType->toString(/*oneLine=*/true) + "'");
  } else if constexpr (std::is_same_v<T, bool>) {
    // bool maps to signless i1, which is BitsType with width <= 1.
    auto *bits = dynamic_cast<const BitsType *>(portType);
    if (!bits || bits->getWidth() > 1)
      throw std::runtime_error(
          "ESI type mismatch: expected BitsType with width <= 1 for "
          "bool, but port type is '" +
          portType->toString(/*oneLine=*/true) + "'");
  } else if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
    auto *sint = dynamic_cast<const SIntType *>(portType);
    if (!sint)
      throw std::runtime_error(
          "ESI type mismatch: expected SIntType for signed integer, "
          "but port type is '" +
          portType->toString(/*oneLine=*/true) + "'");
    if (sint->getWidth() > sizeof(T) * 8)
      throw std::runtime_error(
          "ESI type mismatch: SIntType width " +
          std::to_string(sint->getWidth()) + " does not fit in " +
          std::to_string(sizeof(T) * 8) + "-bit signed integer");
    // Require closest-size match: reject if a smaller C++ type would suffice.
    if (sizeof(T) > 1 && sint->getWidth() <= (sizeof(T) / 2) * 8)
      throw std::runtime_error("ESI type mismatch: SIntType width " +
                               std::to_string(sint->getWidth()) +
                               " should use a smaller C++ type than " +
                               std::to_string(sizeof(T) * 8) + "-bit");
  } else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T>) {
    // Accept UIntType (uiM) or BitsType (iM, signless).
    auto *uintPort = dynamic_cast<const UIntType *>(portType);
    auto *bits = dynamic_cast<const BitsType *>(portType);
    if (!uintPort && !bits)
      throw std::runtime_error(
          "ESI type mismatch: expected UIntType or BitsType for unsigned "
          "integer, but port type is '" +
          portType->toString(/*oneLine=*/true) + "'");
    uint64_t width = uintPort ? uintPort->getWidth() : bits->getWidth();
    if (width > sizeof(T) * 8)
      throw std::runtime_error("ESI type mismatch: bit width " +
                               std::to_string(width) + " does not fit in " +
                               std::to_string(sizeof(T) * 8) +
                               "-bit unsigned integer");
    // Require closest-size match: reject if a smaller C++ type would suffice.
    if (sizeof(T) > 1 && width <= (sizeof(T) / 2) * 8)
      throw std::runtime_error("ESI type mismatch: bit width " +
                               std::to_string(width) +
                               " should use a smaller C++ type than " +
                               std::to_string(sizeof(T) * 8) + "-bit");
  } else {
    throw std::runtime_error(
        std::string("Cannot verify type compatibility for C++ type '") +
        typeid(T).name() + "' against ESI port type '" +
        portType->toString(/*oneLine=*/true) + "'");
  }
}

//===----------------------------------------------------------------------===//
// TypedWritePort<T, CheckValue>
//
// When CheckValue is true, write() and tryWrite() verify that the value fits
// in the ESI port's actual bit width before sending.
//===----------------------------------------------------------------------===//

template <typename T, bool CheckValue = false>
class TypedWritePort {
public:
  explicit TypedWritePort(WriteChannelPort &port) : inner(port) {}

  void connect(const ChannelPort::ConnectOptions &opts = {}) {
    verifyTypeCompatibility<T>(inner.getType());
    inner.connect(opts);
  }

  void write(const T &data) {
    if constexpr (CheckValue)
      checkValueRange(data);
    // MessageData::from takes T& (non-const); safe to cast since it only reads.
    inner.write(MessageData::from(const_cast<T &>(data)));
  }

  bool tryWrite(const T &data) {
    if constexpr (CheckValue)
      checkValueRange(data);
    return inner.tryWrite(MessageData::from(const_cast<T &>(data)));
  }

  bool flush() { return inner.flush(); }
  void disconnect() { inner.disconnect(); }
  bool isConnected() const { return inner.isConnected(); }

  WriteChannelPort &raw() { return inner; }
  const WriteChannelPort &raw() const { return inner; }

private:
  WriteChannelPort &inner;

  void checkValueRange(const T &data) {
    static_assert(std::is_integral_v<T>,
                  "Value range checking only supported for integral types");
    auto *bvType = dynamic_cast<const BitVectorType *>(inner.getType());
    if (!bvType)
      return;
    uint64_t width = bvType->getWidth();
    if (width >= sizeof(T) * 8)
      return; // Full-width; any value is valid.
    if constexpr (std::is_signed_v<T>) {
      int64_t minVal = -(int64_t(1) << (width - 1));
      int64_t maxVal = (int64_t(1) << (width - 1)) - 1;
      if (data < minVal || data > maxVal)
        throw std::runtime_error("Value " + std::to_string(data) +
                                 " out of range for " + std::to_string(width) +
                                 "-bit signed type");
    } else {
      uint64_t maxVal = (uint64_t(1) << width) - 1;
      if (static_cast<uint64_t>(data) > maxVal)
        throw std::runtime_error("Value " + std::to_string(data) +
                                 " out of range for " + std::to_string(width) +
                                 "-bit unsigned type");
    }
  }
};

/// Specialization for void — write takes no data argument.
template <>
class TypedWritePort<void> {
public:
  explicit TypedWritePort(WriteChannelPort &port) : inner(port) {}

  void connect(const ChannelPort::ConnectOptions &opts = {}) {
    verifyTypeCompatibility<void>(inner.getType());
    inner.connect(opts);
  }

  void write() {
    uint8_t zero = 0;
    inner.write(MessageData(&zero, 1));
  }

  bool tryWrite() {
    uint8_t zero = 0;
    return inner.tryWrite(MessageData(&zero, 1));
  }

  bool flush() { return inner.flush(); }
  void disconnect() { inner.disconnect(); }
  bool isConnected() const { return inner.isConnected(); }

  WriteChannelPort &raw() { return inner; }
  const WriteChannelPort &raw() const { return inner; }

private:
  WriteChannelPort &inner;
};

//===----------------------------------------------------------------------===//
// TypedReadPort<T>
//===----------------------------------------------------------------------===//

template <typename T>
class TypedReadPort {
public:
  explicit TypedReadPort(ReadChannelPort &port) : inner(port) {}

  void connect(const ChannelPort::ConnectOptions &opts = {}) {
    verifyTypeCompatibility<T>(inner.getType());
    inner.connect(opts);
  }

  void connect(std::function<bool(T)> callback,
               const ChannelPort::ConnectOptions &opts = {}) {
    verifyTypeCompatibility<T>(inner.getType());
    inner.connect([cb = std::move(callback)](
                      MessageData data) -> bool { return cb(*data.as<T>()); },
                  opts);
  }

  T read() {
    MessageData outData;
    inner.read(outData);
    return *outData.as<T>();
  }

  std::future<T> readAsync() {
    auto innerFuture = inner.readAsync();
    return std::async(std::launch::deferred,
                      [f = std::move(innerFuture)]() mutable -> T {
                        MessageData data = f.get();
                        return *data.as<T>();
                      });
  }

  void disconnect() { inner.disconnect(); }
  bool isConnected() const { return inner.isConnected(); }

  ReadChannelPort &raw() { return inner; }
  const ReadChannelPort &raw() const { return inner; }

private:
  ReadChannelPort &inner;
};

/// Specialization for void — read discards data and returns nothing.
template <>
class TypedReadPort<void> {
public:
  explicit TypedReadPort(ReadChannelPort &port) : inner(port) {}

  void connect(const ChannelPort::ConnectOptions &opts = {}) {
    verifyTypeCompatibility<void>(inner.getType());
    inner.connect(opts);
  }

  void connect(std::function<bool()> callback,
               const ChannelPort::ConnectOptions &opts = {}) {
    verifyTypeCompatibility<void>(inner.getType());
    inner.connect(
        [cb = std::move(callback)](MessageData) -> bool { return cb(); }, opts);
  }

  void read() {
    MessageData outData;
    inner.read(outData);
    // Discard data.
  }

  std::future<void> readAsync() {
    auto innerFuture = inner.readAsync();
    return std::async(std::launch::deferred,
                      [f = std::move(innerFuture)]() mutable -> void {
                        f.get(); // Discard data.
                      });
  }

  void disconnect() { inner.disconnect(); }
  bool isConnected() const { return inner.isConnected(); }

  ReadChannelPort &raw() { return inner; }
  const ReadChannelPort &raw() const { return inner; }

private:
  ReadChannelPort &inner;
};

//===----------------------------------------------------------------------===//
// TypedFunction<ArgT, ResultT>
//
// Non-owning wrapper around FuncService::Function that provides strongly-typed
// call() and connect() APIs. Implicitly constructible from Function* (the
// return type of BundlePort::getAs<FuncService::Function>()).
//===----------------------------------------------------------------------===//

template <typename ArgT, typename ResultT>
class TypedFunction {
public:
  /// Implicit conversion from Function* (returned by getAs<>()).
  // NOLINTNEXTLINE(google-explicit-constructor)
  TypedFunction(services::FuncService::Function *func) : inner(func) {}

  void connect() {
    if (!inner)
      throw std::runtime_error(
          "TypedFunction: null Function pointer (getAs failed or wrong type)");
    verifyTypeCompatibility<ArgT>(inner->getArgType());
    verifyTypeCompatibility<ResultT>(inner->getResultType());
    inner->connect();
  }

  std::future<ResultT> call(const ArgT &arg) {
    auto f = inner->call(MessageData::from(const_cast<ArgT &>(arg)));
    return std::async(std::launch::deferred,
                      [fut = std::move(f)]() mutable -> ResultT {
                        MessageData data = fut.get();
                        return *data.as<ResultT>();
                      });
  }

  services::FuncService::Function &raw() { return *inner; }
  const services::FuncService::Function &raw() const { return *inner; }

private:
  services::FuncService::Function *inner;
};

/// Partial specialization: void argument, typed result.
template <typename ResultT>
class TypedFunction<void, ResultT> {
public:
  // NOLINTNEXTLINE(google-explicit-constructor)
  TypedFunction(services::FuncService::Function *func) : inner(func) {}

  void connect() {
    if (!inner)
      throw std::runtime_error(
          "TypedFunction: null Function pointer (getAs failed or wrong type)");
    verifyTypeCompatibility<void>(inner->getArgType());
    verifyTypeCompatibility<ResultT>(inner->getResultType());
    inner->connect();
  }

  std::future<ResultT> call() {
    uint8_t zero = 0;
    auto f = inner->call(MessageData(&zero, 1));
    return std::async(std::launch::deferred,
                      [fut = std::move(f)]() mutable -> ResultT {
                        MessageData data = fut.get();
                        return *data.as<ResultT>();
                      });
  }

  services::FuncService::Function &raw() { return *inner; }
  const services::FuncService::Function &raw() const { return *inner; }

private:
  services::FuncService::Function *inner;
};

/// Partial specialization: typed argument, void result.
template <typename ArgT>
class TypedFunction<ArgT, void> {
public:
  // NOLINTNEXTLINE(google-explicit-constructor)
  TypedFunction(services::FuncService::Function *func) : inner(func) {}

  void connect() {
    if (!inner)
      throw std::runtime_error(
          "TypedFunction: null Function pointer (getAs failed or wrong type)");
    verifyTypeCompatibility<ArgT>(inner->getArgType());
    verifyTypeCompatibility<void>(inner->getResultType());
    inner->connect();
  }

  std::future<void> call(const ArgT &arg) {
    auto f = inner->call(MessageData::from(const_cast<ArgT &>(arg)));
    return std::async(std::launch::deferred,
                      [fut = std::move(f)]() mutable -> void { fut.get(); });
  }

  services::FuncService::Function &raw() { return *inner; }
  const services::FuncService::Function &raw() const { return *inner; }

private:
  services::FuncService::Function *inner;
};

/// Full specialization: void argument, void result.
template <>
class TypedFunction<void, void> {
public:
  // NOLINTNEXTLINE(google-explicit-constructor)
  TypedFunction(services::FuncService::Function *func) : inner(func) {}

  void connect() {
    if (!inner)
      throw std::runtime_error(
          "TypedFunction: null Function pointer (getAs failed or wrong type)");
    verifyTypeCompatibility<void>(inner->getArgType());
    verifyTypeCompatibility<void>(inner->getResultType());
    inner->connect();
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

template <typename ArgT, typename ResultT>
class TypedCallback {
public:
  // NOLINTNEXTLINE(google-explicit-constructor)
  TypedCallback(services::CallService::Callback *cb) : inner(cb) {}

  void connect(std::function<ResultT(const ArgT &)> callback,
               bool quick = false) {
    if (!inner)
      throw std::runtime_error(
          "TypedCallback: null Callback pointer (getAs failed or wrong type)");
    verifyTypeCompatibility<ArgT>(inner->getArgType());
    verifyTypeCompatibility<ResultT>(inner->getResultType());
    inner->connect(
        [cb = std::move(callback)](const MessageData &argData) -> MessageData {
          ResultT result = cb(*argData.as<ArgT>());
          return MessageData::from(result);
        },
        quick);
  }

  services::CallService::Callback &raw() { return *inner; }
  const services::CallService::Callback &raw() const { return *inner; }

private:
  services::CallService::Callback *inner;
};

/// Partial specialization: void argument, typed result.
template <typename ResultT>
class TypedCallback<void, ResultT> {
public:
  // NOLINTNEXTLINE(google-explicit-constructor)
  TypedCallback(services::CallService::Callback *cb) : inner(cb) {}

  void connect(std::function<ResultT()> callback, bool quick = false) {
    if (!inner)
      throw std::runtime_error(
          "TypedCallback: null Callback pointer (getAs failed or wrong type)");
    verifyTypeCompatibility<void>(inner->getArgType());
    verifyTypeCompatibility<ResultT>(inner->getResultType());
    inner->connect(
        [cb = std::move(callback)](const MessageData &) -> MessageData {
          ResultT result = cb();
          return MessageData::from(result);
        },
        quick);
  }

  services::CallService::Callback &raw() { return *inner; }
  const services::CallService::Callback &raw() const { return *inner; }

private:
  services::CallService::Callback *inner;
};

/// Partial specialization: typed argument, void result.
template <typename ArgT>
class TypedCallback<ArgT, void> {
public:
  // NOLINTNEXTLINE(google-explicit-constructor)
  TypedCallback(services::CallService::Callback *cb) : inner(cb) {}

  void connect(std::function<void(const ArgT &)> callback, bool quick = false) {
    if (!inner)
      throw std::runtime_error(
          "TypedCallback: null Callback pointer (getAs failed or wrong type)");
    verifyTypeCompatibility<ArgT>(inner->getArgType());
    verifyTypeCompatibility<void>(inner->getResultType());
    inner->connect(
        [cb = std::move(callback)](const MessageData &argData) -> MessageData {
          cb(*argData.as<ArgT>());
          return MessageData();
        },
        quick);
  }

  services::CallService::Callback &raw() { return *inner; }
  const services::CallService::Callback &raw() const { return *inner; }

private:
  services::CallService::Callback *inner;
};

/// Full specialization: void argument, void result.
template <>
class TypedCallback<void, void> {
public:
  // NOLINTNEXTLINE(google-explicit-constructor)
  TypedCallback(services::CallService::Callback *cb) : inner(cb) {}

  void connect(std::function<void()> callback, bool quick = false) {
    if (!inner)
      throw std::runtime_error(
          "TypedCallback: null Callback pointer (getAs failed or wrong type)");
    verifyTypeCompatibility<void>(inner->getArgType());
    verifyTypeCompatibility<void>(inner->getResultType());
    inner->connect(
        [cb = std::move(callback)](const MessageData &) -> MessageData {
          cb();
          return MessageData();
        },
        quick);
  }

  services::CallService::Callback &raw() { return *inner; }
  const services::CallService::Callback &raw() const { return *inner; }

private:
  services::CallService::Callback *inner;
};

} // namespace esi

#endif // ESI_TYPED_PORTS_H
