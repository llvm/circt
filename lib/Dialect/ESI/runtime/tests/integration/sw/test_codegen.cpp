// Driver for the codegen / port-kind coverage integration test. Each probe
// below targets exactly one combination of port kind (function / callback /
// channel / MMIO / metric) and codegen path (typed scalar / typed struct /
// void specialization / indexed group). Each probe is small and self-checking
// so that a runtime regression in any one path lights up exactly one probe.
//
// The binary supports a ``--probe NAME`` flag that runs only one probe; the
// pytest harness uses this to surface each probe as a separate pytest test.
// With no ``--probe`` flag, every probe runs in sequence.

#include "test_codegen/CallServiceCallback.h"
#include "test_codegen/CallbackWindowedList.h"
#include "test_codegen/ChannelWindowedListRead.h"
#include "test_codegen/ChannelWindowedListWrite.h"
#include "test_codegen/CustomServiceDeclChannel.h"
#include "test_codegen/IndexedFuncGroup.h"
#include "test_codegen/MmioReadWrite.h"
#include "test_codegen/TelemetryMetric.h"
#include "test_codegen/TypedFuncArrayResult.h"
#include "test_codegen/TypedFuncMultiArg.h"
#include "test_codegen/TypedFuncNestedStruct.h"
#include "test_codegen/TypedFuncStruct.h"
#include "test_codegen/TypedFuncSubByteSigned.h"
#include "test_codegen/TypedFuncVoidArg.h"
#include "test_codegen/TypedFuncVoidResult.h"
#include "test_codegen/TypedFuncWindowedList.h"
#include "test_codegen/TypedReadChannelStruct.h"
#include "test_codegen/TypedWriteChannelByte.h"

#include "esi/Accelerator.h"
#include "esi/CLI.h"
#include "esi/Manifest.h"
#include "esi/Services.h"
#include "esi/TypedPorts.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace esi;

// Resolve a child instance by AppID name from the top-level accelerator.
static esi::HWModule *findInst(Accelerator *accel, const char *appidName) {
  auto it = accel->getChildren().find(AppID(appidName));
  if (it == accel->getChildren().end())
    throw std::runtime_error(std::string("test_codegen instance '") +
                             appidName + "' not found");
  return it->second;
}

//===----------------------------------------------------------------------===//
// Function: typed multi-arg call via emplace ctor.
//===----------------------------------------------------------------------===//
static void runTypedFuncMultiArg(Accelerator *accel) {
  esi_system::TypedFuncMultiArg mod(
      findInst(accel, "typed_func_multi_arg_inst"));
  auto c = mod.connect();

  // The emplace-style call() forwards its arguments into the generated
  // arg struct's constructor, so we never have to spell that struct out.
  uint32_t got = c->call(7u, 6u).get();
  if (got != 42u)
    throw std::runtime_error("typed_func_multi_arg: expected 42, got " +
                             std::to_string(got));
  std::cout << "typed_func_multi_arg ok\n";
}

//===----------------------------------------------------------------------===//
// Function: void argument (typed-result specialization).
//===----------------------------------------------------------------------===//
static void runTypedFuncVoidArg(Accelerator *accel) {
  esi_system::TypedFuncVoidArg mod(findInst(accel, "typed_func_void_arg_inst"));
  auto c = mod.connect();

  uint32_t got = c->call().get();
  if (got != 0xCAFEF00Du)
    throw std::runtime_error(
        "typed_func_void_arg: expected 0xCAFEF00D, got 0x" + toHex(got));
  std::cout << "typed_func_void_arg ok\n";
}

//===----------------------------------------------------------------------===//
// Function: void return (typed-arg specialization).
//===----------------------------------------------------------------------===//
static void runTypedFuncVoidResult(Accelerator *accel) {
  esi_system::TypedFuncVoidResult mod(
      findInst(accel, "typed_func_void_result_inst"));
  auto c = mod.connect();

  // Just asserts that the future resolves without throwing. A void result is
  // the wire-level zero byte; any failure to consume that byte would surface
  // here as a hung future or a deserializer exception.
  c->call(esi_system::AckArgs(0x5A, 0x1234)).get();
  std::cout << "typed_func_void_result ok\n";
}

//===----------------------------------------------------------------------===//
// Callback: HW-initiated call into the host (triggered via an MMIO write).
//===----------------------------------------------------------------------===//
static void runCallServiceCallback(Accelerator *accel) {
  esi_system::CallServiceCallback mod(
      findInst(accel, "call_service_callback_inst"));
  auto c = mod.connect();

  // Install the user callback. The handler stores what it saw and signals a
  // flag; the driver thread polls the flag (with a timeout) so this works
  // both for inline-from-callback-thread dispatch and for service-thread
  // dispatch.
  std::atomic<bool> got_call(false);
  esi_system::NotifyArgs seen{};
  c->callback.connect([&](const esi_system::NotifyArgs &a) {
    seen = a;
    got_call.store(true, std::memory_order_release);
  });

  // Trigger the callback by writing the payload to the MMIO command region
  // at offset 0x10. The HW module forwards the bottom 32 bits of the write
  // data into the callback as ``payload`` and uses a fixed ``tag = 0xA5``.
  constexpr uint32_t kPayload = 0xDEADBEEFu;
  c->trigger.write(0x10, static_cast<uint64_t>(kPayload));

  // Wait up to ~5s for the callback to fire.
  using clock = std::chrono::steady_clock;
  auto deadline = clock::now() + std::chrono::seconds(5);
  while (!got_call.load(std::memory_order_acquire) && clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  if (!got_call.load(std::memory_order_acquire))
    throw std::runtime_error(
        "call_service_callback: callback did not fire within timeout");

  if (seen.tag != 0xA5)
    throw std::runtime_error(
        "call_service_callback: wrong tag, expected 0xA5 got 0x" +
        toHex(static_cast<uint64_t>(seen.tag)));
  if (seen.payload != kPayload)
    throw std::runtime_error(
        "call_service_callback: wrong payload, expected 0x" + toHex(kPayload) +
        " got 0x" + toHex(seen.payload));
  std::cout << "call_service_callback ok\n";
}

//===----------------------------------------------------------------------===//
// To-host channel: TypedReadPort<EventStruct> polling.
//===----------------------------------------------------------------------===//
static void runTypedReadChannelStruct(Accelerator *accel) {
  esi_system::TypedReadChannelStruct mod(
      findInst(accel, "typed_read_channel_struct_inst"));
  auto c = mod.connect();

  // The constant on the HW side bounds how many events get pushed.
  constexpr size_t kNum = esi_system::TypedReadChannelStruct::num_events;
  for (size_t i = 1; i <= kNum; ++i) {
    auto ev = c->data.read();
    if (!ev)
      throw std::runtime_error(
          "typed_read_channel_struct: null read result at i=" +
          std::to_string(i));
    if (ev->ts != i)
      throw std::runtime_error(
          "typed_read_channel_struct: wrong ts at i=" + std::to_string(i) +
          ", got " + std::to_string(ev->ts));
    int32_t expected = -static_cast<int32_t>(i);
    if (ev->val != expected)
      throw std::runtime_error(
          "typed_read_channel_struct: wrong val at i=" + std::to_string(i) +
          ", got " + std::to_string(ev->val));
  }
  std::cout << "typed_read_channel_struct ok (" << kNum << " events)\n";
}

//===----------------------------------------------------------------------===//
// From-host channel: TypedWritePort<uint8_t> + MMIO read-back accumulator.
//===----------------------------------------------------------------------===//
static void runTypedWriteChannelByte(Accelerator *accel) {
  esi_system::TypedWriteChannelByte mod(
      findInst(accel, "typed_write_channel_byte_inst"));
  auto c = mod.connect();

  // Send a sequence and accumulate the expected XOR. The HW receiver is
  // always-ready and XORs every byte into a register whose value is exposed
  // via the ``accumulator`` MMIO read port.
  static constexpr uint8_t kBytes[] = {0x11, 0x22, 0x44, 0x88, 0x10, 0x55};
  uint8_t expected = 0;
  for (uint8_t b : kBytes) {
    c->data.write(b);
    expected ^= b;
  }

  // Poll the accumulator MMIO until it matches (or we time out). A small
  // poll loop covers the case where the last byte hasn't drained yet.
  using clock = std::chrono::steady_clock;
  auto deadline = clock::now() + std::chrono::seconds(2);
  uint8_t got = 0;
  while (clock::now() < deadline) {
    uint64_t resp = c->accumulator.read(0);
    got = static_cast<uint8_t>(resp & 0xff);
    if (got == expected)
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  if (got != expected)
    throw std::runtime_error(
        "typed_write_channel_byte: accumulator mismatch (expected 0x" +
        toHex(static_cast<uint64_t>(expected)) + ", got 0x" +
        toHex(static_cast<uint64_t>(got)) + ")");
  std::cout << "typed_write_channel_byte ok (acc=0x"
            << toHex(static_cast<uint64_t>(expected)) << ")\n";
}

//===----------------------------------------------------------------------===//
// MMIO region: read-write loopback at offset 0x10.
//===----------------------------------------------------------------------===//
static void runMmioReadWrite(Accelerator *accel) {
  esi_system::MmioReadWrite mod(findInst(accel, "mmio_read_write_inst"));
  auto c = mod.connect();

  // Write a 64-bit token, then read it back. The HW's storage register is
  // shared across all offsets, so ``offset`` here is just for completeness.
  constexpr uint64_t kToken = 0xA5A51234'56789ABCULL;
  c->region.write(0x10, kToken);
  uint64_t got = c->region.read(0x10);
  if (got != kToken)
    throw std::runtime_error("mmio_read_write: round-trip mismatch (wrote 0x" +
                             toHex(kToken) + ", read 0x" + toHex(got) + ")");
  std::cout << "mmio_read_write ok (round-trip 0x" << toHex(kToken) << ")\n";
}

//===----------------------------------------------------------------------===//
// Telemetry: free-running cycle counter is monotonic between reads.
//===----------------------------------------------------------------------===//
static void runTelemetryMetric(Accelerator *accel) {
  esi_system::TelemetryMetric mod(findInst(accel, "telemetry_metric_inst"));
  auto c = mod.connect();

  uint64_t first = c->cycleCount.readInt();
  // Sleep enough wall-time for the simulator to advance many cycles even
  // under heavy load.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  uint64_t second = c->cycleCount.readInt();

  if (second <= first)
    throw std::runtime_error(
        "telemetry_metric: counter did not advance (first=" +
        std::to_string(first) + ", second=" + std::to_string(second) + ")");
  std::cout << "telemetry_metric ok (advanced by " << (second - first) << ")\n";
}

//===----------------------------------------------------------------------===//
// Indexed function group: exercise every entry of IndexedPorts<TypedFunction>.
//===----------------------------------------------------------------------===//
static void runIndexedFuncGroup(Accelerator *accel) {
  // Single IndexedFuncGroup module exposes N typed-function ports under the
  // same appid name ``call`` with indices 0..N-1; codegen groups them into a
  // single ``IndexedPorts<TypedFunction<...>>`` member that the driver
  // iterates with ``c->call[idx]``.
  esi_system::IndexedFuncGroup mod(findInst(accel, "indexed_func_group_inst"));
  auto c = mod.connect();
  constexpr size_t kN = esi_system::IndexedFuncGroup::num_entries;
  for (uint32_t idx = 0; idx < kN; ++idx) {
    constexpr uint16_t kArg = 100;
    uint16_t got = c->call[idx](kArg).get();
    uint16_t expected = static_cast<uint16_t>(kArg + (idx + 1));
    if (got != expected)
      throw std::runtime_error("indexed_func_group[" + std::to_string(idx) +
                               "]: expected " + std::to_string(expected) +
                               ", got " + std::to_string(got));
  }
  std::cout << "indexed_func_group ok (" << kN << " entries)\n";
}

//===----------------------------------------------------------------------===//
// Custom-`@esi.ServiceDecl` raw-channel byte loopback. Exercises bundle ports
// backed by a custom service decl rather than the standard `ChannelService`,
// across two indexed instances. The HW also exposes void (Bits(0)) bundles
// for elaboration coverage; the C++ driver does not exercise them because
// the runtime's blocking ``ReadChannelPort::read`` does not surface a
// completion for zero-byte messages.
//===----------------------------------------------------------------------===//
static void runCustomServiceDeclChannel(Accelerator *accel, uint32_t idx) {
  auto it =
      accel->getChildren().find(AppID("custom_service_decl_channel", idx));
  if (it == accel->getChildren().end())
    throw std::runtime_error("custom_service_decl_channel[" +
                             std::to_string(idx) + "]: instance not found");
  esi_system::CustomServiceDeclChannel mod(it->second);
  auto c = mod.connect();

  // Byte channel: send a unique byte per instance and verify the echo so a
  // crossed-wires bug between the two CustomServiceDeclChannel instances
  // would be caught (same-AppID-name multi-instance regression).
  TypedWritePort<uint8_t> toHw(c->byte_in.getRawWrite("recv"));
  TypedReadPort<uint8_t> fromHw(c->byte_out.getRawRead("send"));
  toHw.connect();
  fromHw.connect();

  uint8_t sendVal = static_cast<uint8_t>(0x40 + idx);
  toHw.write(sendVal);
  std::unique_ptr<uint8_t> got = fromHw.read();
  if (!got || *got != sendVal)
    throw std::runtime_error(
        "custom_service_decl_channel[" + std::to_string(idx) +
        "]: byte loopback mismatch (sent 0x" +
        toHex(static_cast<uint64_t>(sendVal)) + ", got 0x" +
        toHex(static_cast<uint64_t>(got ? *got : 0u)) + ")");

  std::cout << "custom_service_decl_channel_" << idx << " ok (byte 0x"
            << toHex(static_cast<uint64_t>(sendVal)) << ")\n";
}

static void runCustomServiceDeclChannel0(Accelerator *accel) {
  runCustomServiceDeclChannel(accel, 0);
}
static void runCustomServiceDeclChannel1(Accelerator *accel) {
  runCustomServiceDeclChannel(accel, 1);
}

//===----------------------------------------------------------------------===//
// Typed function: small struct -> small struct.
//===----------------------------------------------------------------------===//
static void runTypedFuncStruct(Accelerator *accel) {
  esi_system::TypedFuncStruct mod(findInst(accel, "typed_func_struct_inst"));
  auto c = mod.connect();

  esi_system::StructArgs arg(0x1234, static_cast<int8_t>(-7));
  esi_system::StructResult res = c->call(arg).get();
  int8_t expectedX = static_cast<int8_t>(arg.b + 1);
  if (res.x != expectedX || res.y != arg.b)
    throw std::runtime_error(
        "typed_func_struct: wrong result (b=" + std::to_string(arg.b) +
        " x=" + std::to_string(res.x) + " y=" + std::to_string(res.y) + ")");
  std::cout << "typed_func_struct ok (b=" << (int)arg.b
            << " -> x=" << (int)res.x << " y=" << (int)res.y << ")\n";
}

//===----------------------------------------------------------------------===//
// Typed function: nested odd-bit-width struct round-trip.
//===----------------------------------------------------------------------===//
static void runTypedFuncNestedStruct(Accelerator *accel) {
  esi_system::TypedFuncNestedStruct mod(
      findInst(accel, "typed_func_nested_struct_inst"));
  auto c = mod.connect();

  esi_system::OddStruct arg;
  arg.a = 0xabc;
  arg.b = static_cast<int8_t>(-17);
  arg.inner.p = 5;
  arg.inner.q = static_cast<int8_t>(-7);
  arg.inner.r[0] = 3;
  arg.inner.r[1] = 4;

  esi_system::OddStruct res = c->call(arg).get();
  uint16_t expA = static_cast<uint16_t>(arg.a + 1);
  int8_t expB = static_cast<int8_t>(arg.b - 3);
  uint8_t expP = static_cast<uint8_t>(arg.inner.p + 5);
  int8_t expQ = static_cast<int8_t>(arg.inner.q + 2);
  uint8_t expR0 = static_cast<uint8_t>(arg.inner.r[0] + 1);
  uint8_t expR1 = static_cast<uint8_t>(arg.inner.r[1] + 2);
  if (res.a != expA || res.b != expB || res.inner.p != expP ||
      res.inner.q != expQ || res.inner.r[0] != expR0 || res.inner.r[1] != expR1)
    throw std::runtime_error("typed_func_nested_struct: result mismatch");
  std::cout << "typed_func_nested_struct ok (a=" << res.a << " b=" << (int)res.b
            << " p=" << (int)res.inner.p << " q=" << (int)res.inner.q << " r=["
            << (int)res.inner.r[0] << "," << (int)res.inner.r[1] << "])\n";
}

//===----------------------------------------------------------------------===//
// Typed function: ``si4 -> si4`` identity. Probes sign extension at a
// sub-byte width through the typed facade.
//===----------------------------------------------------------------------===//
static void runTypedFuncSubByteSigned(Accelerator *accel) {
  esi_system::TypedFuncSubByteSigned mod(
      findInst(accel, "typed_func_subbyte_signed_inst"));
  auto c = mod.connect();

  for (int8_t arg : {static_cast<int8_t>(5), static_cast<int8_t>(-3),
                     static_cast<int8_t>(-8), static_cast<int8_t>(7)}) {
    int8_t got = c->call(arg).get();
    if (got != arg)
      throw std::runtime_error(
          "typed_func_subbyte_signed: arg=" + std::to_string(arg) +
          " got=" + std::to_string(got));
  }
  std::cout << "typed_func_subbyte_signed ok (4 values)\n";
}

//===----------------------------------------------------------------------===//
// Typed function with an array result.
//===----------------------------------------------------------------------===//
static void runTypedFuncArrayResult(Accelerator *accel) {
  esi_system::TypedFuncArrayResult mod(
      findInst(accel, "typed_func_array_result_inst"));
  auto c = mod.connect();

  esi_system::TypedFuncArrayResult::callArgs arg{static_cast<int8_t>(-3)};
  esi_system::ArrayResult res = c->call(arg).get();
  int8_t a = res[0];
  int8_t b = res[1];
  int8_t expect0 = arg[0];
  int8_t expect1 = static_cast<int8_t>(arg[0] + 1);
  bool ok = (a == expect0 && b == expect1) || (a == expect1 && b == expect0);
  if (!ok)
    throw std::runtime_error("typed_func_array_result: result mismatch");
  std::cout << "typed_func_array_result ok ([" << (int)a << "," << (int)b
            << "])\n";
}

//===----------------------------------------------------------------------===//
// Typed function over a windowed list payload. Doubles each element of the
// input list and reads the result back as another serial-burst window.
// Exercises the auto serial<->parallel windowed-list converters and the
// `SerialListTypeDeserializer` end-to-end.
//===----------------------------------------------------------------------===//
static void runTypedFuncWindowedList(Accelerator *accel) {
  esi_system::TypedFuncWindowedList mod(
      findInst(accel, "typed_func_windowed_list_inst"));
  auto c = mod.connect();

  using ArgT = esi_system::TypedFuncWindowedList::callArgs;
  using ResT = esi_system::TypedFuncWindowedList::callResult;
  std::vector<esi_system::TransformListItem> input;
  for (uint32_t v : {3u, 5u, 7u, 9u, 11u})
    input.emplace_back(v);

  ArgT arg(input);
  ResT result = c->call(arg).get();

  if (result.data_count() != input.size())
    throw std::runtime_error(
        "typed_func_windowed_list: wrong result size (got " +
        std::to_string(result.data_count()) + ")");
  size_t i = 0;
  if (result.data_count() != input.size())
    throw std::runtime_error(
        "typed_func_windowed_list: wrong result size (got " +
        std::to_string(result.data_count()) + ")");
  for (const esi_system::TransformListItem &item : result.data()) {
    uint32_t expected = input[i].v + input[i].v;
    if (item.v != expected)
      throw std::runtime_error("typed_func_windowed_list: element " +
                               std::to_string(i) + " expected " +
                               std::to_string(expected) + ", got " +
                               std::to_string(item.v));
    ++i;
  }
  std::cout << "typed_func_windowed_list ok (" << input.size()
            << " items doubled)\n";
}

//===----------------------------------------------------------------------===//
// To-host channel of windowed list-with-header. Exercises the typed read path
// for serial-burst bulk transfers.
//===----------------------------------------------------------------------===//
static void runChannelWindowedListRead(Accelerator *accel) {
  esi_system::ChannelWindowedListRead mod(
      findInst(accel, "channel_windowed_list_read_inst"));
  auto c = mod.connect();

  using WinT = esi_system::ChannelWindowedListRead::dataData;

  // Arm one burst via the MMIO trigger. The HW only emits when triggered, so
  // free-running emission can't fill the host's polling buffer.
  c->trigger.write(0x10, 0u);

  std::unique_ptr<WinT> got = c->data.read();
  if (!got)
    throw std::runtime_error("channel_windowed_list_read: null read result");
  // The HW emits one burst with ``[10, 20, 30, 40]`` and ``tag = 0xCAFE``.
  static constexpr uint16_t kTag = 0xCAFE;
  static const uint32_t kExpected[] = {10u, 20u, 30u, 40u};
  if (got->tag() != kTag)
    throw std::runtime_error(
        "channel_windowed_list_read: wrong tag, expected 0x" +
        toHex(static_cast<uint64_t>(kTag)) + " got 0x" +
        toHex(static_cast<uint64_t>(got->tag())));
  if (got->items_count() != 4)
    throw std::runtime_error(
        "channel_windowed_list_read: wrong item count, expected 4 got " +
        std::to_string(got->items_count()));
  size_t i = 0;
  for (uint32_t v : got->items()) {
    if (v != kExpected[i])
      throw std::runtime_error("channel_windowed_list_read: element " +
                               std::to_string(i) + " expected " +
                               std::to_string(kExpected[i]) + " got " +
                               std::to_string(v));
    ++i;
  }
  std::cout << "channel_windowed_list_read ok (tag=0x"
            << toHex(static_cast<uint64_t>(kTag)) << ", items=[10,20,30,40])\n";
}

//===----------------------------------------------------------------------===//
// From-host channel of windowed list-with-header. Exercises the typed write
// path: the host constructs a complete burst from a header tag plus a list of
// items, and the HW AND-reduces each beat against the expected pattern. The
// driver verifies success via the ``match`` MMIO read region.
//===----------------------------------------------------------------------===//
static void runChannelWindowedListWrite(Accelerator *accel) {
  esi_system::ChannelWindowedListWrite mod(
      findInst(accel, "channel_windowed_list_write_inst"));
  auto c = mod.connect();

  using WinT = esi_system::ChannelWindowedListWrite::dataData;

  static constexpr uint16_t kTag = 0xCAFE;
  std::vector<uint32_t> items{10u, 20u, 30u, 40u};
  c->data.write(WinT(kTag, items));

  // Poll the match flag MMIO until the burst has been processed (or time
  // out). The HW updates the latch on the burst-end beat.
  using clock = std::chrono::steady_clock;
  auto deadline = clock::now() + std::chrono::seconds(5);
  uint64_t match = 0;
  while (clock::now() < deadline) {
    match = c->match.read(0);
    if (match & 1)
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  if (!(match & 1))
    throw std::runtime_error(
        "channel_windowed_list_write: HW did not report a match within "
        "timeout (got 0x" +
        toHex(match) + ")");
  std::cout << "channel_windowed_list_write ok (tag=0x"
            << toHex(static_cast<uint64_t>(kTag)) << ", items=[10,20,30,40])\n";
}

//===----------------------------------------------------------------------===//
// Callback with windowed list argument: HW sends a serial-burst windowed
// list (tag + items) into a host callback. Verifies that the
// `SerialListTypeDeserializer` works end-to-end through the
// `TypedCallback<WindowT, void>` path.
//===----------------------------------------------------------------------===//
static void runCallbackWindowedList(Accelerator *accel) {
  esi_system::CallbackWindowedList mod(
      findInst(accel, "callback_windowed_list_inst"));
  auto c = mod.connect();

  using WinT = esi_system::CallbackWindowedList::callbackArgs;

  std::atomic<bool> got_call(false);
  c->callback.connect([&](const WinT &arg) {
    static constexpr uint16_t kTag = 0xCAFE;
    static const uint32_t kExpected[] = {10u, 20u, 30u, 40u};
    if (arg.tag() != kTag)
      throw std::runtime_error(
          "callback_windowed_list: wrong tag, expected 0x" +
          toHex(static_cast<uint64_t>(kTag)) + " got 0x" +
          toHex(static_cast<uint64_t>(arg.tag())));
    if (arg.items_count() != 4)
      throw std::runtime_error(
          "callback_windowed_list: wrong item count, expected 4 got " +
          std::to_string(arg.items_count()));
    size_t i = 0;
    for (uint32_t v : arg.items()) {
      if (v != kExpected[i])
        throw std::runtime_error("callback_windowed_list: element " +
                                 std::to_string(i) + " expected " +
                                 std::to_string(kExpected[i]) + " got " +
                                 std::to_string(v));
      ++i;
    }

    std::cout << "callback_windowed_list ok (tag=0x"
              << toHex(static_cast<uint64_t>(kTag))
              << ", items=[10,20,30,40])\n";

    got_call.store(true, std::memory_order_release);
  });

  // Arm the burst via MMIO trigger.
  c->trigger.write(0x10, 0u);

  using clock = std::chrono::steady_clock;
  auto deadline = clock::now() + std::chrono::seconds(5);
  while (!got_call.load(std::memory_order_acquire) && clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  if (!got_call.load(std::memory_order_acquire))
    throw std::runtime_error(
        "callback_windowed_list: callback did not fire within timeout");
}

// Probe registry: maps probe names (as used by ``--probe`` and emitted in
// the success messages) to their driver function. The registry order
// determines the default ``--probe=all`` execution sequence.
using ProbeFn = void (*)(Accelerator *);
static const std::vector<std::pair<std::string, ProbeFn>> &probes() {
  static const std::vector<std::pair<std::string, ProbeFn>> kProbes = {
      {"typed_func_multi_arg", &runTypedFuncMultiArg},
      {"typed_func_void_arg", &runTypedFuncVoidArg},
      {"typed_func_void_result", &runTypedFuncVoidResult},
      {"call_service_callback", &runCallServiceCallback},
      {"typed_read_channel_struct", &runTypedReadChannelStruct},
      {"typed_write_channel_byte", &runTypedWriteChannelByte},
      {"mmio_read_write", &runMmioReadWrite},
      {"telemetry_metric", &runTelemetryMetric},
      {"indexed_func_group", &runIndexedFuncGroup},
      {"custom_service_decl_channel_0", &runCustomServiceDeclChannel0},
      {"custom_service_decl_channel_1", &runCustomServiceDeclChannel1},
      {"typed_func_struct", &runTypedFuncStruct},
      {"typed_func_nested_struct", &runTypedFuncNestedStruct},
      {"typed_func_subbyte_signed", &runTypedFuncSubByteSigned},
      {"typed_func_array_result", &runTypedFuncArrayResult},
      {"typed_func_windowed_list", &runTypedFuncWindowedList},
      {"channel_windowed_list_read", &runChannelWindowedListRead},
      {"channel_windowed_list_write", &runChannelWindowedListWrite},
      {"callback_windowed_list", &runCallbackWindowedList},
  };
  return kProbes;
}

int main(int argc, const char *argv[]) {
  CliParser cli("test-codegen");
  cli.description("Per-port-kind coverage tests for ESI runtime + facade "
                  "codegen. Run a single probe with --probe NAME or run all "
                  "probes (in registry order) by omitting the flag.");
  std::string probeName;
  cli.add_option("--probe", probeName,
                 "Run only the named probe. Without this flag, every probe "
                 "runs in sequence.");
  if (int rc = cli.esiParse(argc, argv))
    return rc;
  if (!cli.get_help_ptr()->empty())
    return 0;

  Context &ctxt = cli.getContext();
  AcceleratorConnection *conn = cli.connect();
  try {
    const auto &info = *conn->getService<services::SysInfo>();
    Manifest manifest(ctxt, info.getJsonManifest());
    Accelerator *accel = manifest.buildAccelerator(*conn);
    conn->getServiceThread()->addPoll(*accel);

    if (probeName.empty()) {
      // Default sequence: every probe in registry order.
      for (const auto &[name, fn] : probes())
        fn(accel);
    } else {
      ProbeFn fn = nullptr;
      for (const auto &[name, candidate] : probes()) {
        if (name == probeName) {
          fn = candidate;
          break;
        }
      }
      if (!fn) {
        std::cerr << "test-codegen: unknown probe '" << probeName << "'\n";
        conn->disconnect();
        return 2;
      }
      fn(accel);
    }

    conn->disconnect();
  } catch (std::exception &e) {
    ctxt.getLogger().error("test-codegen", e.what());
    conn->disconnect();
    return 1;
  }
  return 0;
}
