// Driver for the SerializationProbes integration test. Each function below
// targets exactly one of the on-the-wire serialization invariants the host
// must agree with hardware on (byte order, sign extension, struct-field
// order, sub-byte field packing, array element order). Mismatches surface
// as wrong-but-distinguishable output rather than coincidentally-correct
// answers; see the corresponding HW module's docstring for the rationale.

#include "serialization_probes/BitPackProbe.h"
#include "serialization_probes/ByteRotate1.h"
#include "serialization_probes/PackProbe.h"
#include "serialization_probes/SignProbe.h"
#include "serialization_probes/SignProbe13.h"
// Note: serialization_probes/ArrayProbe.h is intentionally not included.
// runArrayProbe drives the raw FuncService port directly to test the wire
// byte order; see the comment on that function for the rationale.

#include "probe_runner.h"

#include "esi/Accelerator.h"
#include "esi/Manifest.h"
#include "esi/Services.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace esi;

// Resolve a child instance by AppID name from the top-level accelerator.
// The probe modules are instantiated under named appids so the runtime
// hierarchy carries one instance per probe.
static esi::HWModule *findProbe(Accelerator *accel, const char *appidName) {
  auto it = accel->getChildren().find(AppID(appidName));
  if (it == accel->getChildren().end())
    throw std::runtime_error(std::string("probe instance '") + appidName +
                             "' not found in accelerator hierarchy");
  return it->second;
}

// The constant byte sequence shared by ``byte_pattern_const`` and
// ``byte_pattern_echo_eq``. Must stay in lock-step with ``_BYTE_PATTERN``
// in the PyCDE source.
static constexpr std::array<uint8_t, 8> kBytePattern = {0x12, 0x34, 0x56, 0x78,
                                                        0x9A, 0xBC, 0xDE, 0xF0};

static int runByteRotate1(Accelerator *accel) {
  esi_system::ByteRotate1 mod(findProbe(accel, "byte_rotate1_inst"));
  auto connected = mod.connect();

  uint64_t arg = 0x0102030405060708ULL;
  uint64_t expect = 0x0203040506070801ULL; // (arg << 8) | (arg >> 56).
  uint64_t got = connected->byte_rotate1(arg).get();
  if (got != expect)
    throw std::runtime_error("byte_rotate1 mismatch: got 0x" + toHex(got) +
                             " expected 0x" + toHex(expect));
  std::cout << "byte_rotate1 ok: 0x" << std::hex << got << std::dec << "\n";
  return 0;
}

// Look up a FuncService::Function port by AppID on a probe instance, skipping
// the generated facade. Used by the raw-byte probes below so the test depends
// only on what the runtime actually puts on (or reads off) the wire.
static services::FuncService::Function *findRawFunc(esi::HWModule *probe,
                                                    const char *portName) {
  auto it = probe->getPorts().find(AppID(portName));
  if (it == probe->getPorts().end())
    throw std::runtime_error(std::string("port '") + portName +
                             "' not found on probe instance");
  auto *func = it->second.getAs<services::FuncService::Function>();
  if (!func)
    throw std::runtime_error(std::string("port '") + portName +
                             "' is not a FuncService::Function");
  return func;
}

static int runBytePatternConst(Accelerator *accel) {
  // Bypasses the generated facade and the typed deserializer so the test
  // asserts on the *wire* bytes directly.
  auto *func = findRawFunc(findProbe(accel, "byte_pattern_const_inst"),
                           "byte_pattern_const");
  func->connect();

  uint8_t trigger = 0; // value is ignored by the hardware.
  MessageData resMsg = func->call(MessageData(&trigger, sizeof(trigger))).get();

  if (resMsg.getSize() != kBytePattern.size())
    throw std::runtime_error("byte_pattern_const: wrong response size (got " +
                             std::to_string(resMsg.getSize()) + ", expected " +
                             std::to_string(kBytePattern.size()) + ")");
  const uint8_t *got = resMsg.getBytes();
  for (size_t i = 0; i < kBytePattern.size(); ++i) {
    if (got[i] != kBytePattern[i]) {
      std::stringstream ss;
      ss << "byte_pattern_const: wire byte " << i << " mismatch (got 0x"
         << std::hex << static_cast<unsigned>(got[i]) << " expected 0x"
         << static_cast<unsigned>(kBytePattern[i]) << ")";
      throw std::runtime_error(ss.str());
    }
  }
  std::cout << "byte_pattern_const ok\n";
  return 0;
}

static int runBytePatternEchoEq(Accelerator *accel) {
  // Bypasses the generated facade and the typed serializer so the bytes the
  // hardware sees come straight off the wire.
  auto *func = findRawFunc(findProbe(accel, "byte_pattern_echo_eq_inst"),
                           "byte_pattern_echo_eq");
  func->connect();

  MessageData resMsg =
      func->call(MessageData(kBytePattern.data(), kBytePattern.size())).get();
  if (resMsg.getSize() != 1)
    throw std::runtime_error("byte_pattern_echo_eq: wrong response size (got " +
                             std::to_string(resMsg.getSize()) +
                             ", expected 1)");
  uint8_t got = *resMsg.getBytes();
  if (got != 1)
    throw std::runtime_error(
        "byte_pattern_echo_eq: hardware reported byte mismatch (got " +
        std::to_string(got) + ")");
  std::cout << "byte_pattern_echo_eq ok\n";
  return 0;
}

static int runSignProbe(Accelerator *accel) {
  esi_system::SignProbe mod(findProbe(accel, "sign_probe_inst"));
  auto connected = mod.connect();

  // Probe several signed values, including the boundaries where two's-
  // complement and sign-extension bugs would show up.
  struct Case {
    int16_t arg;
    int16_t plus_one;
    int16_t neg;
    uint8_t sign_bit;
  };
  Case cases[] = {
      {0, 1, 0, 0},
      {-1, 0, 1, 1},
      {1, 2, -1, 0},
      {INT16_MAX, static_cast<int16_t>(INT16_MIN), -INT16_MAX, 0},
      // INT16_MIN: -INT16_MIN is undefined in two's complement (overflows to
      // itself), which is the *expected* behavior the hardware reproduces.
      {INT16_MIN, static_cast<int16_t>(INT16_MIN + 1),
       static_cast<int16_t>(INT16_MIN), 1},
  };
  for (const Case &c : cases) {
    esi_system::SignResult r = connected->sign_probe(c.arg).get();
    if (r.plus_one != c.plus_one || r.neg != c.neg || r.sign_bit != c.sign_bit)
      throw std::runtime_error("sign_probe mismatch for arg=" +
                               std::to_string(c.arg));
  }
  std::cout << "sign_probe ok\n";
  return 0;
}

static int runSignProbe13(Accelerator *accel) {
  esi_system::SignProbe13 mod(findProbe(accel, "sign_probe13_inst"));
  auto connected = mod.connect();

  // si13 ranges from -4096 (= -2^12) to 4095 (= 2^12 - 1). The boundary
  // cases are where width-bounded sign extension and saturating-wrap
  // arithmetic differ from si16 behavior; getting plus_one or neg right
  // for both -4096 and 4095 forces the host serializer/deserializer to
  // use the manifest's bit width, not sizeof(int16_t).
  static constexpr int16_t kMin = -4096;
  static constexpr int16_t kMax = 4095;
  struct Case {
    int16_t arg;
    int16_t plus_one;
    int16_t neg;
    uint8_t sign_bit;
  };
  Case cases[] = {
      {0, 1, 0, 0},
      {-1, 0, 1, 1},
      {1, 2, -1, 0},
      {kMax, kMin, static_cast<int16_t>(-kMax), 0}, // 4095 + 1 wraps to -4096.
      // -4096 is the si13 minimum: -kMin overflows back to -4096 (just like
      // -INT16_MIN does for si16), and kMin + 1 = -4095.
      {kMin, static_cast<int16_t>(kMin + 1), kMin, 1},
  };
  for (const Case &c : cases) {
    esi_system::SignResult13 r = connected->sign_probe13(c.arg).get();
    if (r.plus_one != c.plus_one || r.neg != c.neg || r.sign_bit != c.sign_bit)
      throw std::runtime_error(
          "sign_probe13 mismatch for arg=" + std::to_string(c.arg) +
          " got plus_one=" + std::to_string(r.plus_one) + " neg=" +
          std::to_string(r.neg) + " sign_bit=" + std::to_string(r.sign_bit));
  }

  // TODO: Out-of-range si13 args. The generated facade declares
  // `sign_probe13Args = int16_t`, and the runtime's `toMessageData` for
  // int16_t -> si13 just copies the low 2 wire bytes with no truncation.
  // So a host that does ordinary int16_t arithmetic and passes a value
  // outside [-4096, 4095] silently sends a wrong si13 to the hardware.
  // The expected behavior is one of: (a) the codegen emits a wrapper class
  // (e.g. `sint13_t` with a 13-bit bitfield) that masks/sign-extends on
  // construction so this round-trip works, or (b) the runtime rejects the
  // value at write time. Either way the assertion below should hold.
  // Re-enable once one of those is implemented.
  //
  // {
  //   int16_t out_of_range = 4096; // legal int16_t, illegal si13.
  //   esi_system::SignResult13 r = connected->sign_probe13(out_of_range).get();
  //   if (r.plus_one != static_cast<int16_t>(out_of_range + 1))
  //     throw std::runtime_error(
  //         "sign_probe13 out-of-range arg mismatch: arg=" +
  //         std::to_string(out_of_range) +
  //         " plus_one=" + std::to_string(r.plus_one));
  // }

  std::cout << "sign_probe13 ok\n";
  return 0;
}

static int runPackProbe(Accelerator *accel) {
  esi_system::PackProbe mod(findProbe(accel, "pack_probe_inst"));
  auto connected = mod.connect();

  esi_system::PackStruct arg{};
  arg.a = 0x01;
  arg.b = 0x0002;
  arg.c = 0x03;
  arg.d = 0x00000004;

  esi_system::PackStruct r = connected->pack_probe(arg).get();
  if (r.a != 0xA1 || r.b != 0xB002 || r.c != 0xC3 || r.d != 0xD0000004)
    throw std::runtime_error("pack_probe field mismatch");
  std::cout << "pack_probe ok: a=0x" << std::hex << (unsigned)r.a << " b=0x"
            << r.b << " c=0x" << (unsigned)r.c << " d=0x" << r.d << std::dec
            << "\n";
  return 0;
}

static int runBitPackProbe(Accelerator *accel) {
  esi_system::BitPackProbe mod(findProbe(accel, "bit_pack_probe_inst"));
  auto connected = mod.connect();

  esi_system::BitPackArg arg{};
  // Distinct values within each width: x=001, y=10001, z=1001, w=1100. A
  // shift error to a different field width would produce a value outside
  // these picks.
  arg.x = 0b001;
  arg.y = 0b10001;
  arg.z = 0b1001;
  arg.w = 0b1100;

  esi_system::BitPackResult r = connected->bit_pack_probe(arg).get();
  if (r.w_field != arg.x || r.z_field != arg.y || r.y_field != arg.z ||
      r.x_field != arg.w)
    throw std::runtime_error("bit_pack_probe field rotation mismatch");
  std::cout << "bit_pack_probe ok\n";
  return 0;
}

// Per-index sentinels (10, 20, 30, 40) make element order observable.
// The PyCDE source reads the argument as `arg[i]` and returns
// `arg[i] + 10*(i+1)` packed back into the result array at logical
// index `i`. With the request bytes [0x01, 0x02, 0x03, 0x04], a wire
// convention that places logical element `i` at wire byte `i` produces
// a response of [11, 22, 33, 44]; a reverse-on-wire convention would
// produce [41, 32, 23, 14]. Either way each byte uniquely identifies
// its source index so the actual convention is observable from the
// failure message.
static int runArrayProbe(Accelerator *accel) {
  // Driven through the raw FuncService port with explicitly constructed
  // wire bytes -- not via the generated facade. Two reasons:
  //
  //   1. `TypedFunction<std::array<T,N>, ..., SkipTypeCheck=true>` would use
  //      `TypedPorts`' POD (memcpy) (de)serialization. That path is
  //      symmetric: it would round-trip a wrong-but-mirrored byte order as
  //      a false positive without ever exercising what the hardware
  //      actually puts on (or reads off) the wire.
  //   2. The runtime's typed `ArrayType` (de)serializer in `types.py`
  //      reverses element order on the wire. Whether the hardware
  //      (PyCDE-emitted RTL) also reverses is exactly the convention this
  //      probe is meant to nail down. By writing/reading raw bytes the
  //      test pins down the wire format independent of any host-side
  //      mirror of that convention.

  auto *func = findRawFunc(findProbe(accel, "array_probe_inst"), "array_probe");
  func->connect();

  static constexpr std::array<uint8_t, 4> kRequest = {0x01, 0x02, 0x03, 0x04};
  MessageData resMsg =
      func->call(MessageData(kRequest.data(), kRequest.size())).get();
  if (resMsg.getSize() != kRequest.size())
    throw std::runtime_error("array_probe: wrong response size (got " +
                             std::to_string(resMsg.getSize()) + ", expected " +
                             std::to_string(kRequest.size()) + ")");

  static constexpr std::array<uint8_t, 4> kExpect = {11, 22, 33, 44};
  const uint8_t *got = resMsg.getBytes();
  for (size_t i = 0; i < kExpect.size(); ++i) {
    if (got[i] != kExpect[i]) {
      std::stringstream ss;
      ss << "array_probe: wire byte " << i << " mismatch (got 0x" << std::hex
         << static_cast<unsigned>(got[i]) << " expected 0x"
         << static_cast<unsigned>(kExpect[i]) << ")";
      throw std::runtime_error(ss.str());
    }
  }
  std::cout << "array_probe ok (wire order: natural element-index)\n";
  return 0;
}

ESI_PROBE_REGISTRY("serialization-probes",
                   "Hardware-vs-host serialization correctness probes for ESI.",
                   {"byte_rotate1", &runByteRotate1},
                   {"byte_pattern_const", &runBytePatternConst},
                   {"byte_pattern_echo_eq", &runBytePatternEchoEq},
                   {"sign_probe", &runSignProbe},
                   {"sign_probe13", &runSignProbe13},
                   {"pack_probe", &runPackProbe},
                   {"bit_pack_probe", &runBitPackProbe},
                   {"array_probe", &runArrayProbe}, );
