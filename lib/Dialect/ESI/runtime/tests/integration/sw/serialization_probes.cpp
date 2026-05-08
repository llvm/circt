// Driver for the SerializationProbes integration test. Each function below
// targets exactly one of the on-the-wire serialization invariants the host
// must agree with hardware on (byte order, sign extension, struct-field
// order, sub-byte field packing, array element order). Mismatches surface
// as wrong-but-distinguishable output rather than coincidentally-correct
// answers; see the corresponding HW module's docstring for the rationale.

#include "serialization_probes/ArrayProbe.h"
#include "serialization_probes/BitPackProbe.h"
#include "serialization_probes/ByteRotate1.h"
#include "serialization_probes/PackProbe.h"
#include "serialization_probes/SignProbe.h"

#include "esi/Accelerator.h"
#include "esi/CLI.h"
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

static void runByteRotate1(Accelerator *accel) {
  esi_system::ByteRotate1 mod(findProbe(accel, "byte_rotate1_inst"));
  auto connected = mod.connect();

  uint64_t arg = 0x0102030405060708ULL;
  uint64_t expect = 0x0203040506070801ULL; // (arg << 8) | (arg >> 56).
  uint64_t got = connected->byte_rotate1(arg).get();
  if (got != expect)
    throw std::runtime_error("byte_rotate1 mismatch: got 0x" + toHex(got) +
                             " expected 0x" + toHex(expect));
  std::cout << "byte_rotate1 ok: 0x" << std::hex << got << std::dec << "\n";
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

static void runBytePatternConst(Accelerator *accel) {
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
}

static void runBytePatternEchoEq(Accelerator *accel) {
  // Bypasses the generated facade and the typed serializer so the bytes the
  // hardware sees come straight off the wire.
  auto *func = findRawFunc(findProbe(accel, "byte_pattern_echo_eq_inst"),
                           "byte_pattern_echo_eq");
  func->connect();

  MessageData resMsg =
      func->call(MessageData(kBytePattern.data(), kBytePattern.size())).get();
  if (resMsg.getSize() != 1)
    throw std::runtime_error(
        "byte_pattern_echo_eq: wrong response size (got " +
        std::to_string(resMsg.getSize()) + ", expected 1)");
  uint8_t got = *resMsg.getBytes();
  if (got != 1)
    throw std::runtime_error(
        "byte_pattern_echo_eq: hardware reported byte mismatch (got " +
        std::to_string(got) + ")");
  std::cout << "byte_pattern_echo_eq ok\n";
}

static void runSignProbe(Accelerator *accel) {
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
    if (r.plus_one != c.plus_one || r.neg != c.neg ||
        r.sign_bit != c.sign_bit)
      throw std::runtime_error("sign_probe mismatch for arg=" +
                               std::to_string(c.arg));
  }
  std::cout << "sign_probe ok\n";
}

static void runPackProbe(Accelerator *accel) {
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
}

static void runBitPackProbe(Accelerator *accel) {
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
}

static void runArrayProbe(Accelerator *accel) {
  // FIXME: bypass the generated facade for this probe.
  // verifyTypeCompatibility<std::array<T,N>>() against an ArrayType is not
  // implemented in the runtime today, so the generated facade's connect()
  // throws "Cannot verify type compatibility for C++ type 'St5arrayIhLm4EE'".
  // Once that gap is closed, switch this back to:
  //   esi_system::ArrayProbe mod(findProbe(accel, "array_probe_inst"));
  //   auto connected = mod.connect();
  //   ... connected->array_probe(arg).get();
  esi::HWModule *probe = findProbe(accel, "array_probe_inst");
  auto it = probe->getPorts().find(AppID("array_probe"));
  if (it == probe->getPorts().end())
    throw std::runtime_error("array_probe port not found on instance");
  auto *funcPort = it->second.getAs<services::FuncService::Function>();
  if (!funcPort)
    throw std::runtime_error("array_probe is not a FuncService::Function");
  TypedFunction<std::array<uint8_t, 4>, std::array<uint8_t, 4>,
                /*SkipTypeCheck=*/true>
      array_probe(funcPort);
  array_probe.connect();

  std::array<uint8_t, 4> arg{1, 2, 3, 4};
  std::array<uint8_t, 4> r = array_probe(arg).get();
  if (r[0] != 11 || r[1] != 22 || r[2] != 33 || r[3] != 44)
    throw std::runtime_error("array_probe element-order mismatch: got [" +
                             std::to_string(r[0]) + "," +
                             std::to_string(r[1]) + "," +
                             std::to_string(r[2]) + "," +
                             std::to_string(r[3]) + "]");
  std::cout << "array_probe ok: [" << (int)r[0] << "," << (int)r[1] << ","
            << (int)r[2] << "," << (int)r[3] << "]\n";
}

int main(int argc, const char *argv[]) {
  CliParser cli("serialization-probes");
  cli.description(
      "Hardware-vs-host serialization correctness probes for ESI.");
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

    runByteRotate1(accel);
    runBytePatternConst(accel);
    runBytePatternEchoEq(accel);
    runSignProbe(accel);
    runPackProbe(accel);
    runBitPackProbe(accel);
    runArrayProbe(accel);

    conn->disconnect();
  } catch (std::exception &e) {
    ctxt.getLogger().error("serialization-probes", e.what());
    conn->disconnect();
    return 1;
  }
  return 0;
}
