//===- esiaccel.cpp - ESI runtime python bindings ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Simply wrap the C++ API into a Python module called 'esiaccel'.
//
//===----------------------------------------------------------------------===//

#include "esi/Accelerator.h"
#include "esi/Services.h"

#include "esi/backends/Cosim.h"

#include <ranges>
#include <sstream>

// nanobind includes.
// Python world does not respect constness. So it doesn't make sense to have
// const checks. Disable related warnings.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#endif
#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace nb = nanobind;

using namespace esi;
using namespace esi::services;

namespace nanobind {
namespace detail {

template <>
struct type_hook<ChannelPort> {
  static const std::type_info *get(const ChannelPort *port) {
    if (dynamic_cast<const WriteChannelPort *>(port))
      return &typeid(WriteChannelPort);
    if (dynamic_cast<const ReadChannelPort *>(port))
      return &typeid(ReadChannelPort);
    return &typeid(ChannelPort);
  }
};

template <>
struct type_hook<Service> {
  static const std::type_info *get(const Service *svc) {
    if (dynamic_cast<const MMIO *>(svc))
      return &typeid(MMIO);
    if (dynamic_cast<const SysInfo *>(svc))
      return &typeid(SysInfo);
    if (dynamic_cast<const HostMem *>(svc))
      return &typeid(HostMem);
    if (dynamic_cast<const TelemetryService *>(svc))
      return &typeid(TelemetryService);
    return &typeid(Service);
  }
};

/// Nanobind doesn't have a built-in type caster for std::any.
/// We must provide one which knows about all of the potential types which the
/// any might be.
template <>
struct type_caster<std::any> {
  NB_TYPE_CASTER(std::any, const_name("object"))

  static handle from_cpp(const std::any &src, rv_policy /* policy */,
                         cleanup_list * /* cleanup */) {
    const std::type_info &t = src.type();
    if (t == typeid(std::string))
      return nb::str(std::any_cast<std::string>(src).c_str()).release();
    else if (t == typeid(int64_t))
      return nb::int_(std::any_cast<int64_t>(src)).release();
    else if (t == typeid(uint64_t))
      return nb::int_(std::any_cast<uint64_t>(src)).release();
    else if (t == typeid(double))
      return nb::float_(std::any_cast<double>(src)).release();
    else if (t == typeid(bool))
      return nb::bool_(std::any_cast<bool>(src)).release();
    else if (t == typeid(std::nullptr_t))
      return nb::none().release();
    return nb::none().release();
  }
};
} // namespace detail
} // namespace nanobind

/// Resolve a Type to the Python wrapper object.
nb::object getPyType(std::optional<const Type *> t) {
  nb::object typesModule = nb::module_::import_("esiaccel.types");
  if (!t)
    return nb::none();
  return typesModule.attr("_get_esi_type")(*t);
}

// NOLINTNEXTLINE(readability-identifier-naming)
NB_MODULE(esiCppAccel, m) {
  // TODO: Fix leaks! The one I know of is in the callback code -- if one
  // registers a python callback it creates a leak.
  nb::set_leak_warnings(false);

  nb::class_<Type>(m, "Type")
      .def(nb::init<const Type::ID &>(), nb::arg("id"))
      .def_prop_ro("id", &Type::getID)
      .def("__repr__", [](Type &t) { return "<" + t.getID() + ">"; });
  nb::class_<ChannelType, Type>(m, "ChannelType")
      .def(nb::init<const Type::ID &, const Type *>(), nb::arg("id"),
           nb::arg("inner"))
      .def_prop_ro("inner", &ChannelType::getInner, nb::rv_policy::reference);
  nb::enum_<BundleType::Direction>(m, "Direction")
      .value("To", BundleType::Direction::To)
      .value("From", BundleType::Direction::From)
      .export_values();
  nb::class_<BundleType, Type>(m, "BundleType")
      .def(nb::init<const Type::ID &, const BundleType::ChannelVector &>(),
           nb::arg("id"), nb::arg("channels"))
      .def_prop_ro("channels", &BundleType::getChannels,
                   nb::rv_policy::reference);
  nb::class_<VoidType, Type>(m, "VoidType")
      .def(nb::init<const Type::ID &>(), nb::arg("id"));
  nb::class_<AnyType, Type>(m, "AnyType")
      .def(nb::init<const Type::ID &>(), nb::arg("id"));
  nb::class_<TypeAliasType, Type>(m, "TypeAliasType")
      .def(nb::init<const Type::ID &, std::string, const Type *>(),
           nb::arg("id"), nb::arg("name"), nb::arg("inner_type"))
      .def_prop_ro("name", &TypeAliasType::getName)
      .def_prop_ro("inner", &TypeAliasType::getInnerType,
                   nb::rv_policy::reference);
  nb::class_<BitVectorType, Type>(m, "BitVectorType")
      .def(nb::init<const Type::ID &, uint64_t>(), nb::arg("id"),
           nb::arg("width"))
      .def_prop_ro("width", &BitVectorType::getWidth);
  nb::class_<BitsType, BitVectorType>(m, "BitsType")
      .def(nb::init<const Type::ID &, uint64_t>(), nb::arg("id"),
           nb::arg("width"));
  nb::class_<IntegerType, BitVectorType>(m, "IntegerType")
      .def(nb::init<const Type::ID &, uint64_t>(), nb::arg("id"),
           nb::arg("width"));
  nb::class_<SIntType, IntegerType>(m, "SIntType")
      .def(nb::init<const Type::ID &, uint64_t>(), nb::arg("id"),
           nb::arg("width"));
  nb::class_<UIntType, IntegerType>(m, "UIntType")
      .def(nb::init<const Type::ID &, uint64_t>(), nb::arg("id"),
           nb::arg("width"));
  nb::class_<StructType, Type>(m, "StructType")
      .def(nb::init<const Type::ID &, const StructType::FieldVector &, bool>(),
           nb::arg("id"), nb::arg("fields"), nb::arg("reverse") = true)
      .def_prop_ro("fields", &StructType::getFields, nb::rv_policy::reference)
      .def_prop_ro("reverse", &StructType::isReverse);
  nb::class_<ArrayType, Type>(m, "ArrayType")
      .def(nb::init<const Type::ID &, const Type *, uint64_t>(), nb::arg("id"),
           nb::arg("element_type"), nb::arg("size"))
      .def_prop_ro("element", &ArrayType::getElementType,
                   nb::rv_policy::reference)
      .def_prop_ro("size", &ArrayType::getSize);

  nb::class_<Constant>(m, "Constant")
      .def_prop_ro("value", [](Constant &c) { return c.value; })
      .def_prop_ro("type", [](Constant &c) { return getPyType(*c.type); });

  nb::class_<AppID>(m, "AppID")
      .def(nb::init<std::string, std::optional<uint32_t>>(), nb::arg("name"),
           nb::arg("idx") = std::nullopt)
      .def_prop_ro("name", [](AppID &id) { return id.name; })
      .def_prop_ro("idx",
                   [](AppID &id) -> nb::object {
                     if (id.idx)
                       return nb::cast(id.idx);
                     return nb::none();
                   })
      .def("__repr__",
           [](AppID &id) {
             std::string ret = "<" + id.name;
             if (id.idx)
               ret = ret + "[" + std::to_string(*id.idx) + "]";
             ret = ret + ">";
             return ret;
           })
      .def("__eq__", [](AppID &a, AppID &b) { return a == b; })
      .def("__hash__", [](AppID &id) {
        return utils::hash_combine(std::hash<std::string>{}(id.name),
                                   std::hash<uint32_t>{}(id.idx.value_or(-1)));
      });
  nb::class_<AppIDPath>(m, "AppIDPath").def("__repr__", &AppIDPath::toStr);

  nb::class_<ModuleInfo>(m, "ModuleInfo")
      .def_prop_ro("name", [](ModuleInfo &info) { return info.name; })
      .def_prop_ro("summary", [](ModuleInfo &info) { return info.summary; })
      .def_prop_ro("version", [](ModuleInfo &info) { return info.version; })
      .def_prop_ro("repo", [](ModuleInfo &info) { return info.repo; })
      .def_prop_ro("commit_hash",
                   [](ModuleInfo &info) { return info.commitHash; })
      .def_prop_ro("constants", [](ModuleInfo &info) { return info.constants; })
      // TODO: "extra" field.
      .def("__repr__", [](ModuleInfo &info) {
        std::string ret;
        std::stringstream os(ret);
        os << info;
        return os.str();
      });

  nb::enum_<Logger::Level>(m, "LogLevel")
      .value("Debug", Logger::Level::Debug)
      .value("Info", Logger::Level::Info)
      .value("Warning", Logger::Level::Warning)
      .value("Error", Logger::Level::Error)
      .export_values();
  nb::class_<Logger>(m, "Logger");

  nb::class_<services::Service>(m, "Service")
      .def("get_service_symbol", &services::Service::getServiceSymbol);

  nb::class_<SysInfo, services::Service>(m, "SysInfo")
      .def("esi_version", &SysInfo::getEsiVersion)
      .def("json_manifest", &SysInfo::getJsonManifest)
      .def("cycle_count", &SysInfo::getCycleCount,
           "Get the current cycle count of the accelerator system")
      .def("core_clock_frequency", &SysInfo::getCoreClockFrequency,
           "Get the core clock frequency of the accelerator system in Hz");

  nb::class_<MMIO::RegionDescriptor>(m, "MMIORegionDescriptor")
      .def_prop_ro("base", [](MMIO::RegionDescriptor &r) { return r.base; })
      .def_prop_ro("size", [](MMIO::RegionDescriptor &r) { return r.size; });
  nb::class_<services::MMIO, services::Service>(m, "MMIO")
      .def("read", &services::MMIO::read)
      .def("write", &services::MMIO::write)
      .def_prop_ro("regions", &services::MMIO::getRegions,
                   nb::rv_policy::reference);

  nb::class_<services::HostMem::HostMemRegion>(m, "HostMemRegion")
      .def_prop_ro("ptr",
                   [](services::HostMem::HostMemRegion &mem) {
                     return reinterpret_cast<uintptr_t>(mem.getPtr());
                   })
      .def_prop_ro("size", &services::HostMem::HostMemRegion::getSize);

  nb::class_<services::HostMem::Options>(m, "HostMemOptions")
      .def(nb::init<>())
      .def_rw("writeable", &services::HostMem::Options::writeable)
      .def_rw("use_large_pages", &services::HostMem::Options::useLargePages)
      .def("__repr__", [](services::HostMem::Options &opts) {
        std::string ret = "HostMemOptions(";
        if (opts.writeable)
          ret += "writeable ";
        if (opts.useLargePages)
          ret += "use_large_pages";
        ret += ")";
        return ret;
      });

  nb::class_<services::HostMem, services::Service>(m, "HostMem")
      .def("allocate", &services::HostMem::allocate, nb::arg("size"),
           nb::arg("options") = services::HostMem::Options(),
           nb::rv_policy::take_ownership)
      .def(
          "map_memory",
          [](HostMem &self, uintptr_t ptr, size_t size, HostMem::Options opts) {
            return self.mapMemory(reinterpret_cast<void *>(ptr), size, opts);
          },
          nb::arg("ptr"), nb::arg("size"),
          nb::arg("options") = services::HostMem::Options())
      .def(
          "unmap_memory",
          [](HostMem &self, uintptr_t ptr) {
            return self.unmapMemory(reinterpret_cast<void *>(ptr));
          },
          nb::arg("ptr"));
  nb::class_<services::TelemetryService, services::Service>(m,
                                                            "TelemetryService");

  nb::class_<std::future<MessageData>>(m, "MessageDataFuture")
      .def("valid", [](std::future<MessageData> &f) { return f.valid(); })
      .def("wait",
           [](std::future<MessageData> &f) {
             // Yield the GIL while waiting for the future to complete, in case
             // of python callbacks occurring from other threads while waiting.
             nb::gil_scoped_release release{};
             f.wait();
           })
      .def("get", [](std::future<MessageData> &f) {
        std::optional<MessageData> data;
        {
          // Yield the GIL while waiting for the future to complete, in case of
          // python callbacks occurring from other threads while waiting.
          nb::gil_scoped_release release{};
          data.emplace(f.get());
        }
        return nb::bytearray((const char *)data->getBytes(), data->getSize());
      });

  nb::class_<ChannelPort::ConnectOptions>(m, "ConnectOptions")
      .def(nb::init<>())
      .def_rw("buffer_size", &ChannelPort::ConnectOptions::bufferSize,
              nb::arg("buffer_size").none())
      .def_rw("translate_message",
              &ChannelPort::ConnectOptions::translateMessage);

  nb::class_<ChannelPort>(m, "ChannelPort")
      .def("connect", &ChannelPort::connect, nb::arg("options"),
           "Connect with specified options")
      .def("disconnect", &ChannelPort::disconnect)
      .def_prop_ro("type", &ChannelPort::getType, nb::rv_policy::reference);

  nb::class_<WriteChannelPort, ChannelPort>(m, "WriteChannelPort")
      .def("write",
           [](WriteChannelPort &p, nb::bytearray data) {
             std::vector<uint8_t> dataVec((const uint8_t *)data.c_str(),
                                          (const uint8_t *)data.c_str() +
                                              data.size());
             p.write(dataVec);
           })
      .def("tryWrite", [](WriteChannelPort &p, nb::bytearray data) {
        std::vector<uint8_t> dataVec((const uint8_t *)data.c_str(),
                                     (const uint8_t *)data.c_str() +
                                         data.size());
        return p.tryWrite(dataVec);
      });
  nb::class_<ReadChannelPort, ChannelPort>(m, "ReadChannelPort")
      .def(
          "read",
          [](ReadChannelPort &p) -> nb::bytearray {
            MessageData data;
            p.read(data);
            return nb::bytearray((const char *)data.getBytes(), data.getSize());
          },
          "Read data from the channel. Blocking.")
      .def("read_async", &ReadChannelPort::readAsync);

  nb::class_<BundlePort>(m, "BundlePort")
      .def_prop_ro("id", &BundlePort::getID)
      .def_prop_ro("channels", &BundlePort::getChannels,
                   nb::rv_policy::reference)
      .def("getWrite", &BundlePort::getRawWrite, nb::rv_policy::reference)
      .def("getRead", &BundlePort::getRawRead, nb::rv_policy::reference);

  nb::class_<ServicePort, BundlePort>(m, "ServicePort");

  nb::class_<MMIO::MMIORegion, ServicePort>(m, "MMIORegion")
      .def_prop_ro("descriptor", &MMIO::MMIORegion::getDescriptor)
      .def("read", &MMIO::MMIORegion::read)
      .def("write", &MMIO::MMIORegion::write);

  nb::class_<FuncService::Function, ServicePort>(m, "Function")
      .def("call",
           [](FuncService::Function &self,
              nb::bytearray msg) -> std::future<MessageData> {
             std::vector<uint8_t> dataVec((const uint8_t *)msg.c_str(),
                                          (const uint8_t *)msg.c_str() +
                                              msg.size());
             MessageData data(dataVec);
             return self.call(data);
           })
      .def("connect", &FuncService::Function::connect);

  nb::class_<CallService::Callback, ServicePort>(m, "Callback")
      .def("connect", [](CallService::Callback &self,
                         std::function<nb::object(nb::object)> pyCallback) {
        // TODO: Under certain conditions this will cause python to crash. I
        // don't remember how to replicate these crashes, but IIRC they are
        // deterministic.
        self.connect([pyCallback](const MessageData &req) -> MessageData {
          nb::gil_scoped_acquire acquire{};
          std::vector<uint8_t> arg(req.getBytes(),
                                   req.getBytes() + req.getSize());
          nb::bytes argObj((const char *)arg.data(), arg.size());
          auto ret = pyCallback(argObj);
          if (ret.is_none())
            return MessageData();
          nb::bytearray retBytes = nb::cast<nb::bytearray>(ret);
          std::vector<uint8_t> dataVec((const uint8_t *)retBytes.c_str(),
                                       (const uint8_t *)retBytes.c_str() +
                                           retBytes.size());
          return MessageData(dataVec);
        });
      });

  nb::class_<TelemetryService::Metric, ServicePort>(m, "Metric")
      .def("connect", &TelemetryService::Metric::connect)
      .def("read", &TelemetryService::Metric::read)
      .def("readInt", &TelemetryService::Metric::readInt);

  // Store this variable (not commonly done) as the "children" method needs for
  // "Instance" to be defined first.
  auto hwmodule =
      nb::class_<HWModule>(m, "HWModule")
          .def_prop_ro("info", &HWModule::getInfo)
          .def_prop_ro("ports", &HWModule::getPorts, nb::rv_policy::reference)
          .def_prop_ro("services", &HWModule::getServices,
                       nb::rv_policy::reference);

  // In order to inherit methods from "HWModule", it needs to be defined first.
  nb::class_<Instance, HWModule>(m, "Instance")
      .def_prop_ro("id", &Instance::getID);

  nb::class_<Accelerator, HWModule>(m, "Accelerator");

  // Since this returns a vector of Instance*, we need to define Instance first
  // or else stubgen complains.
  hwmodule.def_prop_ro("children", &HWModule::getChildren,
                       nb::rv_policy::reference);

  auto accConn = nb::class_<AcceleratorConnection>(m, "AcceleratorConnection");

  nb::class_<Context>(
      m, "Context",
      "An ESI context owns everything -- types, accelerator connections, and "
      "the accelerator facade (aka Accelerator) itself. It MUST NOT be garbage "
      "collected while the accelerator is still in use. When it is destroyed, "
      "all accelerator connections are disconnected.")
      .def(nb::init<>(), "Create a context with a default logger.")
      .def("connect", &Context::connect, nb::rv_policy::reference)
      .def("set_stdio_logger", [](Context &ctxt, Logger::Level level) {
        ctxt.setLogger(std::make_unique<StreamLogger>(level));
      });

  accConn
      .def(
          "sysinfo",
          [](AcceleratorConnection &acc) {
            return acc.getService<services::SysInfo>({});
          },
          nb::rv_policy::reference)
      .def(
          "get_service_mmio",
          [](AcceleratorConnection &acc) {
            return acc.getService<services::MMIO>({});
          },
          nb::rv_policy::reference)
      .def(
          "get_service_hostmem",
          [](AcceleratorConnection &acc) {
            return acc.getService<services::HostMem>({});
          },
          nb::rv_policy::reference)
      .def("get_accelerator", &AcceleratorConnection::getAccelerator,
           nb::rv_policy::reference);

  nb::class_<Manifest>(m, "Manifest")
      .def(nb::init<Context &, std::string>())
      .def_prop_ro("api_version", &Manifest::getApiVersion)
      .def(
          "build_accelerator",
          [&](Manifest &m, AcceleratorConnection &conn) -> Accelerator * {
            auto *acc = m.buildAccelerator(conn);
            conn.getServiceThread()->addPoll(*acc);
            return acc;
          },
          nb::rv_policy::reference)
      .def_prop_ro("type_table",
                   [](Manifest &m) {
                     std::vector<nb::object> ret;
                     std::ranges::transform(m.getTypeTable(),
                                            std::back_inserter(ret), getPyType);
                     return ret;
                   })
      .def_prop_ro("module_infos", &Manifest::getModuleInfos);
}
