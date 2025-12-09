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

// pybind11 includes
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <pybind11/functional.h>
#include <pybind11/stl.h>

using namespace esi;
using namespace esi::services;

namespace pybind11 {
/// Pybind11 needs a little help downcasting with non-bound instances.
template <>
struct polymorphic_type_hook<ChannelPort> {
  static const void *get(const ChannelPort *port, const std::type_info *&type) {
    if (auto p = dynamic_cast<const WriteChannelPort *>(port)) {
      type = &typeid(WriteChannelPort);
      return p;
    }
    if (auto p = dynamic_cast<const ReadChannelPort *>(port)) {
      type = &typeid(ReadChannelPort);
      return p;
    }
    return port;
  }
};
template <>
struct polymorphic_type_hook<Service> {
  static const void *get(const Service *svc, const std::type_info *&type) {
    if (auto p = dynamic_cast<const MMIO *>(svc)) {
      type = &typeid(MMIO);
      return p;
    }
    if (auto p = dynamic_cast<const SysInfo *>(svc)) {
      type = &typeid(SysInfo);
      return p;
    }
    if (auto p = dynamic_cast<const HostMem *>(svc)) {
      type = &typeid(HostMem);
      return p;
    }
    if (auto p = dynamic_cast<const TelemetryService *>(svc)) {
      type = &typeid(TelemetryService);
      return p;
    }
    return svc;
  }
};

namespace detail {
/// Pybind11 doesn't have a built-in type caster for std::any
/// (https://github.com/pybind/pybind11/issues/1590). We must provide one which
/// knows about all of the potential types which the any might be.
template <>
struct type_caster<std::any> {
public:
  PYBIND11_TYPE_CASTER(std::any, const_name("object"));

  static handle cast(std::any src, return_value_policy /* policy */,
                     handle /* parent */) {
    const std::type_info &t = src.type();
    if (t == typeid(std::string))
      return py::str(std::any_cast<std::string>(src));
    else if (t == typeid(int64_t))
      return py::int_(std::any_cast<int64_t>(src));
    else if (t == typeid(uint64_t))
      return py::int_(std::any_cast<uint64_t>(src));
    else if (t == typeid(double))
      return py::float_(std::any_cast<double>(src));
    else if (t == typeid(bool))
      return py::bool_(std::any_cast<bool>(src));
    else if (t == typeid(std::nullptr_t))
      return py::none();
    return py::none();
  }
};
} // namespace detail
} // namespace pybind11

/// Resolve a Type to the Python wrapper object.
py::object getPyType(std::optional<const Type *> t) {
  py::object typesModule = py::module_::import("esiaccel.types");
  if (!t)
    return py::none();
  return typesModule.attr("_get_esi_type")(*t);
}

// NOLINTNEXTLINE(readability-identifier-naming)
PYBIND11_MODULE(esiCppAccel, m) {
  py::class_<Type>(m, "Type")
      .def(py::init<const Type::ID &>(), py::arg("id"))
      .def_property_readonly("id", &Type::getID)
      .def("__repr__", [](Type &t) { return "<" + t.getID() + ">"; });
  py::class_<ChannelType, Type>(m, "ChannelType")
      .def(py::init<const Type::ID &, const Type *>(), py::arg("id"),
           py::arg("inner"))
      .def_property_readonly("inner", &ChannelType::getInner,
                             py::return_value_policy::reference);
  py::enum_<BundleType::Direction>(m, "Direction")
      .value("To", BundleType::Direction::To)
      .value("From", BundleType::Direction::From)
      .export_values();
  py::class_<BundleType, Type>(m, "BundleType")
      .def(py::init<const Type::ID &, const BundleType::ChannelVector &>(),
           py::arg("id"), py::arg("channels"))
      .def_property_readonly("channels", &BundleType::getChannels,
                             py::return_value_policy::reference);
  py::class_<VoidType, Type>(m, "VoidType")
      .def(py::init<const Type::ID &>(), py::arg("id"));
  py::class_<AnyType, Type>(m, "AnyType")
      .def(py::init<const Type::ID &>(), py::arg("id"));
  py::class_<BitVectorType, Type>(m, "BitVectorType")
      .def(py::init<const Type::ID &, uint64_t>(), py::arg("id"),
           py::arg("width"))
      .def_property_readonly("width", &BitVectorType::getWidth);
  py::class_<BitsType, BitVectorType>(m, "BitsType")
      .def(py::init<const Type::ID &, uint64_t>(), py::arg("id"),
           py::arg("width"));
  py::class_<IntegerType, BitVectorType>(m, "IntegerType")
      .def(py::init<const Type::ID &, uint64_t>(), py::arg("id"),
           py::arg("width"));
  py::class_<SIntType, IntegerType>(m, "SIntType")
      .def(py::init<const Type::ID &, uint64_t>(), py::arg("id"),
           py::arg("width"));
  py::class_<UIntType, IntegerType>(m, "UIntType")
      .def(py::init<const Type::ID &, uint64_t>(), py::arg("id"),
           py::arg("width"));
  py::class_<StructType, Type>(m, "StructType")
      .def(py::init<const Type::ID &, const StructType::FieldVector &, bool>(),
           py::arg("id"), py::arg("fields"), py::arg("reverse") = true)
      .def_property_readonly("fields", &StructType::getFields,
                             py::return_value_policy::reference)
      .def_property_readonly("reverse", &StructType::isReverse);
  py::class_<ArrayType, Type>(m, "ArrayType")
      .def(py::init<const Type::ID &, const Type *, uint64_t>(), py::arg("id"),
           py::arg("element_type"), py::arg("size"))
      .def_property_readonly("element", &ArrayType::getElementType,
                             py::return_value_policy::reference)
      .def_property_readonly("size", &ArrayType::getSize);

  py::class_<Constant>(m, "Constant")
      .def_property_readonly("value", [](Constant &c) { return c.value; })
      .def_property_readonly(
          "type", [](Constant &c) { return getPyType(*c.type); },
          py::return_value_policy::reference);

  py::class_<AppID>(m, "AppID")
      .def(py::init<std::string, std::optional<uint32_t>>(), py::arg("name"),
           py::arg("idx") = std::nullopt)
      .def_property_readonly("name", [](AppID &id) { return id.name; })
      .def_property_readonly("idx",
                             [](AppID &id) -> py::object {
                               if (id.idx)
                                 return py::cast(id.idx);
                               return py::none();
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
  py::class_<AppIDPath>(m, "AppIDPath").def("__repr__", &AppIDPath::toStr);

  py::class_<ModuleInfo>(m, "ModuleInfo")
      .def_property_readonly("name", [](ModuleInfo &info) { return info.name; })
      .def_property_readonly("summary",
                             [](ModuleInfo &info) { return info.summary; })
      .def_property_readonly("version",
                             [](ModuleInfo &info) { return info.version; })
      .def_property_readonly("repo", [](ModuleInfo &info) { return info.repo; })
      .def_property_readonly("commit_hash",
                             [](ModuleInfo &info) { return info.commitHash; })
      .def_property_readonly("constants",
                             [](ModuleInfo &info) { return info.constants; })
      // TODO: "extra" field.
      .def("__repr__", [](ModuleInfo &info) {
        std::string ret;
        std::stringstream os(ret);
        os << info;
        return os.str();
      });

  py::enum_<Logger::Level>(m, "LogLevel")
      .value("Debug", Logger::Level::Debug)
      .value("Info", Logger::Level::Info)
      .value("Warning", Logger::Level::Warning)
      .value("Error", Logger::Level::Error)
      .export_values();
  py::class_<Logger>(m, "Logger");

  py::class_<services::Service>(m, "Service");

  py::class_<SysInfo, services::Service>(m, "SysInfo")
      .def("esi_version", &SysInfo::getEsiVersion)
      .def("json_manifest", &SysInfo::getJsonManifest);

  py::class_<MMIO::RegionDescriptor>(m, "MMIORegionDescriptor")
      .def_property_readonly("base",
                             [](MMIO::RegionDescriptor &r) { return r.base; })
      .def_property_readonly("size",
                             [](MMIO::RegionDescriptor &r) { return r.size; });
  py::class_<services::MMIO, services::Service>(m, "MMIO")
      .def("read", &services::MMIO::read)
      .def("write", &services::MMIO::write)
      .def_property_readonly("regions", &services::MMIO::getRegions,
                             py::return_value_policy::reference);

  py::class_<services::HostMem::HostMemRegion>(m, "HostMemRegion")
      .def_property_readonly("ptr",
                             [](services::HostMem::HostMemRegion &mem) {
                               return reinterpret_cast<uintptr_t>(mem.getPtr());
                             })
      .def_property_readonly("size",
                             &services::HostMem::HostMemRegion::getSize);

  py::class_<services::HostMem::Options>(m, "HostMemOptions")
      .def(py::init<>())
      .def_readwrite("writeable", &services::HostMem::Options::writeable)
      .def_readwrite("use_large_pages",
                     &services::HostMem::Options::useLargePages)
      .def("__repr__", [](services::HostMem::Options &opts) {
        std::string ret = "HostMemOptions(";
        if (opts.writeable)
          ret += "writeable ";
        if (opts.useLargePages)
          ret += "use_large_pages";
        ret += ")";
        return ret;
      });

  py::class_<services::HostMem, services::Service>(m, "HostMem")
      .def("allocate", &services::HostMem::allocate, py::arg("size"),
           py::arg("options") = services::HostMem::Options(),
           py::return_value_policy::take_ownership)
      .def(
          "map_memory",
          [](HostMem &self, uintptr_t ptr, size_t size, HostMem::Options opts) {
            return self.mapMemory(reinterpret_cast<void *>(ptr), size, opts);
          },
          py::arg("ptr"), py::arg("size"),
          py::arg("options") = services::HostMem::Options())
      .def(
          "unmap_memory",
          [](HostMem &self, uintptr_t ptr) {
            return self.unmapMemory(reinterpret_cast<void *>(ptr));
          },
          py::arg("ptr"));

  // py::class_<std::__basic_future<MessageData>>(m, "MessageDataFuture");
  py::class_<std::future<MessageData>>(m, "MessageDataFuture")
      .def("valid",
           [](std::future<MessageData> &f) {
             // For some reason, if we just pass the function pointer, pybind11
             // sees `std::__basic_future` as the type and pybind11_stubgen
             // emits an error.
             return f.valid();
           })
      .def("wait",
           [](std::future<MessageData> &f) {
             // Yield the GIL while waiting for the future to complete, in case
             // of python callbacks occuring from other threads while waiting.
             py::gil_scoped_release release{};
             f.wait();
           })
      .def("get", [](std::future<MessageData> &f) {
        std::optional<MessageData> data;
        {
          // Yield the GIL while waiting for the future to complete, in case of
          // python callbacks occuring from other threads while waiting.
          py::gil_scoped_release release{};
          data.emplace(f.get());
        }
        return py::bytearray((const char *)data->getBytes(), data->getSize());
      });

  py::class_<ChannelPort::ConnectOptions>(m, "ConnectOptions")
      .def(py::init<>())
      .def_readwrite("buffer_size", &ChannelPort::ConnectOptions::bufferSize)
      .def_readwrite("translate_message",
                     &ChannelPort::ConnectOptions::translateMessage);

  py::class_<ChannelPort>(m, "ChannelPort")
      .def("connect", &ChannelPort::connect, py::arg("options"),
           "Connect with specified options")
      .def("disconnect", &ChannelPort::disconnect)
      .def_property_readonly("type", &ChannelPort::getType,
                             py::return_value_policy::reference);

  py::class_<WriteChannelPort, ChannelPort>(m, "WriteChannelPort")
      .def("write",
           [](WriteChannelPort &p, py::bytearray &data) {
             py::buffer_info info(py::buffer(data).request());
             std::vector<uint8_t> dataVec((uint8_t *)info.ptr,
                                          (uint8_t *)info.ptr + info.size);
             p.write(dataVec);
           })
      .def("tryWrite", [](WriteChannelPort &p, py::bytearray &data) {
        py::buffer_info info(py::buffer(data).request());
        std::vector<uint8_t> dataVec((uint8_t *)info.ptr,
                                     (uint8_t *)info.ptr + info.size);
        return p.tryWrite(dataVec);
      });
  py::class_<ReadChannelPort, ChannelPort>(m, "ReadChannelPort")
      .def(
          "read",
          [](ReadChannelPort &p) -> py::bytearray {
            MessageData data;
            p.read(data);
            return py::bytearray((const char *)data.getBytes(), data.getSize());
          },
          "Read data from the channel. Blocking.")
      .def("read_async", &ReadChannelPort::readAsync);

  py::class_<BundlePort>(m, "BundlePort")
      .def_property_readonly("id", &BundlePort::getID)
      .def_property_readonly("channels", &BundlePort::getChannels,
                             py::return_value_policy::reference)
      .def("getWrite", &BundlePort::getRawWrite,
           py::return_value_policy::reference)
      .def("getRead", &BundlePort::getRawRead,
           py::return_value_policy::reference);

  py::class_<ServicePort, BundlePort>(m, "ServicePort");

  py::class_<MMIO::MMIORegion, ServicePort>(m, "MMIORegion")
      .def_property_readonly("descriptor", &MMIO::MMIORegion::getDescriptor)
      .def("read", &MMIO::MMIORegion::read)
      .def("write", &MMIO::MMIORegion::write);

  py::class_<FuncService::Function, ServicePort>(m, "Function")
      .def(
          "call",
          [](FuncService::Function &self,
             py::bytearray msg) -> std::future<MessageData> {
            py::buffer_info info(py::buffer(msg).request());
            std::vector<uint8_t> dataVec((uint8_t *)info.ptr,
                                         (uint8_t *)info.ptr + info.size);
            MessageData data(dataVec);
            return self.call(data);
          },
          py::return_value_policy::take_ownership)
      .def("connect", &FuncService::Function::connect);

  py::class_<CallService::Callback, ServicePort>(m, "Callback")
      .def("connect", [](CallService::Callback &self,
                         std::function<py::object(py::object)> pyCallback) {
        // TODO: Under certain conditions this will cause python to crash. I
        // don't remember how to replicate these crashes, but IIRC they are
        // deterministic.
        self.connect([pyCallback](const MessageData &req) -> MessageData {
          py::gil_scoped_acquire acquire{};
          std::vector<uint8_t> arg(req.getBytes(),
                                   req.getBytes() + req.getSize());
          py::bytearray argObj((const char *)arg.data(), arg.size());
          auto ret = pyCallback(argObj);
          if (ret.is_none())
            return MessageData();
          py::buffer_info info(py::buffer(ret).request());
          std::vector<uint8_t> dataVec((uint8_t *)info.ptr,
                                       (uint8_t *)info.ptr + info.size);
          return MessageData(dataVec);
        });
      });

  py::class_<TelemetryService::Metric, ServicePort>(m, "Metric")
      .def("connect", &TelemetryService::Metric::connect)
      .def("read", &TelemetryService::Metric::read)
      .def("readInt", &TelemetryService::Metric::readInt);

  // Store this variable (not commonly done) as the "children" method needs for
  // "Instance" to be defined first.
  auto hwmodule =
      py::class_<HWModule>(m, "HWModule")
          .def_property_readonly("info", &HWModule::getInfo)
          .def_property_readonly("ports", &HWModule::getPorts,
                                 py::return_value_policy::reference)
          .def_property_readonly("services", &HWModule::getServices,
                                 py::return_value_policy::reference);

  // In order to inherit methods from "HWModule", it needs to be defined first.
  py::class_<Instance, HWModule>(m, "Instance")
      .def_property_readonly("id", &Instance::getID);

  py::class_<Accelerator, HWModule>(m, "Accelerator");

  // Since this returns a vector of Instance*, we need to define Instance first
  // or else pybind11-stubgen complains.
  hwmodule.def_property_readonly("children", &HWModule::getChildren,
                                 py::return_value_policy::reference);

  auto accConn = py::class_<AcceleratorConnection>(m, "AcceleratorConnection");

  py::class_<Context>(
      m, "Context",
      "An ESI context owns everything -- types, accelerator connections, and "
      "the accelerator facade (aka Accelerator) itself. It MUST NOT be garbage "
      "collected while the accelerator is still in use. When it is destroyed, "
      "all accelerator connections are disconnected.")
      .def(py::init<>(), "Create a context with a default logger.",
           py::return_value_policy::take_ownership)
      .def("connect", &Context::connect, py::return_value_policy::reference)
      .def("set_stdio_logger", [](Context &ctxt, Logger::Level level) {
        ctxt.setLogger(std::make_unique<StreamLogger>(level));
      });

  accConn
      .def(
          "sysinfo",
          [](AcceleratorConnection &acc) {
            return acc.getService<services::SysInfo>({});
          },
          py::return_value_policy::reference)
      .def(
          "get_service_mmio",
          [](AcceleratorConnection &acc) {
            return acc.getService<services::MMIO>({});
          },
          py::return_value_policy::reference)
      .def(
          "get_service_hostmem",
          [](AcceleratorConnection &acc) {
            return acc.getService<services::HostMem>({});
          },
          py::return_value_policy::reference)
      .def("get_accelerator", &AcceleratorConnection::getAccelerator,
           py::return_value_policy::reference);

  py::class_<Manifest>(m, "Manifest")
      .def(py::init<Context &, std::string>())
      .def_property_readonly("api_version", &Manifest::getApiVersion)
      .def(
          "build_accelerator",
          [&](Manifest &m, AcceleratorConnection &conn) -> Accelerator * {
            auto *acc = m.buildAccelerator(conn);
            conn.getServiceThread()->addPoll(*acc);
            return acc;
          },
          py::return_value_policy::reference)
      .def_property_readonly("type_table",
                             [](Manifest &m) {
                               std::vector<py::object> ret;
                               std::ranges::transform(m.getTypeTable(),
                                                      std::back_inserter(ret),
                                                      getPyType);
                               return ret;
                             })
      .def_property_readonly("module_infos", &Manifest::getModuleInfos);
}
