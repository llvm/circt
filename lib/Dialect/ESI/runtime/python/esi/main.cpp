#include "esi/Accelerator.h"

// pybind11 includes
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
namespace py = pybind11;

using namespace esi;

PYBIND11_MODULE(esi, m) {
  py::class_<Accelerator>(m, "Accelerator")
      .def_static("connect", &Accelerator::connect,
                  py::return_value_policy::take_ownership)
      .def("sysinfo", &Accelerator::sysInfo,
           py::return_value_policy::reference_internal);
  py::class_<SysInfo>(m, "SysInfo")
      .def("esi_version", &SysInfo::esiVersion)
      .def("raw_json_manifest", &SysInfo::rawJsonManifest);
}
