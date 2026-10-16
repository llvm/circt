//===- SimModule.cpp - Sim API nanobind module ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIRCTModules.h"
#include "circt-c/Dialect/Sim.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

#include <nanobind/stl/string.h>
#include <stdexcept>
#include <vector>

namespace nb = nanobind;
using namespace mlir::python::nanobind_adaptors;

void circt::python::populateDialectSimSubmodule(nb::module_ &m) {
  m.doc() = "Sim dialect Python native extension";

  nb::enum_<SimDPIDirection>(m, "DPIDirection")
      .value("INPUT", SIM_DPI_DIRECTION_INPUT)
      .value("OUTPUT", SIM_DPI_DIRECTION_OUTPUT)
      .value("INOUT", SIM_DPI_DIRECTION_INOUT)
      .value("RETURN", SIM_DPI_DIRECTION_RETURN)
      .value("REF", SIM_DPI_DIRECTION_REF);

  mlir_type_subclass(m, "DPIFunctionType", simTypeIsADPIFunction)
      .def_classmethod(
          "get",
          [](nb::object cls, nb::list pyArguments, MlirContext ctx) {
            std::vector<std::string> names;
            std::vector<SimDPIArgument> arguments;
            names.reserve(pyArguments.size());
            for (auto pyArgument : pyArguments) {
              auto argument = nb::cast<nb::tuple>(pyArgument);
              if (argument.size() != 3)
                throw std::invalid_argument(
                    "DPI arguments must be (name, type, direction) tuples");
              auto type = nb::cast<MlirType>(argument[1]);
              if (!mlirContextEqual(ctx, mlirTypeGetContext(type)))
                throw std::invalid_argument(
                    "DPI argument type belongs to a different context");
              names.push_back(nb::cast<std::string>(argument[0]));
              arguments.push_back({mlirStringRefCreate(names.back().data(),
                                                       names.back().size()),
                                   type,
                                   nb::cast<SimDPIDirection>(argument[2])});
            }
            return cls(
                simDPIFunctionTypeGet(ctx, arguments.size(), arguments.data()));
          },
          "Create a DPI function type from (name, type, direction) tuples.",
          nb::arg("cls"), nb::arg("arguments"), nb::arg("context") = nb::none())
      .def_property_readonly(
          "arguments",
          [](MlirType self) {
            nb::list arguments;
            intptr_t count = simDPIFunctionTypeGetNumArguments(self);
            for (intptr_t i = 0; i < count; ++i) {
              auto argument = simDPIFunctionTypeGetArgument(self, i);
              arguments.append(nb::make_tuple(
                  nb::str(argument.name.data, argument.name.length),
                  argument.type, argument.direction));
            }
            return arguments;
          },
          "Arguments as (name, type, direction) tuples in declaration order.")
      .def_property_readonly("function_type", [](MlirType self) {
        return simDPIFunctionTypeGetFunctionType(self);
      });
}
