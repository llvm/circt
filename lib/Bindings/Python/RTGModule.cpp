//===- RTGModule.cpp - RTG API nanobind module ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIRCTModules.h"

#include "circt-c/Dialect/RTG.h"

#include "mlir/Bindings/Python/NanobindAdaptors.h"

#include <nanobind/nanobind.h>

namespace nb = nanobind;

using namespace circt;
using namespace mlir::python::nanobind_adaptors;

/// Populate the rtg python module.
void circt::python::populateDialectRTGSubmodule(nb::module_ &m) {
  m.doc() = "RTG dialect Python native extension";

  //===--------------------------------------------------------------------===//
  // Types
  //===--------------------------------------------------------------------===//

  // Sequence Types
  //===--------------------------------------------------------------------===//

  mlir_type_subclass(m, "SequenceType", rtgTypeIsASequence)
      .def_classmethod(
          "get",
          [](nb::object cls, std::vector<MlirType> &elementTypes,
             MlirContext ctxt) {
            return cls(rtgSequenceTypeGet(ctxt, elementTypes.size(),
                                          elementTypes.data()));
          },
          nb::arg("self"), nb::arg("elementTypes") = std::vector<MlirType>(),
          nb::arg("ctxt") = nullptr)
      .def_property_readonly(
          "num_elements",
          [](MlirType self) { return rtgSequenceTypeGetNumElements(self); })
      .def("get_element", [](MlirType self, unsigned i) {
        return rtgSequenceTypeGetElement(self, i);
      });

  mlir_type_subclass(m, "RandomizedSequenceType", rtgTypeIsARandomizedSequence)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt) {
            return cls(rtgRandomizedSequenceTypeGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);

  // Common Datastructure Types
  //===--------------------------------------------------------------------===//

  mlir_type_subclass(m, "SetType", rtgTypeIsASet)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirType elementType) {
            return cls(rtgSetTypeGet(elementType));
          },
          nb::arg("self"), nb::arg("element_type"))
      .def_property_readonly("element_type", [](MlirType self) {
        return rtgSetTypeGetElementType(self);
      });

  mlir_type_subclass(m, "BagType", rtgTypeIsABag)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirType elementType) {
            return cls(rtgBagTypeGet(elementType));
          },
          nb::arg("self"), nb::arg("element_type"))
      .def_property_readonly("element_type", [](MlirType self) {
        return rtgBagTypeGetElementType(self);
      });

  mlir_type_subclass(m, "DictType", rtgTypeIsADict)
      .def_classmethod(
          "get",
          [](nb::object cls,
             const std::vector<std::pair<MlirAttribute, MlirType>> &entries,
             MlirContext ctxt) {
            std::vector<MlirAttribute> names;
            std::vector<MlirType> types;
            for (auto entry : entries) {
              names.push_back(entry.first);
              types.push_back(entry.second);
            }
            return cls(
                rtgDictTypeGet(ctxt, types.size(), names.data(), types.data()));
          },
          nb::arg("self"),
          nb::arg("entries") =
              std::vector<std::pair<MlirAttribute, MlirType>>(),
          nb::arg("ctxt") = nullptr);

  mlir_type_subclass(m, "ArrayType", rtgTypeIsAArray)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirType elementType, MlirContext ctxt) {
            return cls(rtgArrayTypeGet(elementType));
          },
          nb::arg("self"), nb::arg("element_type"), nb::arg("ctxt") = nullptr)
      .def_property_readonly("element_type", [](MlirType self) {
        return rtgArrayTypeGetElementType(self);
      });

  mlir_type_subclass(m, "MapType", rtgTypeIsAMap)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirType keyType, MlirType valueType) {
            return cls(rtgMapTypeGet(keyType, valueType));
          },
          nb::arg("self"), nb::arg("key_type"), nb::arg("value_type"))
      .def_property_readonly(
          "key_type", [](MlirType self) { return rtgMapTypeGetKeyType(self); })
      .def_property_readonly("value_type", [](MlirType self) {
        return rtgMapTypeGetValueType(self);
      });

  mlir_type_subclass(m, "TupleType", rtgTypeIsATuple)
      .def_classmethod(
          "get",
          [](nb::object cls, const std::vector<MlirType> &fieldTypes,
             MlirContext ctxt) {
            return cls(
                rtgTupleTypeGet(ctxt, fieldTypes.size(), fieldTypes.data()));
          },
          nb::arg("self"), nb::arg("field_types") = std::vector<MlirType>(),
          nb::arg("ctxt") = nullptr)
      .def_property_readonly("fields", [](MlirType self) {
        std::vector<MlirType> fields;
        for (intptr_t i = 0; i < rtgTypeGetNumFields(self); ++i)
          fields.push_back(rtgTupleTypeGetFieldType(self, i));
        return fields;
      });

  mlir_type_subclass(m, "StringType", rtgTypeIsAString)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt) {
            return cls(rtgStringTypeGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);

  // Types for ISA targets
  //===--------------------------------------------------------------------===//

  mlir_type_subclass(m, "LabelType", rtgTypeIsALabel)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctxt) {
            return cls(rtgLabelTypeGet(ctxt));
          },
          nb::arg("self"), nb::arg("ctxt") = nullptr);

  mlir_type_subclass(m, "ImmediateType", rtgTypeIsAImmediate)
      .def_classmethod(
          "get",
          [](nb::object cls, uint32_t width, MlirContext ctx) {
            return cls(rtgImmediateTypeGet(ctx, width));
          },
          nb::arg("self"), nb::arg("width"), nb::arg("ctx") = nullptr)
      .def_property_readonly("width", [](MlirType self) {
        return rtgImmediateTypeGetWidth(self);
      });

  mlir_type_subclass(m, "MemoryBlockType", rtgTypeIsAMemoryBlock)
      .def_classmethod(
          "get",
          [](nb::object cls, uint32_t addressWidth, MlirContext ctxt) {
            return cls(rtgMemoryBlockTypeGet(ctxt, addressWidth));
          },
          nb::arg("self"), nb::arg("address_width"), nb::arg("ctxt") = nullptr)
      .def_property_readonly("address_width", [](MlirType self) {
        return rtgMemoryBlockTypeGetAddressWidth(self);
      });

  mlir_type_subclass(m, "MemoryType", rtgTypeIsAMemory)
      .def_classmethod(
          "get",
          [](nb::object cls, uint32_t addressWidth, MlirContext ctxt) {
            return cls(rtgMemoryTypeGet(ctxt, addressWidth));
          },
          nb::arg("self"), nb::arg("address_width"), nb::arg("ctxt") = nullptr)
      .def_property_readonly("address_width", [](MlirType self) {
        return rtgMemoryTypeGetAddressWidth(self);
      });

  //===--------------------------------------------------------------------===//
  // Attributes
  //===--------------------------------------------------------------------===//

  mlir_attribute_subclass(m, "DefaultContextAttr", rtgAttrIsADefaultContextAttr)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirType type, MlirContext ctxt) {
            return cls(rtgDefaultContextAttrGet(ctxt, type));
          },
          nb::arg("self"), nb::arg("type"), nb::arg("ctxt") = nullptr);

  mlir_attribute_subclass(m, "AnyContextAttr", rtgAttrIsAAnyContextAttr)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirType type, MlirContext ctxt) {
            return cls(rtgAnyContextAttrGet(ctxt, type));
          },
          nb::arg("self"), nb::arg("type"), nb::arg("ctxt") = nullptr);

  mlir_attribute_subclass(m, "MapAttr", rtgAttrIsAMap)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirType mapType,
             const std::vector<std::pair<MlirAttribute, MlirAttribute>>
                 &entries,
             MlirContext ctxt) {
            std::vector<MlirAttribute> keys;
            std::vector<MlirAttribute> values;
            for (auto entry : entries) {
              keys.push_back(entry.first);
              values.push_back(entry.second);
            }
            return cls(rtgMapAttrGet(ctxt, mapType, keys.size(), keys.data(),
                                     values.data()));
          },
          nb::arg("self"), nb::arg("map_type"),
          nb::arg("entries") =
              std::vector<std::pair<MlirAttribute, MlirAttribute>>(),
          nb::arg("ctxt") = nullptr)
      .def(
          "lookup",
          [](MlirAttribute self, MlirAttribute key) {
            auto val = rtgMapAttrLookup(self, key);
            if (mlirAttributeIsNull(val))
              return nb::none();
            return nb::cast(val);
          },
          nb::arg("key"),
          "Look up the value associated with the given key. Returns None if "
          "the key is not found.");

  // Attributes for ISA targets
  //===--------------------------------------------------------------------===//

  nb::enum_<RTGLabelVisibility>(m, "LabelVisibility")
      .value("LOCAL", RTG_LABEL_VISIBILITY_LOCAL)
      .value("GLOBAL", RTG_LABEL_VISIBILITY_GLOBAL)
      .value("EXTERNAL", RTG_LABEL_VISIBILITY_EXTERNAL)
      .export_values();

  mlir_attribute_subclass(m, "LabelVisibilityAttr",
                          rtgAttrIsALabelVisibilityAttr)
      .def_classmethod(
          "get",
          [](nb::object cls, RTGLabelVisibility visibility, MlirContext ctxt) {
            return cls(rtgLabelVisibilityAttrGet(ctxt, visibility));
          },
          nb::arg("self"), nb::arg("visibility"), nb::arg("ctxt") = nullptr)
      .def_property_readonly("value", [](MlirAttribute self) {
        return rtgLabelVisibilityAttrGetValue(self);
      });

  mlir_attribute_subclass(m, "ImmediateAttr", rtgAttrIsAImmediate)
      .def_classmethod(
          "get",
          [](nb::object cls, uint32_t width, uint64_t value, MlirContext ctx) {
            return cls(rtgImmediateAttrGet(ctx, width, value));
          },
          nb::arg("self"), nb::arg("width"), nb::arg("value"),
          nb::arg("ctx") = nullptr)
      .def_property_readonly(
          "width",
          [](MlirAttribute self) { return rtgImmediateAttrGetWidth(self); })
      .def_property_readonly("value", [](MlirAttribute self) {
        return rtgImmediateAttrGetValue(self);
      });

  mlir_attribute_subclass(m, "VirtualRegisterConfigAttr",
                          rtgAttrIsAVirtualRegisterConfig)
      .def_classmethod(
          "get",
          [](nb::object cls, const std::vector<MlirAttribute> &allowedRegs,
             MlirContext ctxt) {
            return cls(rtgVirtualRegisterConfigAttrGet(ctxt, allowedRegs.size(),
                                                       allowedRegs.data()));
          },
          nb::arg("self"), nb::arg("allowed_regs"), nb::arg("ctxt") = nullptr)
      .def_property_readonly("regs", [](MlirAttribute self) {
        std::vector<MlirAttribute> regs;
        for (unsigned
                 i = 0,
                 numRegs = rtgVirtualRegisterConfigAttrGetNumRegisters(self);
             i < numRegs; ++i)
          regs.push_back(rtgVirtualRegisterConfigAttrGetRegister(self, i));
        return regs;
      });

  mlir_attribute_subclass(m, "LabelAttr", rtgAttrIsALabel)
      .def_classmethod(
          "get",
          [](nb::object cls, const std::string &name, MlirContext ctxt) {
            MlirStringRef nameRef =
                mlirStringRefCreate(name.data(), name.size());
            return cls(rtgLabelAttrGet(ctxt, nameRef));
          },
          nb::arg("self"), nb::arg("name"), nb::arg("ctxt") = nullptr)
      .def_property_readonly("name", [](MlirAttribute self) {
        MlirStringRef name = rtgLabelAttrGetName(self);
        return nb::str(name.data, name.length);
      });
}
