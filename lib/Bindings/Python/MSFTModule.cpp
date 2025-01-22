//===- MSFTModule.cpp - MSFT API nanobind module --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIRCTModules.h"

#include "circt-c/Dialect/MSFT.h"
#include "circt/Dialect/MSFT/MSFTDialect.h"

#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

#include "NanobindUtils.h"
namespace nb = nanobind;

// using namespace circt;
// using namespace circt::msft;
using namespace mlir::python::nanobind_adaptors;

static nb::handle getPhysLocationAttr(MlirAttribute attr) {
  return nb::module_::import_("circt.dialects.msft")
      .attr("PhysLocationAttr")(attr)
      .release();
}

class PrimitiveDB {
public:
  PrimitiveDB(MlirContext ctxt) { db = circtMSFTCreatePrimitiveDB(ctxt); }
  ~PrimitiveDB() { circtMSFTDeletePrimitiveDB(db); }
  bool addPrimitive(MlirAttribute locAndPrim) {
    return mlirLogicalResultIsSuccess(
        circtMSFTPrimitiveDBAddPrimitive(db, locAndPrim));
  }
  bool isValidLocation(MlirAttribute loc) {
    return circtMSFTPrimitiveDBIsValidLocation(db, loc);
  }

  CirctMSFTPrimitiveDB db;
};

class PlacementDB {
public:
  PlacementDB(MlirModule top, PrimitiveDB *seed) {
    db = circtMSFTCreatePlacementDB(top, seed ? seed->db
                                              : CirctMSFTPrimitiveDB{nullptr});
  }
  ~PlacementDB() { circtMSFTDeletePlacementDB(db); }
  MlirOperation place(MlirOperation instOp, MlirAttribute loc,
                      std::string subpath, MlirLocation srcLoc) {
    auto cSubpath = mlirStringRefCreate(subpath.c_str(), subpath.size());
    return circtMSFTPlacementDBPlace(db, instOp, loc, cSubpath, srcLoc);
  }
  void removePlacement(MlirOperation locOp) {
    circtMSFTPlacementDBRemovePlacement(db, locOp);
  }
  bool movePlacement(MlirOperation locOp, MlirAttribute newLoc) {
    return mlirLogicalResultIsSuccess(
        circtMSFTPlacementDBMovePlacement(db, locOp, newLoc));
  }
  MlirOperation getInstanceAt(MlirAttribute loc) {
    return circtMSFTPlacementDBGetInstanceAt(db, loc);
  }
  nb::handle getNearestFreeInColumn(PrimitiveType prim, uint64_t column,
                                    uint64_t nearestToY) {
    auto cPrim = static_cast<CirctMSFTPrimitiveType>(prim);
    MlirAttribute nearest = circtMSFTPlacementDBGetNearestFreeInColumn(
        db, cPrim, column, nearestToY);
    if (!nearest.ptr)
      return nb::none();
    return getPhysLocationAttr(nearest);
  }
  void walkPlacements(
      nb::callable pycb,
      std::tuple<nb::object, nb::object, nb::object, nb::object> bounds,
      nb::object prim, nb::object walkOrder) {

    auto handleNone = [](nb::object o) {
      return o.is_none() ? -1 : nb::cast<int64_t>(o);
    };
    int64_t cBounds[4] = {
        handleNone(std::get<0>(bounds)), handleNone(std::get<1>(bounds)),
        handleNone(std::get<2>(bounds)), handleNone(std::get<3>(bounds))};
    CirctMSFTPrimitiveType cPrim;
    if (prim.is_none())
      cPrim = -1;
    else
      cPrim = nb::cast<CirctMSFTPrimitiveType>(prim);

    CirctMSFTWalkOrder cWalkOrder;
    if (!walkOrder.is_none())
      cWalkOrder = nb::cast<CirctMSFTWalkOrder>(walkOrder);
    else
      cWalkOrder = CirctMSFTWalkOrder{CirctMSFTDirection::NONE,
                                      CirctMSFTDirection::NONE};

    circtMSFTPlacementDBWalkPlacements(
        db,
        [](MlirAttribute loc, MlirOperation locOp, void *userData) {
          nb::gil_scoped_acquire gil;
          nb::callable pycb = *((nb::callable *)(userData));
          pycb(loc, locOp);
        },
        cBounds, cPrim, cWalkOrder, &pycb);
  }

private:
  CirctMSFTPlacementDB db;
};

class PyLocationVecIterator {
public:
  /// Get item at the specified position, translating a nullptr to None.
  static std::optional<MlirAttribute> getItem(MlirAttribute locVec,
                                              intptr_t pos) {
    MlirAttribute loc = circtMSFTLocationVectorAttrGetElement(locVec, pos);
    if (loc.ptr == nullptr)
      return std::nullopt;
    return loc;
  }

  PyLocationVecIterator(MlirAttribute attr) : attr(attr) {}
  PyLocationVecIterator &dunderIter() { return *this; }

  std::optional<MlirAttribute> dunderNext() {
    if (nextIndex >= circtMSFTLocationVectorAttrGetNumElements(attr)) {
      throw nb::stop_iteration();
    }
    return getItem(attr, nextIndex++);
  }

  static void bind(nb::module_ &m) {
    nb::class_<PyLocationVecIterator>(m, "LocationVectorAttrIterator")
        .def("__iter__", &PyLocationVecIterator::dunderIter)
        .def("__next__", &PyLocationVecIterator::dunderNext);
  }

private:
  MlirAttribute attr;
  intptr_t nextIndex = 0;
};

/// Populate the msft python module.
void circt::python::populateDialectMSFTSubmodule(nb::module_ &m) {
  mlirMSFTRegisterPasses();

  m.doc() = "MSFT dialect Python native extension";

  m.def("replaceAllUsesWith", &circtMSFTReplaceAllUsesWith);

  nb::enum_<PrimitiveType>(m, "PrimitiveType")
      .value("M20K", PrimitiveType::M20K)
      .value("DSP", PrimitiveType::DSP)
      .value("FF", PrimitiveType::FF)
      .export_values();

  nb::enum_<CirctMSFTDirection>(m, "Direction")
      .value("NONE", CirctMSFTDirection::NONE)
      .value("ASC", CirctMSFTDirection::ASC)
      .value("DESC", CirctMSFTDirection::DESC)
      .export_values();

  mlir_attribute_subclass(m, "PhysLocationAttr",
                          circtMSFTAttributeIsAPhysLocationAttribute)
      .def_classmethod(
          "get",
          [](nb::object cls, PrimitiveType devType, uint64_t x, uint64_t y,
             uint64_t num, MlirContext ctxt) {
            return cls(circtMSFTPhysLocationAttrGet(ctxt, (uint64_t)devType, x,
                                                    y, num));
          },
          "Create a physical location attribute", nb::arg(),
          nb::arg("dev_type"), nb::arg("x"), nb::arg("y"), nb::arg("num"),
          nb::arg("ctxt") = nb::none())
      .def_property_readonly(
          "devtype",
          [](MlirAttribute self) {
            return (PrimitiveType)circtMSFTPhysLocationAttrGetPrimitiveType(
                self);
          })
      .def_property_readonly("x",
                             [](MlirAttribute self) {
                               return circtMSFTPhysLocationAttrGetX(self);
                             })
      .def_property_readonly("y",
                             [](MlirAttribute self) {
                               return circtMSFTPhysLocationAttrGetY(self);
                             })
      .def_property_readonly("num", [](MlirAttribute self) {
        return circtMSFTPhysLocationAttrGetNum(self);
      });

  mlir_attribute_subclass(m, "PhysicalBoundsAttr",
                          circtMSFTAttributeIsAPhysicalBoundsAttr)
      .def_classmethod(
          "get",
          [](nb::object cls, uint64_t xMin, uint64_t xMax, uint64_t yMin,
             uint64_t yMax, MlirContext ctxt) {
            auto physicalBounds =
                circtMSFTPhysicalBoundsAttrGet(ctxt, xMin, xMax, yMin, yMax);
            return cls(physicalBounds);
          },
          "Create a PhysicalBounds attribute", nb::arg("cls"), nb::arg("xMin"),
          nb::arg("xMax"), nb::arg("yMin"), nb::arg("yMax"),
          nb::arg("context") = nb::none());

  mlir_attribute_subclass(m, "LocationVectorAttr",
                          circtMSFTAttributeIsALocationVectorAttribute)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirType type, std::vector<nb::handle> pylocs,
             MlirContext ctxt) {
            // Get a LocationVector being sensitive to None in the list of
            // locations.
            SmallVector<MlirAttribute> locs;
            for (auto attrHandle : pylocs)
              if (attrHandle.is_none())
                locs.push_back({nullptr});
              else
                locs.push_back(mlirPythonCapsuleToAttribute(
                    mlirApiObjectToCapsule(attrHandle).ptr()));
            return cls(circtMSFTLocationVectorAttrGet(ctxt, type, locs.size(),
                                                      locs.data()));
          },
          "Create a LocationVector attribute", nb::arg("cls"), nb::arg("type"),
          nb::arg("locs"), nb::arg("context") = nb::none())
      .def("reg_type", &circtMSFTLocationVectorAttrGetType)
      .def("__len__", &circtMSFTLocationVectorAttrGetNumElements)
      .def("__getitem__", &PyLocationVecIterator::getItem,
           "Get the location at the specified position", nb::arg("pos"))
      .def("__iter__",
           [](MlirAttribute arr) { return PyLocationVecIterator(arr); });
  PyLocationVecIterator::bind(m);

  nb::class_<PrimitiveDB>(m, "PrimitiveDB")
      .def(nb::init<MlirContext>(), nb::arg("ctxt") = nb::none())
      .def("add_primitive", &PrimitiveDB::addPrimitive,
           "Inform the DB about a new placement.", nb::arg("loc_and_prim"))
      .def("is_valid_location", &PrimitiveDB::isValidLocation,
           "Query the DB as to whether or not a primitive exists.",
           nb::arg("loc"));

  nb::class_<PlacementDB>(m, "PlacementDB")
      .def(nb::init<MlirModule, PrimitiveDB *>(), nb::arg("top"),
           nb::arg("seed") = nullptr)
      .def("place", &PlacementDB::place, "Place a dynamic instance.",
           nb::arg("dyn_inst"), nb::arg("location"), nb::arg("subpath"),
           nb::arg("src_location") = nb::none())
      .def("remove_placement", &PlacementDB::removePlacement,
           "Remove a placement.", nb::arg("location"))
      .def("move_placement", &PlacementDB::movePlacement,
           "Move a placement to another location.", nb::arg("old_location"),
           nb::arg("new_location"))
      .def("get_nearest_free_in_column", &PlacementDB::getNearestFreeInColumn,
           "Find the nearest free primitive location in column.",
           nb::arg("prim_type"), nb::arg("column"), nb::arg("nearest_to_y"))
      .def("get_instance_at", &PlacementDB::getInstanceAt,
           "Get the instance at location. Returns None if nothing exists "
           "there. Otherwise, returns (path, subpath, op) of the instance "
           "there.")
      .def("walk_placements", &PlacementDB::walkPlacements,
           "Walk the placements, with possible bounds. Bounds are (xmin, xmax, "
           "ymin, ymax) with 'None' being unbounded.",
           nb::arg("callback"),
           nb::arg("bounds") =
               std::make_tuple(nb::none(), nb::none(), nb::none(), nb::none()),
           nb::arg("prim_type") = nb::none(),
           nb::arg("walk_order") = nb::none());

  nb::class_<CirctMSFTWalkOrder>(m, "WalkOrder")
      .def(nb::init<CirctMSFTDirection, CirctMSFTDirection>(),
           nb::arg("columns") = CirctMSFTDirection::NONE,
           nb::arg("rows") = CirctMSFTDirection::NONE);
}
