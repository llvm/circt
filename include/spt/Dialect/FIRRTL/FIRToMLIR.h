//===- FIRRTL/FIRToMLIR.h - .fir to FIRRTL dialect parser -------*- C++ -*-===//
//
// Defines the interface to the .fir file parser.
//
//===----------------------------------------------------------------------===//

#ifndef SPT_DIALECT_FIRRTL_FIRTOMLIR_H
#define SPT_DIALECT_FIRRTL_FIRTOMLIR_H

namespace spt {
namespace firrtl {

void registerFIRRTLToMLIRTranslation();

} // namespace firrtl
} // namespace spt

#endif // SPT_DIALECT_FIRRTL_FIRTOMLIR_H
