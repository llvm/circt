# TODO: Make the check source feature optional as an argument on *_add_library.
set(LLVM_OPTIONAL_SOURCES
  Comb.cpp
  ESI.cpp
  MSFT.cpp
  HW.cpp
  LLHD.cpp
  Seq.cpp
  SV.cpp
)

add_mlir_public_c_api_library(CIRCTCAPIComb
  Comb.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTComb
  )

add_mlir_public_c_api_library(CIRCTCAPIESI
  ESI.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTESI
  )

add_mlir_public_c_api_library(CIRCTCAPIMSFT
  MSFT.cpp

  DEPENDS
  MLIRTransforms

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRTransforms
  CIRCTMSFT
  )

add_mlir_public_c_api_library(CIRCTCAPIHW
  HW.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTHW
  )

add_mlir_public_c_api_library(CIRCTCAPILLHD
  LLHD.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTLLHD
  )

add_mlir_public_c_api_library(CIRCTCAPISeq
  Seq.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTSeq
  )

add_mlir_public_c_api_library(CIRCTCAPISV
  SV.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTSV
  CIRCTSVTransforms
  )
