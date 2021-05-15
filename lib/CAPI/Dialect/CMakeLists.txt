# TODO: Make the check source feature optional as an argument on *_add_library.
set(LLVM_OPTIONAL_SOURCES
  Comb.cpp
  ESI.cpp
  MSFT.cpp
  HW.cpp
  Seq.cpp
  SV.cpp
)

add_circt_library(CIRCTCAPIComb

  Comb.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir-c

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTComb
  )

add_circt_library(CIRCTCAPIESI

  ESI.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir-c

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTESI
  )

add_circt_library(CIRCTCAPIMSFT

  MSFT.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir-c

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTMSFT
  CIRCTMSFTExportTcl
  )

add_circt_library(CIRCTCAPIHW

  HW.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir-c

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTHW
  )

add_circt_library(CIRCTCAPISeq

  Seq.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir-c

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTSeq
  )

add_circt_library(CIRCTCAPISV

  SV.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir-c

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTSV
  CIRCTSVTransforms
  )
