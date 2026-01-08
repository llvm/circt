add_circt_conversion_library(CIRCTFSMToSMT FSMToSMT.cpp

                                 DEPENDS CIRCTConversionPassIncGen

                                     LINK_COMPONENTS Core

                                         LINK_LIBS PUBLIC MLIRSMT

                                             CIRCTComb CIRCTVerif CIRCTHW
                                                 CIRCTEmit CIRCTFSM CIRCTSupport
                                                     MLIRTransforms)