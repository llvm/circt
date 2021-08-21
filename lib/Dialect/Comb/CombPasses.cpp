#include "circt/Dialect/Comb/CombOps.h"

void circt::comb::registerCombPasses() {
  circt::comb::registerCombAnalysisPasses();
}
