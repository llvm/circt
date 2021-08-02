#include "circt/Query/Query.h"

using namespace circt::query;

namespace querytool {
namespace parser {

Filter parse(llvm::StringRef source, bool &errored);

} /* namespace parser */
} /* namespace querytool */
