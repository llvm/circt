#include <iostream>
#include <optional>
#include <string>
#include <vector>


using namespace llvm;
using namespace mlir;
using namespace circt;
using namespace std;

struct transition {
  string from;
  string to;
  string guard;
  bool isGuard;
  vector<string> action;
  bool isAction;
  string output;
  bool isOutput;
};
