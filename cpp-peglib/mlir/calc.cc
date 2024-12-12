#include <assert.h>
#include <iostream>
#include <peglib.h>

#include <fstream>
#include <string>

using namespace peg;
using namespace std;

#include "arith.h"

std::string readExpressionFromFile(const std::string& filename) {
    std::ifstream file(filename);
    std::string line;
    if (getline(file, line)) {
        return line;
    }
    return "";
}


int main(void) {
  // (2) Make a parser
  parser parser(R"(
        # Grammar for Calculator...
        Additive    <- Multiplicative '+' Additive / Multiplicative
        Multiplicative   <- Primary '*' Multiplicative^cond / Primary
        Primary     <- '(' Additive ')' / Number
        Number      <- < [0-9]+ >
        %whitespace <- [ \t]*
        cond <- '' { error_message "missing multiplicative" }
    )");

  assert(static_cast<bool>(parser) == true);

  // (3) Setup actions
  parser["Additive"] = [](const SemanticValues &vs) {
    switch (vs.choice()) {
    case 0: // "Multiplicative '+' Additive"
      arith_work(any_cast<int>(vs[0]),any_cast<int>(vs[1]),ArithOp::ADD);
      return 0;
    default:
      return any_cast<int>(vs[0]);
    }
  };

  parser["Multiplicative"] = [](const SemanticValues &vs) {
    switch (vs.choice()) {
    case 0:
      arith_work(any_cast<int>(vs[0]),any_cast<int>(vs[1]),ArithOp::MUL);
      return 0;
    default: // "Primary"
      return any_cast<int>(vs[0]);
    }
  };

  parser["Number"] = [](const SemanticValues &vs) {
    return vs.token_to_number<int>();
  };

  // (4) Parse
  parser.enable_packrat_parsing(); // Enable packrat parsing.

  std::string expr = readExpressionFromFile("input.txt");
  parser.parse(expr);
}
