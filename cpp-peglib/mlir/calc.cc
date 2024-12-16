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
        Start       <- ArrayExpr / ScalarExpr
        ArrayExpr   <- Array '+' Array / Array
        ScalarExpr  <- Additive
        Additive    <- Multiplicative '+' Additive / Multiplicative
        Multiplicative   <- Primary '*' Multiplicative^cond / Primary
        Array       <- '[' Numbers ']'
        Primary     <- '(' ScalarExpr ')' / Number
        Numbers     <- Number (',' Number)*
        Number      <- < [0-9]+ >
        %whitespace <- [ \t]*
        cond <- '' { error_message "missing multiplicative" }
    )");

  assert(static_cast<bool>(parser) == true);

  parser["Start"] = [](const SemanticValues& vs) {
    return vs[0];
  };

  parser["ArrayExpr"] = [](const SemanticValues& vs) {
    switch (vs.choice()) {
      case 0: { // Array '+' Array
        auto left = std::any_cast<std::vector<int>>(vs[0]);
        auto right = std::any_cast<std::vector<int>>(vs[1]);
        simd_work(left,right);
        std::vector<int> result;
        for (size_t i = 0; i < left.size(); i++) {
          result.push_back(left[i] + right[i]);
        }
        return result;
      }
      default: // Array
        return std::any_cast<std::vector<int>>(vs[0]);
    }
  };

  parser["ScalarExpr"] = [](const SemanticValues& vs) {
    return std::any_cast<int>(vs[0]);
  };

  parser["Additive"] = [](const SemanticValues& vs) {
    switch (vs.choice()) {
      case 0: // Multiplicative '+' Additive
        arith_work(any_cast<int>(vs[0]),any_cast<int>(vs[1]),ArithOp::ADD);
        return 0;
      default: // Multiplicative
        return std::any_cast<int>(vs[0]);
    }
  };

  parser["Multiplicative"] = [](const SemanticValues& vs) {
    switch (vs.choice()) {
      case 0: // Primary '*' Multiplicative
        arith_work(any_cast<int>(vs[0]),any_cast<int>(vs[1]),ArithOp::MUL);
        return 0;
      default: // Primary
        return std::any_cast<int>(vs[0]);
    }
  };

  parser["Array"] = [](const SemanticValues& vs) {
    return std::any_cast<std::vector<int>>(vs[0]);
  };

  parser["Numbers"] = [](const SemanticValues& vs) {
    std::vector<int> numbers;
    numbers.push_back(std::any_cast<int>(vs[0]));
    for (size_t i = 1; i < vs.size(); i++) {
      numbers.push_back(std::any_cast<int>(vs[i]));
    }
    return numbers;
  };

  parser["Number"] = [](const SemanticValues& vs) {
    return vs.token_to_number<int>();
  };

  // (4) Parse
  parser.enable_packrat_parsing(); // Enable packrat parsing.

  std::string expr = readExpressionFromFile("input.txt");
  parser.parse(expr);
}
