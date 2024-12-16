#ifndef ARITH_H
#define ARITH_H
#include <vector>

enum class ArithOp {
    ADD,  // Addition
    SUB,  // Subtraction 
    MUL,  // Multiplication
    DIV   // Division
};
int arith_work(int first,int second,ArithOp type);
int simd_work(const std::vector<int> &values1,const std::vector<int> &values2);

#endif // ARITH_H