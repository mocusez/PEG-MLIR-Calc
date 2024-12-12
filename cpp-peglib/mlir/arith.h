#ifndef ARITH_H
#define ARITH_H

enum class ArithOp {
    ADD,  // Addition
    SUB,  // Subtraction 
    MUL,  // Multiplication
    DIV   // Division
};
int arith_work(int first,int second,ArithOp type);

#endif // ARITH_H