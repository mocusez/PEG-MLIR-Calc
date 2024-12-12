# Naive MLIR Calculator

Inspire By the https://github.com/yhirose/cpp-peglib

🚨🚨**Warning: Work in Progress**🚨🚨

## How to use

Setup CMake environment on Debian-sid with MLIR Environment

then

```
echo "30*2" >> input.txt
./build/cpp-peglib/mlir/mlir-calc
```

## Result

### Mul

Input:

```
30*2
```

output：

```
module {
  func.func @main() -> i32 {
    %c30_i32 = arith.constant 30 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = arith.addi %c30_i32, %c2_i32 : i32
    return %0 : i32
  }
}
```

### Add

Input:

```
50+5
```

output：

```
module {
  func.func @main() -> i32 {
    %c50_i32 = arith.constant 50 : i32
    %c5_i32 = arith.constant 5 : i32
    %0 = arith.addi %c50_i32, %c5_i32 : i32
    return %0 : i32
  }
}
```

