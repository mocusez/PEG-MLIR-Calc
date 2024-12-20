# Naive MLIR Calculator

Inspire By the https://github.com/yhirose/cpp-peglib

And reference work from [MLIR Toy Dialect](https://github.com/llvm/llvm-project/tree/main/mlir/docs/Tutorials/Toy)

## How to use

```
git clone https://github.com/mocusez/PEG-MLIR-Calc
```

Setup CMake  MLIR environment on Debian-sid with MLIR Environment  -> [CMake_MLIR_Toy](https://github.com/mocusez/CMake_MLIR_Toy)

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

### Vector Add(SIMD)

input: `[1,2,3,6]+[2,3,8,5]`

output:

```
module {
  func.func @main() -> vector<4xi32> {
    %cst = arith.constant dense<[1, 2, 3, 6]> : vector<4xi32>
    %cst_0 = arith.constant dense<[2, 3, 8, 5]> : vector<4xi32>
    %0 = arith.addi %cst, %cst_0 : vector<4xi32>
    return %0 : vector<4xi32>
  }
}
```



## Use Passes

input: `50+5`

output:

```
module {
  func.func @main() -> i32 {
    %c55_i32 = arith.constant 55 : i32
    return %c55_i32 : i32
  }
}
```



## Lower to LLVM Dialect

input: `50+5`

output:

```
module {
  llvm.func @main() -> i32 {
    %0 = llvm.mlir.constant(55 : i32) : i32
    llvm.return %0 : i32
  }
}
```



## Output to LLVM IR

Just for reference, test at x86_64 Linux



input: `50+5`

output: 

```llvm IR
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define noundef i32 @main() local_unnamed_addr #0 {
  ret i32 55
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) }
```



input: `[1,2,3,6]+[2,3,8,5]`

output: 

```
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define noundef <4 x i32> @main() local_unnamed_addr #0 {
  ret <4 x i32> <i32 3, i32 5, i32 11, i32 11>
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) }
```



## Run on JIT

Just for reference, test at x86_64 Linux

input: `50+5`

output: `55`



input: `[1,2,3,6]+[2,3,8,5]`

output: `[3, 5, 11, 11]`
