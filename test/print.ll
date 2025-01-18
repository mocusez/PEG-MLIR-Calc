; filepath: runtime.ll

; Declare the external printf function
declare i32 @printf(i8* nocapture readonly, ...) nounwind

; Global string constants
@.str.newline = private unnamed_addr constant [2 x i8] c"\0A\00"
@.str.close = private unnamed_addr constant [2 x i8] c"]\00"
@.str.comma = private unnamed_addr constant [3 x i8] c", \00"
@.str.f32 = private unnamed_addr constant [3 x i8] c"%f\00"
@.str.open = private unnamed_addr constant [2 x i8] c"[\00"
@.str.i64 = private unnamed_addr constant [3 x i8] c"%d\00"

; Function implementations
define void @printNewline() {
    %1 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.newline, i64 0, i64 0))
    ret void
}

define void @printClose() {
    %1 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.close, i64 0, i64 0))
    ret void
}

define void @printComma() {
    %1 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.comma, i64 0, i64 0))
    ret void
}

define void @printF32(float %f) {
    %1 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.f32, i64 0, i64 0), float %f)
    ret void
}

define void @printOpen() {
    %1 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.open, i64 0, i64 0))
    ret void
}

define void @printI64(i64 %i) {
    %1 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.i64, i64 0, i64 0), i64 %i)
    ret void
}