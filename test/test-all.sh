#!/bin/bash

# Define test cases as associative array
declare -A test_cases=(
    ["add.sh"]="55"
    ["mul.sh"]="60"
    ["simd_add.sh"]="[3, 5, 11, 11]"
)

# Run all tests
for script in "${!test_cases[@]}"; do
    expected="${test_cases[$script]}"
    
    # Make script executable
    chmod +x "$script"
    
    # Run test
    output=$(./"$script")
    if [ "$output" = "$expected" ]; then
        echo "✅ $script test passed: output is $expected"
    else
        echo "❌ $script test failed: Expected $expected, got $output"
        exit 1
    fi
done

echo "🎉 All tests passed!"
exit 0