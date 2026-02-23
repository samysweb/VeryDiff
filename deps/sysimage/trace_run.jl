using VeryDiff

sysimage_dir = @__DIR__

# --epsilon 0.05 $sysimage_dir/../../test/examples/nets/ACASXU_run2a_1_1_batch_2000.onnx $sysimage_dir/../../test/examples/nets/ACASXU_run2a_1_1_batch_2000_pruned5.onnx $sysimage_dir/../../test/examples/specs/prop_1.vnnlib

VeryDiff.run_cmd([
    "--epsilon", "0.05",
    "$sysimage_dir/../../test/examples/nets/ACASXU_run2a_1_1_batch_2000.onnx",
    "$sysimage_dir/../../test/examples/nets/ACASXU_run2a_1_1_batch_2000_pruned5.onnx",
    "$sysimage_dir/../../test/examples/specs/prop_1.vnnlib"
])

# --naive --epsilon 0.05 $sysimage_dir/../../test/examples/nets/ACASXU_run2a_1_1_batch_2000.onnx $sysimage_dir/../../test/examples/nets/ACASXU_run2a_1_1_batch_2000_pruned5.onnx $sysimage_dir/../../test/examples/specs/prop_1.vnnlib

VeryDiff.run_cmd([
    "--naive", "--epsilon", "0.05",
    "$sysimage_dir/../../test/examples/nets/ACASXU_run2a_1_1_batch_2000.onnx",
    "$sysimage_dir/../../test/examples/nets/ACASXU_run2a_1_1_batch_2000_pruned5.onnx",
    "$sysimage_dir/../../test/examples/specs/prop_1.vnnlib"
])

# --epsilon 0.005 $sysimage_dir/../../test/examples/nets/ACASXU_run2a_1_1_batch_2000.onnx $sysimage_dir/../../test/examples/nets/ACASXU_run2a_1_1_batch_2000_pruned5.onnx $sysimage_dir/../../test/examples/specs/prop_1.vnnlib

VeryDiff.run_cmd([
    "--epsilon", "0.005",
    "$sysimage_dir/../../test/examples/nets/ACASXU_run2a_1_1_batch_2000.onnx",
    "$sysimage_dir/../../test/examples/nets/ACASXU_run2a_1_1_batch_2000_pruned5.onnx",
    "$sysimage_dir/../../test/examples/specs/prop_1.vnnlib"
])

# --top-1 $sysimage_dir/../../test/examples/nets/mnist_relu_3_100.onnx $sysimage_dir/../../test/examples/nets/mnist_relu_3_100_pruned5.onnx $sysimage_dir/../../test/examples/specs/mnist_0_local_15.vnnlib

VeryDiff.run_cmd([
    "--top-1",
    "$sysimage_dir/../../test/examples/nets/mnist_relu_3_100.onnx",
    "$sysimage_dir/../../test/examples/nets/mnist_relu_3_100_pruned5.onnx",
    "$sysimage_dir/../../test/examples/specs/mnist_0_local_15.vnnlib"
])

# --naive --top-1 $sysimage_dir/../../test/examples/nets/mnist_relu_3_100.onnx $sysimage_dir/../../test/examples/nets/mnist_relu_3_100_pruned5.onnx $sysimage_dir/../../test/examples/specs/mnist_0_local_15.vnnlib

VeryDiff.run_cmd([
    "--naive",
    "--top-1",
    "$sysimage_dir/../../test/examples/nets/mnist_relu_3_100.onnx",
    "$sysimage_dir/../../test/examples/nets/mnist_relu_3_100_pruned5.onnx",
    "$sysimage_dir/../../test/examples/specs/mnist_0_local_15.vnnlib"
])

# --top-1 $sysimage_dir/../../test/examples/nets/mnist_relu_3_100.onnx $sysimage_dir/../../test/examples/nets/mnist_relu_3_100_pruned5.onnx $sysimage_dir/../../test/examples/specs/mnist_7_local_15.vnnlib

VeryDiff.run_cmd([
    "--top-1",
    "$sysimage_dir/../../test/examples/nets/mnist_relu_3_100.onnx",
    "$sysimage_dir/../../test/examples/nets/mnist_relu_3_100_pruned5.onnx",
    "$sysimage_dir/../../test/examples/specs/mnist_7_local_15.vnnlib"
])

# --top-1-delta 0.999 $sysimage_dir/../../test/examples/nets/2_80-1.onnx $sysimage_dir/../../test/examples/nets/2_80-1-0.1.onnx  $sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib

VeryDiff.run_cmd([
    "--top-1-delta", "0.999",
    "$sysimage_dir/../../test/examples/nets/2_80-1.onnx",
    "$sysimage_dir/../../test/examples/nets/2_80-1-0.1.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

# --top-1-delta 0.5 $sysimage_dir/../../test/examples/nets/2_80-1.onnx $sysimage_dir/../../test/examples/nets/2_80-1-0.1.onnx  $sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib

VeryDiff.run_cmd([
    "--top-1-delta", "0.5",
    "$sysimage_dir/../../test/examples/nets/2_80-1.onnx",
    "$sysimage_dir/../../test/examples/nets/2_80-1-0.1.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

# --naive --top-1-delta 0.76 $sysimage_dir/../../test/examples/nets/2_80-1.onnx $sysimage_dir/../../test/examples/nets/2_80-1-0.1.onnx  $sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib

VeryDiff.run_cmd([
    "--naive", "--top-1-delta", "0.76",
    "$sysimage_dir/../../test/examples/nets/2_80-1.onnx",
    "$sysimage_dir/../../test/examples/nets/2_80-1-0.1.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

# --top-1-delta 0.999 $sysimage_dir/../../test/examples/nets/2_80-1.onnx $sysimage_dir/../../test/examples/nets/2_80-1-0.1.onnx  $sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib

VeryDiff.run_cmd([
    "--top-1-delta", "0.9999",
    "$sysimage_dir/../../test/examples/nets/2_80-1.onnx",
    "$sysimage_dir/../../test/examples/nets/2_80-1-0.1.onnx",
    "$sysimage_dir/../../test/examples/specs/sigma_0.1.vnnlib"
])

# german_credit-not-norm.onnx german_credit-not-norm.onnx test/examples/specs-new/german-credit-bounds.vnnlib --robustness-delta=0.75 --input-epsilon=4e-2
VeryDiff.run_cmd([
    "$sysimage_dir/../../test/examples/nets-new/german_credit-not-norm.onnx",
    "-",
    "$sysimage_dir/../../test/examples/specs-new/german-credit-bounds.vnnlib",
    "--robustness-delta=0.75",
    "--input-epsilon=2e-2"
])

VeryDiff.run_cmd([
    "$sysimage_dir/../../test/examples/nets-new/german_credit-not-norm.onnx",
    "-",
    "$sysimage_dir/../../test/examples/specs-new/german-credit-bounds.vnnlib",
    "--robustness-delta=0.75",
    "--input-epsilon=2e-2",
    "--only-split-diff"
])