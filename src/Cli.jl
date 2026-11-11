
using ArgParse
using VNNLib

function parse_commandline(cmd_args)
    s = ArgParseSettings()
    @add_arg_table s begin
        "net1"
            help        =   "First NN file (ONNX)"
            required    =   true
        "net2"
            help        =   "Second NN file (ONNX)"
            required    =   true
        "spec"
            help        =   "Input specification file (VNNLIB)"
            required    =   true
        "--epsilon"
            help        =   "Verify Epsilon Equivalence; provides the epsilon value"
            arg_type    =   Float64
            default     =   -Inf64
        "--top-1"
            help        =   "Verify Top-1 Equivalence"
            action      =   :store_true
        "--top-1-delta"
            help        =   "Verify δ-Top-1 Equivalence; provides the delta value"
            arg_type    =   Float64
            default     =   -Inf64
        "--robustness-delta"
            help        =   "Verify confidence based robustness with given delta"
            arg_type    =   Float64
            default     =   -Inf64
        "--input-epsilon"
            help        =   "Epsilon value for input perturbation in confidence based robustness"
            arg_type    =   Float64
            default     =   0.0
        "--only-split-diff"
            help        =   "Only split on differential dimensions (only relevant for robustness verification)"
            action      =   :store_true
            default     =   false
        "--timeout"
            help        =   "Timeout for verification"
            arg_type    =   Int
            default     =   0
        "--naive"
            help        =   "Use naive verification (without differential verification)"
            action      =   :store_true
    end
    return parse_args(cmd_args, s)
end

function run_cmd(args)
    parsed_args = parse_commandline(args)

    epsilon = parsed_args["epsilon"]
    top_1 = parsed_args["top-1"]
    top_1_delta = parsed_args["top-1-delta"]
    robustness_delta = parsed_args["robustness-delta"]
    input_epsilon = parsed_args["input-epsilon"]

    net1 = parsed_args["net1"]
    if !isfile(net1)
        error("File not found: $net1")
        return 1
    end
    try
        net1 = load_onnx_model(net1)
    catch
        error("Failed to parse network: $net1")
        return 1
    end
    if parsed_args["net2"] == "-"
        @assert robustness_delta != -Inf64 "For confidence based equivalence, both networks must be specified"
        net2 = parsed_args["net1"]
    else
        net2 = parsed_args["net2"]
    end
    if !isfile(net2)
        error("File not found: $net2")
        return 1
    end
    try
        net2 = load_onnx_model(net2)
    catch
        error("Failed to parse network: $net2")
        return 1
    end
    spec = parsed_args["spec"]
    if !isfile(spec)
        error("File not found: $spec")
        return 1
    end
    n_inputs = nothing
    try
        spec, n_inputs, _ = get_ast(spec)
    catch
        error("Failed to parse specification: $spec")
        return 1
    end

    timeout = parsed_args["timeout"]
    if timeout <= 0
        timeout = Inf
    end

    # Choose property
    property = nothing
    split_heuristic = nothing
    if epsilon != -Inf64
        if parsed_args["naive"]
            property = get_epsilon_property_naive(epsilon)
        else
            property = get_epsilon_property(epsilon)
        end
        split_heuristic = epsilon_split_heuristic
    end
    if top_1
        @assert isnothing(property) "Cannot specify both epsilon and Top-1"
        property = get_top1_property(naive=parsed_args["naive"])
        split_heuristic = top1_configure_split_heuristic(1)
    end
    if top_1_delta != -Inf64
        @assert isnothing(property) "Cannot specify both epsilon and Top-1"
        @assert 0.5 <= top_1_delta < 1.0 "Invalid delta value for Top-1; must be in [0.5,1)"
        property = get_top1_property(delta=top_1_delta, naive=parsed_args["naive"])
        split_heuristic = top1_configure_split_heuristic(1)
    end
    if robustness_delta != -Inf64
        @assert isnothing(property) "Cannot specify both epsilon and robustness delta"
        @assert 0.5 <= robustness_delta < 1.0 "Invalid delta value for robustness; must be in [0.5,1)"
        property = get_top1_property(delta=robustness_delta, naive=parsed_args["naive"])
        if parsed_args["only-split-diff"]
            split_heuristic = top1_configure_split_heuristic(2)
        else
            split_heuristic = top1_configure_split_heuristic(1)
        end
        @assert input_epsilon > 0.0 "Input epsilon must be positive for confidence based robustness verification"
    end
    if isnothing(property)
        error("No property specified")
        return 1
    end

    if parsed_args["naive"]
        println("Using naive verification")
        VeryDiff.USE_DIFFZONO[] = false
    else
        println("Using differential verification")
        VeryDiff.USE_DIFFZONO[] = true
    end

    result = SAFE
    try
        # Run verification
        for (bounds, _, _, num) in spec
            passed_time = @timed begin
                current_result = verify_network(net1, net2, bounds[1:n_inputs,:], property, split_heuristic, timeout=timeout, init_eps=input_epsilon)
            end
            timeout -= passed_time[:time]
            if current_result == UNSAFE
                result = UNSAFE
                break
            elseif current_result == UNKNOWN
                result = UNKNOWN
                break
            end
            if timeout <= 0
                result = UNKNOWN
                break
            end
        end
    catch e
        println("Caught an exception: $e")
        showerror(stdout, e, catch_backtrace())
        println("Aborting verification.")
        result = UNKNOWN
    end
    if result == SAFE
        return 20
    elseif result == UNSAFE
        return 10
    else
        return 0
    end
end


function main_VeryDiff():Cint
    result = run_cmd(ARGS)
    return convert(Cint,result)
end