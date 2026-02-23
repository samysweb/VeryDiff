#using Random

@enum VerificationStatus UNKNOWN SAFE UNSAFE

function verify_network(
    N1 :: OnnxNet{LayerIdT,NShapeIn, NShapeOut},
    N2 :: OnnxNet{LayerIdT,NShapeIn, NShapeOut},
    bounds,
    property_check,
    split_heuristic;
    timeout=Inf,
    init_eps=0.0) where {LayerIdT,NShapeIn,NShapeOut}
    global FIRST_ROUND[] = true
    verification_result = nothing
    # Prepare Zonotope Initialization
    input_dim = size(bounds,1)
    low = @view bounds[:,1]
    high = @view bounds[:,2]
    mid = (high.+low) ./ 2
    distance = mid .- low
    input_dim = length(low)

    # Initialize Zonotope
    non_zero_indices = findall((!).(iszero.(distance)))
    distance = distance[non_zero_indices]
    if init_eps > 0.0
        if !all(init_eps .<= (2.0 .* distance))
            println("ERROR: Initial epsilon too large for given bounds!")
            println("Inital Epsilon: $(init_eps); Max Epsilon: $(2.0*minimum(distance))")
            throw(ErrorException("Initial epsilon too large for given bounds!"))
        end
        distance .-= (init_eps/2)
        distance1_secondary = fill(init_eps/2, input_dim)
        distance2_secondary = fill(init_eps/2, input_dim)
        mid1_secondary = zeros(Float64, input_dim)
        mid2_secondary = zeros(Float64, input_dim)
    else
        println("Differential Zonotope initialized with zero perturbation.")
        distance1_secondary = nothing
        distance2_secondary = nothing
        mid1_secondary = nothing
        mid2_secondary = nothing
    end

    N = GeminiNetwork(N1,N2)

    N1 = executable_network(N1)
    N2 = executable_network(N2)

    println("Network initialized.")
    if N.diff_layers[end] isa VeryDiff.Definitions.DiffLayer{VNNLib.OnnxParser.ONNXSoftmax{S},VNNLib.OnnxParser.ONNXSoftmax{S},VNNLib.OnnxParser.ONNXSoftmax{S}} where S
        pop!(N.diff_layers)
        @warn "Removed final Softmax layer from differential network for verification."
        @warn "VeryDiff assumes this is handled by the choice of an appropriate property!"
    end
    @assert length(N.inputs) == 1 "VeryDiff currently only supports single input networks."

    # Statistics
    total_zonos = 1

    #Config
    num_threads = Threads.nthreads()
    println("Running with $(num_threads) threads")
    work_queue = Queue()
    push!(work_queue,
        VerificationTask(
            mid, distance, non_zero_indices,
            distance1_secondary, mid1_secondary,
            distance2_secondary, mid2_secondary,
            nothing, Inf, 1.0 )
    )
    @Debugger.propagation_init_hook(N)
    verification_result = worker_function(
        work_queue,
        1,
        N, N1, N2,
        property_check,
        split_heuristic,
        num_threads;
        timeout=timeout)
    if verification_result == SAFE
        println("SAFE")
    elseif verification_result == UNSAFE
        println("UNSAFE")
    else
        println("UNKNOWN")
    end
    work_queue=nothing
    return verification_result
end

function worker_function(work_queue, threadid, N,N1,N2,property_check, split_heuristic, num_threads;timeout=Inf)
    try
        thread_result = worker_function_internal(work_queue, threadid,N,N1,N2,num_threads, property_check, split_heuristic, timeout=timeout)
        return thread_result
    catch e
        println("[Thread $(threadid)] Caught exception: $(e)")
        showerror(stdout, e, catch_backtrace())
        thread_result = UNKNOWN
        return thread_result
    end
end
function worker_function_internal(work_queue, threadid, N,N1,N2,num_threads, property_check, split_heuristic;timeout=Inf)
    starttime = time_ns()
    prop_state = PropState(true)
    k = 0
    total_zonos=0
    generated_zonos = 0
    splits = 0
    is_verified = SAFE
    should_terminate = length(work_queue) == 0
    do_not_split = false
    total_work = 0.0
    first=true
    loop_time = @elapsed begin
    while !should_terminate
        verification_task = pop!(work_queue)
        prepare_prop_state!(prop_state, verification_task)
        if k == 0
            println("[Thread $(threadid)] Time to first task: $(round((time_ns()-starttime)/1e9;digits=2))s")
        end
        total_zonos+=1
        Zin = prop_state.zono_storage.zonotopes[1].zonotope
        prop_state = propagate!(N,prop_state)
        Zout = prop_state.zono_storage.zonotopes[end].zonotope
        if first
            println("Zono Bounds:")
            bounds = zono_bounds(Zout.∂Z)
            println(bounds[:,1])
            println(bounds[:,2])
            first=false
        end
        prop_satisfied, cex, heuristics_info, verification_status, distance_bound = property_check(N1, N2, Zin, Zout, verification_task.verification_status)
        global FIRST_ROUND[] = false
        if !prop_satisfied
            if !isnothing(cex)
                @assert all(zono_bounds(Zin.Z₁)[:,1] .<= cex[1] .&& cex[1] .<= zono_bounds(Zin.Z₁)[:,2])
                println("\nFound counterexample: $(cex)")
                should_terminate = true
                is_verified = UNSAFE
            elseif !do_not_split
                splits += 1
                split_d = split_heuristic(Zin,Zout,heuristics_info, verification_task)
                Z1, Z2 = split_zono(split_d, verification_task,verification_status, distance_bound)
                Zin=nothing
                push!(work_queue, Z1)
                push!(work_queue, Z2)
                generated_zonos+=2
            end
        else
            total_work += verification_task.work_share
        end
        if (time_ns()-starttime)/1e9 > timeout
            println("\n\nTIMEOUT REACHED")
            println("UNKNOWN")
            should_terminate = true
            is_verified = UNKNOWN
        end
        should_terminate |= length(work_queue) == 0
        k+=1
        if k%100 == 0
            if length(work_queue) > 0
                top_task = peek_queue(work_queue)
                println("[Thread $(threadid)] Processed $(total_zonos) (Work Done: $(round(100*total_work;digits=5))%; Expected: $(total_zonos/total_work); Bound: $(top_task.distance_bound))")
            else
                println("[Thread $(threadid)] Processed $(total_zonos) (Work Done: $(round(100*total_work;digits=5))%; Expected: $(total_zonos/total_work); Queue empty!")
            end
        end
        reset_ps!(prop_state)
    end
    end
    empty!(work_queue)
    if do_not_split
        println("\n\nTIMEOUT REACHED")
        println("UNKNOWN")
        is_verified = UNKNOWN
    end
    println("[Thread $(threadid)] Total splits: $(splits)")
    print("Processed $(total_zonos) zonotopes (Work Done: $(round(100*total_work;digits=1))%); Generated $(generated_zonos) ($(loop_time/k)s/loop)\n")
    return is_verified
end

function split_zono(distance_d, verification_task :: VerificationTask, verification_status, distance_bound)
    # TODO(steuber): Make split prettier?
    work_share_new = verification_task.work_share / 2.0
    if distance_d <= size(verification_task.distance_indices,1)
        #println("Splitting on input dimension $(verification_task.distance_indices[distance_d])")
        input_pos = verification_task.distance_indices[distance_d]
        distance1 = verification_task.distance[distance_d]/2
        distance2 = verification_task.distance[distance_d]/2
        mid1 = verification_task.middle[input_pos] - distance1
        mid2 = verification_task.middle[input_pos] + distance2
        distance1_vec = deepcopy(verification_task.distance)
        distance1_vec[distance_d] = distance1
        middle1_vec = deepcopy(verification_task.middle)
        middle1_vec[input_pos] = mid1
        distance2_vec = verification_task.distance
        distance2_vec[distance_d] = distance2
        middle2_vec = verification_task.middle
        middle2_vec[input_pos] = mid2
        Z1 = VerificationTask(
            middle1_vec, distance1_vec,
            deepcopy(verification_task.distance_indices),
            deepcopy(verification_task.distance1_secondary),
            deepcopy(verification_task.middle1_secondary),
            deepcopy(verification_task.distance2_secondary),
            deepcopy(verification_task.middle2_secondary),
            deepcopy(verification_status),
            distance_bound,
            work_share_new,
            verification_task.task_bounds)
        Z2 = VerificationTask(
            middle2_vec, distance2_vec,
            verification_task.distance_indices,
            verification_task.distance1_secondary,
            verification_task.middle1_secondary,
            verification_task.distance2_secondary,
            verification_task.middle2_secondary,
            verification_status,
            distance_bound,
            work_share_new,
            deepcopy(verification_task.task_bounds))
        return Z1, Z2
    elseif distance_d <= size(verification_task.distance_indices,1) + size(verification_task.distance1_secondary,1)
        input_pos = distance_d - size(verification_task.distance_indices,1)
        #println("Splitting on 1 differential dimension $(input_pos)")
        diff_distance1 = verification_task.distance1_secondary[input_pos]/2
        diff_distance2 = verification_task.distance1_secondary[input_pos]/2
        diff_mid1 = verification_task.middle1_secondary[input_pos] - diff_distance1
        diff_mid2 = verification_task.middle1_secondary[input_pos] + diff_distance2
        distance1_secondary = deepcopy(verification_task.distance1_secondary)
        distance1_secondary[input_pos] = diff_distance1
        middle1_secondary = deepcopy(verification_task.middle1_secondary)
        middle1_secondary[input_pos] = diff_mid1
        distance2_secondary = deepcopy(verification_task.distance1_secondary)
        distance2_secondary[input_pos] = diff_distance2
        middle2_secondary = deepcopy(verification_task.middle1_secondary)
        middle2_secondary[input_pos] = diff_mid2
        Z1 = VerificationTask(
            deepcopy(verification_task.middle), deepcopy(verification_task.distance),
            deepcopy(verification_task.distance_indices),
            distance1_secondary,
            middle1_secondary,
            deepcopy(verification_task.distance2_secondary),
            deepcopy(verification_task.middle2_secondary),
            deepcopy(verification_status),
            distance_bound,
            work_share_new,
            verification_task.task_bounds)
        Z2 = VerificationTask(
            verification_task.middle, verification_task.distance,
            verification_task.distance_indices,
            distance2_secondary,
            middle2_secondary,
            verification_task.distance2_secondary,
            verification_task.middle2_secondary,
            verification_status,
            distance_bound,
            work_share_new,
            deepcopy(verification_task.task_bounds))
        return Z1, Z2
    else
        input_pos = distance_d - size(verification_task.distance_indices,1) - size(verification_task.distance1_secondary,1)
        #println("Splitting on 2 differential dimension $(input_pos)")
        diff_distance1 = verification_task.distance2_secondary[input_pos]/2
        diff_distance2 = verification_task.distance2_secondary[input_pos]/2
        diff_mid1 = verification_task.middle2_secondary[input_pos] - diff_distance1
        diff_mid2 = verification_task.middle2_secondary[input_pos] + diff_distance2
        distance1_secondary = deepcopy(verification_task.distance2_secondary)
        distance1_secondary[input_pos] = diff_distance1
        middle1_secondary = deepcopy(verification_task.middle2_secondary)
        middle1_secondary[input_pos] = diff_mid1
        distance2_secondary = deepcopy(verification_task.distance2_secondary)
        distance2_secondary[input_pos] = diff_distance2
        middle2_secondary = deepcopy(verification_task.middle2_secondary)
        middle2_secondary[input_pos] = diff_mid2
        Z1 = VerificationTask(
            deepcopy(verification_task.middle), deepcopy(verification_task.distance),
            deepcopy(verification_task.distance_indices),
            deepcopy(verification_task.distance1_secondary),
            deepcopy(verification_task.middle1_secondary),
            distance1_secondary,
            middle1_secondary,
            deepcopy(verification_status),
            distance_bound,
            work_share_new,
            verification_task.task_bounds)
        Z2 = VerificationTask(
            verification_task.middle, verification_task.distance,
            verification_task.distance_indices,
            verification_task.distance1_secondary,
            verification_task.middle1_secondary,
            distance2_secondary,
            middle2_secondary,
            verification_status,
            distance_bound,
            work_share_new,
            deepcopy(verification_task.task_bounds))
        return Z1, Z2
    end
end