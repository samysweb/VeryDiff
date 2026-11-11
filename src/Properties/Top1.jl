const TOP1_FOUND_CONCRETE_DELTA = Ref{Bool}(false)

function get_top1_property(;delta=zero(Float64),naive=false)
    if !iszero(delta)
        @assert 0.5 <= delta && delta <= 1.0
        dist=log(delta/(1-delta))
    else
        dist=0.0
    end
    return (N1, N2, Zin, Zout, verification_status) -> begin
        global TOP1_FOUND_CONCRETE_DELTA
        if VeryDiff.FIRST_ROUND[]
            TOP1_FOUND_CONCRETE_DELTA[] = false
        end
        if isnothing(verification_status)
            verification_status = Dict{Tuple{Int,Int},Bool}()
        end
        input_dim = length(Zin.Z₁.c)
        res1 = N1(Zin.Z₁.c)
        res2 = N2(Zin.Z₂.c)
        argmax_N1 = argmax(res1)
        argmax_N2 = argmax(res2)
        # Output of N1/N2 already has Softmax applied now, so no need to compute here.
        softmax_N1 = res1
        if argmax_N1 != argmax_N2
            if iszero(delta) || softmax_N1[argmax_N1] >= delta
                println("Found cex")
                println("N1 Probability: $(softmax_N1[argmax_N1]) >= $delta")
                return false, (Zin.Z₁.c, (argmax_N1, argmax_N2)), nothing, nothing, 0.0
            else
                second_largest = sort(res1,rev=true)[2]
                if !iszero(delta) && res1[argmax_N1]-second_largest >= dist
                    println("Found spurious cex")
                    println("N1 Probability: $(softmax_N1[argmax_N1]) < $delta")
                    println("but difference $(res1[argmax_N1]-second_largest) >= $dist (approximate bound)")
                end
            end
        end
        property_satisfied = true
        distance_bound = 0.0
        any_feasible = false
        common_generator_indices = union(
            Zout.Z₁.generator_ids,
            union(Zout.Z₂.generator_ids, Zout.∂Z.generator_ids))
        variable_offsets = Int64[1]
        for id in common_generator_indices
            for curZ in [Zout.∂Z, Zout.Z₁, Zout.Z₂]
                id_pos = attempt_find_index_position(curZ.generator_ids, id)
                if id_pos > 0
                    next_offset = variable_offsets[end] + size(curZ.Gs[id_pos], 2)
                    push!(variable_offsets, next_offset)
                    break
                end
            end
        end
        indices₁ = intersect_indices(common_generator_indices, Zout.Z₁.generator_ids)
        indices₂ = intersect_indices(common_generator_indices, Zout.Z₂.generator_ids)
        ∂indices = intersect_indices(common_generator_indices, Zout.∂Z.generator_ids)

        in_indices₁ = intersect_indices(common_generator_indices, Zin.Z₁.generator_ids)
        in_indices₂ = intersect_indices(common_generator_indices, Zin.Z₂.generator_ids)
        output_dim = length(Zout.Z₁.c)
        for top_index in 1:output_dim
            if USE_GUROBI[]
                model = Model(() -> Gurobi.Optimizer(GRB_ENV[]))
            else
                model = Model(GLPK.Optimizer)
            end
            set_time_limit_sec(model, 10)
            var_num = variable_offsets[end]-1
            @variable(model,-1.0 <= x[1:var_num] <= 1.0)
            
            # Constraint 1: Maximal output of first network is top_index
            offset1_start = variable_offsets[indices₁[1]]
            offset1_end = variable_offsets[indices₁[1]+1] - 1
            lhs = ((@view Zout.Z₁.Gs[1][1:end .!= top_index,:]) .- (@view Zout.Z₁.Gs[1][top_index:top_index,:]))*x[offset1_start:offset1_end]
            rhs = (Zout.Z₁.c[top_index] .- (@view Zout.Z₁.c[1:end .!= top_index])) .- dist
            
            for i in 2:length(Zout.Z₁.Gs)
                curG = Zout.Z₁.Gs[i]
                curIdx = indices₁[i]
                offset_start = variable_offsets[curIdx]
                offset_end = variable_offsets[curIdx + 1] - 1
                lhs .+= ((@view curG[1:end .!= top_index, :]) .- (@view curG[top_index:top_index,:])) * x[offset_start:offset_end]
            end
            @constraint(model, lhs .<= rhs)

            if !naive
                # Constraint 2:
                # Output difference between networks is given by the differential zonotope (∂Z)
                #       ∂Z - (Z₁ - Z₂) = 0
                # <->   (∂G + ∂c) - ((G₁ + c₁) - (G₂ + c₂)) = 0
                # <->   ∂G + ∂c - G₁ - c₁ + G₂ + c₂ = 0
                # <->   ∂G - G₁ + G₂ = c₁ - c₂ - ∂c
                G2 = zeros(output_dim, var_num)
                for (i, curIdx) in enumerate(∂indices)
                    offset_start = variable_offsets[curIdx]
                    offset_end = variable_offsets[curIdx + 1] - 1
                    G2[:,offset_start:offset_end] .= Zout.∂Z.Gs[i]
                end

                for (i, curIdx) in enumerate(indices₁)
                    offset_start = variable_offsets[curIdx]
                    offset_end = variable_offsets[curIdx + 1] - 1
                    G2[:,offset_start:offset_end] .-= Zout.Z₁.Gs[i]
                end

                for (i, curIdx) in enumerate(indices₂)
                    offset_start = variable_offsets[curIdx]
                    offset_end = variable_offsets[curIdx + 1] - 1
                    G2[:, offset_start:offset_end] .+= Zout.Z₂.Gs[i]
                end
                @constraint(model,
                    G2*x .== (Zout.Z₁.c .- Zout.∂Z.c .- Zout.Z₂.c)
                )
            end

            # Check if model is feasible
            @objective(model,Max,0)
            optimize!(model)
            
            if termination_status(model) == MOI.INFEASIBLE
                # Model is infeasible -> top_index is never maximal with delta
                for other_index in 1:output_dim
                    verification_status[(top_index,other_index)]=true
                end
            else
                # Model is feasible -> top_index can be maximal in NN1 with delta
                # Now check that all output dimensions of NN2 (other than top_index)
                # are less than the output at top_index

                # ...but before we do that:
                # Check if we found concrete evidence for feasibility of confidence delta
                if !iszero(delta) && !TOP1_FOUND_CONCRETE_DELTA[]
                    input1 = copy(Zin.Z₁.c)
                    for (i, curIdx) in enumerate(in_indices₁)
                        offset_start = variable_offsets[curIdx]
                        offset_end = offset_start + size(Zin.Z₁.Gs[i],2) - 1
                        input1 .+= Zin.Z₁.Gs[i] * value.(x[offset_start:offset_end])
                    end
                    # input2 = copy(Zin.Z₂.c)
                    # for (i, curIdx) in enumerate(in_indices₂)
                    #     offset_start = variable_offsets[curIdx]
                    #     offset_end = offset_start + size(Zin.Z₂.Gs[i],2)
                    #     input2 .+= Zin.Z₂.Gs[i] * value.(x[offset_start:offset_end])
                    # end
                    res1 = N1(input1)
                    argmax_N1 = argmax(res1)
                    # Output of N1/N2 already has Softmax applied now, so no need to compute here.
                    softmax_N1 = res1
                    if softmax_N1[argmax_N1] >= delta
                        println("[TOP-1] required confidence ($(softmax_N1[argmax_N1])≥$delta) is feasible for index $argmax_N1")
                        TOP1_FOUND_CONCRETE_DELTA[]=true
                    else
                        #println("[TOP-1] did not find required confidence yet.")
                    end
                end
                # ...also before we do that:
                # Record that we found at least one feasible top_index
                any_feasible = true
                
                # ...now we'll check the output dimensions of NN2:
                for other_index in 1:output_dim
                    if other_index != top_index && !haskey(verification_status, (top_index,other_index))
                        # We are indeed considering an output index other than top_index
                        # This other index has also not yet been verified

                        # Set up objective:
                        # Maximize gap between other_index and top_index in NN2
                        a = zeros(var_num)

                        for (i, curIdx) in enumerate(indices₂)
                            offset_start = variable_offsets[curIdx]
                            offset_end = variable_offsets[curIdx + 1] - 1
                            a[offset_start:offset_end] .= Zout.Z₂.Gs[i][other_index,:] .- Zout.Z₂.Gs[i][top_index,:]
                        end
                        @objective(model,Max,a'*x)
                        
                        # Compute threshold for objective
                        threshold = Zout.Z₂.c[top_index]-Zout.Z₂.c[other_index]

                        # If the optimal value is < threshold, then the property is satisfied
                        # (it is impossible for other_index to be greater than top_index)
                        # otherwise (optimal >= threshold) we may have found a counterexample

                        if USE_GUROBI[] # we are using GUROBI -> set objective/bound thresholds
                            set_optimizer_attribute(model, "Cutoff", threshold-1e-6)
                        end
                        optimize!(model)

                        model_status = termination_status(model)
                        # Model MUST be feasible since we did not add any constraints
                        @assert model_status != MOI.INFEASIBLE

                        # Model should be optimal or have reached the objective limit
                        # any other status -> split and retry
                        if model_status != MOI.OPTIMAL && model_status != MOI.OBJECTIVE_LIMIT
                            println("[GUROBI] Irregular model status: $model_status")
                            property_satisfied = false
                            if has_values(model)
                                distance_bound = max(distance_bound, objective_value(model))
                            end
                            continue
                        end
                        
                        # LP has reached optimization limit -> property is satisfied
                        # i.e. other_index cannot exceed top_index
                        if model_status == MOI.OBJECTIVE_LIMIT || objective_value(model) < threshold
                            verification_status[(top_index,other_index)]=true
                        else
                            # Potentially we found a counterexample
                            # -> check that
                            distance_bound = max(distance_bound, objective_value(model))
                            input1 = copy(Zin.Z₁.c)
                            for (i, curIdx) in enumerate(in_indices₁)
                                offset_start = variable_offsets[curIdx]
                                offset_end = offset_start + size(Zin.Z₁.Gs[i],2) - 1
                                input1 .+= Zin.Z₁.Gs[i] * value.(x[offset_start:offset_end])
                            end
                            # Important: Use input generated by Z₂ here
                            # Important due to robustness queries where input Z₁ and Z₂ differ
                            input2 = copy(Zin.Z₂.c)
                            for (i, curIdx) in enumerate(in_indices₂)
                                offset_start = variable_offsets[curIdx]
                                offset_end = offset_start + size(Zin.Z₂.Gs[i],2) - 1
                                input2 .+= Zin.Z₂.Gs[i] * value.(x[offset_start:offset_end])
                            end
                            res1 = N1(input1)
                            res2 = N2(input2)
                            argmax_N1 = argmax(res1)
                            argmax_N2 = argmax(res2)
                            # Output of N1/N2 already has Softmax applied now, so no need to compute here.
                            softmax_N1 = res1
                            if argmax_N1 != argmax_N2
                                # N1 and N2 indeed differ in their classification for input
                                # But does N1 have enough confidence?
                                if iszero(delta) || softmax_N1[argmax_N1] >= delta
                                    # N1 has sufficient confidence -> concrete counterexample
                                    println("Found cex")
                                    second_most = sort(softmax_N1,rev=true)[2]
                                    println("N1 ($argmax_N1): $(softmax_N1[argmax_N1]) (vs. $second_most)")
                                    # Output of N1/N2 already has Softmax applied now, so no need to compute here.
                                    softmax_N2 = res2
                                    println("N2 ($argmax_N2): $(softmax_N2[argmax_N2])")
                                    println("N1 Probability: $(softmax_N1[argmax_N1]) >= $delta")
                                    return false, (input1, (argmax_N1, argmax_N2)), nothing, nothing, 0.0
                                else
                                    # N1 does not have enough confidence
                                    second_largest = sort(res1,rev=true)[2]
                                    if !iszero(delta) && res1[argmax_N1]-second_largest >= dist
                                        println("Found spurious cex")
                                        println("N1 Probability: $(softmax_N1[argmax_N1]) < $delta")
                                        println("but difference $(res1[argmax_N1]-second_largest) >= $dist (approximate bound)")
                                    end
                                    property_satisfied = false
                                end
                            else
                                # Counterexample was spurious in the sense that N1 and N2
                                # have the same maximum
                                property_satisfied = false
                            end
                        end
                    end
                end
            end
            # generator_importance .*= sum(abs,Zin.Z₁.G,dims=1)[1,:]
        end
        @assert !iszero(delta) || any_feasible "One output must be maximal, but our analysis says there is no maximum -- this smells like a bug!"
        # if property_satisfied
        #     println("Zonotope Top 1 Equivalent!")
        # end
        return property_satisfied, nothing, nothing, verification_status, distance_bound
    end
end

function top1_configure_split_heuristic(mode)
    return (Zin,Zout,_heuristics_info,verification_task) -> begin
        distance_indices = verification_task.distance_indices
        if VeryDiff.NEW_HEURISTIC[]
            diff_weights = zeros(size(Zin.Z₁.influence[1],1))
            for (g1,inf1) in zip(Zout.Z₁.Gs,Zout.Z₁.influence)
                diff_weights .+= sum(abs, abs.(g1) * abs.(inf1'),dims=1)[1,:]
            end
            for (g2,inf2) in zip(Zout.Z₂.Gs,Zout.Z₂.influence)
                diff_weights .+= sum(abs,abs.(g2) * abs.(inf2'),dims=1)[1,:]
            end
        else
            throw("Old heuristic no longer implemented")
        end
        # println(diff_weights)


        offset = 0
        if mode == 2
            diff_weights = @view diff_weights[(size(distance_indices,1)+1):end]
            offset = size(distance_indices,1)
        end

        d = argmax(
            diff_weights
        )[1] + offset
        #return distance_indices[d]
        # println("Splitting on dimension $d")
        return d
    end
end