using Statistics
using CairoMakie
using Associations

module MakeDatasets

using Statistics
using Associations

export change_datasets

"""
change_datasets(init_data_x, init_data_y, loss_func, max_iter, δ; temp=0.4, randomize_num=10, if_corr=true, if_mi=false)

Performs an iterative perturbation of two datasets while preserving specified statistical properties.

# Arguments
* `init_data_x::Vector{Float64}`: Initial dataset for x.
* `init_data_y::Vector{Float64}`: Initial dataset for y.
* `loss_func::Function`: Function that evaluates whether a perturbation should be accepted.
* `max_iter::Int64`: Maximum number of accepted perturbation steps.
* `δ::Float64`: Step size of the perturbation.

# Keyword Arguments
* `temp::Float64=0.4`: Temperature parameter controlling stochastic acceptance.
* `randomize_num::Int64=10`: Number of data points to perturb at each step.
* `if_corr::Bool=true`: Whether to preserve correlation between x and y.
* `if_mi::Bool=false`: Whether to preserve mutual information between x and y.

# Returns
* A tuple `(new_data_x, new_data_y, prob_itr)` where:
* `new_data_x`, `new_data_y`: The modified datasets.
* `prob_itr`: Acceptance ratio (`max_iter / total_trials`).
"""
function change_datasets(
    init_data_x:: Vector{Float64},
    init_data_y:: Vector{Float64},
    loss_func:: Function,
    max_iter:: Int64,
    δ:: Float64;
    temp:: Float64 = 0.4,
    randomize_num:: Int64 = 10,
    if_corr:: Bool = true,
    if_mi:: Bool = false,
)

    current_data_x = copy(init_data_x)
    current_data_y = copy(init_data_y)

    itr = 0
    true_itr = 0
    while itr < max_iter
        test_x, test_y = perturb(current_data_x, current_data_y, loss_func, δ, temp = temp, randomize_num = randomize_num)
        if is_error_ok(test_x, test_y, current_data_x, current_data_y, if_corr, if_mi)
            current_data_x = test_x
            current_data_y = test_y
            itr += 1
        end
        true_itr += 1
    end

    prob_itr = max_iter / true_itr

    return current_data_x, current_data_y, prob_itr
end

function perturb(
    data_x:: Vector{Float64},
    data_y:: Vector{Float64},
    loss_func:: Function,
    δ:: Float64;
    temp:: Float64 = 0.4,
    randomize_num:: Int64 = 10,
)

    loss_check = true
    test_x = copy(data_x)
    test_y = copy(data_y)

    while loss_check
        test_x, test_y = move_random_points(data_x, data_y, δ, randomize_num = randomize_num)

        if loss_func(test_x, test_y, data_x, data_y) || temp > rand()
            loss_check = false
        end
    end

    return test_x, test_y
end

function move_random_points(
    data_x:: Vector{Float64},
    data_y:: Vector{Float64},
    δ:: Float64;
    randomize_num:: Int64 = 10,
)

    test_x = copy(data_x)
    test_y = copy(data_y)

    data_num = length(data_x)

    randomize = Int.(ceil.(rand(randomize_num) * data_num))

    for itr ∈ randomize
        test_x[itr] = test_x[itr] + δ * randn()
        test_y[itr] = test_y[itr] + δ * randn()
    end

    return test_x, test_y
end

function is_error_ok(
    test_x:: Vector{Float64},
    test_y:: Vector{Float64},
    data_x:: Vector{Float64},
    data_y:: Vector{Float64},
    if_corr:: Bool,
    if_mi:: Bool
)
    est = KSG1(MIShannon(); k = 5)
    x̄ = Int(floor(mean(test_x) * 1000))
    ȳ = Int(floor(mean(test_y) * 1000))
    xₛ = Int(floor(std(test_x) * 1000))
    yₛ = Int(floor(std(test_y) * 1000))
    corr = 0.0
    if if_corr
        corr = Int(floor(cor(test_x, test_y) * 1000))
    end

    mi = 0.0
    if if_mi
        mi = Int(floor(association(est, test_x, test_y) * 1000))
    end

    x̄_data = Int(floor(mean(data_x) * 1000))
    ȳ_data = Int(floor(mean(data_y) * 1000))
    xₛ_data = Int(floor(std(data_x) * 1000))
    yₛ_data = Int(floor(std(data_y) * 1000))

    corr_data = 0.0
    if if_corr
        corr_data = Int(floor(cor(data_x, data_y) * 1000))
    end

    mi_data = 0.0
    if if_mi
        mi_data = Int(floor(association(est, data_x, data_y) * 1000))
    end

    error = true
    error = error && (x̄ == x̄_data)
    error = error && (ȳ == ȳ_data)
    error = error && (xₛ == xₛ_data)
    error = error && (yₛ == yₛ_data)

    if if_corr
        error = error && (corr == corr_data)
    end

    if if_mi
        error = error && (mi == mi_data)
    end

    return error
end

end


function loss_func_circle(test_x, test_y, data_x, data_y)

    delta_test = mean(abs.(sqrt.(test_x.^2 .+ test_y.^2) .- 1.0).^5)
    delta_data = mean(abs.(sqrt.(data_x.^2 .+ data_y.^2) .- 1.0).^5)

    return delta_test < delta_data
end

function loss_func_line(test_x, test_y, data_x, data_y)

    delta_test = 0.0
    delta_data = 0.0
    data_num = length(test_x)
    for i = 1:data_num
        delta_test += min(abs(test_x[i] - test_y[i]), abs(test_x[i] + test_y[i])) / data_num
        delta_data += min(abs(data_x[i] - data_y[i]), abs(data_x[i] + data_y[i])) / data_num
    end

    return delta_test < delta_data
end


function MAIN()

    init_data_x = randn(1000)
    init_data_y = randn(1000)

    max_iter = 20000
    δ = 0.01

    temp_arr = [0.4, 0.35, 0.2, 0.1]


    println("correlation")
    test_x, test_y = init_data_x, init_data_y
    prob_corr = zeros(4)
    for itr = 1:4
        println("--- $itr ---")
        test_x, test_y, prob_corr[itr] = MakeDatasets.change_datasets(test_x, test_y,loss_func_line,max_iter,δ,temp = temp_arr[itr], if_corr = true, if_mi = false)
    end

    with_theme(
            Theme(
                fonts = (
                    regular = "CMU Serif",
                    bold = "CMU Serif Bold",
                    italic = "CMU Serif Italic",
                ),
            )
    ) do
        p1 = Figure()
        ax = Axis(p1[1, 1]; aspect = 1, limits = ((-3, 3), (-3, 3)))
        scatter!(ax, init_data_x, init_data_y, label = "init")
        save("init.pdf", p1)

        p2 = Figure()
        ax = Axis(p2[1, 1]; aspect = 1, limits = ((-3, 3), (-3, 3)))
        scatter!(ax, test_x, test_y, label = "correlation")
        save("fix_correlation.pdf", p2)
    end

    println("MI")
    test_mi_x, test_mi_y = init_data_x, init_data_y
    prob_mi = zeros(4)
    for itr = 1:4
        println("--- $itr ---")
        test_mi_x, test_mi_y, prob_mi[itr] = MakeDatasets.change_datasets(test_mi_x,test_mi_y,loss_func_line,max_iter,δ,temp = temp_arr[itr], if_corr = false, if_mi = true)
    end


    with_theme(
            Theme(
                fonts = (
                    regular = "CMU Serif",
                    bold = "CMU Serif Bold",
                    italic = "CMU Serif Italic",
                ),
            )
    ) do
        p3 = Figure()
        ax = Axis(p3[1, 1]; aspect = 1, limits = ((-3, 3), (-3, 3)))
        scatter!(ax, test_mi_x, test_mi_y, label = "mutual_information")
        save("fix_mutual_information.pdf", p3)
    end

    println("x average   : init : $(mean(init_data_x))")
    println("            : corr : $(mean(test_x))")
    println("            :  mi  : $(mean(test_mi_x))")
    println("y average   : init : $(mean(init_data_y))")
    println("            : corr : $(mean(test_y))")
    println("            :  mi  : $(mean(test_mi_y))")
    println("x std       : init : $(std(init_data_x))")
    println("            : corr : $(std(test_x))")
    println("            :  mi  : $(std(test_mi_x))")
    println("y std       : init : $(std(init_data_y))")
    println("            : corr : $(std(test_y))")
    println("            :  mi  : $(std(test_mi_y))")
    println("correlation : init : $(cor(init_data_x, init_data_y))")
    println("            : corr : $(cor(test_x, test_y))")
    println("            :  mi  : $(cor(test_mi_x, test_mi_y))")

    est = KSG1(MIShannon(); k = 5)
    println("MI          : init : $(association(est, init_data_x, init_data_y))")
    println("            : corr : $(association(est, test_x, test_y))")
    println("            :  mi  : $(association(est, test_mi_x, test_mi_y))")


    with_theme(
            Theme(
                fonts = (
                    regular = "CMU Serif",
                    bold = "CMU Serif Bold",
                    italic = "CMU Serif Italic",
                ),
            )
    ) do
        p4 = Figure()
        ax = Axis(p4[1, 1]; xlabel = "temperature", ylabel = "probability")
        qqplot!(ax, temp_arr, prob_corr, label = "fix correlation", qqline = :fit)
        qqplot!(ax, temp_arr, prob_mi, label = "fix mutual information", qqline = :fit)
        axislegend(position = :lt)
        save("prob_scaling.pdf", p4)
    end

end

MAIN()
