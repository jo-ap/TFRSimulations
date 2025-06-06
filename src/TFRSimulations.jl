module TFRSimulations

using Colors
using ColorSchemes
using DifferentialEquations 
using GLMakie
using Latexify
using LaTeXStrings
using RuntimeGeneratedFunctions
using TikhonovFenichelReductions
import Oscar: FracFieldElem, QQMPolyRingElem

RuntimeGeneratedFunctions.init(@__MODULE__)

## export

export simgui

## globals 

# default settings
# size for Makie plots
const FIG_SIZE = (800,600)
# range for sliders 
const SLIDER_RANGE = LinRange(0, 10, 101)
# format value of slider
const SLIDER_FORMAT = x -> "$(round(x; digits=1))"
# number of time points for time series plot
const N_TP = 500
# algorithm for ode solver
const ODE_ALG = AutoTsit5(Rosenbrock23())

## GUI for simulations of ODEProblems

# plot ODESolution with series
Makie.convert_arguments(T::Type{<:Series}, sol::ODESolution) =  Makie.convert_arguments(T, sol.t, hcat(sol.u...))

# get default colors for plot
function default_colors(n::Int)
  if n <= 7
    return Makie.wong_colors()
  elseif n <= 10
    return colorschemes[:seaborn_colorblind].colors
  end
  return get(colorschemes[:batlow], range(0.0, 1.0, length = n))
end
  
## Generate Julia code from Reduction type and parse to simgui

# build julia function for ODE solver
function _generate_function_expression(
    f::Union{Vector{FracFieldElem{QQMPolyRingElem}}, Vector{QQMPolyRingElem}},
    u::Vector{QQMPolyRingElem},
    p::Vector{QQMPolyRingElem}
  )
  str = "function(du,u,p,t)\n" * 
    join(string.(u), ", ") * " = u\n" * 
    join(string.(p), ", ") * " = p\n" *
    "du .= [\n" *
    join(string.(f), ",\n") *
    "\n]\n" *
    "return nothing\n" * 
    "end"
  str = replace(str, "//" => "/")
  return Meta.parse(str)
end
_generate_function_expression(reduction::Reduction) = _generate_function_expression(reduction.g, reduction.problem.x, reduction.problem.p)

function _to_func(
    M::Union{Vector{FracFieldElem{QQMPolyRingElem}}, Vector{QQMPolyRingElem}},
    u::Vector{QQMPolyRingElem},
    p::Vector{QQMPolyRingElem}
  )
  str = "function(u,p)\n" * 
    join(string.(u), ", ") * " = u\n" * 
    join(string.(p), ", ") * " = p\n" *
    "return [\n" *
    join(string.(M), ",\n") *
    "\n]\n" *
    "end"
  str = replace(str, "//" => "/")
  return Meta.parse(str)
end

## GUI for reduction

function idx_small_all(reduction)
  i_sf = collect(1:length(reduction.problem.p))[reduction.problem.idx_slow_fast]
  return i_sf[.!reduction.sf_separation]
end

function pad_lims(x; padding=0.1, min_range=0.01)
  m, M = x
  Δ = max(M - m, min_range)
  return (m - padding*Δ, M + padding*Δ)
end

function get_axis_lims(sol::ODESolution)
  t_lims  = (sol.t[1], sol.t[end])
  y_lims = (minimum(minimum.(sol.u)), maximum(maximum.(sol.u)))
  return t_lims, y_lims
end

function get_axis_lims(sol::ODESolution, sol2::ODESolution)
  t_lims  = (sol.t[1], sol.t[end])
  y_lims1 = (minimum(minimum.(sol.u)), maximum(maximum.(sol.u)))
  y_lims2 = (minimum(minimum.(sol2.u)), maximum(maximum.(sol2.u)))
  y_lims = (min(y_lims1[1], y_lims2[1]), max(y_lims1[2], y_lims2[2]))
  return t_lims, y_lims
end

function simgui(
    reduction::Reduction;
    x0::Vector{Float64}=ones(Float64, length(reduction.problem.x)),
    p::Vector{Float64}=abs.(randn(length(reduction.problem.p))),
    include_full_system::Bool=true,
    colors=default_colors(length(reduction.problem.x)),
    slider_range::Union{Vector{<:AbstractVector}, AbstractVector}=SLIDER_RANGE,
    ode_alg::SciMLBase.AbstractODEAlgorithm=ODE_ALG,
    n_tp::Int=N_TP,
    latexify::Bool=false
  )

  # multiply small parameters by ε
  i_small = idx_small_all(reduction)
  function update_parameters(p)
    ε = p[end]
    _p = p[1:end-1]
    _p[i_small] .= ε*_p[i_small]
    return _p
  end

  # create function for ODE solver
  f! = @RuntimeGeneratedFunction(_generate_function_expression(reduction))
  if include_full_system 
    f_orig! = @RuntimeGeneratedFunction(_generate_function_expression(reduction.problem.f, reduction.problem.x, reduction.problem.p))
  end

  # set initial condition on slow manifold
  update_initial_condition = @RuntimeGeneratedFunction(_to_func(reduction.M, reduction.problem.x, reduction.problem.p))
  
  # init figure and axes
  fig = Figure(size = (1000, 750));
  ax_sol = Axis(fig[1,1]);

  # Sliders
  p₀ = [p..., 0.1, x0...]
  name_parameters = [string.(reduction.problem.p)..., "ε"]
  name_components = string.(reduction.problem.x)
  slider_name = [name_parameters; [n * "(0)" for n in name_components]]
  if latexify
    slider_name = Latexify.latexify.(slider_name)
    name_components = Latexify.latexify.(name_components)
  end
  if isa(slider_range, Vector{<:AbstractVector})
    @assert length(slider_range) == length([p; x0]) "you must either provide a single slider range or one for parameter and state varieble, i.e. a vector with the same length as [p; x0]"
    slider_range = [slider_range[1:length(p)]..., LinRange(0.01, 1, 100), slider_range[length(p)+1:end]...]
  else 
    slider_range = [[slider_range for _ in 1:(length(p))]..., LinRange(0.01,1,100), [slider_range for _ in x0]...]
  end
  slider = [
    (label = slider_name[i],
      range = slider_range[i],
      format = i in i_small ? x ->  "ε⋅$(round(x, digits = 3))" : x -> "$(round(x, digits = 3))")
    for i in eachindex(slider_name)
  ]
  # create a label grid to hold all sliders and add to figure
  lsgrid = SliderGrid(fig, slider..., tellheight=false)
  # unpack values of sliders for further use
  sliderobservables = [s.value for s in lsgrid.sliders]
  for i = 1:length(sliderobservables)
    set_close_to!(lsgrid.sliders[i], p₀[i])
  end
  slider_params = lift(sliderobservables...) do slvalues...
    [slvalues...]
  end
  start_on_M_btn = Button(fig, label = "x(0) on slow manifold", tellwidth=false)
  print_parameters_btn = Button(fig, label = "print parameters", tellwidth=false)
  fig[1,2] = vgrid!(lsgrid, print_parameters_btn, start_on_M_btn)

  # Time slider
  time_grid = SliderGrid(fig[2,1], (label = L"T_{max}", range = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10_000, 20_000, 50_000], format = x->"$x"))
  T_end = time_grid.sliders[1].value
  _tspan = @lift (0.0, Float64($T_end))

  # Simulation
  # unpack parameters and initial conditions
  _p = @lift update_parameters($(slider_params)[1:length(p)+1])
  _u₀ = @lift $(slider_params)[(length(p)+2):end]
  _u₀_slow_manifold = @lift update_initial_condition($_u₀, $_p)

  on(start_on_M_btn.clicks) do clicks
    for i in eachindex(_u₀_slow_manifold[])
      set_close_to!(lsgrid.sliders[length(p)+1+i], _u₀_slow_manifold[][i])
    end
  end
  on(print_parameters_btn.clicks) do clicks 
    println("p = " * string(slider_params[][1:length(p)]))
    println("u₀ = " * string(slider_params[][length(p)+2:end]))
  end
  
  # ODE solution (reduction)
  ode_problem = @lift ODEProblem(f!, $_u₀, $_tspan, $_p)
  sol = @lift solve($ode_problem, alg=ode_alg; saveat=LinRange(0, $T_end, n_tp)) 
  series!(ax_sol, sol; linewidth=3, labels=name_components, color=colors)
  axislegend(ax_sol)
  
  # include solution of full system
  if include_full_system 
    ode_problem_full = @lift ODEProblem(f_orig!, $_u₀, $_tspan, $_p)
    sol_full = @lift solve($ode_problem_full, alg=ode_alg; saveat=LinRange(0, $T_end, n_tp)) 
    series!(ax_sol, sol_full; linewidth=3, color=colors, linestyle=:dot)
    axis_lims = @lift get_axis_lims($sol, $sol_full)
  else
    axis_lims = @lift get_axis_lims($sol)
  end

  # update limits
  on(axis_lims) do axis_lims
    x_lims = pad_lims(axis_lims[1])
    y_lims = pad_lims(axis_lims[2])
    # if solutions explode, upper limits can become inf 
    # in that case the corresponding lines can leave the axis
    if any(isinf.(Float32.(y_lims)))
      autolimits!(ax_sol)
    else
      xlims!(ax_sol, x_lims)
      ylims!(ax_sol, y_lims)
    end
   end

  # update plot now
  sol[] = sol[]

  # layout
  rowgap!(lsgrid.layout, 7)
  colsize!(fig.layout, 1, Relative(2/3))
  
  return fig
end

end # module TFRSimulations
