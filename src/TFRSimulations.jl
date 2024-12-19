module TFRSimulations

import Oscar

using DifferentialEquations 
using GLMakie
using TikhonovFenichelReductions
using Latexify
using LaTeXStrings

## export

export gui

## globals 

# default range for sliders 
SLIDER_RANGE = LinRange(0, 10, 101)
# max time span for auto detection of steady state
TSPAN = (0.0, 1000.0);
# minimum time to integrate (until steady state can be detected and integration stopped)
MIN_T = 5
# number of time points for time series plot
N_TP = 1000
# font size for labels in legend
LABEL_FS = 15
# algorithm for ode solver
ODE_ALG = AutoTsit5(Rosenbrock23())

## Generate Julia function from Oscar types

function _eval_poly(u, f) 
  if f == 0
    return Float64(0) 
  else
    return sum([Float64(c)*prod(u.^a) for (c, a) in Oscar.coefficients_and_exponents(f)]);
  end
end

_to_ode_func(g::Oscar.QQMPolyRingElem) = function (du,u,p,t) 
  du = _eval_poly([u; p],g) 
  return
end
function _to_ode_func(g::Oscar.AbstractAlgebra.Generic.FracFieldElem{Oscar.QQMPolyRingElem})
  g_num = numerator(g)
  g_den = denominator(g)
  return function (du,u,p,t)
    du = _eval_poly([u;p], g_num)/_eval_poly([u;p], g_den)
    return
  end
end
function _to_ode_func(g::Union{Vector{Oscar.AbstractAlgebra.Generic.FracFieldElem{Oscar.QQMPolyRingElem}}, Vector{Oscar.QQMPolyRingElem}})
  p_func = numerator.(g)
  q_func = denominator.(g)
  return function (du,u,p,t)
    du .= [_eval_poly([u;p], _p)/_eval_poly([u;p], _q) for (_p,_q) in zip(p_func, q_func)]
    return
  end
end 


_to_func(g::Oscar.QQMPolyRingElem) = (u,p) -> _eval_poly([u; p],g)
function _to_func(g::Oscar.AbstractAlgebra.Generic.FracFieldElem{Oscar.QQMPolyRingElem})
  g_num = numerator(g)
  g_den = denominator(g)
  (u,p) -> _eval_poly([u;p], g_num)/_eval_poly([u;p], g_den)
end
function _to_func(g::Union{Vector{Oscar.AbstractAlgebra.Generic.FracFieldElem{Oscar.QQMPolyRingElem}}, Vector{Oscar.QQMPolyRingElem}})
  p_func = numerator.(g)
  q_func = denominator.(g)
  (u,p) -> [_eval_poly([u;p], _p)/_eval_poly([u;p], _q) for (_p,_q) in zip(p_func, q_func)]
end 

## Simulate system and plot solution

# simulate ODE system
function simulate(f::Function, θ, u₀; tspan=TSPAN, n_tp=501, stop=:auto, min_t=nothing, alg=ODE_ALG)
  prob = ODEProblem(f, u₀, tspan, θ)
  # detect if steady state is reached
  if stop == :auto
    t_end = isnothing(min_t) ? 1.0 : float(min_t)
    sol = solve(prob, alg, reltol=1e-9, abstol=1e-9, callback=TerminateSteadyState(min_t=min_t))
    t_end = max(t_end, round(sol.t[end], digits=1))
    # remake problem
    tspan = (tspan[1], t_end) 
    prob =  remake(prob, tspan=tspan)
  end
  # save solution in optimal time range with given resolution
  sol = solve(prob, alg, saveat=LinRange(tspan[1], tspan[2], n_tp), reltol=1e-9, abstol=1e-9)
  return sol, tspan
end


## GUI

# plot ODESolution with series
Makie.convert_arguments(T::Type{<:Series}, sol::ODESolution) =  Makie.convert_arguments(T, sol.t, hcat(sol.u...))

# pad limits for axis
function pad_lims(x; padding=0.1, min_range=0.01)
    m, M = x
    Δ = M - m + min_range
    return (m - padding*Δ, M + padding*Δ)
end

## GUI for reduction

function idx_small_all(reduction)
  i_sf = collect(1:length(reduction.idx_slow_fast))[reduction.idx_slow_fast]
  return i_sf[.!reduction.sf_separation]
end

function gui(reduction::TikhonovFenichelReductions.Reduction,
             g;
             x0::Vector{Float64}=Float64[],
             p::Vector{Float64}=Float64[],
             plotfunction::NamedTuple=NamedTuple(),
             slider_range=SLIDER_RANGE,
             alg=ODE_ALG,
             n_tp=N_TP,
             min_t=MIN_T,
             tspan=TSPAN,
             latexify=true,
             colors::AbstractVector=[],
             labels::AbstractVector=[])

  # p = isempty(p) ? (; zip([Symbol.(reduction.p)..., :ε], [[1.0 for _ in reduction.p]..., 0.1])...) : (; p..., ε=0.1)
  # x0 = isempty(x0) ? (; zip(Symbol.(reduction.x), [0.1 for _ in reduction.x])...) : x0
  p = isempty(p) ? [[1.0 for _ in reduction.p]..., 0.1] : [p...; 0.1]
  x0 = isempty(x0) ? [0.1 for _ in reduction.x] : x0
  slider_range = [[slider_range for _ in 1:(length(p)-1)]..., LinRange(0.01,1,100), [slider_range for _ in x0]...]

  # create function for ODE solver
  f! = _to_ode_func(g)

  # multiply small parameters by ε
  i_small = idx_small_all(reduction)
  function update_parameters(p)
    ε = p[end]
    __p = p[1:end-1]
    __p[i_small] .= ε*__p[i_small]
    return __p
  end

  # set initial condition on slow manifold
  update_initial_condition = _to_func(reduction.M)
  
  # init figure and axes
  fig = Figure(size = (1000, 750));
  ax_sol = Axis(fig[1,1]);

  # Sliders
  p₀ = [p; x0]
  name_parameters = [string.(reduction.p)..., "ε"]
  name_components = string.(reduction.x)
  slider_name = [name_parameters; [n * "(0)" for n in name_components]]
  if latexify
    slider_name = Latexify.latexify.(slider_name)
    name_components = Latexify.latexify.(name_components)
  end
  slider_range = slider_range == [] ? [SLIDER_RANGE for _ in [p, x0]] : slider_range
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
  fig[1,2] = vgrid!(lsgrid, start_on_M_btn)

  # Time slider
  time_grid = SliderGrid(fig, (label = L"T_{max}", range = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000], format = x->"$x"))
  T_end = time_grid.sliders[1].value
  T_toggle = Toggle(fig, active = true)
  T_toggle_active = T_toggle.active
  T_label = Label(fig, "detect steady state")
  fig[2,1] = hgrid!(
    time_grid, T_toggle, T_label
  )
  stop_simulate = @lift $T_toggle_active ? :auto : :no
  _tspan = @lift $T_toggle_active ? tspan : (0.0, Float64($T_end))

  # Simulation
  # unpack parameters and initial conditions
  _p = @lift update_parameters($(slider_params)[1:length(p)])
  _u₀ = @lift $(slider_params)[(length(p)+1):end]
  _u₀_slow_manifold = @lift update_initial_condition($_u₀, $_p)

  on(start_on_M_btn.clicks) do clicks
    println("slow manifold: $(_u₀_slow_manifold[])")
    for i in eachindex(_u₀_slow_manifold[])
      set_close_to!(lsgrid.sliders[length(p)+i], _u₀_slow_manifold[][i])
    end
  end
  
  # ODE solution
  sim = @lift simulate(f!, $_p, $_u₀; stop=$stop_simulate, tspan=$_tspan, alg=ODE_ALG, n_tp=n_tp);
  sol = @lift $(sim)[1]
  default_colors = length(x0)<=7 ? Makie.wong_colors() : :tab20
  colors = length(colors) == 0 ? default_colors : colors
  labels = length(labels) == 0 ? name_components : labels
 
  series!(ax_sol, sol; linewidth=2, labels=labels, color=colors)
 
  # # add plot function
  # if length(plotfunction) > 0
  #   _g = @lift plotfunction.g($(sol).u, $_p, $(sol).t, $idx_fp)
  #   _t = @lift $(_g)[1]
  #   _M = @lift $(_g)[2]
  #   _colors = haskey(plotfunction, :colors) == 0 ? default_colors : plotfunction.colors
  #   _labels = haskey(plotfunction, :labels) == 0 ? name_components : plotfunction.labels
  #   _ls = haskey(plotfunction, :linestyle) == 0 ? :dot : plotfunction.linestyle
  #   series!(ax_sol, _t, _M; linewidth=2, linestyle=_ls, labels=_labels, color=_colors)
  # end
  axislegend(ax_sol)

  # update limits
  on(sol) do sol
    x_lims = pad_lims((sol.t[1], sol.t[end]))
    y_lims = pad_lims((minimum(minimum.(sol.u)), maximum(maximum.(sol.u))))
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
  # colsize!(fig.layout, 3, Relative(1/3))
  
  return fig
end

end # module TFRSimulations
