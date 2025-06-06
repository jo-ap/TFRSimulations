# TFRSimulations.jl 
A simple package that uses
[DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/) and
[Makie.jl](https://docs.makie.org/stable/) to create a GUI for simulations of a
reduced ODE system obtained with
[TikhonovFenichelReductions.jl](https://jo-ap.github.io/TikhonovFenichelReductions.jl/stable/).

For `reduction::Reduction` you can call 
```julia 
simgui(reduction)
```
to get a simple GUI that allows to investigate the behaviour of the reduced
system (together with the original one):
![GUI with an example](https://github.com/jo-ap/TFRSimulations/blob/main/example.png)
The example shows the reduction corresponding to the MacArthur-Rosenzweig model,
as demonstrated
[here](https://jo-ap.github.io/TikhonovFenichelReductions.jl/stable/gettingstarted/).
The approximate solution of the reduced system is shown with solid lines and the
one for the full system with dotted lines.
