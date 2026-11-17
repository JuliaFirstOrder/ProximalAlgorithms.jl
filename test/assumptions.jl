using ProximalAlgorithms: get_assumptions

@testset "get_assumptions function" begin
    @test length(get_assumptions(ProximalAlgorithms.CGIteration)) == 2
    @test length(get_assumptions(ProximalAlgorithms.ADMMIteration)) == 2
    @test length(get_assumptions(ProximalAlgorithms.DavisYinIteration)) == 3
    @test length(get_assumptions(ProximalAlgorithms.DouglasRachfordIteration)) == 2
    @test length(get_assumptions(ProximalAlgorithms.FastForwardBackwardIteration)) == 2
    @test length(get_assumptions(ProximalAlgorithms.FastProximalGradientIteration)) == 2
    @test length(get_assumptions(ProximalAlgorithms.ForwardBackwardIteration)) == 2
    @test length(get_assumptions(ProximalAlgorithms.ProximalGradientIteration)) == 2
    @test length(get_assumptions(ProximalAlgorithms.LiLinIteration)) == 2
    @test length(get_assumptions(ProximalAlgorithms.PANOCIteration)) == 2
    @test length(get_assumptions(ProximalAlgorithms.PANOCplusIteration)) == 2
    @test length(get_assumptions(ProximalAlgorithms.AFBAIteration)) == 3
    @test length(get_assumptions(ProximalAlgorithms.VuCondatIteration)) == 3
    @test length(get_assumptions(ProximalAlgorithms.ChambollePockIteration)) == 2
    @test length(get_assumptions(ProximalAlgorithms.SFISTAIteration)) == 2
    @test length(get_assumptions(ProximalAlgorithms.ZeroFPRIteration)) == 2
end