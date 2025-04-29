from pde_solver_api.solvers.schrodinger_circle import solve_schrodinger_circle

def test_energy_levels():
    E, _, _, _ = solve_schrodinger_circle(num_points=30, num_eigenvalues=3)
    assert E[0] > 5 and E[0] < 6  # Примерный уровень энергии для круга радиуса 1
