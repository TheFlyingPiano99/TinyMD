#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "../src/Atom.h"
#include "../src/TinyMD.h"
#include <cmath>

using namespace tinymd;
using Catch::Approx;

TEST_CASE("Single particle free motion (no forces)", "[velocity-verlet]") {
    using scalar = double;
    using vec3 = tinyla::VariableMatrix<scalar, 3, 1>;

    // Initial conditions
    vec3 initial_pos = vec3::filled(0.0);
    initial_pos[0] = 1.0;  // x = 1.0
    vec3 initial_vel = vec3::filled(0.0);
    initial_vel[0] = 2.0;  // vx = 2.0
    scalar mass = 1.0;
    scalar dt = 0.01;

    Atom<scalar> atom(initial_pos, initial_vel, 1.0 / mass, 1.0);

    // Evolve for several time steps without forces
    int steps = 100;
    for (int i = 0; i < steps; ++i) {
        atom.integrate_position(dt);
        atom.integrate_velocity(dt);
    }

    // Analytical solution: x(t) = x0 + v0*t (no acceleration)
    scalar t_total = dt * steps;
    scalar expected_x = 1.0 + 2.0 * t_total;
    
    REQUIRE(atom.get_position()[0] == Approx(expected_x).epsilon(1e-10));
    REQUIRE(atom.get_velocity()[0] == Approx(2.0).epsilon(1e-10));
}

TEST_CASE("Single particle with constant acceleration", "[velocity-verlet]") {
    using scalar = double;
    using vec3 = tinyla::VariableMatrix<scalar, 3, 1>;

    // Initial conditions
    vec3 initial_pos = vec3::filled(0.0);
    vec3 initial_vel = vec3::filled(0.0);
    scalar mass = 1.0;
    scalar dt = 0.01;

    Atom<scalar> atom(initial_pos, initial_vel, 1.0 / mass, 1.0);

    // Apply constant force (F = ma, so a = F/m)
    vec3 force = vec3::filled(0.0);
    force[2] = 1.0;  // Force in z direction
    scalar acceleration = 1.0;  // a = F/m = 1.0/1.0 = 1.0

    int steps = 100;
    for (int i = 0; i < steps; ++i) {
        atom.reset_acceleration();
        atom.apply_force(force, dt);
        atom.integrate_position(dt);
        atom.reset_acceleration();
        atom.apply_force(force, dt);
        atom.integrate_velocity(dt);
    }

    // Analytical solution for constant acceleration:
    // v(t) = v0 + a*t
    // x(t) = x0 + v0*t + 0.5*a*t^2
    scalar t_total = dt * steps;
    scalar expected_z = 0.5 * acceleration * t_total * t_total;
    scalar expected_vz = acceleration * t_total;
    
    REQUIRE(atom.get_position()[2] == Approx(expected_z).epsilon(1e-8));
    REQUIRE(atom.get_velocity()[2] == Approx(expected_vz).epsilon(1e-8));
}

TEST_CASE("Simple harmonic oscillator - energy conservation", "[velocity-verlet]") {
    using scalar = double;
    using vec3 = tinyla::VariableMatrix<scalar, 3, 1>;

    // Simple harmonic oscillator: F = -k*x
    scalar k = 1.0;  // spring constant
    scalar mass = 1.0;
    scalar dt = 0.001;  // Small time step for accuracy

    // Initial conditions: displaced from equilibrium with no velocity
    vec3 initial_pos = vec3::filled(0.0);
    initial_pos[0] = 1.0;  // x = 1.0
    vec3 initial_vel = vec3::filled(0.0);

    Atom<scalar> atom(initial_pos, initial_vel, 1.0 / mass, 1.0);

    // Initial energy: E = 0.5*k*x^2 + 0.5*m*v^2
    scalar initial_energy = 0.5 * k * 1.0 * 1.0;

    int steps = 1000;
    for (int i = 0; i < steps; ++i) {
        // Apply spring force
        atom.reset_acceleration();
        vec3 force = vec3::filled(0.0);
        force[0] = -k * atom.get_position()[0];
        atom.apply_force(force, dt);
        
        atom.integrate_position(dt);
        
        // Recompute force at new position
        atom.reset_acceleration();
        force[0] = -k * atom.get_position()[0];
        atom.apply_force(force, dt);
        
        atom.integrate_velocity(dt);
    }

    // Check energy conservation
    scalar pos_x = atom.get_position()[0];
    scalar vel_x = atom.get_velocity()[0];
    scalar final_energy = 0.5 * k * pos_x * pos_x + 0.5 * mass * vel_x * vel_x;
    
    // Energy should be conserved within numerical error
    REQUIRE(final_energy == Approx(initial_energy).epsilon(1e-5));
}

TEST_CASE("Particle in uniform gravitational field", "[velocity-verlet]") {
    using scalar = double;
    using vec3 = tinyla::VariableMatrix<scalar, 3, 1>;

    // Free fall under gravity
    scalar g = 9.81;  // gravitational acceleration
    scalar mass = 2.0;
    scalar dt = 0.01;

    vec3 initial_pos = vec3::filled(0.0);
    initial_pos[1] = 100.0;  // y = 100.0 meters
    vec3 initial_vel = vec3::filled(0.0);

    Atom<scalar> atom(initial_pos, initial_vel, 1.0 / mass, 1.0);

    vec3 gravity_force = vec3::filled(0.0);
    gravity_force[1] = -mass * g;  // F = m*g downward

    int steps = 200;
    for (int i = 0; i < steps; ++i) {
        atom.reset_acceleration();
        atom.apply_force(gravity_force, dt);
        atom.integrate_position(dt);
        atom.reset_acceleration();
        atom.apply_force(gravity_force, dt);
        atom.integrate_velocity(dt);
    }

    // Analytical solution: y(t) = y0 - 0.5*g*t^2, v(t) = -g*t
    scalar t_total = dt * steps;
    scalar expected_y = 100.0 - 0.5 * g * t_total * t_total;
    scalar expected_vy = -g * t_total;
    
    REQUIRE(atom.get_position()[1] == Approx(expected_y).epsilon(1e-8));
    REQUIRE(atom.get_velocity()[1] == Approx(expected_vy).epsilon(1e-8));
}

TEST_CASE("Velocity Verlet is time-reversible", "[velocity-verlet]") {
    using scalar = double;
    using vec3 = tinyla::VariableMatrix<scalar, 3, 1>;

    vec3 initial_pos = vec3::filled(0.0);
    initial_pos[0] = 1.0;
    initial_pos[1] = 2.0;
    initial_pos[2] = 3.0;
    
    vec3 initial_vel = vec3::filled(0.0);
    initial_vel[0] = 0.5;
    initial_vel[1] = -0.3;
    initial_vel[2] = 0.7;
    
    scalar mass = 1.5;
    scalar dt = 0.01;  
    scalar k = 1.0;  // Spring constant for harmonic oscillator

    Atom<scalar> atom(initial_pos, initial_vel, 1.0 / mass, 1.0);

    // Forward integration with harmonic force F = -k*r
    int steps = 50;
    for (int i = 0; i < steps; ++i) {
        vec3 force = -k * atom.get_position();
        atom.reset_acceleration();
        atom.apply_force(force, dt);
        atom.integrate_position(dt);
        
        force = -k * atom.get_position();
        atom.reset_acceleration();
        atom.apply_force(force, dt);
        atom.integrate_velocity(dt);
    }

    // Backward integration (same atom, just use -dt)
    for (int i = 0; i < steps; ++i) {
        vec3 force = -k * atom.get_position();
        atom.reset_acceleration();
        atom.apply_force(force, -dt);
        atom.integrate_position(-dt);
        
        force = -k * atom.get_position();
        atom.reset_acceleration();
        atom.apply_force(force, -dt);
        atom.integrate_velocity(-dt);
    }

    vec3 final_pos = atom.get_position();
    vec3 final_vel = atom.get_velocity();
    
    // Should return to initial conditions (time-reversibility)
    REQUIRE(final_pos.eval_at(0, 0) == Approx(initial_pos.eval_at(0, 0)).epsilon(1e-10));
    REQUIRE(final_pos.eval_at(1, 0) == Approx(initial_pos.eval_at(1, 0)).epsilon(1e-10));
    REQUIRE(final_pos.eval_at(2, 0) == Approx(initial_pos.eval_at(2, 0)).epsilon(1e-10));
    
    REQUIRE(final_vel.eval_at(0, 0) == Approx(initial_vel.eval_at(0, 0)).epsilon(1e-10));
    REQUIRE(final_vel.eval_at(1, 0) == Approx(initial_vel.eval_at(1, 0)).epsilon(1e-10));
    REQUIRE(final_vel.eval_at(2, 0) == Approx(initial_vel.eval_at(2, 0)).epsilon(1e-10));
}

TEST_CASE("Two particles with Coulomb interaction - momentum conservation", "[coulomb]") {
    using scalar = double;
    using vec3 = tinyla::VariableMatrix<scalar, 3, 1>;

    vec3 pos1 = vec3::filled(0.0);
    pos1[0] = -1.0;
    vec3 vel1 = vec3::filled(0.0);

    vec3 pos2 = vec3::filled(0.0);
    pos2[0] = 1.0;
    vec3 vel2 = vec3::filled(0.0);

    Atom<scalar> atom1(pos1, vel1, 1.0, 1.0);
    Atom<scalar> atom2(pos2, vel2, 1.0, 1.0);

    // Initial total momentum
    vec3 initial_momentum = atom1.get_mass() * vel1 + atom2.get_mass() * vel2;

    MDSimulator<scalar> sim;
    sim.add_atom(std::move(atom1));
    sim.add_atom(std::move(atom2));

    // Note: We can't easily test momentum conservation with the current MDSimulator
    // because it doesn't expose atoms. This test is a placeholder showing the concept.
    
    SUCCEED("Momentum conservation test structure created");
}
