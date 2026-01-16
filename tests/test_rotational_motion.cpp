#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "../src/Atom.h"
#define _USE_MATH_DEFINES
#include <cmath>

using namespace tinymd;
using Catch::Approx;

TEST_CASE("Rotation with constant torque produces expected angular displacement", "[rotation]") {
    using scalar = double;
    using vec3 = tinyla::VariableMatrix<scalar, 3, 1>;

    vec3 initial_pos = vec3::filled(0.0);
    vec3 initial_vel = vec3::filled(0.0);
    scalar mass = 1.0;
    scalar dt = 0.001;

    Atom<scalar> atom(initial_pos, initial_vel, 1.0 / mass, 1.0);

    // Apply constant torque about z-axis
    vec3 torque = vec3::filled(0.0);
    torque[2] = 1.0;

    // For a sphere: I_inv = (5/2) * (1/m) = 2.5
    // Angular acceleration: α = I_inv * τ = 2.5
    scalar expected_alpha = 2.5;

    int steps = 100;
    for (int i = 0; i < steps; ++i) {
        atom.reset_force_and_torque();
        atom.apply_torque_in_world_frame(torque);
        atom.integrate_position_and_rotation(dt);
        atom.reset_force_and_torque();
        atom.apply_torque_in_world_frame(torque);
        atom.integrate_velocity_and_angular_velocity(dt);
    }

    scalar t_total = dt * steps;
    scalar expected_angle = 0.5 * expected_alpha * t_total * t_total;
    
    vec3 euler_angles = atom.get_euler_angles();
    
    REQUIRE(static_cast<double>(euler_angles[2]) == Approx(expected_angle).epsilon(1e-6));
}

TEST_CASE("Quaternion remains normalized during rotation", "[rotation]") {
    using scalar = double;
    using vec3 = tinyla::VariableMatrix<scalar, 3, 1>;

    vec3 initial_pos = vec3::filled(0.0);
    vec3 initial_vel = vec3::filled(0.0);
    scalar mass = 1.0;
    scalar dt = 0.001;

    Atom<scalar> atom(initial_pos, initial_vel, 1.0 / mass, 1.0);

    vec3 torque = vec3::filled(0.0);
    torque[0] = 1.0;
    torque[1] = 0.5;
    torque[2] = 0.7;

    for (int i = 0; i < 500; ++i) {
        atom.reset_force_and_torque();
        atom.apply_torque_in_world_frame(torque);
        atom.integrate_position_and_rotation(dt);
        atom.reset_force_and_torque();
        atom.apply_torque_in_world_frame(torque);
        atom.integrate_velocity_and_angular_velocity(dt);
        
        auto q = atom.get_rotation();
        REQUIRE(static_cast<double>(norm(q)) == Approx(1.0).epsilon(1e-7));
    }
}

TEST_CASE("Time reversibility of rotational integration", "[rotation]") {
    using scalar = double;
    using vec3 = tinyla::VariableMatrix<scalar, 3, 1>;
    using quat = tinyla::Quaternion<scalar>;

    vec3 initial_pos = vec3::filled(0.0);
    vec3 initial_vel = vec3::filled(0.0);
    scalar mass = 1.0;
    scalar dt = 0.001;

    Atom<scalar> atom(initial_pos, initial_vel, 1.0 / mass, 1.0);

    vec3 initial_euler = vec3::filled(0.0);
    initial_euler[0] = 0.1;
    initial_euler[1] = 0.2;
    initial_euler[2] = 0.3;
    atom.set_rotation_from_euler_angles(initial_euler);
    
    quat initial_rotation = atom.get_rotation();

    vec3 torque = vec3::filled(0.0);
    torque[0] = 0.5;
    torque[1] = 0.3;
    torque[2] = 0.7;

    int steps = 50;
    for (int i = 0; i < steps; ++i) {
        atom.reset_force_and_torque();
        atom.apply_torque_in_world_frame(torque);
        atom.integrate_position_and_rotation(dt);
        atom.reset_force_and_torque();
        atom.apply_torque_in_world_frame(torque);
        atom.integrate_velocity_and_angular_velocity(dt);
    }

    for (int i = 0; i < steps; ++i) {
        atom.reset_force_and_torque();
        atom.apply_torque_in_world_frame(torque);
        atom.integrate_position_and_rotation(-dt);
        atom.reset_force_and_torque();
        atom.apply_torque_in_world_frame(torque);
        atom.integrate_velocity_and_angular_velocity(-dt);
    }

    quat final_rotation = atom.get_rotation();
    
    REQUIRE(static_cast<double>(final_rotation.w()) == Approx(static_cast<double>(initial_rotation.w())).epsilon(1e-6));
    REQUIRE(static_cast<double>(final_rotation.x()) == Approx(static_cast<double>(initial_rotation.x())).epsilon(1e-6));
    REQUIRE(static_cast<double>(final_rotation.y()) == Approx(static_cast<double>(initial_rotation.y())).epsilon(1e-6));
    REQUIRE(static_cast<double>(final_rotation.z()) == Approx(static_cast<double>(initial_rotation.z())).epsilon(1e-6));
}

TEST_CASE("Free rotation without torque", "[rotation]") {
    using scalar = double;
    using vec3 = tinyla::VariableMatrix<scalar, 3, 1>;
    using quat = tinyla::Quaternion<scalar>;

    vec3 initial_pos = vec3::filled(0.0);
    vec3 initial_vel = vec3::filled(0.0);
    scalar mass = 1.0;
    scalar dt = 0.001;

    Atom<scalar> atom(initial_pos, initial_vel, 1.0 / mass, 1.0);

    // Apply torque initially to get it spinning
    vec3 torque = vec3::filled(0.0);
    torque[2] = 50.0;
    
    for (int i = 0; i < 10; ++i) {
        atom.reset_force_and_torque();
        atom.apply_torque_in_world_frame(torque);
        atom.integrate_position_and_rotation(dt);
        atom.reset_force_and_torque();
        atom.apply_torque_in_world_frame(torque);
        atom.integrate_velocity_and_angular_velocity(dt);
    }
    
    atom.reset_force_and_torque();
    quat q_initial = atom.get_rotation();
    
    // Now let it spin freely
    for (int i = 0; i < 100; ++i) {
        atom.integrate_position_and_rotation(dt);
        atom.integrate_velocity_and_angular_velocity(dt);
    }

    quat q_final = atom.get_rotation();
    
    // Should have rotated
    scalar quaternion_diff = static_cast<double>(norm(q_final - q_initial));
    REQUIRE(quaternion_diff > 0.01);
    
    // Quaternion should still be normalized
    REQUIRE(static_cast<double>(norm(q_final)) == Approx(1.0).epsilon(1e-8));
}

TEST_CASE("Combined translation and rotation", "[rotation]") {
    using scalar = double;
    using vec3 = tinyla::VariableMatrix<scalar, 3, 1>;

    vec3 initial_pos = vec3::filled(0.0);
    vec3 initial_vel = vec3::filled(0.0);
    initial_vel[0] = 1.0;
    scalar mass = 1.0;
    scalar dt = 0.01;

    Atom<scalar> atom(initial_pos, initial_vel, 1.0 / mass, 1.0);

    vec3 force = vec3::filled(0.0);
    force[1] = 0.5;
    
    vec3 torque = vec3::filled(0.0);
    torque[2] = 1.0;

    for (int i = 0; i < 100; ++i) {
        atom.reset_force_and_torque();
        atom.apply_force(force);
        atom.apply_torque_in_world_frame(torque);
        atom.integrate_position_and_rotation(dt);
        atom.reset_force_and_torque();
        atom.apply_force(force);
        atom.apply_torque_in_world_frame(torque);
        atom.integrate_velocity_and_angular_velocity(dt);
    }

    vec3 final_pos = atom.get_position();
    vec3 euler_angles = atom.get_euler_angles();
    
    // Both translation and rotation should have occurred
    REQUIRE(std::fabs(static_cast<double>(final_pos[0])) > 0.1);
    REQUIRE(std::fabs(static_cast<double>(final_pos[1])) > 0.01);
    REQUIRE(std::fabs(static_cast<double>(euler_angles[2])) > 0.1);
}

TEST_CASE("Rod with gyroscopic effects - torque-free precession", "[rotation][gyroscopic]") {
    using scalar = double;
    using vec3 = tinyla::VariableMatrix<scalar, 3, 1>;
    using mat3 = tinyla::VariableMatrix<scalar, 3, 3>;

    vec3 initial_pos = vec3::filled(0.0);
    vec3 initial_vel = vec3::filled(0.0);
    scalar mass = 1.0;
    scalar dt = 0.0001;

    Atom<scalar> atom(initial_pos, initial_vel, 1.0 / mass, 1.0);

    // Create a rod-like inertia tensor (moment of inertia larger about perpendicular axes)
    // For a rod along z-axis: I_xx = I_yy >> I_zz
    mat3 I_body = mat3::identity();
    I_body[0][0] = 5.0;  // Large moment about x
    I_body[1][1] = 5.0;  // Large moment about y
    I_body[2][2] = 0.1;  // Small moment about z (rod axis)
    
    mat3 I_inv_body = mat3::identity();
    I_inv_body[0][0] = 1.0 / 5.0;
    I_inv_body[1][1] = 1.0 / 5.0;
    I_inv_body[2][2] = 1.0 / 0.1;
    
    atom.set_inertia_tensor_body_frame(I_body);
    atom.set_inv_inertia_tensor_body_frame(I_inv_body);

    // Initial angular velocity: spinning fast about z-axis, slight tilt in x direction
    vec3 omega_initial = vec3::filled(0.0);
    omega_initial[0] = 0.5;   // Small component in x
    omega_initial[2] = 10.0;  // Large component in z (spinning)
    
    atom.set_angular_velocity_world_frame(omega_initial);

    // Store initial angular momentum (should be conserved with no external torque)
    vec3 L_initial = atom.get_angular_momentum_world_frame();
    scalar L_mag_initial = static_cast<double>(tinyla::norm(L_initial));

    // Evolve with no external torque (gyroscopic effects only)
    int steps = 1000;
    for (int i = 0; i < steps; ++i) {
        atom.reset_force_and_torque();
        atom.integrate_position_and_rotation(dt);
        atom.reset_force_and_torque();
        atom.integrate_velocity_and_angular_velocity(dt);
    }

    // Angular momentum magnitude should be conserved (no external torque)
    // Relaxed tolerance due to numerical integration over many steps
    vec3 L_final = atom.get_angular_momentum_world_frame();
    scalar L_mag_final = static_cast<double>(tinyla::norm(L_final));
    
    REQUIRE(L_mag_final == Approx(L_mag_initial).epsilon(1e-3));
    
    // The angular velocity vector should have precessed (direction changed)
    vec3 omega_final = atom.get_angular_velocity_world_frame();
    
    // The x-component should have changed due to gyroscopic precession
    REQUIRE(std::fabs(static_cast<double>(omega_final[0]) - static_cast<double>(omega_initial[0])) > 0.01);
}

TEST_CASE("Asymmetric object - verify rotation angle with constant torque", "[rotation][gyroscopic]") {
    using scalar = double;
    using vec3 = tinyla::VariableMatrix<scalar, 3, 1>;
    using mat3 = tinyla::VariableMatrix<scalar, 3, 3>;

    vec3 initial_pos = vec3::filled(0.0);
    vec3 initial_vel = vec3::filled(0.0);
    scalar mass = 1.0;
    scalar dt = 0.001;

    Atom<scalar> atom(initial_pos, initial_vel, 1.0 / mass, 1.0);

    // Create an ellipsoid-like inertia tensor with different principal moments
    mat3 I_body = mat3::identity();
    I_body[0][0] = 2.0;
    I_body[1][1] = 3.0;
    I_body[2][2] = 4.0;
    
    mat3 I_inv_body = mat3::identity();
    I_inv_body[0][0] = 1.0 / 2.0;
    I_inv_body[1][1] = 1.0 / 3.0;
    I_inv_body[2][2] = 1.0 / 4.0;
    
    atom.set_inertia_tensor_body_frame(I_body);
    atom.set_inv_inertia_tensor_body_frame(I_inv_body);

    // Apply torque about z-axis only
    vec3 torque = vec3::filled(0.0);
    torque[2] = 1.0;

    int steps = 100;
    for (int i = 0; i < steps; ++i) {
        atom.reset_force_and_torque();
        atom.apply_torque_in_world_frame(torque);
        atom.integrate_position_and_rotation(dt);
        atom.reset_force_and_torque();
        atom.apply_torque_in_world_frame(torque);
        atom.integrate_velocity_and_angular_velocity(dt);
    }

    scalar t_total = dt * steps;
    
    // For torque about z-axis: α_z = I_inv_zz * τ_z = 0.25 * 1.0 = 0.25
    scalar expected_alpha_z = 0.25;
    scalar expected_angle = 0.5 * expected_alpha_z * t_total * t_total;
    
    vec3 euler_angles = atom.get_euler_angles();
    
    // The rotation about z should match the expected value
    REQUIRE(static_cast<double>(euler_angles[2]) == Approx(expected_angle).epsilon(1e-6));
}

TEST_CASE("Gyroscopic precession - spinning disc with applied torque", "[rotation][gyroscopic]") {
    using scalar = double;
    using vec3 = tinyla::VariableMatrix<scalar, 3, 1>;
    using mat3 = tinyla::VariableMatrix<scalar, 3, 3>;

    vec3 initial_pos = vec3::filled(0.0);
    vec3 initial_vel = vec3::filled(0.0);
    scalar mass = 1.0;
    scalar dt = 0.0001;

    Atom<scalar> atom(initial_pos, initial_vel, 1.0 / mass, 1.0);

    // Disc-like inertia tensor (spinning about z, I_zz smaller than I_xx, I_yy)
    mat3 I_body = mat3::identity();
    I_body[0][0] = 4.0;
    I_body[1][1] = 4.0;
    I_body[2][2] = 2.0;
    
    mat3 I_inv_body = mat3::identity();
    I_inv_body[0][0] = 1.0 / 4.0;
    I_inv_body[1][1] = 1.0 / 4.0;
    I_inv_body[2][2] = 1.0 / 2.0;
    
    atom.set_inertia_tensor_body_frame(I_body);
    atom.set_inv_inertia_tensor_body_frame(I_inv_body);

    // Fast spin about z-axis
    vec3 omega_initial = vec3::filled(0.0);
    omega_initial[2] = 50.0;
    atom.set_angular_velocity_world_frame(omega_initial);

    // Record initial orientation
    vec3 euler_initial = atom.get_euler_angles();

    // Apply torque perpendicular to spin axis (about x-axis)
    vec3 torque = vec3::filled(0.0);
    torque[0] = 2.0;  // Increased torque for more visible effect

    int steps = 1000;
    for (int i = 0; i < steps; ++i) {
        atom.reset_force_and_torque();
        atom.apply_torque_in_world_frame(torque);
        atom.integrate_position_and_rotation(dt);
        atom.reset_force_and_torque();
        atom.apply_torque_in_world_frame(torque);
        atom.integrate_velocity_and_angular_velocity(dt);
    }

    // Angular velocity should have changed due to torque and gyroscopic effects
    vec3 omega_final = atom.get_angular_velocity_world_frame();
    
    // The x-component should have increased due to applied torque
    REQUIRE(static_cast<double>(omega_final[0]) > 0.5);
    
    // The z-component should still be present (spinning continues)
    REQUIRE(std::fabs(static_cast<double>(omega_final[2])) > 40.0);
    
    // Verify significant rotation has occurred
    vec3 euler_final = atom.get_euler_angles();
    scalar total_rotation = std::sqrt(
        static_cast<double>(euler_final[0] * euler_final[0] + 
        euler_final[1] * euler_final[1] + 
        euler_final[2] * euler_final[2])
    );
    REQUIRE(total_rotation > 1.0);
}
