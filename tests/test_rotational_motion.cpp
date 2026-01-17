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

TEST_CASE("Sign reversal of torque should reverse rotation", "[rotation]") {
    using scalar = double;
    using vec3 = tinyla::VariableMatrix<scalar, 3, 1>;
    using quat = tinyla::Quaternion<scalar>;

    vec3 initial_pos = vec3::filled(0.0);
    vec3 initial_vel = vec3::filled(0.0);
    scalar mass = 1.0;
    scalar dt = 0.001;

    Atom<scalar> atom(initial_pos, initial_vel, 1.0 / mass, 1.0);

    vec3 initial_euler = vec3::filled(0.0);
    initial_euler[2] = 0.1;
    atom.set_rotation_from_euler_angles(initial_euler);
    
    quat initial_rotation = atom.get_rotation();
    vec3 initial_omega = atom.get_angular_velocity_world_frame();

    // Apply positive torque for some time
    vec3 torque = vec3::filled(0.0);
    torque[2] = 2.0;

    int steps = 100;
    scalar t_total = dt * steps;
    
    for (int i = 0; i < steps; ++i) {
        atom.reset_force_and_torque();
        atom.apply_torque_in_world_frame(torque);
        atom.integrate_position_and_rotation(dt);
        atom.reset_force_and_torque();
        atom.apply_torque_in_world_frame(torque);
        atom.integrate_velocity_and_angular_velocity(dt);
    }

    // Check that we've rotated significantly
    quat mid_rotation = atom.get_rotation();
    vec3 mid_euler = atom.get_euler_angles();
    vec3 mid_omega = atom.get_angular_velocity_world_frame();
    
    REQUIRE(std::fabs(static_cast<double>(mid_euler[2]) - 0.1) > 0.01);

    // Now apply opposite torque for the same time period
    // This should bring angular velocity to zero and add more rotation
    torque[2] = -2.0;
    for (int i = 0; i < steps; ++i) {
        atom.reset_force_and_torque();
        atom.apply_torque_in_world_frame(torque);
        atom.integrate_position_and_rotation(dt);
        atom.reset_force_and_torque();
        atom.apply_torque_in_world_frame(torque);
        atom.integrate_velocity_and_angular_velocity(dt);
    }

    vec3 final_euler = atom.get_euler_angles();
    vec3 final_omega = atom.get_angular_velocity_world_frame();
    
    // Angular velocity should return to initial (zero) - within numerical precision
    REQUIRE(std::fabs(static_cast<double>(final_omega[2]) - static_cast<double>(initial_omega[2])) < 1e-10);
    
    // But rotation should NOT return to initial - it should be:
    // θ_final = θ_initial + 0.5·α·T² + 0.5·α·T² = θ_initial + α·T²
    // where α = I_inv * τ = 2.5 * 2.0 = 5.0
    scalar alpha = 2.5 * 2.0;
    scalar expected_total_rotation = 0.1 + alpha * t_total * t_total;
    REQUIRE(static_cast<double>(final_euler[2]) == Approx(expected_total_rotation).epsilon(1e-5));
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

TEST_CASE("Non-spherical object - verify specific rotation amount under constant torque", "[rotation][gyroscopic]") {
    using scalar = double;
    using vec3 = tinyla::VariableMatrix<scalar, 3, 1>;
    using mat3 = tinyla::VariableMatrix<scalar, 3, 3>;

    vec3 initial_pos = vec3::filled(0.0);
    vec3 initial_vel = vec3::filled(0.0);
    scalar mass = 1.0;
    scalar dt = 0.001;

    Atom<scalar> atom(initial_pos, initial_vel, 1.0 / mass, 1.0);

    // Create a rod-like inertia tensor aligned with z-axis
    // I_zz is small (easy to spin about z), I_xx = I_yy are large
    mat3 I_body = mat3::identity();
    I_body[0][0] = 10.0;
    I_body[1][1] = 10.0;
    I_body[2][2] = 1.0;
    
    mat3 I_inv_body = mat3::identity();
    I_inv_body[0][0] = 0.1;
    I_inv_body[1][1] = 0.1;
    I_inv_body[2][2] = 1.0;
    
    atom.set_inertia_tensor_body_frame(I_body);
    atom.set_inv_inertia_tensor_body_frame(I_inv_body);

    // Apply constant torque about z-axis
    vec3 torque = vec3::filled(0.0);
    torque[2] = 2.0;

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
    
    // For torque about z-axis: α_z = I_inv_zz * τ_z = 1.0 * 2.0 = 2.0
    scalar expected_alpha = 2.0;
    scalar expected_angle = 0.5 * expected_alpha * t_total * t_total;
    
    vec3 euler_angles = atom.get_euler_angles();
    
    // Verify the rotation about z-axis matches the expected value
    REQUIRE(static_cast<double>(euler_angles[2]) == Approx(expected_angle).epsilon(1e-6));
    
    // Also verify the angular velocity
    vec3 omega = atom.get_angular_velocity_world_frame();
    scalar expected_omega_z = expected_alpha * t_total;
    REQUIRE(static_cast<double>(omega[2]) == Approx(expected_omega_z).epsilon(1e-6));
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

TEST_CASE("Disc rotation - verify rotation angle matches expected value", "[rotation][gyroscopic]") {
    using scalar = double;
    using vec3 = tinyla::VariableMatrix<scalar, 3, 1>;
    using mat3 = tinyla::VariableMatrix<scalar, 3, 3>;

    vec3 initial_pos = vec3::filled(0.0);
    vec3 initial_vel = vec3::filled(0.0);
    scalar mass = 1.0;
    scalar dt = 0.001;

    Atom<scalar> atom(initial_pos, initial_vel, 1.0 / mass, 1.0);

    // Disc-like inertia tensor - easier to spin about z (disc's axis of symmetry)
    mat3 I_body = mat3::identity();
    I_body[0][0] = 8.0;
    I_body[1][1] = 8.0;
    I_body[2][2] = 4.0;
    
    mat3 I_inv_body = mat3::identity();
    I_inv_body[0][0] = 1.0 / 8.0;
    I_inv_body[1][1] = 1.0 / 8.0;
    I_inv_body[2][2] = 1.0 / 4.0;
    
    atom.set_inertia_tensor_body_frame(I_body);
    atom.set_inv_inertia_tensor_body_frame(I_inv_body);

    // Apply torque about x-axis
    vec3 torque = vec3::filled(0.0);
    torque[0] = 4.0;

    int steps = 50;
    for (int i = 0; i < steps; ++i) {
        atom.reset_force_and_torque();
        atom.apply_torque_in_world_frame(torque);
        atom.integrate_position_and_rotation(dt);
        atom.reset_force_and_torque();
        atom.apply_torque_in_world_frame(torque);
        atom.integrate_velocity_and_angular_velocity(dt);
    }

    scalar t_total = dt * steps;
    
    // For torque about x-axis: α_x = I_inv_xx * τ_x = 0.125 * 4.0 = 0.5
    scalar expected_alpha = 0.5;
    scalar expected_angle_x = 0.5 * expected_alpha * t_total * t_total;
    
    vec3 euler_angles = atom.get_euler_angles();
    
    // Verify the rotation about x-axis (roll) matches the expected value
    REQUIRE(std::fabs(static_cast<double>(euler_angles[0])) == Approx(expected_angle_x).epsilon(1e-5));
    
    // Also verify the angular velocity about x
    vec3 omega = atom.get_angular_velocity_world_frame();
    scalar expected_omega_x = expected_alpha * t_total;
    REQUIRE(std::fabs(static_cast<double>(omega[0])) == Approx(expected_omega_x).epsilon(1e-5));
}
