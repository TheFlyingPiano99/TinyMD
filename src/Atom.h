#pragma once
#define _USE_MATH_DEFINES
#include "TinyLA.h"
#include <cmath>

namespace tinymd {

    /*
        * Class: Atom
        * ------------
        * Represents an atom in the molecular dynamics simulation.
        * Each atom has position, velocity, mass, charge, and other properties.
        * The class provides methods to evolve the atom's state over time
        * and apply forces to it.
        * The values are in Hartree atomic units.
    */
    template<tinyla::RealType T>
    class Atom {
    using vec3 = tinyla::VariableMatrix<T, 3, 1>;
    using mat3 = tinyla::VariableMatrix<T, 3, 3>;
    using quat = tinyla::Quaternion<T>;
    using scalar = T;
        
    public:
        Atom(const vec3& position, const vec3& velocity, scalar reciprocal_mass, scalar effective_radius)
                : m_position(position),
                m_velocity(velocity),
                m_force(vec3::filled(static_cast<T>(0))), 
                m_reciprocal_mass(reciprocal_mass),
                m_rotation(quat{1, 0, 0, 0}),
                m_inertia_tensor_body_frame(mat3::identity() * (static_cast<T>(2.0) / static_cast<T>(5.0)) / m_reciprocal_mass),
                m_inv_inertia_tensor_body_frame(mat3::identity() * (static_cast<T>(5.0) / static_cast<T>(2.0)) * m_reciprocal_mass),
                m_angular_velocity_world_frame(vec3::filled(static_cast<T>(0))),
                m_torque_world_frame(vec3::filled(static_cast<T>(0))),
                m_charge(1.0), m_effective_radius(effective_radius), m_id(next_id++)
            {
                update_world_frame_inv_inertia_tensor();
            };

        /*
            * Evolves the atom's position and rotation using the Velocity Verlet integration method.
            *
            * delta_time: The time step for the evolution between step n and step n+1.
        */
        void integrate_position_and_rotation(scalar delta_time) {
            // Translational motion:
            {
                // Calculate acceleration at time n
                auto acceleration = m_force * m_reciprocal_mass;
                
                // Update velocity (half step): v_n+1/2 = v_n + 1/2 * a_n * dt
                m_velocity += static_cast<T>(0.5) * acceleration * delta_time;
                
                // Update position using half-step velocity: r_n+1 = r_n + v_n+1/2 * dt
                m_position += m_velocity * delta_time;
            }

            // Rotational motion:
            {
                // For gyroscopic term, we need: ω × (I * ω) where I is the inertia tensor
                // Angular momentum: L = I * ω
                vec3 angular_momentum = m_inertia_tensor_world_frame * m_angular_velocity_world_frame;
                
                // Gyroscopic term: ω × L
                vec3 gyroscopic_term = cross(m_angular_velocity_world_frame, angular_momentum);
                
                // Effective torque including gyroscopic correction: τ_eff = τ - ω × L
                vec3 effective_torque = m_torque_world_frame - gyroscopic_term;
                
                // Calculate angular acceleration: α = I^(-1) * τ_eff
                vec3 angular_acceleration_world_frame = m_inv_inertia_tensor_world_frame * effective_torque;
                
                // Update angular velocity (half step): ω_n+1/2 = ω_n + 1/2 * α_n * dt
                m_angular_velocity_world_frame += static_cast<T>(0.5) * angular_acceleration_world_frame * delta_time;
                
                // Update rotation using quaternion derivative: dq/dt = 1/2 * ω_quat * q
                // where ω_quat is the angular velocity as a pure quaternion (w=0, v=ω)
                auto angular_velocity_quat = quat{0, m_angular_velocity_world_frame.x(), m_angular_velocity_world_frame.y(), m_angular_velocity_world_frame.z()};
                auto q_dot = static_cast<T>(0.5) * angular_velocity_quat * m_rotation;
                
                // Integrate rotation: q_n+1 = q_n + dq/dt * dt
                m_rotation = m_rotation + q_dot * delta_time;
                
                // Normalize quaternion
                m_rotation = m_rotation / norm(m_rotation);
                update_world_frame_inv_inertia_tensor();
            }
        }

        /*
            * Evolves the atom's velocity using the Velocity Verlet integration method.
            *
            * delta_time: The time step for the evolution between step n and step n+1.
        */
        void integrate_velocity_and_angular_velocity(scalar delta_time) {
            // Translational motion:
            {
                auto acceleration = m_force * m_reciprocal_mass;
                m_velocity += static_cast<T>(0.5) * acceleration * delta_time;    // v_n+1 = v_n+1/2 + 1/2 * a_n+1 * dt
            }

            // Rotational motion:
            {
                // For gyroscopic term, we need: ω × (I * ω) where I is the inertia tensor
                // Angular momentum: L = I * ω
                vec3 angular_momentum = m_inertia_tensor_world_frame * m_angular_velocity_world_frame;
                
                // Gyroscopic term: ω × L
                vec3 gyroscopic_term = cross(m_angular_velocity_world_frame, angular_momentum);
                
                // Effective torque including gyroscopic correction: τ_eff = τ - ω × L
                vec3 effective_torque = m_torque_world_frame - gyroscopic_term;
                
                // Calculate angular acceleration: α = I^(-1) * τ_eff
                vec3 angular_acceleration_world_frame = m_inv_inertia_tensor_world_frame * effective_torque;
                
                // Complete velocity update: ω_n+1 = ω_n+1/2 + 1/2 * α_n+1 * dt
                m_angular_velocity_world_frame += static_cast<T>(0.5) * angular_acceleration_world_frame * delta_time;
            }
        }

        int get_status() const {
            return 0; // 0 indicates normal status
        }

        bool operator==(const Atom<T>& other) const {
            return m_id == other.m_id;
        }

        [[nodiscard]] uint32_t get_id() const {
            return m_id;
        }

        [[nodiscard]] vec3 get_position() const {
            return m_position;
        }

        [[nodiscard]] vec3 get_velocity() const {
            return m_velocity;
        }

        [[nodiscard]] vec3 get_acceleration() const {
            return m_force * m_reciprocal_mass;
        }

        [[nodiscard]] quat get_rotation() const {
            return m_rotation;
        }

        [[nodiscard]] vec3 get_angular_velocity() const {
            return m_angular_velocity_world_frame;
        }

        [[nodiscard]] vec3 get_angular_velocity_world_frame() const {
            return m_angular_velocity_world_frame;
        }

        [[nodiscard]] vec3 get_angular_momentum_world_frame() const {
            return m_inertia_tensor_world_frame * m_angular_velocity_world_frame;
        }

        [[nodiscard]] vec3 get_euler_angles() const {
            return vec3::euler_angles(m_rotation);
        }

        [[nodiscard]] scalar get_mass() const {
            return 1.0 / m_reciprocal_mass;
        }

        [[nodiscard]] scalar get_reciprocal_mass() const {
            return m_reciprocal_mass;
        }

        [[nodiscard]] scalar get_effective_radius() const {
            return m_effective_radius;
        }

        [[nodiscard]] scalar get_charge() const {
            return m_charge;
        }

        void reset_force_and_torque() {
            m_force = vec3{0, 0, 0};
            m_torque_world_frame = vec3{0, 0, 0};
        }

        void apply_force(const vec3& force) {
            m_force += force;
        }

        void apply_torque_in_world_frame(const vec3& torque_wf) {
            m_torque_world_frame += torque_wf;
        }

        /*
         * Sets the rotation from Euler angles (ZYX convention: roll-pitch-yaw).
         * 
         * euler_angles: vec3 containing (roll, pitch, yaw) in radians
         *               roll (x): rotation about X-axis
         *               pitch (y): rotation about Y-axis  
         *               yaw (z): rotation about Z-axis
         */
        void set_rotation_from_euler_angles(const vec3& euler_angles_xyz) {
            m_rotation = quat::rotation_from_euler_angles(euler_angles_xyz);
        }

        void set_angular_velocity_world_frame(const vec3& omega) {
            m_angular_velocity_world_frame = omega;
        }

        void set_inertia_tensor_body_frame(const mat3& I) {
            m_inertia_tensor_body_frame = I;
            update_world_frame_inv_inertia_tensor();
        }

        void set_inv_inertia_tensor_body_frame(const mat3& I_inv) {
            m_inv_inertia_tensor_body_frame = I_inv;
            update_world_frame_inv_inertia_tensor();
        }
        
    private:

        void update_world_frame_inv_inertia_tensor() {
            auto rotation_mat = mat3::rotation_matrix(m_rotation);
            m_inv_inertia_tensor_world_frame = rotation_mat * m_inv_inertia_tensor_body_frame * transpose(rotation_mat);
            m_inertia_tensor_world_frame = rotation_mat * m_inertia_tensor_body_frame * transpose(rotation_mat);
        }

        vec3 m_position;    // Bohr radius
        vec3 m_velocity;    // Bohr radius per atomic unit of time
        vec3 m_force; // Bohr radius per atomic unit of time squared
        scalar m_reciprocal_mass; // Reciprocal of mass in Hartree atomic units
        quat m_rotation;    // Orientation as a rotation quaternion
        vec3 m_angular_velocity_world_frame;    // Radians per atomic unit of time
        vec3 m_torque_world_frame;
        mat3 m_inertia_tensor_body_frame;
        mat3 m_inv_inertia_tensor_body_frame;
        mat3 m_inertia_tensor_world_frame;
        mat3 m_inv_inertia_tensor_world_frame;
        scalar m_charge;      // Elementary charge
        scalar m_effective_radius; // In Bohr radius
        uint32_t m_id;
        static inline uint32_t next_id = 0;
    };

    template Atom<double>;
}

