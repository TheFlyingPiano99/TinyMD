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
                m_rotation_body_to_world(quat{1, 0, 0, 0}),
                m_inertia_tensor_body_frame(mat3::identity() * (static_cast<T>(2.0) / static_cast<T>(5.0)) / m_reciprocal_mass),
                m_inv_inertia_tensor_body_frame(mat3::identity() * (static_cast<T>(5.0) / static_cast<T>(2.0)) * m_reciprocal_mass),
                m_angular_velocity_body_frame(vec3::filled(static_cast<T>(0))),
                m_torque_world_frame(vec3::filled(static_cast<T>(0))),
                m_charge(1.0), m_effective_radius(effective_radius), m_id(next_id++)
            {
            };

        /*
            * Evolves the atom's position and rotation using the Velocity Verlet integration method.
            *
            * delta_time: The time step for the evolution between step n and step n+1.
        */
        void integrate_position_and_rotation(scalar delta_time) {
            // Translational motion:
            {
                auto acceleration = m_force * m_reciprocal_mass;
                m_velocity += static_cast<T>(0.5) * acceleration * delta_time;
                clamp_velocity();
                m_position += m_velocity * delta_time;
            }

            // Rotational motion:
            {
                vec3 torque_body_frame = rotate_vector_by_quaternion(m_torque_world_frame, conjugate(m_rotation_body_to_world));
                vec3 angular_momentum_body_frame = m_inertia_tensor_body_frame * m_angular_velocity_body_frame;
                vec3 gyroscopic_torque_body_frame = cross(m_angular_velocity_body_frame, angular_momentum_body_frame);
                vec3 angular_acceleration_body_frame = m_inv_inertia_tensor_body_frame * (torque_body_frame - gyroscopic_torque_body_frame);
                m_angular_velocity_body_frame += static_cast<T>(0.5) * angular_acceleration_body_frame * delta_time;
                clamp_angular_velocity();
                T ang_velocity_norm = norm(m_angular_velocity_body_frame);
                constexpr T epsilon = static_cast<T>(1e-8);
                if (ang_velocity_norm > epsilon) {  // Normal case
                    m_rotation_body_to_world = m_rotation_body_to_world * quat{
                        std::cos(ang_velocity_norm * delta_time * static_cast<T>(0.5)),
                        (m_angular_velocity_body_frame.x() / ang_velocity_norm) * std::sin(ang_velocity_norm * delta_time * static_cast<T>(0.5)),
                        (m_angular_velocity_body_frame.y() / ang_velocity_norm) * std::sin(ang_velocity_norm * delta_time * static_cast<T>(0.5)),
                        (m_angular_velocity_body_frame.z() / ang_velocity_norm) * std::sin(ang_velocity_norm * delta_time * static_cast<T>(0.5))
                    };
                }
                else {  // For very small angular velocities, approximate the rotation
                    m_rotation_body_to_world = m_rotation_body_to_world * quat{
                        static_cast<T>(1),
                        m_angular_velocity_body_frame.x() * delta_time * static_cast<T>(0.5),
                        m_angular_velocity_body_frame.y() * delta_time * static_cast<T>(0.5),
                        m_angular_velocity_body_frame.z() * delta_time * static_cast<T>(0.5)                    
                    };
                }
                m_rotation_body_to_world = m_rotation_body_to_world / norm(m_rotation_body_to_world);   // Normalize quaternion
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
                clamp_velocity();
            }

            // Rotational motion:
            {
                vec3 torque_body_frame = rotate_vector_by_quaternion(m_torque_world_frame, conjugate(m_rotation_body_to_world));
                vec3 angular_momentum_body_frame = m_inertia_tensor_body_frame * m_angular_velocity_body_frame;
                vec3 gyroscopic_torque_body_frame = cross(m_angular_velocity_body_frame, angular_momentum_body_frame);
                vec3 angular_acceleration_body_frame = m_inv_inertia_tensor_body_frame * (torque_body_frame - gyroscopic_torque_body_frame);
                m_angular_velocity_body_frame += static_cast<T>(0.5) * angular_acceleration_body_frame * delta_time;
                clamp_angular_velocity();
            }
        }

        inline void clamp_velocity() {
            float n = norm(m_velocity);
            if (n > max_velocity) {
                m_velocity = (m_velocity / n) * max_velocity;
            }
        }

        inline void clamp_angular_velocity() {
            float ang_n = norm(m_angular_velocity_body_frame);
            if (ang_n > max_angular_velocity) {
                m_angular_velocity_body_frame = (m_angular_velocity_body_frame / ang_n) * max_angular_velocity;
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
            return m_rotation_body_to_world;
        }

        [[nodiscard]] vec3 get_angular_velocity_body_frame() const {
            return m_angular_velocity_body_frame;
        }

        [[nodiscard]] vec3 get_angular_velocity_world_frame() const {
            return m_angular_velocity_body_frame;
        }

        [[nodiscard]] vec3 get_euler_angles() const {
            return vec3::euler_angles(m_rotation_body_to_world);
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
            m_rotation_body_to_world = quat::rotation_from_euler_angles(euler_angles_xyz);
        }

        void set_angular_velocity_world_frame(const vec3& omega) {
            m_angular_velocity_body_frame = omega;
        }

        void set_inertia_tensor_body_frame(const mat3& I) {
            m_inertia_tensor_body_frame = I;
        }

        void set_inv_inertia_tensor_body_frame(const mat3& I_inv) {
            m_inv_inertia_tensor_body_frame = I_inv;
        }
        
    private:

        vec3 m_position;    // Bohr radius
        vec3 m_velocity;    // Bohr radius per atomic unit of time
        vec3 m_force; // Bohr radius per atomic unit of time squared
        scalar m_reciprocal_mass; // Reciprocal of mass in Hartree atomic units
        quat m_rotation_body_to_world;    // Orientation as a rotation quaternion
        vec3 m_angular_velocity_body_frame;    // Radians per atomic unit of time
        vec3 m_torque_world_frame;
        mat3 m_inertia_tensor_body_frame;
        mat3 m_inv_inertia_tensor_body_frame;
        scalar m_charge;      // Elementary charge
        scalar m_effective_radius; // In Bohr radius
        uint32_t m_id;
        static inline uint32_t next_id = 0;
        static constexpr auto max_velocity = 1000.0;
        static constexpr auto max_angular_velocity = 1000.0;
    };

    template Atom<double>;
}

