#pragma once
#include "TinyLA.h"

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
    using scalar = T;
        
    public:
        Atom(const vec3& position, const vec3& velocity, scalar reciprocal_mass, scalar effective_radius)
            : m_position(position), m_velocity(velocity), m_acceleration(vec3::filled(static_cast<T>(0))), 
              m_reciprocal_mass(reciprocal_mass), 
              m_charge(1.0), m_effective_radius(effective_radius), m_id(next_id++) {};

        /*
            * Evolves the atom's position using the Velocity Verlet integration method.
            *
            * delta_time: The time step for the evolution.
        */
        void integrate_position(scalar delta_time) {
            m_position += m_velocity * delta_time + static_cast<T>(0.5) * m_acceleration * delta_time * delta_time; // r_n+1 = r_n + v_n * dt + 0.5 * a_n * dt^2
            m_velocity += static_cast<T>(0.5) * m_acceleration * delta_time;    // v_n+1/2 = v_n + 1/2 * a_n * dt
        }

        /*
            * Evolves the atom's velocity using the Velocity Verlet integration method.
            *
            * delta_time: The time step for the evolution.
        */
        void integrate_velocity(scalar delta_time) {
            m_velocity += static_cast<T>(0.5) * m_acceleration * delta_time;    // v_n+1 = v_n+1/2 + 1/2 * a_n+1 * dt
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
            return m_acceleration;
        }

        [[nodiscard]] vec3 get_rotation() const {
            return m_rotation;
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

        void reset_acceleration() {
            m_acceleration = vec3::filled(static_cast<T>(0));
        }

        void apply_force(const vec3& force, scalar delta_time) {
            m_acceleration += force * m_reciprocal_mass;
        }
        
    private:
        vec3 m_position;    // Bohr radius
        vec3 m_velocity;    // Bohr radius per atomic unit of time
        vec3 m_acceleration; // Bohr radius per atomic unit of time squared
        vec3 m_rotation;    // Not used currently
        scalar m_reciprocal_mass; // Reciprocal of mass in Hartree atomic units
        scalar m_charge;      // Elementary charge
        scalar m_effective_radius; // In Bohr radius
        uint32_t m_id;
        static inline uint32_t next_id = 0;
    };

    template Atom<double>;
}

