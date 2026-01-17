#include "TinyMD.h"
#include <print>
#include <format>
#include <fstream>

namespace tinymd {

    auto to_string(const tinyla::dvec3& vec) {
        return std::format("({}, {}, {})", static_cast<double>(vec.x()), static_cast<double>(vec.y()), static_cast<double>(vec.z()));
    };

    template<typename T>
    auto to_string(const tinyla::Quaternion<T>& quat) {
        return std::format("({} + i{} + j{} + k{})", static_cast<double>(quat.real()), static_cast<double>(quat.i()), static_cast<double>(quat.j()), static_cast<double>(quat.k()));
    };

    template<tinyla::RealType T>
    MDSimulator<T>::MDSimulator() {
        // Constructor implementation (if any)
    }

    template<tinyla::RealType T>
    void MDSimulator<T>::run() {
        for (m_current_step = 0; m_current_step < m_step_count; ++m_current_step) {

            // Reset forces and torques before recalculating:
            for (auto& atom : m_atoms) {
                atom.reset_force_and_torque();
            }

            // Compute interaction forces:
            for (size_t i = 0; i < m_atoms.size(); ++i) {
                for (size_t j = i + 1; j < m_atoms.size(); ++j) {
                    if (i == j) continue;
                    interact_atoms(m_atoms[i], m_atoms[j]);
                }
            }

            // Evolve all atom positions using v_n and a_n to get r_n+1 and v_n+1/2:
            std::vector<Atom<T>*> to_remove;
            for (auto& atom : m_atoms) {
                atom.integrate_position_and_rotation(m_delta_time);
            }

            // Reset forces and torques before recalculating:
            for (auto& atom : m_atoms) {
                atom.reset_force_and_torque();
            }

            // Compute interaction forces again before velocity update from v_n+1/2 to v_n+1:
            for (size_t i = 0; i < m_atoms.size(); ++i) {
                for (size_t j = i + 1; j < m_atoms.size(); ++j) {
                    if (i == j) continue;
                    interact_atoms(m_atoms[i], m_atoms[j]);
                }
            }

            // Evolve all atom velocity from v_n+1/2 to v_n+1:
            for (auto& atom : m_atoms) {
                atom.integrate_velocity_and_angular_velocity(m_delta_time);
            }

            // Check for atoms to remove (e.g., if they went out of bounds)
            for (auto& atom : m_atoms) {
                auto status = atom.get_status();
                if (status != 0) { // Example condition
                    to_remove.push_back(&atom);
                }
            }

            // Remove atoms that need to be removed:
            for (const auto atom_ptr : to_remove) {
                auto it = std::find(m_atoms.begin(), m_atoms.end(), *atom_ptr);
                if (it != m_atoms.end()) {
                    m_atoms.erase(it);
                }
            }

            // Debug print:
            if (m_print_debug && (m_current_step % m_print_interval == 0)) {
                std::println("Step {}\nNo. of atoms = {}", m_current_step, m_atoms.size());
                print_atom_states();
            }
            bool is_export_csv = true;
            if (is_export_csv && !m_export_path.empty()) {
                export_csv(m_export_path, m_current_step > 0);
            }
        }
    }

    template<tinyla::RealType T>
    void MDSimulator<T>::interact_atoms(Atom<T>& atom1, Atom<T>& atom2) {
        coulomb_interaction(atom1, atom2);
    }

    template<tinyla::RealType T>
    void MDSimulator<T>::coulomb_interaction(Atom<T>& atom1, Atom<T>& atom2) {
        constexpr T k_e = static_cast<T>(1.0); // Coulomb constant in Hartree atomic units
        
        // Calculate distance vector from atom2 to atom1
        vec3 r_vec = atom1.get_position() - atom2.get_position();
        T r = norm(r_vec);
        T r_squared = r * r;

        // Avoid division by zero
        if (r < static_cast<T>(1e-10)) {
            return;
        }
        
        // Calculate Coulomb force magnitude: F = k * q1 * q2 / r²
        T q1 = atom1.get_charge();
        T q2 = atom2.get_charge();
        T force_magnitude = k_e * q1 * q2 / r_squared;
        
        // Force direction (unit vector)
        vec3 force_direction = r_vec / r;
        
        // Force vector
        vec3 force = force_direction * force_magnitude;
        
        // Apply forces (Newton's third law: equal and opposite)
        atom1.apply_force(force);
        atom2.apply_force(-force);

        // Test torque by explicitly applying:
        double sign = ((m_current_step / 500) % 2 == 0) ? 1.0 : -1.0;
        atom1.apply_torque_in_world_frame(sign * vec3{0.01, 0.01, 0.01});
        atom2.apply_torque_in_world_frame(sign * vec3{0.01, 0.01, 0.01});
    }

    template<tinyla::RealType T>
    void MDSimulator<T>::print_atom_states() const {
        for (const auto& atom : m_atoms) {
            std::println("Atom ID: {}, r = {}, r' = {}, q = {}, omega = {}", atom.get_id(), to_string(atom.get_position()), to_string(atom.get_velocity()), to_string(atom.get_rotation()), to_string(atom.get_angular_velocity_world_frame()));
        }
    }

    template<tinyla::RealType T>
    void MDSimulator<T>::export_csv(const std::string& filename, bool append) const {
        std::ofstream file(filename, append ? std::ios_base::app : std::ios_base::out);
        
        if (!file.is_open()) {
            throw std::runtime_error(std::format("Failed to open file for writing: {}", filename));
        }

        if (!append) {
            // Write header
            file << "position_x,position_y,position_z,velocity_x,velocity_y,velocity_z,rotation_r,rotation_i,rotation_j,rotation_k,angular_velocity_x,angular_velocity_y,angular_velocity_z,atom_id,simulation_step\n";
        }

        // Write atom positions
        for (const auto& atom : m_atoms) {
            auto pos = atom.get_position();
            auto vel = atom.get_velocity();
            auto rot = atom.get_rotation();
            auto ang_vel = atom.get_angular_velocity_world_frame();
            file << std::format("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n", 
                static_cast<double>(pos.eval_at(0, 0)),
                static_cast<double>(pos.eval_at(1, 0)),
                static_cast<double>(pos.eval_at(2, 0)),
                static_cast<double>(vel.eval_at(0, 0)),
                static_cast<double>(vel.eval_at(1, 0)),
                static_cast<double>(vel.eval_at(2, 0)),
                static_cast<double>(rot.eval_at(0, 0)),
                static_cast<double>(rot.eval_at(1, 0)),
                static_cast<double>(rot.eval_at(2, 0)),
                static_cast<double>(rot.eval_at(3, 0)),
                static_cast<double>(ang_vel.eval_at(0, 0)),
                static_cast<double>(ang_vel.eval_at(1, 0)),
                static_cast<double>(ang_vel.eval_at(2, 0)),
                atom.get_id(),
                m_current_step
            );
        }
        file.close();
    }

} // namespace tinymd