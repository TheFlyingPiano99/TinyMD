#pragma once
#include "Atom.h"
#include <vector>
#include <string>


namespace tinymd {

    /*
        * Class: MDSimulator
        * -------------------
        * A simple molecular dynamics simulator that manages a collection of atoms
        * and simulates their interactions over time.
        * The simulator uses the Hartree atomic units system.
    */
    template<tinyla::RealType T>
    class MDSimulator {
        using vec3 = tinyla::VariableMatrix<T, 3, 1>;
        using scalar = T;
    public:

        MDSimulator();

        /*
            * Runs the molecular dynamics simulation for the configured number of steps.
        */
        void run();

        void add_atom(Atom<T>&& atom) {
            m_atoms.push_back(std::forward<Atom<T>>(atom));
        }

        auto get_atoms() const {
            return m_atoms;
        }

        void set_time_step(scalar delta_time) {
            m_delta_time = delta_time;
        }

        void set_step_count(uint32_t step_count) {
            m_step_count = step_count;
        }

        void set_debug_print(bool enable) {
            m_print_debug = enable;
        }

        void print_atom_states() const;

        /*
            * Exports the current positions of all atoms to a CSV file.
            * The CSV format includes headers: atom_id, x, y, z
            * This is useful for visualization and analysis in external tools like Blender.
            * 
            * filename: Path to the output CSV file
            * throws: std::runtime_error if the file cannot be opened for writing
        */
        void export_csv(const std::string& filename, bool append = false) const;

        void set_export_path(const std::string& path) {
            m_export_path = path;
        }

    private:
        void interact_atoms(Atom<T>& atom1, Atom<T>& atom2);
        void coulomb_interaction(Atom<T>& atom1, Atom<T>& atom2);

        std::vector<Atom<scalar>> m_atoms;
        uint32_t m_step_count = 100;
        uint32_t m_current_step = 0;
        scalar m_delta_time = static_cast<scalar>(0.001);
        bool m_print_debug = true;
        std::string m_export_path;
    };

    template class MDSimulator<double>;

}
