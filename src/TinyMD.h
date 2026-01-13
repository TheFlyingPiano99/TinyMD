#pragma once
#include "Atom.h"
#include <vector>

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

        void run();

        void add_atom(Atom<T>&& atom) {
            m_atoms.push_back(std::forward<Atom<T>>(atom));
        }

    private:
        void interact_atoms(Atom<T>& atom1, Atom<T>& atom2);

        std::vector<Atom<scalar>> m_atoms;
        uint32_t m_step_count = 100;
        scalar m_delta_time = static_cast<scalar>(0.001);
        bool m_print_debug = true;
    };

    template class MDSimulator<double>;

}
