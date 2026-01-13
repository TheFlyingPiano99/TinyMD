#include "TinyMD.h"
#include <print>

int main(int argc, char** argv) {
    std::println("Launching TinyMD Simulator...");
    auto simulator = tinymd::MDSimulator<double>{};

    // Init atoms:
    auto atom0 = tinymd::Atom<double>{
        tinyla::dvec3{1.0, 1.0, 1.0},
        tinyla::dvec3{0.0, 0.0, 0.0},
        1.0,
        5.0
    };
    simulator.add_atom(std::move(atom0));
    auto atom1 = tinymd::Atom<double>{
        tinyla::dvec3{10.0, 1.0, 1.0},
        tinyla::dvec3{0.0, 0.0, 0.0},
        1.0,
        5.0
    };
    simulator.add_atom(std::move(atom1));
    auto atom2 = tinymd::Atom<double>{
        tinyla::dvec3{20.0, 1.0, 1.0},
        tinyla::dvec3{0.0, 0.0, 0.0},
        1.0,
        5.0
    };
    simulator.add_atom(std::move(atom2));

    // Run simulation:
    simulator.run();

    return 0;
}