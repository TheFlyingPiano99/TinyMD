#include "TinyMD.h"
#include <print>
#include <filesystem>

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
        tinyla::dvec3{5.0, 1.0, 1.0},
        tinyla::dvec3{0.0, 0.0, 0.0},
        1.0,
        5.0
    };
    simulator.add_atom(std::move(atom1));
    auto atom2 = tinymd::Atom<double>{
        tinyla::dvec3{10.0, 1.0, 1.0},
        tinyla::dvec3{0.0, 0.0, 0.0},
        1.0,
        5.0
    };
    simulator.add_atom(std::move(atom2));

    simulator.set_time_step(0.1);
    simulator.set_step_count(1000);

    // Export atom positions to CSV for visualization in Blender
    auto output_dir = std::filesystem::path("E:/coding/CppProjects/TinyMD/output");
    std::filesystem::create_directories(output_dir);
    auto csv_path = output_dir / "atom_positions.csv";
    simulator.set_export_path(csv_path.string());
    // Run simulation:
    simulator.run();

    return 0;
}