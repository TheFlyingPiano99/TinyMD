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

    /*
    for (int i = 0; i < 10000; ++i) {
        simulator.add_atom(tinymd::Atom<double>{
            tinyla::dvec3{15.0 + i * 4.0, 1.0, 1.0},
            tinyla::dvec3{0.0, 0.0, 0.0},
            1.0,
            5.0
        });
    }
    */
    

    simulator.set_time_step(0.1);
    simulator.set_step_count(10000);

    // Export atom positions to CSV for visualization in Blender
    auto output_dir = std::filesystem::path("E:/coding/CppProjects/TinyMD/output");
    std::filesystem::create_directories(output_dir);
    auto csv_path = output_dir / "simulation_data.csv";
    simulator.set_export_path(csv_path.string());
    simulator.set_export_csv(false);

    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

    // Run simulation:
    simulator.run();

    std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::println("Simulation completed in {} s", duration / 1000.0);

    return 0;
}