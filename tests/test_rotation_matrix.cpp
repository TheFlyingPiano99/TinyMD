#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "../src/Atom.h"
#define _USE_MATH_DEFINES
#include <cmath>

using namespace tinymd;
using Catch::Approx;

TEST_CASE("Identity rotation matrix to Euler angles", "[rotation]") {
    using scalar = double;
    using vec3 = tinyla::VariableMatrix<scalar, 3, 1>;
    using mat3 = tinyla::VariableMatrix<scalar, 3, 3>;

    // Create identity rotation matrix
    mat3 identity = mat3::identity();
    
    vec3 euler_angles = Atom<scalar>::extract_euler_angles_from_rotation_matrix(identity);
    
    // Identity matrix should give zero Euler angles
    REQUIRE(static_cast<double>(euler_angles[0]) == Approx(0.0).margin(1e-10));  // roll
    REQUIRE(static_cast<double>(euler_angles[1]) == Approx(0.0).margin(1e-10));  // pitch
    REQUIRE(static_cast<double>(euler_angles[2]) == Approx(0.0).margin(1e-10));  // yaw
}

TEST_CASE("Zero Euler angles to rotation matrix", "[rotation]") {
    using scalar = double;
    using vec3 = tinyla::VariableMatrix<scalar, 3, 1>;
    using mat3 = tinyla::VariableMatrix<scalar, 3, 3>;

    vec3 zero_angles = vec3::filled(0.0);
    mat3 rotation = Atom<scalar>::construct_rotation_matrix_from_euler_angles(zero_angles);
    
    // Should produce identity matrix
    mat3 identity = mat3::identity();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            REQUIRE(rotation(i, j) == Approx(identity(i, j)).margin(1e-10));
        }
    }
}

TEST_CASE("Rotation about X-axis (roll only)", "[rotation]") {
    using scalar = double;
    using vec3 = tinyla::VariableMatrix<scalar, 3, 1>;
    using mat3 = tinyla::VariableMatrix<scalar, 3, 3>;

    scalar angle = M_PI / 4.0;  // 45 degrees
    vec3 euler_angles = vec3::filled(0.0);
    euler_angles[0] = angle;  // roll
    
    mat3 rotation = Atom<scalar>::construct_rotation_matrix_from_euler_angles(euler_angles);
    
    // Rotation about X-axis:
    // [1    0         0    ]
    // [0  cos(θ)  -sin(θ) ]
    // [0  sin(θ)   cos(θ) ]
    REQUIRE(static_cast<double>(rotation.at(0, 0)) == Approx(1.0).margin(1e-10));
    REQUIRE(static_cast<double>(rotation.at(0, 1)) == Approx(0.0).margin(1e-10));
    REQUIRE(static_cast<double>(rotation.at(0, 2)) == Approx(0.0).margin(1e-10));
    REQUIRE(static_cast<double>(rotation.at(1, 0)) == Approx(0.0).margin(1e-10));
    REQUIRE(static_cast<double>(rotation.at(1, 1)) == Approx(std::cos(angle)).margin(1e-10));
    REQUIRE(static_cast<double>(rotation.at(1, 2)) == Approx(-std::sin(angle)).margin(1e-10));
    REQUIRE(static_cast<double>(rotation.at(2, 0)) == Approx(0.0).margin(1e-10));
    REQUIRE(static_cast<double>(rotation.at(2, 1)) == Approx(std::sin(angle)).margin(1e-10));
    REQUIRE(static_cast<double>(rotation.at(2, 2)) == Approx(std::cos(angle)).margin(1e-10));
}

TEST_CASE("Rotation about Y-axis (pitch only)", "[rotation]") {
    using scalar = double;
    using vec3 = tinyla::VariableMatrix<scalar, 3, 1>;
    using mat3 = tinyla::VariableMatrix<scalar, 3, 3>;

    scalar angle = M_PI / 3.0;  // 60 degrees
    vec3 euler_angles = vec3::filled(0.0);
    euler_angles[1] = angle;  // pitch
    
    mat3 rotation = Atom<scalar>::construct_rotation_matrix_from_euler_angles(euler_angles);
    
    // Rotation about Y-axis:
    // [ cos(θ)  0  sin(θ) ]
    // [   0     1    0    ]
    // [-sin(θ)  0  cos(θ) ]
    REQUIRE(static_cast<double>(rotation.at(0, 0)) == Approx(std::cos(angle)).margin(1e-10));
    REQUIRE(static_cast<double>(rotation.at(0, 1)) == Approx(0.0).margin(1e-10));
    REQUIRE(static_cast<double>(rotation.at(0, 2)) == Approx(std::sin(angle)).margin(1e-10));
    REQUIRE(static_cast<double>(rotation.at(1, 0)) == Approx(0.0).margin(1e-10));
    REQUIRE(static_cast<double>(rotation.at(1, 1)) == Approx(1.0).margin(1e-10));
    REQUIRE(static_cast<double>(rotation.at(1, 2)) == Approx(0.0).margin(1e-10));
    REQUIRE(static_cast<double>(rotation.at(2, 0)) == Approx(-std::sin(angle)).margin(1e-10));
    REQUIRE(static_cast<double>(rotation.at(2, 1)) == Approx(0.0).margin(1e-10));
    REQUIRE(static_cast<double>(rotation.at(2, 2)) == Approx(std::cos(angle)).margin(1e-10));
}

TEST_CASE("Rotation about Z-axis (yaw only)", "[rotation]") {
    using scalar = double;
    using vec3 = tinyla::VariableMatrix<scalar, 3, 1>;
    using mat3 = tinyla::VariableMatrix<scalar, 3, 3>;

    scalar angle = M_PI / 6.0;  // 30 degrees
    vec3 euler_angles = vec3::filled(0.0);
    euler_angles[2] = angle;  // yaw
    
    mat3 rotation = Atom<scalar>::construct_rotation_matrix_from_euler_angles(euler_angles);
    
    // Rotation about Z-axis:
    // [cos(θ)  -sin(θ)  0]
    // [sin(θ)   cos(θ)  0]
    // [  0        0     1]
    REQUIRE(static_cast<double>(rotation.at(0, 0)) == Approx(std::cos(angle)).margin(1e-10));
    REQUIRE(static_cast<double>(rotation.at(0, 1)) == Approx(-std::sin(angle)).margin(1e-10));
    REQUIRE(static_cast<double>(rotation.at(0, 2)) == Approx(0.0).margin(1e-10));
    REQUIRE(static_cast<double>(rotation.at(1, 0)) == Approx(std::sin(angle)).margin(1e-10));
    REQUIRE(static_cast<double>(rotation.at(1, 1)) == Approx(std::cos(angle)).margin(1e-10));
    REQUIRE(static_cast<double>(rotation.at(1, 2)) == Approx(0.0).margin(1e-10));
    REQUIRE(static_cast<double>(rotation.at(2, 0)) == Approx(0.0).margin(1e-10));
    REQUIRE(static_cast<double>(rotation.at(2, 1)) == Approx(0.0).margin(1e-10));
    REQUIRE(static_cast<double>(rotation.at(2, 2)) == Approx(1.0).margin(1e-10));
}

TEST_CASE("Round-trip conversion: Euler -> Matrix -> Euler", "[rotation]") {
    using scalar = double;
    using vec3 = tinyla::VariableMatrix<scalar, 3, 1>;

    SECTION("General angles") {
        vec3 original_angles = vec3::filled(0.0);
        original_angles[0] = 0.5;   // roll
        original_angles[1] = 0.3;   // pitch
        original_angles[2] = -0.7;  // yaw
        
        auto rotation = Atom<scalar>::construct_rotation_matrix_from_euler_angles(original_angles);
        vec3 recovered_angles = Atom<scalar>::extract_euler_angles_from_rotation_matrix(rotation);
        
        REQUIRE(recovered_angles[0] == Approx(original_angles[0]).margin(1e-10));
        REQUIRE(recovered_angles[1] == Approx(original_angles[1]).margin(1e-10));
        REQUIRE(recovered_angles[2] == Approx(original_angles[2]).margin(1e-10));
    }
    
    SECTION("Multiple test angles") {
        std::vector<vec3> test_angles = {
            {0.1, 0.2, 0.3},
            {-0.5, 0.4, 0.9},
            {M_PI/4, M_PI/6, M_PI/3},
            {-M_PI/3, M_PI/5, -M_PI/4}
        };
        
        for (const auto& angles : test_angles) {
            auto rotation = Atom<scalar>::construct_rotation_matrix_from_euler_angles(angles);
            vec3 recovered = Atom<scalar>::extract_euler_angles_from_rotation_matrix(rotation);
            
            REQUIRE(recovered(0) == Approx(angles[0]).margin(1e-10));
            REQUIRE(recovered(1) == Approx(angles[1]).margin(1e-10));
            REQUIRE(recovered(2) == Approx(angles[2]).margin(1e-10));
        }
    }
}

TEST_CASE("Round-trip conversion: Matrix -> Euler -> Matrix", "[rotation]") {
    using scalar = double;
    using vec3 = tinyla::VariableMatrix<scalar, 3, 1>;
    using mat3 = tinyla::VariableMatrix<scalar, 3, 3>;

    // Create a rotation matrix from known Euler angles
    vec3 angles = vec3::filled(0.0);
    angles[0] = 0.6;
    angles[1] = 0.4;
    angles[2] = -0.8;
    
    mat3 original_matrix = Atom<scalar>::construct_rotation_matrix_from_euler_angles(angles);
    vec3 extracted_angles = Atom<scalar>::extract_euler_angles_from_rotation_matrix(original_matrix);
    mat3 reconstructed_matrix = Atom<scalar>::construct_rotation_matrix_from_euler_angles(extracted_angles);
    
    // Compare matrices element by element
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            REQUIRE(reconstructed_matrix(i, j) == Approx(original_matrix(i, j)).margin(1e-10));
        }
    }
}

TEST_CASE("Rotation matrix properties", "[rotation]") {
    using scalar = double;
    using vec3 = tinyla::VariableMatrix<scalar, 3, 1>;
    using mat3 = tinyla::VariableMatrix<scalar, 3, 3>;

    vec3 angles = vec3::filled(0.0);
    angles[0] = 0.7;
    angles[1] = -0.4;
    angles[2] = 1.2;
    
    mat3 rotation = Atom<scalar>::construct_rotation_matrix_from_euler_angles(angles);
    
    SECTION("Orthogonality: R * R^T = I") {
        mat3 should_be_identity = rotation * rotation.transposed();
        mat3 identity = mat3::identity();
        
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                REQUIRE(should_be_identity(i, j) == Approx(identity(i, j)).margin(1e-10));
            }
        }
    }
    
    SECTION("Determinant should be 1") {
        scalar det = static_cast<double>(rotation.at(0, 0)) * (static_cast<double>(rotation.at(1, 1)) * static_cast<double>(rotation.at(2, 2)) - static_cast<double>(rotation.at(1, 2)) * static_cast<double>(rotation.at(2, 1)))
                   - static_cast<double>(rotation.at(0, 1)) * (static_cast<double>(rotation.at(1, 0)) * static_cast<double>(rotation.at(2, 2)) - static_cast<double>(rotation.at(1, 2)) * static_cast<double>(rotation.at(2, 0)))
                   + static_cast<double>(rotation.at(0, 2)) * (static_cast<double>(rotation.at(1, 0)) * static_cast<double>(rotation.at(2, 1)) - static_cast<double>(rotation.at(1, 1)) * static_cast<double>(rotation.at(2, 0)));
        
        REQUIRE(det == Approx(1.0).margin(1e-10));
    }
}

TEST_CASE("Gimbal lock at pitch = +90 degrees", "[rotation]") {
    using scalar = double;
    using vec3 = tinyla::VariableMatrix<scalar, 3, 1>;
    using mat3 = tinyla::VariableMatrix<scalar, 3, 3>;

    vec3 angles = vec3::filled(0.0);
    angles[0] = 0.3;                      // roll
    angles[1] = M_PI / 2.0;               // pitch = 90 degrees (gimbal lock)
    angles[2] = 0.5;                      // yaw
    
    mat3 rotation = Atom<scalar>::construct_rotation_matrix_from_euler_angles(angles);
    vec3 extracted_angles = Atom<scalar>::extract_euler_angles_from_rotation_matrix(rotation);
    
    // At gimbal lock, roll is set to 0 and yaw absorbs the combined rotation
    REQUIRE(extracted_angles[1] == Approx(M_PI / 2.0).margin(1e-10));  // pitch preserved
    
    // Reconstruct to verify matrix is still correct
    mat3 reconstructed = Atom<scalar>::construct_rotation_matrix_from_euler_angles(extracted_angles);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            REQUIRE(reconstructed(i, j) == Approx(rotation(i, j)).margin(1e-10));
        }
    }
}

TEST_CASE("Gimbal lock at pitch = -90 degrees", "[rotation]") {
    using scalar = double;
    using vec3 = tinyla::VariableMatrix<scalar, 3, 1>;
    using mat3 = tinyla::VariableMatrix<scalar, 3, 3>;

    vec3 angles = vec3::filled(0.0);
    angles[0] = 0.4;                      // roll
    angles[1] = -M_PI / 2.0;              // pitch = -90 degrees (gimbal lock)
    angles[2] = 0.6;                      // yaw
    
    mat3 rotation = Atom<scalar>::construct_rotation_matrix_from_euler_angles(angles);
    vec3 extracted_angles = Atom<scalar>::extract_euler_angles_from_rotation_matrix(rotation);
    
    // At gimbal lock, roll is set to 0 and yaw absorbs the combined rotation
    REQUIRE(extracted_angles[1] == Approx(-M_PI / 2.0).margin(1e-10));  // pitch preserved
    
    // Reconstruct to verify matrix is still correct
    mat3 reconstructed = Atom<scalar>::construct_rotation_matrix_from_euler_angles(extracted_angles);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            REQUIRE(reconstructed(i, j) == Approx(rotation(i, j)).margin(1e-10));
        }
    }
}

TEST_CASE("Combined rotations", "[rotation]") {
    using scalar = double;
    using vec3 = tinyla::VariableMatrix<scalar, 3, 1>;
    using mat3 = tinyla::VariableMatrix<scalar, 3, 3>;

    vec3 angles = vec3::filled(0.0);
    angles[0] = M_PI / 4.0;   // roll = 45°
    angles[1] = M_PI / 6.0;   // pitch = 30°
    angles[2] = M_PI / 3.0;   // yaw = 60°
    
    mat3 rotation = Atom<scalar>::construct_rotation_matrix_from_euler_angles(angles);
    vec3 extracted = Atom<scalar>::extract_euler_angles_from_rotation_matrix(rotation);
    
    REQUIRE(extracted(0) == Approx(angles[0]).margin(1e-10));
    REQUIRE(extracted(1) == Approx(angles[1]).margin(1e-10));
    REQUIRE(extracted(2) == Approx(angles[2]).margin(1e-10));
}

TEST_CASE("Atom set_euler_angles and get_euler_angles integration", "[rotation]") {
    using scalar = double;
    using vec3 = tinyla::VariableMatrix<scalar, 3, 1>;

    vec3 position = vec3::filled(0.0);
    vec3 velocity = vec3::filled(0.0);
    
    Atom<scalar> atom(position, velocity, 1.0, 1.0);
    
    vec3 euler_angles = vec3::filled(0.0);
    euler_angles[0] = 0.5;
    euler_angles[1] = 0.3;
    euler_angles[2] = -0.8;
    
    atom.set_euler_angles(euler_angles);
    vec3 retrieved_angles = atom.get_euler_angles();
    
    REQUIRE(retrieved_angles[0] == Approx(euler_angles[0]).margin(1e-10));
    REQUIRE(retrieved_angles[1] == Approx(euler_angles[1]).margin(1e-10));
    REQUIRE(retrieved_angles[2] == Approx(euler_angles[2]).margin(1e-10));
}

TEST_CASE("Large angle values", "[rotation]") {
    using scalar = double;
    using vec3 = tinyla::VariableMatrix<scalar, 3, 1>;

    SECTION("Full rotations") {
        vec3 angles = vec3::filled(0.0);
        angles[0] = 2.0 * M_PI;  // Full rotation
        angles[2] = 2.0 * M_PI;  // Full rotation
        
        auto rotation = Atom<scalar>::construct_rotation_matrix_from_euler_angles(angles);
        
        // Should be close to identity
        auto identity = decltype(rotation)::identity();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                REQUIRE(rotation(i, j) == Approx(identity(i, j)).margin(1e-10));
            }
        }
    }
    
    SECTION("Near-boundary pitch values") {
        vec3 angles = vec3::filled(0.0);
        angles[1] = M_PI / 2.0 - 0.01;  // Just below gimbal lock
        
        auto rotation = Atom<scalar>::construct_rotation_matrix_from_euler_angles(angles);
        vec3 extracted = Atom<scalar>::extract_euler_angles_from_rotation_matrix(rotation);
        
        REQUIRE(extracted(1) == Approx(angles[1]).margin(1e-10));
    }
}
