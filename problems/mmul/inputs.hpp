// ============================================================================
// inputs.hpp  —  USER band (Phase 1, Shape B). Form-generated in production;
// hand-authored here. Owns the ENTIRE I/O boundary of the problem:
//   scalars, data generators, KTT argument registration, and which buffer is
//   validated. The engine skeleton is arg-agnostic and only consumes the
//   `Inputs` struct returned by DefineInputs().
//
// Rule: generators may reference SCALARS, never tuning parameters.
// ============================================================================
#pragma once
#include <vector>
#include <random>
#include <Ktt.h>

// --- scalars (host consts; also runtime kernel args, see DefineInputs) -------
inline constexpr int kSizeM = 1024;
inline constexpr int kSizeN = 1024;
inline constexpr int kSizeK = 1024;

// --- data generators (one per buffer) ---------------------------------------
inline std::vector<float> gen_mat_a() {
    std::vector<float> v(static_cast<size_t>(kSizeM) * kSizeK);
    std::mt19937 rng(1234u);
    std::uniform_real_distribution<float> d(-2.0f, 2.0f);
    for (auto& x : v) x = d(rng);
    return v;
}
inline std::vector<float> gen_mat_b() {
    std::vector<float> v(static_cast<size_t>(kSizeN) * kSizeK);
    std::mt19937 rng(5678u);
    std::uniform_real_distribution<float> d(-2.0f, 2.0f);
    for (auto& x : v) x = d(rng);
    return v;
}
inline std::vector<float> gen_mat_c() {
    return std::vector<float>(static_cast<size_t>(kSizeM) * kSizeN, 0.0f);
}

// --- boundary exposed to the engine skeleton + the LLM regions --------------
struct Inputs {
    ktt::ArgumentId kSizeM_, kSizeN_, kSizeK_;
    ktt::ArgumentId mat_a, mat_b, mat_c;
    ktt::ArgumentId validated;                  // buffer checked against the reference
    std::vector<ktt::ArgumentId> boundary;      // args in reference-signature order
};

inline Inputs DefineInputs(ktt::Tuner& t) {
    Inputs in;
    in.kSizeM_ = t.AddArgumentScalar(kSizeM);
    in.kSizeN_ = t.AddArgumentScalar(kSizeN);
    in.kSizeK_ = t.AddArgumentScalar(kSizeK);
    in.mat_a = t.AddArgumentVector(gen_mat_a(), ktt::ArgumentAccessType::ReadOnly);
    in.mat_b = t.AddArgumentVector(gen_mat_b(), ktt::ArgumentAccessType::ReadOnly);
    in.mat_c = t.AddArgumentVector(gen_mat_c(), ktt::ArgumentAccessType::WriteOnly);
    in.validated = in.mat_c;
    in.boundary  = {in.kSizeM_, in.kSizeN_, in.kSizeK_, in.mat_a, in.mat_b, in.mat_c};
    return in;
}
