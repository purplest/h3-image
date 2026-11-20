// h3-image-specific Metal kernels (SPEC.md parts 2-5).
//
// The Makefile concatenates src/gpu/shaders.metal (the migrated h3.c kernel
// set, h3i_* names) followed by this file into build/h3i_shaders.metal, which
// the GPU runtime compiles at startup in safe-math mode.
//
// Kernels land here per the build plan:
//   Phase 2:  h3i_convert_<f32|f16|bf16|q8_0|q2_k|q3_k|q4_k|q5_k|q6_k>_<bf16|f32>
//   Phase 11: h3i_bf16_to_f16, h3i_f16_to_bf16_checked, h3i_transpose_bf16_f16
//
// Intentionally empty until Phase 2.
