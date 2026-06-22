// Enzyme-native repro for the CbrtOp derivative bug.
// This is test/lit_tests/diffrules/stablehlo/cbrt.mlir's @main, with ONE change:
// the inputs [8.0, 27.0] are flipped to [-8.0, -27.0]. The true gradients are the
// SAME magnitudes (1/(3*cbrt(x)^2) is sign-independent), so %expected is unchanged.
// The author's test passes on the positive inputs and would FAIL here:
// the rule emits `stablehlo.power %arg0, -0.6666` which is NaN for negative bases
// (StableHLO power spec example: power([-36.0],[1.1]) = -nan), so the computed
// gradient is NaN, not 0.0833 / 0.0370.
//
// Run (their own third RUN line, no extra harness):
//   enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops \
//     --arith-raise --enzyme-hlo-opt | stablehlo-translate - --interpret

func.func @cbrt(%x : tensor<2xf32>) -> tensor<2xf32> {
  %y = stablehlo.cbrt %x : (tensor<2xf32>) -> tensor<2xf32>
  func.return %y : tensor<2xf32>
}

func.func @main() {
  %x = stablehlo.constant dense<[-8.0, -27.0]> : tensor<2xf32>          // was [8.0, 27.0]
  %out = stablehlo.constant dense<[-2.0, -3.0]> : tensor<2xf32>         // cbrt(-8), cbrt(-27): real
  %expected = stablehlo.constant dense<[0.083333336, 0.037037037]> : tensor<2xf32>  // UNCHANGED

  %dx = stablehlo.constant dense<1.0> : tensor<2xf32>

  %fwd:2 = enzyme.fwddiff @cbrt(%x, %dx) {
    activity=[#enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>)

  check.expect_almost_eq %fwd#0, %out : tensor<2xf32>        // primal: cbrt(-8)=-2 OK
  check.expect_almost_eq %fwd#1, %expected : tensor<2xf32>   // gradient: NaN != 0.0833  -> FAILS

  func.return
}
