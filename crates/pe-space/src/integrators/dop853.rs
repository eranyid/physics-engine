//! Dormand–Prince 8(5,3) / DOP853 adaptive Runge–Kutta (Hairer–Nørsett–Wanner).
//!
//! Coefficients are the published DOP853 tableau as transcribed in SciPy's
//! BSD-licensed `scipy/integrate/_ivp/dop853_coefficients.py` (SciPy 1.14.1),
//! which reproduces Hairer's `dop853.f` tables from *Solving Ordinary
//! Differential Equations I*. Attribution: [`THIRD_PARTY.md`](../../../../THIRD_PARTY.md).
//!
//! - **12** stages for the 8th-order step; weights `B = A[12, 0..12)`.
//! - Embedded **5th**- and **3rd**-order error estimators (`E5`, `E3`) combined
//!   as in Hairer/SciPy for step-size control.
//! - Optional dense output (7th-order interpolant) via 4 extra stages.
//!
//! Determinism: portable [`pe_math::libm_ops`] only; no FMA.

use pe_math::libm_ops;
use pe_math::Vec3;

use crate::field::GravityField;
use crate::integrator::{AdaptiveStep, Integrator, StepResult};
use crate::state::NBodyState;

use super::accel_buf;

/// Number of RK stages in the 8th-order step (SciPy `N_STAGES`).
pub const N_STAGES: usize = 12;
/// Extended stage count including dense-output stages (`N_STAGES_EXTENDED`).
pub const N_STAGES_EXTENDED: usize = 16;
/// Dense-output polynomial power (`INTERPOLATOR_POWER`).
pub const INTERPOLATOR_POWER: usize = 7;

/// Nodes `c_i` (length [`N_STAGES_EXTENDED`]).
#[allow(clippy::excessive_precision)]
pub const C: [f64; N_STAGES_EXTENDED] = [
    0.0,
    0.526_001_519_587_677_318_785_587_544_488e-1,
    0.789_002_279_381_515_978_178_381_316_732e-1,
    0.118_350_341_907_227_396_726_757_197_510,
    0.281_649_658_092_772_603_273_242_802_490,
    0.333_333_333_333_333_333_333_333_333_333,
    0.25,
    0.307_692_307_692_307_692_307_692_307_692,
    0.651_282_051_282_051_282_051_282_051_282,
    0.6,
    0.857_142_857_142_857_142_857_142_857_142,
    1.0,
    1.0,
    0.1,
    0.2,
    0.777_777_777_777_777_777_777_777_777_778,
];

/// Nonzero Butcher `A` entries as `(i, j, a_ij)` for `i > j`.
/// Stages `0..12` advance the step; row 12 holds the order-8 weights `B`;
/// rows 13..15 are dense-output stages.
#[allow(clippy::excessive_precision)]
const A_ENTRIES: &[(usize, usize, f64)] = &[
    (1, 0, 5.260_015_195_876_773_187_855_875_444_88e-2),
    (2, 0, 1.972_505_698_453_789_945_445_953_291_83e-2),
    (2, 1, 5.917_517_095_361_369_836_337_859_875_49e-2),
    (3, 0, 2.958_758_547_680_684_918_168_929_937_75e-2),
    (3, 2, 8.876_275_643_042_054_754_506_789_813_24e-2),
    (4, 0, 2.413_651_341_592_666_855_023_697_986_65e-1),
    (4, 2, -8.845_494_793_282_860_853_448_649_627_17e-1),
    (4, 3, 9.248_340_032_617_920_031_157_379_665_43e-1),
    (5, 0, 3.703_703_703_703_703_703_703_703_703_7e-2),
    (5, 3, 1.708_286_087_294_738_712_796_044_821_73e-1),
    (5, 4, 1.254_676_875_668_224_250_166_918_141_23e-1),
    (6, 0, 3.710_937_5e-2),
    (6, 3, 1.702_522_110_195_440_393_149_780_602_72e-1),
    (6, 4, 6.021_653_898_045_596_068_502_193_972_83e-2),
    (6, 5, -1.757_812_5e-2),
    (7, 0, 3.709_200_011_850_479_271_087_793_198_36e-2),
    (7, 3, 1.703_839_257_122_399_938_102_140_547_05e-1),
    (7, 4, 1.072_620_304_463_732_846_518_091_991_68e-1),
    (7, 5, -1.531_943_774_862_440_175_279_361_582_36e-2),
    (7, 6, 8.273_789_163_814_022_887_584_737_660_02e-3),
    (8, 0, 6.241_109_587_160_757_171_144_295_778_12e-1),
    (8, 3, -3.360_892_629_446_941_294_068_571_098_25),
    (8, 4, -8.682_193_468_417_260_068_181_898_914_53e-1),
    (8, 5, 2.759_209_969_944_670_830_494_156_007_97e1),
    (8, 6, 2.015_406_755_047_789_340_861_867_889_79e1),
    (8, 7, -4.348_988_418_106_995_884_773_662_551_44e1),
    (9, 0, 4.776_625_364_382_643_658_904_339_085_27e-1),
    (9, 3, -2.488_114_619_971_667_641_926_425_864_68),
    (9, 4, -5.902_908_268_368_429_963_714_464_757_43e-1),
    (9, 5, 2.123_005_144_818_119_423_472_889_498_97e1),
    (9, 6, 1.527_923_363_288_242_358_325_969_229_38e1),
    (9, 7, -3.328_821_096_898_486_291_944_532_655_87e1),
    (9, 8, -2.033_120_170_850_862_613_582_229_285_93e-2),
    (10, 0, -9.371_424_300_859_873_257_170_402_165_8e-1),
    (10, 3, 5.186_372_428_844_063_708_300_238_532_09),
    (10, 4, 1.091_437_348_996_729_578_185_002_546_54),
    (10, 5, -8.149_787_010_746_926_125_139_972_673_57),
    (10, 6, -1.852_006_565_999_695_986_415_661_807_01e1),
    (10, 7, 2.273_948_709_935_050_428_189_700_567_34e1),
    (10, 8, 2.493_605_552_679_652_389_870_893_967_62),
    (10, 9, -3.046_764_471_898_219_500_382_366_902_2),
    (11, 0, 2.273_310_147_516_538_207_923_597_684_49),
    (11, 3, -1.053_449_546_673_725_019_840_666_898_79e1),
    (11, 4, -2.000_872_058_224_862_499_096_757_184_44),
    (11, 5, -1.795_893_186_311_879_891_727_659_505_34e1),
    (11, 6, 2.794_888_452_941_996_005_084_998_088_37e1),
    (11, 7, -2.858_998_277_135_023_694_740_655_086_74),
    (11, 8, -8.872_856_933_530_629_544_335_492_892_58),
    (11, 9, 1.236_056_717_579_430_306_472_662_015_28e1),
    (11, 10, 6.433_927_460_157_635_303_559_704_840_46e-1),
    // Order-8 weights B = A[12, 0..12)
    (12, 0, 5.429_373_411_656_876_223_805_357_663_63e-2),
    (12, 5, 4.450_312_892_752_408_881_441_139_505_66),
    (12, 6, 1.891_517_899_314_500_383_042_815_990_44),
    (12, 7, -5.801_203_960_010_584_781_467_211_422_7),
    (12, 8, 3.111_643_669_578_198_944_089_160_623_7e-1),
    (12, 9, -1.521_609_496_625_160_785_561_788_068_05e-1),
    (12, 10, 2.013_654_008_040_303_483_747_765_375_01e-1),
    (12, 11, 4.471_061_572_777_259_051_768_855_690_43e-2),
    // Dense-output stages
    (13, 0, 5.616_750_228_304_795_233_929_092_196_81e-2),
    (13, 6, 2.535_002_102_166_248_110_887_947_653_33e-1),
    (13, 7, -2.462_390_374_708_024_899_174_414_754_41e-1),
    (13, 8, -1.241_914_232_638_163_604_690_101_406_26e-1),
    (13, 9, 1.532_917_982_787_656_973_120_632_268_5e-1),
    (13, 10, 8.201_052_295_634_689_884_916_666_020_57e-3),
    (13, 11, 7.567_897_660_545_699_761_386_035_895_84e-3),
    (13, 12, -8.298e-3),
    (14, 0, 3.183_464_816_350_214_050_607_684_732_61e-2),
    (14, 5, 2.830_090_967_236_677_552_883_229_614_02e-2),
    (14, 6, 5.354_198_830_743_856_762_237_973_843_72e-2),
    (14, 7, -5.492_374_857_139_098_846_465_693_403_06e-2),
    (14, 10, -1.083_473_286_972_493_228_585_093_169_94e-4),
    (14, 11, 3.825_710_908_356_584_129_549_201_923_23e-4),
    (14, 12, -3.404_650_086_874_045_608_029_771_144_92e-4),
    (14, 13, 1.413_124_436_746_325_002_780_746_183_66e-1),
    (15, 0, -4.288_963_015_837_919_234_085_735_386_92e-1),
    (15, 5, -4.697_621_415_361_163_843_144_494_472_06),
    (15, 6, 7.683_421_196_062_599_041_842_409_538_78),
    (15, 7, 4.068_989_818_397_110_079_702_135_543_31),
    (15, 8, 3.567_271_874_552_811_092_706_695_430_21e-1),
    (15, 12, -1.399_024_165_159_014_621_294_180_097_34e-3),
    (15, 13, 2.947_514_789_152_772_338_955_627_214_9),
    (15, 14, -9.150_958_472_179_870_010_818_701_871_38),
];

/// Order-8 weights `B_i` (SciPy `B = A[12, :12]`).
#[allow(clippy::excessive_precision)]
pub const B: [f64; N_STAGES] = [
    5.429_373_411_656_876_223_805_357_663_63e-2,
    0.0,
    0.0,
    0.0,
    0.0,
    4.450_312_892_752_408_881_441_139_505_66,
    1.891_517_899_314_500_383_042_815_990_44,
    -5.801_203_960_010_584_781_467_211_422_7,
    3.111_643_669_578_198_944_089_160_623_7e-1,
    -1.521_609_496_625_160_785_561_788_068_05e-1,
    2.013_654_008_040_303_483_747_765_375_01e-1,
    4.471_061_572_777_259_051_768_855_690_43e-2,
];

/// Embedded 3rd-order error weights (length `N_STAGES+1`, last entry unused).
#[allow(clippy::excessive_precision)]
pub const E3: [f64; N_STAGES + 1] = [
    5.429_373_411_656_876_223_805_357_663_63e-2 - 0.244_094_488_188_976_377_952_755_905_512,
    0.0,
    0.0,
    0.0,
    0.0,
    4.450_312_892_752_408_881_441_139_505_66,
    1.891_517_899_314_500_383_042_815_990_44,
    -5.801_203_960_010_584_781_467_211_422_7,
    3.111_643_669_578_198_944_089_160_623_7e-1 - 0.733_846_688_281_611_857_341_361_741_547,
    -1.521_609_496_625_160_785_561_788_068_05e-1,
    2.013_654_008_040_303_483_747_765_375_01e-1,
    4.471_061_572_777_259_051_768_855_690_43e-2 - 0.220_588_235_294_117_647_058_823_529_412e-1,
    0.0,
];

/// Embedded 5th-order error weights.
#[allow(clippy::excessive_precision)]
pub const E5: [f64; N_STAGES + 1] = [
    0.131_200_449_941_948_807_325_010_299_6e-1,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.122_515_644_637_620_444_072_056_975_3e1,
    -0.495_758_949_657_250_191_521_407_995_2,
    0.166_437_718_245_498_653_696_153_041_5e1,
    -0.350_328_848_749_973_681_688_648_729_0,
    0.334_179_118_713_017_479_029_731_884_1,
    0.819_232_064_851_157_124_657_074_261_3e-1,
    -0.223_553_078_638_862_952_588_442_784_5e-1,
    0.0,
];

/// Dense-output `D` matrix rows (`INTERPOLATOR_POWER-3` × `N_STAGES_EXTENDED`).
#[allow(clippy::excessive_precision)]
const D_ENTRIES: &[(usize, usize, f64)] = &[
    (0, 0, -0.842_893_827_610_901_286_513_534_911_42e1),
    (0, 5, 0.566_714_953_519_377_769_625_317_835_90),
    (0, 6, -0.306_894_994_594_989_169_127_973_047_27e1),
    (0, 7, 0.238_466_765_651_206_982_877_281_496_80e1),
    (0, 8, 0.211_703_458_244_502_827_671_551_499_46e1),
    (0, 9, -0.871_391_583_777_972_992_067_899_074_90),
    (0, 10, 0.224_043_743_026_078_827_585_417_716_50e1),
    (0, 11, 0.631_578_778_769_468_818_155_702_492_90),
    (0, 12, -0.889_903_364_513_333_108_206_981_174_00e-1),
    (0, 13, 0.181_485_055_208_547_272_566_564_049_62e2),
    (0, 14, -0.919_463_239_247_835_540_004_519_844_36e1),
    (0, 15, -0.443_603_638_759_489_396_643_105_720_00e1),
    (1, 0, 0.104_275_086_425_791_346_034_131_510_09e2),
    (1, 5, 0.242_283_491_775_258_182_884_301_753_19e3),
    (1, 6, 0.165_200_451_717_270_281_985_053_948_87e3),
    (1, 7, -0.374_546_754_722_690_202_795_183_121_52e3),
    (1, 8, -0.221_136_668_531_253_060_362_709_385_78e2),
    (1, 9, 0.773_343_266_847_226_383_896_038_988_08e1),
    (1, 10, -0.306_740_847_310_893_981_820_612_136_26e2),
    (1, 11, -0.933_213_052_643_022_787_295_672_217_06e1),
    (1, 12, 0.156_972_381_217_708_438_861_310_910_75e2),
    (1, 13, -0.311_394_032_195_651_776_772_828_504_11e2),
    (1, 14, -0.935_292_435_884_447_838_657_138_626_64e1),
    (1, 15, 0.358_168_414_863_940_837_524_658_985_40e2),
    (2, 0, 0.199_850_532_420_024_338_209_876_536_17e2),
    (2, 5, -0.387_037_308_749_351_765_551_059_017_42e3),
    (2, 6, -0.189_178_138_195_167_568_828_308_383_28e3),
    (2, 7, 0.527_808_159_205_423_649_005_610_166_86e3),
    (2, 8, -0.115_739_025_399_596_301_261_418_711_34e2),
    (2, 9, 0.688_123_269_469_630_001_696_669_226_61e1),
    (2, 10, -0.100_060_509_669_108_384_031_838_609_80e1),
    (2, 11, 0.777_713_779_805_344_320_928_692_657_40),
    (2, 12, -0.277_820_575_235_350_840_659_320_043_39e1),
    (2, 13, -0.601_966_952_312_641_207_582_673_808_46e2),
    (2, 14, 0.843_204_055_066_771_610_181_599_037_84e2),
    (2, 15, 0.119_922_911_361_827_893_280_351_300_30e2),
    (3, 0, -0.256_939_334_627_037_490_033_125_861_29e2),
    (3, 5, -0.154_189_748_690_236_433_740_539_936_27e3),
    (3, 6, -0.231_529_379_176_045_495_675_360_391_09e3),
    (3, 7, 0.357_639_117_910_614_123_782_853_499_10e3),
    (3, 8, 0.934_053_241_836_243_100_039_076_917_04e2),
    (3, 9, -0.374_583_231_364_516_331_568_751_393_51e2),
    (3, 10, 0.104_099_649_508_962_300_451_472_461_84e3),
    (3, 11, 0.298_402_934_266_605_031_233_443_635_79e2),
    (3, 12, -0.435_334_565_900_111_437_544_321_750_58e2),
    (3, 13, 0.963_245_539_591_882_829_483_949_506_00e2),
    (3, 14, -0.391_772_616_756_154_391_652_314_861_72e2),
    (3, 15, -0.149_726_836_257_985_625_814_221_252_76e3),
];

fn a(i: usize, j: usize) -> f64 {
    for &(ii, jj, v) in A_ENTRIES {
        if ii == i && jj == j {
            return v;
        }
    }
    0.0
}

fn d_coeff(row: usize, col: usize) -> f64 {
    for &(r, c, v) in D_ENTRIES {
        if r == row && c == col {
            return v;
        }
    }
    0.0
}

/// Adaptive DOP853 controller.
#[derive(Clone, Debug)]
pub struct Dop853 {
    /// Step-size / tolerance settings.
    pub ctrl: AdaptiveStep,
    /// When true, retain dense-output stages after an accepted step.
    pub dense_output: bool,
    /// Dense-output polynomial coefficients after the last accepted step
    /// (empty if unused). Length `INTERPOLATOR_POWER` vectors of `(x,v)` packs.
    dense_f: Vec<Vec<f64>>,
    /// Left endpoint state for dense output (`x‖v` pack).
    dense_y0: Vec<f64>,
    /// Left / right times of the dense bracket.
    dense_t0: f64,
    dense_t1: f64,
}

impl Default for Dop853 {
    fn default() -> Self {
        Self::new(AdaptiveStep::default())
    }
}

impl Dop853 {
    /// Construct with adaptive controller (dense output off).
    pub fn new(ctrl: AdaptiveStep) -> Self {
        Self {
            ctrl,
            dense_output: false,
            dense_f: Vec::new(),
            dense_y0: Vec::new(),
            dense_t0: 0.0,
            dense_t1: 0.0,
        }
    }

    /// Enable optional dense output after accepted steps.
    pub fn with_dense_output(mut self, enable: bool) -> Self {
        self.dense_output = enable;
        self
    }
}

/// Construct DOP853 with the given adaptive controller.
pub fn dormand_prince_8_7(ctrl: AdaptiveStep) -> Dop853 {
    Dop853::new(ctrl)
}

/// Free-function one-shot with default tolerances.
pub fn dop853(field: &dyn GravityField, state: &mut NBodyState, t: f64, dt: f64) -> StepResult {
    let mut integ = Dop853::new(AdaptiveStep {
        dt,
        ..AdaptiveStep::default()
    });
    integ.step(field, state, t, dt)
}

struct Phase {
    x: Vec<Vec3>,
    v: Vec<Vec3>,
}

fn pack_phase(p: &Phase) -> Vec<f64> {
    let n = p.x.len();
    let mut out = Vec::with_capacity(n * 6);
    for i in 0..n {
        out.extend_from_slice(&p.x[i].to_array());
        out.extend_from_slice(&p.v[i].to_array());
    }
    out
}

fn unpack_phase(mass_n: usize, y: &[f64]) -> Phase {
    let mut x = Vec::with_capacity(mass_n);
    let mut v = Vec::with_capacity(mass_n);
    for i in 0..mass_n {
        let o = i * 6;
        x.push(Vec3::new(y[o], y[o + 1], y[o + 2]));
        v.push(Vec3::new(y[o + 3], y[o + 4], y[o + 5]));
    }
    Phase { x, v }
}

fn add_scaled_y(y0: &[f64], ks: &[Vec<f64>], scales: &[f64], h: f64) -> Vec<f64> {
    let mut y = y0.to_vec();
    for (s, k) in scales.iter().zip(ks.iter()) {
        let f = h * *s;
        for (yi, ki) in y.iter_mut().zip(k.iter()) {
            *yi += ki * f;
        }
    }
    y
}

fn deriv_pack(field: &dyn GravityField, mass: &[f64], y: &[f64], t: f64) -> Vec<f64> {
    let n = mass.len();
    let p = unpack_phase(n, y);
    let scratch = NBodyState::new(mass.to_vec(), p.x.clone(), p.v.clone());
    let a = accel_buf(field, &scratch, t);
    let mut out = Vec::with_capacity(n * 6);
    for i in 0..n {
        out.extend_from_slice(&p.v[i].to_array());
        out.extend_from_slice(&a[i].to_array());
    }
    out
}

fn err_norm_combined(
    err5: &[f64],
    err3: &[f64],
    y: &[f64],
    y_new: &[f64],
    atol: f64,
    rtol: f64,
) -> f64 {
    // SciPy DOP853: scale = atol + max(|y|,|y_new|)*rtol; then
    // denom = ||e5||² + 0.01||e3||²; return |h| * ||e5||² / sqrt(denom * n)
    // is applied outside — here we return the dimensionless combined norm of
    // (err5, err3) / scale (caller multiplies by |h|).
    let n = err5.len() as f64;
    let mut e5_sq = 0.0;
    let mut e3_sq = 0.0;
    for i in 0..err5.len() {
        let sc = atol + rtol * libm_ops::fabs(y[i]).max(libm_ops::fabs(y_new[i]));
        let a = err5[i] / sc;
        let b = err3[i] / sc;
        e5_sq += a * a;
        e3_sq += b * b;
    }
    if e5_sq == 0.0 && e3_sq == 0.0 {
        return 0.0;
    }
    let denom = e5_sq + 0.01 * e3_sq;
    e5_sq / libm_ops::sqrt(denom * n)
}

impl Dop853 {
    /// Evaluate dense output at time `t` inside the last accepted step bracket.
    ///
    /// Returns packed `(x‖v)` or `None` if dense output was not enabled / no step.
    pub fn interpolate_pack(&self, t: f64) -> Option<Vec<f64>> {
        if self.dense_f.is_empty() || self.dense_t1 <= self.dense_t0 {
            return None;
        }
        if t < self.dense_t0 || t > self.dense_t1 {
            return None;
        }
        let h = self.dense_t1 - self.dense_t0;
        let x = (t - self.dense_t0) / h;
        let dim = self.dense_y0.len();
        let mut y = vec![0.0; dim];
        // Horner-like product from Hairer/SciPy Dop853DenseOutput.
        for (i, f) in self.dense_f.iter().rev().enumerate() {
            for d in 0..dim {
                y[d] += f[d];
            }
            let factor = if i % 2 == 0 { x } else { 1.0 - x };
            for d in 0..dim {
                y[d] *= factor;
            }
        }
        for d in 0..dim {
            y[d] += self.dense_y0[d];
        }
        Some(y)
    }

    /// Dense-output position of body `i` at time `t` (requires [`Self::dense_output`]).
    pub fn interpolate_position(&self, t: f64, body: usize) -> Option<Vec3> {
        let y = self.interpolate_pack(t)?;
        let o = body * 6;
        if o + 2 >= y.len() {
            return None;
        }
        Some(Vec3::new(y[o], y[o + 1], y[o + 2]))
    }

    fn store_dense(
        &mut self,
        field: &dyn GravityField,
        mass: &[f64],
        y0: &[f64],
        y1: &[f64],
        k: &[Vec<f64>],
        t0: f64,
        h: f64,
        f_new: &[f64],
    ) {
        let n_ext = N_STAGES_EXTENDED;
        let mut k_ext: Vec<Vec<f64>> = k.to_vec();
        // k currently has N_STAGES+1 = 13 entries (stages 0..11 + f_new at 12).
        debug_assert_eq!(k_ext.len(), N_STAGES + 1);
        // Compute extra stages 13,14,15 (indices in extended tableau).
        for stage in (N_STAGES + 1)..n_ext {
            let mut scales = vec![0.0; stage];
            for j in 0..stage {
                scales[j] = a(stage, j);
            }
            let y_s = add_scaled_y(y0, &k_ext, &scales, h);
            let d = deriv_pack(field, mass, &y_s, t0 + C[stage] * h);
            k_ext.push(d);
        }
        let dim = y0.len();
        let f_old = &k_ext[0];
        let mut f_poly = vec![vec![0.0; dim]; INTERPOLATOR_POWER];
        // F[0] = Δy; F[1] = h f_old − Δy; F[2] = 2 Δy − h (f_new + f_old);
        // F[3..] = h D · K
        for d in 0..dim {
            let dy = y1[d] - y0[d];
            f_poly[0][d] = dy;
            f_poly[1][d] = h * f_old[d] - dy;
            f_poly[2][d] = 2.0 * dy - h * (f_new[d] + f_old[d]);
        }
        for row in 0..(INTERPOLATOR_POWER - 3) {
            for d in 0..dim {
                let mut acc = 0.0;
                for col in 0..n_ext {
                    acc += d_coeff(row, col) * k_ext[col][d];
                }
                f_poly[row + 3][d] = h * acc;
            }
        }
        self.dense_f = f_poly;
        self.dense_y0 = y0.to_vec();
        self.dense_t0 = t0;
        self.dense_t1 = t0 + h;
    }
}

impl Integrator for Dop853 {
    fn name(&self) -> &'static str {
        "dop853"
    }

    fn step(
        &mut self,
        field: &dyn GravityField,
        state: &mut NBodyState,
        t: f64,
        dt: f64,
    ) -> StepResult {
        let mut h = dt.clamp(self.ctrl.dt_min, self.ctrl.dt_max);
        let mass = state.mass.clone();
        let n = state.n();
        let base = Phase {
            x: state.x.clone(),
            v: state.v.clone(),
        };
        let y0 = pack_phase(&base);

        for _attempt in 0..16 {
            let mut ks: Vec<Vec<f64>> = Vec::with_capacity(N_STAGES + 1);
            // Stage 0
            ks.push(deriv_pack(field, &mass, &y0, t));
            for stage in 1..N_STAGES {
                let mut scales = vec![0.0; stage];
                for j in 0..stage {
                    scales[j] = a(stage, j);
                }
                let y_s = add_scaled_y(&y0, &ks, &scales, h);
                ks.push(deriv_pack(field, &mass, &y_s, t + C[stage] * h));
            }
            // y_new from B weights
            let mut y_new = y0.clone();
            for s in 0..N_STAGES {
                let f = h * B[s];
                for d in 0..y_new.len() {
                    y_new[d] += ks[s][d] * f;
                }
            }
            let f_new = deriv_pack(field, &mass, &y_new, t + h);
            ks.push(f_new.clone());

            // Error estimate (combined 5th/3rd)
            let dim = y0.len();
            let mut err5 = vec![0.0; dim];
            let mut err3 = vec![0.0; dim];
            for s in 0..=N_STAGES {
                for d in 0..dim {
                    err5[d] += E5[s] * ks[s][d];
                    err3[d] += E3[s] * ks[s][d];
                }
            }
            let err_unit =
                err_norm_combined(&err5, &err3, &y0, &y_new, self.ctrl.atol, self.ctrl.rtol);
            let err = libm_ops::fabs(h) * err_unit;

            // SciPy uses error_estimator_order = 7 → exponent −1/8
            let order_est = 7.0;
            let mut accept = err <= 1.0;
            if err == 0.0 {
                accept = true;
            }
            let factor = if err == 0.0 {
                10.0
            } else {
                0.9 * libm_ops::pow(1.0 / err, 1.0 / (order_est + 1.0))
            };
            let factor = factor.clamp(0.2, 10.0);
            let h_new = (h * factor).clamp(self.ctrl.dt_min, self.ctrl.dt_max);

            if accept || h <= self.ctrl.dt_min * 1.000_000_1 {
                let p_new = unpack_phase(n, &y_new);
                for i in 0..n {
                    let dx = p_new.x[i] - state.x[i];
                    state.add_position(i, dx);
                    state.v[i] = p_new.v[i];
                }
                if self.dense_output {
                    self.store_dense(field, &mass, &y0, &y_new, &ks, t, h, &f_new);
                }
                self.ctrl.dt = h_new;
                return StepResult {
                    t: t + h,
                    dt_taken: h,
                    err_est: err,
                };
            }
            h = h_new;
        }

        let h = self.ctrl.dt_min;
        let mut rk = crate::integrators::Rk4;
        let res = rk.step(field, state, t, h);
        self.ctrl.dt = h;
        res
    }
}

/// One adaptive DOP853 step for a general first-order ODE `y' = f(t, y)`.
///
/// Used by analytic oracles (convergence order, Arenstorf). Returns
/// `(y_new, t_new, dt_taken, err_est, h_next)`.
pub fn ode_step<F>(
    f: &F,
    t: f64,
    y: &[f64],
    dt: f64,
    ctrl: &mut AdaptiveStep,
) -> (Vec<f64>, f64, f64, f64)
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    let mut h = dt.clamp(ctrl.dt_min, ctrl.dt_max);
    let y0 = y.to_vec();
    for _ in 0..16 {
        let mut ks: Vec<Vec<f64>> = Vec::with_capacity(N_STAGES + 1);
        ks.push(f(t, &y0));
        for stage in 1..N_STAGES {
            let mut scales = vec![0.0; stage];
            for j in 0..stage {
                scales[j] = a(stage, j);
            }
            let y_s = add_scaled_y(&y0, &ks, &scales, h);
            ks.push(f(t + C[stage] * h, &y_s));
        }
        let mut y_new = y0.clone();
        for s in 0..N_STAGES {
            let fac = h * B[s];
            for d in 0..y_new.len() {
                y_new[d] += ks[s][d] * fac;
            }
        }
        let f_new = f(t + h, &y_new);
        ks.push(f_new);

        let dim = y0.len();
        let mut err5 = vec![0.0; dim];
        let mut err3 = vec![0.0; dim];
        for s in 0..=N_STAGES {
            for d in 0..dim {
                err5[d] += E5[s] * ks[s][d];
                err3[d] += E3[s] * ks[s][d];
            }
        }
        let err_unit = err_norm_combined(&err5, &err3, &y0, &y_new, ctrl.atol, ctrl.rtol);
        let err = libm_ops::fabs(h) * err_unit;
        let order_est = 7.0;
        let accept = err <= 1.0 || err == 0.0 || h <= ctrl.dt_min * 1.000_000_1;
        let factor = if err == 0.0 {
            10.0
        } else {
            0.9 * libm_ops::pow(1.0 / err, 1.0 / (order_est + 1.0))
        };
        let factor = factor.clamp(0.2, 10.0);
        let h_new = (h * factor).clamp(ctrl.dt_min, ctrl.dt_max);
        if accept {
            ctrl.dt = h_new;
            return (y_new, t + h, h, err);
        }
        h = h_new;
    }
    ctrl.dt = ctrl.dt_min;
    (y0, t, 0.0, -1.0)
}

/// Fixed-step (non-adaptive) DOP853 advance — for measured convergence order.
pub fn ode_step_fixed<F>(f: &F, t: f64, y: &[f64], h: f64) -> Vec<f64>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    let y0 = y.to_vec();
    let mut ks: Vec<Vec<f64>> = Vec::with_capacity(N_STAGES);
    ks.push(f(t, &y0));
    for stage in 1..N_STAGES {
        let mut scales = vec![0.0; stage];
        for j in 0..stage {
            scales[j] = a(stage, j);
        }
        let y_s = add_scaled_y(&y0, &ks, &scales, h);
        ks.push(f(t + C[stage] * h, &y_s));
    }
    let mut y_new = y0;
    for s in 0..N_STAGES {
        let fac = h * B[s];
        for d in 0..y_new.len() {
            y_new[d] += ks[s][d] * fac;
        }
    }
    y_new
}

/// Verify Butcher consistency: row sums `Σ_j A_{i j} = c_i` and `Σ B = 1`.
///
/// Returns max absolute residual (should be ≲ 1e-14 for the published tables).
pub fn butcher_consistency_residual() -> (f64, f64) {
    let mut max_row = 0.0_f64;
    for i in 1..N_STAGES {
        let mut s = 0.0;
        for j in 0..i {
            s += a(i, j);
        }
        let r = libm_ops::fabs(s - C[i]);
        if r > max_row {
            max_row = r;
        }
    }
    let mut bsum = 0.0;
    for &bi in &B {
        bsum += bi;
    }
    (max_row, libm_ops::fabs(bsum - 1.0))
}

#[cfg(test)]
mod coeff_tests {
    use super::*;

    #[test]
    fn butcher_row_sums_and_weights() {
        let (row, b) = butcher_consistency_residual();
        assert!(row < 1e-14, "row-sum residual {row}");
        assert!(b < 1e-14, "B-sum residual {b}");
    }
}
