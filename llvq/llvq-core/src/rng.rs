//! Minimal deterministic RNG (SplitMix64) for tests and benches.
//!
//! Kept in the library (not `#[cfg(test)]`) so integration tests and future
//! benches can use it without pulling an external `rand` dependency.

/// SplitMix64 — tiny, fast, and statistically fine for test sampling.
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    #[inline]
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}
