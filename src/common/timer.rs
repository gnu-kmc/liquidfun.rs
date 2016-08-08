extern crate std;

use std::time::Instant;

struct Timer {
    start: Instant
}

impl Timer {
    pub fn new() -> Timer {
        Timer{start: Instant::now()}
    }

    /// Reset the timer.
    pub fn reset(&mut self) {
        self.start = Instant::now();
    }

    /// Get the time since construction or the last reset.
    pub fn get_milliseconds(&self) -> f32 {
        (Instant::now().duration_since(self.start).as_secs() * 1000u64) as f32 +
        (Instant::now().duration_since(self.start).subsec_nanos() / 1000u32) as f32
    }

    /// Get platform specific tick count
    fn get_ticks(&self) -> i64 {
        unimplemented!();
    }
}
