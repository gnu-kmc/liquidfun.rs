extern crate std;

/// Calculates min/max/mean of a set of samples
struct Stat {
    count: i32,
    total: f64,
    min: f32,
    max: f32
}

impl Stat {
    fn new() -> Stat {
        Stat{count: 0, total: 0.0f64, min: std::f32::MAX, max: -std::f32::MAX}
    }

    /// Record a samples
    fn record(&mut self, t: f32) {
        self.total += t as f64;
        self.min = self.min.min(t);
        self.max = self.max.max(t);
        self.count += 1;
    }

    /// Returns the number of recorded samples
    fn get_count(&self) -> i32 {
        self.count
    }

    /// Returns the mean of all recorded samples,
    /// Returns 0 if there are no recorded samples
    fn get_mean(&self) -> f32 {
        if self.count == 0 {
            return 0.0f32;
        }
        (self.total / (self.count as f64)) as f32
    }

    /// Returns the min of all recorded samples,
    /// std::f32::MAX if there are no recored samples
    fn get_min(&self) -> f32 {
        self.min
    }

    /// Returns the max of all recorded samples,
    /// -std::f32::MAX if there are no recorded samples
    fn get_max(&self) -> f32 {
        self.max
    }

    /// Erase all recorded samples
    fn clear(&mut self) {
        self.count = 0;
        self.total = 0.0f64;
        self.min = std::f32::MAX;
        self.max = -std::f32::MAX;
    }
}
