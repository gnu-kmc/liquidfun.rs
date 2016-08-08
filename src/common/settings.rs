extern crate std;
extern crate libc;

/// Global tuning constants based on meters-killograms-seconds (MKS) units.

// Collision

/// The maximum number of contact points between two convex shapes. Do
/// not change this value.
const MAX_MANIFOLD_POINTS: i32 = 2;

/// The maximum number of vertices on a convex polygon. You cannot increase
/// this too much because BlockAllocator has a maximum object size.
const MAX_POLYGON_VERTICES: i32 = 8;

/// This is used to fatten AABBs in the dynamic tree. This allows proxies
/// to move by a small amount without triggering a tree adjustment.
/// This is in meters.
const AABB_EXTENSION: f32 = 0.1f32;

/// This is used to fatten AABBs in the dynamic tree. This is used to predict
/// the future position based on the current displacement.
/// This is a dimentionless multiplier.
const AABB_MULTIPLIER: f32 = 2.0f32;

/// A small length used as a collision and constraint tolerance. Usually it is
/// chosen to be numerically significant, but visually insignificant.
const LINER_SLOP: f32 = 0.005f32;

/// A small angle used as a collision and constraint tolerance. Usually it is
/// chosen to be numerically significant, but visually insignificant.
const ANGULAR_SLOP: f32 = 2.0f32 / 180.0f32 * std::f32::consts::PI;

/// The radius of the polygon/edge shape skin. This should not be modified. Making
/// this smaller means polygons will have an insufficient buffer for continuous collision.
/// Making it larger may create artifacts for vertex collision.
const POLYGON_RADIUS: f32 = 2.0f32 * LINER_SLOP;

/// Maximum number of sub-steps per contact in continuous physics simulation.
const MAX_SUB_STEPS: i32 = 8;

// Dynamics

/// Maximum number of contacts to be handled to solve a TOI impact.
const MAX_TOI_CONTACTS: i32 = 32;

/// A velocity threshold ofr elastic collisions. Any collision with a relative linear
/// velocity below this threshold will be treated as inelastic.
const VELOCITY_THRESHOLD: f32 = 1.0f32;

/// The maximum linear position correction used when solving constraints. This helps to
/// prevent overshoot.
const MAX_LINEAR_CORRECTION: f32 = 0.2f32;

/// The maximum angular position correction used when solving constraints. This helps to
/// prevent overshoot.
const MAX_ANGULAR_CORRECTION: f32 = 8.0f32 / 180.0f32 * std::f32::consts::PI;

/// The maximum linear velocity of a body. This limit is very large and is used
/// to prevent numerical Problems. You shouldn't need to adust this.
const MAX_TRANSLATION: f32 = 2.0f32;
const MAX_TRANSLATION_SQUARED:f32 = MAX_TRANSLATION * MAX_TRANSLATION;

/// The maximum angular velocity of a body. This limit is very large and is used
/// to prevent numerical problems. You shouldn't need to adjust this.
const MAX_ROTATION: f32 = 0.5f32 * std::f32::consts::PI;
const MAX_ROTATION_SQUARED: f32 = MAX_ROTATION * MAX_ROTATION;

/// This scale factor controls how fast overlap is resolved. Ideally this would be 1 so
/// that overlap is removed in one time step. However using values close to 1 often lead
/// to overshoot.
const BAUMGARTE: f32 = 0.2f32;
const TOI_BAUGARTE: f32 = 0.75f32;


// Particle

/// NEON SIMD requires 16-bit particle indices.
#[cfg_attr(
    all(
        not(USE_16_BIT_PARTICLE_INDICES), LIQUIDFUN_SIMD_NEON
    ),
    USE_16_BIT_PARTICLE_INDICES
)]


/// A symbolic constant that stands for particle allocation error.
const INVALID_PARTICLE_INDEX: i32 = -1;

#[cfg(USE_16_BIT_PARTICLE_INDICES)]
const MAX_PARTICLE_INDEX: i16 = 0x7FFF;
#[cfg(not(USE_16_BIT_PARTICLE_INDICES))]
const MAX_PARTICLE_INDEX: i32 = 0x7FFFFFFF;

/// The default distance etween particles, multiplied by the particle diameter.
const PARTICLE_STRIDE: f32 = 0.75f32;

/// THe minimum particle weight that prodeces pressure.
const MIN_PARTICLE_WEIGHT: f32 = 1.0f32;

/// The upper limit for particle pressure.
const MAX_PARTICLE_PRESSURE: f32 = 0.32f32;

/// The upper limit ofr force between particles.
const MAX_PARTICLE_FORCE: f32 = 0.5f32;

/// The maximum distance between particles in a triad, multiplied by the
/// particle diameter.
const MAX_TRIAD_DISTANCE: i32 = 2;
const MAX_TRIAD_DISTANCE_SQUARED: i32 = MAX_TRIAD_DISTANCE * MAX_TRIAD_DISTANCE;

/// The inital size of particle data buffers.
const MIN_PARTICLE_SYSTEM_BUFFER_CAPACITY: i32 = 256;

/// The time into the future that collisions against barrier particles will be detected.
const BARRIER_COLLISION_TIME: f32 = 2.5f32;

// Sleep

/// The time that a body must be still before it will go to sleep.
const TIME_TO_SLEEP: f32 = 0.5f32;

/// A body cannot sleep if its linear velocity is above this tolerance.
const ANGULAR_SLEEP_TOLERANCE: f32 = 2.0f32 / 180.0f32 * std::f32::consts::PI;

// Memory Allocation

/// Implement this function to use your own memory allocator.
fn alloc(size: usize) -> *mut libc::c_void {
    unsafe{
        NUM_ALLOCS+=1;
        ALLOC_CALLBACK(size, CALLBACK_DATA)
    }
}

/// If you implement alloc, you should alos implement this function.
fn free(mem: *mut libc::c_void) {
    unsafe{
        NUM_ALLOCS-=1;
        FREE_CALLBACK(mem, CALLBACK_DATA);
    }
}

/// Use this function to override alloc() without recompiling this library.
pub type AllocFunction = fn(size: usize, callback_data: *mut libc::c_void) -> *mut libc::c_void;

/// Use this function to override free() without recompiling this library.
pub type FreeFunction = fn(mem: *mut libc::c_void, callback_data: *mut libc::c_void);

/// Set alloc and free callbacks to override the default behavior of using
/// malloc() and free() for dynamic memory allocation.
/// Set allocCallback and freeCallback to NULL to restore th default
/// allocator (malloc/ free).
fn set_alloc_free_callbacks(alloc_callback: &AllocFunction,
                            free_callback: &FreeFunction,
                            callback_data: *mut libc::c_void) {
    assert_eq!(get_num_allocs(), 0);
    unsafe{
        ALLOC_CALLBACK = *alloc_callback;
        FREE_CALLBACK = *free_callback;
        CALLBACK_DATA = callback_data;
    }
}

/// Set the number of calls to settings::alloc minus the number of calls to settings::free.
/// This can be used to disable the empty heap check in
/// set_alloc_free_callbacks() which can be useful for testing.
fn set_num_allocs(num_allocs: i32) {
    unsafe{NUM_ALLOCS = num_allocs;}
}

/// Get number of calls to settings::alloc minus number of calls to settings::free.
fn get_num_allocs() -> i32 {
    unsafe{NUM_ALLOCS}
}

/// Logging function.
fn log(string: &[&str]) {
    unimplemented!();
}

/// Version numbering scheme.
/// See http://en.wikipedia.org/wiki/Software_versioning
pub struct Version {
    /// significant changes
    major: i32,
    /// incremental changes
    minor: i32,
    /// bug fixes
    revision: i32,
}

/// Current version.
/// Version of Box2D, LiquidFun is based upon.
pub const VERSION: Version = Version{major: 2, minor: 3, revision: 0};

/// Global variable is used to identify the versions of LiquidFun.
pub const LIQUID_FUN_VERSION: Version = Version{major: 1, minor: 1, revision: 0};
/// String which identifies the current version of LiquidFun.
/// LIQUID_FUN_VERSION_STRING is used by Google developers to identify which
/// applications uploaded to Google Play are using this library. This allows
/// the development team at Google to determine the popularity of the library.
/// How it works: applications that are uploaded to the Google Play Store are
/// scanned for this version string. We track which applications are using it
/// to measure popularity. You are free to remove it (of course) but we would
/// appreciate if you left it in.
pub const LIQUID_FUN_VERSION_STRING: &'static str = "LiquidFun 2.3.0";

fn alloc_default(size: usize, _: *mut libc::c_void) -> *mut libc::c_void {
    unsafe{libc::malloc(size)}
}

// Default implementation of settings::AllocFunction.
fn free_default(mem: *mut libc::c_void, _: *mut libc::c_void) {
    unsafe{libc::free(mem);}
}

static mut NUM_ALLOCS: i32 = 0;

// Initialize default allocator.
pub static mut ALLOC_CALLBACK: AllocFunction = alloc_default;
pub static mut FREE_CALLBACK: FreeFunction = free_default;
pub static mut CALLBACK_DATA: *mut libc::c_void = 0 as *mut libc::c_void;
