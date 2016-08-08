extern crate std;

use std::ops::{Add, Sub, Div, AddAssign, SubAssign, MulAssign, Neg};
use std::option::Option;

/// This function is used to ensure that a floating point number is not a NaN or infinity.
#[inline]
fn is_valid(x: f32) -> bool {
    x.is_finite()
}

/// This is a approximate yet fast inverse square-root.
#[inline]
fn inv_sqrt(x: f32) -> f32 {
    unsafe{
        let xhalf = 0.5f32 * x;
        let i: i32 = std::mem::transmute(x);
        let j: i32 = 0x5f3759dfi32 - (i >> 1i32);
        let y: f32 = std::mem::transmute(j);
        y * (1.5f32 - xhalf * y * y)
    }
}

/// A 2D column vecotr.
#[derive(Clone, Copy)]
pub struct Vec2 {
    x: f32,
    y: f32
}

impl Vec2 {
    /// Default constructor does nothing (for performance).
    pub fn new() -> Vec2 {
        Vec2{x: 0.0f32, y: 0.0f32}
    }

    /// Construct using coordinates.
    pub fn new_with_coordinates(x: f32, y: f32) -> Vec2 {
        Vec2{x: x, y: y}
    }

    /// set this vecotr to all zeros.
    pub fn set_zero(&mut self) {
        self.x = 0.0f32;
        self.y = 0.0f32;
    }

    /// Set this vecotr to some specified coordinates.
    pub fn set(&mut self, x: f32, y: f32) {
        self.x = x;
        self.y = y;
    }

    /// Get the length of this vecor (the norm).
    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    /// Get the length squared. For performance, use this instead of
    /// Vec2::length (if possible).
    pub fn length_squared(&self) -> f32 {
        self.x * self.x + self.y * self.y
    }

    /// Comvert this vector into a unit vector. Returns the length.
    pub fn normalize(&mut self) -> f32 {
        let length = self.length();
        if length < std::f32::EPSILON {
            return 0.0f32;
        }
        let inv_length = 1.0f32 / length;
        self.x *= inv_length;
        self.y *= inv_length;

        length
    }

    /// Does this vector contain finite coordinates?
    pub fn is_valid(&self) -> bool {
        is_valid(self.x) && is_valid(self.y)
    }

    /// Get the skew vector such that dot(skew_vec, other) == cross(vec, other)
    pub fn skew(&self) -> Vec2 {
        Vec2{x: -self.y, y: self.x}
    }
}

/// Useful constant
const VEC2_ZERO: Vec2 = Vec2{x: 0.0f32, y: 0.0f32};

/// Add two vectors component-wise.
impl Add for Vec2 {
    type Output = Vec2;
    #[inline]
    fn add(self, v: Vec2) -> Vec2 {
        Vec2::new_with_coordinates(self.x + v.x, self.y + v.y)
    }
}

/// Add a float to a vector.
impl Add<f32> for Vec2 {
    type Output = Vec2;
    #[inline]
    fn add(self, f: f32) -> Vec2 {
        Vec2::new_with_coordinates(self.x + f, self.y + f)
    }
}

/// Subtract two vectors component-wise.
impl Sub for Vec2 {
    type Output = Vec2;
    #[inline]
    fn sub(self, v: Vec2) -> Vec2 {
        Vec2::new_with_coordinates(self.x - v.x, self.y - v.y)
    }
}

/// substract a float from a vector.
impl Sub<f32> for Vec2 {
    type Output = Vec2;
    #[inline]
    fn sub(self, f: f32) -> Vec2 {
        Vec2::new_with_coordinates(self.x - f, self.y - f)
    }
}

/// Multiply a float with a vector.
impl std::ops::Mul<f32> for Vec2 {
    type Output = Vec2;
    #[inline]
    fn mul(self, f: f32) -> Vec2 {
        Vec2::new_with_coordinates(self.x * f, self.y * f)
    }
}

impl std::ops::Mul<Vec2> for f32 {
    type Output = Vec2;
    #[inline]
    fn mul(self, v: Vec2) -> Vec2 {
        Vec2::new_with_coordinates(self * v.x, self * v.y)
    }
}

/// Divide a vector by a float.
impl Div<f32> for Vec2 {
    type Output = Vec2;
    #[inline]
    fn div(self, f: f32) -> Vec2 {
        Vec2::new_with_coordinates(self.x / f, self.y / f)
    }
}

/// Negate this vector.
impl Neg for Vec2 {
    type Output = Vec2;
    fn neg(self) -> Vec2 {
        Vec2::new_with_coordinates(-self.x, -self.y)
    }
}

/// Add a vector to this vector.
impl AddAssign for Vec2 {
    fn add_assign(&mut self, rhs: Vec2) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

/// Subtract a vecgor from this vector.
impl SubAssign for Vec2 {
    fn sub_assign(&mut self, rhs: Vec2) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

/// Multiply this vecor by a scalar.
impl MulAssign<f32> for Vec2 {
    fn mul_assign(&mut self, f: f32) {
        self.x *= f;
        self.y *= f;
    }
}

/// A 3D column vector with 3 elements.
#[derive(Clone, Copy)]
pub struct Vec3 {
    x: f32,
    y: f32,
    z: f32
}

impl Vec3 {
    /// Default construcor does nothing (for performance).
    pub fn new() -> Vec3 {
        Vec3{x: 0.0f32, y: 0.0f32, z: 0.0f32}
    }

    /// Construct using coordinates.
    pub fn new_with_coordinates(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3{x: x, y: y, z: z}
    }

    /// Set this vecotr to all zeros.
    pub fn set_zero(&mut self) {
        self.x = 0.0f32;
        self.y = 0.0f32;
        self.z = 0.0f32;
    }

    /// Set this vecotr to some specified coordinates.
    pub fn set(&mut self, x: f32, y: f32, z:f32) {
        self.x = x;
        self.y = y;
        self.z = z;
    }

    /// Get the length of this vector (the norm).
    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Convert this vector into a unit vector. Returns the length.
    pub fn normalize(&mut self) -> f32 {
        let length = self.length();
        if length < std::f32::EPSILON {
            return 0.0f32;
        }
        let inv_length = 1.0f32 / length;
        self.x *= inv_length;
        self.y *= inv_length;
        self.z *= inv_length;

        length
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    #[inline]
    fn add(self, v: Vec3) -> Vec3 {
        Vec3::new_with_coordinates(
            self.x + v.x,
            self.y + v.y,
            self.z + v.z
        )
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    #[inline]
    fn sub(self, v: Vec3) -> Vec3 {
        Vec3::new_with_coordinates(
            self.x - v.x,
            self.y - v.y,
            self.z - v.z
        )
    }
}

impl std::ops::Mul<Vec3> for f32 {
    type Output = Vec3;
    #[inline]
    fn mul(self, v: Vec3) -> Vec3 {
        Vec3::new_with_coordinates(self * v.x, self * v.y, self * v.z)
    }
}

/// Negate this vecotr.
impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3::new_with_coordinates(-self.x, -self.y, -self.z)
    }
}

/// Add a vector to this vector.
impl AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Vec3) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

/// Subtract a vector from this vector.
impl SubAssign for Vec3 {
    fn sub_assign(&mut self, rhs: Vec3) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

/// Multiply this vector by a scalar.
impl MulAssign<f32> for Vec3 {
    fn mul_assign(&mut self, f: f32) {
        self.x *= f;
        self.y *= f;
        self.z *= f;
    }
}

/// A 4D column vector with 4 elements.
#[derive(Clone, Copy)]
struct Vec4 {
    x: f32,
    y: f32,
    z: f32,
    w: f32
}

impl Vec4 {
    /// Default constructor does nothing (for performance).
    pub fn new() -> Vec4 {
        Vec4{x: 0.0f32, y: 0.0f32, z: 0.0f32, w: 0.0f32}
    }

    /// construct using coordinates.
    pub fn new_with_coordinates(x: f32, y: f32, z: f32, w: f32) -> Vec4 {
        Vec4{x: x, y: y, z: z, w: w}
    }
}

/// A 2-by-2 matrix. Stored in column-major order.
#[derive(Clone, Copy)]
pub struct Mat22 {
    ex: Vec2,
    ey: Vec2
}

impl Mat22 {
    /// The default constructor does nothing (for performance).
    pub fn new() -> Mat22 {
        Mat22{ex: Vec2::new(), ey: Vec2::new()}
    }

    /// Construct this matrix using colunns.
    pub fn new_with_columns(c1: &Vec2, c2: &Vec2) -> Mat22 {
        Mat22{ex: c1.clone(), ey: c2.clone()}
    }

    /// Initialize this matrix using columns.
    pub fn set(&mut self, c1: &Vec2, c2:&Vec2) {
        self.ex = c1.clone();
        self.ey = c2.clone();
    }

    /// Set this to the identity matrix.
    pub fn set_identity(&mut self) {
        self.ex.x = 1.0f32; self.ey.y = 0.0f32;
        self.ex.y = 0.0f32; self.ey.y = 1.0f32;
    }

    /// Set this matrix to all zeros.
    pub fn set_zero(&mut self) {
        self.ex.x = 0.0f32; self.ey.x = 0.0f32;
        self.ex.y = 0.0f32; self.ey.y = 0.0f32;
    }

    /// Returns None when Det(self) == 0.0f32
    pub fn get_inverse(&self) -> Option<Mat22> {
        let mut m = Mat22::new();
        let a = self.ex.x; let b = self.ey.x;
        let c = self.ex.y; let d = self.ey.y;
        let det = a * d - b * c;
        if det != 0.0f32 {
            m.ex.x =  det * d; m.ey.x = -det * b;
            m.ex.y = -det * c; m.ey.y =  det * a;
            Some(m)
        } else {
            None
        }
    }

    /// Solve A * x = b, where b is a column vector. This is more efficient
    /// than computing the inverse in one-shot cases.
    pub fn solve(&self, b: Vec2) -> Option<Vec2> {
        let a11 = self.ex.x; let a12 = self.ey.x;
        let a21 = self.ex.y; let a22 = self.ey.y;
        let det = a11 * a22 - a12 * a21;
        if det != 0.0f32 {
            Some(Vec2{
                x: (a22 * b.x - a12 * b.y) / det,
                y: (a11 * b.y - a21 * b.x) / det
            })
        } else {
            None
        }
    }
}

impl Add for Mat22 {
    type Output = Mat22;
    #[inline]
    fn add(self, rhs: Mat22) -> Mat22 {
        let c1 = self.ex + rhs.ex;
        let c2 = self.ey + rhs.ey;
        Mat22::new_with_columns(&c1, &c2)
    }
}

/// A 3-by-3 matrix. Stored in column-major order.
#[derive(Clone, Copy)]
pub struct Mat33 {
    ex: Vec3,
    ey: Vec3,
    ez: Vec3
}

impl Mat33 {
    /// The default constructor does nothing (for performance).
    pub fn new() -> Mat33 {
        Mat33{
            ex: Vec3::new(),
            ey: Vec3::new(),
            ez: Vec3::new()
        }
    }

    /// Construct this matrix using columns.
    pub fn new_with_columns(c1: &Vec3, c2: &Vec3, c3: &Vec3) -> Mat33 {
        Mat33{
            ex: c1.clone(),
            ey: c2.clone(),
            ez: c3.clone()
        }
    }

    /// Set this matrix to all zeros.
    pub fn set_zero(&mut self) {
        self.ex.set_zero();
        self.ey.set_zero();
        self.ez.set_zero();
    }

    /// Solve A * x = b, where b is a column vector. This is more efficient
    /// than computing the inverse in one-shot cases.
    pub fn solve33(&self, b: &Vec3) -> Vec3 {
        let mut det = Vec3::dot(&self.ex, &Vec3::cross(&self.ey, &self.ez));
        if det != 0.0f32 {
            det = 1.0f32 / det;
        }
        let mut x = Vec3::new();
        x.x = det * Vec3::dot(b, &Vec3::cross(&self.ey, &self.ez));
        x.y = det * Vec3::dot(&self.ex, &Vec3::cross(b, &self.ez));
        x.z = det * Vec3::dot(&self.ex, &Vec3::cross(&self.ey, b));

        x
    }

    /// Solve A * x = b, where b is a column vector. this is more efficient
    /// than computing the inverse in one-shot cases. Solve only the upper
    /// 2-by-2 matrix equation.
    pub fn solve22(&self, b: &Vec2) -> Vec2 {
        let (a11, a12) = (self.ex.x, self.ey.x);
        let (a21, a22) = (self.ex.y, self.ey.y);
        let mut det = a11 * a22 - a12 * a21;
        if det != 0.0f32 {
            det = 1.0f32 / det;
        }
        let mut x = Vec2::new();
        x.x = det * (a22 * b.x - a12 * b.y);
        x.y = det * (a11 * b.y - a21 * b.x);

        x
    }

    /// Get the inverse of this matrix as a 2-by-2.
    /// Returns the zero matrix if singular.
    pub fn get_inverse22(&self, m: &mut Mat33) {
        let (a, b) = (self.ex.x, self.ey.x);
        let (c, d) = (self.ex.y, self.ey.y);
        let mut det = a * d - b * c;
        if det != 0.0f32 {
            det = 1.0f32 / det;
        }
        m.ex.x =  det * d; m.ey.x = -det * b; m.ex.z = 0.0f32;
        m.ex.y = -det * c; m.ey.y =  det * a; m.ey.z = 0.0f32;
        m.ez.x = 0.0f32; m.ez.y = 0.0f32; m.ez.z = 0.0f32;
    }

    /// Get the symmetric inverse of this matrix as a 3-by-3.
    /// Returns the zero matrix if singular.
    pub fn get_sym_inverse33(&self, m: &mut Mat33) {
        let mut det = Vec3::dot(&self.ex, &Vec3::cross(&self.ey, &self.ez));
        if det != 0.0f32 {
            det = 1.0f32 / det;
        }

        let a11 = self.ex.x; let a12 = self.ey.x; let a13 = self.ez.x;
        let a22 = self.ey.y; let a23 = self.ez.y;
        let a33 = self.ez.z;

        m.ex.x = det * (a22 * a33 - a23 * a23);
        m.ex.y = det * (a13 * a23 - a12 * a33);
        m.ex.z = det * (a12 * a23 - a13 * a22);

        m.ey.x = m.ex.y;
        m.ey.y = det * (a11 * a33 - a13 * a13);
        m.ey.z = det * (a13 * a12 - a11 * a23);

        m.ez.x = m.ex.z;
        m.ez.y = m.ey.z;
        m.ez.z = det * (a11 * a22 - a12 * a12);
    }
}

/// rotation
#[derive(Clone, Copy)]
pub struct Rot {
    /// Sine
    s: f32,
    /// cosine
    c: f32
}

impl Rot {
    pub fn new() -> Rot {
        Rot{s: 0.0f32, c: 1.0f32}
    }

    /// Initialize from an angel in radians
    pub fn new_from_angle(angle: f32) -> Rot {
        Rot{s: angle.sin(), c: angle.cos()}
    }

    /// Set using an angle in radians.
    pub fn set(&mut self, angle: f32) {
        self.s = angle.sin();
        self.c = angle.cos();
    }

    /// Set to the identity rotation
    pub fn set_identity(&mut self) {
        self.s = 0.0f32;
        self.c = 1.0f32;
    }

    /// Get the angle in radians
    pub fn get_angle(&self) -> f32 {
        self.s.atan2(self.c)
    }

    /// Get the x-axis
    pub fn get_x_axis(&self) -> Vec2 {
        Vec2{x: self.c, y: self.s}
    }

    /// Get the y-axis
    pub fn get_y_axis(&self) -> Vec2 {
        Vec2{x: -self.s, y: self.c}
    }
}

/// A transform contains translation and rotation. It is used to represent
/// the position and orientation of rigid frames.
#[derive(Clone, Copy)]
pub struct Transform {
    p: Vec2,
    q: Rot
}

impl Transform {
    /// The default constructor does nothing
    pub fn new() -> Transform {
        Transform{p: Vec2::new(), q: Rot::new()}
    }

    /// Initialize using a position vector and a rotaton.
    pub fn new_with(position: &Vec2, rotation: &Rot) -> Transform {
        Transform{p: position.clone(), q: rotation.clone()}
    }

    /// Set this to the identity transform.
    pub fn set_identity(&mut self) {
        self.p.set_zero();
        self.q.set_identity();
    }

    /// Set this based on the position and angle.
    pub fn set(&mut self, position: &Vec2, angle: f32) {
        self.p = position.clone();
        self.q.set(angle);
    }
}

#[cfg(LIQUIDFUN_EXTERNAL_LANGUAGE_API)]
impl Transform {
    /// Get x-coordinate of p.
    pub fn get_position_x(&self) -> f32 {
        self.p.x
    }

    /// Get y-coordinate of p.
    pub fn get_position_y(&self) -> f32 {
        self.p.y
    }

    /// Get sine-component of q.
    pub fn get_rotation_sin(&self) -> f32 {
        self.q.s
    }

    /// Get cosine-component of q.
    pub fn get_rotation_cos(&self) -> f32 {
        self.q.c
    }
}

/// This descrives the motion of a body/shape for TOI computation.
/// Shapes are defined with respect to the body origin, which may
/// no coincide with the center of mass. However, to support dynamics
/// we must interpolate the center of mass position.
pub struct Sweep {
    /// local center of mass position
    local_center: Vec2,
    /// center world positions
    c0: Vec2, c: Vec2,
    /// world angles
    a0: f32, a: f32,
    /// Fraction of the current time step in the range [0,1]
    /// `c0` and `a0` are the positions at alpha0
    alpha0: f32
}

impl Sweep {
    /// Get the interpoolated transform at a specific time.
    /// `beta` is a factor in [0,1], where 0 indicates alpha0.
    #[inline]
    pub fn get_transform(&mut self, xf: &mut Transform, beta: f32) {
        xf.p = (1.0f32 - beta) * self.c0 + beta * self.c;
        xf.q.set((1.0f32 - beta) * self.a0 + beta * self.a);

        // Shift to origin
        xf.p -= Rot::mul(&xf.q, &self.local_center);
    }

    /// Advance the sweep forward, yielding a new initial state.
    /// `alpha` the new initial time.
    #[inline]
    pub fn advane(&mut self, alpha: f32) {
        assert!(self.alpha0 < 1.0f32);
        let beta = (alpha - self.alpha0) / (1.0f32 - self.alpha0);
        self.c0 += beta * (self.c - self.c0);
        self.a0 += beta * (self.a - self.a0);
        self.alpha0 = alpha;
    }

    /// Normalize the angles.
    #[inline]
    pub fn normalize(&mut self) {
        let two_pi = 2.0f32 * std::f32::consts::PI;
        let d = two_pi * (self.a0 / two_pi).floor();
        self.a0 -= d;
        self.a -= d;
    }
}

trait Dot<Lhs = Self, Rhs = Self> {
    type Output;
    fn dot(Lhs, Rhs) -> Self::Output;
}

impl<'a, 'b> Dot<&'a Vec2, &'b Vec2> for Vec2 {
    type Output = f32;
    /// Perform th dot product on two vectors.
    #[inline]
    fn dot(a: &'a Vec2, b: &'b Vec2) -> f32 {
        a.x * b.x + a.y * b.y
    }
}

trait Cross<Lhs = Self, Rhs = Self> {
    type Output;
    fn cross(Lhs, Rhs) -> Self::Output;
}

impl<'a, 'b> Cross<&'a Vec2, &'b Vec2> for Vec2 {
    type Output = f32;
    /// Perform the cross product on two vectors. In 2D this produces a scalar.
    #[inline]
    fn cross(a: &'a Vec2, b: &'b Vec2) -> f32 {
        a.x * b.y - a.y * b.x
    }
}

impl<'a> Cross<&'a Vec2, f32> for Vec2 {
    type Output = Vec2;
    /// Perform the cross product on a vector and a scalar. In 2D this pruduces
    /// a vector.
    #[inline]
    fn cross(a: &'a Vec2, s: f32) -> Vec2 {
        Vec2::new_with_coordinates(s * a.y, -s * a.x)
    }
}

impl<'a> Cross<f32, &'a Vec2> for f32 {
    type Output = Vec2;
    /// Perform the cross product on a scalar and a vector. In 2D this produces
    /// a vector.
    #[inline]
    fn cross(s: f32, a: &'a Vec2) -> Vec2 {
        Vec2::new_with_coordinates(-s * a.y, s * a.x)
    }
}

trait Mul<Lhs = Self, Rhs = Self> {
    type Output;
    fn mul(Lhs, Rhs) -> Self::Output;
}

impl<'a, 'b> Mul<&'a Mat22, &'b Vec2> for Mat22 {
    type Output = Vec2;
    /// Multiply a matrix times a vector. If a rotation    matrix is provided,
    /// then this transforms the vector from one frame to another.
    #[inline]
    fn mul(a: &'a Mat22, v: &'b Vec2) -> Vec2 {
        Vec2::new_with_coordinates(
            a.ex.x * v.x + a.ey.x * v.y,
            a.ex.y * v.x + a.ey.y * v.y
        )
    }
}

trait MulT<Lhs = Self, Rhs = Self> {
    type Output;
    fn mul_t(Lhs, Rhs) -> Self::Output;
}

impl<'a, 'b> MulT<&'a Mat22, &'b Vec2> for Mat22 {
    type Output = Vec2;
    /// Multiply a matrix transpose times a vector. IF a rotation matrix is provided,
    /// then this transforms the vector from one frame to another (inverse transform).
    #[inline]
    fn mul_t(a: &Mat22, v: &Vec2) -> Vec2 {
        Vec2::new_with_coordinates(
            Vec2::dot(v, &a.ex),
            Vec2::dot(v, &a.ey)
        )
    }
}

#[inline]
pub fn distance(a: &Vec2, b: &Vec2) -> f32 {
    (*a - *b).length()
}

#[inline]
pub fn distance_squared(a: &Vec2, b: &Vec2) -> f32 {
    let c = *a - *b;
    Vec2::dot(&c, &c)
}

impl<'a, 'b> Dot<&'a Vec3, &'b Vec3> for Vec3 {
    type Output = f32;
    /// Perform the dot product on two vectors.
    #[inline]
    fn dot(a: &'a Vec3, b: &'b Vec3) -> f32 {
        a.x * b.x + a.y * b.y + a.z * b.z
    }
}

impl<'a, 'b> Cross<&'a Vec3, &'b Vec3> for Vec3 {
    type Output = Vec3;
    /// Perform the cross product on two vectors.
    #[inline]
    fn cross(a: &'a Vec3, b: &'b Vec3) -> Vec3 {
        Vec3::new_with_coordinates(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        )
    }
}

impl<'a, 'b> Mul<&'a Mat22, &'b Mat22> for Mat22 {
    type Output = Mat22;
    /// a * b
    #[inline]
    fn mul(a: &'a Mat22, b: &'b Mat22) -> Mat22 {
        Mat22::new_with_columns(
            &Mat22::mul(a, &b.ex),
            &Mat22::mul(a, &b.ey)
        )
    }
}

impl<'a, 'b> MulT<&'a Mat22, &'b Mat22> for Mat22 {
    type Output = Mat22;
    /// a^T * b
    #[inline]
    fn mul_t(a: &'a Mat22, b: &'b Mat22) -> Mat22 {
        let c1 = Vec2::new_with_coordinates(
            Vec2::dot(&a.ex, &b.ex),
            Vec2::dot(&a.ey, &b.ey)
        );
        let c2 = Vec2::new_with_coordinates(
            Vec2::dot(&a.ex, &b.ey),
            Vec2::dot(&a.ey, &b.ey)
        );
        Mat22::new_with_columns(&c1, &c2)
    }
}

impl<'a, 'b> Mul<&'a Mat33, &'b Vec3> for Mat33 {
    type Output = Vec3;
    /// Multiply a matrix times a vector.
    #[inline]
    fn mul(a: &'a Mat33, v: &'b Vec3) -> Vec3 {
        v.x * a.ex + v.y * a.ey + v.z * a.ez
    }
}

/// Multiply a matrix times a vector.
#[inline]
pub fn mul22(a: &Mat22, v: &Vec2) -> Vec2 {
    Vec2::new_with_coordinates(
        a.ex.x * v.x + a.ey.x * v.y,
        a.ex.y * v.x + a.ey.y * v.y
    )
}

impl<'a, 'b> Mul<&'a Rot, &'b Rot> for Rot {
    type Output = Rot;
    /// Multiply two rotations: q * r
    #[inline]
    fn mul(q: &'a Rot, r: &'b Rot) -> Rot {
        // [qc -qs] * [rc -rs] = [qc*rc-qs*rs -qc*rs-qs*rc]
        // [qs  qc] * [rs  rc]   [qs*rc+qc*rs -qs*rs+qc*rc]
        // s = qs * rc + qc * rs
        // c = qc * rc - qs * rs
        let mut qr = Rot::new();
        qr.s = q.s * r.c + q.c * r.s;
        qr.c = q.c * r.c - q.s * r.s;
        qr
    }
}

impl<'a, 'b> MulT<&'a Rot, &'b Rot> for Rot {
    type Output = Rot;
    /// Transpose multiply two rotations: qT * r
    #[inline]
    fn mul_t(q: &'a Rot, r: &'b Rot) -> Rot {
        // [ qc qs] * [rc -rs] = [qc*rc+qs*rs -qc*rs+qs*rc]
        // [-qs qc] * [rs  rc]   [-qs*rc+qc*rs qs*rs+qc*rc]
        let mut qr = Rot::new();
        qr.s = q.c * r.s - q.s * r.c;
        qr.c = q.c * r.c + q.s * r.s;
        qr
    }
}

impl<'a, 'b> Mul<&'a Rot, &'b Vec2> for Rot {
    type Output = Vec2;
    /// Rotate a vector
    #[inline]
    fn mul(q: &'a Rot, v: &'b Vec2) -> Vec2 {
        Vec2::new_with_coordinates(
            q.c * v.x - q.s * v.y,
            q.s * v.x + q.c * v.y
        )
    }
}

impl<'a, 'b> MulT<&'a Rot, &'b Vec2> for Rot {
    type Output = Vec2;
    /// Inverse rotate a vector
    #[inline]
    fn mul_t(q: &'a Rot, v: &'b Vec2) -> Vec2 {
        Vec2::new_with_coordinates(
            q.c * v.x + q.s * v.y,
            -q.s * v.x + q.c * v.y
        )
    }
}

impl<'a, 'b> Mul<&'a Transform, &'b Vec2> for Transform {
    type Output = Vec2;
    #[inline]
    fn mul(t: &'a Transform, v: &'b Vec2) -> Vec2 {
        Vec2::new_with_coordinates(
            (t.q.c * v.x - t.q.s * v.y) + t.p.x,
            (t.q.s * v.x + t.q.c * v.y) + t.p.y
        )
    }
}

impl<'a, 'b> MulT<&'a Transform, &'b Vec2> for Transform {
    type Output = Vec2;
    #[inline]
    fn mul_t(t: &'a Transform, v: &'b Vec2) -> Vec2 {
        let px = v.x - t.p.x;
        let py = v.y - t.p.y;
        Vec2::new_with_coordinates(
            t.q.c * px + t.q.s * py,
            -t.q.s * px + t.q.c * py
        )
    }
}

impl<'a, 'b> Mul<&'a Transform, &'b Transform> for Transform {
    type Output = Transform;
    // v2 = a.q.rot(b.q.rot(v1) + b.p) + a.p
    //    = (a.q * b.q).rot(v1) + q.q.rot(b.p) + a.p
    #[inline]
    fn mul(a: &'a Transform, b: &'b Transform) -> Transform {
        let mut c = Transform::new();
        c.q = Rot::mul(&a.q, &b.q);
        c.p = Rot::mul(&a.q, &b.p) + a.p;
        c
    }
}

impl<'a, 'b> MulT<&'a Transform, &'b Transform> for Transform {
    type Output = Transform;
    // v2 = a.q' * (b.q * v1 + b.p - a.p)
    //    = a.q' * b.q * v1 + a.q' * (b.p - a.p)
    #[inline]
    fn mul_t(a: &'a Transform, b: &'b Transform) -> Transform {
        let mut c = Transform::new();
        c.q = Rot::mul(&a.q, &b.q);
        c.p = Rot::mul(&a.q, &(b.p - a.p));
        c
    }
}

trait Abs {
    fn abs(&Self) -> Self;
}

impl Abs for Vec2 {
    fn abs(a: &Vec2) -> Vec2 {
        Vec2::new_with_coordinates(
            f32::abs(a.x),
            f32::abs(a.y)
        )
    }
}

impl Abs for Mat22 {
    fn abs(a: &Mat22) -> Mat22 {
        Mat22::new_with_columns(
            &Vec2::abs(&a.ex),
            &Vec2::abs(&a.ey)
        )
    }
}

trait Min {
    fn min(&Self, &Self) -> Self;
}

impl Min for Vec2 {
    #[inline]
    fn min(a: &Vec2, b: &Vec2) -> Vec2 {
        Vec2::new_with_coordinates(
            f32::min(a.x, b.x),
            f32::min(a.y, b.y)
        )
    }
}

trait Max {
    fn max(&Self, &Self) -> Self;
}

impl Max for Vec2 {
    #[inline]
    fn max(a: &Vec2, b: &Vec2) -> Vec2 {
        Vec2::new_with_coordinates(
            f32::max(a.x, b.x),
            f32::max(a.y, b.y)
        )
    }
}

trait Clamp {
    fn clamp(&Self, &Self, &Self) -> Self;
}

impl Clamp for Vec2 {
    #[inline]
    fn clamp(a: &Vec2, low: &Vec2, high: &Vec2) -> Vec2 {
        Vec2::max(low, &Vec2::min(a, high))
    }
}

#[inline]
pub fn swap<T>(a: &mut T, b: &mut T) {
    std::mem::swap(a, b);
}

/// "Next Largest Power of 2
/// Given a binary integer value x, the next largest power of 2 can be computed by a SWAR algorithm
/// that recursively "folds" the upper bits into the lower bits. This process yields a bit vector with
/// the same most significant 1 as x, but all 1's below it. Adding 1 to that value yields the next
/// largest power of 2. FOr a 32-bit value:"
#[inline]
fn next_power_of_two(x: u32) -> u32 {
    let mut y = x;
    y |= y >> 1;
    y |= y >> 2;
    y |= y >> 4;
    y |= y >> 8;
    y |= y >> 16;
    y + 1
}

#[inline]
fn is_power_of_two(x: u32) -> bool {
    x > 0 && (x & (x - 1)) == 0
}
