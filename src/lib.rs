//! Common algorithms for rectangles (clipping, transformation, quadtree, rect packing, etc.)

#![warn(missing_docs)]
#![forbid(unsafe_code)]

use nalgebra::{Matrix3, SimdPartialOrd, Vector2};
use num_traits::{NumAssign, Zero};
use std::fmt::Debug;

pub mod pack;
pub mod quadtree;

/// Arbitrary number.
pub trait Number: NumAssign + 'static + Clone + PartialEq + Debug + PartialOrd + Copy {}

impl<T> Number for T where T: NumAssign + 'static + Clone + PartialEq + Debug + PartialOrd + Copy {}

/// A rectangle defined by position and size.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Rect<T> {
    /// Position of the rectangle.
    pub position: Vector2<T>,
    /// Size of the rectangle, where X - width, Y - height.
    pub size: Vector2<T>,
}

impl<T> Default for Rect<T>
where
    T: Number,
{
    fn default() -> Self {
        Self {
            position: Vector2::new(Zero::zero(), Zero::zero()),
            size: Vector2::new(Zero::zero(), Zero::zero()),
        }
    }
}

/// A version of [Rect] that is optionally None.
/// This simplifies the process of creating a bounding rect from a series of points,
/// as it can start as None and then build an initial rect from the first point.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct OptionRect<T>(Option<Rect<T>>);

impl<T> Default for OptionRect<T> {
    fn default() -> Self {
        Self(None)
    }
}

impl<T> OptionRect<T>
where
    T: Number + SimdPartialOrd,
{
    /// Clip the rectangle to the given bounds.
    #[inline]
    pub fn clip(&mut self, bounds: Rect<T>) {
        if let Some(rect) = self.0 {
            *self = rect.clip_by(bounds);
        }
    }
    /// Extends the rectangle so it will contain the given point.
    #[inline]
    pub fn push(&mut self, p: Vector2<T>) {
        if let Some(rect) = &mut self.0 {
            rect.push(p);
        } else {
            self.0 = Some(Rect::new(p.x, p.y, T::zero(), T::zero()));
        }
    }
    /// Extends the rectangle so it will contain the other rectangle.
    ///
    /// # Notes
    ///
    /// To build a bounding rectangle, initialize an OptionRect to default.
    ///
    /// ```
    /// # use nalgebra::Vector2;
    /// # use rectutils::Rect;
    ///
    /// let vertices = [Vector2::new(1.0, 2.0), Vector2::new(-3.0, 5.0)];
    ///
    /// let mut bounding_rect = OptionRect::default();
    ///
    /// for &v in &vertices {
    ///     bounding_rect.push(v);
    /// }
    ///
    /// // So long as vertices is not empty, bounding_rect is guaranteed to be some.
    /// let bounding_rect = bounding_rect.unwrap();
    /// ```
    #[inline]
    pub fn extend_to_contain(&mut self, other: Rect<T>) {
        if let Some(rect) = &mut self.0 {
            rect.extend_to_contain(other);
        } else {
            self.0 = Some(other);
        }
    }
}

impl<T> From<Rect<T>> for OptionRect<T> {
    fn from(source: Rect<T>) -> Self {
        Self(Some(source))
    }
}
impl<T> From<Option<Rect<T>>> for OptionRect<T> {
    fn from(source: Option<Rect<T>>) -> Self {
        Self(source)
    }
}
impl<T> std::ops::Deref for OptionRect<T> {
    type Target = Option<Rect<T>>;
    fn deref(&self) -> &Option<Rect<T>> {
        &self.0
    }
}
impl<T> std::ops::DerefMut for OptionRect<T> {
    fn deref_mut(&mut self) -> &mut Option<Rect<T>> {
        &mut self.0
    }
}

impl<T> Rect<T>
where
    T: Number,
{
    /// Creates a new rectangle from X, Y, width, height.
    #[inline]
    pub fn new(x: T, y: T, w: T, h: T) -> Self {
        Self {
            position: Vector2::new(x, y),
            size: Vector2::new(w, h),
        }
    }

    /// Create a new rectangle from two diagonally opposite corner points.
    /// In other words, create the smallest rectangle containing both given points.
    pub fn from_points(p0: Vector2<T>, p1: Vector2<T>) -> Self
    where
        T: SimdPartialOrd,
    {
        let inf = p0.inf(&p1);
        let sup = p0.sup(&p1);
        Self {
            position: inf,
            size: sup - inf,
        }
    }

    /// Sets the new position of the rectangle.
    #[inline]
    pub fn with_position(mut self, position: Vector2<T>) -> Self {
        self.position = position;
        self
    }

    /// Sets the new size of the rectangle.
    #[inline]
    pub fn with_size(mut self, size: Vector2<T>) -> Self {
        self.size = size;
        self
    }

    /// Inflates the rectangle by the given amounts. It offsets the rectangle by `(-dw, -dh)` and
    /// increases its size by `(2 * dw, 2 * dh)`.
    #[inline]
    #[must_use = "this method creates new instance of rect"]
    pub fn inflate(&self, dw: T, dh: T) -> Self {
        Self {
            position: Vector2::new(self.position.x - dw, self.position.y - dh),
            size: Vector2::new(self.size.x + dw + dw, self.size.y + dh + dh),
        }
    }

    /// Deflates the rectangle by the given amounts. It offsets the rectangle by `(dw, dh)` and
    /// decreases its size by `(2 * dw, 2 * dh)`.
    #[inline]
    #[must_use = "this method creates new instance of rect"]
    pub fn deflate(&self, dw: T, dh: T) -> Self {
        Self {
            position: Vector2::new(self.position.x + dw, self.position.y + dh),
            size: Vector2::new(self.size.x - (dw + dw), self.size.y - (dh + dh)),
        }
    }

    /// Checks if the given point lies within the bounds of the rectangle.
    #[inline]
    pub fn contains(&self, pt: Vector2<T>) -> bool {
        pt.x >= self.position.x
            && pt.x <= self.position.x + self.size.x
            && pt.y >= self.position.y
            && pt.y <= self.position.y + self.size.y
    }

    /// Returns center point of the rectangle.
    #[inline]
    pub fn center(&self) -> Vector2<T> {
        let two = T::one() + T::one();
        self.position + Vector2::new(self.size.x / two, self.size.y / two)
    }

    /// Extends the rectangle to contain the given point.
    ///
    /// # Notes
    ///
    /// To build bounding rectangle you should use [OptionRect].
    #[inline]
    pub fn push(&mut self, p: Vector2<T>)
    where
        T: SimdPartialOrd,
    {
        let p0 = self.left_top_corner();
        let p1 = self.right_bottom_corner();
        *self = Self::from_points(p.inf(&p0), p.sup(&p1));
    }

    /// Clips the rectangle by some other rectangle and returns a new rectangle that corresponds to
    /// the intersection of both rectangles. If the rectangles does not intersects, the method
    /// returns none.
    #[inline]
    #[must_use = "this method creates new instance of OptionRect"]
    pub fn clip_by(&self, other: Rect<T>) -> OptionRect<T> {
        let mut clipped = *self;

        if other.x() + other.w() < self.x()
            || other.x() > self.x() + self.w()
            || other.y() + other.h() < self.y()
            || other.y() > self.y() + self.h()
        {
            return OptionRect::<T>::default();
        }

        if clipped.position.x < other.position.x {
            clipped.size.x -= other.position.x - clipped.position.x;
            clipped.position.x = other.position.x;
        }

        if clipped.position.y < other.position.y {
            clipped.size.y -= other.position.y - clipped.position.y;
            clipped.position.y = other.position.y;
        }

        let clipped_right_bottom = clipped.right_bottom_corner();
        let other_right_bottom = other.right_bottom_corner();

        if clipped_right_bottom.x > other_right_bottom.x {
            clipped.size.x -= clipped_right_bottom.x - other_right_bottom.x;
        }
        if clipped_right_bottom.y > other_right_bottom.y {
            clipped.size.y -= clipped_right_bottom.y - other_right_bottom.y;
        }

        clipped.into()
    }

    /// Checks if the rectangle intersects with some other rectangle.
    #[inline]
    pub fn intersects(&self, other: Rect<T>) -> bool {
        if other.position.x < self.position.x + self.size.x
            && self.position.x < other.position.x + other.size.x
            && other.position.y < self.position.y + self.size.y
        {
            self.position.y < other.position.y + other.size.y
        } else {
            false
        }
    }

    /// Offsets the given rectangle and returns a new rectangle.
    #[inline]
    #[must_use = "this method creates new instance of rect"]
    pub fn translate(&self, translation: Vector2<T>) -> Self {
        Self {
            position: Vector2::new(
                self.position.x + translation.x,
                self.position.y + translation.y,
            ),
            size: self.size,
        }
    }

    /// Checks if the rectangle intersects a circle represented by a center point and a radius.
    #[inline]
    pub fn intersects_circle(&self, center: Vector2<T>, radius: T) -> bool {
        let r = self.position.x + self.size.x;
        let b = self.position.y + self.size.y;
        // find the closest point to the circle within the rectangle
        let closest_x = if center.x < self.position.x {
            self.position.x
        } else if center.x > r {
            r
        } else {
            center.x
        };
        let closest_y = if center.y < self.position.y {
            self.position.y
        } else if center.y > b {
            b
        } else {
            center.y
        };
        // calculate the distance between the circle's center and this closest point
        let distance_x = center.x - closest_x;
        let distance_y = center.y - closest_y;
        // if the distance is less than the circle's radius, an intersection occurs
        let distance_squared = (distance_x * distance_x) + (distance_y * distance_y);
        distance_squared < (radius * radius)
    }

    /// Extends the rectangle so it will contain the other rectangle.
    #[inline]
    pub fn extend_to_contain(&mut self, other: Rect<T>)
    where
        T: SimdPartialOrd,
    {
        let p0 = self.left_top_corner();
        let p1 = self.right_bottom_corner();
        let o0 = other.left_top_corner();
        let o1 = other.right_bottom_corner();
        *self = Self::from_points(p0.inf(&o0), p1.sup(&o1));
    }

    /// Returns the top left corner of the rectangle.
    #[inline(always)]
    pub fn left_top_corner(&self) -> Vector2<T> {
        self.position
    }

    /// Returns the top right corner of the rectangle.
    #[inline(always)]
    pub fn right_top_corner(&self) -> Vector2<T> {
        Vector2::new(self.position.x + self.size.x, self.position.y)
    }

    /// Returns the bottom right corner of the rectangle.
    #[inline(always)]
    pub fn right_bottom_corner(&self) -> Vector2<T> {
        Vector2::new(self.position.x + self.size.x, self.position.y + self.size.y)
    }

    /// Returns the bottom left corner of the rectangle.
    #[inline(always)]
    pub fn left_bottom_corner(&self) -> Vector2<T> {
        Vector2::new(self.position.x, self.position.y + self.size.y)
    }

    /// Returns width of the rectangle.
    #[inline(always)]
    pub fn w(&self) -> T {
        self.size.x
    }

    /// Returns height of the rectangle.
    #[inline(always)]
    pub fn h(&self) -> T {
        self.size.y
    }

    /// Returns horizontal position of the rectangle.
    #[inline(always)]
    pub fn x(&self) -> T {
        self.position.x
    }

    /// Returns vertical position of the rectangle.
    #[inline(always)]
    pub fn y(&self) -> T {
        self.position.y
    }

    /// Applies an arbitrary affine transformation to the rectangle.
    #[inline]
    #[must_use]
    pub fn transform(&self, matrix: &Matrix3<T>) -> Self {
        let min = self.position;
        let max = self.right_bottom_corner();

        let translation = Vector2::new(matrix[6], matrix[7]);

        let mut transformed_min = translation;
        let mut transformed_max = translation;

        for i in 0..2 {
            for j in 0..2 {
                let a = matrix[(i, j)] * min[j];
                let b = matrix[(i, j)] * max[j];
                if a < b {
                    transformed_min[i] += a;
                    transformed_max[i] += b;
                } else {
                    transformed_min[i] += b;
                    transformed_max[i] += a;
                }
            }
        }

        Self {
            position: transformed_min,
            size: transformed_max - transformed_min,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn intersects1() {
        let rect1 = Rect::new(-1, -2, 4, 6);
        let rect2 = Rect::new(2, 3, 2, 2);
        assert!(rect1.intersects(rect2));
    }
    #[test]
    fn intersects2() {
        let rect1 = Rect::new(0, 0, 4, 6);
        let rect2 = Rect::new(-1, -2, 2, 3);
        assert!(rect1.intersects(rect2));
    }
    #[test]
    fn not_intersects1() {
        let rect1 = Rect::new(-1, -2, 3, 4);
        let rect2 = Rect::new(3, 3, 2, 2);
        assert!(!rect1.intersects(rect2));
    }
    #[test]
    fn not_intersects2() {
        let rect1 = Rect::new(-1, -2, 3, 4);
        let rect2 = Rect::new(2, 1, 2, 2);
        assert!(!rect1.intersects(rect2));
    }
    #[test]
    fn from_points1() {
        let rect = Rect::from_points(Vector2::new(-1, -2), Vector2::new(2, 1));
        assert_eq!(rect, Rect::new(-1, -2, 3, 3));
    }
    #[test]
    fn from_points2() {
        let rect = Rect::from_points(Vector2::new(-1, 1), Vector2::new(2, -2));
        assert_eq!(rect, Rect::new(-1, -2, 3, 3));
    }
    #[test]
    fn rect_extend_to_contain() {
        let mut rect = Rect::new(0.0, 0.0, 1.0, 1.0);

        rect.extend_to_contain(Rect::new(1.0, 1.0, 1.0, 1.0));
        assert_eq!(rect, Rect::new(0.0, 0.0, 2.0, 2.0));

        rect.extend_to_contain(Rect::new(-1.0, -1.0, 1.0, 1.0));
        assert_eq!(rect, Rect::new(-1.0, -1.0, 3.0, 3.0));

        rect.extend_to_contain(Rect::new(10.0, -1.0, 1.0, 15.0));
        assert_eq!(rect, Rect::new(-1.0, -1.0, 12.0, 15.0));
    }
    #[test]
    fn rect_push2() {
        let mut rect = Rect::new(0.0, 0.0, 1.0, 1.0);

        rect.push(Vector2::new(1.0, 1.0));
        assert_eq!(rect, Rect::new(0.0, 0.0, 1.0, 1.0));

        rect.push(Vector2::new(-1.0, -1.0));
        assert_eq!(rect, Rect::new(-1.0, -1.0, 2.0, 2.0));

        rect.push(Vector2::new(10.0, -1.0));
        assert_eq!(rect, Rect::new(-1.0, -1.0, 11.0, 2.0));
    }
    #[test]
    fn option_rect_extend_to_contain() {
        let mut rect = OptionRect::default();

        rect.extend_to_contain(Rect::new(1.0, 1.0, 1.0, 1.0));
        assert_eq!(rect.unwrap(), Rect::new(1.0, 1.0, 1.0, 1.0));

        rect.extend_to_contain(Rect::new(-1.0, -1.0, 1.0, 1.0));
        assert_eq!(rect.unwrap(), Rect::new(-1.0, -1.0, 3.0, 3.0));

        rect.extend_to_contain(Rect::new(10.0, -1.0, 1.0, 15.0));
        assert_eq!(rect.unwrap(), Rect::new(-1.0, -1.0, 12.0, 15.0));
    }
    #[test]
    fn option_rect_push() {
        let mut rect = OptionRect::default();

        rect.push(Vector2::new(1.0, 1.0));
        assert_eq!(rect.unwrap(), Rect::new(1.0, 1.0, 0.0, 0.0));

        rect.push(Vector2::new(-1.0, -1.0));
        assert_eq!(rect.unwrap(), Rect::new(-1.0, -1.0, 2.0, 2.0));

        rect.push(Vector2::new(10.0, -1.0));
        assert_eq!(rect.unwrap(), Rect::new(-1.0, -1.0, 11.0, 2.0));
    }
    #[test]
    fn option_rect_clip() {
        let rect = OptionRect::<i32>::from(Rect::new(0, 0, 10, 10));

        let mut r = rect;
        r.clip(Rect::new(2, 2, 1, 1));
        assert_eq!(r.unwrap(), Rect::new(2, 2, 1, 1));

        let mut r = rect;
        r.clip(Rect::new(0, 0, 15, 15));
        assert_eq!(r.unwrap(), Rect::new(0, 0, 10, 10));

        // When there is no intersection.
        let mut r = OptionRect::default();
        r.clip(Rect::new(0, 0, 10, 10));
        assert!(r.is_none());
        let mut r = rect;
        r.clip(Rect::new(-2, 1, 1, 1));
        assert!(r.is_none());
        let mut r = rect;
        r.clip(Rect::new(11, 1, 1, 1));
        assert!(r.is_none());
        let mut r = rect;
        r.clip(Rect::new(1, -2, 1, 1));
        assert!(r.is_none());
        let mut r = rect;
        r.clip(Rect::new(1, 11, 1, 1));
        assert!(r.is_none());
    }
    #[test]
    fn default_for_rect() {
        assert_eq!(
            Rect::<f32>::default(),
            Rect {
                position: Vector2::new(Zero::zero(), Zero::zero()),
                size: Vector2::new(Zero::zero(), Zero::zero()),
            }
        );
    }

    #[test]
    fn rect_with_position() {
        let rect = Rect::new(0, 0, 1, 1);

        assert_eq!(
            rect.with_position(Vector2::new(1, 1)),
            Rect::new(1, 1, 1, 1)
        );
    }

    #[test]
    fn rect_with_size() {
        let rect = Rect::new(0, 0, 1, 1);

        assert_eq!(
            rect.with_size(Vector2::new(10, 10)),
            Rect::new(0, 0, 10, 10)
        );
    }

    #[test]
    fn rect_inflate() {
        let rect = Rect::new(0, 0, 1, 1);

        assert_eq!(rect.inflate(5, 5), Rect::new(-5, -5, 11, 11));
    }

    #[test]
    fn rect_deflate() {
        let rect = Rect::new(-5, -5, 11, 11);

        assert_eq!(rect.deflate(5, 5), Rect::new(0, 0, 1, 1));
    }

    #[test]
    fn rect_contains() {
        let rect = Rect::new(0, 0, 10, 10);

        assert!(rect.contains(Vector2::new(0, 0)));
        assert!(rect.contains(Vector2::new(0, 10)));
        assert!(rect.contains(Vector2::new(10, 0)));
        assert!(rect.contains(Vector2::new(10, 10)));
        assert!(rect.contains(Vector2::new(5, 5)));

        assert!(!rect.contains(Vector2::new(0, 20)));
    }

    #[test]
    fn rect_center() {
        let rect = Rect::new(0, 0, 10, 10);

        assert_eq!(rect.center(), Vector2::new(5, 5));
    }

    #[test]
    fn rect_push() {
        let mut rect = Rect::new(10, 10, 11, 11);

        rect.push(Vector2::new(0, 0));
        assert_eq!(rect, Rect::new(0, 0, 21, 21));

        rect.push(Vector2::new(0, 20));
        assert_eq!(rect, Rect::new(0, 0, 21, 21));

        rect.push(Vector2::new(20, 20));
        assert_eq!(rect, Rect::new(0, 0, 21, 21));

        rect.push(Vector2::new(30, 30));
        assert_eq!(rect, Rect::new(0, 0, 30, 30));
    }

    #[test]
    fn rect_getters() {
        let rect = Rect::new(0, 0, 1, 1);

        assert_eq!(rect.left_top_corner(), Vector2::new(0, 0));
        assert_eq!(rect.left_bottom_corner(), Vector2::new(0, 1));
        assert_eq!(rect.right_top_corner(), Vector2::new(1, 0));
        assert_eq!(rect.right_bottom_corner(), Vector2::new(1, 1));

        assert_eq!(rect.x(), 0);
        assert_eq!(rect.y(), 0);
        assert_eq!(rect.w(), 1);
        assert_eq!(rect.h(), 1);
    }

    #[test]
    fn rect_clip_by() {
        let rect = Rect::new(0, 0, 10, 10);

        assert_eq!(
            rect.clip_by(Rect::new(2, 2, 1, 1)).unwrap(),
            Rect::new(2, 2, 1, 1)
        );
        assert_eq!(
            rect.clip_by(Rect::new(0, 0, 15, 15)).unwrap(),
            Rect::new(0, 0, 10, 10)
        );

        // When there is no intersection.
        assert!(rect.clip_by(Rect::new(-2, 1, 1, 1)).is_none());
        assert!(rect.clip_by(Rect::new(11, 1, 1, 1)).is_none());
        assert!(rect.clip_by(Rect::new(1, -2, 1, 1)).is_none());
        assert!(rect.clip_by(Rect::new(1, 11, 1, 1)).is_none());
    }

    #[test]
    fn rect_translate() {
        let rect = Rect::new(0, 0, 10, 10);

        assert_eq!(rect.translate(Vector2::new(5, 5)), Rect::new(5, 5, 10, 10));
    }

    #[test]
    fn rect_intersects_circle() {
        let rect = Rect::new(0.0, 0.0, 1.0, 1.0);

        assert!(!rect.intersects_circle(Vector2::new(5.0, 5.0), 1.0));
        assert!(rect.intersects_circle(Vector2::new(0.0, 0.0), 1.0));
        assert!(rect.intersects_circle(Vector2::new(-0.5, -0.5), 1.0));
    }

    #[test]
    fn rect_transform() {
        let rect = Rect::new(0.0, 0.0, 1.0, 1.0);

        assert_eq!(
            rect.transform(&Matrix3::new(
                1.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, //
                0.0, 0.0, 1.0,
            )),
            rect,
        );

        assert_eq!(
            rect.transform(&Matrix3::new(
                2.0, 0.0, 0.0, //
                0.0, 2.0, 0.0, //
                0.0, 0.0, 2.0,
            )),
            Rect::new(0.0, 0.0, 2.0, 2.0),
        );
    }
}
