//! Common algorithms for rectangles (clipping, transformation, quadtree, rect packing, etc.)

#![warn(missing_docs)]
#![forbid(unsafe_code)]

use nalgebra::{Matrix3, Vector2};
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
    /// To build bounding rectangle you should correctly initialize initial rectangle:
    ///
    /// ```
    /// # use nalgebra::Vector2;
    /// # use rectutils::Rect;
    ///
    /// let vertices = [Vector2::new(1.0, 2.0), Vector2::new(-3.0, 5.0)];
    ///
    /// // This is important part, it must have "invalid" state to correctly
    /// // calculate bounding rect. Rect::default will give invalid result!
    /// let mut bounding_rect = Rect::new(f32::MAX, f32::MAX, 0.0, 0.0);
    ///
    /// for &v in &vertices {
    ///     bounding_rect.push(v);
    /// }
    /// ```
    #[inline]
    pub fn push(&mut self, p: Vector2<T>) {
        if p.x < self.position.x {
            self.position.x = p.x;
        }
        if p.y < self.position.y {
            self.position.y = p.y;
        }

        let right_bottom = self.right_bottom_corner();

        if p.x > right_bottom.x {
            self.size.x = p.x - self.position.x;
        }
        if p.y > right_bottom.y {
            self.size.y = p.y - self.position.y;
        }
    }

    /// Clips the rectangle by some other rectangle and returns a new rectangle that corresponds to
    /// the intersection of both rectangles. If the rectangles does not intersects, the method
    /// returns this rectangle.
    #[inline]
    #[must_use = "this method creates new instance of rect"]
    pub fn clip_by(&self, other: Rect<T>) -> Rect<T> {
        let mut clipped = *self;

        if other.x() + other.w() < self.x()
            || other.x() > self.x() + self.w()
            || other.y() + other.h() < self.y()
            || other.y() > self.y() + self.h()
        {
            return clipped;
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

        clipped
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
    pub fn extend_to_contain(&mut self, other: Rect<T>) {
        if other.position.x < self.position.x {
            self.position.x = other.position.x;
        }
        if other.position.y < self.position.y {
            self.position.y = other.position.y;
        }
        let self_right_bottom_corner = self.right_bottom_corner();
        let other_right_bottom_corner = other.right_bottom_corner();
        if other_right_bottom_corner.x > self_right_bottom_corner.x {
            self.size.x += other_right_bottom_corner.x - self_right_bottom_corner.x;
        }
        if other_right_bottom_corner.y > self_right_bottom_corner.y {
            self.size.y += other_right_bottom_corner.y - self_right_bottom_corner.y;
        }
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
