//! Quadrilateral (quad) tree is used for space partitioning and fast spatial queries.

use crate::Rect;
use arrayvec::ArrayVec;
use nalgebra::Vector2;

#[derive(Clone)]
enum QuadTreeNode<T: Clone> {
    Leaf {
        bounds: Rect<f32>,
        ids: Vec<T>,
    },
    Branch {
        bounds: Rect<f32>,
        leaves: [usize; 4],
    },
}

fn split_rect(rect: &Rect<f32>) -> [Rect<f32>; 4] {
    let half_size = rect.size.scale(0.5);
    [
        Rect {
            position: rect.position,
            size: half_size,
        },
        Rect {
            position: Vector2::new(rect.position.x + half_size.x, rect.position.y),
            size: half_size,
        },
        Rect {
            position: rect.position + half_size,
            size: half_size,
        },
        Rect {
            position: Vector2::new(rect.position.x, rect.position.y + half_size.y),
            size: half_size,
        },
    ]
}

/// Quadrilateral (quad) tree is used for space partitioning and fast spatial queries.
#[derive(Clone)]
pub struct QuadTree<T: Clone> {
    nodes: Vec<QuadTreeNode<T>>,
    root: usize,
    split_threshold: usize,
}

impl<T: Clone + 'static> Default for QuadTree<T> {
    fn default() -> Self {
        Self {
            nodes: Default::default(),
            root: Default::default(),
            split_threshold: 16,
        }
    }
}

/// A trait for anything that has rectangular bounds.
pub trait BoundsProvider {
    /// Identifier of the bounds provider.
    type Id: Clone;

    /// Returns bounds of the bounds provider.
    fn bounds(&self) -> Rect<f32>;

    /// Returns id of the bounds provider.
    fn id(&self) -> Self::Id;
}

/// An error, that may occur during the build of the quad tree.
pub enum QuadTreeBuildError {
    /// It means that given split threshold is too low for an algorithm to build quad tree.
    /// Make it larger and try again. Also this might mean that your initial bounds are too small.
    ReachedRecursionLimit,
}

#[derive(Clone)]
struct Entry<I: Clone> {
    id: I,
    bounds: Rect<f32>,
}

fn build_recursive<I>(
    nodes: &mut Vec<QuadTreeNode<I>>,
    bounds: Rect<f32>,
    entries: &[Entry<I>],
    split_threshold: usize,
    depth: usize,
) -> Result<usize, QuadTreeBuildError>
where
    I: Clone + 'static,
{
    if depth >= 64 {
        Err(QuadTreeBuildError::ReachedRecursionLimit)
    } else if entries.len() <= split_threshold {
        let index = nodes.len();
        nodes.push(QuadTreeNode::Leaf {
            bounds,
            ids: entries.iter().map(|e| e.id.clone()).collect::<Vec<_>>(),
        });
        Ok(index)
    } else {
        let leaf_bounds = split_rect(&bounds);
        let mut leaves = [usize::MAX; 4];

        for (leaf, &leaf_bounds) in leaves.iter_mut().zip(leaf_bounds.iter()) {
            let leaf_entries = entries
                .iter()
                .filter_map(|e| {
                    if leaf_bounds.intersects(e.bounds) {
                        Some(e.clone())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();

            *leaf = build_recursive(
                nodes,
                leaf_bounds,
                &leaf_entries,
                split_threshold,
                depth + 1,
            )?;
        }

        let index = nodes.len();
        nodes.push(QuadTreeNode::Branch { bounds, leaves });
        Ok(index)
    }
}

impl<I> QuadTree<I>
where
    I: Clone + 'static,
{
    /// Creates new quad tree from the given initial bounds and the set of objects.
    pub fn new<T>(
        root_bounds: Rect<f32>,
        objects: impl Iterator<Item = T>,
        split_threshold: usize,
    ) -> Result<Self, QuadTreeBuildError>
    where
        T: BoundsProvider<Id = I>,
    {
        let entries = objects
            .filter_map(|o| {
                if root_bounds.intersects(o.bounds()) {
                    Some(Entry {
                        id: o.id(),
                        bounds: o.bounds(),
                    })
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        let mut nodes = Vec::new();
        let root = build_recursive(&mut nodes, root_bounds, &entries, split_threshold, 0)?;
        Ok(Self {
            nodes,
            root,
            split_threshold,
        })
    }

    /// Searches for a leaf node in the tree, that contains the given point and writes ids of the
    /// entities stored in the leaf node to the output storage.
    pub fn point_query<S>(&self, point: Vector2<f32>, storage: &mut S)
    where
        S: QueryStorage<Id = I>,
    {
        self.point_query_recursive(self.root, point, storage)
    }

    fn point_query_recursive<S>(&self, node: usize, point: Vector2<f32>, storage: &mut S)
    where
        S: QueryStorage<Id = I>,
    {
        if let Some(node) = self.nodes.get(node) {
            match node {
                QuadTreeNode::Leaf { bounds, ids } => {
                    if bounds.contains(point) {
                        for id in ids {
                            if !storage.try_push(id.clone()) {
                                return;
                            }
                        }
                    }
                }
                QuadTreeNode::Branch { bounds, leaves } => {
                    if bounds.contains(point) {
                        for &leaf in leaves {
                            self.point_query_recursive(leaf, point, storage)
                        }
                    }
                }
            }
        }
    }

    /// Returns current split threshold, that was used to build the quad tree.
    pub fn split_threshold(&self) -> usize {
        self.split_threshold
    }
}

/// Arbitrary storage for query results.
pub trait QueryStorage {
    /// Id of an entity in the storage.
    type Id;

    /// Tries to push a new id in the storage.
    fn try_push(&mut self, id: Self::Id) -> bool;

    /// Clears the storage.
    fn clear(&mut self);
}

impl<I> QueryStorage for Vec<I> {
    type Id = I;

    fn try_push(&mut self, intersection: I) -> bool {
        self.push(intersection);
        true
    }

    fn clear(&mut self) {
        self.clear()
    }
}

impl<I, const CAP: usize> QueryStorage for ArrayVec<I, CAP> {
    type Id = I;

    fn try_push(&mut self, intersection: I) -> bool {
        self.try_push(intersection).is_ok()
    }

    fn clear(&mut self) {
        self.clear()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::Rect;

    struct TestObject {
        bounds: Rect<f32>,
        id: usize,
    }

    impl BoundsProvider for &TestObject {
        type Id = usize;

        fn bounds(&self) -> Rect<f32> {
            self.bounds
        }

        fn id(&self) -> Self::Id {
            self.id
        }
    }

    #[test]
    fn test_quad_tree() {
        let root_bounds = Rect::new(0.0, 0.0, 200.0, 200.0);
        let objects = vec![
            TestObject {
                bounds: Rect::new(10.0, 10.0, 10.0, 10.0),
                id: 0,
            },
            TestObject {
                bounds: Rect::new(10.0, 10.0, 10.0, 10.0),
                id: 1,
            },
        ];
        // Infinite recursion prevention check (when there are multiple objects share same location).
        assert!(QuadTree::new(root_bounds, objects.iter(), 1).is_err());

        let objects = vec![
            TestObject {
                bounds: Rect::new(10.0, 10.0, 10.0, 10.0),
                id: 0,
            },
            TestObject {
                bounds: Rect::new(20.0, 20.0, 10.0, 10.0),
                id: 1,
            },
        ];
        assert!(QuadTree::new(root_bounds, objects.iter(), 1).is_ok());
    }

    #[test]
    fn default_for_quad_tree() {
        let tree = QuadTree::<u32>::default();

        assert_eq!(tree.split_threshold, 16);
        assert_eq!(tree.root, 0);
    }

    #[test]
    fn quad_tree_point_query() {
        // empty
        let tree = QuadTree::<f32>::default();
        let mut s = Vec::<f32>::new();

        tree.point_query(Vector2::new(0.0, 0.0), &mut s);
        assert_eq!(s, vec![]);

        let root_bounds = Rect::new(0.0, 0.0, 200.0, 200.0);

        // leaf
        let mut s = Vec::<usize>::new();
        let mut pool = Vec::new();
        pool.push(QuadTreeNode::Leaf {
            bounds: root_bounds,
            ids: vec![0, 1],
        });

        let tree = QuadTree {
            root: 0,
            nodes: pool,
            ..Default::default()
        };

        tree.point_query(Vector2::new(10.0, 10.0), &mut s);
        assert_eq!(s, vec![0, 1]);

        // branch
        let mut s = Vec::<usize>::new();
        let mut pool = Vec::new();
        let a = 0;
        pool.push(QuadTreeNode::Leaf {
            bounds: root_bounds,
            ids: vec![0, 1],
        });
        let b = 1;
        pool.push(QuadTreeNode::Branch {
            bounds: root_bounds,
            leaves: [a, a, a, a],
        });

        let tree = QuadTree {
            root: b,
            nodes: pool,
            ..Default::default()
        };

        tree.point_query(Vector2::new(10.0, 10.0), &mut s);
        assert_eq!(s, vec![0, 1, 0, 1, 0, 1, 0, 1]);
    }

    #[test]
    fn quad_tree_split_threshold() {
        let tree = QuadTree::<u32>::default();

        assert_eq!(tree.split_threshold(), tree.split_threshold);
    }

    #[test]
    fn query_storage_for_vec() {
        let mut s = vec![1];

        let res = QueryStorage::try_push(&mut s, 2);
        assert!(res);
        assert_eq!(s, vec![1, 2]);

        QueryStorage::clear(&mut s);
        assert!(s.is_empty());
    }

    #[test]
    fn query_storage_for_array_vec() {
        let mut s = ArrayVec::<i32, 3>::new();

        let res = QueryStorage::try_push(&mut s, 1);
        assert!(res);
        assert!(!s.is_empty());

        QueryStorage::clear(&mut s);
        assert!(s.is_empty());
    }
}
