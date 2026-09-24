//! Explicit KV cache for autoregressive generation.
//!
//! ## Layout contract
//!
//! Each layer stores K and V as flattened `[position][kv_head][head_dim]`
//! F32 buffers: position `p` of layer `l` occupies
//! `k[l][p * kv_heads * head_dim .. (p + 1) * kv_heads * head_dim]`.
//!
//! ## Staging contract
//!
//! `seq_len` is the number of **committed** history positions. A forward at
//! position `p` reads history `[0, p)` and computes its current K/V
//! separately; the caller appends the current K/V for **every** layer and
//! only then calls [`KvCache::commit`] once to advance the history. Staged
//! (appended but uncommitted) entries are never visible to history reads.
//! Position `p` is therefore always written at slot `seq_len == p` and read
//! back only after commit.

use crate::error::{Error, Result};

/// Per-model KV cache holding K and V for every layer.
#[derive(Debug, Clone)]
pub struct KvCache {
    layers: usize,
    kv_heads: usize,
    head_dim: usize,
    capacity: usize,
    seq_len: usize,
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl KvCache {
    /// Create an empty cache for `layers` layers with room for `capacity`
    /// positions. All dimensions must be non-zero.
    pub fn new(layers: usize, kv_heads: usize, head_dim: usize, capacity: usize) -> Result<Self> {
        if layers == 0 || kv_heads == 0 || head_dim == 0 || capacity == 0 {
            return Err(Error(
                "KV cache requires non-zero layers, KV heads, head dimension, and capacity"
                    .to_string(),
            ));
        }
        let width = kv_heads
            .checked_mul(head_dim)
            .ok_or_else(|| Error("KV cache width overflows".to_string()))?;
        let elems = capacity
            .checked_mul(width)
            .ok_or_else(|| Error("KV cache allocation size overflows".to_string()))?;
        Ok(Self {
            layers,
            kv_heads,
            head_dim,
            capacity,
            seq_len: 0,
            k: vec![vec![0.0f32; elems]; layers],
            v: vec![vec![0.0f32; elems]; layers],
        })
    }

    /// Number of layers in this cache.
    pub fn layers(&self) -> usize {
        self.layers
    }

    /// Number of KV heads.
    pub fn kv_heads(&self) -> usize {
        self.kv_heads
    }

    /// Width of one head.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Position capacity in tokens.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of committed history positions.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// True when no history has been committed.
    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }

    /// Values per position: `kv_heads * head_dim`.
    pub fn position_width(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// Stage the current token's K/V for one layer at slot `seq_len`.
    ///
    /// Both slices use `[kv_head][head_dim]` layout. The entry becomes
    /// visible to history reads only after [`KvCache::commit`].
    pub fn append(&mut self, layer: usize, k: &[f32], v: &[f32]) -> Result<()> {
        if layer >= self.layers {
            return Err(Error(format!(
                "KV layer {layer} out of bounds for {} layers",
                self.layers
            )));
        }
        if self.seq_len >= self.capacity {
            return Err(Error("KV cache is full".to_string()));
        }
        let width = self.position_width();
        if k.len() != width || v.len() != width {
            return Err(Error(format!(
                "KV append size mismatch: expected {width}, got k {}, v {}",
                k.len(),
                v.len()
            )));
        }
        let offset = self.seq_len * width;
        self.k[layer][offset..offset + width].copy_from_slice(k);
        self.v[layer][offset..offset + width].copy_from_slice(v);
        Ok(())
    }

    /// Commit staged entries: advance the history length by one.
    ///
    /// The caller must have appended the current position for every layer
    /// before committing; commit itself only moves the visible boundary.
    pub fn commit(&mut self) {
        self.seq_len += 1;
    }

    /// Committed K history `[0, seq_len)` for one layer.
    pub fn k_history(&self, layer: usize) -> Result<&[f32]> {
        self.get_layer(layer, true, 0, self.seq_len)
    }

    /// Committed V history `[0, seq_len)` for one layer.
    pub fn v_history(&self, layer: usize) -> Result<&[f32]> {
        self.get_layer(layer, false, 0, self.seq_len)
    }

    /// Committed K positions `[start, end)` for one layer.
    pub fn k_range(&self, layer: usize, start: usize, end: usize) -> Result<&[f32]> {
        self.get_layer(layer, true, start, end)
    }

    /// Committed V positions `[start, end)` for one layer.
    pub fn v_range(&self, layer: usize, start: usize, end: usize) -> Result<&[f32]> {
        self.get_layer(layer, false, start, end)
    }

    /// Discard committed history (buffers are retained, not reallocated).
    pub fn clear(&mut self) {
        self.seq_len = 0;
    }

    /// Grow position capacity, preserving committed and staged entries.
    /// Shrinking (or equal size) is a no-op that succeeds.
    pub fn grow_to(&mut self, new_capacity: usize) -> Result<()> {
        if new_capacity <= self.capacity {
            return Ok(());
        }
        let width = self.position_width();
        let elems = new_capacity
            .checked_mul(width)
            .ok_or_else(|| Error("KV cache grown size overflows".to_string()))?;
        for buffer in self.k.iter_mut().chain(self.v.iter_mut()) {
            buffer.resize(elems, 0.0);
        }
        self.capacity = new_capacity;
        Ok(())
    }

    fn get_layer(&self, layer: usize, is_key: bool, start: usize, end: usize) -> Result<&[f32]> {
        if layer >= self.layers {
            return Err(Error(format!(
                "KV layer {layer} out of bounds for {} layers",
                self.layers
            )));
        }
        if start > end || end > self.seq_len {
            return Err(Error(format!(
                "KV read range {start}..{end} is outside committed history 0..{}",
                self.seq_len
            )));
        }
        let width = self.position_width();
        let buffer = if is_key {
            &self.k[layer]
        } else {
            &self.v[layer]
        };
        Ok(&buffer[start * width..end * width])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_cache_has_no_history() {
        let cache = KvCache::new(2, 2, 4, 8).unwrap();
        assert!(cache.is_empty());
        assert_eq!(cache.seq_len(), 0);
        assert_eq!(cache.k_history(0).unwrap(), &[]);
        assert_eq!(cache.v_history(1).unwrap(), &[]);
    }

    #[test]
    fn append_commit_makes_history_visible() {
        let mut cache = KvCache::new(1, 1, 2, 4).unwrap();
        cache.append(0, &[1.0, 2.0], &[3.0, 4.0]).unwrap();
        // Staged but not committed: still invisible.
        assert!(cache.is_empty());
        assert_eq!(cache.k_history(0).unwrap(), &[]);
        cache.commit();
        assert_eq!(cache.seq_len(), 1);
        assert_eq!(cache.k_history(0).unwrap(), &[1.0, 2.0]);
        assert_eq!(cache.v_history(0).unwrap(), &[3.0, 4.0]);
    }

    #[test]
    fn positions_are_position_major() {
        let mut cache = KvCache::new(1, 2, 2, 4).unwrap();
        // Position 0: kv_head 0 = [1, 2], kv_head 1 = [3, 4].
        cache.append(0, &[1.0, 2.0, 3.0, 4.0], &[0.0; 4]).unwrap();
        cache.commit();
        // Position 1: kv_head 0 = [5, 6], kv_head 1 = [7, 8].
        cache.append(0, &[5.0, 6.0, 7.0, 8.0], &[0.0; 4]).unwrap();
        cache.commit();
        assert_eq!(
            cache.k_history(0).unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
        assert_eq!(cache.k_range(0, 1, 2).unwrap(), &[5.0, 6.0, 7.0, 8.0]);
        assert!(cache.k_range(0, 0, 3).is_err());
        assert!(cache.k_range(0, 2, 1).is_err());
    }

    #[test]
    fn layers_are_independent() {
        let mut cache = KvCache::new(2, 1, 1, 4).unwrap();
        cache.append(0, &[1.0], &[2.0]).unwrap();
        cache.append(1, &[3.0], &[4.0]).unwrap();
        cache.commit();
        assert_eq!(cache.k_history(0).unwrap(), &[1.0]);
        assert_eq!(cache.k_history(1).unwrap(), &[3.0]);
        assert_eq!(cache.v_history(1).unwrap(), &[4.0]);
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn append_validates_bounds_and_sizes() {
        let mut cache = KvCache::new(1, 1, 2, 1).unwrap();
        assert!(cache.append(1, &[0.0, 0.0], &[0.0, 0.0]).is_err());
        assert!(cache.append(0, &[0.0], &[0.0, 0.0]).is_err());
        cache.append(0, &[1.0, 2.0], &[3.0, 4.0]).unwrap();
        cache.commit();
        assert!(cache.append(0, &[1.0, 2.0], &[3.0, 4.0]).is_err());
        assert!(cache.k_history(5).is_err());
        assert!(KvCache::new(0, 1, 1, 1).is_err());
    }

    #[test]
    fn grow_preserves_history() {
        let mut cache = KvCache::new(1, 1, 2, 1).unwrap();
        cache.append(0, &[1.0, 2.0], &[3.0, 4.0]).unwrap();
        cache.commit();
        cache.grow_to(4).unwrap();
        assert_eq!(cache.capacity(), 4);
        assert_eq!(cache.k_history(0).unwrap(), &[1.0, 2.0]);
        cache.append(0, &[5.0, 6.0], &[7.0, 8.0]).unwrap();
        cache.commit();
        assert_eq!(cache.k_history(0).unwrap(), &[1.0, 2.0, 5.0, 6.0]);
        cache.grow_to(1).unwrap();
        assert_eq!(cache.capacity(), 4);
    }
}
