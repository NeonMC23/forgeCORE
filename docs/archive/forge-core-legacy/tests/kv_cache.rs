//! KV-cache contract tests: staging, layout, ordering, growth.
use forge_core::kv::KvCache;

#[test]
fn empty_cache_reads_empty_history() {
    let cache = KvCache::new(2, 2, 4, 8).unwrap();
    assert!(cache.is_empty());
    assert_eq!(cache.seq_len(), 0);
    assert_eq!(cache.k_history(0).unwrap(), &[] as &[f32]);
    assert_eq!(cache.v_history(0).unwrap(), &[] as &[f32]);
    assert_eq!(cache.k_range(1, 0, 0).unwrap(), &[] as &[f32]);
}

#[test]
fn one_position_round_trips() {
    let mut cache = KvCache::new(1, 1, 2, 4).unwrap();
    cache.append(0, &[1.0, 2.0], &[3.0, 4.0]).unwrap();
    cache.commit();
    assert_eq!(cache.seq_len(), 1);
    assert!(!cache.is_empty());
    assert_eq!(cache.k_history(0).unwrap(), &[1.0, 2.0]);
    assert_eq!(cache.v_history(0).unwrap(), &[3.0, 4.0]);
}

#[test]
fn multiple_positions_preserve_order() {
    let mut cache = KvCache::new(1, 1, 1, 8).unwrap();
    for position in 0..4 {
        cache
            .append(0, &[position as f32], &[-(position as f32)])
            .unwrap();
        cache.commit();
    }
    assert_eq!(cache.k_history(0).unwrap(), &[0.0, 1.0, 2.0, 3.0]);
    assert_eq!(cache.v_history(0).unwrap(), &[0.0, -1.0, -2.0, -3.0]);
    assert_eq!(cache.k_range(0, 1, 3).unwrap(), &[1.0, 2.0]);
    assert_eq!(cache.v_range(0, 2, 4).unwrap(), &[-2.0, -3.0]);
}

#[test]
fn layout_is_position_major_over_kv_heads() {
    // 2 KV heads x head_dim 3: position p occupies one contiguous
    // 6-value span [head0(3), head1(3)].
    let mut cache = KvCache::new(1, 2, 3, 4).unwrap();
    cache
        .append(0, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[0.0; 6])
        .unwrap();
    cache.commit();
    cache
        .append(0, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[0.0; 6])
        .unwrap();
    cache.commit();
    assert_eq!(
        cache.k_history(0).unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    );
}

#[test]
fn staged_entries_are_invisible_until_commit() {
    let mut cache = KvCache::new(2, 1, 2, 4).unwrap();
    cache.append(0, &[1.0, 2.0], &[3.0, 4.0]).unwrap();
    // Only layer 0 staged so far: no layer exposes the new position.
    assert_eq!(cache.k_history(0).unwrap(), &[] as &[f32]);
    assert_eq!(cache.k_history(1).unwrap(), &[] as &[f32]);
    cache.append(1, &[5.0, 6.0], &[7.0, 8.0]).unwrap();
    cache.commit();
    assert_eq!(cache.seq_len(), 1);
    assert_eq!(cache.k_history(0).unwrap(), &[1.0, 2.0]);
    assert_eq!(cache.k_history(1).unwrap(), &[5.0, 6.0]);
}

#[test]
fn cache_rejects_out_of_bounds_access() {
    let mut cache = KvCache::new(1, 1, 2, 1).unwrap();
    assert!(cache.append(1, &[0.0, 0.0], &[0.0, 0.0]).is_err());
    assert!(cache.k_history(1).is_err());
    assert!(cache.k_range(0, 0, 1).is_err());
    cache.append(0, &[1.0, 2.0], &[3.0, 4.0]).unwrap();
    cache.commit();
    assert!(cache.append(0, &[1.0, 2.0], &[3.0, 4.0]).is_err());
    assert!(cache.v_range(0, 0, 2).is_err());
    assert!(cache.v_range(0, 1, 0).is_err());
}

#[test]
fn growth_preserves_committed_and_staged_data() {
    let mut cache = KvCache::new(1, 1, 2, 1).unwrap();
    cache.append(0, &[1.0, 2.0], &[3.0, 4.0]).unwrap();
    cache.commit();
    cache.grow_to(8).unwrap();
    assert_eq!(cache.capacity(), 8);
    assert_eq!(cache.k_history(0).unwrap(), &[1.0, 2.0]);
    cache.append(0, &[5.0, 6.0], &[7.0, 8.0]).unwrap();
    cache.commit();
    assert_eq!(cache.k_history(0).unwrap(), &[1.0, 2.0, 5.0, 6.0]);
    assert_eq!(cache.v_history(0).unwrap(), &[3.0, 4.0, 7.0, 8.0]);
}
