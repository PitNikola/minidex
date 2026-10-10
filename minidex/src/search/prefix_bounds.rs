//! One bounded, immutable vocabulary summary for successive prefixes. It is populated by an
//! ordinary search, so index writes, disk format and file access do not change.

use crate::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

const BLOCK_TERMS: usize = 64;
const MAX_BLOCKS: usize = 2048;
const MAX_KEY_BYTES: usize = 64 * 1024;
const MAX_DATA_BYTES: usize = 352 * 1024;
const MIN_TERMS: usize = 256;
const MAX_PREFIX_BYTES: usize = 128;

#[derive(Clone, Copy)]
pub(crate) struct Block {
    start: (u32, u32),
    end: (u32, u32),
    pub(crate) upper: u64,
    data: (u32, u32),
}

/// The lease is the last field of its owner, so vectors are freed before another builder may
/// reserve the budget. Old summaries held by concurrent readers also keep this lease alive.
struct Lease(Arc<AtomicBool>);
impl Drop for Lease {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

pub(crate) struct Bounds {
    generation: usize,
    segment: usize,
    prefix: String,
    pub(crate) disabled: AtomicBool,
    pub(crate) blocks: Vec<Block>,
    keys: Vec<u8>,
    data: Vec<u8>,
    pub(crate) vocabulary_complete: bool,
    _lease: Lease,
}

impl Bounds {
    pub(crate) fn start(&self, block: Block) -> &[u8] {
        &self.keys[block.start.0 as usize..block.start.1 as usize]
    }
    pub(crate) fn end(&self, block: Block) -> &[u8] {
        &self.keys[block.end.0 as usize..block.end.1 as usize]
    }
    pub(crate) fn matching_blocks(&self, prefix: &[u8], end: &[u8], ranked: bool) -> Vec<usize> {
        let mut indices: Vec<_> = self
            .blocks
            .iter()
            .enumerate()
            .filter(|(_, b)| self.end(**b) >= prefix && self.start(**b) < end)
            .map(|(i, _)| i)
            .collect();
        if ranked {
            indices.sort_unstable_by(|&a, &b| {
                self.blocks[b]
                    .upper
                    .cmp(&self.blocks[a].upper)
                    .then(a.cmp(&b))
            });
        }
        indices
    }
    /// Replay exactly the FST term order, using front-coded names and original posting offsets.
    /// Only the queried prefix needs decoding. Most blocks share it in full, needing no names.
    pub(crate) fn for_each_posting(
        &self,
        block: Block,
        prefix: &[u8],
        mut visit: impl FnMut(u64, bool),
    ) {
        let all_match =
            self.start(block).starts_with(prefix) && self.end(block).starts_with(prefix);
        let mut head = Vec::new();
        if !all_match {
            head.reserve_exact(prefix.len());
        }
        let mut data = &self.data[block.data.0 as usize..block.data.1 as usize];
        while !data.is_empty() {
            let offset = u64::from_le_bytes(data[..8].try_into().unwrap());
            let shared = u16::from_le_bytes(data[8..10].try_into().unwrap()) as usize;
            let len = u16::from_le_bytes(data[10..12].try_into().unwrap()) as usize;
            let suffix = &data[12..12 + len];
            let term_len = shared + len;
            if !all_match {
                head.truncate(shared.min(prefix.len()));
                let needed = prefix.len().saturating_sub(head.len()).min(len);
                head.extend_from_slice(&suffix[..needed]);
            }
            if all_match || head == prefix {
                visit(offset, term_len == prefix.len());
            }
            data = &data[12 + len..];
        }
    }
}

#[derive(Default)]
pub(crate) struct PrefixBounds {
    cached: Mutex<Option<Arc<Bounds>>>,
    reserved: Arc<AtomicBool>,
}

pub(crate) enum Prepared {
    Reuse(Arc<Bounds>),
    Build(Box<Builder>),
    None,
}

impl PrefixBounds {
    pub(crate) fn reuse(&self, generation: usize, segment: usize, prefix: &str) -> Prepared {
        let Ok(cached) = self.cached.lock() else {
            return Prepared::None;
        };
        if let Some(bounds) = cached.as_ref()
            && bounds.generation == generation
            && bounds.segment == segment
            && prefix.starts_with(&bounds.prefix)
            && (!bounds.disabled.load(Ordering::Acquire) || bounds.vocabulary_complete)
        {
            Prepared::Reuse(Arc::clone(bounds))
        } else {
            Prepared::None
        }
    }
    pub(crate) fn prepare(&self, generation: usize, segment: usize, prefix: &str) -> Prepared {
        if prefix.len() > MAX_PREFIX_BYTES {
            return Prepared::None;
        }
        let old = {
            let Ok(mut cached) = self.cached.lock() else {
                return Prepared::None;
            };
            if let Some(bounds) = cached.as_ref()
                && bounds.generation == generation
                && bounds.segment == segment
                && prefix.starts_with(&bounds.prefix)
            {
                if !bounds.disabled.load(Ordering::Acquire) || bounds.vocabulary_complete {
                    return Prepared::Reuse(Arc::clone(bounds));
                }
                if prefix == bounds.prefix {
                    return Prepared::None;
                }
            }
            cached.take()
        };
        drop(old);
        if self
            .reserved
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Prepared::None;
        }
        Prepared::Build(Box::new(Builder {
            bounds: Bounds {
                generation,
                segment,
                prefix: prefix.to_owned(),
                disabled: AtomicBool::new(false),
                blocks: Vec::with_capacity(MAX_BLOCKS),
                keys: Vec::with_capacity(MAX_KEY_BYTES),
                data: Vec::with_capacity(MAX_DATA_BYTES),
                vocabulary_complete: true,
                _lease: Lease(Arc::clone(&self.reserved)),
            },
            terms: 0,
            start: Vec::new(),
            end: Vec::new(),
            upper: 0,
            overflow: false,
            data_start: 0,
        }))
    }

    pub(crate) fn publish(&self, builder: Box<Builder>, tied: bool) {
        if let Some(bounds) = (*builder).finish(tied)
            && let Ok(mut cached) = self.cached.lock()
        {
            *cached = Some(Arc::new(bounds));
        }
    }
}

pub(crate) struct Builder {
    terms: usize,
    start: Vec<u8>,
    end: Vec<u8>,
    upper: u64,
    overflow: bool,
    data_start: u32,
    bounds: Bounds,
}

/// Memoize the original fold of one large first posting list, rather than a result page.
/// Subsequent expansions still run normally, including every original cutoff and exact flag.
#[derive(PartialEq)]
pub(crate) struct FoldKey {
    pub(crate) generation: usize,
    pub(crate) segment: usize,
    pub(crate) offset: u64,
    pub(crate) kind: Option<crate::Kind>,
    pub(crate) category: Option<u8>,
    pub(crate) volume_mask: u8,
    pub(crate) keep: usize,
    pub(crate) exact: bool,
    pub(crate) exact_token: Option<String>,
}

#[derive(Default)]
struct FoldState {
    key: Option<FoldKey>,
    rows: Vec<(u64, u32)>,
    words: Vec<u64>,
    gathered: usize,
    upper: u64,
    tied: bool,
}

#[derive(Default)]
pub(crate) struct PostingFold(Mutex<FoldState>);
impl PostingFold {
    pub(crate) fn get_into(
        &self,
        key: &FoldKey,
        rows: &mut Vec<(u64, u32)>,
        exact: &mut crate::ExactMatches,
    ) -> Option<(usize, u64, bool)> {
        let state = self.0.lock().ok()?;
        if state.key.as_ref() != Some(key) {
            return None;
        }
        rows.extend_from_slice(&state.rows);
        if key.exact {
            exact.words.copy_from_slice(&state.words);
        }
        Some((state.gathered, state.upper, state.tied))
    }
    pub(crate) fn store(
        &self,
        key: FoldKey,
        rows: &[(u64, u32)],
        exact: &crate::ExactMatches,
        result: (usize, u64, bool),
    ) {
        if rows.len() > 512 || (key.exact && exact.words.len() > 2048) {
            return;
        }
        if let Ok(mut state) = self.0.lock() {
            if state.rows.capacity() == 0 {
                state.rows = Vec::with_capacity(512);
            }
            state.rows.clear();
            state.rows.extend_from_slice(rows);
            state.words.clear();
            if key.exact {
                if state.words.capacity() == 0 {
                    state.words = Vec::with_capacity(2048);
                }
                state.words.extend_from_slice(&exact.words);
            }
            state.key = Some(key);
            state.gathered = result.0;
            state.upper = result.1;
            state.tied = result.2;
        }
    }
}

impl Builder {
    pub(crate) fn add(&mut self, term: &[u8], offset: u64, upper: u64) {
        // Boundary keys are bounded too, including the transient unfinished block.
        if self.overflow {
            return;
        }
        if term.len() > 1024 {
            self.overflow = true;
            return;
        }
        if self.terms.is_multiple_of(BLOCK_TERMS) {
            self.start.clear();
            self.start.extend_from_slice(term);
            self.upper = 0;
            self.data_start = self.bounds.data.len() as u32;
        }
        if self.bounds.vocabulary_complete {
            let shared = if self.terms.is_multiple_of(BLOCK_TERMS) {
                0
            } else {
                self.end
                    .iter()
                    .zip(term)
                    .take_while(|(a, b)| a == b)
                    .count()
            };
            let suffix = &term[shared..];
            if self.bounds.data.len() + 12 + suffix.len() > MAX_DATA_BYTES {
                self.bounds.vocabulary_complete = false;
                self.bounds.data = Vec::new();
            } else {
                self.bounds.data.extend_from_slice(&offset.to_le_bytes());
                self.bounds
                    .data
                    .extend_from_slice(&(shared as u16).to_le_bytes());
                self.bounds
                    .data
                    .extend_from_slice(&(suffix.len() as u16).to_le_bytes());
                self.bounds.data.extend_from_slice(suffix);
            }
        }
        self.end.clear();
        self.end.extend_from_slice(term);
        self.upper = self.upper.max(upper);
        self.terms += 1;
        if self.terms.is_multiple_of(BLOCK_TERMS) {
            self.flush_block();
        }
    }
    fn flush_block(&mut self) {
        if self.bounds.blocks.len() == MAX_BLOCKS
            || self.bounds.keys.len() + self.start.len() + self.end.len() > MAX_KEY_BYTES
        {
            self.overflow = true;
            return;
        }
        let start = self.bounds.keys.len() as u32;
        self.bounds.keys.extend_from_slice(&self.start);
        let mid = self.bounds.keys.len() as u32;
        self.bounds.keys.extend_from_slice(&self.end);
        self.bounds.blocks.push(Block {
            start: (start, mid),
            end: (mid, self.bounds.keys.len() as u32),
            upper: self.upper,
            data: (self.data_start, self.bounds.data.len() as u32),
        });
    }
    fn finish(mut self, tied: bool) -> Option<Bounds> {
        if !self.terms.is_multiple_of(BLOCK_TERMS) {
            self.flush_block();
        }
        if self.overflow || self.terms < MIN_TERMS {
            return None;
        }
        self.bounds.disabled.store(tied, Ordering::Release);
        Some(self.bounds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn posting_fold_restores_original_state_and_keys_exact_terms_and_filters() {
        let cache = PostingFold::default();
        let key = || FoldKey {
            generation: 1,
            segment: 0,
            offset: 10,
            kind: None,
            category: None,
            volume_mask: 15,
            keep: 70,
            exact: true,
            exact_token: Some("report".into()),
        };
        let rows = [(77, 1), (88, 2)];
        let mut exact = crate::ExactMatches::reuse(Vec::new(), 256);
        exact.set(33);
        cache.store(key(), &rows, &exact, (1000, 88, true));
        let mut copied = Vec::new();
        let mut copied_exact = crate::ExactMatches::reuse(Vec::new(), 256);
        assert_eq!(
            cache.get_into(&key(), &mut copied, &mut copied_exact),
            Some((1000, 88, true))
        );
        assert_eq!(copied, rows);
        assert!(copied_exact.contains(33));
        let mut changed = key();
        changed.exact_token = Some("txt".into());
        assert!(
            cache
                .get_into(&changed, &mut Vec::new(), &mut copied_exact)
                .is_none()
        );
        let mut changed = key();
        changed.kind = Some(crate::Kind::File);
        assert!(
            cache
                .get_into(&changed, &mut Vec::new(), &mut copied_exact)
                .is_none()
        );
        let mut changed = key();
        changed.generation = 2;
        assert!(
            cache
                .get_into(&changed, &mut Vec::new(), &mut copied_exact)
                .is_none()
        );
        let state = cache.0.lock().unwrap();
        assert!(
            state.rows.capacity() * size_of::<(u64, u32)>() + state.words.capacity() * 8
                <= 24 * 1024
        );
    }

    #[test]
    fn cached_traversal_preserves_unicode_prefixes_exact_terms_and_order() {
        let cache = PrefixBounds::default();
        let mut terms: Vec<_> = (0..600).map(|i| format!("abé{i:04}.txt")).collect();
        terms.extend(["ab".into(), "abé".into(), "ab中".into(), "ab😀".into()]);
        terms.sort_unstable();
        let Prepared::Build(mut build) = cache.prepare(1, 0, "ab") else {
            panic!()
        };
        for (i, term) in terms.iter().enumerate() {
            build.add(term.as_bytes(), i as u64, i as u64);
        }
        cache.publish(build, true);
        let Prepared::Reuse(bounds) = cache.prepare(1, 0, "ab") else {
            panic!()
        };
        assert!(
            bounds.blocks.capacity() * size_of::<Block>()
                + bounds.keys.capacity()
                + bounds.data.capacity()
                + bounds.prefix.capacity()
                + size_of::<Bounds>()
                < 512 * 1024
        );
        for prefix in [
            "ab",
            "abé",
            "abé00",
            "abé0001.txt",
            "ab中",
            "ab😀",
            "abmissing",
        ] {
            let expected: Vec<_> = terms
                .iter()
                .enumerate()
                .filter(|(_, t)| t.starts_with(prefix))
                .map(|(i, t)| (i as u64, t == prefix))
                .collect();
            let mut end = prefix.as_bytes().to_vec();
            end.push(0xff);
            let mut actual = Vec::new();
            for i in bounds.matching_blocks(prefix.as_bytes(), &end, false) {
                bounds.for_each_posting(bounds.blocks[i], prefix.as_bytes(), |offset, exact| {
                    actual.push((offset, exact))
                });
            }
            assert_eq!(actual, expected, "{prefix}");
        }
    }
    #[test]
    fn active_readers_keep_the_shared_memory_lease_until_their_vectors_are_freed() {
        let cache = PrefixBounds::default();
        let Prepared::Build(mut build) = cache.prepare(1, 0, "ab") else {
            panic!()
        };
        for i in 0..256 {
            build.add(format!("ab{i:04}").as_bytes(), i, i);
        }
        cache.publish(build, false);
        let Prepared::Reuse(reader) = cache.prepare(1, 0, "ab1") else {
            panic!()
        };
        assert!(matches!(cache.prepare(2, 0, "cd"), Prepared::None));
        drop(reader);
        assert!(matches!(cache.prepare(2, 0, "cd"), Prepared::Build(_)));
    }
    #[test]
    fn oversized_vocabularies_fall_back_and_release_the_budget() {
        let cache = PrefixBounds::default();
        let Prepared::Build(mut build) = cache.prepare(1, 0, "ab") else {
            panic!()
        };
        for i in 0..MAX_BLOCKS * BLOCK_TERMS + 1 {
            build.add(format!("ab{i:06}").as_bytes(), 0, 1);
        }
        cache.publish(build, false);
        assert!(matches!(cache.prepare(1, 0, "ab"), Prepared::Build(_)));
    }
    #[test]
    fn tied_cutoffs_disable_reordering_for_descendant_prefixes() {
        let cache = PrefixBounds::default();
        let Prepared::Build(mut build) = cache.prepare(1, 0, "ab") else {
            panic!()
        };
        for i in 0..256 {
            build.add(format!("ab{i:04}").as_bytes(), i, 1);
        }
        cache.publish(build, true);
        let Prepared::Reuse(bounds) = cache.prepare(1, 0, "ab1") else {
            panic!()
        };
        assert!(bounds.disabled.load(Ordering::Acquire));
        let mut offsets = Vec::new();
        for i in bounds.matching_blocks(b"ab1", b"ab1\xff", false) {
            bounds.for_each_posting(bounds.blocks[i], b"ab1", |offset, _| offsets.push(offset));
        }
        assert!(offsets.is_empty());
    }
}
