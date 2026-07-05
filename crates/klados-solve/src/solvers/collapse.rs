//! Coarsen-Collapse master solve.
//!
//! The minimum-components partition over a fixed column pool is a MAX-WEIGHT
//! INDEPENDENT SET in the *column-crossing graph* (vertices = agreement-block
//! columns, edge = two columns share an embedding node in some tree, weight =
//! `|c| - 1`). Non-crossing columns are automatically leaf-disjoint (two columns
//! sharing a leaf share that leaf's parent node), and any leaf left uncovered by
//! the chosen columns becomes a singleton (weight 0). So `k = n - MWIS`.
//!
//! On the LP-coarsened pool this graph has small treewidth (measured 8-10 on
//! typical irreducible cores), so bucket elimination solves the master exactly
//! in milliseconds — where the generic pool-MIP stalls and branch-and-price
//! takes minutes.

use crate::solvers::bp::column::AfColumn;

const NEG: i64 = i64::MIN / 4;

/// Do two multi-leaf columns cross? (share an embedding node in some tree).
fn columns_cross(a: &AfColumn, b: &AfColumn) -> bool {
    for (na, nb) in a
        .coverage()
        .iter_per_tree()
        .zip(b.coverage().iter_per_tree())
    {
        // both node lists are sorted+deduped
        let (mut i, mut j) = (0usize, 0usize);
        while i < na.len() && j < nb.len() {
            match na[i].cmp(&nb[j]) {
                std::cmp::Ordering::Equal => return true,
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
            }
        }
    }
    false
}

/// Exact max-weight independent set via bucket elimination on a min-degree
/// elimination order. Returns the selected-vertex mask, or `None` if the
/// induced treewidth exceeds `tw_cap` (caller falls back).
pub fn mwis_bucket_elim(
    n: usize,
    adj: &[Vec<usize>],
    weight: &[i64],
    tw_cap: usize,
) -> Option<Vec<bool>> {
    if n == 0 {
        return Some(Vec::new());
    }
    use std::collections::BTreeSet;

    // Min-degree elimination order with fill-in; abort if width blows the cap.
    let mut work: Vec<BTreeSet<usize>> = adj.iter().map(|a| a.iter().copied().collect()).collect();
    let mut alive = vec![true; n];
    let mut order = Vec::with_capacity(n);
    let mut tw = 0usize;
    for _ in 0..n {
        let v = (0..n)
            .filter(|&x| alive[x])
            .min_by_key(|&x| work[x].len())
            .unwrap();
        let nb: Vec<usize> = work[v].iter().copied().collect();
        tw = tw.max(nb.len());
        if tw > tw_cap {
            return None;
        }
        for i in 0..nb.len() {
            for j in (i + 1)..nb.len() {
                work[nb[i]].insert(nb[j]);
                work[nb[j]].insert(nb[i]);
            }
        }
        for &u in &nb {
            work[u].remove(&v);
        }
        alive[v] = false;
        order.push(v);
    }
    let mut posn = vec![0usize; n];
    for (i, &v) in order.iter().enumerate() {
        posn[v] = i;
    }

    struct Factor {
        scope: Vec<usize>, // sorted by posn
        table: Vec<i64>,   // index = Σ bit_k << k, bit_k = assignment of scope[k]
    }
    let bucket_of = |scope: &[usize]| *scope.iter().min_by_key(|&&u| posn[u]).unwrap();

    let mut buckets: Vec<Vec<Factor>> = (0..n).map(|_| Vec::new()).collect();
    for v in 0..n {
        buckets[v].push(Factor {
            scope: vec![v],
            table: vec![0, weight[v]],
        });
    }
    for i in 0..n {
        for &j in &adj[i] {
            if i < j {
                let scope = if posn[i] < posn[j] {
                    vec![i, j]
                } else {
                    vec![j, i]
                };
                let b = bucket_of(&scope);
                buckets[b].push(Factor {
                    scope,
                    table: vec![0, 0, 0, NEG], // (1,1) forbidden
                });
            }
        }
    }

    // trace[v] = (rest scope, choice[rest-assignment] = optimal x_v)
    let mut trace: Vec<Option<(Vec<usize>, Vec<u8>)>> = (0..n).map(|_| None).collect();
    for &v in &order {
        let factors = std::mem::take(&mut buckets[v]);
        if factors.is_empty() {
            continue;
        }
        let mut sset = BTreeSet::new();
        for f in &factors {
            for &u in &f.scope {
                sset.insert(u);
            }
        }
        let mut scope: Vec<usize> = sset.into_iter().collect();
        scope.sort_by_key(|&u| posn[u]);
        let s = scope.len();
        let mut spos = vec![0usize; n];
        for (k, &u) in scope.iter().enumerate() {
            spos[u] = k;
        }
        let mut combined = vec![0i64; 1usize << s];
        for f in &factors {
            let fmap: Vec<usize> = f.scope.iter().map(|&u| spos[u]).collect();
            for (assign, c) in combined.iter_mut().enumerate() {
                let mut fidx = 0usize;
                for (k, &sp) in fmap.iter().enumerate() {
                    fidx |= ((assign >> sp) & 1) << k;
                }
                let val = f.table[fidx];
                *c = if *c <= NEG || val <= NEG {
                    NEG
                } else {
                    *c + val
                };
            }
        }
        let vk = spos[v];
        let rest: Vec<usize> = scope.iter().copied().filter(|&u| u != v).collect();
        let rs = rest.len();
        let restpos: Vec<usize> = rest.iter().map(|&u| spos[u]).collect();
        let mut newtab = vec![NEG; 1usize << rs];
        let mut choice = vec![0u8; 1usize << rs];
        for (assign, &val) in combined.iter().enumerate() {
            let mut ri = 0usize;
            for (k, &sp) in restpos.iter().enumerate() {
                ri |= ((assign >> sp) & 1) << k;
            }
            if val > newtab[ri] {
                newtab[ri] = val;
                choice[ri] = ((assign >> vk) & 1) as u8;
            }
        }
        trace[v] = Some((rest.clone(), choice));
        if !rest.is_empty() {
            let b = bucket_of(&rest);
            buckets[b].push(Factor {
                scope: rest,
                table: newtab,
            });
        }
    }

    // Traceback in reverse elimination order.
    let mut assign = vec![0u8; n];
    for &v in order.iter().rev() {
        if let Some((rest, choice)) = &trace[v] {
            let mut ri = 0usize;
            for (k, &u) in rest.iter().enumerate() {
                ri |= (assign[u] as usize) << k;
            }
            assign[v] = choice[ri];
        }
    }
    Some(assign.iter().map(|&x| x == 1).collect())
}

/// Leaves covered by the LARGEST connected component of the column-crossing
/// graph over the support (multi-leaf columns with value > eps). This is the
/// "coupled block": the region where an unconverged core's LP support is thin,
/// and where a restricted re-pricing (smaller instance ⇒ CG converges) pays off.
/// Returns empty if the largest component is trivial or spans (almost) the whole
/// core (nothing to gain from restricting).
pub fn coupled_leaves(columns: &[AfColumn], values: &[f64], n: usize) -> Vec<u32> {
    const EPS: f64 = 1.0e-9;
    let lim = columns.len().min(values.len());
    let idx: Vec<usize> = (0..lim)
        .filter(|&i| columns[i].size() >= 2 && values[i] > EPS)
        .collect();
    let m = idx.len();
    if m == 0 {
        return Vec::new();
    }
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); m];
    for a in 0..m {
        for b in (a + 1)..m {
            if columns_cross(&columns[idx[a]], &columns[idx[b]]) {
                adj[a].push(b);
                adj[b].push(a);
            }
        }
    }
    let mut comp = vec![usize::MAX; m];
    let mut nc = 0usize;
    for s in 0..m {
        if comp[s] != usize::MAX {
            continue;
        }
        comp[s] = nc;
        let mut st = vec![s];
        while let Some(u) = st.pop() {
            for &w in &adj[u] {
                if comp[w] == usize::MAX {
                    comp[w] = nc;
                    st.push(w);
                }
            }
        }
        nc += 1;
    }
    use std::collections::BTreeSet;
    let mut comp_leaves: Vec<BTreeSet<u32>> = vec![BTreeSet::new(); nc];
    for a in 0..m {
        for &l in columns[idx[a]].labels() {
            comp_leaves[comp[a]].insert(l);
        }
    }
    match comp_leaves.into_iter().max_by_key(|s| s.len()) {
        Some(s) if s.len() >= 20 && s.len() <= n * 9 / 10 => s.into_iter().collect(),
        _ => Vec::new(),
    }
}

/// Greedy max-weight independent set: take vertices by descending weight,
/// skipping any adjacent to an already-taken one. Used as a per-component
/// fallback when a single component's treewidth exceeds the cap.
fn greedy_mwis(n: usize, adj: &[Vec<usize>], weight: &[i64]) -> Vec<bool> {
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&v| std::cmp::Reverse(weight[v]));
    let mut sel = vec![false; n];
    let mut blocked = vec![false; n];
    for v in order {
        if blocked[v] || weight[v] <= 0 {
            continue;
        }
        sel[v] = true;
        for &u in &adj[v] {
            blocked[u] = true;
        }
    }
    sel
}

/// Exact MWIS that first splits the graph into CONNECTED COMPONENTS — MWIS is
/// separable across them — and solves each with bucket elimination. A single
/// dense component whose induced treewidth exceeds `tw_cap` falls back to greedy
/// for that component ALONE (the rest stay exact). This is what lets the master
/// scale to large cores: the whole crossing graph often has tw > cap, but its
/// components are individually small (validated: 8661-core → largest comp tw ≤ 18).
pub fn mwis_components(n: usize, adj: &[Vec<usize>], weight: &[i64], tw_cap: usize) -> Vec<bool> {
    use std::collections::HashMap;
    let mut comp = vec![usize::MAX; n];
    let mut ncomp = 0usize;
    for s in 0..n {
        if comp[s] != usize::MAX {
            continue;
        }
        comp[s] = ncomp;
        let mut stack = vec![s];
        while let Some(u) = stack.pop() {
            for &w in &adj[u] {
                if comp[w] == usize::MAX {
                    comp[w] = ncomp;
                    stack.push(w);
                }
            }
        }
        ncomp += 1;
    }
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); ncomp];
    for v in 0..n {
        members[comp[v]].push(v);
    }
    let mut sel = vec![false; n];
    for mem in &members {
        if mem.len() == 1 {
            // isolated column: weight ≥ 1, no conflicts ⇒ always in the MWIS
            sel[mem[0]] = weight[mem[0]] > 0;
            continue;
        }
        let local: HashMap<usize, usize> = mem.iter().enumerate().map(|(k, &g)| (g, k)).collect();
        let ladj: Vec<Vec<usize>> = mem
            .iter()
            .map(|&g| {
                adj[g]
                    .iter()
                    .filter_map(|w| local.get(w).copied())
                    .collect()
            })
            .collect();
        let lw: Vec<i64> = mem.iter().map(|&g| weight[g]).collect();
        let lsel = mwis_bucket_elim(mem.len(), &ladj, &lw, tw_cap)
            .unwrap_or_else(|| greedy_mwis(mem.len(), &ladj, &lw));
        for (k, &g) in mem.iter().enumerate() {
            if lsel[k] {
                sel[g] = true;
            }
        }
    }
    sel
}

/// Solve the master over the LP *support* via the tree-DP. `values` are the
/// converged LP column values; only support columns (value > eps) with ≥2 leaves
/// enter the crossing graph — that is the LP-coarsened pool where treewidth is
/// small. Returns the forest as label groups (chosen columns + singletons for
/// uncovered leaves). Never bails: each crossing-graph component is solved
/// independently, so a locally-dense component can't sink the whole master.
pub fn master_via_tree_dp(
    columns: &[AfColumn],
    values: &[f64],
    n: usize,
    tw_cap: usize,
) -> Option<Vec<Vec<u32>>> {
    const EPS: f64 = 1.0e-9;
    let lim = columns.len().min(values.len());
    let idx: Vec<usize> = (0..lim)
        .filter(|&i| columns[i].size() >= 2 && values[i] > EPS)
        .collect();
    let m = idx.len();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); m];
    for a in 0..m {
        for b in (a + 1)..m {
            if columns_cross(&columns[idx[a]], &columns[idx[b]]) {
                adj[a].push(b);
                adj[b].push(a);
            }
        }
    }
    let weight: Vec<i64> = idx.iter().map(|&i| columns[i].size() as i64 - 1).collect();
    let sel = mwis_components(m, &adj, &weight, tw_cap);
    let mut covered = vec![false; n + 1];
    let mut groups: Vec<Vec<u32>> = Vec::new();
    for a in 0..m {
        if sel[a] {
            let labs = columns[idx[a]].labels().to_vec();
            for &l in &labs {
                covered[l as usize] = true;
            }
            groups.push(labs);
        }
    }
    for l in 1..=n as u32 {
        if !covered[l as usize] {
            groups.push(vec![l]);
        }
    }
    Some(groups)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Brute-force MWIS for validation.
    fn brute(n: usize, adj: &[Vec<usize>], w: &[i64]) -> i64 {
        let mut best = 0;
        for mask in 0..(1u32 << n) {
            let mut ok = true;
            let mut val = 0;
            for v in 0..n {
                if mask >> v & 1 == 1 {
                    val += w[v];
                    for &u in &adj[v] {
                        if u > v && mask >> u & 1 == 1 {
                            ok = false;
                        }
                    }
                }
            }
            if ok && val > best {
                best = val;
            }
        }
        best
    }

    fn val_of(sel: &[bool], w: &[i64]) -> i64 {
        sel.iter()
            .zip(w)
            .filter(|(s, _)| **s)
            .map(|(_, &x)| x)
            .sum()
    }

    fn is_independent(sel: &[bool], adj: &[Vec<usize>]) -> bool {
        for v in 0..sel.len() {
            if sel[v] {
                for &u in &adj[v] {
                    if sel[u] {
                        return false;
                    }
                }
            }
        }
        true
    }

    #[test]
    fn mwis_matches_bruteforce() {
        // deterministic pseudo-random graphs
        let mut state: u64 = 0x1234_5678;
        let mut rng = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..200 {
            let n = 2 + (rng() % 10) as usize; // up to 11 vertices
            let mut adj = vec![Vec::new(); n];
            for i in 0..n {
                for j in (i + 1)..n {
                    if rng() % 3 == 0 {
                        adj[i].push(j);
                        adj[j].push(i);
                    }
                }
            }
            let w: Vec<i64> = (0..n).map(|_| 1 + (rng() % 5) as i64).collect();
            let sel = mwis_bucket_elim(n, &adj, &w, n).unwrap();
            assert!(is_independent(&sel, &adj), "not independent");
            assert_eq!(val_of(&sel, &w), brute(n, &adj, &w), "wrong MWIS value");
        }
    }

    #[test]
    fn tw_cap_bails() {
        // a clique of 6 has treewidth 5; cap 3 must bail.
        let n = 6;
        let mut adj = vec![Vec::new(); n];
        for i in 0..n {
            for j in (i + 1)..n {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
        let w = vec![1i64; n];
        assert!(mwis_bucket_elim(n, &adj, &w, 3).is_none());
        assert!(mwis_bucket_elim(n, &adj, &w, 5).is_some());
    }
}
