use std::io::Cursor;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use klados_core::kernelize::{self, KernelizeConfig};
use klados_core::lower_bound::maf_bounds;
use klados_core::{Instance, Label, SolverStats, Tree};

use crate::HeuristicSolver;
use crate::max_sat::max_sat_problem::{ClauseKind, Lit, MaxSatProblem, VarId};

mod max_sat_problem;

const SOLVER_PATH: &str = "open-wbo/open-wbo_release";
const SOLVER_ARGS: &[&str] = &[];

pub struct MaxSatSolver {
    child_pid: Arc<AtomicU32>,
}

impl Default for MaxSatSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl MaxSatSolver {
    pub fn new() -> Self {
        Self {
            child_pid: Arc::new(AtomicU32::new(0)),
        }
    }

    pub fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>> {
        println!("c Start solve");
        if instance.trees.is_empty() {
            return None;
        }

        if instance.num_trees() == 1 {
            return Some(instance.trees.clone());
        }

        // Kernelize first to reduce the instance size before encoding.
        let kern_config = KernelizeConfig::default();
        let kern = kernelize::kernelize(instance, &kern_config);
        let reduced = &kern.instance;

        // Compute bounds on the reduced instance.
        // Use the UB as k (instead of n), which dramatically shrinks the encoding.
        let bounds = maf_bounds(&reduced.trees, reduced.num_leaves);
        let param_reduction = kern.param_reduction;

        let leaves: Vec<_> = reduced.reference_tree().leaves().collect();

        let n = leaves.len();
        // k capped at UB (adjusted for param reduction already subtracted by kernelize).
        // Ensure k >= 1 to avoid degenerate encodings.
        let k = bounds.upper.max(1);
        let m = reduced.num_trees();

        println!("c Kernelized: {}→{} leaves, LB={}, UB={}", instance.num_leaves, n, bounds.lower, k);

        let mut msp = MaxSatProblem::new();

        // Variables

        println!("c Creating variables");
        let mut l = vec![vec![VarId(0); n]; k];
        for i in 0..k {
            for j in 0..n {
                l[i][j] = msp.add_var();
            }
        }

        let mut u = vec![VarId(0); k];
        for i in 0..k {
            u[i] = msp.add_var();
        }

        let num_nodes = 2 * n - 1;

        let mut w = vec![vec![vec![VarId(0); num_nodes]; k]; m];
        for q in 0..m {
            for i in 0..k {
                for v in 0..num_nodes {
                    w[q][i][v] = msp.add_var();
                }
            }
        }

        let mut s = vec![vec![VarId(0); n]; k];
        for i in 0..k {
            for j in 0..n {
                s[i][j] = msp.add_var();
            }
        }

        let mut t = vec![vec![vec![VarId(0); num_nodes]; k]; m];
        for q in 0..m {
            for i in 0..k {
                for v in 0..num_nodes {
                    t[q][i][v] = msp.add_var();
                }
            }
        }

        //Clauses

        //  H1a
        println!("c H1");
        for j in 0..n {
            let mut lits: Vec<Lit> = Vec::with_capacity(k);

            for i in 0..k {
                lits.push(l[i][j].pos());
            }

            msp.add_clause(&lits, ClauseKind::Hard);
        }

        //  H1b
        for j in 0..n {
            for i in 0..k - 1 {
                msp.add_clause(&[l[i][j].neg(), s[i][j].pos()], ClauseKind::Hard);
            }

            for i in 1..k - 1 {
                msp.add_clause(&[s[i - 1][j].neg(), s[i][j].pos()], ClauseKind::Hard);
            }

            for i in 1..k {
                msp.add_clause(&[s[i - 1][j].neg(), l[i][j].neg()], ClauseKind::Hard);
            }
        }

        //  H2
        println!("c H2");
        for q in 0..m {
            for v in 0..num_nodes {
                for i in 0..k - 1 {
                    msp.add_clause(&[w[q][i][v].neg(), t[q][i][v].pos()], ClauseKind::Hard);
                }

                for i in 1..k - 1 {
                    msp.add_clause(&[t[q][i - 1][v].neg(), t[q][i][v].pos()], ClauseKind::Hard);
                }

                for i in 1..k {
                    msp.add_clause(&[t[q][i - 1][v].neg(), w[q][i][v].neg()], ClauseKind::Hard);
                }
            }
        }

        //  H3
        println!("c H3");
        for q in 0..m {
            let tree = &reduced.trees[q];
            for i in 0..k {
                for j in 0..n {
                    for k_idx in j + 1..n {
                        let label_j = (j + 1) as Label;
                        let label_k = (k_idx + 1) as Label;
                        let node_j = tree.node_by_label(label_j);
                        let node_k = tree.node_by_label(label_k);
                        let nca = tree.nearest_common_ancestor(node_j, node_k);

                        let mut cur = node_j;
                        while cur != nca {
                            if !tree.is_leaf(cur) {
                                msp.add_clause(
                                    &[
                                        l[i][j].neg(),
                                        l[i][k_idx].neg(),
                                        w[q][i][cur as usize].pos(),
                                    ],
                                    ClauseKind::Hard,
                                );
                            }
                            cur = tree.parent[cur as usize];
                        }

                        cur = node_k;
                        while cur != nca {
                            if !tree.is_leaf(cur) {
                                msp.add_clause(
                                    &[
                                        l[i][j].neg(),
                                        l[i][k_idx].neg(),
                                        w[q][i][cur as usize].pos(),
                                    ],
                                    ClauseKind::Hard,
                                );
                            }
                            cur = tree.parent[cur as usize];
                        }

                        msp.add_clause(
                            &[
                                l[i][j].neg(),
                                l[i][k_idx].neg(),
                                w[q][i][nca as usize].pos(),
                            ],
                            ClauseKind::Hard,
                        );
                    }
                }
            }
        }

        //  H4
        println!("c H4");
        for a in 0..n {
            for b in a + 1..n {
                for c in b + 1..n {
                    let label_a = (a + 1) as Label;
                    let label_b = (b + 1) as Label;
                    let label_c = (c + 1) as Label;

                    let mut impossible = false;
                    let mut first_odd: Option<Label> = None;
                    for tree in &reduced.trees {
                        let node_a = tree.node_by_label(label_a);
                        let node_b = tree.node_by_label(label_b);
                        let node_c = tree.node_by_label(label_c);
                        let nca_ab = tree.nearest_common_ancestor(node_a, node_b);
                        let nca_ac = tree.nearest_common_ancestor(node_a, node_c);
                        let nca_bc = tree.nearest_common_ancestor(node_b, node_c);

                        let depth_ab = tree.depth[nca_ab as usize];
                        let depth_ac = tree.depth[nca_ac as usize];
                        let depth_bc = tree.depth[nca_bc as usize];

                        let odd_leaf = if depth_ab > depth_ac && depth_ab > depth_bc {
                            label_c
                        } else if depth_ac > depth_ab && depth_ac > depth_bc {
                            label_b
                        } else {
                            label_a
                        };

                        if let Some(first) = first_odd {
                            if odd_leaf != first {
                                impossible = true;
                                break;
                            }
                        } else {
                            first_odd = Some(odd_leaf);
                        }
                    }

                    if impossible {
                        for i in 0..k {
                            msp.add_clause(
                                &[l[i][a].neg(), l[i][b].neg(), l[i][c].neg()],
                                ClauseKind::Hard,
                            );
                        }
                    }
                }
            }
        }

        //  H5
        println!("c H5");
        for i in 0..k {
            for j in 0..n {
                msp.add_clause(&[l[i][j].neg(), u[i].pos()], ClauseKind::Hard);
            }
        }

        //  S1
        println!("c S1");
        for i in 0..k {
            msp.add_clause(&[u[i].neg()], ClauseKind::Soft { weight: 1.0 });
        }

        //  O1
        println!("c O1");
        for i in 0..k - 1 {
            msp.add_clause(&[u[i + 1].neg(), u[i].pos()], ClauseKind::Hard);
        }

        //  O2
        println!("c O2");
        msp.add_clause(&[l[0][0].pos()], ClauseKind::Hard);

        //  O3
        println!("c O3");
        for i in 0..k - 1 {
            for j in 0..n {
                let mut lits: Vec<Lit> = Vec::new();

                lits.push(l[i + 1][j].neg());

                for jp in 0..j {
                    lits.push(l[i][jp].pos());
                }

                msp.add_clause(&lits, ClauseKind::Hard);
            }
        }

        println!("c Building DIMACS");

        let mut dimacs = Vec::new();
        msp.write_dimacs(&mut Cursor::new(&mut dimacs))
            .expect("Failed to write DIMACS");

        println!("c Starting open-wbo");

        let mut child = Command::new(SOLVER_PATH)
            .args(SOLVER_ARGS)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("Failed to spawn solver");

        self.child_pid.store(child.id() as u32, Ordering::SeqCst);

        use std::io::Write;
        child
            .stdin
            .as_mut()
            .unwrap()
            .write_all(&dimacs)
            .expect("Failed to write to solver stdin");
        drop(child.stdin.take());

        let output = child.wait_with_output().expect("Failed to wait for solver");
        self.child_pid.store(0, Ordering::SeqCst);
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();

        let true_vars = parse_solution(&stdout);

        let mut components: Vec<Vec<usize>> = vec![Vec::new(); k];
        for i in 0..k {
            for j in 0..n {
                let var_id = l[i][j].0;
                if true_vars.contains(&var_id) {
                    components[i].push(j);
                }
            }
        }

        let mut reduced_components: Vec<Tree> = Vec::new();
        for comp_leaves in components {
            if comp_leaves.is_empty() {
                continue;
            }
            let labels: Vec<Label> = comp_leaves.iter().map(|&j| (j + 1) as Label).collect();
            let tree = extract_induced_subtree(&reduced.trees[0], &labels);
            reduced_components.push(tree);
        }

        // Expand solution back to original label space.
        let result = kernelize::expand_solution(
            reduced_components,
            &kern,
            &instance.trees[0],
            instance.num_leaves,
        );

        let _ = param_reduction; // accounted for by kernelize::expand_solution
        return Some(result);
    }
}

fn parse_solution(output: &str) -> std::collections::HashSet<usize> {
    let mut true_vars = std::collections::HashSet::new();
    for line in output.lines() {
        if line.starts_with("v ") {
            for lit in line[2..].split_whitespace() {
                if let Ok(var) = lit.parse::<isize>() {
                    if var > 0 {
                        true_vars.insert((var - 1) as usize);
                    }
                }
            }
        }
    }
    true_vars
}

fn extract_induced_subtree(tree: &Tree, labels: &[Label]) -> Tree {
    use klados_core::NONE;

    if labels.len() == 1 {
        return build_single_leaf_tree(labels[0]);
    }

    let label_set: std::collections::HashSet<Label> = labels.iter().copied().collect();

    fn build_subtree(
        tree: &Tree,
        node: klados_core::NodeId,
        label_set: &std::collections::HashSet<Label>,
    ) -> Option<Vec<Label>> {
        if tree.is_leaf(node) {
            if let Some(label) = tree.leaf_label(node) {
                if label_set.contains(&label) {
                    return Some(vec![label]);
                }
            }
            return None;
        }

        let left = tree.left[node as usize];
        let right = tree.right[node as usize];

        let left_result = if left != NONE {
            build_subtree(tree, left, label_set)
        } else {
            None
        };
        let right_result = if right != NONE {
            build_subtree(tree, right, label_set)
        } else {
            None
        };

        match (left_result, right_result) {
            (Some(l), Some(r)) => {
                let mut result = vec![0];
                result.extend(l);
                result.extend(r);
                Some(result)
            }
            (Some(l), None) => Some(l),
            (None, Some(r)) => Some(r),
            (None, None) => None,
        }
    }

    let mut nca = tree.node_by_label(labels[0]);
    for &label in &labels[1..] {
        let node = tree.node_by_label(label);
        nca = tree.nearest_common_ancestor(nca, node);
    }

    let subtree_labels = build_subtree(tree, nca, &label_set).unwrap_or_else(|| labels.to_vec());
    build_tree_from_labels(&subtree_labels)
}

fn build_tree_from_labels(labels: &[Label]) -> Tree {
    use pace26io::binary_tree::TopDownCursor;

    struct LabelsCursor<'a> {
        labels: &'a [Label],
        pos: usize,
    }

    impl<'a> TopDownCursor for LabelsCursor<'a> {
        fn children(&self) -> Option<(Self, Self)> {
            if self.labels[self.pos] != 0 {
                return None;
            }

            let mut pos = self.pos + 1;
            let start_left = pos;
            let mut depth = 0;
            loop {
                if self.labels[pos] == 0 {
                    depth += 1;
                } else {
                    depth -= 1;
                }
                pos += 1;
                if depth < 0 {
                    break;
                }
            }
            let end_left = pos;

            Some((
                LabelsCursor {
                    labels: self.labels,
                    pos: start_left,
                },
                LabelsCursor {
                    labels: self.labels,
                    pos: end_left,
                },
            ))
        }

        fn leaf_label(&self) -> Option<pace26io::binary_tree::Label> {
            if self.labels[self.pos] != 0 {
                Some(pace26io::binary_tree::Label(self.labels[self.pos]))
            } else {
                None
            }
        }
    }

    let max_label = labels
        .iter()
        .filter(|&&l| l != 0)
        .max()
        .copied()
        .unwrap_or(1);
    let cursor = LabelsCursor { labels, pos: 0 };
    Tree::from_cursor(cursor, max_label)
}

fn build_single_leaf_tree(label: Label) -> Tree {
    use pace26io::binary_tree::TopDownCursor;

    struct LeafCursor {
        label: Label,
    }

    impl TopDownCursor for LeafCursor {
        fn children(&self) -> Option<(Self, Self)> {
            None
        }

        fn leaf_label(&self) -> Option<pace26io::binary_tree::Label> {
            Some(pace26io::binary_tree::Label(self.label))
        }
    }

    Tree::from_cursor(LeafCursor { label }, label)
}

impl HeuristicSolver for MaxSatSolver {
    fn name(&self) -> &'static str {
        "max-sat"
    }

    fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>> {
        MaxSatSolver::solve(self, instance)
    }

    fn stats(&self) -> &SolverStats {
        static EMPTY: SolverStats = SolverStats {
            nodes_explored: 0,
            branches_pruned: 0,
            lower_bound: 0,
            upper_bound: None,
        };
        &EMPTY
    }

    fn sigterm_handler(&self) {
        let pid = self.child_pid.load(Ordering::SeqCst);
        if pid != 0 {
            unsafe {
                libc::kill(pid as i32, libc::SIGTERM);
            }
        }
    }
}
