//! Usage:
//! ```ignore
//! let mut problem = MaxSatProblem::new();
//! let x = problem.add_var();
//! let y = problem.add_var();
//! let z = problem.add_var();
//!
//! // Add clauses with weights (higher weight = more important)
//! problem.add_clause(&[x.pos(), y.neg()], 1.0);      // x OR NOT y
//! problem.add_clause(&[y.pos(), z.pos()], 2.0);      // y OR z
//! problem.add_clause(&[x.neg(), z.neg()], 1.0);      // NOT x OR NOT z
//!

use std::io::{self, Write};
use std::ops::Range;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VarId(pub usize);

#[derive(Clone, Copy, Debug)]
pub struct Lit {
    pub var: VarId,
    pub polarity: bool,
}

impl VarId {
    pub fn pos(self) -> Lit {
        Lit {
            var: self,
            polarity: true,
        }
    }

    pub fn neg(self) -> Lit {
        Lit {
            var: self,
            polarity: false,
        }
    }
}

#[allow(dead_code)]
impl Lit {
    pub fn positive(var: VarId) -> Self {
        var.pos()
    }

    pub fn negative(var: VarId) -> Self {
        var.neg()
    }
}

#[derive(Clone, Debug)]
pub enum ClauseKind {
    Hard,
    Soft { weight: f64 },
}

#[derive(Clone, Debug)]
pub struct Clause {
    pub lits: Vec<Lit>,
    pub kind: ClauseKind,
}

pub struct MaxSatProblem {
    num_vars: usize,
    clauses: Vec<Clause>,
}

impl Default for MaxSatProblem {
    fn default() -> Self {
        Self::new()
    }
}

impl MaxSatProblem {
    pub fn new() -> Self {
        Self {
            num_vars: 0,
            clauses: Vec::new(),
        }
    }

    #[allow(dead_code)]
    pub fn with_capacity(_num_vars: usize, num_clauses: usize) -> Self {
        Self {
            num_vars: 0,
            clauses: Vec::with_capacity(num_clauses),
        }
    }

    pub fn add_var(&mut self) -> VarId {
        let id = VarId(self.num_vars);
        self.num_vars += 1;
        id
    }

    #[allow(dead_code)]
    pub fn add_vars(&mut self, n: usize) -> Range<VarId> {
        let start = self.num_vars;
        self.num_vars += n;
        VarId(start)..VarId(self.num_vars)
    }

    pub fn add_clause(&mut self, lits: &[Lit], kind: ClauseKind) {
        self.clauses.push(Clause {
            lits: lits.to_vec(),
            kind,
        });
    }

    #[allow(dead_code)]
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    #[allow(dead_code)]
    pub fn num_clauses(&self) -> usize {
        self.clauses.len()
    }

    pub fn write_dimacs<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        let num_soft = self
            .clauses
            .iter()
            .filter(|c| matches!(c.kind, ClauseKind::Soft { .. }))
            .count();
        let top = num_soft + 1;

        writeln!(writer, "c MaxSAT instance")?;
        writeln!(
            writer,
            "p wcnf {} {} {}",
            self.num_vars,
            self.clauses.len(),
            top
        )?;

        for clause in &self.clauses {
            let weight = match &clause.kind {
                ClauseKind::Hard => top,
                ClauseKind::Soft { weight } => (*weight as usize).max(1),
            };

            write!(writer, "{}", weight)?;
            for lit in &clause.lits {
                let var_num = lit.var.0 + 1;
                let lit_num: isize = if lit.polarity {
                    var_num as isize
                } else {
                    -(var_num as isize)
                };
                write!(writer, " {}", lit_num)?;
            }
            writeln!(writer, " 0")?;
        }

        Ok(())
    }
}
