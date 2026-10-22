//! pychromy: Chromy's sequential sampling, from Python.
//!
//! The engine is the dependency-free crate `chromy-core`, shared with the R
//! package `chromy`. The engine takes its randomness as arguments, so this
//! binding carries its own small generator (SplitMix64): a `seed` makes a
//! selection exactly reproducible within Python. R and Python reproduce
//! themselves, not each other — each language consumes its own generator.
//!
//! Matrices arrive as lists of columns, so a caller with pandas writes
//! `[df[c].tolist() for c in cols]` and a caller with plain lists writes
//! them directly; nothing here needs numpy.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use chromy_core as core;

fn err<T>(r: Result<T, core::Error>) -> PyResult<T> {
    r.map_err(|e| PyValueError::new_err(e.to_string()))
}

/// SplitMix64: deterministic uniforms from a seed, documented and fixed, so
/// a published design can name its seed and be replayed.
struct Rng(u64);

impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
    fn uniforms(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.uniform()).collect()
    }
}

fn head_mode(head: &str) -> PyResult<core::Head> {
    match head {
        "all" => Ok(core::Head::All),
        "residual" => Ok(core::Head::Residual),
        other => Err(PyValueError::new_err(format!(
            "head must be 'all' or 'residual', not '{other}'"
        ))),
    }
}

fn columns_to_matrix(columns: Vec<Vec<f64>>) -> PyResult<(Vec<f64>, usize)> {
    if columns.is_empty() {
        return Err(PyValueError::new_err("at least one column is needed"));
    }
    let n_rows = columns[0].len();
    if columns.iter().any(|c| c.len() != n_rows) {
        return Err(PyValueError::new_err("the columns have different lengths"));
    }
    let mut flat = Vec::with_capacity(n_rows * columns.len());
    for column in &columns {
        flat.extend_from_slice(column);
    }
    Ok((flat, n_rows))
}

fn keys_to_core(keys: &[Bound<'_, PyAny>]) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<String>>, Vec<(bool, usize)>)> {
    let mut numeric: Vec<Vec<f64>> = Vec::new();
    let mut categorical: Vec<Vec<String>> = Vec::new();
    let mut shape: Vec<(bool, usize)> = Vec::new();
    for key in keys {
        if let Ok(values) = key.extract::<Vec<String>>() {
            shape.push((false, categorical.len()));
            categorical.push(values);
        } else if let Ok(values) = key.extract::<Vec<f64>>() {
            shape.push((true, numeric.len()));
            numeric.push(values);
        } else {
            return Err(PyValueError::new_err(
                "each sort key must be a sequence of strings or of numbers",
            ));
        }
    }
    Ok((numeric, categorical, shape))
}

/// The selection itself, deterministic given the uniforms.
///
/// `lam` holds the expected hits in list order; `uniforms` one draw in
/// [0, 1) per unit; `priority` (0-based, lower wins) breaks the tie when
/// more than one unit could absorb the floating-point drift. Returns a dict
/// with the realised `hits`, the `effective_lambda` the selection actually
/// used (the denominators of the weights), and the accounting.
#[pyfunction]
#[pyo3(signature = (lam, uniforms, priority=None))]
fn pmr(
    py: Python<'_>,
    lam: Vec<f64>,
    uniforms: Vec<f64>,
    priority: Option<Vec<usize>>,
) -> PyResult<Py<PyDict>> {
    let priority = priority.unwrap_or_else(|| (0..lam.len()).collect());
    let s = err(core::pmr(&lam, &uniforms, &priority))?;
    let out = PyDict::new(py);
    out.set_item("hits", s.hits)?;
    out.set_item("effective_lambda", s.effective_lambda)?;
    out.set_item("sample_size", s.sample_size)?;
    out.set_item("eps_sum", s.eps_sum)?;
    out.set_item("corrected_unit", s.corrected_unit)?;
    out.set_item("adjusted_crossings", s.adjusted_crossings)?;
    out.set_item("max_adjustment", s.max_adjustment)?;
    Ok(out.into())
}

/// Hierarchical serpentine (or nested) order over the sort keys.
///
/// `keys` is a list of columns, outermost first, each a sequence of strings
/// or of numbers; string keys rank by byte order, never by locale. Returns
/// the 0-based order in which to list the frame.
#[pyfunction]
#[pyo3(signature = (keys, method="serpentine"))]
fn serpentine_order(keys: Vec<Bound<'_, PyAny>>, method: &str) -> PyResult<Vec<usize>> {
    let (numeric, categorical, shape) = keys_to_core(&keys)?;
    let core_keys: Vec<core::Key> = shape
        .iter()
        .map(|&(is_num, i)| {
            if is_num {
                core::Key::Numeric(&numeric[i])
            } else {
                core::Key::Categorical(&categorical[i])
            }
        })
        .collect();
    match method {
        "serpentine" => err(core::serpentine(&core_keys)),
        "nested" => err(core::nested(&core_keys)),
        other => Err(PyValueError::new_err(format!(
            "method must be 'serpentine' or 'nested', not '{other}'"
        ))),
    }
}

/// The maximal (MPPS) size measure over several auxiliary variables.
///
/// `columns` is a list of non-negative numeric columns, one per variable;
/// `n_j` the relative sample size each variable is to command (one positive
/// factor per column, equal when omitted). Returns one size per unit.
#[pyfunction]
#[pyo3(signature = (columns, n_j=None))]
fn mpps_size(columns: Vec<Vec<f64>>, n_j: Option<Vec<f64>>) -> PyResult<Vec<f64>> {
    let n_cols = columns.len();
    let (flat, n_rows) = columns_to_matrix(columns)?;
    let n_j = n_j.unwrap_or_else(|| vec![1.0; n_cols]);
    err(core::mpps(&flat, n_rows, &n_j))
}

/// Expected hits proportional to size, summing to `n`.
///
/// Certainty units are resolved by iteration; with `min_replacement` the
/// expected hits may exceed one and a unit may be selected more than once.
#[pyfunction]
#[pyo3(signature = (size, n, min_replacement=false))]
fn expected_hits(size: Vec<f64>, n: u64, min_replacement: bool) -> PyResult<Vec<f64>> {
    err(core::expected_hits(&size, n, min_replacement))
}

/// The allocation that equalises anticipated coefficients of variation.
///
/// Searches for the per-variable factors under which every variable's total
/// is anticipated with the same CV. Returns a dict with the factors `n_j`,
/// the anticipated `cv` of each informative column, `used`, `iterations`
/// and `converged`.
#[pyfunction]
#[pyo3(signature = (columns, n, max_iterations=300, tolerance=1e-3, damping=0.5))]
fn eqcv_allocation(
    py: Python<'_>,
    columns: Vec<Vec<f64>>,
    n: u64,
    max_iterations: usize,
    tolerance: f64,
    damping: f64,
) -> PyResult<Py<PyDict>> {
    let (flat, n_rows) = columns_to_matrix(columns)?;
    let a = err(core::eqcv(
        &flat,
        n_rows,
        n,
        core::EqcvOptions {
            max_iterations,
            tolerance,
            damping,
        },
    ))?;
    let out = PyDict::new(py);
    out.set_item("n_j", a.n_j)?;
    out.set_item("cv", a.cv)?;
    out.set_item("used", a.used)?;
    out.set_item("iterations", a.iterations)?;
    out.set_item("converged", a.converged)?;
    Ok(out.into())
}

fn selection_dict(py: Python<'_>, s: &core::Selected) -> PyResult<Py<PyDict>> {
    let out = PyDict::new(py);
    out.set_item("head", s.head)?;
    out.set_item("list", &s.list)?;
    out.set_item("hits", &s.hits)?;
    out.set_item("effective_lambda", &s.effective_lambda)?;
    match &s.accounting {
        Some(a) => {
            let acc = PyDict::new(py);
            acc.set_item("eps_sum", a.eps_sum)?;
            acc.set_item("corrected_unit", a.corrected_unit)?;
            acc.set_item("adjusted_crossings", a.adjusted_crossings)?;
            acc.set_item("max_adjustment", a.max_adjustment)?;
            out.set_item("accounting", acc)?;
        }
        None => out.set_item("accounting", py.None())?,
    }
    Ok(out.into())
}

/// The full procedure: a random head, the rotated list, the selection.
///
/// `lam` holds the expected hits per frame unit (from `expected_hits`);
/// `order` the 0-based listing (from `serpentine_order`), the frame's own
/// order when omitted. The head is drawn with probability proportional to
/// expected hits among the eligible units — every unit with `head='all'`,
/// only fractional ones with `head='residual'`. Give a `seed` for a
/// reproducible selection, or `uniforms` (and `u_head`) to control every
/// draw yourself. Returns a dict with the `head` position, the rotated
/// `list` of frame units, their `hits`, the `effective_lambda` per frame
/// unit, and the floating-point `accounting`.
#[pyfunction]
#[pyo3(signature = (lam, order=None, head="all", seed=None, u_head=None, uniforms=None))]
fn select(
    py: Python<'_>,
    lam: Vec<f64>,
    order: Option<Vec<usize>>,
    head: &str,
    seed: Option<u64>,
    u_head: Option<f64>,
    uniforms: Option<Vec<f64>>,
) -> PyResult<Py<PyDict>> {
    let order = order.unwrap_or_else(|| (0..lam.len()).collect());
    let mode = head_mode(head)?;
    let needed = core::required_uniforms(&lam, &order, mode);
    let mut rng = seed.map(Rng);
    let u_head = match (u_head, &mut rng) {
        (Some(u), _) => u,
        (None, Some(r)) => r.uniform(),
        (None, None) => {
            if uniforms.is_none() {
                return Err(PyValueError::new_err(
                    "give a seed, or u_head and uniforms: the selection needs randomness from somewhere",
                ));
            }
            return Err(PyValueError::new_err(
                "uniforms were given without u_head; the head needs its own draw",
            ));
        }
    };
    let uniforms = match (uniforms, &mut rng) {
        (Some(u), _) => u,
        (None, Some(r)) => r.uniforms(needed),
        (None, None) => unreachable!(),
    };
    let s = err(core::select(&lam, &order, mode, u_head, &uniforms))?;
    selection_dict(py, &s)
}

/// A whole design in one call: size to expected hits, keys to order,
/// selection to weighted sample.
///
/// `size` is the size measure (one non-negative value per unit); `keys`,
/// when given, lists the sort columns for the serpentine (or nested) order.
/// Returns a dict of parallel lists over the *selected* units — `unit` the
/// frame index, `position` in the selection list, `expected_hits`, `hits`,
/// `weight` — plus the full `order`, `head` and `accounting`.
#[pyfunction]
#[pyo3(signature = (size, n, keys=None, method="serpentine", head="all", min_replacement=false, seed=None))]
#[allow(clippy::too_many_arguments)]
fn sample(
    py: Python<'_>,
    size: Vec<f64>,
    n: u64,
    keys: Option<Vec<Bound<'_, PyAny>>>,
    method: &str,
    head: &str,
    min_replacement: bool,
    seed: Option<u64>,
) -> PyResult<Py<PyDict>> {
    let lam = err(core::expected_hits(&size, n, min_replacement))?;
    let order = match keys {
        Some(keys) => serpentine_order(keys, method)?,
        None => (0..lam.len()).collect(),
    };
    let mode = head_mode(head)?;
    let seed = seed.ok_or_else(|| {
        PyValueError::new_err("sample needs a seed; use select() to pass uniforms yourself")
    })?;
    let mut rng = Rng(seed);
    let u_head = rng.uniform();
    let uniforms = rng.uniforms(core::required_uniforms(&lam, &order, mode));
    let s = err(core::select(&lam, &order, mode, u_head, &uniforms))?;

    let drawn: Vec<usize> = (0..s.hits.len()).filter(|&p| s.hits[p] > 0).collect();
    let out = PyDict::new(py);
    out.set_item("unit", drawn.iter().map(|&p| s.list[p]).collect::<Vec<_>>())?;
    out.set_item("position", &drawn)?;
    out.set_item(
        "expected_hits",
        drawn
            .iter()
            .map(|&p| s.effective_lambda[s.list[p]])
            .collect::<Vec<_>>(),
    )?;
    out.set_item("hits", drawn.iter().map(|&p| s.hits[p]).collect::<Vec<_>>())?;
    out.set_item(
        "weight",
        drawn
            .iter()
            .map(|&p| 1.0 / s.effective_lambda[s.list[p]])
            .collect::<Vec<_>>(),
    )?;
    out.set_item("order", s.list.clone())?;
    out.set_item("head", s.head)?;
    match &s.accounting {
        Some(a) => {
            let acc = PyDict::new(py);
            acc.set_item("eps_sum", a.eps_sum)?;
            acc.set_item("corrected_unit", a.corrected_unit)?;
            acc.set_item("adjusted_crossings", a.adjusted_crossings)?;
            acc.set_item("max_adjustment", a.max_adjustment)?;
            out.set_item("accounting", acc)?;
        }
        None => out.set_item("accounting", py.None())?,
    }
    Ok(out.into())
}

/// The multiplicity estimator with successive-difference variance.
///
/// `y_columns` holds the study variables of the *selected* units, one
/// column per variable, in the same order as the sample lists from
/// `sample()`. Returns a dict with the estimated `total` per column, the
/// `se` (None when only one hit was random), and `random_draws`.
#[pyfunction]
#[pyo3(signature = (y_columns, positions, expected_hits, hits))]
fn estimate_total(
    py: Python<'_>,
    y_columns: Vec<Vec<f64>>,
    positions: Vec<usize>,
    expected_hits: Vec<f64>,
    hits: Vec<u64>,
) -> PyResult<Py<PyDict>> {
    let (flat, n_rows) = columns_to_matrix(y_columns)?;
    let units: Vec<usize> = (0..n_rows).collect();
    let e = err(core::multiplicity_estimate(
        &flat,
        n_rows,
        &units,
        &positions,
        &expected_hits,
        &hits,
    ))?;
    let se = PyList::empty(py);
    for v in &e.sd_variance {
        if v.is_nan() {
            se.append(py.None())?;
        } else {
            se.append(v.sqrt())?;
        }
    }
    let out = PyDict::new(py);
    out.set_item("total", e.total)?;
    out.set_item("se", se)?;
    out.set_item("random_draws", e.random_draws)?;
    Ok(out.into())
}

#[pymodule]
fn pychromy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pmr, m)?)?;
    m.add_function(wrap_pyfunction!(serpentine_order, m)?)?;
    m.add_function(wrap_pyfunction!(mpps_size, m)?)?;
    m.add_function(wrap_pyfunction!(expected_hits, m)?)?;
    m.add_function(wrap_pyfunction!(eqcv_allocation, m)?)?;
    m.add_function(wrap_pyfunction!(select, m)?)?;
    m.add_function(wrap_pyfunction!(sample, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_total, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
