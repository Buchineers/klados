//! Instance list resolution for batch processing.
//!
//! Provides uniform resolution of STRIDE `.lst` files containing instance
//! digests, and wrapping raw paths with optional base directory resolution.

use std::path::{Path, PathBuf};

/// A resolved instance entry: digest (STRIDE hash) and absolute path to the file.
#[derive(Clone, Debug)]
pub struct InstanceEntry {
    /// STRIDE digest (e.g. `"a1b2c3...zz"`)
    pub digest: String,
    /// Absolute filesystem path to the instance file
    pub path: PathBuf,
}

/// Parse a `.lst` file containing STRIDE digests (one per line, `s:` prefix optional).
///
/// Resolves each digest to a path under `base_dir/stride-downloads/XX/XX/XXXX...`
/// where `XX/XX/` are the first 4 chars of the digest.
pub fn parse_list_file(list_file: &Path) -> Result<(Vec<InstanceEntry>, PathBuf), Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(list_file)?;
    let base_dir = list_file.parent().unwrap_or(Path::new("."));

    let entries: Vec<InstanceEntry> = content
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                return None;
            }
            let digest = line.strip_prefix("s:").unwrap_or(line).to_string();
            let rel = format!(
                "stride-downloads/{}/{}/{}",
                &digest[..2],
                &digest[2..4],
                &digest[4..]
            );
            let path = {
                let p = base_dir.join(&rel);
                if p.exists() { p } else { PathBuf::from(&rel) }
            };
            Some(InstanceEntry { digest, path })
        })
        .collect();

    Ok((entries, base_dir.to_path_buf()))
}

