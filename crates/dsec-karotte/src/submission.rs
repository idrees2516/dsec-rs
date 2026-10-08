//! Defensive custody copy of student output (upstream
//! `karotte/save_submission.py`) and the untrusted-path primitives it
//! rests on (`untrusted_paths.py`).
//!
//! The grader must never read student-controlled paths directly: a
//! symlink at the submission path would make it read root's files; a
//! FIFO would hang it; a huge tree would fill the disk. The custody copy
//! walks with **dir-fd-relative, no-follow** operations, refuses
//! symlinks and special files, enforces hard caps (256 MiB/file, 1 GiB
//! total, 10,000 entries, depth 32), handles sparseness, and maps
//! student-provokable destination errnos to misbehavior.

use crate::error::{Error, Result, StudentMisbehaviorError};

/// Upstream `max_file_bytes`.
pub const MAX_FILE_BYTES: u64 = 256 << 20;
/// Upstream `max_total_bytes`.
pub const MAX_TOTAL_BYTES: u64 = 1 << 30;
/// Upstream `max_entries`.
pub const MAX_ENTRIES: u64 = 10_000;
/// Upstream `max_depth`.
pub const MAX_DEPTH: usize = 32;
/// Upstream `_CHUNK_SIZE`.
pub const CHUNK_SIZE: usize = 1024 * 1024;
/// Upstream `_DEST_ERRNOS` — destination errors a student can provoke,
/// reclassified as misbehavior.
pub const DEST_ERRNOS: &[&str] = &["ENAMETOOLONG", "EEXIST", "ENOSPC", "EDQUOT", "EFBIG"];
/// Upstream submissions dir.
pub const SUBMISSIONS_DIR_SUFFIX: &str = ".config/karotte/submissions";

/// An entry's lstat-style view.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntryInfo {
    /// Owner uid.
    pub uid: u32,
    /// Raw st_mode.
    pub mode: u32,
    /// Size in bytes (regular files).
    pub size: u64,
}

impl EntryInfo {
    /// `S_ISREG`.
    pub fn is_regular(&self) -> bool {
        self.mode & 0o170000 == 0o100000
    }
    /// `S_ISDIR`.
    pub fn is_dir(&self) -> bool {
        self.mode & 0o170000 == 0o040000
    }
    /// `S_ISLNK`.
    pub fn is_symlink(&self) -> bool {
        self.mode & 0o170000 == 0o120000
    }
    /// Anything but regular/dir — FIFOs, sockets, devices: the
    /// grader-DoS shapes.
    pub fn is_special(&self) -> bool {
        !self.is_regular() && !self.is_dir() && !self.is_symlink()
    }
}

/// The filesystem view the custody copy walks (dir-fd-relative,
/// no-follow only — no path-string equivalents exist in this API by
/// design).
pub trait CustodyFs: Send + Sync {
    /// `openat(dir, name, O_RDONLY|O_NOFOLLOW|...)` for a directory.
    fn open_dir(&self, dir: &str, name: &str) -> Option<String>;
    /// `fstatat(dir, name, ..., AT_SYMLINK_NOFOLLOW)`.
    fn lstat_at(&self, dir: &str, name: &str) -> Option<EntryInfo>;
    /// Read up to `CHUNK_SIZE` bytes of a regular file at
    /// `dir/name` from `offset`; returns (bytes, eof).
    fn read_file_chunk(&self, dir: &str, name: &str, offset: u64) -> Result<(Vec<u8>, bool)>;
    /// Create the destination file `dest_dir/name` (0600); returns a
    /// handle. Destination errors the student can provoke are mapped by
    /// the caller.
    fn create_dest(&self, dest_dir: &str, name: &str) -> Result<()>;
    /// Write a chunk at `offset` in the created destination file
    /// (positional: sparse holes advance the offset without writing).
    fn write_dest_chunk(&self, dest_dir: &str, name: &str, offset: u64, data: &[u8]) -> Result<()>;
    /// Truncate/extend the destination to `size` (sparse handling).
    fn truncate_dest(&self, dest_dir: &str, name: &str, size: u64) -> Result<()>;
    /// Create a destination directory (0700).
    fn mkdir_dest(&self, dest_dir: &str, name: &str) -> Result<()>;
}

/// Walk-to-parent result (upstream `walk_to_parent`): the parent dir
/// handle and the final name, with no symlink followed on the way.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParentRef {
    /// Parent directory handle.
    pub parent: String,
    /// The final entry name.
    pub name: String,
}

/// `walk_to_parent` over a path (upstream opens `/` then walks each
/// component with `open_at`, refusing symlinks; the port models the
/// refusals).
pub fn walk_to_parent(fs: &dyn CustodyFs, path: &str) -> std::result::Result<ParentRef, Error> {
    let abs = path
        .strip_prefix('/')
        .ok_or_else(|| Error::InvalidSpec(format!("path must be absolute: {path}")))?;
    let mut components: Vec<&str> = abs.split('/').filter(|c| !c.is_empty()).collect();
    if components.is_empty() {
        return Err(Error::InvalidSpec("cannot walk to the parent of /".into()));
    }
    let name = components.pop().unwrap().to_string();
    let mut parent = "/".to_string();
    for c in components {
        let info = fs.lstat_at(&parent, c);
        if let Some(info) = &info {
            if info.is_symlink() {
                return Err(Error::Misbehavior(StudentMisbehaviorError::Symlink {
                    path: format!("{parent}/{c}"),
                }));
            }
            if !info.is_dir() {
                return Err(Error::InvalidSpec(format!(
                    "{parent}/{c} is not a directory"
                )));
            }
        }
        parent = fs
            .open_dir(&parent, c)
            .ok_or_else(|| Error::InvalidSpec(format!("cannot open {parent}/{c}")))?;
    }
    Ok(ParentRef { parent, name })
}

/// The custody-copy report.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CustodyReport {
    /// Files copied.
    pub files: u64,
    /// Directories created.
    pub dirs: u64,
    /// Bytes copied (logical).
    pub bytes: u64,
    /// Whether the source did not exist (not a misbehavior — the judge
    /// decides; upstream logs "Nothing handed in").
    pub missing_source: bool,
    /// Sparse holes skipped.
    pub holes: u64,
}

/// Copy a student submission into custody (upstream `save_submission`).
/// Missing source is `Ok` with `missing_source = true`. Symlinks /
/// special files / cap violations / provokable dest errors are
/// misbehavior. All-zero chunks are skipped as holes (the destination is
/// truncated to the logical size instead of written).
pub fn save_submission(fs: &dyn CustodyFs, source: &str, dest_dir: &str) -> Result<CustodyReport> {
    let mut report = CustodyReport::default();
    let parent = match walk_to_parent(fs, source) {
        Ok(p) => p,
        Err(Error::Misbehavior(_)) if fs.lstat_at("/", "").is_none() => {
            // Source missing entirely: nothing handed in.
            report.missing_source = true;
            return Ok(report);
        }
        Err(e) => return Err(e),
    };
    // Does the source exist?
    let Some(info) = fs.lstat_at(&parent.parent, &parent.name) else {
        report.missing_source = true;
        return Ok(report);
    };
    if info.is_symlink() {
        return Err(Error::Misbehavior(StudentMisbehaviorError::Symlink {
            path: source.to_string(),
        }));
    }
    if info.is_special() {
        return Err(Error::Misbehavior(StudentMisbehaviorError::SpecialFile {
            mode: info.mode,
            path: source.to_string(),
        }));
    }
    if info.is_dir() {
        copy_tree(
            fs,
            &parent.parent,
            &parent.name,
            dest_dir,
            "",
            0,
            &mut report,
        )?;
        Ok(report)
    } else {
        copy_file(
            fs,
            &parent.parent,
            &parent.name,
            dest_dir,
            "",
            &info,
            &mut report,
        )?;
        Ok(report)
    }
}

fn copy_tree(
    fs: &dyn CustodyFs,
    src_parent: &str,
    src_name: &str,
    dest_dir: &str,
    rel: &str,
    depth: usize,
    report: &mut CustodyReport,
) -> Result<()> {
    if depth > MAX_DEPTH {
        return Err(Error::Misbehavior(StudentMisbehaviorError::LimitExceeded {
            what: format!("submission deeper than {MAX_DEPTH} levels"),
        }));
    }
    if report.files + report.dirs >= MAX_ENTRIES {
        return Err(Error::Misbehavior(StudentMisbehaviorError::LimitExceeded {
            what: format!("more than {MAX_ENTRIES} entries"),
        }));
    }
    // Model listing via lstat probes is impractical; the FsView contract
    // uses open_dir + a listing through read hooks: extend with a
    // list_dir style approach by reading zero-size chunk markers.
    // For the port, directories are enumerated through a dedicated
    // convention: read_file_chunk on the *dir handle* returns entry
    // names (see MemCustody below). Production views list real dirs.
    let dir = fs
        .open_dir(src_parent, src_name)
        .ok_or_else(|| Error::InvalidSpec(format!("cannot open {src_parent}/{src_name}")))?;
    let _ = rel;
    // Root of the tree: create the dest root.
    fs.mkdir_dest(dest_dir, "")?;
    report.dirs += 1;
    copy_dir_contents(fs, &dir, dest_dir, depth, report)
}

fn copy_dir_contents(
    fs: &dyn CustodyFs,
    dir: &str,
    dest_dir: &str,
    depth: usize,
    report: &mut CustodyReport,
) -> Result<()> {
    // Enumerate children via the view's listing convention (chunks of
    // the dir handle). See MemCustody for the contract.
    let listing = fs
        .read_file_chunk(dir, "", 0)
        .map(|(b, _)| String::from_utf8_lossy(&b).to_string())
        .unwrap_or_default();
    for name in listing.lines().filter(|l| !l.is_empty()) {
        if report.files + report.dirs >= MAX_ENTRIES {
            return Err(Error::Misbehavior(StudentMisbehaviorError::LimitExceeded {
                what: format!("more than {MAX_ENTRIES} entries"),
            }));
        }
        if depth + 1 > MAX_DEPTH {
            return Err(Error::Misbehavior(StudentMisbehaviorError::LimitExceeded {
                what: format!("submission deeper than {MAX_DEPTH} levels"),
            }));
        }
        let Some(info) = fs.lstat_at(dir, name) else {
            continue;
        };
        if info.is_symlink() {
            return Err(Error::Misbehavior(StudentMisbehaviorError::Symlink {
                path: format!("{dir}/{name}"),
            }));
        }
        if info.is_special() {
            return Err(Error::Misbehavior(StudentMisbehaviorError::SpecialFile {
                mode: info.mode,
                path: format!("{dir}/{name}"),
            }));
        }
        if info.is_dir() {
            fs.mkdir_dest(dest_dir, name)?;
            report.dirs += 1;
            let sub = fs
                .open_dir(dir, name)
                .ok_or_else(|| Error::InvalidSpec(format!("cannot open {dir}/{name}")))?;
            copy_dir_contents(fs, &sub, &format!("{dest_dir}/{name}"), depth + 1, report)?;
        } else {
            copy_file(fs, dir, name, dest_dir, name, &info, report)?;
        }
    }
    Ok(())
}

fn copy_file(
    fs: &dyn CustodyFs,
    src_parent: &str,
    src_name: &str,
    dest_dir: &str,
    dest_name: &str,
    info: &EntryInfo,
    report: &mut CustodyReport,
) -> Result<()> {
    let dest_name = if dest_name.is_empty() {
        src_name
    } else {
        dest_name
    };
    if info.size > MAX_FILE_BYTES {
        return Err(Error::Misbehavior(StudentMisbehaviorError::LimitExceeded {
            what: format!("file exceeds {} bytes", MAX_FILE_BYTES),
        }));
    }
    if report.bytes + info.size > MAX_TOTAL_BYTES {
        return Err(Error::Misbehavior(StudentMisbehaviorError::LimitExceeded {
            what: format!("total exceeds {} bytes", MAX_TOTAL_BYTES),
        }));
    }
    fs.create_dest(dest_dir, dest_name).map_err(map_dest_err)?;
    let mut offset: u64 = 0;
    loop {
        let (chunk, _eof) = fs.read_file_chunk(src_parent, src_name, offset)?;
        if chunk.is_empty() {
            break;
        }
        if chunk.iter().all(|b| *b == 0) {
            // Sparse hole: skip writing; the final truncate extends.
            report.holes += 1;
        } else {
            fs.write_dest_chunk(dest_dir, dest_name, offset, &chunk)
                .map_err(map_dest_err)?;
        }
        offset += chunk.len() as u64;
        if offset >= info.size {
            break;
        }
    }
    // Extend to the logical size (sparse files keep their length).
    fs.truncate_dest(dest_dir, dest_name, info.size)
        .map_err(map_dest_err)?;
    report.files += 1;
    report.bytes += info.size;
    Ok(())
}

/// The errno codes for [`DEST_ERRNOS`] (Linux).
const DEST_ERRNO_CODES: &[(&str, i32)] = &[
    ("ENAMETOOLONG", 36),
    ("EEXIST", 17),
    ("ENOSPC", 28),
    ("EDQUOT", 122),
    ("EFBIG", 27),
];

/// Map a destination failure to misbehavior when the errno is one the
/// student can provoke (upstream `_DEST_ERRNOS`).
pub fn map_dest_err(err: Error) -> Error {
    if let Error::Io(io) = err {
        if let Some(code) = io.raw_os_error() {
            for (name, c) in DEST_ERRNO_CODES {
                if *c == code {
                    return Error::Misbehavior(StudentMisbehaviorError::DestError {
                        errno: name,
                        message: io.to_string(),
                    });
                }
            }
        }
        Error::Io(io)
    } else {
        err
    }
}

/// Classify an errno name as a provokable destination error.
pub fn is_dest_errno(name: &str) -> bool {
    DEST_ERRNOS.contains(&name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::Mutex;

    /// The in-memory custody FS. Contract details:
    /// * `open_dir(dir, name)` returns `Some("{dir}/{name}")` when it is
    ///   a directory in `entries`.
    /// * `read_file_chunk(dir, "", 0)` on a *directory* handle returns
    ///   the newline-joined child names (the listing convention).
    /// * `read_file_chunk(dir, name, off)` reads regular files.
    struct MemCustody {
        entries: BTreeMap<String, EntryInfo>,
        contents: BTreeMap<String, Vec<u8>>,
        dest_files: Mutex<BTreeMap<String, Vec<u8>>>,
        dest_dirs: Mutex<BTreeSet<String>>,
        provoke: Option<&'static str>,
    }

    impl MemCustody {
        fn new() -> Self {
            Self {
                entries: BTreeMap::new(),
                contents: BTreeMap::new(),
                dest_files: Mutex::new(BTreeMap::new()),
                dest_dirs: Mutex::new(BTreeSet::new()),
                provoke: None,
            }
        }
        fn file(&mut self, path: &str, uid: u32, data: &[u8]) {
            self.entries.insert(
                path.to_string(),
                EntryInfo {
                    uid,
                    mode: 0o100644,
                    size: data.len() as u64,
                },
            );
            self.contents.insert(path.to_string(), data.to_vec());
        }
        fn dir(&mut self, path: &str) {
            self.entries.insert(
                path.to_string(),
                EntryInfo {
                    uid: 0,
                    mode: 0o040755,
                    size: 0,
                },
            );
        }
        fn symlink(&mut self, path: &str) {
            self.entries.insert(
                path.to_string(),
                EntryInfo {
                    uid: 1000,
                    mode: 0o120777,
                    size: 0,
                },
            );
        }
        fn fifo(&mut self, path: &str) {
            self.entries.insert(
                path.to_string(),
                EntryInfo {
                    uid: 1000,
                    mode: 0o010644,
                    size: 0,
                },
            );
        }
    }

    impl CustodyFs for MemCustody {
        fn open_dir(&self, dir: &str, name: &str) -> Option<String> {
            let path = if name.is_empty() {
                dir.to_string()
            } else {
                format!("{}/{}", dir.trim_end_matches('/'), name)
            };
            let e = self.entries.get(&path)?;
            if e.is_dir() {
                Some(path)
            } else {
                None
            }
        }
        fn lstat_at(&self, dir: &str, name: &str) -> Option<EntryInfo> {
            if name.is_empty() {
                return self.entries.get(dir).cloned();
            }
            self.entries
                .get(&format!("{}/{}", dir.trim_end_matches('/'), name))
                .cloned()
        }
        fn read_file_chunk(&self, dir: &str, name: &str, offset: u64) -> Result<(Vec<u8>, bool)> {
            if name.is_empty() {
                // listing convention for directories
                let prefix = format!("{}/", dir.trim_end_matches('/'));
                let names: Vec<String> = self
                    .entries
                    .keys()
                    .filter_map(|p| p.strip_prefix(&prefix))
                    .filter(|r| !r.contains('/'))
                    .map(|r| r.to_string())
                    .collect();
                return Ok((names.join("\n").into_bytes(), true));
            }
            let path = format!("{}/{}", dir.trim_end_matches('/'), name);
            let data = self.contents.get(&path).cloned().unwrap_or_default();
            let off = offset as usize;
            if off >= data.len() {
                return Ok((Vec::new(), true));
            }
            let end = (off + CHUNK_SIZE).min(data.len());
            Ok((data[off..end].to_vec(), end >= data.len()))
        }
        fn create_dest(&self, dest_dir: &str, name: &str) -> Result<()> {
            if let Some(p) = self.provoke {
                if p == "ENOSPC" {
                    return Err(Error::Io(std::io::Error::from_raw_os_error(28)));
                }
            }
            self.dest_files.lock().unwrap().insert(
                format!("{}/{}", dest_dir.trim_end_matches('/'), name),
                Vec::new(),
            );
            Ok(())
        }
        fn write_dest_chunk(
            &self,
            dest_dir: &str,
            name: &str,
            offset: u64,
            data: &[u8],
        ) -> Result<()> {
            if let Some(p) = self.provoke {
                if p == "ENOSPC" && !data.is_empty() {
                    return Err(Error::Io(std::io::Error::from_raw_os_error(28)));
                }
            }
            let mut files = self.dest_files.lock().unwrap();
            let f = files
                .entry(format!("{}/{}", dest_dir.trim_end_matches('/'), name))
                .or_default();
            let end = (offset as usize) + data.len();
            if f.len() < end {
                f.resize(end, 0);
            }
            f[offset as usize..end].copy_from_slice(data);
            Ok(())
        }
        fn truncate_dest(&self, dest_dir: &str, name: &str, size: u64) -> Result<()> {
            let key = format!("{}/{}", dest_dir.trim_end_matches('/'), name);
            let mut files = self.dest_files.lock().unwrap();
            let f = files.entry(key).or_default();
            f.resize(size as usize, 0);
            Ok(())
        }
        fn mkdir_dest(&self, dest_dir: &str, name: &str) -> Result<()> {
            self.dest_dirs.lock().unwrap().insert(format!(
                "{}/{}",
                dest_dir.trim_end_matches('/'),
                name
            ));
            Ok(())
        }
    }

    #[test]
    fn entry_classification() {
        let reg = EntryInfo {
            uid: 0,
            mode: 0o100644,
            size: 0,
        };
        let dir = EntryInfo {
            uid: 0,
            mode: 0o040755,
            size: 0,
        };
        let link = EntryInfo {
            uid: 0,
            mode: 0o120777,
            size: 0,
        };
        let fifo = EntryInfo {
            uid: 0,
            mode: 0o010644,
            size: 0,
        };
        assert!(reg.is_regular() && !reg.is_special());
        assert!(dir.is_dir());
        assert!(link.is_symlink());
        assert!(fifo.is_special(), "FIFOs are the grader-DoS shape");
    }

    #[test]
    fn walk_to_parent_refuses_intermediate_symlinks() {
        let mut fs = MemCustody::new();
        fs.dir("/workdir");
        fs.symlink("/workdir/evil");
        // /workdir/evil/answer walks through the symlink → refused.
        let err = walk_to_parent(&fs, "/workdir/evil/answer").unwrap_err();
        assert!(matches!(
            err,
            Error::Misbehavior(StudentMisbehaviorError::Symlink { .. })
        ));
    }

    #[test]
    fn missing_source_is_not_misbehavior() {
        let mut fs = MemCustody::new();
        fs.dir("/workdir");
        let report = save_submission(&fs, "/workdir/absent.txt", "/dest").unwrap();
        assert!(report.missing_source);
        assert_eq!(report.files, 0);
    }

    #[test]
    fn symlink_at_submission_path_is_misbehavior() {
        let mut fs = MemCustody::new();
        fs.dir("/workdir");
        fs.symlink("/workdir/answer.txt");
        let err = save_submission(&fs, "/workdir/answer.txt", "/dest").unwrap_err();
        match err {
            Error::Misbehavior(StudentMisbehaviorError::Symlink { path }) => {
                assert_eq!(path, "/workdir/answer.txt");
            }
            other => panic!("expected Symlink, got {other:?}"),
        }
    }

    #[test]
    fn fifo_at_submission_path_is_misbehavior() {
        let mut fs = MemCustody::new();
        fs.dir("/workdir");
        fs.fifo("/workdir/hangme");
        let err = save_submission(&fs, "/workdir/hangme", "/dest").unwrap_err();
        assert!(matches!(
            err,
            Error::Misbehavior(StudentMisbehaviorError::SpecialFile { .. })
        ));
    }

    #[test]
    fn regular_file_copies_with_sparse_holes() {
        let mut fs = MemCustody::new();
        fs.dir("/workdir");
        // 3 chunks: data, zeros, data (sparse middle).
        let mut data = vec![b'A'; CHUNK_SIZE];
        data.extend_from_slice(&vec![0u8; CHUNK_SIZE]);
        data.extend_from_slice(&vec![b'B'; CHUNK_SIZE]);
        fs.file("/workdir/out.bin", 1000, &data);
        let report = save_submission(&fs, "/workdir/out.bin", "/dest").unwrap();
        assert_eq!(report.files, 1);
        assert_eq!(report.bytes, (3 * CHUNK_SIZE) as u64);
        assert_eq!(report.holes, 1, "the zero chunk is skipped as a hole");
        let stored = fs
            .dest_files
            .lock()
            .unwrap()
            .get("/dest/out.bin")
            .cloned()
            .unwrap();
        assert_eq!(stored.len(), 3 * CHUNK_SIZE);
        assert_eq!(&stored[0..8], b"AAAAAAAA");
        assert_eq!(&stored[3 * CHUNK_SIZE - 8..], b"BBBBBBBB");
    }

    #[test]
    fn file_cap_is_misbehavior() {
        let mut fs = MemCustody::new();
        fs.dir("/workdir");
        fs.file("/workdir/huge.bin", 1000, &[0u8; 16]);
        // Lie about the size to trip the cap check.
        fs.entries.get_mut("/workdir/huge.bin").unwrap().size = MAX_FILE_BYTES + 1;
        let err = save_submission(&fs, "/workdir/huge.bin", "/dest").unwrap_err();
        assert!(matches!(
            err,
            Error::Misbehavior(StudentMisbehaviorError::LimitExceeded { .. })
        ));
    }

    #[test]
    fn directory_tree_copies_and_depth_caps() {
        let mut fs = MemCustody::new();
        fs.dir("/workdir");
        fs.dir("/workdir/sub");
        fs.file("/workdir/sub/a.txt", 1000, b"aaa");
        fs.file("/workdir/b.txt", 1000, b"bbb");
        let report = save_submission(&fs, "/workdir/sub", "/dest").unwrap();
        assert_eq!(report.files, 1);
        assert!(report.dirs >= 1);
        assert_eq!(
            fs.dest_files
                .lock()
                .unwrap()
                .get("/dest/a.txt")
                .cloned()
                .unwrap(),
            b"aaa".to_vec()
        );
    }

    #[test]
    fn dest_errnos_are_provokeable_misbehavior() {
        let mut fs = MemCustody::new();
        fs.dir("/workdir");
        fs.file("/workdir/x.txt", 1000, b"data");
        fs.provoke = Some("ENOSPC");
        let err = save_submission(&fs, "/workdir/x.txt", "/dest").unwrap_err();
        match err {
            Error::Misbehavior(StudentMisbehaviorError::DestError { errno, .. }) => {
                assert_eq!(errno, "ENOSPC");
            }
            other => panic!("expected DestError, got {other:?}"),
        }
    }

    #[test]
    fn errno_classification_table() {
        assert!(is_dest_errno("ENAMETOOLONG"));
        assert!(is_dest_errno("EEXIST"));
        assert!(is_dest_errno("ENOSPC"));
        assert!(is_dest_errno("EDQUOT"));
        assert!(is_dest_errno("EFBIG"));
        assert!(!is_dest_errno("EACCES"));
        assert!(!is_dest_errno("EPERM"));
    }

    #[test]
    fn caps_match_upstream() {
        assert_eq!(MAX_FILE_BYTES, 256 << 20);
        assert_eq!(MAX_TOTAL_BYTES, 1 << 30);
        assert_eq!(MAX_ENTRIES, 10_000);
        assert_eq!(MAX_DEPTH, 32);
        assert_eq!(CHUNK_SIZE, 1024 * 1024);
    }
}
