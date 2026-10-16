//! Proof-of-work solver for chat.deepseek.com (`DeepSeekHashV1`).
//!
//! The site ships a wasm-bindgen module (`assets/pow_solver.wasm`) whose
//! `wasm_solve(retptr, challenge_ptr, len, prefix_ptr, len, difficulty)`
//! finds the answer nonce. We execute that very module via `wasmi`,
//! so the algorithm stays byte-for-byte identical to the browser worker.
//!
//! Prefix convention (from the site's worker code): `{salt}_{expire_at}_`.

use base64::Engine as _;
use serde::Deserialize;
use serde_json::json;
use wasmi::{Engine, Instance, Linker, Memory, Module, Store, TypedFunc};

const WASM: &[u8] = include_bytes!("../assets/pow_solver.wasm");

/// Challenge issued by `/api/v0/chat/create_pow_challenge`.
#[derive(Debug, Clone, Deserialize)]
pub struct PowChallenge {
    pub algorithm: String,
    pub challenge: String,
    pub salt: String,
    pub difficulty: f64,
    pub expire_at: i64,
    pub signature: String,
    #[serde(default)]
    pub target_path: String,
}

/// Executes the vendored wasm solver.
pub struct PowSolver {
    store: Store<()>,
    instance: Instance,
}

impl PowSolver {
    /// Compile and instantiate the embedded module.
    pub fn new() -> Result<Self, String> {
        let engine = Engine::default();
        let module =
            Module::new(&engine, &mut WASM.as_ref()).map_err(|e| format!("PoW wasm parse: {e}"))?;
        let mut store = Store::new(&engine, ());
        let linker: Linker<()> = Linker::new(&engine);
        let instance = linker
            .instantiate_and_start(&mut store, &module)
            .map_err(|e| format!("PoW wasm instantiate: {e}"))?;
        Ok(Self { store, instance })
    }

    /// Solve one challenge; returns the answer value for the response header.
    pub fn solve(&mut self, c: &PowChallenge) -> Result<f64, String> {
        if c.algorithm != "DeepSeekHashV1" {
            return Err(format!("unsupported PoW algorithm: {}", c.algorithm));
        }
        let prefix = format!("{}_{}_", c.salt, c.expire_at);

        let solve: TypedFunc<(i32, i32, i32, i32, i32, f64), ()> = self
            .instance
            .get_typed_func(&self.store, "wasm_solve")
            .map_err(|_| "export wasm_solve not found")?;
        let add_to_stack: TypedFunc<(i32,), i32> = self
            .instance
            .get_typed_func(&self.store, "__wbindgen_add_to_stack_pointer")
            .map_err(|_| "export __wbindgen_add_to_stack_pointer not found")?;
        let alloc: TypedFunc<(i32, i32), i32> = self
            .instance
            .get_typed_func(&self.store, "__wbindgen_export_0")
            .map_err(|_| "export __wbindgen_export_0 (alloc) not found")?;
        let memory: Memory = self
            .instance
            .get_memory(&self.store, "memory")
            .ok_or("export memory not found")?;

        // Reserve return slot: [status: i32][answer: f64]
        let retptr = add_to_stack
            .call(&mut self.store, (-16,))
            .map_err(|e| format!("stack reserve: {e}"))?;

        let result = self.write_and_solve(
            &solve,
            &alloc,
            memory,
            retptr,
            &c.challenge,
            &prefix,
            c.difficulty,
        );

        let _ = add_to_stack.call(&mut self.store, (16,));
        result
    }

    fn write_and_solve(
        &mut self,
        solve: &TypedFunc<(i32, i32, i32, i32, i32, f64), ()>,
        alloc: &TypedFunc<(i32, i32), i32>,
        memory: Memory,
        retptr: i32,
        challenge: &str,
        prefix: &str,
        difficulty: f64,
    ) -> Result<f64, String> {
        let put = |store: &mut Store<()>, s: &str| -> Result<(i32, usize), String> {
            let bytes = s.as_bytes();
            let ptr = alloc
                .call(&mut *store, (bytes.len() as i32, 1))
                .map_err(|e| format!("alloc: {e}"))?;
            let start = ptr as usize;
            let end = start + bytes.len();
            let mem_len = memory.data(&*store).len();
            if end > mem_len {
                return Err(format!(
                    "wasm OOB write: ptr={ptr} len={} mem={mem_len}",
                    bytes.len()
                ));
            }
            memory.data_mut(&mut *store)[start..end].copy_from_slice(bytes);
            Ok((ptr, bytes.len()))
        };
        let (ch_ptr, ch_len) = put(&mut self.store, challenge)?;
        let (pr_ptr, pr_len) = put(&mut self.store, prefix)?;

        solve
            .call(
                &mut self.store,
                (
                    retptr,
                    ch_ptr,
                    ch_len as i32,
                    pr_ptr,
                    pr_len as i32,
                    difficulty,
                ),
            )
            .map_err(|e| format!("wasm_solve trap: {e}"))?;

        let data = memory.data(&self.store);
        if retptr as usize + 16 > data.len() {
            return Err("retptr out of memory bounds".into());
        }
        let status = i32::from_le_bytes(
            data[retptr as usize..retptr as usize + 4]
                .try_into()
                .unwrap(),
        );
        let answer = f64::from_le_bytes(
            data[retptr as usize + 8..retptr as usize + 16]
                .try_into()
                .unwrap(),
        );
        if status == 0 {
            return Err("PoW: no solution found".into());
        }
        if !answer.is_finite() || answer < 0.0 {
            return Err(format!("PoW: suspicious answer {answer}"));
        }
        Ok(answer)
    }
}

/// Serialize the answer exactly like the site does: compact JSON →
/// standard base64 → padding stripped.
pub fn pow_response_header(c: &PowChallenge, answer: f64) -> String {
    // JS `JSON.stringify` prints integral floats without ".0"
    // (e.g. `109585`); mirror that, otherwise the server rejects the header.
    let answer_value = if answer.fract() == 0.0 {
        json!(answer as i64)
    } else {
        json!(answer)
    };
    let payload = json!({
        "algorithm": c.algorithm,
        "challenge": c.challenge,
        "salt": c.salt,
        "answer": answer_value,
        "signature": c.signature,
        "target_path": c.target_path,
    });
    base64::engine::general_purpose::STANDARD
        .encode(payload.to_string())
        .trim_end_matches('=')
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_challenge(difficulty: f64) -> PowChallenge {
        PowChallenge {
            algorithm: "DeepSeekHashV1".into(),
            challenge: "17f75b5e0984f15fc0b8def0c77b48ee4b41b1865f5e8217971722f7870ad4cb".into(),
            salt: "cba9b8a9bf8368b6341c".into(),
            difficulty,
            expire_at: 1785354733914,
            signature: "b9f07214b4519ac8a6b976e22a3c2228e30b0738009e21122135539e69862f6a".into(),
            target_path: "/api/v0/chat/completion".into(),
        }
    }

    #[test]
    fn parses_challenge_json_from_server_shape() {
        let raw = serde_json::json!({
            "algorithm": "DeepSeekHashV1",
            "challenge": "abc",
            "salt": "def",
            "difficulty": 144000,
            "expire_at": 1783681627181_i64,
            "signature": "sig",
            "target_path": "/api/v0/chat/completion"
        });
        let c: PowChallenge = serde_json::from_value(raw).unwrap();
        assert_eq!(c.salt, "def");
        assert_eq!(c.expire_at, 1783681627181);
    }

    #[test]
    fn response_header_is_unpadded_base64_of_compact_json() {
        let c = sample_challenge(144000.0);
        let h = pow_response_header(&c, 123456.0);
        assert!(!h.contains('='));
        let padded = format!("{}{}", h, "=".repeat((4 - h.len() % 4) % 4));
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(padded)
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&decoded).unwrap();
        assert_eq!(v["algorithm"], json!("DeepSeekHashV1"));
        assert_eq!(v["salt"], json!("cba9b8a9bf8368b6341c"));
        assert_eq!(v["answer"], json!(123456));
        assert!(
            v["answer"].is_i64(),
            "integral answer must serialize as integer, not float"
        );
        assert_eq!(v["target_path"], json!("/api/v0/chat/completion"));
    }

    #[test]
    fn rejects_unknown_algorithm_before_touching_wasm() {
        let mut solver = PowSolver::new().unwrap();
        let mut c = sample_challenge(1000.0);
        c.algorithm = "DeepSeekHashV2".into();
        assert!(solver.solve(&c).is_err());
    }

    /// Captured live at 2026-08-25: full server tuple with its accepted
    /// answer. Server challenges are signature-bound (salt/expiry tampering
    /// makes them unsolvable), so this captured pair is our only possible
    /// deterministic fixture. It validates ABI + wasm integrity end-to-end.
    fn real_tuple() -> PowChallenge {
        PowChallenge {
            algorithm: "DeepSeekHashV1".into(),
            challenge: "9122c42a801fe8ece1ba4fc680a4e78509c17d212a97ab2b7896e310d99bd970".into(),
            salt: "7a7b8318a149f271de2a".into(),
            difficulty: 144000.0,
            expire_at: 1787684408111,
            signature: "120ade96d0882ba232b010dec6e0716a9462ce5dcefc519049b65d574eee3e4f8".into(),
            target_path: "/api/v0/chat/completion".into(),
        }
    }

    #[test]
    fn solves_real_captured_tuple_to_known_answer() {
        let mut solver = PowSolver::new().unwrap();
        let answer = solver.solve(&real_tuple()).unwrap();
        assert_eq!(answer, 47460.0);
    }

    /// Difficulty-sweep probe: for the real tuple, collect the minimal
    /// nonce the site-wasm reports across several difficulties. Small `d`
    /// makes each wasmi solve near-instant. The resulting curve pins down
    /// the acceptance rule empirically.
    #[test]
    #[ignore = "one-shot reverse-engineering probe"]
    fn probe_difficulty_curve() {
        let mut solver = PowSolver::new().unwrap();
        let mut base = real_tuple();
        println!("challenge={}", base.challenge);
        println!("prefix={{salt}}_{{expire}}_");
        for d in [16u64, 64, 256, 1024, 4096, 16384, 40000] {
            base.difficulty = d as f64;
            match solver.solve(&base) {
                Ok(a) => println!("D={d:>6}  n_min={a:.0}"),
                Err(e) => println!("D={d:>6}  UNSOLVABLE ({e})"),
            }
        }
        // Second family: perturbed salt — rule must transfer unchanged.
        base.salt = format!("{}b", &real_tuple().salt[..19]);
        for d in [1024u64, 4096] {
            base.difficulty = d as f64;
            match solver.solve(&base) {
                Ok(a) => println!("salt' D={d:>6}  n_min={a:.0}"),
                Err(e) => println!("salt' D={d:>6}  UNSOLVABLE ({e})"),
            }
        }
    }

    /// Decisive fit: hash every nonce once, then find the bit-scale `s`
    /// making rule `be_u64(head) < d << s` reproduce the oracle exactly
    /// (min-n == 47460 at d=144000 AND no solution at d=40000).
    #[test]
    #[ignore = "one-shot reverse-engineering probe"]
    fn fit_scale_rule() {
        use sha3::{Digest, Keccak256, Sha3_256};

        let c = real_tuple();
        let prefix = format!("{}_{}_", c.salt, c.expire_at);
        let n_star = 47460u64;
        let cap = 144_000u64;

        for (hname, hasher) in [("sha3-256", 0u8), ("keccak256", 1)] {
            let mut heads = Vec::with_capacity(cap as usize);
            for n in 0..cap {
                let input = format!("{prefix}{n}");
                let d: [u8; 32] = match hasher {
                    0 => Sha3_256::digest(input.as_bytes()).into(),
                    _ => Keccak256::digest(input.as_bytes()).into(),
                };
                heads.push(u64::from_be_bytes(d[0..8].try_into().unwrap()));
            }

            // Valid s interval: hit at n_star, miss before it, at d=144000.
            let mut valid_s = Vec::new();
            for s in 0..63 {
                let t = match 144000u64.checked_shl(s) {
                    Some(v) => v,
                    None => break,
                };
                if heads[..n_star as usize].iter().all(|&h| h >= t) && heads[n_star as usize] < t {
                    valid_s.push(s);
                }
            }
            println!("{hname}: valid s = {valid_s:?} (hit@{n_star}, misses before)");
            for &s in &valid_s {
                // d=40000 must have NO solution below cap.
                let t40000 = match 40000u64.checked_shl(s) {
                    Some(v) => v,
                    None => continue,
                };
                let none_small = heads[..40_000usize.min(cap as usize)]
                    .iter()
                    .all(|&h| h >= t40000);
                // First-hit prediction at other difficulties.
                let first_at = |d: u64| -> Option<usize> {
                    let t = d.checked_shl(s)?;
                    heads.iter().position(|&h| h < t)
                };
                println!(
                    "  s={s}: d=40000 none_below_40k={none_small}, first@d40000={:?}, first@d16384={:?}",
                    first_at(40000),
                    first_at(16384)
                );
            }
        }
    }

    /// Fit the acceptance rule against mined oracle tuples.
    ///
    /// For rule "v(hash(prefix+n)) < T" every oracle bounds T:
    /// max{v(h_n) : n < n*} < T <= v(h_{n*}). We intersect per-tuple
    /// candidate formulas T = (C_head >> j) * d >> e across ALL tuples;
    /// a combo surviving every tuple IS the algorithm.
    #[test]
    #[ignore = "fitting: needs /tmp/opencode/liho_oracles.json"]
    fn fit_threshold_formula() {
        use sha3::{Digest, Keccak256, Sha3_256};

        let raw = std::fs::read("/tmp/opencode/liho_oracles.json").expect("oracles file");
        let tuples: Vec<serde_json::Value> =
            serde_json::from_str(std::str::from_utf8(&raw).unwrap()).unwrap();

        #[derive(Clone, Copy)]
        enum H {
            Sha3,
            Keccak,
        }
        let hashers = [(H::Sha3, "sha3-256"), (H::Keccak, "keccak256")];

        // Window readers over the digest.
        type Win = fn(&[u8; 32]) -> u64;
        let windows: Vec<(&str, Win)> = vec![
            ("u64be@0", |h| {
                u64::from_be_bytes(h[0..8].try_into().unwrap())
            }),
            ("u32be@0", |h| {
                u32::from_be_bytes(h[0..4].try_into().unwrap()) as u64
            }),
            ("u64le@0", |h| {
                u64::from_le_bytes(h[0..8].try_into().unwrap())
            }),
            ("u32le@0", |h| {
                u32::from_le_bytes(h[0..4].try_into().unwrap()) as u64
            }),
            ("u32be@28", |h| {
                u32::from_be_bytes(h[28..32].try_into().unwrap()) as u64
            }),
        ];

        struct Oracle {
            prefix: String,
            d: u64,
            c_bytes: [u8; 32],
            n_star: usize,
            /// Precomputed head values for each window, index = nonce.
            heads: Vec<[u64; 5]>,
        }

        let mut oracles = Vec::new();
        for t in &tuples {
            let challenge = t["challenge"].as_str().unwrap();
            let salt = t["salt"].as_str().unwrap();
            let expire = t["expire_at"].as_u64().unwrap();
            let d = t["difficulty"].as_f64().unwrap() as u64;
            let n_star = t["answer"].as_f64().unwrap() as usize;
            let mut cb = [0u8; 32];
            for i in 0..32 {
                cb[i] = u8::from_str_radix(&challenge[i * 2..i * 2 + 2], 16).unwrap();
            }
            let prefix = format!("{salt}_{expire}_");
            let mut heads = Vec::with_capacity(n_star + 1);
            for n in 0..=n_star {
                let input = format!("{prefix}{n}");
                let mut row = [0u64; 5];
                for (hi, (h, _)) in hashers.iter().enumerate() {
                    let dig: [u8; 32] = match h {
                        H::Sha3 => Sha3_256::digest(input.as_bytes()).into(),
                        H::Keccak => Keccak256::digest(input.as_bytes()).into(),
                    };
                    for (wi, (_, f)) in windows.iter().enumerate() {
                        row[wi] = f(&dig);
                    }
                }
                heads.push(row);
            }
            oracles.push((
                Oracle {
                    prefix,
                    d,
                    c_bytes: cb,
                    n_star,
                    heads,
                },
                challenge.to_string(),
            ));
        }
        println!("loaded {} oracles", oracles.len());

        // Challenge-head candidates for the formula side.
        let c_variants: Vec<(&str, Box<dyn Fn(&[u8; 32]) -> u64>)> = vec![
            (
                "cBE8",
                Box::new(|c: &[u8; 32]| u64::from_be_bytes(c[0..8].try_into().unwrap())),
            ),
            (
                "cLE8",
                Box::new(|c: &[u8; 32]| u64::from_le_bytes(c[0..8].try_into().unwrap())),
            ),
            (
                "cBE8t",
                Box::new(|c: &[u8; 32]| u64::from_be_bytes(c[24..32].try_into().unwrap())),
            ),
            (
                "cLE8t",
                Box::new(|c: &[u8; 32]| u64::from_le_bytes(c[24..32].try_into().unwrap())),
            ),
        ];

        let mut survivors = Vec::new();
        for &(hname_idx, hlabel) in &hashers {
            for (wname, _) in &windows {
                let wi = windows.iter().position(|(n, _)| n == wname).unwrap();
                for (cname, cf) in &c_variants {
                    let base: Vec<u64> = oracles.iter().map(|(o, _)| cf(&o.c_bytes)).collect();
                    if base.iter().all(|&b| b == 0) {
                        continue;
                    }
                    // Enumerate T-shapes: pure shift, shift+scale-by-d,
                    // divide-by-d-power, product-shift.
                    let mut gens: Vec<(String, Vec<u64>)> = Vec::new();
                    for s in 0..64u32 {
                        gens.push((format!("c>>{s}"), base.iter().map(|&c| c >> s).collect()));
                        gens.push((
                            format!("c<<{s}"),
                            base.iter()
                                .map(|&c| c.checked_shl(s).unwrap_or(0))
                                .collect(),
                        ));
                        gens.push((
                            format!("(c>>{s})*d"),
                            base.iter()
                                .zip(&oracles)
                                .map(|(&c, (o, _))| (c >> s).saturating_mul(o.d))
                                .collect(),
                        ));
                        gens.push((
                            format!("(c<<{s})*d"),
                            base.iter()
                                .zip(&oracles)
                                .map(|(&c, (o, _))| {
                                    c.checked_shl(s).unwrap_or(0).saturating_mul(o.d)
                                })
                                .collect(),
                        ));
                    }
                    for e in 0..40u32 {
                        gens.push((
                            format!("c/(d<<{e})"),
                            base.iter()
                                .zip(&oracles)
                                .map(|(&c, (o, _))| match o.d.checked_shl(e) {
                                    Some(x) if x != 0 => c / x,
                                    _ => 0,
                                })
                                .collect(),
                        ));
                    }

                    for (gname, ts) in &gens {
                        let ok = oracles.iter().enumerate().all(|(oi, (o, _))| {
                            let t = ts[oi];
                            if t == 0 {
                                return false;
                            }
                            let mut lo = 0u64;
                            for hh in &o.heads[..o.n_star] {
                                lo = lo.max(hh[wi]);
                            }
                            lo < t && t <= o.heads[o.n_star][wi]
                        });
                        if ok {
                            survivors.push(format!("{hlabel}/{wname}/{cname}/{gname}"));
                            println!("SURVIVOR: {hlabel}/{wname}/{cname} {gname}");
                        }
                    }
                }
            }
            let _ = hname_idx;
        }
        if survivors.is_empty() {
            println!("NO FORMULA SURVIVED — widen hypothesis classes");
        }
    }

    /// The site's wasm is named `sha3_wasm_bg.wasm`; we brute-force the
    /// hypothesis space {hash × input layout × acceptance condition} against
    /// the captured oracle: the correct combo must yield EXACTLY 47460 as
    /// the minimal nonce. Run once with:
    ///   cargo test --release explore_hypotheses -- --ignored --nocapture
    #[test]
    #[ignore = "one-shot reverse-engineering probe"]
    fn explore_hypotheses() {
        use sha3::{Digest, Keccak256, Sha3_256};

        const EXPECT: u64 = 47460;
        const CAP: u64 = 400_000;

        let c = real_tuple();
        let prefix = format!("{}_{}_", c.salt, c.expire_at);
        let difficulty = c.difficulty as u64; // 144000

        // 256-bit helpers over [u8; 32] (big-endian semantics).
        fn leading_zero_bits(h: &[u8; 32]) -> u32 {
            h.iter()
                .map(|b| b.leading_zeros() as u32)
                .take_while(|z| *z == 8)
                .sum::<u32>()
                + h.iter()
                    .find(|b| **b != 0)
                    .map(|b| b.leading_zeros() as u32)
                    .unwrap_or(0)
        }
        fn be_mod(h: &[u8; 32], d: u64) -> u64 {
            // Horner over 32 bytes using u64 intermediates.
            h.iter().fold(0u64, |acc, &b| {
                (acc.wrapping_mul(256).wrapping_add(b as u64)) % d
            })
        }
        fn lt_target(h: &[u8; 32], target: &[u8; 32]) -> bool {
            h < target
        }

        let k_ceil = (difficulty as f64).log2().ceil() as u32; // 18
        let k_floor = (difficulty as f64).log2().floor() as u32; // 17

        enum Hasher {
            Keccak,
            Sha3,
        }
        impl Hasher {
            fn digest(&self, input: &[u8]) -> [u8; 32] {
                match self {
                    Hasher::Keccak => Keccak256::digest(input).into(),
                    Hasher::Sha3 => Sha3_256::digest(input).into(),
                }
            }
            fn name(&self) -> &'static str {
                match self {
                    Hasher::Keccak => "keccak256",
                    Hasher::Sha3 => "sha3-256",
                }
            }
        }

        let inputs: Vec<(&str, Box<dyn Fn(u64) -> String>)> = vec![
            ("prefix+dec", {
                let p = prefix.clone();
                Box::new(move |n| format!("{p}{n}"))
            }),
            ("dec+prefix", {
                let p = prefix.clone();
                Box::new(move |n| format!("{n}{p}"))
            }),
            ("challenge+prefix+dec", {
                let ch = c.challenge.clone();
                let p = prefix.clone();
                Box::new(move |n| format!("{ch}{p}{n}"))
            }),
            ("prefix+challenge+dec", {
                let ch = c.challenge.clone();
                let p = prefix.clone();
                Box::new(move |n| format!("{p}{ch}{n}"))
            }),
            ("challenge+dec", {
                let ch = c.challenge.clone();
                Box::new(move |n| format!("{ch}{n}"))
            }),
            ("prefix+hex", {
                let p = prefix.clone();
                Box::new(move |n| format!("{p}{n:x}"))
            }),
        ];

        // Threshold for "h < max/d": schoolbook bit-by-bit division of
        // the 256-bit all-ones numerator by difficulty.
        fn div_max(d: u64) -> [u8; 32] {
            let mut q = [0u8; 32];
            let mut rem: u64 = 0;
            for i in 0..256 {
                rem = rem.checked_mul(2).expect("rem must stay < d*2, d small");
                rem |= 1; // numerator bit is 1 (all ones)
                if rem >= d {
                    rem -= d;
                    q[i / 8] |= 0x80 >> (i % 8);
                }
            }
            q
        }
        let threshold = div_max(difficulty);

        let mut hits = Vec::new();
        for hasher in [Hasher::Keccak, Hasher::Sha3] {
            for (iname, mk) in &inputs {
                'cond: for (cname, cond) in [
                    (
                        "lzb>=ceil(log2 d)",
                        Box::new(|h: &[u8; 32]| leading_zero_bits(h) >= k_ceil)
                            as Box<dyn Fn(&[u8; 32]) -> bool>,
                    ),
                    (
                        "lzb>=floor(log2 d)",
                        Box::new(|h: &[u8; 32]| leading_zero_bits(h) >= k_floor),
                    ),
                    (
                        "mod d == 0",
                        Box::new(|h: &[u8; 32]| be_mod(h, difficulty) == 0),
                    ),
                    (
                        "h < max/d",
                        Box::new(|h: &[u8; 32]| lt_target(h, &threshold)),
                    ),
                    (
                        "first4 mod d == 0",
                        Box::new(|h: &[u8; 32]| {
                            u32::from_be_bytes([h[0], h[1], h[2], h[3]]) as u64 % difficulty == 0
                        }),
                    ),
                ] {
                    let mut found = None;
                    for n in 0..CAP {
                        let input = mk(n);
                        let digest = hasher.digest(input.as_bytes());
                        if cond(&digest) {
                            found = Some(n);
                            break;
                        }
                    }
                    if found == Some(EXPECT) {
                        println!(
                            "MATCH ✓ hash={} input={} cond={} → {EXPECT}",
                            hasher.name(),
                            iname,
                            cname
                        );
                        hits.push(format!("{}/{}", hasher.name(), iname));
                    }
                    if !hits.is_empty() && found.is_some() {
                        break 'cond; // keep output readable once we have a family hit
                    }
                }
                if !hits.is_empty() {
                    break;
                }
            }
            if !hits.is_empty() {
                break;
            }
        }
        if hits.is_empty() {
            println!("NO MATCH in hypothesis space — extending search needed");
        }
    }
}
