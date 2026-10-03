//! Phase-1 tokenizer + vocabulary API over two fixtures.
//!
//! `FORGE_TEST_MODEL` (tiny-llama: vocab 32, `add_bos` only, no byte
//! pieces) covers vocab getters and empty-input encodes; its SPM
//! tokenizer cannot encode non-empty text (missing byte fallback
//! throws `std::out_of_range` upstream — see the phase-1 report §7),
//! so that path is deliberately untested here. `FORGE_TEST_MODEL_TOK`
//! (tiny-tok: vocab 269, full `<0xXX>` byte coverage, `add_bos` and
//! `add_eos`) covers real encodes, decodes and round-trips. Each
//! suite skips when its fixture is unset (see `docs/NATIVE.md`).

use forge_core::{
    DecodeOptions, EncodeOptions, Model, SpecialTokens, TokenAttr, Tokenizer, VocabType,
};
use std::path::PathBuf;

fn fixture(var: &str) -> Option<PathBuf> {
    match std::env::var(var) {
        Ok(path) => Some(PathBuf::from(path)),
        Err(_) => {
            println!("SKIP: {var} not set (see docs/NATIVE.md)");
            None
        }
    }
}

fn load(var: &str) -> Option<Model> {
    fixture(var).map(|path| Model::load(&path).expect("fixture model must load"))
}

/// `tiny-tok` id of raw byte `b`: ids 13..269 are `<0x00>`..`<0xFF>`.
fn byte_id(byte: u8) -> u32 {
    13 + u32::from(byte)
}

fn plain_options() -> EncodeOptions {
    let mut options = EncodeOptions::default();
    options.add_special = false;
    options
}

fn parsed_options() -> EncodeOptions {
    let mut options = EncodeOptions::default();
    options.add_special = false;
    options.parse_special = true;
    options
}

fn strip_options() -> DecodeOptions {
    let mut options = DecodeOptions::default();
    options.remove_special = true;
    options
}

fn unparse_options() -> DecodeOptions {
    let mut options = DecodeOptions::default();
    options.unparse_special = true;
    options
}

// -- A. open / metadata --------------------------------------------------------

#[test]
fn tok_fixture_metadata_is_correct() {
    let Some(model) = load("FORGE_TEST_MODEL_TOK") else {
        return;
    };
    let tok = Tokenizer::open(&model).expect("open");
    assert_eq!(tok.n_vocab(), 269);
    assert_eq!(tok.vocab_type(), VocabType::Spm);
    assert!(tok.adds_bos());
    assert!(tok.adds_eos());
    // Upstream auto-detects NL from the `<0x0A>` byte piece (id 23).
    assert_eq!(
        tok.special_tokens(),
        SpecialTokens {
            bos: Some(1),
            eos: Some(2),
            eot: None,
            sep: None,
            nl: Some(23),
            pad: None,
            mask: None,
        }
    );
    // Cross-check against the Model API and the shape-derived count.
    assert_eq!(model.vocab_size().expect("vocab_size"), 269);
    assert_eq!(model.n_params(), 4968);
    let debug = format!("{tok:?}");
    assert!(debug.contains("Spm"), "Debug names the type: {debug}");
}

#[test]
fn tiny_fixture_metadata_is_correct() {
    let Some(model) = load("FORGE_TEST_MODEL") else {
        return;
    };
    let tok = Tokenizer::open(&model).expect("open");
    assert_eq!(tok.n_vocab(), 32);
    assert_eq!(tok.vocab_type(), VocabType::Spm);
    assert!(tok.adds_bos());
    assert!(!tok.adds_eos(), "tiny-llama leaves add_eos unset");
    assert_eq!(tok.special_tokens().bos, Some(1));
    assert_eq!(tok.special_tokens().eos, Some(2));
    assert_eq!(tok.special_tokens().nl, None, "no byte pieces, no NL");
}

// -- B. vocab getters (tiny-tok) -----------------------------------------------

#[test]
fn token_text_score_attr_cover_all_classes() {
    let Some(model) = load("FORGE_TEST_MODEL_TOK") else {
        return;
    };
    let tok = Tokenizer::open(&model).expect("open");
    let cases = [
        (0, "<unk>", TokenAttr::UNKNOWN),
        (1, "<s>", TokenAttr::CONTROL),
        (2, "</s>", TokenAttr::CONTROL),
        (3, "tok0", TokenAttr::NORMAL),
        (12, "tok9", TokenAttr::NORMAL),
        (13, "<0x00>", TokenAttr::BYTE),
        (77, "<0x40>", TokenAttr::BYTE),
        (268, "<0xFF>", TokenAttr::BYTE),
    ];
    for (id, text, attr) in cases {
        assert_eq!(tok.token_text(id).expect("text"), text, "id {id}");
        assert_eq!(tok.token_score(id).expect("score"), 0.0, "id {id}");
        assert_eq!(tok.token_attr(id).expect("attr"), attr, "id {id}");
    }
}

#[test]
fn eog_and_control_matrix() {
    let Some(model) = load("FORGE_TEST_MODEL_TOK") else {
        return;
    };
    let tok = Tokenizer::open(&model).expect("open");
    // Only EOS ends generation; BOS/EOS are the only controls.
    for id in [0, 1, 2, 3, 12, 13, 268] {
        assert_eq!(tok.is_eog(id).expect("eog"), id == 2, "eog id {id}");
        assert_eq!(
            tok.is_control(id).expect("control"),
            id == 1 || id == 2,
            "control id {id}"
        );
    }
}

#[test]
fn out_of_range_ids_are_clean_errors() {
    let Some(model) = load("FORGE_TEST_MODEL_TOK") else {
        return;
    };
    let tok = Tokenizer::open(&model).expect("open");
    // Each failure below must arrive as Error, never as a native
    // throw (terminate) or an unchecked-index UB: the test passing
    // at all is the assertion; messages are checked too.
    let bad = [
        tok.token_text(269)
            .map(|_| ())
            .map_err(|error| error.to_string()),
        tok.token_score(u32::MAX)
            .map(|_| ())
            .map_err(|error| error.to_string()),
        tok.token_attr(1000)
            .map(|_| ())
            .map_err(|error| error.to_string()),
        tok.is_eog(269)
            .map(|_| ())
            .map_err(|error| error.to_string()),
        tok.is_control(500)
            .map(|_| ())
            .map_err(|error| error.to_string()),
    ];
    for (error, id) in bad.iter().zip([269, u32::MAX, 1000, 269, 500]) {
        let message = error.as_ref().expect_err("OOB id must fail");
        assert!(
            message.starts_with("tokenizer error"),
            "OOB maps to tokenizer error: {message}"
        );
        assert!(
            message.contains(&id.to_string()),
            "error names the bad id: {message}"
        );
    }
}

// -- C. encode (tiny-tok) ------------------------------------------------------

#[test]
fn encode_empty_matches_spm_convention() {
    let Some(model) = load("FORGE_TEST_MODEL_TOK") else {
        return;
    };
    let tok = Tokenizer::open(&model).expect("open");
    assert_eq!(
        tok.encode("", &plain_options()).expect("plain"),
        Vec::<u32>::new()
    );
    assert_eq!(
        tok.encode("", &EncodeOptions::default()).expect("special"),
        vec![1, 2]
    );
}

#[test]
fn encode_is_byte_exact_and_deterministic() {
    let Some(model) = load("FORGE_TEST_MODEL_TOK") else {
        return;
    };
    let tok = Tokenizer::open(&model).expect("open");
    // '▁'(E2 96 81, the SPM space prefix) then 't' 'o' 'k' '5' as
    // raw byte pieces: with all scores 0 the Viterbi tie-break lands
    // on bytes rather than the "tok5" word piece. Ids are 13 + byte.
    let expected = vec![239, 163, 142, 129, 124, 120, 66];
    assert_eq!(
        expected,
        vec![
            byte_id(0xe2),
            byte_id(0x96),
            byte_id(0x81),
            byte_id(b't'),
            byte_id(b'o'),
            byte_id(b'k'),
            byte_id(b'5'),
        ]
    );
    let once = tok.encode("tok5", &plain_options()).expect("encode");
    assert_eq!(once, expected);
    let twice = tok.encode("tok5", &plain_options()).expect("again");
    assert_eq!(once, twice, "encode is deterministic");
    assert!(once.iter().all(|id| *id < tok.n_vocab()), "ids are valid");
}

#[test]
fn encode_add_special_wraps_with_bos_eos() {
    let Some(model) = load("FORGE_TEST_MODEL_TOK") else {
        return;
    };
    let tok = Tokenizer::open(&model).expect("open");
    let plain = tok.encode("tok5", &plain_options()).expect("plain");
    let mut expected = vec![1];
    expected.extend(&plain);
    expected.push(2);
    assert_eq!(
        tok.encode("tok5", &EncodeOptions::default())
            .expect("special"),
        expected
    );
}

#[test]
fn parse_special_controls_special_spellings() {
    let Some(model) = load("FORGE_TEST_MODEL_TOK") else {
        return;
    };
    let tok = Tokenizer::open(&model).expect("open");
    // As plaintext: '▁' + '<' 's' '>' bytes.
    assert_eq!(
        tok.encode("<s>", &plain_options()).expect("plain"),
        vec![239, 163, 142, 73, 128, 75]
    );
    // Parsed: the single BOS token.
    assert_eq!(
        tok.encode("<s>", &parsed_options()).expect("parsed"),
        vec![1]
    );
}

#[test]
fn encode_maps_inner_spaces_to_word_boundaries() {
    let Some(model) = load("FORGE_TEST_MODEL_TOK") else {
        return;
    };
    let tok = Tokenizer::open(&model).expect("open");
    assert_eq!(
        tok.encode("a b", &plain_options()).expect("encode"),
        vec![
            byte_id(0xe2),
            byte_id(0x96),
            byte_id(0x81),
            byte_id(b'a'),
            byte_id(0xe2),
            byte_id(0x96),
            byte_id(0x81),
            byte_id(b'b'),
        ]
    );
}

// -- D. decode (tiny-tok) ------------------------------------------------------

#[test]
fn decode_renders_and_skips_specials_by_flag() {
    let Some(model) = load("FORGE_TEST_MODEL_TOK") else {
        return;
    };
    let tok = Tokenizer::open(&model).expect("open");
    let default = DecodeOptions::default();
    assert_eq!(tok.decode(&[5], &default).expect("one"), "tok2");
    assert_eq!(tok.decode(&[1, 5, 2], &default).expect("edges"), "tok2");
    assert_eq!(
        tok.decode(&[1, 5, 2], &strip_options()).expect("strip"),
        "tok2"
    );
    assert_eq!(
        tok.decode(&[1, 5, 2], &unparse_options()).expect("unparse"),
        "<s>tok2</s>"
    );
    assert_eq!(tok.decode(&[], &default).expect("empty"), "");
}

#[test]
fn decode_rejects_bad_ids_and_bad_utf8() {
    let Some(model) = load("FORGE_TEST_MODEL_TOK") else {
        return;
    };
    let tok = Tokenizer::open(&model).expect("open");
    let default = DecodeOptions::default();
    let oob = tok.decode(&[269], &default).expect_err("id 269 must fail");
    assert!(oob.to_string().starts_with("tokenizer error"), "{oob}");
    tok.decode(&[5, 999], &default)
        .expect_err("second id bad must fail");
    // A bare 3-byte-sequence leader is not valid UTF-8 ...
    let bad = tok
        .decode(&[byte_id(0xe2)], &default)
        .expect_err("lone leader must fail");
    assert!(bad.to_string().contains("UTF-8"), "{bad}");
    // ... but the full sequence decodes.
    assert_eq!(
        tok.decode(&[byte_id(0xe2), byte_id(0x96), byte_id(0x81)], &default)
            .expect("full"),
        "▁"
    );
}

// -- E. round-trip (tiny-tok) --------------------------------------------------

#[test]
fn spm_round_trip_maps_spaces_to_word_boundaries() {
    let Some(model) = load("FORGE_TEST_MODEL_TOK") else {
        return;
    };
    let tok = Tokenizer::open(&model).expect("open");
    for text in ["tok5", "hello", "a b", "Hello, world!", "tok0 tok9"] {
        let ids = tok.encode(text, &plain_options()).expect("encode");
        let back = tok.decode(&ids, &DecodeOptions::default()).expect("decode");
        // Upstream bytes come back verbatim: SPM U+2581 stays U+2581.
        assert_eq!(back, format!("▁{}", text.replace(' ', "▁")), "{text:?}");
    }
    // add_special + remove_special round-trips through BOS/EOS too.
    let ids = tok
        .encode("tok5", &EncodeOptions::default())
        .expect("encode");
    assert_eq!(tok.decode(&ids, &strip_options()).expect("decode"), "▁tok5");
}

// -- F. tiny-llama encode limits ------------------------------------------------

#[test]
fn tiny_encode_empty_only() {
    let Some(model) = load("FORGE_TEST_MODEL") else {
        return;
    };
    let tok = Tokenizer::open(&model).expect("open");
    assert_eq!(
        tok.encode("", &plain_options()).expect("plain"),
        Vec::<u32>::new()
    );
    // add_bos with no add_eos: a lone BOS.
    assert_eq!(
        tok.encode("", &EncodeOptions::default()).expect("special"),
        vec![1]
    );
    // Non-empty input is deliberately NOT encoded here: tiny-llama
    // has no byte pieces, so upstream byte fallback throws
    // std::out_of_range (observed SIGABRT; see report §7). Real
    // encodes are covered by the tiny-tok suite above.
}
