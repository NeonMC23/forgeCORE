fn main() {
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("native/staging.cpp")
        .include("native")
        .warnings(false)
        .compile("forge_staging_native");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-lib=dl");
    }
    println!("cargo:rerun-if-changed=native/staging.cpp");
    println!("cargo:rerun-if-changed=native/staging.h");
}
