{
  description = "icp_stitch: offline loop-closure post-processing for dimos memory2 recordings (tag PGO + ICP stitching), all in Rust";

  inputs = {
    gtsam_shim.url = "github:jeff-hykin/gtsam_shim";
    nixpkgs.follows = "gtsam_shim/nixpkgs";
    flake-utils.follows = "gtsam_shim/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, gtsam_shim, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        gtsam = gtsam_shim.packages.${system}.gtsam;
        buildEnv = gtsam_shim.lib.${system}.buildEnv;

        icp_stitch = pkgs.rustPlatform.buildRustPackage ({
          pname = "icp_stitch";
          version = "0.1.0";
          src = ./.;
          cargoLock = {
            lockFile = ./Cargo.lock;
            outputHashes = {
              "gtsam_shim-0.1.0" = "sha256-DPG0WfdsNSJ8jOr2W1PAmvnuk3Mp3hgBf0Z6HHjqRg0=";
              "lcm-msgs-0.1.0" = "sha256-ps+8iBliZpyDB3I+QB5U+L+Jo5idP+GzuJBRnGoN9CU=";
            };
          };
          nativeBuildInputs = [ pkgs.pkg-config ];
          buildInputs = [ gtsam pkgs.eigen pkgs.boost pkgs.tbb ];
          # rerun's sdk build is heavy; the unit tests already run in CI/dev.
          doCheck = false;
          # darwin fixup strips LC_RPATH, breaking the @rpath/libgtsam reference.
          postFixup = pkgs.lib.optionalString pkgs.stdenv.isDarwin ''
            install_name_tool -add_rpath ${gtsam}/lib $out/bin/icp_stitch
          '';
        } // buildEnv);
      in {
        packages.default = icp_stitch;
        packages.icp_stitch = icp_stitch;

        apps.default = {
          type = "app";
          program = "${icp_stitch}/bin/icp_stitch";
        };

        devShells.default = pkgs.mkShell {
          packages = [ pkgs.cargo pkgs.rustc pkgs.clippy pkgs.rustfmt pkgs.pkg-config ];
          buildInputs = [ gtsam pkgs.eigen pkgs.boost pkgs.tbb ];
          env = buildEnv;
        };
      });
}
