export NIX_CONFIG := "extra-experimental-features = nix-command flakes"

fmt:
    treefmt || nix fmt

lint:
    uv run ruff check --fix

test:
    uv run pytest src

check:
    uv run basedpyright --project pyproject.toml

sync:
    uv sync --all-packages

sync-clean:
    uv sync --all-packages --force-reinstall --no-cache

rust-rebuild:
    cargo run --bin stub_gen
    uv sync --reinstall-package exo_pyo3_bindings

build-dashboard:
    #!/usr/bin/env bash
    cd dashboard
    npm install
    npm run build

package:
    uv run pyinstaller --noconfirm packaging/pyinstaller/exo.spec

clean:
    rm -rf **/__pycache__
    rm -rf target/
    rm -rf .venv
    rm -rf dashboard/node_modules
    rm -rf dashboard/.svelte-kit
    rm -rf dashboard/build

kill:
    # Kill process on default port 52415
    lsof -ti:52415 | xargs kill -9 2>/dev/null || true
    # Fallback kill by name
    pkill -9 -f "python -m exo" 2>/dev/null || true
    pkill -9 -f "EXO.app" 2>/dev/null || true
    # Kill the backend binary specifically (in case the app wrapper is gone but backend lingers)
    pkill -9 -f "Contents/Resources/exo/exo" 2>/dev/null || true

build-macos: build-dashboard package
    # Build the macOS app
    cd app/EXO && xcodebuild build -scheme EXO -configuration Release -derivedDataPath build
    # Inject the Python backend
    rm -rf app/EXO/build/Build/Products/Release/EXO.app/Contents/Resources/exo
    mkdir -p app/EXO/build/Build/Products/Release/EXO.app/Contents/Resources
    cp -R dist/exo app/EXO/build/Build/Products/Release/EXO.app/Contents/Resources/exo
    @echo "App built at app/EXO/build/Build/Products/Release/EXO.app"
