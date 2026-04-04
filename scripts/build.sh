#!/usr/bin/env bash
# Local LaTeX build script — no sudo required
# Usage: ./scripts/build.sh [en|zh|all]
set -euo pipefail

TEXLIVE_BIN="/data/liutianxing/texlive/2026/bin/x86_64-linux"
export PATH="$TEXLIVE_BIN:$PATH"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

BUILD_DIR="$PROJECT_ROOT/build"
TARGET="${1:-all}"

build_en() {
    echo "==> Compiling English PDF (pdflatex)..."
    latexmk -pdf -interaction=nonstopmode main.tex
    cp main.pdf "$BUILD_DIR/Control_Theory_Note.pdf"
    echo "==> English PDF: $BUILD_DIR/Control_Theory_Note.pdf"
}

build_zh() {
    echo "==> Compiling Chinese PDF (xelatex)..."
    latexmk -xelatex -interaction=nonstopmode main_zh.tex
    cp main_zh.pdf "$BUILD_DIR/Control_Theory_Note_cn.pdf"
    echo "==> Chinese PDF: $BUILD_DIR/Control_Theory_Note_cn.pdf"
}

clean() {
    echo "==> Cleaning auxiliary files..."
    latexmk -C main.tex main_zh.tex 2>/dev/null || true
    rm -f main.pdf main_zh.pdf main_zh.xdv
}

case "$TARGET" in
    en)    build_en ;;
    zh)    build_zh ;;
    all)   build_en; build_zh ;;
    clean) clean ;;
    *)     echo "Usage: $0 [en|zh|all|clean]"; exit 1 ;;
esac
