# Clangd Setup for ETI (Explicit Template Instantiation) Support

This project uses Explicit Template Instantiation (ETI) with a `*_decl.hpp` / `*_def.hpp` / `*.hpp` file structure. This document explains how clangd is configured to provide full IntelliSense support.

## The Problem

With ETI, template definitions are in `*_def.hpp` files, but when clangd opens these files directly, it doesn't see the template declarations from `*_decl.hpp` files, causing "no template named X" errors.

## The Solution

We use an **automated compile_commands.json augmentation** approach that:
1. Detects all `*_def.hpp` files in the project
2. Adds compilation entries for each with `-include` flags pointing to the corresponding `*_decl.hpp`
3. Requires **zero source code modifications**
4. Automatically runs after CMake configuration

## Files Involved

- **`augment_compile_commands.py`**: Python script that augments `compile_commands.json`
- **`build-*/configure-build.sh`**: Configure scripts that run the augmentation after CMake

## How It Works

### When you run CMake:

1. CMake generates `build-debug/compile_commands.json` with entries for `*.cpp` files
2. The configure script automatically runs `augment_compile_commands.py`
3. The script adds 85 new entries for all `*_def.hpp` files
4. Each entry includes `-include path/to/corresponding_decl.hpp`
5. Clangd reads the augmented `compile_commands.json`
6. ✅ Full IntelliSense in all files, including `*_def.hpp`

### File Structure Example:

```
Domain.hpp           → Unified header (includes _decl, conditionally includes _def)
Domain_decl.hpp      → Template declarations
Domain_def.hpp       → Template definitions (now has compile_commands entry!)
Domain.cpp           → Explicit instantiations
```

## Usage

### Initial Setup (Already Done)

The setup is already complete! Just use clangd normally.

### After Modifying CMake Files

If you modify CMakeLists.txt or reconfigure:

```bash
cd build-debug
./configure-build.sh  # Automatically augments compile_commands.json
```

### Manual Augmentation (if needed)

If you need to manually run the augmentation:

```bash
python3 augment_compile_commands.py [path/to/compile_commands.json]

# Examples:
python3 augment_compile_commands.py                              # Uses build-debug/
python3 augment_compile_commands.py build-debug-debug/compile_commands.json
```

### In Neovim

Just restart clangd to pick up changes:

```vim
:LspRestart
```

## Adding New ETI Files

When you create new `*_decl.hpp` / `*_def.hpp` pairs:

1. Create your files following the existing naming convention
2. Re-run CMake or the augmentation script
3. Restart clangd

The script automatically detects all `*_def.hpp` files, so no configuration changes needed!

## Verification

To verify clangd can parse a `*_def.hpp` file correctly:

```bash
clangd --check=feddlib/core/FE/Domain_def.hpp 2>&1 | grep "no_template"
```

If the output is empty, it's working! If you see "no template" errors, re-run the augmentation script.

## How This Differs from Source Modification Approach

### This Approach (Chosen):
- ✅ Zero source code changes
- ✅ Automatic via configure scripts
- ✅ Maintains clean ETI structure
- ⚠️  Requires running script after CMake changes

### Alternative (Not Used):
- ❌ Requires adding `#ifdef __clangd__` includes to 85 files
- ✅ Would work without scripts
- ✅ More explicit in source code

We chose the automated approach to keep the source code pristine.

## Troubleshooting

### Clangd shows "no template" errors in *_def.hpp files

**Solution**: Re-run the augmentation script and restart clangd:
```bash
python3 augment_compile_commands.py
# In neovim: :LspRestart
```

### Script reports "No corresponding _decl.hpp" warnings

This is normal for any `*_def.hpp` files without matching `*_decl.hpp` files. Verify the file actually needs declarations.

### compile_commands.json gets overwritten

The configure scripts automatically re-augment after CMake. If you run `cmake` directly (not via configure script), manually run:
```bash
python3 augment_compile_commands.py
```

## Benefits

✅ **Full IntelliSense** in all header files  
✅ **No source modifications** required  
✅ **Automatic** via configure scripts  
✅ **Scalable** - works with any number of ETI files  
✅ **Standard approach** - uses only clangd's documented features  

---

**Last Updated**: January 19, 2026  
**Clangd Version**: 17.0.0+  
**Project**: FEDDLib with Trilinos ETI structure
