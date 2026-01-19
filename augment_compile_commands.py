#!/usr/bin/env python3
"""
Augment compile_commands.json with entries for *_def.hpp files.
Each _def.hpp entry will force-include its corresponding _decl.hpp file.
"""

import json
import sys
from pathlib import Path

def augment_compile_commands(compile_commands_path):
    # Load existing compile commands
    with open(compile_commands_path, 'r') as f:
        commands = json.load(f)
    
    # Find all *_def.hpp files in the project
    project_root = Path(compile_commands_path).parent.parent
    def_files = list(project_root.rglob('*_def.hpp'))
    
    print(f"Found {len(def_files)} *_def.hpp files")
    
    # Use the first compilation command as a template
    if not commands:
        print("Error: No compilation commands found!")
        return
    
    template = commands[0].copy()
    
    # Track how many entries we add
    added = 0
    
    for def_file in def_files:
        # Calculate the corresponding _decl.hpp file
        decl_file = def_file.parent / def_file.name.replace('_def.hpp', '_decl.hpp')
        
        if not decl_file.exists():
            print(f"Warning: No corresponding _decl.hpp for {def_file.relative_to(project_root)}")
            continue
        
        # Check if this _def.hpp already has an entry
        if any(str(def_file) in cmd.get('file', '') for cmd in commands):
            continue
        
        # Create a new entry for this _def.hpp file
        # Make paths relative to project root for consistency
        def_file_rel = def_file.relative_to(project_root)
        decl_file_rel = decl_file.relative_to(project_root)
        
        new_entry = {
            'directory': template['directory'],
            'file': str(def_file_rel),
            'command': template['command'].replace(
                template['file'], str(def_file_rel)
            ) + f' -include {project_root / decl_file_rel}'
        }
        
        commands.append(new_entry)
        added += 1
    
    print(f"Added {added} new entries for *_def.hpp files")
    
    # Write back the augmented compile commands
    with open(compile_commands_path, 'w') as f:
        json.dump(commands, f, indent=2)
    
    print(f"Updated {compile_commands_path}")

if __name__ == '__main__':
    # Accept compile_commands.json path as argument, default to build-debug
    if len(sys.argv) > 1:
        compile_commands = Path(sys.argv[1])
    else:
        compile_commands = Path('build-debug/compile_commands.json')
    
    if not compile_commands.exists():
        print(f"Error: {compile_commands} not found!")
        sys.exit(1)
    
    augment_compile_commands(compile_commands)
