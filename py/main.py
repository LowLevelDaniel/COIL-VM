#!/usr/bin/env python3
"""
COIL VM - Enhanced Debug Version
A virtual machine that interprets COIL instructions directly
following specification version 1.0.0
"""

import os
import sys
import argparse
import traceback
from coil_vm_core import COILVirtualMachine


# Import the improved MemoryManager class
# from memory_manager_fix import MemoryManager


def parse_memory_size(memory_str):
    """
    Parse memory size string with robust error handling
    """
    if not memory_str:
        print("Warning: No memory size specified, using default (1MB)")
        return 1024 * 1024  # Default to 1MB
    
    # Strip whitespace and convert to uppercase
    memory_str = memory_str.strip().upper()
    
    # Handle both full and shorthand units
    units = {
        'B': 1,
        'K': 1024,
        'KB': 1024,
        'M': 1024 * 1024,
        'MB': 1024 * 1024,
        'G': 1024 * 1024 * 1024,
        'GB': 1024 * 1024 * 1024
    }
    
    # Try to extract the numeric part and unit
    numeric_part = ''
    unit_part = ''
    
    for char in memory_str:
        if char.isdigit() or char == '.':
            numeric_part += char
        else:
            unit_part += char
    
    # If we couldn't extract a numeric part, use default
    if not numeric_part:
        print(f"Warning: Invalid memory size format: {memory_str}, using default (1MB)")
        return 1024 * 1024  # Default to 1MB
    
    # If there's no unit specified, assume bytes
    if not unit_part:
        unit_part = 'B'
    
    # Check if the unit is recognized
    if unit_part not in units:
        print(f"Warning: Unknown memory unit: {unit_part}, using default (1MB)")
        return 1024 * 1024  # Default to 1MB
    
    try:
        # Convert the numeric part to an integer or float
        if '.' in numeric_part:
            size_value = float(numeric_part)
        else:
            size_value = int(numeric_part)
        
        # Multiply by the unit factor
        result = int(size_value * units[unit_part])
        
        # Ensure we have a positive size
        if result <= 0:
            print(f"Warning: Memory size must be positive, using default (1MB)")
            return 1024 * 1024  # Default to 1MB
        
        print(f"Memory size parsed: {numeric_part} {unit_part} = {result} bytes")
        return result
    
    except (ValueError, TypeError):
        print(f"Warning: Invalid memory size format: {memory_str}, using default (1MB)")
        return 1024 * 1024  # Default to 1MB


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='COIL VM Debug - Version 1.0.0')
    
    parser.add_argument('file', help='COIL object file (.cof) to execute')
    
    parser.add_argument('-m', '--memory', type=str, default='1MB',
                        help='Memory size (e.g., 1MB, 16KB, 1M, 16K)')
    
    parser.add_argument('-e', '--entry', type=str,
                        help='Override entry point (symbol name or hex address)')
    
    parser.add_argument('-d', '--dump', type=str,
                        help='Dump memory region (format: addr,size or start-end)')
    
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')
    
    parser.add_argument('--debug', action='store_true',
                        help='Enable COF loader debug output')
    
    parser.add_argument('--examine', action='store_true',
                        help='Examine COF file structure without executing')
    
    parser.add_argument('--fix-negative', action='store_true',
                        help='Attempt to fix negative addresses')
    
    return parser.parse_args()


def main():
    """Main entry point with enhanced error handling"""
    args = parse_arguments()
    
    # Parse memory size
    memory_size = parse_memory_size(args.memory)
    
    # Check if file exists
    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        return 1
    
    print(f"\n=== COIL VM Debug - Version 1.0.0 ===")
    print(f"Loading file: {args.file}")
    print(f"Memory size: {memory_size} bytes")
    
    try:
        # Create VM instance
        vm = COILVirtualMachine()
        
        # Replace memory manager with enhanced version
        # vm.memory_manager = MemoryManager(memory_size)
        
        # If using original memory manager, set to specified size
        vm.memory_manager.memory_size = memory_size
        vm.memory_manager.memory = bytearray(memory_size)
        
        # Enable debug output for the loader if requested
        if hasattr(vm.loader, 'set_debug') and args.debug:
            vm.loader.set_debug(True)
        
        # Read the COF file content for examination
        with open(args.file, 'rb') as f:
            cof_data = f.read()
        
        print(f"File size: {len(cof_data)} bytes")
        
        if len(cof_data) < 32:
            print("Error: File too small to be a valid COF file (minimum 32 bytes required)")
            return 1
        
        # Check magic number
        magic = int.from_bytes(cof_data[0:4], byteorder='little')
        if magic != 0x434F494C:  # 'COIL'
            print(f"Warning: Magic number mismatch: 0x{magic:08x}, expected 0x434F494C")
        else:
            print(f"Magic number OK: 0x{magic:08x} ('COIL')")
        
        # Print basic header info
        version_major = cof_data[4]
        version_minor = cof_data[5]
        version_patch = cof_data[6]
        flags = cof_data[7]
        target = int.from_bytes(cof_data[8:10], byteorder='little')
        section_count = int.from_bytes(cof_data[10:12], byteorder='little')
        entrypoint = int.from_bytes(cof_data[12:16], byteorder='little')
        
        print(f"Version: {version_major}.{version_minor}.{version_patch}")
        print(f"Flags: 0x{flags:02x}")
        print(f"Target: 0x{target:04x}")
        print(f"Section count: {section_count}")
        print(f"Entry point: 0x{entrypoint:08x}")
        
        if args.examine:
            # Detailed examination of the file structure
            print("\n=== COF File Structure ===")
            
            # String table info
            str_tab_off = int.from_bytes(cof_data[16:20], byteorder='little')
            str_tab_size = int.from_bytes(cof_data[20:24], byteorder='little')
            print(f"String table: offset=0x{str_tab_off:x}, size={str_tab_size}")
            
            # Symbol table info
            sym_tab_off = int.from_bytes(cof_data[24:28], byteorder='little')
            sym_tab_size = int.from_bytes(cof_data[28:32], byteorder='little')
            print(f"Symbol table: offset=0x{sym_tab_off:x}, size={sym_tab_size}")
            
            # Section headers
            print("\n=== Section Headers ===")
            offset = 32  # Start after header
            
            for i in range(section_count):
                if offset + 36 > len(cof_data):
                    print(f"Error: Section header {i} extends beyond end of file")
                    break
                
                name_offset = int.from_bytes(cof_data[offset:offset+4], byteorder='little')
                type_value = int.from_bytes(cof_data[offset+4:offset+8], byteorder='little')
                flags = int.from_bytes(cof_data[offset+8:offset+12], byteorder='little')
                section_offset = int.from_bytes(cof_data[offset+12:offset+16], byteorder='little')
                size = int.from_bytes(cof_data[offset+16:offset+20], byteorder='little')
                
                # Get section name from string table if possible
                section_name = "<unknown>"
                if str_tab_off > 0 and name_offset < str_tab_size:
                    # Find null terminator
                    name_end = str_tab_off + name_offset
                    while name_end < len(cof_data) and cof_data[name_end] != 0:
                        name_end += 1
                    
                    if name_end < len(cof_data):
                        section_name = cof_data[str_tab_off + name_offset:name_end].decode('utf-8', errors='replace')
                
                print(f"Section {i}: name='{section_name}', type={type_value}, flags=0x{flags:x}")
                print(f"  offset=0x{section_offset:x}, size={size}")
                
                if section_offset + size > len(cof_data):
                    print(f"  WARNING: Section data extends beyond end of file")
                
                offset += 36  # Move to next section header
            
            return 0  # Exit after examination if requested
        
        # Try to load and execute the file
        try:
            print("\n=== Loading COF File ===")
            entry_point = vm.load_file(args.file)
            print(f"File loaded successfully, entry point: 0x{entry_point:08x}")
            
            # Override entry point if specified
            if args.entry:
                try:
                    if args.entry.startswith('0x'):
                        entry_point = int(args.entry, 16)
                    else:
                        entry_point = int(args.entry)
                    print(f"Overriding entry point to: 0x{entry_point:08x}")
                except ValueError:
                    # Try as symbol name
                    symbol_addr = vm.loader.get_symbol_address(args.entry)
                    if symbol_addr is not None:
                        entry_point = symbol_addr
                        print(f"Resolved symbol '{args.entry}' to address: 0x{entry_point:08x}")
                    else:
                        print(f"Warning: Could not resolve entry point: {args.entry}")
            
            if args.dump:
                # Parse memory dump request
                dump_parts = args.dump.split(',')
                if len(dump_parts) == 2:
                    # Format: addr,size
                    try:
                        addr = int(dump_parts[0], 0)
                        size = int(dump_parts[1], 0)
                        print(f"\n=== Memory Dump: 0x{addr:08x} ({size} bytes) ===")
                        # Dump memory
                        if hasattr(vm.memory_manager, 'dump_range'):
                            vm.memory_manager.dump_range(addr, size)
                        else:
                            # Basic dump implementation
                            data = vm.memory_manager.read_bytes(addr, size)
                            for i in range(0, len(data), 16):
                                chunk = data[i:i+16]
                                hex_values = ' '.join(f'{b:02x}' for b in chunk)
                                ascii_values = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in chunk)
                                print(f"0x{addr+i:08x}: {hex_values.ljust(48)} | {ascii_values}")
                    except Exception as e:
                        print(f"Error during memory dump: {e}")
            
            print("\n=== Executing Program ===")
            vm.execute(entry_point)
            print("Program execution completed successfully")
            
            return 0
            
        except Exception as e:
            print(f"Error: {e}")
            if args.verbose:
                traceback.print_exc()
            
            # Special handling for common errors
            if "Address out of range" in str(e) and "-" in str(e):
                print("\n=== Potential negative address issue detected ===")
                print("This may be due to improper sign extension or section relocation.")
                print("Try running with --fix-negative flag to attempt automatic correction.")
            
            return 1
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())