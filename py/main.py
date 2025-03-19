#!/usr/bin/env python3
"""
COIL VM - Command Line Interface
A virtual machine that interprets COIL instructions directly
following specification version 1.0.0
"""

import os
import sys
import argparse
from coil_vm_core import COILVirtualMachine
from coil_vm_debugger import COILDebugger, COILDebuggerInterface


class COILVirtualMachineCLI:
    """Command line interface for the COIL VM"""
    
    def __init__(self):
        """Initialize the CLI"""
        self.vm = COILVirtualMachine()
        self.debugger = None
    
    def parse_arguments(self):
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(description='COIL Virtual Machine - Version 1.0.0')
        
        parser.add_argument('file', help='COIL object file (.cof) to execute')
        
        parser.add_argument('-d', '--debug', action='store_true',
                            help='Start in debug mode')
        
        parser.add_argument('-b', '--break', dest='breakpoints', action='append',
                            help='Set breakpoint at address (can be used multiple times)')
        
        parser.add_argument('-e', '--entry', type=str,
                            help='Override entry point (symbol name or hex address)')
        
        parser.add_argument('-v', '--verbose', action='store_true',
                            help='Enable verbose output')
        
        parser.add_argument('-m', '--memory', type=str, default='1MB',
                            help='Memory size (e.g., 1MB, 16KB)')
        
        parser.add_argument('--trace', action='store_true',
                            help='Enable instruction tracing')
        
        parser.add_argument('--limit', type=int, default=1000000,
                            help='Maximum number of instructions to execute')
        
        return parser.parse_args()
    
    def parse_memory_size(self, memory_str):
        """Parse memory size string (e.g., 1MB, 16KB) to bytes"""
        memory_str = memory_str.upper()
        units = {
            'B': 1,
            'KB': 1024,
            'MB': 1024 * 1024,
            'GB': 1024 * 1024 * 1024
        }
        
        # Find the unit
        unit = 'B'
        for u in units:
            if memory_str.endswith(u):
                unit = u
                memory_str = memory_str[:-len(u)]
                break
        
        try:
            size = int(memory_str) * units[unit]
            return size
        except ValueError:
            print(f"Error: Invalid memory size format: {memory_str}")
            return 1024 * 1024  # Default to 1MB
    
    def parse_address(self, addr_str):
        """Parse a string as an address (hex or decimal)"""
        try:
            if addr_str.startswith('0x'):
                return int(addr_str, 16)
            else:
                return int(addr_str)
        except ValueError:
            # Might be a symbol name, return None and handle later
            return None
    
    def run(self):
        """Run the COIL VM with the given arguments"""
        args = self.parse_arguments()
        
        # Configure VM memory
        memory_size = self.parse_memory_size(args.memory)
        self.vm.memory_manager = MemoryManager(memory_size)
        
        if args.verbose:
            print(f"COIL VM - Version 1.0.0")
            print(f"Memory size: {memory_size} bytes")
        
        # Check if file exists
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            return 1
        
        try:
            # Load the file
            if args.verbose:
                print(f"Loading file: {args.file}")
            
            entry_point = self.vm.load_file(args.file)
            
            # Override entry point if specified
            if args.entry:
                addr = self.parse_address(args.entry)
                if addr is not None:
                    entry_point = addr
                else:
                    # Try as symbol name
                    symbol_addr = self.vm.loader.get_symbol_address(args.entry)
                    if symbol_addr is not None:
                        entry_point = symbol_addr
                    else:
                        print(f"Error: Could not resolve entry point: {args.entry}")
                        return 1
            
            if args.verbose:
                print(f"Entry point: 0x{entry_point:08x}")
            
            # Setup debugging if requested
            if args.debug or args.breakpoints or args.trace:
                self.debugger = COILDebugger(self.vm)
                
                # Set breakpoints if specified
                if args.breakpoints:
                    for bp in args.breakpoints:
                        addr = self.parse_address(bp)
                        if addr is not None:
                            self.debugger.add_breakpoint(addr)
                        else:
                            # Try as symbol name
                            symbol_addr = self.vm.loader.get_symbol_address(bp)
                            if symbol_addr is not None:
                                self.debugger.add_breakpoint(symbol_addr)
                            else:
                                print(f"Warning: Could not resolve breakpoint: {bp}")
                
                # Enable tracing if requested
                if args.trace:
                    self.debugger.enable()
                
                # Start debugger if in debug mode
                if args.debug:
                    self.debugger.enable()
                    debugger_interface = COILDebuggerInterface(self.vm, self.debugger)
                    debugger_interface.run()
                    return 0
            
            # Execute the program
            if args.verbose:
                print("Executing program...")
            
            # Set instruction limit
            instruction_limit = args.limit
            
            # Run the program
            try:
                # If tracing is enabled, we need a custom execution loop
                if args.trace:
                    self._run_with_tracing(entry_point, instruction_limit, args.verbose)
                else:
                    self.vm.execute(entry_point)
                
                if args.verbose:
                    print("Program execution completed successfully")
                return 0
                
            except Exception as e:
                print(f"Error during execution: {e}")
                return 1
                
        except Exception as e:
            print(f"Error: {e}")
            return 1
    
    def _run_with_tracing(self, start_address, limit, verbose):
        """Run the program with instruction tracing"""
        self.vm.register_manager.set_register('RIP', start_address)
        
        instruction_count = 0
        running = True
        
        while running and instruction_count < limit:
            # Get current instruction pointer
            current_rip = self.vm.register_manager.get_register('RIP', 0)
            
            # Decode the instruction
            try:
                instruction = self.vm.instruction_interpreter.decode_instruction(current_rip)
            except Exception as e:
                print(f"Error decoding instruction at 0x{current_rip:08x}: {e}")
                break
            
            # Print trace information
            if verbose:
                opcode_name = self.debugger._get_opcode_name(instruction['opcode'])
                operands = self.debugger._format_operands(instruction['operands'])
                print(f"0x{current_rip:08x}: {opcode_name} {operands}")
            
            # Let debugger check breakpoints
            if self.debugger and self.debugger.check_breakpoint(current_rip, instruction):
                # If we hit a breakpoint, start the interactive debugger
                debugger_interface = COILDebuggerInterface(self.vm, self.debugger)
                debugger_interface.run()
                # After debugger exits, continue execution
            
            # Update RIP to next instruction
            self.vm.register_manager.set_register('RIP', current_rip + instruction['size'])
            
            # Execute the instruction
            handler = self.vm.instruction_interpreter.opcode_handlers.get(instruction['opcode'])
            if handler:
                running = handler(instruction)
            else:
                print(f"Unknown opcode: {instruction['opcode']:#04x}")
                break
            
            instruction_count += 1
        
        if instruction_count >= limit:
            print(f"Instruction limit reached ({limit})")


def main():
    """Main entry point"""
    cli = COILVirtualMachineCLI()
    sys.exit(cli.run())


if __name__ == "__main__":
    main()