# COIL VM - Debugger Component

class COILDebugger:
    """Debugger for the COIL Virtual Machine"""
    
    def __init__(self, vm):
        self.vm = vm
        self.breakpoints = set()
        self.watchpoints = {}  # address -> callback
        self.instruction_trace = []
        self.max_trace_size = 1000
        self.is_active = False
        self.step_mode = False
        self.instruction_counter = 0
        self.last_registers = {}
    
    def add_breakpoint(self, address):
        """Add a breakpoint at the specified address"""
        self.breakpoints.add(address)
        print(f"Breakpoint set at 0x{address:08x}")
    
    def remove_breakpoint(self, address):
        """Remove a breakpoint at the specified address"""
        if address in self.breakpoints:
            self.breakpoints.remove(address)
            print(f"Breakpoint removed from 0x{address:08x}")
        else:
            print(f"No breakpoint found at 0x{address:08x}")
    
    def add_watchpoint(self, address, callback=None):
        """Add a watchpoint to monitor memory access"""
        self.watchpoints[address] = callback
        print(f"Watchpoint set at 0x{address:08x}")
    
    def remove_watchpoint(self, address):
        """Remove a watchpoint"""
        if address in self.watchpoints:
            del self.watchpoints[address]
            print(f"Watchpoint removed from 0x{address:08x}")
        else:
            print(f"No watchpoint found at 0x{address:08x}")
    
    def enable(self):
        """Enable the debugger"""
        self.is_active = True
        self.instruction_trace.clear()
        self.instruction_counter = 0
        self._save_register_state()
    
    def disable(self):
        """Disable the debugger"""
        self.is_active = False
        self.step_mode = False
    
    def step(self):
        """Execute a single instruction"""
        self.step_mode = True
        # The VM's instruction interpreter will call back to check_breakpoint
        # after each instruction when the debugger is active
    
    def continue_execution(self):
        """Continue execution until next breakpoint"""
        self.step_mode = False
    
    def check_breakpoint(self, address, instruction):
        """Check if execution should pause at current address"""
        if not self.is_active:
            return False
        
        self.instruction_counter += 1
        
        # Track instruction in trace buffer
        if len(self.instruction_trace) >= self.max_trace_size:
            self.instruction_trace.pop(0)
        
        self.instruction_trace.append({
            'address': address,
            'instruction': instruction
        })
        
        # Check if we're at a breakpoint
        is_at_breakpoint = address in self.breakpoints
        
        # Check register changes
        reg_changes = self._check_register_changes()
        
        # Determine if we should pause execution
        should_pause = is_at_breakpoint or self.step_mode
        
        if should_pause:
            self._print_current_state(address, instruction, reg_changes)
            
            # If at a breakpoint, switch to step mode to let user control execution
            if is_at_breakpoint:
                self.step_mode = True
        
        return should_pause
    
    def check_memory_access(self, address, is_write, value=None):
        """Check if memory access triggers a watchpoint"""
        if not self.is_active:
            return False
        
        if address in self.watchpoints:
            callback = self.watchpoints[address]
            
            if callback:
                # Execute custom callback if provided
                callback(address, is_write, value)
            else:
                # Default behavior
                access_type = "write to" if is_write else "read from"
                value_str = f"value 0x{value:x}" if value is not None else ""
                print(f"Watchpoint hit: {access_type} 0x{address:08x} {value_str}")
            
            # Switch to step mode when a watchpoint is hit
            self.step_mode = True
            return True
        
        return False
    
    def print_instructions(self, count=10):
        """Print the last N instructions executed"""
        if not self.instruction_trace:
            print("No instructions in trace buffer")
            return
        
        count = min(count, len(self.instruction_trace))
        start = len(self.instruction_trace) - count
        
        print(f"Last {count} instructions:")
        for i in range(start, len(self.instruction_trace)):
            entry = self.instruction_trace[i]
            instr = entry['instruction']
            addr = entry['address']
            
            opcode_name = self._get_opcode_name(instr['opcode'])
            operands = self._format_operands(instr['operands'])
            
            print(f"  0x{addr:08x}: {opcode_name} {operands}")
    
    def print_registers(self):
        """Print current register values"""
        print("Register values:")
        rm = self.vm.register_manager
        
        # General purpose registers
        for i in range(16):
            quad_val = rm.get_register('RQ', i)
            long_val = rm.get_register('RL', i)
            print(f"  RQ{i}: 0x{quad_val:016x}  RL{i}: 0x{long_val:08x}")
        
        # Special registers
        print(f"  RSP: 0x{rm.rsp:016x}")
        print(f"  RBP: 0x{rm.rbp:016x}")
        print(f"  RIP: 0x{rm.rip:016x}")
        
        # Flags
        flags = rm.rf
        flag_strings = []
        if flags & rm.FLAG_ZERO:
            flag_strings.append("ZERO")
        if flags & rm.FLAG_SIGN:
            flag_strings.append("SIGN")
        if flags & rm.FLAG_OVERFLOW:
            flag_strings.append("OVERFLOW")
        if flags & rm.FLAG_CARRY:
            flag_strings.append("CARRY")
        if flags & rm.FLAG_PARITY:
            flag_strings.append("PARITY")
        
        flag_str = " ".join(flag_strings) if flag_strings else "none"
        print(f"  RF: 0x{flags:08x} ({flag_str})")
    
    def hexdump(self, address, length=64):
        """Print a hexdump of memory"""
        print(f"Memory dump from 0x{address:08x} ({length} bytes):")
        
        # Ensure we can read the memory
        mem = self.vm.memory_manager
        try:
            data = mem.read_bytes(address, length)
        except MemoryError:
            print("  Error: Unable to read memory at specified address")
            return
        
        # Format as hexdump
        bytes_per_line = 16
        for i in range(0, length, bytes_per_line):
            line_data = data[i:i+bytes_per_line]
            hex_values = " ".join(f"{b:02x}" for b in line_data)
            
            # Pad hex values to align ASCII representation
            padding = "   " * (bytes_per_line - len(line_data))
            
            # ASCII representation
            ascii_repr = "".join(chr(b) if 32 <= b <= 126 else "." for b in line_data)
            
            print(f"  0x{address+i:08x}: {hex_values}{padding}  |{ascii_repr}|")
    
    def disassemble(self, address, count=10):
        """Disassemble instructions at the given address"""
        print(f"Disassembly at 0x{address:08x}:")
        
        current_addr = address
        for _ in range(count):
            try:
                instr = self.vm.instruction_interpreter.decode_instruction(current_addr)
                opcode_name = self._get_opcode_name(instr['opcode'])
                operands = self._format_operands(instr['operands'])
                
                print(f"  0x{current_addr:08x}: {opcode_name} {operands}")
                
                current_addr += instr['size']
            except Exception as e:
                print(f"  Error disassembling at 0x{current_addr:08x}: {e}")
                break
    
    def print_sections(self):
        """Print information about loaded sections"""
        sections = self.vm.memory_manager.sections
        
        if not sections:
            print("No sections loaded")
            return
        
        print("Loaded sections:")
        for name, info in sections.items():
            flag_str = self._format_section_flags(info['flags'])
            print(f"  {name}: 0x{info['start']:08x} - 0x{info['start']+info['size']:08x} ({info['size']} bytes) {flag_str}")
    
    def print_symbols(self):
        """Print information about loaded symbols"""
        if not hasattr(self.vm.loader, 'symbols'):
            print("No symbol information available")
            return
        
        symbols = self.vm.loader.symbols
        
        if not symbols:
            print("No symbols loaded")
            return
        
        print("Loaded symbols:")
        for name, symbol in symbols.items():
            addr = symbol['value']
            if symbol['section_index'] > 0:
                # Adjust for section base address if needed
                # This is simplified - would need section mapping in a real implementation
                print(f"  {name}: 0x{addr:08x} + section {symbol['section_index']}")
            else:
                print(f"  {name}: 0x{addr:08x}")
    
    def _check_register_changes(self):
        """Check which registers have changed since last instruction"""
        rm = self.vm.register_manager
        changes = {}
        
        current_state = {}
        
        # Check general purpose registers
        for i in range(16):
            for reg_type in ['RQ', 'RL', 'RW', 'RB']:
                key = f"{reg_type}{i}"
                current_state[key] = rm.get_register(reg_type, i)
                
                if key in self.last_registers and current_state[key] != self.last_registers[key]:
                    changes[key] = (self.last_registers[key], current_state[key])
        
        # Check special registers
        for reg in ['RSP', 'RBP', 'RIP', 'RF']:
            current_state[reg] = rm.get_register(reg, 0)
            
            if reg in self.last_registers and current_state[reg] != self.last_registers[reg]:
                changes[reg] = (self.last_registers[reg], current_state[reg])
        
        # Update last known register values
        self.last_registers = current_state
        
        return changes
    
    def _save_register_state(self):
        """Save current register state for comparison"""
        rm = self.vm.register_manager
        
        # Save general purpose registers
        for i in range(16):
            for reg_type in ['RQ', 'RL', 'RW', 'RB']:
                self.last_registers[f"{reg_type}{i}"] = rm.get_register(reg_type, i)
        
        # Save special registers
        for reg in ['RSP', 'RBP', 'RIP', 'RF']:
            self.last_registers[reg] = rm.get_register(reg, 0)
    
    def _print_current_state(self, address, instruction, reg_changes):
        """Print current execution state when paused"""
        opcode_name = self._get_opcode_name(instruction['opcode'])
        operands = self._format_operands(instruction['operands'])
        
        print(f"\nStopped at 0x{address:08x}: {opcode_name} {operands}")
        
        # Print changed registers
        if reg_changes:
            print("Register changes:")
            for reg, (old_val, new_val) in reg_changes.items():
                print(f"  {reg}: 0x{old_val:x} -> 0x{new_val:x}")
        
        # Print flags
        flags = self.vm.register_manager.rf
        flag_str = ""
        if flags & self.vm.register_manager.FLAG_ZERO:
            flag_str += "Z"
        if flags & self.vm.register_manager.FLAG_SIGN:
            flag_str += "S"
        if flags & self.vm.register_manager.FLAG_OVERFLOW:
            flag_str += "O"
        if flags & self.vm.register_manager.FLAG_CARRY:
            flag_str += "C"
        if flags & self.vm.register_manager.FLAG_PARITY:
            flag_str += "P"
        
        print(f"Flags: {flag_str if flag_str else 'none'}")
    
    def _get_opcode_name(self, opcode):
        """Convert opcode value to mnemonic"""
        opcodes = {
            0x00: "NOP",
            0x01: "SYMB",
            0x02: "BR",
            0x03: "BRC",
            0x04: "CALL",
            0x05: "RET",
            0x06: "INT",
            0x09: "SYSC",
            0x10: "ADD",
            0x11: "SUB",
            0x12: "MUL",
            0x13: "DIV",
            0x20: "AND",
            0x21: "OR",
            0x22: "XOR",
            0x23: "NOT",
            0x30: "CMP",
            0x40: "MOV",
            0x41: "LOAD",
            0x42: "STORE",
            0x50: "PUSH",
            0x51: "POP"
        }
        return opcodes.get(opcode, f"UNK_{opcode:02x}")
    
    def _format_operands(self, operands):
        """Format instruction operands for display"""
        if not operands:
            return ""
        
        operand_strs = []
        
        for op in operands:
            qualifier = op['qualifier']
            
            if qualifier == 0x01:  # IMM
                operand_strs.append(f"#{op.get('value', 0):x}")
            elif qualifier == 0x03:  # REG
                reg_index = op.get('reg_index', 0)
                # Determine register prefix based on type
                op_type = op.get('type', 0)
                width = op.get('width', 3)  # Default to 64-bit (TYPE_WIDTH_64)
                
                reg_prefix = "R"
                if width == 0:
                    reg_prefix += "B"  # 8-bit
                elif width == 1:
                    reg_prefix += "W"  # 16-bit
                elif width == 2:
                    reg_prefix += "L"  # 32-bit
                else:
                    reg_prefix += "Q"  # 64-bit
                
                operand_strs.append(f"{reg_prefix}{reg_index}")
            elif qualifier == 0x04:  # MEM
                operand_strs.append(f"[0x{op.get('address', 0):x}]")
            else:
                operand_strs.append(f"<op:{qualifier:02x}>")
        
        return ", ".join(operand_strs)
    
    def _format_section_flags(self, flags):
        """Format section flags for display"""
        flag_strs = []
        
        if flags & 0x01:  # WRITE
            flag_strs.append("W")
        if flags & 0x02:  # EXEC
            flag_strs.append("X")
        if flags & 0x04:  # ALLOC
            flag_strs.append("A")
        
        return "[" + "".join(flag_strs) + "]" if flag_strs else ""


# Example debugger command interface
class COILDebuggerInterface:
    """Interactive command interface for the COIL debugger"""
    
    def __init__(self, vm, debugger):
        self.vm = vm
        self.debugger = debugger
        self.running = False
        self.commands = {
            'h': self.cmd_help,
            'help': self.cmd_help,
            'q': self.cmd_quit,
            'quit': self.cmd_quit,
            'b': self.cmd_breakpoint,
            'break': self.cmd_breakpoint,
            'c': self.cmd_continue,
            'cont': self.cmd_continue,
            's': self.cmd_step,
            'step': self.cmd_step,
            'r': self.cmd_registers,
            'reg': self.cmd_registers,
            'm': self.cmd_memory,
            'mem': self.cmd_memory,
            'd': self.cmd_disassemble,
            'disasm': self.cmd_disassemble,
            'i': self.cmd_info,
            'info': self.cmd_info,
            'w': self.cmd_watchpoint,
            'watch': self.cmd_watchpoint,
            't': self.cmd_trace,
            'trace': self.cmd_trace
        }
    
    def run(self):
        """Run the debugger interface"""
        self.running = True
        self.debugger.enable()
        print("COIL Debugger started")
        print("Type 'help' for a list of commands")
        
        while self.running:
            try:
                cmd = input("(coil-dbg) ").strip()
                if not cmd:
                    continue
                
                parts = cmd.split()
                cmd_name = parts[0].lower()
                args = parts[1:]
                
                if cmd_name in self.commands:
                    self.commands[cmd_name](args)
                else:
                    print(f"Unknown command: {cmd_name}")
                    print("Type 'help' for a list of commands")
            
            except KeyboardInterrupt:
                print("\nInterrupted")
                self.running = False
            except EOFError:
                print("\nExiting")
                self.running = False
            except Exception as e:
                print(f"Error: {e}")
    
    def cmd_help(self, args):
        """Show help information"""
        print("Available commands:")
        print("  h, help            - Show this help")
        print("  q, quit            - Exit the debugger")
        print("  b, break <addr>    - Set breakpoint at address")
        print("  c, cont            - Continue execution")
        print("  s, step            - Execute one instruction")
        print("  r, reg             - Show register values")
        print("  m, mem <addr> [len]- Show memory at address")
        print("  d, disasm <addr> [count] - Disassemble at address")
        print("  i, info <what>     - Show info (sections, symbols)")
        print("  w, watch <addr>    - Set watchpoint at address")
        print("  t, trace [count]   - Show instruction trace")
    
    def cmd_quit(self, args):
        """Exit the debugger"""
        self.running = False
        self.debugger.disable()
        print("Exiting debugger")
    
    def cmd_breakpoint(self, args):
        """Set or clear breakpoint"""
        if not args:
            print("Usage: break <address> [clear]")
            return
        
        try:
            if args[0].startswith("0x"):
                addr = int(args[0], 16)
            else:
                addr = int(args[0])
            
            # Check for 'clear' argument
            if len(args) > 1 and args[1].lower() == "clear":
                self.debugger.remove_breakpoint(addr)
            else:
                self.debugger.add_breakpoint(addr)
        
        except ValueError:
            print(f"Invalid address: {args[0]}")
    
    def cmd_continue(self, args):
        """Continue execution"""
        self.debugger.continue_execution()
        print("Continuing execution...")
    
    def cmd_step(self, args):
        """Step one instruction"""
        self.debugger.step()
        print("Stepping...")
    
    def cmd_registers(self, args):
        """Show register values"""
        self.debugger.print_registers()
    
    def cmd_memory(self, args):
        """Show memory dump"""
        if not args:
            print("Usage: mem <address> [length]")
            return
        
        try:
            if args[0].startswith("0x"):
                addr = int(args[0], 16)
            else:
                addr = int(args[0])
            
            length = 64  # Default
            if len(args) > 1:
                length = int(args[1])
            
            self.debugger.hexdump(addr, length)
        
        except ValueError:
            print(f"Invalid address or length")
    
    def cmd_disassemble(self, args):
        """Disassemble code"""
        if not args:
            print("Usage: disasm <address> [count]")
            return
        
        try:
            if args[0].startswith("0x"):
                addr = int(args[0], 16)
            else:
                addr = int(args[0])
            
            count = 10  # Default
            if len(args) > 1:
                count = int(args[1])
            
            self.debugger.disassemble(addr, count)
        
        except ValueError:
            print(f"Invalid address or count")
    
    def cmd_info(self, args):
        """Show various information"""
        if not args:
            print("Usage: info <sections|symbols>")
            return
        
        topic = args[0].lower()
        
        if topic == "sections":
            self.debugger.print_sections()
        elif topic == "symbols":
            self.debugger.print_symbols()
        else:
            print(f"Unknown info topic: {topic}")
            print("Available topics: sections, symbols")
    
    def cmd_watchpoint(self, args):
        """Set or clear watchpoint"""
        if not args:
            print("Usage: watch <address> [clear]")
            return
        
        try:
            if args[0].startswith("0x"):
                addr = int(args[0], 16)
            else:
                addr = int(args[0])
            
            # Check for 'clear' argument
            if len(args) > 1 and args[1].lower() == "clear":
                self.debugger.remove_watchpoint(addr)
            else:
                self.debugger.add_watchpoint(addr)
        
        except ValueError:
            print(f"Invalid address: {args[0]}")
    
    def cmd_trace(self, args):
        """Show instruction trace"""
        count = 10  # Default
        if args:
            try:
                count = int(args[0])
            except ValueError:
                print(f"Invalid count: {args[0]}")
                return
        
        self.debugger.print_instructions(count)


# Example usage
def start_debugger(vm):
    """Start the COIL debugger"""
    debugger = COILDebugger(vm)
    interface = COILDebuggerInterface(vm, debugger)
    interface.run()


# Integration with the VM's instruction interpreter
def _integrate_debugger(vm, debugger):
    """Integrate the debugger with the VM's instruction interpreter"""
    # Save the original execute method
    original_execute = vm.instruction_interpreter.execute
    
    # Monkey patch the execute method to add debugging hooks
    def debug_execute(start_address):
        vm.register_manager.set_register('RIP', start_address)
        
        running = True
        while running:
            current_rip = vm.register_manager.get_register('RIP', 0)
            instruction = vm.instruction_interpreter.decode_instruction(current_rip)
            
            # Check for breakpoint or step action
            if debugger.check_breakpoint(current_rip, instruction):
                # Handle interactive debugging here
                # This would integrate with the debugger interface
                pass
            
            # Update RIP to next instruction
            vm.register_manager.set_register('RIP', current_rip + instruction['size'])
            
            # Execute the instruction
            handler = vm.instruction_interpreter.opcode_handlers.get(instruction['opcode'])
            if handler:
                running = handler(instruction)
            else:
                raise ValueError(f"Unknown opcode: {instruction['opcode']:#04x}")
            
            # Check if we should stop execution
            if not running:
                break
        
        return current_rip
    
    # Replace the execute method
    vm.instruction_interpreter.execute = debug_execute