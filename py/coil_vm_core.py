# COIL VM - Core Components
# Following COIL Specification Version 1.0.0

class COILVirtualMachine:
    """Main VM class that coordinates all components"""
    
    def __init__(self):
        self.register_manager = RegisterManager()
        self.memory_manager = MemoryManager()
        self.type_system = TypeSystem()
        self.instruction_interpreter = InstructionInterpreter(self)
        self.loader = COFLoader(self)
        
    def load_file(self, filename):
        """Load a COF file into the VM"""
        return self.loader.load(filename)
        
    def execute(self, entry_point=None):
        """Execute loaded code starting at entry_point"""
        if entry_point is None:
            entry_point = self.loader.get_entry_point()
        
        return self.instruction_interpreter.execute(entry_point)
    
    def reset(self):
        """Reset the VM state"""
        self.register_manager.reset()
        self.memory_manager.reset()


class RegisterManager:
    """Manages virtual registers according to COIL spec"""
    
    def __init__(self):
        # Initialize general purpose registers
        # 16 registers in each width category
        self.byte_registers = [0] * 16  # 8-bit  (RB0-RB15)
        self.word_registers = [0] * 16  # 16-bit (RW0-RW15)
        self.long_registers = [0] * 16  # 32-bit (RL0-RL15)
        self.quad_registers = [0] * 16  # 64-bit (RQ0-RQ15)
        
        # Special registers
        self.rsp = 0  # Stack pointer
        self.rbp = 0  # Base pointer
        self.rip = 0  # Instruction pointer
        self.rf = 0   # Flags register
        
        # Segment registers (for architectures that support memory segmentation)
        self.segment_registers = [0] * 6  # S0-S5
        
        # Flag bits within RF register
        self.FLAG_ZERO = 0x01      # Zero flag
        self.FLAG_SIGN = 0x02      # Sign flag
        self.FLAG_OVERFLOW = 0x04  # Overflow flag
        self.FLAG_CARRY = 0x08     # Carry flag
        self.FLAG_PARITY = 0x10    # Parity flag
    
    def get_register(self, reg_type, reg_num):
        """Get value from specified register"""
        if reg_type == 'RB':
            return self.byte_registers[reg_num]
        elif reg_type == 'RW':
            return self.word_registers[reg_num]
        elif reg_type == 'RL':
            return self.long_registers[reg_num]
        elif reg_type == 'RQ':
            return self.quad_registers[reg_num]
        elif reg_type == 'RSP':
            return self.rsp
        elif reg_type == 'RBP':
            return self.rbp
        elif reg_type == 'RIP':
            return self.rip
        elif reg_type == 'RF':
            return self.rf
        elif reg_type.startswith('S'):
            s_num = int(reg_type[1:])
            return self.segment_registers[s_num]
        else:
            raise ValueError(f"Unknown register: {reg_type}{reg_num}")
    
    def set_register(self, reg_type, reg_num, value):
        """Set value in specified register"""
        if reg_type == 'RB':
            self.byte_registers[reg_num] = value & 0xFF  # Mask to 8 bits
        elif reg_type == 'RW':
            self.word_registers[reg_num] = value & 0xFFFF  # Mask to 16 bits
        elif reg_type == 'RL':
            self.long_registers[reg_num] = value & 0xFFFFFFFF  # Mask to 32 bits
        elif reg_type == 'RQ':
            self.quad_registers[reg_num] = value & 0xFFFFFFFFFFFFFFFF  # Mask to 64 bits
        elif reg_type == 'RSP':
            self.rsp = value
        elif reg_type == 'RBP':
            self.rbp = value
        elif reg_type == 'RIP':
            self.rip = value
        elif reg_type == 'RF':
            self.rf = value
        elif reg_type.startswith('S'):
            s_num = int(reg_type[1:])
            self.segment_registers[s_num] = value
        else:
            raise ValueError(f"Unknown register: {reg_type}{reg_num}")
    
    def set_flag(self, flag, value):
        """Set a specific flag bit"""
        if value:
            self.rf |= flag
        else:
            self.rf &= ~flag
    
    def get_flag(self, flag):
        """Get a specific flag bit"""
        return bool(self.rf & flag)
    
    def update_flags_after_operation(self, result, width, carry=False, overflow=False):
        """Update flag register after an operation"""
        # Determine the max value for the given width
        if width == 8:    # byte
            mask = 0xFF
            sign_bit = 0x80
        elif width == 16:  # word
            mask = 0xFFFF
            sign_bit = 0x8000
        elif width == 32:  # long
            mask = 0xFFFFFFFF
            sign_bit = 0x80000000
        elif width == 64:  # quad
            mask = 0xFFFFFFFFFFFFFFFF
            sign_bit = 0x8000000000000000
        else:
            raise ValueError(f"Invalid register width: {width}")
        
        # Mask the result to the appropriate width
        result &= mask
        
        # Set zero flag
        self.set_flag(self.FLAG_ZERO, result == 0)
        
        # Set sign flag
        self.set_flag(self.FLAG_SIGN, bool(result & sign_bit))
        
        # Set carry flag
        self.set_flag(self.FLAG_CARRY, carry)
        
        # Set overflow flag
        self.set_flag(self.FLAG_OVERFLOW, overflow)
        
        # Set parity flag (count number of set bits)
        count = bin(result & 0xFF).count('1')
        self.set_flag(self.FLAG_PARITY, count % 2 == 0)  # Even parity
    
    def reset(self):
        """Reset all registers to 0"""
        for i in range(16):
            self.byte_registers[i] = 0
            self.word_registers[i] = 0
            self.long_registers[i] = 0
            self.quad_registers[i] = 0
        
        self.rsp = 0
        self.rbp = 0
        self.rip = 0
        self.rf = 0
        
        for i in range(6):
            self.segment_registers[i] = 0


class MemoryManager:
    """Enhanced memory manager with better error handling and debugging"""
    
    def __init__(self, memory_size=1024*1024):  # Default 1MB memory
        """Initialize memory manager with the specified size"""
        self.memory_size = max(1024, memory_size)  # Ensure minimum size
        print(f"Initializing memory manager with {self.memory_size} bytes")
        self.memory = bytearray(self.memory_size)
        self.sections = {}  # Map of named sections
    
    def _check_address(self, address, operation="access", size=1):
        """
        Check if an address (and range) is valid
        Provides detailed error information for debugging
        """
        # Convert to unsigned 32-bit if given a negative value
        # This handles potential sign extension issues from 32 to 64 bit
        if address < 0:
            print(f"Warning: Converting negative address 0x{address:x} ({address}) to unsigned")
            address = address & 0xFFFFFFFF
            print(f"Converted to: 0x{address:x} ({address})")
        
        # Check if the address is within memory bounds
        if address < 0 or address >= self.memory_size:
            raise MemoryError(
                f"Address out of range during {operation}: 0x{address:x} ({address}), "
                f"valid range is 0 to 0x{self.memory_size-1:x} ({self.memory_size-1})"
            )
        
        # Check if the address range fits in memory
        if address + size > self.memory_size:
            raise MemoryError(
                f"Memory operation at 0x{address:x} with size {size} exceeds memory bounds "
                f"(limit: 0x{self.memory_size-1:x})"
            )
        
        return address  # Return possibly adjusted address
    
    def read_byte(self, address):
        """Read a byte from memory with enhanced error checking"""
        address = self._check_address(address, "read_byte")
        return self.memory[address]
    
    def write_byte(self, address, value):
        """Write a byte to memory with enhanced error checking"""
        address = self._check_address(address, "write_byte")
        self.memory[address] = value & 0xFF
    
    def read_word(self, address):
        """Read a 16-bit word from memory with enhanced error checking"""
        address = self._check_address(address, "read_word", 2)
        return (self.memory[address] | 
                (self.memory[address + 1] << 8))
    
    def write_word(self, address, value):
        """Write a 16-bit word to memory with enhanced error checking"""
        address = self._check_address(address, "write_word", 2)
        self.memory[address] = value & 0xFF
        self.memory[address + 1] = (value >> 8) & 0xFF
    
    def read_long(self, address):
        """Read a 32-bit long from memory with enhanced error checking"""
        address = self._check_address(address, "read_long", 4)
        return (self.memory[address] | 
                (self.memory[address + 1] << 8) |
                (self.memory[address + 2] << 16) |
                (self.memory[address + 3] << 24))
    
    def write_long(self, address, value):
        """Write a 32-bit long to memory with enhanced error checking"""
        address = self._check_address(address, "write_long", 4)
        self.memory[address] = value & 0xFF
        self.memory[address + 1] = (value >> 8) & 0xFF
        self.memory[address + 2] = (value >> 16) & 0xFF
        self.memory[address + 3] = (value >> 24) & 0xFF
    
    def read_quad(self, address):
        """Read a 64-bit quad from memory with enhanced error checking"""
        address = self._check_address(address, "read_quad", 8)
        return (self.memory[address] | 
                (self.memory[address + 1] << 8) |
                (self.memory[address + 2] << 16) |
                (self.memory[address + 3] << 24) |
                (self.memory[address + 4] << 32) |
                (self.memory[address + 5] << 40) |
                (self.memory[address + 6] << 48) |
                (self.memory[address + 7] << 56))
    
    def write_quad(self, address, value):
        """Write a 64-bit quad to memory with enhanced error checking"""
        address = self._check_address(address, "write_quad", 8)
        self.memory[address] = value & 0xFF
        self.memory[address + 1] = (value >> 8) & 0xFF
        self.memory[address + 2] = (value >> 16) & 0xFF
        self.memory[address + 3] = (value >> 24) & 0xFF
        self.memory[address + 4] = (value >> 32) & 0xFF
        self.memory[address + 5] = (value >> 40) & 0xFF
        self.memory[address + 6] = (value >> 48) & 0xFF
        self.memory[address + 7] = (value >> 56) & 0xFF
    
    def read_bytes(self, address, length):
        """Read block of bytes from memory with enhanced error checking"""
        address = self._check_address(address, "read_bytes", length)
        return bytes(self.memory[address:address + length])
    
    def write_bytes(self, address, data):
        """Write block of bytes to memory with enhanced error checking"""
        length = len(data)
        address = self._check_address(address, "write_bytes", length)
        self.memory[address:address + length] = data
    
    def add_section(self, name, start_address, size, flags):
        """Add a named memory section with validation"""
        if start_address < 0:
            raise ValueError(f"Section {name} has negative start address: {start_address}")
        
        if size <= 0:
            raise ValueError(f"Section {name} has invalid size: {size}")
        
        if start_address + size > self.memory_size:
            raise MemoryError(
                f"Section {name} (0x{start_address:x}-0x{start_address+size-1:x}) "
                f"exceeds memory size (0x{self.memory_size-1:x})"
            )
        
        self.sections[name] = {
            'start': start_address,
            'size': size,
            'flags': flags
        }
        print(f"Added section: {name} at 0x{start_address:x} size {size} bytes")
    
    def get_section(self, name):
        """Get information about a named section"""
        return self.sections.get(name)
    
    def reset(self):
        """Reset all memory to zero"""
        self.memory = bytearray(self.memory_size)
        self.sections = {}
        print("Memory reset complete")
    
    def dump_range(self, start, size, width=16):
        """Dump a range of memory for debugging purposes"""
        if start < 0 or start >= self.memory_size:
            print(f"Error: Cannot dump memory at invalid address: 0x{start:x}")
            return
        
        end = min(start + size, self.memory_size)
        
        print(f"Memory dump from 0x{start:08x} to 0x{end-1:08x}:")
        for addr in range(start, end, width):
            row_data = self.memory[addr:min(addr+width, end)]
            hex_values = " ".join(f"{b:02x}" for b in row_data)
            ascii_repr = "".join(chr(b) if 32 <= b <= 126 else "." for b in row_data)
            
            # Pad hex values to align ASCII representation
            padding = "   " * (width - len(row_data))
            
            print(f"  0x{addr:08x}: {hex_values}{padding}  |{ascii_repr}|")


class TypeSystem:
    """Implements the COIL type system"""
    
    def __init__(self):
        # Type constants from the COIL specification
        self.COIL_TYPE_INT = 0x00    # Signed integer
        self.COIL_TYPE_UINT = 0x01   # Unsigned integer
        self.COIL_TYPE_FLOAT = 0x02  # Floating point
        
        # Extended types (reserved for future)
        self.COIL_TYPE_VEC = 0x10    # Vector type
        
        # Composite types (reserved for future)
        self.COIL_TYPE_STRUCT = 0x20 # Structure type
        self.COIL_TYPE_UNION = 0x21  # Union type
        self.COIL_TYPE_PACK = 0x22   # Packed structure
        
        # Memory types (reserved for future)
        self.COIL_TYPE_ARR = 0x40    # Array type
        
        # Other types
        self.COIL_TYPE_VOID = 0xF0   # Void type
        self.COIL_TYPE_BOOL = 0xF1   # Boolean type
        self.COIL_TYPE_LINT = 0xF2   # Largest native integer
        self.COIL_TYPE_FINT = 0xF3   # Fastest native integer
        self.COIL_TYPE_PTR = 0xF4    # Pointer type
        
        # Parameter types
        self.COIL_TYPE_PARAM2 = 0xFD # Parameter type 2
        self.COIL_TYPE_PARAM1 = 0xFE # Parameter type 1
        self.COIL_TYPE_PARAM0 = 0xFF # Parameter type 0
        
        # Type widths
        self.WIDTH_8 = 0x00    # 8-bit
        self.WIDTH_16 = 0x01   # 16-bit
        self.WIDTH_32 = 0x02   # 32-bit
        self.WIDTH_64 = 0x03   # 64-bit
        self.WIDTH_128 = 0x04  # 128-bit (future)
    
    def get_type_size(self, type_code, width_code=None):
        """Get the size in bytes for a given type"""
        if type_code in [self.COIL_TYPE_INT, self.COIL_TYPE_UINT]:
            if width_code == self.WIDTH_8:
                return 1
            elif width_code == self.WIDTH_16:
                return 2
            elif width_code == self.WIDTH_32:
                return 4
            elif width_code == self.WIDTH_64:
                return 8
            else:
                raise ValueError(f"Invalid width code for integer: {width_code}")
        
        elif type_code == self.COIL_TYPE_FLOAT:
            if width_code == self.WIDTH_32:
                return 4
            elif width_code == self.WIDTH_64:
                return 8
            else:
                raise ValueError(f"Invalid width code for float: {width_code}")
        
        elif type_code == self.COIL_TYPE_BOOL:
            return 1
        
        elif type_code == self.COIL_TYPE_PTR:
            return 8  # 64-bit pointers by default
        
        elif type_code == self.COIL_TYPE_VOID:
            return 0
        
        elif type_code == self.COIL_TYPE_LINT:
            return 8  # 64-bit by default
        
        elif type_code == self.COIL_TYPE_FINT:
            return 4  # 32-bit by default
        
        else:
            raise ValueError(f"Unsupported type: {type_code}")
    
    def is_signed(self, type_code):
        """Check if a type is signed"""
        return type_code == self.COIL_TYPE_INT
    
    def is_integer(self, type_code):
        """Check if a type is an integer type"""
        return type_code in [self.COIL_TYPE_INT, self.COIL_TYPE_UINT]
    
    def is_float(self, type_code):
        """Check if a type is a floating point type"""
        return type_code == self.COIL_TYPE_FLOAT
    
    def get_conversion_cost(self, src_type, src_width, dst_type, dst_width):
        """Calculate the cost of converting between types"""
        # Higher cost means more risky conversion
        # 0 means no conversion needed
        # -1 means impossible conversion
        
        # Same type and width, no conversion needed
        if src_type == dst_type and src_width == dst_width:
            return 0
        
        # Integer to integer conversions
        if self.is_integer(src_type) and self.is_integer(dst_type):
            # Widening - safe
            if src_width < dst_width:
                # Unsigned to signed is safest
                if src_type == self.COIL_TYPE_UINT and dst_type == self.COIL_TYPE_INT:
                    return 1
                # Same signedness is also safe
                elif src_type == dst_type:
                    return 2
                # Signed to unsigned loses sign info
                else:
                    return 5
            # Narrowing - potential data loss
            else:
                return 10
        
        # Float to float conversions
        elif self.is_float(src_type) and self.is_float(dst_type):
            # Widening - safe
            if src_width < dst_width:
                return 1
            # Narrowing - potential precision loss
            else:
                return 8
        
        # Integer to float - potential precision loss for large integers
        elif self.is_integer(src_type) and self.is_float(dst_type):
            return 6
        
        # Float to integer - potential data loss
        elif self.is_float(src_type) and self.is_integer(dst_type):
            return 12
        
        # Other conversions not directly supported
        return -1


class InstructionInterpreter:
    """Interprets and executes COIL instructions"""
    
    def __init__(self, vm):
        self.vm = vm
        
        # Define opcode handlers
        self.opcode_handlers = {
            0x00: self._handle_nop,         # NOP
            0x10: self._handle_add,         # ADD
            0x11: self._handle_sub,         # SUB
            0x12: self._handle_mul,         # MUL
            0x13: self._handle_div,         # DIV
            0x30: self._handle_cmp,         # CMP
            0x40: self._handle_mov,         # MOV
            0x41: self._handle_load,        # LOAD
            0x42: self._handle_store,       # STORE
            0x50: self._handle_push,        # PUSH
            0x51: self._handle_pop,         # POP
            # Add more handlers as needed
        }
        
        # Operand qualifier constants
        self.OPQUAL_IMM = 0x01  # Immediate value
        self.OPQUAL_VAR = 0x02  # Variable
        self.OPQUAL_REG = 0x03  # Register
        self.OPQUAL_MEM = 0x04  # Memory address
        self.OPQUAL_LBL = 0x05  # Label
        self.OPQUAL_STR = 0x06  # String
        self.OPQUAL_SYM = 0x07  # Symbol
        self.OPQUAL_REL = 0x08  # Relative offset
    
    def decode_instruction(self, address):
        """Decode instruction at given address with detailed debug info"""
        mem = self.vm.memory_manager
        
        try:
            # Instruction format:
            # [opcode(1)][qualifier(1)][operand_count(1)][operands...]
            opcode = mem.read_byte(address)
            qualifier = mem.read_byte(address + 1)
            operand_count = mem.read_byte(address + 2)
            
            print(f"DEBUG: Decoded header - opcode: 0x{opcode:02x}, qualifier: 0x{qualifier:02x}, operand_count: {operand_count}")
            
            # Check for potentially invalid instruction format
            if operand_count > 8:  # Arbitrary reasonable limit
                print(f"WARNING: Suspicious operand count: {operand_count} - might not be a valid instruction")
                
            operands = []
            current_pos = address + 3
            
            for i in range(operand_count):
                print(f"DEBUG: Parsing operand {i+1} at offset 0x{current_pos - address:x} from instruction start")
                
                # Operand format: [qualifier(1)][type(1)][data(variable)]
                op_qualifier = mem.read_byte(current_pos)
                op_type = mem.read_byte(current_pos + 1)
                
                print(f"DEBUG: Operand {i+1} - qualifier: 0x{op_qualifier:02x}, type: 0x{op_type:02x}")
                
                # Check for suspicious operand qualifiers
                valid_qualifiers = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]
                if op_qualifier not in valid_qualifiers:
                    # This is likely not a real instruction - check if it's ASCII
                    if 32 <= opcode <= 126 and 32 <= qualifier <= 126 and 32 <= operand_count <= 126:
                        ascii_chars = chr(opcode) + chr(qualifier) + chr(operand_count)
                        print(f"ERROR: This looks like ASCII text, not a COIL instruction: '{ascii_chars}...'")
                        
                        # Show a longer ASCII representation
                        try:
                            text_bytes = mem.read_bytes(address, 32)
                            ascii_text = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in text_bytes)
                            print(f"ASCII Text found: '{ascii_text}'")
                        except Exception:
                            pass
                            
                    raise ValueError(f"Unsupported operand qualifier: {op_qualifier} (might not be a valid instruction)")
                
                # Handle based on qualifier
                if op_qualifier == self.OPQUAL_IMM:
                    # Immediate value - size depends on type
                    data_size = self.vm.type_system.get_type_size(op_type)
                    if data_size == 1:
                        value = mem.read_byte(current_pos + 2)
                    elif data_size == 2:
                        value = mem.read_word(current_pos + 2)
                    elif data_size == 4:
                        value = mem.read_long(current_pos + 2)
                    elif data_size == 8:
                        value = mem.read_quad(current_pos + 2)
                    else:
                        raise ValueError(f"Unsupported immediate size: {data_size}")
                    
                    print(f"DEBUG: Immediate value: 0x{value:x}")
                    
                    operands.append({
                        'qualifier': op_qualifier,
                        'type': op_type,
                        'value': value
                    })
                    current_pos += 2 + data_size
                    
                elif op_qualifier == self.OPQUAL_REG:
                    # Register reference
                    reg_index = mem.read_byte(current_pos + 2)
                    print(f"DEBUG: Register index: {reg_index}")
                    
                    operands.append({
                        'qualifier': op_qualifier,
                        'type': op_type,
                        'reg_index': reg_index
                    })
                    current_pos += 3
                    
                elif op_qualifier == self.OPQUAL_MEM:
                    # Memory address - using 64-bit addresses
                    address_value = mem.read_quad(current_pos + 2)
                    print(f"DEBUG: Memory address: 0x{address_value:x}")
                    
                    operands.append({
                        'qualifier': op_qualifier,
                        'type': op_type,
                        'address': address_value
                    })
                    current_pos += 10
                    
                else:
                    # Other operand types
                    print(f"DEBUG: Unimplemented qualifier: 0x{op_qualifier:02x}")
                    # For now, just skip this operand - implement as needed
                    current_pos += 2  # Skip the qualifier and type, but can't determine full size
                    
                    operands.append({
                        'qualifier': op_qualifier,
                        'type': op_type
                    })
            
            # Calculate instruction size for RIP update
            instruction_size = current_pos - address
            
            return {
                'opcode': opcode,
                'qualifier': qualifier,
                'operand_count': operand_count,
                'operands': operands,
                'size': instruction_size
            }
        except Exception as e:
            print(f"ERROR during instruction decoding at 0x{address:08x}: {e}")
            # Dump more context for debugging
            try:
                print("Memory context around this address:")
                for offset in range(-16, 32, 16):
                    if address + offset >= 0:
                        data = mem.read_bytes(address + offset, 16)
                        hex_dump = ' '.join(f'{b:02x}' for b in data)
                        ascii_dump = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in data)
                        print(f"0x{address+offset:08x}: {hex_dump}  |{ascii_dump}|")
            except Exception:
                pass
            raise

    def execute(self, start_address):
        """Execute instructions starting at given address"""
        self.vm.register_manager.set_register('RIP', 0, start_address)
        
        running = True
        while running:
            current_rip = self.vm.register_manager.get_register('RIP', 0)
            instruction = self.decode_instruction(current_rip)
            
            # Update RIP to next instruction
            self.vm.register_manager.set_register('RIP', 0, current_rip + instruction['size'])
            
            # Execute the instruction
            handler = self.opcode_handlers.get(instruction['opcode'])
            if handler:
                running = handler(instruction)
            else:
                raise ValueError(f"Unknown opcode: {instruction['opcode']:#04x}")
            
            # Check if we should stop execution
            if not running:
                break
        
        return current_rip
    
    def _handle_nop(self, instruction):
        """Handle NOP instruction"""
        # Do nothing
        return True
    
    def _handle_add(self, instruction):
        """Handle ADD instruction (opcode 0x10)"""
        # ADD dest, src1, src2
        if len(instruction['operands']) != 3:
            raise ValueError("ADD requires 3 operands")
        
        dest = instruction['operands'][0]
        src1 = instruction['operands'][1]
        src2 = instruction['operands'][2]
        
        # Get source values
        src1_value = self._get_operand_value(src1)
        src2_value = self._get_operand_value(src2)
        
        # Perform addition
        result = src1_value + src2_value
        
        # Store result
        self._set_operand_value(dest, result)
        
        # Update flags
        # For simplicity, just using 64-bit flags
        width = 64
        carry = (src1_value + src2_value) > ((1 << width) - 1)
        
        # Simple overflow check for signed integers
        if instruction['qualifier'] & 0x01:  # Check if ARITH_SIGNED bit is set
            src1_sign = (src1_value >> (width - 1)) & 1
            src2_sign = (src2_value >> (width - 1)) & 1
            result_sign = (result >> (width - 1)) & 1
            overflow = (src1_sign == src2_sign) and (result_sign != src1_sign)
        else:
            overflow = False
        
        self.vm.register_manager.update_flags_after_operation(result, width, carry, overflow)
        
        return True
    
    def _handle_sub(self, instruction):
        """Handle SUB instruction (opcode 0x11)"""
        # SUB dest, src1, src2
        if len(instruction['operands']) != 3:
            raise ValueError("SUB requires 3 operands")
        
        dest = instruction['operands'][0]
        src1 = instruction['operands'][1]
        src2 = instruction['operands'][2]
        
        # Get source values
        src1_value = self._get_operand_value(src1)
        src2_value = self._get_operand_value(src2)
        
        # Perform subtraction
        result = src1_value - src2_value
        
        # Store result
        self._set_operand_value(dest, result)
        
        # Update flags - for simplicity, assume 64-bit operations
        width = 64
        carry = src1_value < src2_value
        
        # Overflow check for signed integers
        if instruction['qualifier'] & 0x01:  # Check if ARITH_SIGNED bit is set
            src1_sign = (src1_value >> (width - 1)) & 1
            src2_sign = (src2_value >> (width - 1)) & 1
            result_sign = (result >> (width - 1)) & 1
            overflow = (src1_sign != src2_sign) and (result_sign != src1_sign)
        else:
            overflow = False
        
        self.vm.register_manager.update_flags_after_operation(result, width, carry, overflow)
        
        return True
    
    def _handle_mul(self, instruction):
        """Handle MUL instruction (opcode 0x12)"""
        # MUL dest, src1, src2
        if len(instruction['operands']) != 3:
            raise ValueError("MUL requires 3 operands")
        
        dest = instruction['operands'][0]
        src1 = instruction['operands'][1]
        src2 = instruction['operands'][2]
        
        # Get source values
        src1_value = self._get_operand_value(src1)
        src2_value = self._get_operand_value(src2)
        
        # Perform multiplication
        result = src1_value * src2_value
        
        # Store result
        self._set_operand_value(dest, result)
        
        # Update flags - simple implementation
        width = 64
        overflow = (result >> width) != 0
        carry = overflow
        
        self.vm.register_manager.update_flags_after_operation(result, width, carry, overflow)
        
        return True
    
    def _handle_div(self, instruction):
        """Handle DIV instruction (opcode 0x13)"""
        # DIV dest, src1, src2
        if len(instruction['operands']) != 3:
            raise ValueError("DIV requires 3 operands")
        
        dest = instruction['operands'][0]
        src1 = instruction['operands'][1]
        src2 = instruction['operands'][2]
        
        # Get source values
        src1_value = self._get_operand_value(src1)
        src2_value = self._get_operand_value(src2)
        
        # Check for division by zero
        if src2_value == 0:
            raise ZeroDivisionError("Division by zero")
        
        # Perform division
        result = src1_value // src2_value
        
        # Store result
        self._set_operand_value(dest, result)
        
        # Update flags - simple implementation
        width = 64
        self.vm.register_manager.update_flags_after_operation(result, width, False, False)
        
        return True
    
    def _handle_cmp(self, instruction):
        """Handle CMP instruction (opcode 0x30)"""
        # CMP src1, src2
        if len(instruction['operands']) != 2:
            raise ValueError("CMP requires 2 operands")
        
        src1 = instruction['operands'][0]
        src2 = instruction['operands'][1]
        
        # Get source values
        src1_value = self._get_operand_value(src1)
        src2_value = self._get_operand_value(src2)
        
        # Perform comparison (effectively src1 - src2)
        result = src1_value - src2_value
        
        # Update flags
        width = 64
        carry = src1_value < src2_value
        
        # Simple overflow check for signed integers
        src1_sign = (src1_value >> (width - 1)) & 1
        src2_sign = (src2_value >> (width - 1)) & 1
        result_sign = (result >> (width - 1)) & 1
        overflow = (src1_sign != src2_sign) and (result_sign != src1_sign)
        
        self.vm.register_manager.update_flags_after_operation(result, width, carry, overflow)
        
        return True
    
    def _handle_mov(self, instruction):
        """Handle MOV instruction (opcode 0x40)"""
        # MOV dest, src
        if len(instruction['operands']) != 2:
            raise ValueError("MOV requires 2 operands")
        
        dest = instruction['operands'][0]
        src = instruction['operands'][1]
        
        # Get source value
        src_value = self._get_operand_value(src)
        
        # Store to destination
        self._set_operand_value(dest, src_value)
        
        return True
    
    def _handle_load(self, instruction):
        """Handle LOAD instruction (opcode 0x41)"""
        # LOAD dest, addr
        if len(instruction['operands']) != 2:
            raise ValueError("LOAD requires 2 operands")
        
        dest = instruction['operands'][0]
        addr = instruction['operands'][1]
        
        # Get memory address
        addr_value = self._get_operand_value(addr)
        
        # Determine load size based on destination type
        dest_type = dest['type']
        type_size = self.vm.type_system.get_type_size(dest_type)
        
        # Load from memory
        if type_size == 1:
            value = self.vm.memory_manager.read_byte(addr_value)
        elif type_size == 2:
            value = self.vm.memory_manager.read_word(addr_value)
        elif type_size == 4:
            value = self.vm.memory_manager.read_long(addr_value)
        elif type_size == 8:
            value = self.vm.memory_manager.read_quad(addr_value)
        else:
            raise ValueError(f"Unsupported load size: {type_size}")
        
        # Store to destination
        self._set_operand_value(dest, value)
        
        return True
    
    def _handle_store(self, instruction):
        """Handle STORE instruction (opcode 0x42)"""
        # STORE addr, src
        if len(instruction['operands']) != 2:
            raise ValueError("STORE requires 2 operands")
        
        addr = instruction['operands'][0]
        src = instruction['operands'][1]
        
        # Get memory address and source value
        addr_value = self._get_operand_value(addr)
        src_value = self._get_operand_value(src)
        
        # Determine store size based on source type
        src_type = src['type']
        type_size = self.vm.type_system.get_type_size(src_type)
        
        # Store to memory
        if type_size == 1:
            self.vm.memory_manager.write_byte(addr_value, src_value)
        elif type_size == 2:
            self.vm.memory_manager.write_word(addr_value, src_value)
        elif type_size == 4:
            self.vm.memory_manager.write_long(addr_value, src_value)
        elif type_size == 8:
            self.vm.memory_manager.write_quad(addr_value, src_value)
        else:
            raise ValueError(f"Unsupported store size: {type_size}")
        
        return True
    
    def _handle_push(self, instruction):
        """Handle PUSH instruction (opcode 0x50)"""
        # PUSH src
        if len(instruction['operands']) != 1:
            raise ValueError("PUSH requires 1 operand")
        
        src = instruction['operands'][0]
        
        # Get source value
        src_value = self._get_operand_value(src)
        
        # Determine push size based on source type
        src_type = src['type']
        type_size = self.vm.type_system.get_type_size(src_type)
        
        # Update stack pointer
        rsp = self.vm.register_manager.get_register('RSP', 0)
        rsp -= type_size
        self.vm.register_manager.set_register('RSP', 0, rsp)
        
        # Store value on stack
        if type_size == 1:
            self.vm.memory_manager.write_byte(rsp, src_value)
        elif type_size == 2:
            self.vm.memory_manager.write_word(rsp, src_value)
        elif type_size == 4:
            self.vm.memory_manager.write_long(rsp, src_value)
        elif type_size == 8:
            self.vm.memory_manager.write_quad(rsp, src_value)
        else:
            raise ValueError(f"Unsupported push size: {type_size}")
        
        return True
    
    def _handle_pop(self, instruction):
        """Handle POP instruction (opcode 0x51)"""
        # POP dest
        if len(instruction['operands']) != 1:
            raise ValueError("POP requires 1 operand")
        
        dest = instruction['operands'][0]
        
        # Determine pop size based on destination type
        dest_type = dest['type']
        type_size = self.vm.type_system.get_type_size(dest_type)
        
        # Get current stack pointer
        rsp = self.vm.register_manager.get_register('RSP', 0)
        
        # Load value from stack
        if type_size == 1:
            value = self.vm.memory_manager.read_byte(rsp)
        elif type_size == 2:
            value = self.vm.memory_manager.read_word(rsp)
        elif type_size == 4:
            value = self.vm.memory_manager.read_long(rsp)
        elif type_size == 8:
            value = self.vm.memory_manager.read_quad(rsp)
        else:
            raise ValueError(f"Unsupported pop size: {type_size}")
        
        # Update stack pointer
        rsp += type_size
        self.vm.register_manager.set_register('RSP', 0, rsp)
        
        # Store value to destination
        self._set_operand_value(dest, value)
        
        return True
    
    def _get_operand_value(self, operand):
        """Get the value of an operand"""
        qualifier = operand['qualifier']
        
        if qualifier == self.OPQUAL_IMM:
            # Immediate value
            return operand['value']
            
        elif qualifier == self.OPQUAL_REG:
            # Register value
            reg_type = None
            reg_width = None
            
            # Determine register type and width from the operand type
            op_type = operand['type']
            if op_type == self.vm.type_system.COIL_TYPE_INT or op_type == self.vm.type_system.COIL_TYPE_UINT:
                op_width = operand.get('width', self.vm.type_system.WIDTH_64)  # Default to 64-bit
                
                if op_width == self.vm.type_system.WIDTH_8:
                    reg_type = 'RB'
                elif op_width == self.vm.type_system.WIDTH_16:
                    reg_type = 'RW'
                elif op_width == self.vm.type_system.WIDTH_32:
                    reg_type = 'RL'
                else:  # Default to 64-bit
                    reg_type = 'RQ'
            else:
                # Default to 64-bit register for other types
                reg_type = 'RQ'
            
            reg_index = operand['reg_index']
            return self.vm.register_manager.get_register(reg_type, reg_index)
            
        elif qualifier == self.OPQUAL_MEM:
            # Memory address - get value at address
            address = operand['address']
            op_type = operand['type']
            type_size = self.vm.type_system.get_type_size(op_type)
            
            if type_size == 1:
                return self.vm.memory_manager.read_byte(address)
            elif type_size == 2:
                return self.vm.memory_manager.read_word(address)
            elif type_size == 4:
                return self.vm.memory_manager.read_long(address)
            elif type_size == 8:
                return self.vm.memory_manager.read_quad(address)
            else:
                raise ValueError(f"Unsupported memory read size: {type_size}")
        
        else:
            raise ValueError(f"Unsupported operand qualifier: {qualifier}")
    
    def _set_operand_value(self, operand, value):
        """Set the value of an operand"""
        qualifier = operand['qualifier']
        
        if qualifier == self.OPQUAL_REG:
            # Register value
            reg_type = None
            
            # Determine register type and width from the operand type
            op_type = operand['type']
            if op_type == self.vm.type_system.COIL_TYPE_INT or op_type == self.vm.type_system.COIL_TYPE_UINT:
                op_width = operand.get('width', self.vm.type_system.WIDTH_64)  # Default to 64-bit
                
                if op_width == self.vm.type_system.WIDTH_8:
                    reg_type = 'RB'
                elif op_width == self.vm.type_system.WIDTH_16:
                    reg_type = 'RW'
                elif op_width == self.vm.type_system.WIDTH_32:
                    reg_type = 'RL'
                else:  # Default to 64-bit
                    reg_type = 'RQ'
            else:
                # Default to 64-bit register for other types
                reg_type = 'RQ'
            
            reg_index = operand['reg_index']
            self.vm.register_manager.set_register(reg_type, reg_index, value)
            
        elif qualifier == self.OPQUAL_MEM:
            # Memory address - set value at address
            address = operand['address']
            op_type = operand['type']
            type_size = self.vm.type_system.get_type_size(op_type)
            
            if type_size == 1:
                self.vm.memory_manager.write_byte(address, value)
            elif type_size == 2:
                self.vm.memory_manager.write_word(address, value)
            elif type_size == 4:
                self.vm.memory_manager.write_long(address, value)
            elif type_size == 8:
                self.vm.memory_manager.write_quad(address, value)
            else:
                raise ValueError(f"Unsupported memory write size: {type_size}")
        
        else:
            raise ValueError(f"Unsupported operand qualifier for setting value: {qualifier}")


class COFLoader:
    """Enhanced COF loader with better validation and error handling"""
    
    def __init__(self, vm):
        self.vm = vm
        self.entry_point = 0
        self.loaded_sections = {}
        self.symbols = {}
        self.string_table = bytearray()
        self.debug_mode = False  # Set to True to enable verbose debug output
    
    def set_debug(self, debug):
        """Enable or disable debug output"""
        self.debug_mode = debug
    
    def debug_print(self, message):
        """Print a debug message if debug mode is enabled"""
        if self.debug_mode:
            print(f"[COF Loader] {message}")
    
    def load(self, filename):
        """Load a COF file into the VM with enhanced error checking"""
        try:
            with open(filename, 'rb') as file:
                data = file.read()
            
            self.debug_print(f"Loaded {len(data)} bytes from file: {filename}")
            return self._load_from_memory(data)
            
        except FileNotFoundError:
            raise FileNotFoundError(f"COF file not found: {filename}")
        except PermissionError:
            raise PermissionError(f"Permission denied when accessing COF file: {filename}")
        except Exception as e:
            raise RuntimeError(f"Error loading COF file: {str(e)}")
    
    def _load_from_memory(self, data):
        """Load a COF file from memory with enhanced validation"""
        # Validate minimum file size for the header
        if len(data) < 32:
            raise ValueError(f"COF file too small ({len(data)} bytes), minimum is 32 bytes for the header")
        
        # Validate COF magic number - should be 'COIL' (0x434F494C)
        magic = int.from_bytes(data[0:4], byteorder='little')
        if magic != 0x434F494C:
            raise ValueError(f"Invalid COF file: magic number mismatch: 0x{magic:x}, expected 0x434F494C")
        
        # Parse COF header
        version_major = data[4]
        version_minor = data[5]
        version_patch = data[6]
        flags = data[7]
        target = int.from_bytes(data[8:10], byteorder='little')
        section_count = int.from_bytes(data[10:12], byteorder='little')
        entrypoint = int.from_bytes(data[12:16], byteorder='little')
        str_tab_off = int.from_bytes(data[16:20], byteorder='little')
        str_tab_size = int.from_bytes(data[20:24], byteorder='little')
        sym_tab_off = int.from_bytes(data[24:28], byteorder='little')
        sym_tab_size = int.from_bytes(data[28:32], byteorder='little')
        
        self.debug_print(f"COF Version: {version_major}.{version_minor}.{version_patch}")
        self.debug_print(f"Target: 0x{target:04x}")
        self.debug_print(f"Sections: {section_count}")
        self.debug_print(f"Entry point: 0x{entrypoint:08x}")
        self.debug_print(f"String table: offset=0x{str_tab_off:x}, size={str_tab_size}")
        self.debug_print(f"Symbol table: offset=0x{sym_tab_off:x}, size={sym_tab_size}")
        
        # Validate header offsets against file size
        if str_tab_off > 0 and str_tab_off + str_tab_size > len(data):
            raise ValueError(
                f"String table (offset=0x{str_tab_off:x}, size={str_tab_size}) "
                f"extends beyond end of file (size={len(data)})"
            )
            
        if sym_tab_off > 0 and sym_tab_off + sym_tab_size > len(data):
            raise ValueError(
                f"Symbol table (offset=0x{sym_tab_off:x}, size={sym_tab_size}) "
                f"extends beyond end of file (size={len(data)})"
            )
        
        self.entry_point = entrypoint
        
        # Load string table if present
        if str_tab_size > 0 and str_tab_off > 0:
            self.string_table = data[str_tab_off:str_tab_off + str_tab_size]
            self.debug_print(f"Loaded string table ({str_tab_size} bytes)")
        else:
            self.debug_print("No string table found or empty string table")
        
        # Load sections
        section_offset = 32  # Start after header
        
        for section_idx in range(section_count):
            # Check if we have enough data for the section header
            if section_offset + 36 > len(data):
                raise ValueError(f"Section header {section_idx} extends beyond end of file")
            
            section = self._parse_section_header(data, section_offset, section_idx)
            section_offset += 36  # Size of section header
            
            section_name = self._get_string(section['name_offset'])
            self.debug_print(f"Section {section_idx}: {section_name}, offset=0x{section['offset']:x}, size={section['size']}")
            
            # Validate section data offset and size
            if section['size'] > 0:
                if section['offset'] >= len(data):
                    raise ValueError(
                        f"Section {section_idx} ({section_name}) data offset (0x{section['offset']:x}) "
                        f"is beyond end of file"
                    )
                
                if section['offset'] + section['size'] > len(data):
                    raise ValueError(
                        f"Section {section_idx} ({section_name}) data "
                        f"(offset=0x{section['offset']:x}, size={section['size']}) "
                        f"extends beyond end of file"
                    )
                
                # Extract section data
                section_data = data[section['offset']:section['offset'] + section['size']]
                
                # Calculate a suitable address for the section in VM memory
                # For simplicity, we'll use a fixed offset approach
                # In a real implementation, this would use a more sophisticated memory allocation strategy
                base_address = 0x10000  # Start at 64KB
                section_address = base_address + section_idx * 0x10000  # Give each section 64KB of space
                
                # Ensure we don't exceed memory size
                max_address = section_address + section['size']
                if max_address > self.vm.memory_manager.memory_size:
                    raise MemoryError(
                        f"Section {section_name} (size={section['size']}) would exceed VM memory size "
                        f"at address 0x{section_address:x}"
                    )
                
                self.debug_print(f"Loading section {section_name} to address 0x{section_address:x}")
                
                # Load section data into VM memory
                try:
                    self.vm.memory_manager.write_bytes(section_address, section_data)
                except Exception as e:
                    raise RuntimeError(f"Error loading section {section_name} data: {str(e)}")
                
                # Register the section in the VM
                self.vm.memory_manager.add_section(
                    section_name, 
                    section_address, 
                    section['size'], 
                    section['flags']
                )
                
                # Store mapping information for symbols
                self.loaded_sections[section_idx] = {
                    'name': section_name,
                    'address': section_address,
                    'size': section['size'],
                    'flags': section['flags']
                }
        
        # Load symbol table if present
        if sym_tab_size > 0 and sym_tab_off > 0:
            symbol_count = sym_tab_size // 16  # Each symbol entry is 16 bytes
            
            self.debug_print(f"Loading {symbol_count} symbols from symbol table")
            
            for i in range(symbol_count):
                offset = sym_tab_off + i * 16
                
                # Check if we have enough data for the symbol entry
                if offset + 16 > len(data):
                    raise ValueError(f"Symbol entry {i} extends beyond end of file")
                
                symbol = self._parse_symbol(data, offset)
                symbol_name = self._get_string(symbol['name_offset'])
                
                # Adjust symbol value based on section
                adjusted_value = symbol['value']
                if symbol['section_index'] > 0 and symbol['section_index'] in self.loaded_sections:
                    section = self.loaded_sections[symbol['section_index']]
                    adjusted_value += section['address']
                
                self.debug_print(
                    f"Symbol {i}: {symbol_name}, value=0x{symbol['value']:x}, "
                    f"adjusted=0x{adjusted_value:x}, section={symbol['section_index']}"
                )
                
                # Store the symbol
                self.symbols[symbol_name] = {
                    'name_offset': symbol['name_offset'],
                    'value': symbol['value'],
                    'adjusted_value': adjusted_value,
                    'size': symbol['size'],
                    'type': symbol['type'],
                    'binding': symbol['binding'],
                    'visibility': symbol['visibility'],
                    'section_index': symbol['section_index']
                }
        
        # Adjust entry point if it's relative to a section
        if entrypoint > 0 and entrypoint < 0x10000:  # Heuristic: small entry points are likely section-relative
            # Try to find a code section
            for section_idx, section in self.loaded_sections.items():
                if section['flags'] & 0x02:  # Check for EXEC flag
                    self.debug_print(f"Adjusting entry point relative to code section {section['name']}")
                    self.entry_point = section['address'] + entrypoint
                    break
        
        self.debug_print(f"Final entry point: 0x{self.entry_point:08x}")
        return self.entry_point
    
    def _parse_section_header(self, data, offset, section_idx):
        """Parse a section header from the COF file with validation"""
        name_offset = int.from_bytes(data[offset:offset+4], byteorder='little')
        type_value = int.from_bytes(data[offset+4:offset+8], byteorder='little')
        flags = int.from_bytes(data[offset+8:offset+12], byteorder='little')
        data_offset = int.from_bytes(data[offset+12:offset+16], byteorder='little')
        size = int.from_bytes(data[offset+16:offset+20], byteorder='little')
        link = int.from_bytes(data[offset+20:offset+24], byteorder='little')
        info = int.from_bytes(data[offset+24:offset+28], byteorder='little')
        alignment = int.from_bytes(data[offset+28:offset+32], byteorder='little')
        entsize = int.from_bytes(data[offset+32:offset+36], byteorder='little')
        
        # Validate critical fields
        if name_offset >= len(self.string_table) and len(self.string_table) > 0:
            self.debug_print(f"Warning: Section {section_idx} name offset (0x{name_offset:x}) is beyond end of string table")
        
        if size < 0:
            raise ValueError(f"Section {section_idx} has negative size: {size}")
        
        return {
            'name_offset': name_offset,
            'type': type_value,
            'flags': flags,
            'offset': data_offset,
            'size': size,
            'link': link,
            'info': info,
            'alignment': alignment,
            'entsize': entsize
        }
    
    def _parse_symbol(self, data, offset):
        """Parse a symbol from the COF file with validation"""
        name_offset = int.from_bytes(data[offset:offset+4], byteorder='little')
        value = int.from_bytes(data[offset+4:offset+8], byteorder='little')
        size = int.from_bytes(data[offset+8:offset+12], byteorder='little')
        type_value = data[offset+12]
        binding = data[offset+13]
        visibility = data[offset+14]
        section_index = data[offset+15]
        
        # Validate name offset
        if name_offset >= len(self.string_table) and len(self.string_table) > 0:
            self.debug_print(f"Warning: Symbol name offset (0x{name_offset:x}) is beyond end of string table")
        
        return {
            'name_offset': name_offset,
            'value': value,
            'size': size,
            'type': type_value,
            'binding': binding,
            'visibility': visibility,
            'section_index': section_index
        }
    
    def _get_string(self, offset):
        """Get a string from the string table with bounds checking"""
        if not self.string_table:
            return f"<string_{offset}>"
        
        if offset < 0 or offset >= len(self.string_table):
            return f"<invalid_offset_{offset}>"
        
        # Find the end of the string (null terminator)
        end = offset
        while end < len(self.string_table) and self.string_table[end] != 0:
            end += 1
        
        if end >= len(self.string_table):
            # String not null-terminated, return what we have
            return self.string_table[offset:].decode('utf-8', errors='replace')
        
        try:
            return self.string_table[offset:end].decode('utf-8')
        except UnicodeDecodeError:
            # Handle invalid UTF-8 sequences
            return self.string_table[offset:end].decode('utf-8', errors='replace')
    
    def get_entry_point(self):
        """Get the entry point for the loaded COF file"""
        return self.entry_point
    
    def get_symbol_address(self, name):
        """Get the address of a symbol by name"""
        if name in self.symbols:
            symbol = self.symbols[name]
            
            # Return adjusted value if available
            if 'adjusted_value' in symbol:
                return symbol['adjusted_value']
            
            # For absolute symbols
            if symbol['section_index'] == 0:
                return symbol['value']
            
            # For section-relative symbols
            if symbol['section_index'] in self.loaded_sections:
                section = self.loaded_sections[symbol['section_index']]
                return section['address'] + symbol['value']
        
        return None