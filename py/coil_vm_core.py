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
    """Manages memory for the VM"""
    
    def __init__(self, memory_size=1024*1024):  # Default 1MB memory
        self.memory = bytearray(memory_size)
        self.memory_size = memory_size
        self.sections = {}  # Map of named sections
    
    def read_byte(self, address):
        """Read a byte from memory"""
        self._check_address(address)
        return self.memory[address]
    
    def write_byte(self, address, value):
        """Write a byte to memory"""
        self._check_address(address)
        self.memory[address] = value & 0xFF
    
    def read_word(self, address):
        """Read a 16-bit word from memory"""
        self._check_address(address + 1)
        return (self.memory[address] | 
                (self.memory[address + 1] << 8))
    
    def write_word(self, address, value):
        """Write a 16-bit word to memory"""
        self._check_address(address + 1)
        self.memory[address] = value & 0xFF
        self.memory[address + 1] = (value >> 8) & 0xFF
    
    def read_long(self, address):
        """Read a 32-bit long from memory"""
        self._check_address(address + 3)
        return (self.memory[address] | 
                (self.memory[address + 1] << 8) |
                (self.memory[address + 2] << 16) |
                (self.memory[address + 3] << 24))
    
    def write_long(self, address, value):
        """Write a 32-bit long to memory"""
        self._check_address(address + 3)
        self.memory[address] = value & 0xFF
        self.memory[address + 1] = (value >> 8) & 0xFF
        self.memory[address + 2] = (value >> 16) & 0xFF
        self.memory[address + 3] = (value >> 24) & 0xFF
    
    def read_quad(self, address):
        """Read a 64-bit quad from memory"""
        self._check_address(address + 7)
        return (self.memory[address] | 
                (self.memory[address + 1] << 8) |
                (self.memory[address + 2] << 16) |
                (self.memory[address + 3] << 24) |
                (self.memory[address + 4] << 32) |
                (self.memory[address + 5] << 40) |
                (self.memory[address + 6] << 48) |
                (self.memory[address + 7] << 56))
    
    def write_quad(self, address, value):
        """Write a 64-bit quad to memory"""
        self._check_address(address + 7)
        self.memory[address] = value & 0xFF
        self.memory[address + 1] = (value >> 8) & 0xFF
        self.memory[address + 2] = (value >> 16) & 0xFF
        self.memory[address + 3] = (value >> 24) & 0xFF
        self.memory[address + 4] = (value >> 32) & 0xFF
        self.memory[address + 5] = (value >> 40) & 0xFF
        self.memory[address + 6] = (value >> 48) & 0xFF
        self.memory[address + 7] = (value >> 56) & 0xFF
    
    def read_bytes(self, address, length):
        """Read block of bytes from memory"""
        self._check_address(address + length - 1)
        return self.memory[address:address + length]
    
    def write_bytes(self, address, data):
        """Write block of bytes to memory"""
        length = len(data)
        self._check_address(address + length - 1)
        self.memory[address:address + length] = data
    
    def add_section(self, name, start_address, size, flags):
        """Add a named memory section"""
        if start_address + size > self.memory_size:
            raise MemoryError(f"Section {name} exceeds memory size")
        
        self.sections[name] = {
            'start': start_address,
            'size': size,
            'flags': flags
        }
    
    def get_section(self, name):
        """Get information about a named section"""
        return self.sections.get(name)
    
    def _check_address(self, address):
        """Check if an address is valid"""
        if address < 0 or address >= self.memory_size:
            raise MemoryError(f"Address out of range: {address}")
    
    def reset(self):
        """Reset all memory to zero"""
        self.memory = bytearray(self.memory_size)
        self.sections = {}


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
        """Decode instruction at given address and update RIP"""
        mem = self.vm.memory_manager
        
        # Instruction format:
        # [opcode(1)][qualifier(1)][operand_count(1)][operands...]
        opcode = mem.read_byte(address)
        qualifier = mem.read_byte(address + 1)
        operand_count = mem.read_byte(address + 2)
        
        operands = []
        current_pos = address + 3
        
        for _ in range(operand_count):
            # Operand format: [qualifier(1)][type(1)][data(variable)]
            op_qualifier = mem.read_byte(current_pos)
            op_type = mem.read_byte(current_pos + 1)
            
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
                
                operands.append({
                    'qualifier': op_qualifier,
                    'type': op_type,
                    'value': value
                })
                current_pos += 2 + data_size
                
            elif op_qualifier == self.OPQUAL_REG:
                # Register reference
                reg_index = mem.read_byte(current_pos + 2)
                operands.append({
                    'qualifier': op_qualifier,
                    'type': op_type,
                    'reg_index': reg_index
                })
                current_pos += 3
                
            elif op_qualifier == self.OPQUAL_MEM:
                # Memory address - using 64-bit addresses
                address_value = mem.read_quad(current_pos + 2)
                operands.append({
                    'qualifier': op_qualifier,
                    'type': op_type,
                    'address': address_value
                })
                current_pos += 10
                
            else:
                # Other operand types can be added as needed
                raise ValueError(f"Unsupported operand qualifier: {op_qualifier}")
        
        # Calculate instruction size for RIP update
        instruction_size = current_pos - address
        
        return {
            'opcode': opcode,
            'qualifier': qualifier,
            'operand_count': operand_count,
            'operands': operands,
            'size': instruction_size
        }
    
    def execute(self, start_address):
        """Execute instructions starting at given address"""
        self.vm.register_manager.set_register('RIP', start_address)
        
        running = True
        while running:
            current_rip = self.vm.register_manager.get_register('RIP', 0)
            instruction = self.decode_instruction(current_rip)
            
            # Update RIP to next instruction
            self.vm.register_manager.set_register('RIP', current_rip + instruction['size'])
            
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
    """Loads COIL Object Format (COF) files"""
    
    def __init__(self, vm):
        self.vm = vm
        self.entry_point = 0
        self.loaded_sections = {}
        self.symbols = {}
        self.string_table = bytearray()
    
    def load(self, filename):
        """Load a COF file into the VM"""
        with open(filename, 'rb') as file:
            data = file.read()
        
        return self._load_from_memory(data)
    
    def _load_from_memory(self, data):
        """Load a COF file from memory"""
        # Validate COF magic number - should be 'COIL' (0x434F494C)
        magic = int.from_bytes(data[0:4], byteorder='little')
        if magic != 0x434F494C:
            raise ValueError(f"Invalid COF file: magic number mismatch: {magic:#x}")
        
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
        
        self.entry_point = entrypoint
        
        # Load string table
        if str_tab_size > 0:
            self.string_table = data[str_tab_off:str_tab_off + str_tab_size]
        
        # Load sections
        section_offset = 32  # Start after header
        for _ in range(section_count):
            section = self._parse_section_header(data, section_offset)
            section_offset += 36  # Size of section header
            
            # Load section data into memory
            if section['size'] > 0:
                section_data = data[section['offset']:section['offset'] + section['size']]
                
                # Allocate memory for the section
                section_address = self.vm.memory_manager.memory_size - section['size']
                self.vm.memory_manager.write_bytes(section_address, section_data)
                
                # Register section
                self.vm.memory_manager.add_section(
                    self._get_string(section['name_offset']), 
                    section_address, 
                    section['size'], 
                    section['flags']
                )
                
                # Store mapped section
                self.loaded_sections[section['name_offset']] = {
                    'address': section_address,
                    'size': section['size'],
                    'flags': section['flags']
                }
        
        # Load symbol table
        if sym_tab_size > 0:
            symbol_count = sym_tab_size // 16  # Each symbol entry is 16 bytes
            for i in range(symbol_count):
                offset = sym_tab_off + i * 16
                symbol = self._parse_symbol(data, offset)
                
                symbol_name = self._get_string(symbol['name_offset'])
                self.symbols[symbol_name] = symbol
        
        return self.entry_point
    
    def _parse_section_header(self, data, offset):
        """Parse a section header from the COF file"""
        section = {}
        section['name_offset'] = int.from_bytes(data[offset:offset+4], byteorder='little')
        section['type'] = int.from_bytes(data[offset+4:offset+8], byteorder='little')
        section['flags'] = int.from_bytes(data[offset+8:offset+12], byteorder='little')
        section['offset'] = int.from_bytes(data[offset+12:offset+16], byteorder='little')
        section['size'] = int.from_bytes(data[offset+16:offset+20], byteorder='little')
        section['link'] = int.from_bytes(data[offset+20:offset+24], byteorder='little')
        section['info'] = int.from_bytes(data[offset+24:offset+28], byteorder='little')
        section['alignment'] = int.from_bytes(data[offset+28:offset+32], byteorder='little')
        section['entsize'] = int.from_bytes(data[offset+32:offset+36], byteorder='little')
        return section
    
    def _parse_symbol(self, data, offset):
        """Parse a symbol from the COF file"""
        symbol = {}
        symbol['name_offset'] = int.from_bytes(data[offset:offset+4], byteorder='little')
        symbol['value'] = int.from_bytes(data[offset+4:offset+8], byteorder='little')
        symbol['size'] = int.from_bytes(data[offset+8:offset+12], byteorder='little')
        symbol['type'] = data[offset+12]
        symbol['binding'] = data[offset+13]
        symbol['visibility'] = data[offset+14]
        symbol['section_index'] = data[offset+15]
        return symbol
    
    def _get_string(self, offset):
        """Get a string from the string table"""
        if offset < 0 or offset >= len(self.string_table):
            return ""
        
        # Strings are null-terminated
        end = offset
        while end < len(self.string_table) and self.string_table[end] != 0:
            end += 1
        
        return self.string_table[offset:end].decode('utf-8')
    
    def get_entry_point(self):
        """Get the entry point for the loaded COF file"""
        return self.entry_point
    
    def get_symbol_address(self, name):
        """Get the address of a symbol by name"""
        if name in self.symbols:
            symbol = self.symbols[name]
            
            # For absolute symbols
            if symbol['section_index'] == 0:
                return symbol['value']
            
            # For section-relative symbols
            section_name_offset = None
            for section_offset, section in self.loaded_sections.items():
                section_name = self._get_string(section_offset)
                if section_name == self._get_string(symbol['section_index']):
                    section_name_offset = section_offset
                    break
            
            if section_name_offset is not None:
                section = self.loaded_sections[section_name_offset]
                return section['address'] + symbol['value']
        
        return None