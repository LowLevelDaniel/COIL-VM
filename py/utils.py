# COIL VM - COF Utilities and Example Usage

import struct
import binascii
from enum import Enum

class COFHeaderBuilder:
    """Helps build a valid COF header"""
    
    def __init__(self):
        self.magic = 0x434F494C  # 'COIL'
        self.version_major = 1
        self.version_minor = 0
        self.version_patch = 0
        self.flags = 0
        self.target = 0x0002  # Default: x86-64
        self.section_count = 0
        self.entrypoint = 0
        self.str_tab_off = 0
        self.str_tab_size = 0
        self.sym_tab_off = 0
        self.sym_tab_size = 0
    
    def set_target(self, target_id):
        """Set the target architecture"""
        self.target = target_id
        return self
    
    def set_executable(self, is_executable):
        """Set whether this is an executable (has entrypoint)"""
        if is_executable:
            self.flags |= 0x01  # COF_FLAG_EXECUTABLE
        else:
            self.flags &= ~0x01
        return self
    
    def set_entrypoint(self, entrypoint):
        """Set the entrypoint offset"""
        self.entrypoint = entrypoint
        return self
    
    def set_section_count(self, count):
        """Set the number of sections"""
        self.section_count = count
        return self
    
    def set_string_table(self, offset, size):
        """Set string table information"""
        self.str_tab_off = offset
        self.str_tab_size = size
        return self
    
    def set_symbol_table(self, offset, size):
        """Set symbol table information"""
        self.sym_tab_off = offset
        self.sym_tab_size = size
        return self
    
    def build(self):
        """Build the COF header as bytes"""
        header = struct.pack(
            '<IBBBBHHIIIII8s',
            self.magic,
            self.version_major,
            self.version_minor,
            self.version_patch,
            self.flags,
            self.target,
            self.section_count,
            self.entrypoint,
            self.str_tab_off,
            self.str_tab_size,
            self.sym_tab_off,
            self.sym_tab_size,
            b'\x00' * 8  # padding
        )
        return header


class COFSectionBuilder:
    """Helps build a valid COF section header"""
    
    def __init__(self, name_offset):
        self.name_offset = name_offset
        self.type = 1  # Default: CODE section
        self.flags = 0
        self.offset = 0
        self.size = 0
        self.link = 0
        self.info = 0
        self.alignment = 8  # Default alignment: 8 bytes
        self.entsize = 0
    
    def set_type(self, section_type):
        """Set the section type"""
        self.type = section_type
        return self
    
    def set_flags(self, flags):
        """Set section flags"""
        self.flags = flags
        return self
    
    def set_offset_and_size(self, offset, size):
        """Set the offset and size of section data"""
        self.offset = offset
        self.size = size
        return self
    
    def set_alignment(self, alignment):
        """Set section alignment"""
        self.alignment = alignment
        return self
    
    def build(self):
        """Build the section header as bytes"""
        header = struct.pack(
            '<IIIIIIIIII',
            self.name_offset,
            self.type,
            self.flags,
            self.offset,
            self.size,
            self.link,
            self.info,
            self.alignment,
            self.entsize,
            0  # padding
        )
        return header


class COFSymbolBuilder:
    """Helps build a valid COF symbol entry"""
    
    def __init__(self, name_offset):
        self.name_offset = name_offset
        self.value = 0
        self.size = 0
        self.type = 0  # SYM_TYPE_NOTYPE
        self.binding = 1  # SYM_BIND_GLOBAL
        self.visibility = 0  # SYM_VIS_DEFAULT
        self.section_index = 0
    
    def set_value(self, value):
        """Set symbol value"""
        self.value = value
        return self
    
    def set_size(self, size):
        """Set symbol size"""
        self.size = size
        return self
    
    def set_type(self, sym_type):
        """Set symbol type"""
        self.type = sym_type
        return self
    
    def set_binding(self, binding):
        """Set symbol binding"""
        self.binding = binding
        return self
    
    def set_section_index(self, index):
        """Set symbol section index"""
        self.section_index = index
        return self
    
    def build(self):
        """Build the symbol entry as bytes"""
        entry = struct.pack(
            '<IIBBBB',
            self.name_offset,
            self.value,
            self.size,
            self.type,
            self.binding,
            self.visibility,
            self.section_index
        )
        return entry


class COFBuilder:
    """Helper class to build a complete COF file"""
    
    def __init__(self):
        self.header_builder = COFHeaderBuilder()
        self.sections = []
        self.section_data = []
        self.string_table = bytearray(b'\x00')  # First byte must be null
        self.symbol_table = []
    
    def add_string(self, string):
        """Add a string to the string table and return its offset"""
        if not string:
            return 0
        
        # Check if string already exists in the table
        string_bytes = string.encode('utf-8') + b'\x00'
        current_index = 1
        while current_index < len(self.string_table):
            end = current_index
            while end < len(self.string_table) and self.string_table[end] != 0:
                end += 1
            
            if end < len(self.string_table):
                existing_string = self.string_table[current_index:end]
                if existing_string == string_bytes[:-1]:  # Compare without null terminator
                    return current_index
            
            current_index = end + 1
        
        # String not found, add it
        offset = len(self.string_table)
        self.string_table.extend(string_bytes)
        return offset
    
    def add_section(self, name, section_type, flags, data=b''):
        """Add a section to the COF file"""
        name_offset = self.add_string(name)
        
        section_builder = COFSectionBuilder(name_offset)
        section_builder.set_type(section_type)
        section_builder.set_flags(flags)
        
        self.sections.append(section_builder)
        self.section_data.append(data)
        
        return len(self.sections) - 1  # Return section index
    
    def add_symbol(self, name, value, sym_type, binding, section_index=0):
        """Add a symbol to the COF file"""
        name_offset = self.add_string(name)
        
        symbol_builder = COFSymbolBuilder(name_offset)
        symbol_builder.set_value(value)
        symbol_builder.set_type(sym_type)
        symbol_builder.set_binding(binding)
        symbol_builder.set_section_index(section_index)
        
        self.symbol_table.append(symbol_builder)
        
        return len(self.symbol_table) - 1  # Return symbol index
    
    def set_entrypoint(self, offset):
        """Set the entrypoint offset"""
        self.header_builder.set_executable(True)
        self.header_builder.set_entrypoint(offset)
    
    def build(self):
        """Build the complete COF file"""
        # Calculate offsets
        header_size = 32  # COF header size
        section_header_size = 40  # Section header size
        
        # Calculate total size for section headers
        section_headers_size = len(self.sections) * section_header_size
        
        # String table offset
        str_tab_off = header_size + section_headers_size
        str_tab_size = len(self.string_table)
        
        # Symbol table offset
        sym_tab_off = str_tab_off + str_tab_size
        sym_tab_size = len(self.symbol_table) * 16  # Each symbol entry is 16 bytes
        
        # Section data offset
        section_data_off = sym_tab_off + sym_tab_size
        
        # Update header with offsets
        self.header_builder.set_section_count(len(self.sections))
        self.header_builder.set_string_table(str_tab_off, str_tab_size)
        self.header_builder.set_symbol_table(sym_tab_off, sym_tab_size)
        
        # Build COF file
        cof_data = bytearray(self.header_builder.build())
        
        # Add section headers
        current_data_offset = section_data_off
        for i, section_builder in enumerate(self.sections):
            section_data = self.section_data[i]
            section_builder.set_offset_and_size(current_data_offset, len(section_data))
            cof_data.extend(section_builder.build())
            current_data_offset += len(section_data)
        
        # Add string table
        cof_data.extend(self.string_table)
        
        # Add symbol table
        for symbol_builder in self.symbol_table:
            cof_data.extend(symbol_builder.build())
        
        # Add section data
        for section_data in self.section_data:
            cof_data.extend(section_data)
        
        return cof_data


class Opcodes(Enum):
    """COIL instruction opcodes"""
    NOP = 0x00
    BR = 0x02
    BRC = 0x03
    CALL = 0x04
    RET = 0x05
    ADD = 0x10
    SUB = 0x11
    MUL = 0x12
    DIV = 0x13
    AND = 0x20
    OR = 0x21
    XOR = 0x22
    NOT = 0x23
    SHL = 0x24
    SHR = 0x25
    CMP = 0x30
    MOV = 0x40
    LOAD = 0x41
    STORE = 0x42
    PUSH = 0x50
    POP = 0x51


class OperandQualifier(Enum):
    """COIL operand qualifiers"""
    IMM = 0x01  # Immediate value
    VAR = 0x02  # Variable
    REG = 0x03  # Register
    MEM = 0x04  # Memory address
    LBL = 0x05  # Label
    STR = 0x06  # String
    SYM = 0x07  # Symbol
    REL = 0x08  # Relative offset


class COILType(Enum):
    """COIL type codes"""
    INT = 0x00    # Signed integer
    UINT = 0x01   # Unsigned integer
    FLOAT = 0x02  # Floating point
    VOID = 0xF0   # Void type
    BOOL = 0xF1   # Boolean
    LINT = 0xF2   # Largest native integer
    FINT = 0xF3   # Fastest native integer
    PTR = 0xF4    # Pointer type


class TypeWidth(Enum):
    """COIL type width codes"""
    W8 = 0x00    # 8-bit
    W16 = 0x01   # 16-bit
    W32 = 0x02   # 32-bit
    W64 = 0x03   # 64-bit
    W128 = 0x04  # 128-bit (future use)


class InstructionBuilder:
    """Helper class to build COIL instructions"""
    
    def __init__(self, opcode, qualifier=0):
        self.opcode = opcode
        self.qualifier = qualifier
        self.operands = []
    
    def add_imm_int(self, value, width=TypeWidth.W64.value):
        """Add an immediate integer operand"""
        self.operands.append({
            'qualifier': OperandQualifier.IMM.value,
            'type': COILType.INT.value,
            'width': width,
            'value': value
        })
        return self
    
    def add_imm_uint(self, value, width=TypeWidth.W64.value):
        """Add an immediate unsigned integer operand"""
        self.operands.append({
            'qualifier': OperandQualifier.IMM.value,
            'type': COILType.UINT.value,
            'width': width,
            'value': value
        })
        return self
    
    def add_reg(self, reg_index, type_code=COILType.INT.value, width=TypeWidth.W64.value):
        """Add a register operand"""
        self.operands.append({
            'qualifier': OperandQualifier.REG.value,
            'type': type_code,
            'width': width,
            'reg_index': reg_index
        })
        return self
    
    def add_mem(self, address, type_code=COILType.INT.value, width=TypeWidth.W64.value):
        """Add a memory operand"""
        self.operands.append({
            'qualifier': OperandQualifier.MEM.value,
            'type': type_code,
            'width': width,
            'address': address
        })
        return self
    
    def build(self):
        """Build the instruction as bytes"""
        # Start with opcode, qualifier, and operand count
        instruction = bytearray([self.opcode, self.qualifier, len(self.operands)])
        
        # Add each operand
        for operand in self.operands:
            qualifier = operand['qualifier']
            type_code = operand['type']
            
            instruction.extend([qualifier, type_code])
            
            if qualifier == OperandQualifier.IMM.value:
                # Immediate value
                width = operand.get('width', TypeWidth.W64.value)
                value = operand['value']
                
                if width == TypeWidth.W8.value:
                    instruction.extend([value & 0xFF])
                elif width == TypeWidth.W16.value:
                    instruction.extend([value & 0xFF, (value >> 8) & 0xFF])
                elif width == TypeWidth.W32.value:
                    instruction.extend([
                        value & 0xFF, 
                        (value >> 8) & 0xFF, 
                        (value >> 16) & 0xFF, 
                        (value >> 24) & 0xFF
                    ])
                else:  # Default to 64-bit
                    instruction.extend([
                        value & 0xFF, 
                        (value >> 8) & 0xFF, 
                        (value >> 16) & 0xFF, 
                        (value >> 24) & 0xFF,
                        (value >> 32) & 0xFF, 
                        (value >> 40) & 0xFF, 
                        (value >> 48) & 0xFF, 
                        (value >> 56) & 0xFF
                    ])
            
            elif qualifier == OperandQualifier.REG.value:
                # Register
                reg_index = operand['reg_index']
                instruction.extend([reg_index])
            
            elif qualifier == OperandQualifier.MEM.value:
                # Memory address (64-bit)
                address = operand['address']
                instruction.extend([
                    address & 0xFF, 
                    (address >> 8) & 0xFF, 
                    (address >> 16) & 0xFF, 
                    (address >> 24) & 0xFF,
                    (address >> 32) & 0xFF, 
                    (address >> 40) & 0xFF, 
                    (address >> 48) & 0xFF, 
                    (address >> 56) & 0xFF
                ])
        
        return bytes(instruction)


# Example: Create a simple COF file with a "Hello, World" program
def create_example_cof(filename):
    """Create a simple example COF file that prints "Hello, World!" when executed"""
    builder = COFBuilder()
    
    # Add code section
    code_section_idx = builder.add_section(
        ".text",             # Section name
        1,                   # Section type (CODE)
        0x03                 # Flags (WRITE | EXEC)
    )
    
    # Add data section for "Hello, World!" string
    hello_str = "Hello, World!\n"
    data = hello_str.encode('ascii')
    data_section_idx = builder.add_section(
        ".data",             # Section name
        2,                   # Section type (DATA)
        0x01,                # Flags (WRITE)
        data                 # Section data
    )
    
    # Add main symbol
    builder.add_symbol(
        "main",              # Symbol name
        0,                   # Value (offset in section)
        2,                   # Type (FUNC)
        1,                   # Binding (GLOBAL)
        code_section_idx     # Section index
    )
    
    # Add hello_str symbol
    builder.add_symbol(
        "hello_str",         # Symbol name
        0,                   # Value (offset in section)
        1,                   # Type (OBJECT)
        1,                   # Binding (GLOBAL)
        data_section_idx     # Section index
    )
    
    # Create a simple program
    # This is a placeholder - in a real implementation, this would be more complex
    # and would actually print "Hello, World!" when executed
    
    # Initialize registers and call print function
    instructions = [
        # MOV RQ0, hello_str address
        InstructionBuilder(Opcodes.MOV.value).add_reg(0).add_imm_uint(0x1000).build(),
        
        # MOV RQ1, hello_str length
        InstructionBuilder(Opcodes.MOV.value).add_reg(1).add_imm_uint(len(hello_str)).build(),
        
        # MOV RQ4, 1 (stdout file descriptor)
        InstructionBuilder(Opcodes.MOV.value).add_reg(4).add_imm_uint(1).build(),
        
        # MOV RQ0, 1 (syscall number for write)
        InstructionBuilder(Opcodes.MOV.value).add_reg(0).add_imm_uint(1).build(),
        
        # System call (write syscall)
        # This is a simplified representation as SYSC isn't fully implemented in our example
        InstructionBuilder(0x09).add_reg(0).build(),  # SYSC RQ0
        
        # Exit program
        InstructionBuilder(Opcodes.MOV.value).add_reg(0).add_imm_uint(60).build(),  # exit syscall number
        InstructionBuilder(Opcodes.MOV.value).add_reg(4).add_imm_uint(0).build(),   # exit code 0
        InstructionBuilder(0x09).add_reg(0).build(),  # SYSC RQ0
    ]
    
    # Combine all instructions
    code = b''.join(instructions)
    
    # Update code section
    builder.section_data[code_section_idx] = code
    
    # Set entrypoint
    builder.set_entrypoint(0)  # Start at the beginning of the code section
    
    # Build and write COF file
    cof_data = builder.build()
    with open(filename, 'wb') as f:
        f.write(cof_data)
    
    print(f"Created example COF file: {filename}")
    print(f"  Code section size: {len(code)} bytes")
    print(f"  Total COF size: {len(cof_data)} bytes")
    
    return filename


# Example usage of the COIL VM
def run_example():
    from coil_vm_core import COILVirtualMachine
    
    # Create an example COF file
    cof_file = create_example_cof("hello_world.cof")
    
    # Initialize the VM
    vm = COILVirtualMachine()
    
    # Load the COF file
    entry_point = vm.load_file(cof_file)
    
    # Execute the loaded code
    vm.execute(entry_point)
    
    print("Execution completed")


if __name__ == "__main__":
    run_example()